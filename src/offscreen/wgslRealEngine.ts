import type { GpuPacingMode, LoadSource, WgslKvMode } from '../shared/protocol.ts';
import {
  computeDecodePostStepBreatherMs,
  computeDecodeSliceBounds,
  computePrefillSliceBreatherMs,
  getGpuPacingPolicy,
  resolveExtensionAssetUrl,
  selectDecodeSliceCount,
  selectPrefillChunkSize,
  sleepMs,
} from './gpuPacing.ts';
import {
  computeAuxLayout,
  QWEN35_FULL_ATTN_LAYERS,
  QWEN35_HIDDEN,
  QWEN35_INTERMEDIATE,
  QWEN35_NUM_LAYERS,
  QWEN35_VOCAB_SIZE,
  type AuxLayout,
  type WgslSlabManifest,
} from './onnxWeightLoader.ts';
import type { OpfsModelCache } from './opfsCache.ts';
import { sampleNextToken, type SamplingOptions } from './sampling.ts';
import {
  acquireWgslDevice,
  createValidatedPipeline,
  getWgslShaderSource,
  GPU_MAP_READ,
  GPU_SHADER_COMPUTE,
  GPU_USAGE,
  loadWgslWasm,
  readGpuF32Slice,
  streamUrlToGpuBuffer,
  type GpuBufferLike,
  type GpuDeviceLike,
  type WasmEngineExports,
} from './wgslWebgpu.ts';

const GQA_NUM_SPLITS = 32;
const PREFILL_NUM_SPLITS = 16;
const PREFILL_MAX_CHUNK = 64;
const UNIFORM_MAX_SLOTS = 512;
const MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024;
const BUNDLED_MODEL_MISSING_MSG =
  'Bundled model files missing in models/qwen35-4b/. Run npm run bundle:model.';

export interface WgslLoadStats {
  coldLoadMs: number;
  repackMs: number;
  opfsReadMs: number;
  pipelineCompileMs: number;
  loadedFromCache: boolean;
  loadSource: LoadSource;
  weightMb: number;
  maxSeqCapacity: number;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  maxBufferSize: number;
}

export interface WgslCpuSnapshot {
  seqPos: number;
  kvMode: WgslKvMode;
  dnState: Uint8Array;
  dnConv: Uint8Array;
  kvSlices0: Uint8Array[]; // 4 layers x 2 planes (K, V)
  kvSlices1: Uint8Array[];
}

interface DispatchStep {
  pipeline: unknown;
  bindGroup: unknown;
  uniformByteOffset: number;
  wx: number;
  wy: number;
  wz: number;
}

export class WgslQwen35Engine {
  private device: GpuDeviceLike | null = null;
  private wasm: WasmEngineExports | null = null;
  private allocatedBuffers: GpuBufferLike[] = [];

  private kvMode: WgslKvMode = 'int8';
  private maxSeqCapacity = 131584;
  private seqPos = 0;
  private auxLayout!: AuxLayout;

  private weightSlab0!: GpuBufferLike;
  private weightSlab1!: GpuBufferLike;
  private weightSlab2!: GpuBufferLike;
  private auxBuf!: GpuBufferLike;
  private dnStateBuf!: GpuBufferLike;
  private dnConvBuf!: GpuBufferLike;
  private kvSlab0!: GpuBufferLike;
  private kvSlab1!: GpuBufferLike;
  private tokenIdsBuf!: GpuBufferLike;
  private hiddenBuf!: GpuBufferLike;
  private normBuf!: GpuBufferLike;
  private workBuf!: GpuBufferLike;
  private attnOutBuf!: GpuBufferLike;
  private gqaSplitBuf!: GpuBufferLike;
  private logitsBuf!: GpuBufferLike;
  private tokOutBuf!: GpuBufferLike;
  private staging4B!: GpuBufferLike;
  private decodeUniformBuf!: GpuBufferLike;
  private prefillUniformBuf!: GpuBufferLike;

  private embedPipe!: unknown;
  private rmsNormPipe!: unknown;
  private swigluPipe!: unknown;
  private gemvPipe!: unknown;
  private gemmPipe!: unknown;
  private convPipe!: unknown;
  private gdnRecPipe!: unknown;
  private qkRopePipe!: unknown;
  private gqaSplitPipe!: unknown;
  private gqaReducePipe!: unknown;
  private gqaPrefillPipe!: unknown;
  private arg1Pipe!: unknown;
  private arg2Pipe!: unknown;

  private bgEmbed!: unknown;
  private bgRmsNorm!: unknown;
  private bgSwiGlu!: unknown;
  private bgW0NormToWork!: unknown;
  private bgW1NormToWork!: unknown;
  private bgW0AttnToHidden!: unknown;
  private bgW1AttnToHidden!: unknown;
  private bgLmHead!: unknown;
  private bgConv!: unknown;
  private bgGdnRec!: unknown;
  private bgQkRope!: unknown;
  private bgGqaSplit0!: unknown;
  private bgGqaSplit1!: unknown;
  private bgGqaReduce!: unknown;
  private bgGqaPrefill0!: unknown;
  private bgGqaPrefill1!: unknown;

  // Separate prefill bind groups (using prefillUniformBuf) so static decodeUniformBuf stays untouched
  private pBgEmbed!: unknown;
  private pBgRmsNorm!: unknown;
  private pBgSwiGlu!: unknown;
  private pBgW0NormToWork!: unknown;
  private pBgW1NormToWork!: unknown;
  private pBgW0AttnToHidden!: unknown;
  private pBgW1AttnToHidden!: unknown;
  private pBgConv!: unknown;
  private pBgGdnRec!: unknown;
  private pBgQkRope!: unknown;
  private pBgGqaReduce!: unknown;
  private pBgLmHead!: unknown;
  private bgArg1!: unknown;
  private bgArg2!: unknown;

  private decodeSteps: DispatchStep[] = [];
  private decodeLayerEndStepIndices: number[] = [];
  private decodeRopePosByteOffsets: number[] = [];
  private decodeGqaPosByteOffsets: number[] = [];
  private lastDecodeGpuMs = 0;
  private layerOffsets: Array<{
    isFullAttn: boolean;
    dnIdx: number;
    faIdx: number;
    w: number[];
  }> = [];

  private weightMb = 0;
  private dnSingleBytes = 0;
  private dnConvSingleBytes = 0;
  private kvSingleBytes = 0;
  private kvPlaneU32s = 0;
  private kvRowU32s = 0;

  get activeSeqPos(): number {
    return this.seqPos;
  }

  get kvCacheMode(): WgslKvMode {
    return this.kvMode;
  }

  get maxContextTokens(): number {
    return this.maxSeqCapacity;
  }

  getMemoryStats(): { weightMb: number; kvCacheMb: number; deltaNetStateMb: number; estVramMb: number } {
    const rowBytes = this.kvMode === 'int8' ? 65 * 4 : 128 * 4;
    const kvActiveBytes = 8 * 2 * 4 * this.seqPos * rowBytes;
    const kvCacheMb = kvActiveBytes / (1024 * 1024);
    const deltaNetStateMb = (24 * (this.dnSingleBytes + this.dnConvSingleBytes)) / (1024 * 1024);
    return {
      weightMb: this.weightMb,
      kvCacheMb,
      deltaNetStateMb,
      estVramMb: this.weightMb + kvCacheMb + deltaNetStateMb,
    };
  }

  private track(buf: GpuBufferLike): GpuBufferLike {
    this.allocatedBuffers.push(buf);
    return buf;
  }

  private async fetchBundledManifest(assetBaseUrl?: string): Promise<WgslSlabManifest> {
    const manifestUrl = resolveExtensionAssetUrl(
      'models/qwen35-4b/manifest.json',
      assetBaseUrl,
    );
    let res: Response;
    try {
      res = await fetch(manifestUrl);
    } catch {
      throw new Error(BUNDLED_MODEL_MISSING_MSG);
    }
    if (!res.ok) {
      throw new Error(BUNDLED_MODEL_MISSING_MSG);
    }
    let parsed: Partial<WgslSlabManifest> | null = null;
    try {
      parsed = (await res.json()) as Partial<WgslSlabManifest>;
    } catch {
      throw new Error(BUNDLED_MODEL_MISSING_MSG);
    }
    if (
      !parsed ||
      parsed.version !== 1 ||
      !parsed.complete ||
      !Number.isInteger(parsed.slab0Bytes) ||
      (parsed.slab0Bytes ?? 0) <= 0 ||
      !Number.isInteger(parsed.slab1Bytes) ||
      (parsed.slab1Bytes ?? 0) <= 0 ||
      !Number.isInteger(parsed.slab2Bytes) ||
      (parsed.slab2Bytes ?? 0) <= 0 ||
      !Number.isInteger(parsed.auxBytes) ||
      (parsed.auxBytes ?? 0) <= 0
    ) {
      throw new Error(BUNDLED_MODEL_MISSING_MSG);
    }
    return parsed as WgslSlabManifest;
  }

  async load(
    _opfsCache: OpfsModelCache | null,
    options: {
      assetBaseUrl?: string;
      modelId: string;
      kvMode?: WgslKvMode;
      maxContext?: number;
      pacingMode?: GpuPacingMode;
      onProgress?: (phase: string, pct: number) => void;
    },
  ): Promise<WgslLoadStats> {
    this.dispose();
    const loadStart = performance.now();

    try {
      const wasm = await loadWgslWasm(options.assetBaseUrl);
      this.wasm = wasm;
      const { device, hasShaderF16, hasSubgroups, maxBufferSize } = await acquireWgslDevice();
      this.device = device;

      if (!hasShaderF16 || !hasSubgroups) {
        throw new Error(
          `WebGPU device lacks required features (shader-f16=${hasShaderF16}, subgroups=${hasSubgroups}).`,
        );
      }

      this.kvMode = options.kvMode ?? 'int8';
      this.maxSeqCapacity =
        Math.min(options.maxContext ?? 131072, this.kvMode === 'fp16' ? 65536 : 131072) + 512;
      this.auxLayout = computeAuxLayout();

      const manifest = await this.fetchBundledManifest(options.assetBaseUrl);
      const policy = getGpuPacingPolicy(options.pacingMode ?? 'balanced');

      const gpuStreamStart = performance.now();
      options.onProgress?.('Streaming bundled Qwen3.5-4B Q4 slabs -> WebGPU VRAM...', 0);

      this.weightSlab0 = this.track(
        device.createBuffer({ size: manifest.slab0Bytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }),
      );
      this.weightSlab1 = this.track(
        device.createBuffer({ size: manifest.slab1Bytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }),
      );
      this.weightSlab2 = this.track(
        device.createBuffer({ size: manifest.slab2Bytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }),
      );
      this.auxBuf = this.track(
        device.createBuffer({ size: manifest.auxBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }),
      );

      const streamSpecs: Array<{
        key: 'slab0' | 'slab1' | 'slab2' | 'aux';
        buf: GpuBufferLike;
        expectedBytes: number;
      }> = [
        { key: 'slab0', buf: this.weightSlab0, expectedBytes: manifest.slab0Bytes },
        { key: 'slab1', buf: this.weightSlab1, expectedBytes: manifest.slab1Bytes },
        { key: 'slab2', buf: this.weightSlab2, expectedBytes: manifest.slab2Bytes },
        { key: 'aux', buf: this.auxBuf, expectedBytes: manifest.auxBytes },
      ];
      const totalBundleBytes = streamSpecs.reduce((sum, spec) => sum + spec.expectedBytes, 0);
      let completedBytes = 0;

      for (const spec of streamSpecs) {
        const defaultFile = spec.key === 'aux' ? 'aux_params.bin' : `${spec.key}.bin`;
        const shardList =
          manifest.shards?.[spec.key] ?? [{ file: defaultFile, bytes: spec.expectedBytes }];
        const shardBytesSum = shardList.reduce((sum, shard) => sum + shard.bytes, 0);
        if (shardBytesSum !== spec.expectedBytes) {
          throw new Error(
            `Manifest shard sum mismatch for ${spec.key}: got ${shardBytesSum}, expected ${spec.expectedBytes}.`,
          );
        }

        let shardOffset = 0;
        for (const shard of shardList) {
          const url = resolveExtensionAssetUrl(
            `models/qwen35-4b/${shard.file}`,
            options.assetBaseUrl,
          );
          await streamUrlToGpuBuffer(device, url, spec.buf, shard.bytes, {
            dstByteOffset: shardOffset,
            yieldMs: policy.loadChunkYieldMs,
            onProgress: (loadedBytes) => {
              const combined = completedBytes + loadedBytes;
              const pct = totalBundleBytes > 0 ? (combined / totalBundleBytes) * 86 : 86;
              options.onProgress?.(
                `Streaming bundled ${shard.file} -> WebGPU VRAM...`,
                Math.min(86, pct),
              );
            },
          });
          // Flush each <= 128 MiB shard cleanly so Dawn staging shared memory stays bounded.
          await device.queue.onSubmittedWorkDone();
          shardOffset += shard.bytes;
          completedBytes += shard.bytes;
        }
      }
      const opfsReadMs = performance.now() - gpuStreamStart;

    options.onProgress?.('Compiling Qwen3.5-4B WGSL compute pipelines...', 93);
    const compileStart = performance.now();

    const dynStorage4Bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const dynRwStorageBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const gdnRecBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const qkRopeBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const staticBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const staticRead2Bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const dyn4Layout = device.createPipelineLayout({ bindGroupLayouts: [dynStorage4Bgl] });
    const dynRwLayout = device.createPipelineLayout({ bindGroupLayouts: [dynRwStorageBgl] });
    const gdnLayout = device.createPipelineLayout({ bindGroupLayouts: [gdnRecBgl] });
    const qkRopeLayout = device.createPipelineLayout({ bindGroupLayouts: [qkRopeBgl] });

    [
      this.embedPipe,
      this.rmsNormPipe,
      this.swigluPipe,
      this.gemvPipe,
      this.gemmPipe,
      this.convPipe,
      this.gdnRecPipe,
      this.qkRopePipe,
      this.gqaSplitPipe,
      this.gqaReducePipe,
      this.gqaPrefillPipe,
      this.arg1Pipe,
      this.arg2Pipe,
    ] = await Promise.all([
      createValidatedPipeline(device, 'real-embed-lookup', getWgslShaderSource(wasm, 10), dyn4Layout),
      createValidatedPipeline(device, 'real-rmsnorm-batch', getWgslShaderSource(wasm, 11), dyn4Layout),
      createValidatedPipeline(device, 'real-swiglu-batch', getWgslShaderSource(wasm, 12), dyn4Layout),
      createValidatedPipeline(device, 'real-q4-subgroups', getWgslShaderSource(wasm, 2), dyn4Layout),
      createValidatedPipeline(device, 'real-q4-gemm-f16', getWgslShaderSource(wasm, 13), dyn4Layout),
      createValidatedPipeline(device, 'real-gdn-conv1d', getWgslShaderSource(wasm, 14), dynRwLayout),
      createValidatedPipeline(device, 'real-gdn-recurrent', getWgslShaderSource(wasm, 15), gdnLayout),
      createValidatedPipeline(device, 'real-qk-norm-rope', getWgslShaderSource(wasm, 16), qkRopeLayout),
      createValidatedPipeline(
        device,
        `real-gqa-${this.kvMode}-split`,
        getWgslShaderSource(wasm, this.kvMode === 'int8' ? 5 : 4),
        dynRwLayout,
      ),
      createValidatedPipeline(device, 'real-gqa-reduce', getWgslShaderSource(wasm, 9), dyn4Layout),
      createValidatedPipeline(
        device,
        `real-gqa-${this.kvMode}-prefill`,
        getWgslShaderSource(wasm, this.kvMode === 'int8' ? 18 : 17),
        dynRwLayout,
      ),
      createValidatedPipeline(
        device,
        'real-argmax-pass1',
        getWgslShaderSource(wasm, 7),
        device.createPipelineLayout({ bindGroupLayouts: [staticBgl] }),
      ),
      createValidatedPipeline(
        device,
        'real-argmax-pass2',
        getWgslShaderSource(wasm, 8),
        device.createPipelineLayout({ bindGroupLayouts: [staticRead2Bgl] }),
      ),
    ]);

    const pipelineCompileMs = performance.now() - compileStart;

    // Verify new real-model compute kernels against CPU reference before running user prompts
    await this.verifyRealKernels(device, dynStorage4Bgl, gdnRecBgl, qkRopeBgl, dynRwStorageBgl);

    // Allocate hybrid state buffers
    this.dnSingleBytes = 32 * 128 * 128 * 4;
    this.dnConvSingleBytes = 8192 * 3 * 4;
    this.dnStateBuf = this.track(
      device.createBuffer({ size: 24 * this.dnSingleBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST }),
    );
    this.dnConvBuf = this.track(
      device.createBuffer({ size: 24 * this.dnConvSingleBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST }),
    );

    this.kvRowU32s = this.kvMode === 'int8' ? 65 : 128;
    this.kvPlaneU32s = 4 * this.maxSeqCapacity * this.kvRowU32s;
    this.kvSingleBytes = Number(wasm.full_attn_layer_kv_bytes(this.maxSeqCapacity, this.kvMode === 'int8' ? 1 : 0));
    const kvSlabBytes = 4 * this.kvSingleBytes;
    this.kvSlab0 = this.track(
      device.createBuffer({ size: kvSlabBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST }),
    );
    this.kvSlab1 = this.track(
      device.createBuffer({ size: kvSlabBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST }),
    );

    // Allocate non-aliasing activation buffers sized for PREFILL_MAX_CHUNK
    this.tokenIdsBuf = this.track(
      device.createBuffer({ size: PREFILL_MAX_CHUNK * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }),
    );
    this.hiddenBuf = this.track(
      device.createBuffer({ size: PREFILL_MAX_CHUNK * QWEN35_HIDDEN * 4, usage: GPU_USAGE.STORAGE }),
    );
    this.normBuf = this.track(
      device.createBuffer({ size: PREFILL_MAX_CHUNK * QWEN35_HIDDEN * 4, usage: GPU_USAGE.STORAGE }),
    );
    this.workBuf = this.track(
      device.createBuffer({ size: PREFILL_MAX_CHUNK * QWEN35_INTERMEDIATE * 2 * 4, usage: GPU_USAGE.STORAGE }),
    );
    this.attnOutBuf = this.track(
      device.createBuffer({ size: PREFILL_MAX_CHUNK * QWEN35_INTERMEDIATE * 4, usage: GPU_USAGE.STORAGE }),
    );
    this.gqaSplitBuf = this.track(
      device.createBuffer({
        size: Math.max(16 * GQA_NUM_SPLITS, PREFILL_MAX_CHUNK * 16 * PREFILL_NUM_SPLITS) * 260 * 4,
        usage: GPU_USAGE.STORAGE,
      }),
    );
    this.logitsBuf = this.track(
      device.createBuffer({ size: QWEN35_VOCAB_SIZE * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC }),
    );

    const argmaxWgCount = Math.ceil(QWEN35_VOCAB_SIZE / 256);
    const partValBuf = this.track(device.createBuffer({ size: argmaxWgCount * 4, usage: GPU_USAGE.STORAGE }));
    const partIdxBuf = this.track(device.createBuffer({ size: argmaxWgCount * 4, usage: GPU_USAGE.STORAGE }));
    this.tokOutBuf = this.track(
      device.createBuffer({ size: 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC }),
    );
    this.staging4B = this.track(
      device.createBuffer({ size: 4, usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST }),
    );

    const uArg1 = this.track(device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
    const uArg2 = this.track(device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
    device.queue.writeBuffer(uArg1, 0, new Uint32Array([QWEN35_VOCAB_SIZE, 0, 0, 0]));
    device.queue.writeBuffer(uArg2, 0, new Uint32Array([argmaxWgCount, 0, 0, 0]));

    this.bgArg1 = device.createBindGroup({
      layout: staticBgl,
      entries: [
        { binding: 0, resource: { buffer: uArg1 } },
        { binding: 1, resource: { buffer: this.logitsBuf } },
        { binding: 2, resource: { buffer: partValBuf } },
        { binding: 3, resource: { buffer: partIdxBuf } },
      ],
    });
    this.bgArg2 = device.createBindGroup({
      layout: staticRead2Bgl,
      entries: [
        { binding: 0, resource: { buffer: uArg2 } },
        { binding: 1, resource: { buffer: partValBuf } },
        { binding: 2, resource: { buffer: partIdxBuf } },
        { binding: 3, resource: { buffer: this.tokOutBuf } },
      ],
    });

    const uniformBytes = UNIFORM_MAX_SLOTS * 256;
    this.decodeUniformBuf = this.track(
      device.createBuffer({ size: uniformBytes, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }),
    );
    this.prefillUniformBuf = this.track(
      device.createBuffer({ size: uniformBytes, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }),
    );

    const makeDyn4 = (uBuf: GpuBufferLike, b1: GpuBufferLike, b2: GpuBufferLike, b3: GpuBufferLike) =>
      device.createBindGroup({
        layout: dynStorage4Bgl,
        entries: [
          { binding: 0, resource: { buffer: uBuf, size: 32 } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
        ],
      });
    const makeDynRw = (uBuf: GpuBufferLike, b1: GpuBufferLike, b2: GpuBufferLike, b3: GpuBufferLike) =>
      device.createBindGroup({
        layout: dynRwStorageBgl,
        entries: [
          { binding: 0, resource: { buffer: uBuf, size: 32 } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
        ],
      });

    // Bind groups bound to decodeUniformBuf
    this.bgEmbed = makeDyn4(this.decodeUniformBuf, this.tokenIdsBuf, this.weightSlab2, this.hiddenBuf);
    this.bgRmsNorm = makeDyn4(this.decodeUniformBuf, this.auxBuf, this.hiddenBuf, this.normBuf);
    this.bgSwiGlu = makeDyn4(this.decodeUniformBuf, this.auxBuf, this.workBuf, this.attnOutBuf);
    this.bgW0NormToWork = makeDyn4(this.decodeUniformBuf, this.weightSlab0, this.normBuf, this.workBuf);
    this.bgW1NormToWork = makeDyn4(this.decodeUniformBuf, this.weightSlab1, this.normBuf, this.workBuf);
    this.bgW0AttnToHidden = makeDyn4(this.decodeUniformBuf, this.weightSlab0, this.attnOutBuf, this.hiddenBuf);
    this.bgW1AttnToHidden = makeDyn4(this.decodeUniformBuf, this.weightSlab1, this.attnOutBuf, this.hiddenBuf);
    this.bgLmHead = makeDyn4(this.decodeUniformBuf, this.weightSlab2, this.normBuf, this.logitsBuf);
    this.bgGqaReduce = makeDyn4(this.decodeUniformBuf, this.workBuf, this.gqaSplitBuf, this.attnOutBuf);
    this.bgConv = makeDynRw(this.decodeUniformBuf, this.auxBuf, this.dnConvBuf, this.workBuf);
    this.bgGqaSplit0 = makeDynRw(this.decodeUniformBuf, this.workBuf, this.kvSlab0, this.gqaSplitBuf);
    this.bgGqaSplit1 = makeDynRw(this.decodeUniformBuf, this.workBuf, this.kvSlab1, this.gqaSplitBuf);
    this.bgGdnRec = device.createBindGroup({
      layout: gdnRecBgl,
      entries: [
        { binding: 0, resource: { buffer: this.decodeUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.auxBuf } },
        { binding: 2, resource: { buffer: this.workBuf } },
        { binding: 3, resource: { buffer: this.dnStateBuf } },
        { binding: 4, resource: { buffer: this.attnOutBuf } },
      ],
    });
    this.bgQkRope = device.createBindGroup({
      layout: qkRopeBgl,
      entries: [
        { binding: 0, resource: { buffer: this.decodeUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.auxBuf } },
        { binding: 2, resource: { buffer: this.workBuf } },
      ],
    });

    // Prefill bind groups bound to prefillUniformBuf
    this.pBgEmbed = makeDyn4(this.prefillUniformBuf, this.tokenIdsBuf, this.weightSlab2, this.hiddenBuf);
    this.pBgRmsNorm = makeDyn4(this.prefillUniformBuf, this.auxBuf, this.hiddenBuf, this.normBuf);
    this.pBgSwiGlu = makeDyn4(this.prefillUniformBuf, this.auxBuf, this.workBuf, this.attnOutBuf);
    this.pBgW0NormToWork = makeDyn4(this.prefillUniformBuf, this.weightSlab0, this.normBuf, this.workBuf);
    this.pBgW1NormToWork = makeDyn4(this.prefillUniformBuf, this.weightSlab1, this.normBuf, this.workBuf);
    this.pBgW0AttnToHidden = makeDyn4(this.prefillUniformBuf, this.weightSlab0, this.attnOutBuf, this.hiddenBuf);
    this.pBgW1AttnToHidden = makeDyn4(this.prefillUniformBuf, this.weightSlab1, this.attnOutBuf, this.hiddenBuf);
    this.pBgLmHead = makeDyn4(this.prefillUniformBuf, this.weightSlab2, this.normBuf, this.logitsBuf);
    this.pBgConv = makeDynRw(this.prefillUniformBuf, this.auxBuf, this.dnConvBuf, this.workBuf);
    this.bgGqaPrefill0 = makeDynRw(this.prefillUniformBuf, this.workBuf, this.kvSlab0, this.gqaSplitBuf);
    this.bgGqaPrefill1 = makeDynRw(this.prefillUniformBuf, this.workBuf, this.kvSlab1, this.gqaSplitBuf);
    this.pBgGqaReduce = makeDyn4(this.prefillUniformBuf, this.workBuf, this.gqaSplitBuf, this.attnOutBuf);
    this.pBgGdnRec = device.createBindGroup({
      layout: gdnRecBgl,
      entries: [
        { binding: 0, resource: { buffer: this.prefillUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.auxBuf } },
        { binding: 2, resource: { buffer: this.workBuf } },
        { binding: 3, resource: { buffer: this.dnStateBuf } },
        { binding: 4, resource: { buffer: this.attnOutBuf } },
      ],
    });
    this.pBgQkRope = device.createBindGroup({
      layout: qkRopeBgl,
      entries: [
        { binding: 0, resource: { buffer: this.prefillUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.auxBuf } },
        { binding: 2, resource: { buffer: this.workBuf } },
      ],
    });

    // Compute exact Q4 weight u32 offsets matching onnxWeightLoader.ts order
    this.layerOffsets = [];
    let wOff0 = 0;
    let wOff1 = 0;
    let dnIdx = 0;
    let faIdx = 0;
    for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
      const inSlab0 = l < 16;
      const takeW = (rows: number, cols: number): number => {
        const u32s = (rows * wasm.q4_row_u32s(cols)) >>> 0;
        if (inSlab0) {
          const cur = wOff0;
          wOff0 += u32s;
          return cur;
        }
        const cur = wOff1;
        wOff1 += u32s;
        return cur;
      };
      const isFullAttn = QWEN35_FULL_ATTN_LAYERS.has(l);
      if (!isFullAttn) {
        const w = [
          takeW(8192, QWEN35_HIDDEN),
          takeW(4096, QWEN35_HIDDEN),
          takeW(64, QWEN35_HIDDEN),
          takeW(QWEN35_HIDDEN, 4096),
          takeW(QWEN35_INTERMEDIATE, QWEN35_HIDDEN),
          takeW(QWEN35_INTERMEDIATE, QWEN35_HIDDEN),
          takeW(QWEN35_HIDDEN, QWEN35_INTERMEDIATE),
        ];
        this.layerOffsets.push({ isFullAttn: false, dnIdx: dnIdx++, faIdx: 0, w });
      } else {
        const w = [
          takeW(8192, QWEN35_HIDDEN),
          takeW(1024, QWEN35_HIDDEN),
          takeW(1024, QWEN35_HIDDEN),
          takeW(QWEN35_HIDDEN, 4096),
          takeW(QWEN35_INTERMEDIATE, QWEN35_HIDDEN),
          takeW(QWEN35_INTERMEDIATE, QWEN35_HIDDEN),
          takeW(QWEN35_HIDDEN, QWEN35_INTERMEDIATE),
        ];
        this.layerOffsets.push({ isFullAttn: true, dnIdx: 0, faIdx: faIdx++, w });
      }
    }

      if (wOff0 * 4 !== manifest.slab0Bytes || wOff1 * 4 !== manifest.slab1Bytes) {
        throw new Error(
          `Slab size mismatch: planned (${wOff0 * 4}, ${wOff1 * 4}) vs manifest (${manifest.slab0Bytes}, ${manifest.slab1Bytes})`,
        );
      }
      if (this.auxLayout.totalF32s * 4 !== manifest.auxBytes) {
        throw new Error(
          `Aux size mismatch: planned (${this.auxLayout.totalF32s * 4}) vs manifest (${manifest.auxBytes})`,
        );
      }

      this.buildStaticDecodePlan();
      this.weightMb =
        (manifest.slab0Bytes + manifest.slab1Bytes + manifest.slab2Bytes + manifest.auxBytes) /
        (1024 * 1024);

      await this.resetState();
      options.onProgress?.('Ready', 100);

      return {
        coldLoadMs: performance.now() - loadStart,
        repackMs: 0,
        opfsReadMs,
        pipelineCompileMs,
        loadedFromCache: true,
        loadSource: 'bundled',
        weightMb: this.weightMb,
        maxSeqCapacity: this.maxSeqCapacity,
        hasShaderF16,
        hasSubgroups,
        maxBufferSize,
      };
    } catch (err) {
      this.dispose();
      throw err;
    }
  }

  private buildStaticDecodePlan(): void {
    const device = this.device!;
    const uniformTable = new ArrayBuffer(UNIFORM_MAX_SLOTS * 256);
    const u32View = new Uint32Array(uniformTable);
    const f32View = new Float32Array(uniformTable);

    this.decodeSteps = [];
    this.decodeLayerEndStepIndices = [];
    this.decodeRopePosByteOffsets = [];
    this.decodeGqaPosByteOffsets = [];
    let nextSlot = 0;

    const push = (
      pipeline: unknown,
      bindGroup: unknown,
      vals: [number, number, number, number, number, number, number, number],
      wx: number,
      wy = 1,
      isFloat5 = false,
    ): number => {
      const slot = nextSlot++;
      const byteOff = slot * 256;
      const uIdx = byteOff / 4;
      for (let i = 0; i < 8; i++) {
        if (isFloat5 && i === 5) f32View[uIdx + i] = vals[i];
        else u32View[uIdx + i] = vals[i];
      }
      this.decodeSteps.push({ pipeline, bindGroup, uniformByteOffset: byteOff, wx, wy, wz: 1 });
      return byteOff;
    };

    const gemvWx = (rows: number) => Math.ceil(rows / 4);

    // 0. Embedding lookup for single token
    push(this.embedPipe, this.bgEmbed, [1, QWEN35_HIDDEN, 0, 0, 0, 0, 0, 0], 1, 80);

    for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
      const la = this.auxLayout.layers[l];
      const lo = this.layerOffsets[l];
      const bgNormToWork = l < 16 ? this.bgW0NormToWork : this.bgW1NormToWork;
      const bgAttnToHidden = l < 16 ? this.bgW0AttnToHidden : this.bgW1AttnToHidden;

      // Pre-attn RMSNorm
      push(this.rmsNormPipe, this.bgRmsNorm, [1, QWEN35_HIDDEN, la.inputNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);

      if (!lo.isFullAttn) {
        const [wQkv, wZ, wAb, wOut, wGate, wUp, wDown] = lo.w;
        push(this.gemvPipe, bgNormToWork, [8192, QWEN35_HIDDEN, wQkv, 0, 0, 0, 0, 0], gemvWx(8192));
        push(this.gemvPipe, bgNormToWork, [4096, QWEN35_HIDDEN, wZ, 0, 8192, 0, 0, 0], gemvWx(4096));
        push(this.gemvPipe, bgNormToWork, [64, QWEN35_HIDDEN, wAb, 0, 12288, 0, 0, 0], gemvWx(64));
        push(this.convPipe, this.bgConv, [1, la.convWOff, lo.dnIdx * 24576, 0, 0, 0, 0, 0], 64);
        push(this.gdnRecPipe, this.bgGdnRec, [1, lo.dnIdx * 524288, la.dnAuxOff, 0, 8192, 12288, 0, 0], 32);
        push(this.gemvPipe, bgAttnToHidden, [QWEN35_HIDDEN, 4096, wOut, 0, 0, 1, 0, 0], gemvWx(QWEN35_HIDDEN));

        push(this.rmsNormPipe, this.bgRmsNorm, [1, QWEN35_HIDDEN, la.postNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);
        push(this.gemvPipe, bgNormToWork, [QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wGate, 0, 0, 0, 0, 0], gemvWx(QWEN35_INTERMEDIATE));
        push(this.gemvPipe, bgNormToWork, [QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wUp, 0, QWEN35_INTERMEDIATE, 0, 0, 0], gemvWx(QWEN35_INTERMEDIATE));
        push(this.swigluPipe, this.bgSwiGlu, [QWEN35_INTERMEDIATE, 0, QWEN35_INTERMEDIATE, 0, 0, 0, 0, 0], Math.ceil(QWEN35_INTERMEDIATE / 256));
        push(this.gemvPipe, bgAttnToHidden, [QWEN35_HIDDEN, QWEN35_INTERMEDIATE, wDown, 0, 0, 1, 0, 0], gemvWx(QWEN35_HIDDEN));
      } else {
        const [wQg, wK, wV, wO, wGate, wUp, wDown] = lo.w;
        push(this.gemvPipe, bgNormToWork, [8192, QWEN35_HIDDEN, wQg, 0, 0, 0, 0, 0], gemvWx(8192));
        push(this.gemvPipe, bgNormToWork, [1024, QWEN35_HIDDEN, wK, 0, 8192, 0, 0, 0], gemvWx(1024));
        push(this.gemvPipe, bgNormToWork, [1024, QWEN35_HIDDEN, wV, 0, 9216, 0, 0, 0], gemvWx(1024));

        const ropeSlot = push(
          this.qkRopePipe,
          this.bgQkRope,
          [1, 0, 0, 8192, la.qNormOff, la.kNormOff, this.auxLayout.invFreqOff, 0],
          20,
          1,
        );
        this.decodeRopePosByteOffsets.push(ropeSlot);

        const inKv0 = lo.faIdx < 4;
        const localKv = inKv0 ? lo.faIdx : lo.faIdx - 4;
        const kvBaseU32 = localKv * (this.kvSingleBytes / 4);
        const gqaSlot = push(
          this.gqaSplitPipe,
          inKv0 ? this.bgGqaSplit0 : this.bgGqaSplit1,
          [kvBaseU32, this.maxSeqCapacity, 0, 0, 8192, 9216, GQA_NUM_SPLITS, 512],
          16,
          GQA_NUM_SPLITS,
        );
        this.decodeGqaPosByteOffsets.push(gqaSlot);

        push(this.gqaReducePipe, this.bgGqaReduce, [GQA_NUM_SPLITS, 256, 0, 512, 0, 0, 0, 0], 16);
        push(this.gemvPipe, bgAttnToHidden, [QWEN35_HIDDEN, 4096, wO, 0, 0, 1, 0, 0], gemvWx(QWEN35_HIDDEN));

        push(this.rmsNormPipe, this.bgRmsNorm, [1, QWEN35_HIDDEN, la.postNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);
        push(this.gemvPipe, bgNormToWork, [QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wGate, 0, 0, 0, 0, 0], gemvWx(QWEN35_INTERMEDIATE));
        push(this.gemvPipe, bgNormToWork, [QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wUp, 0, QWEN35_INTERMEDIATE, 0, 0, 0], gemvWx(QWEN35_INTERMEDIATE));
        push(this.swigluPipe, this.bgSwiGlu, [QWEN35_INTERMEDIATE, 0, QWEN35_INTERMEDIATE, 0, 0, 0, 0, 0], Math.ceil(QWEN35_INTERMEDIATE / 256));
        push(this.gemvPipe, bgAttnToHidden, [QWEN35_HIDDEN, QWEN35_INTERMEDIATE, wDown, 0, 0, 1, 0, 0], gemvWx(QWEN35_HIDDEN));
      }
      if (l < QWEN35_NUM_LAYERS - 1) {
        this.decodeLayerEndStepIndices.push(this.decodeSteps.length);
      }
    }

    // Final RMSNorm + LM Head
    push(this.rmsNormPipe, this.bgRmsNorm, [1, QWEN35_HIDDEN, this.auxLayout.finalNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);
    push(this.gemvPipe, this.bgLmHead, [QWEN35_VOCAB_SIZE, QWEN35_HIDDEN, 0, 0, 0, 0, 0, 0], gemvWx(QWEN35_VOCAB_SIZE));
    this.decodeLayerEndStepIndices.push(this.decodeSteps.length);
    if (this.decodeLayerEndStepIndices.length !== QWEN35_NUM_LAYERS) {
      throw new Error(
        `Invalid decode layer boundary table length: expected ${QWEN35_NUM_LAYERS}, got ${this.decodeLayerEndStepIndices.length}.`,
      );
    }

    device.queue.writeBuffer(this.decodeUniformBuf, 0, uniformTable);
  }

  async resetState(): Promise<void> {
    this.lastDecodeGpuMs = 0;
    if (!this.device) {
      this.seqPos = 0;
      return;
    }
    const enc = this.device.createCommandEncoder();
    if (typeof enc.clearBuffer === 'function') {
      enc.clearBuffer(this.dnStateBuf);
      enc.clearBuffer(this.dnConvBuf);
    } else {
      const zeros = new Uint8Array(this.dnSingleBytes);
      for (let i = 0; i < 24; i++) {
        this.device.queue.writeBuffer(this.dnStateBuf, i * this.dnSingleBytes, zeros);
        this.device.queue.writeBuffer(this.dnConvBuf, i * this.dnConvSingleBytes, zeros.subarray(0, this.dnConvSingleBytes));
      }
    }
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this.seqPos = 0;
  }

  async snapshotState(): Promise<WgslCpuSnapshot> {
    if (!this.device) {
      throw new Error('WGSL engine not loaded.');
    }
    const rowBytes = this.kvRowU32s * 4;
    const perSeqKvBytes = 8 * 2 * 4 * this.seqPos * rowBytes;
    const totalBytes = 24 * (this.dnSingleBytes + this.dnConvSingleBytes) + perSeqKvBytes;
    if (totalBytes > MAX_SNAPSHOT_BYTES) {
      throw new Error(
        `Snapshot skipped: hybrid state (${(totalBytes / (1024 * 1024)).toFixed(1)} MiB) exceeds 512 MiB cap.`,
      );
    }

    const readRaw = async (buf: GpuBufferLike, byteOffset: number, byteLen: number): Promise<Uint8Array> => {
      if (byteLen === 0) return new Uint8Array(0);
      const st = this.device!.createBuffer({ size: byteLen, usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST });
      const enc = this.device!.createCommandEncoder();
      enc.copyBufferToBuffer(buf, byteOffset, st, 0, byteLen);
      this.device!.queue.submit([enc.finish()]);
      await st.mapAsync(GPU_MAP_READ);
      const out = new Uint8Array(st.getMappedRange()).slice();
      st.unmap();
      st.destroy();
      return out;
    };

    const dnState = await readRaw(this.dnStateBuf, 0, 24 * this.dnSingleBytes);
    const dnConv = await readRaw(this.dnConvBuf, 0, 24 * this.dnConvSingleBytes);
    const kvSlices0: Uint8Array[] = [];
    const kvSlices1: Uint8Array[] = [];
    const copyBytes = this.seqPos * rowBytes;

    for (let idx = 0; idx < 4; idx++) {
      const kOff = idx * this.kvSingleBytes;
      const vOff = kOff + this.kvPlaneU32s * 4;
      for (let h = 0; h < 4; h++) {
        const hOff = h * this.maxSeqCapacity * rowBytes;
        kvSlices0.push(await readRaw(this.kvSlab0, kOff + hOff, copyBytes));
        kvSlices0.push(await readRaw(this.kvSlab0, vOff + hOff, copyBytes));
        kvSlices1.push(await readRaw(this.kvSlab1, kOff + hOff, copyBytes));
        kvSlices1.push(await readRaw(this.kvSlab1, vOff + hOff, copyBytes));
      }
    }

    return { seqPos: this.seqPos, kvMode: this.kvMode, dnState, dnConv, kvSlices0, kvSlices1 };
  }

  async restoreState(snap: WgslCpuSnapshot): Promise<void> {
    if (!this.device || snap.kvMode !== this.kvMode) {
      throw new Error('Incompatible WGSL state snapshot.');
    }
    await this.resetState();
    this.device.queue.writeBuffer(this.dnStateBuf, 0, snap.dnState as BufferSource);
    this.device.queue.writeBuffer(this.dnConvBuf, 0, snap.dnConv as BufferSource);
    const rowBytes = this.kvRowU32s * 4;
    let cursor = 0;
    for (let idx = 0; idx < 4; idx++) {
      const kOff = idx * this.kvSingleBytes;
      const vOff = kOff + this.kvPlaneU32s * 4;
      for (let h = 0; h < 4; h++) {
        const hOff = h * this.maxSeqCapacity * rowBytes;
        const k0 = snap.kvSlices0[cursor];
        const v0 = snap.kvSlices0[cursor + 1];
        const k1 = snap.kvSlices1[cursor];
        const v1 = snap.kvSlices1[cursor + 1];
        if (k0.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab0, kOff + hOff, k0 as BufferSource);
        if (v0.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab0, vOff + hOff, v0 as BufferSource);
        if (k1.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab1, kOff + hOff, k1 as BufferSource);
        if (v1.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab1, vOff + hOff, v1 as BufferSource);
        cursor += 2;
      }
    }
    await this.device.queue.onSubmittedWorkDone();
    this.seqPos = snap.seqPos;
  }

  private async readFullLogits(): Promise<Float32Array> {
    return readGpuF32Slice(this.device!, this.logitsBuf, 0, QWEN35_VOCAB_SIZE * 4);
  }

  private async prefillChunk(
    chunkTokens: number[],
    computeLogits: boolean,
    sampling: SamplingOptions,
    pacingOptions?: {
      pacingMode?: GpuPacingMode;
      abortSignal?: AbortSignal;
      burstState?: { gpuBurstMs: number };
    },
  ): Promise<number | null> {
    const device = this.device!;
    const M = chunkTokens.length;
    if (M === 0) return null;
    if (this.seqPos + M > this.maxSeqCapacity) {
      throw new Error(`Context overflow: seqPos=${this.seqPos} + ${M} > maxSeqCapacity=${this.maxSeqCapacity}`);
    }

    const startPos = this.seqPos;
    device.queue.writeBuffer(this.tokenIdsBuf, 0, new Uint32Array(chunkTokens));

    const uniformTable = new ArrayBuffer(UNIFORM_MAX_SLOTS * 256);
    const u32View = new Uint32Array(uniformTable);
    const f32View = new Float32Array(uniformTable);
    const steps: DispatchStep[] = [];
    const layerSplitEnds: number[] = [];
    let nextSlot = 0;

    const push = (
      pipeline: unknown,
      bindGroup: unknown,
      vals: [number, number, number, number, number, number, number, number],
      wx: number,
      wy = 1,
      wz = 1,
      isFloat5 = false,
    ) => {
      const slot = nextSlot++;
      const byteOff = slot * 256;
      const uIdx = byteOff / 4;
      for (let i = 0; i < 8; i++) {
        if (isFloat5 && i === 5) f32View[uIdx + i] = vals[i];
        else u32View[uIdx + i] = vals[i];
      }
      steps.push({ pipeline, bindGroup, uniformByteOffset: byteOff, wx, wy, wz });
    };

    const mTiles = Math.ceil(M / 32);
    const prefillSplits = startPos + M <= 512 ? 1 : startPos + M <= 4096 ? 4 : PREFILL_NUM_SPLITS;

    push(this.embedPipe, this.pBgEmbed, [M, QWEN35_HIDDEN, 0, 0, 0, 0, 0, 0], M, 80);

    for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
      const la = this.auxLayout.layers[l];
      const lo = this.layerOffsets[l];
      const bgNormToWork = l < 16 ? this.pBgW0NormToWork : this.pBgW1NormToWork;
      const bgAttnToHidden = l < 16 ? this.pBgW0AttnToHidden : this.pBgW1AttnToHidden;

      push(this.rmsNormPipe, this.pBgRmsNorm, [M, QWEN35_HIDDEN, la.inputNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);

      if (!lo.isFullAttn) {
        const [wQkv, wZ, wAb, wOut, wGate, wUp, wDown] = lo.w;
        push(this.gemmPipe, bgNormToWork, [M, 8192, QWEN35_HIDDEN, wQkv, 0, 0, 0, 0], Math.ceil(8192 / 32), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, 4096, QWEN35_HIDDEN, wZ, 0, M * 8192, 0, 0], Math.ceil(4096 / 32), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, 64, QWEN35_HIDDEN, wAb, 0, M * 12288, 0, 0], Math.ceil(64 / 32), mTiles);
        push(this.convPipe, this.pBgConv, [M, la.convWOff, lo.dnIdx * 24576, 0, 0, 0, 0, 0], 64);
        push(this.gdnRecPipe, this.pBgGdnRec, [M, lo.dnIdx * 524288, la.dnAuxOff, 0, M * 8192, M * 12288, 0, 0], 32);
        push(this.gemmPipe, bgAttnToHidden, [M, QWEN35_HIDDEN, 4096, wOut, 0, 0, 1, 0], Math.ceil(QWEN35_HIDDEN / 32), mTiles);

        push(this.rmsNormPipe, this.pBgRmsNorm, [M, QWEN35_HIDDEN, la.postNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        push(this.gemmPipe, bgNormToWork, [M, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wGate, 0, 0, 0, 0], Math.ceil(QWEN35_INTERMEDIATE / 32), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wUp, 0, M * QWEN35_INTERMEDIATE, 0, 0], Math.ceil(QWEN35_INTERMEDIATE / 32), mTiles);
        push(this.swigluPipe, this.pBgSwiGlu, [M * QWEN35_INTERMEDIATE, 0, M * QWEN35_INTERMEDIATE, 0, 0, 0, 0, 0], Math.ceil((M * QWEN35_INTERMEDIATE) / 256));
        push(this.gemmPipe, bgAttnToHidden, [M, QWEN35_HIDDEN, QWEN35_INTERMEDIATE, wDown, 0, 0, 1, 0], Math.ceil(QWEN35_HIDDEN / 32), mTiles);
      } else {
        const [wQg, wK, wV, wO, wGate, wUp, wDown] = lo.w;
        push(this.gemmPipe, bgNormToWork, [M, 8192, QWEN35_HIDDEN, wQg, 0, 0, 0, 0], Math.ceil(8192 / 32), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, 1024, QWEN35_HIDDEN, wK, 0, M * 8192, 0, 0], Math.ceil(1024 / 32), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, 1024, QWEN35_HIDDEN, wV, 0, M * 9216, 0, 0], Math.ceil(1024 / 32), mTiles);
        push(
          this.qkRopePipe,
          this.pBgQkRope,
          [M, startPos, 0, M * 8192, la.qNormOff, la.kNormOff, this.auxLayout.invFreqOff, 0],
          20,
          M,
        );

        const inKv0 = lo.faIdx < 4;
        const localKv = inKv0 ? lo.faIdx : lo.faIdx - 4;
        const kvBaseU32 = localKv * (this.kvSingleBytes / 4);
        const bgGqaPrefill = inKv0 ? this.bgGqaPrefill0 : this.bgGqaPrefill1;
        push(this.gqaPrefillPipe, bgGqaPrefill, [0, kvBaseU32, this.maxSeqCapacity, startPos, M, M * 8192, M * 9216, 0], 4, M);
        push(this.gqaPrefillPipe, bgGqaPrefill, [1, kvBaseU32, this.maxSeqCapacity, startPos, M, 0, 0, prefillSplits], 4, Math.ceil(M / 2), prefillSplits);
        push(this.gqaReducePipe, this.pBgGqaReduce, [prefillSplits, 256, 0, 512, 0, 0, 0, 0], 16, M);
        push(this.gemmPipe, bgAttnToHidden, [M, QWEN35_HIDDEN, 4096, wO, 0, 0, 1, 0], Math.ceil(QWEN35_HIDDEN / 32), mTiles);

        push(this.rmsNormPipe, this.pBgRmsNorm, [M, QWEN35_HIDDEN, la.postNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        push(this.gemmPipe, bgNormToWork, [M, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wGate, 0, 0, 0, 0], Math.ceil(QWEN35_INTERMEDIATE / 32), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, wUp, 0, M * QWEN35_INTERMEDIATE, 0, 0], Math.ceil(QWEN35_INTERMEDIATE / 32), mTiles);
        push(this.swigluPipe, this.pBgSwiGlu, [M * QWEN35_INTERMEDIATE, 0, M * QWEN35_INTERMEDIATE, 0, 0, 0, 0, 0], Math.ceil((M * QWEN35_INTERMEDIATE) / 256));
        push(this.gemmPipe, bgAttnToHidden, [M, QWEN35_HIDDEN, QWEN35_INTERMEDIATE, wDown, 0, 0, 1, 0], Math.ceil(QWEN35_HIDDEN / 32), mTiles);
      }
      if (l % 4 === 3) {
        layerSplitEnds.push(steps.length);
      }
    }

    const greedy = sampling.temperature <= 0 || sampling.topK === 1;
    if (computeLogits) {
      push(
        this.rmsNormPipe,
        this.pBgRmsNorm,
        [1, QWEN35_HIDDEN, this.auxLayout.finalNormOff, (M - 1) * QWEN35_HIDDEN, 0, 1e-6, 0, 0],
        1,
        1,
        1,
        true,
      );
      push(
        this.gemvPipe,
        this.pBgLmHead,
        [QWEN35_VOCAB_SIZE, QWEN35_HIDDEN, 0, 0, 0, 0, 0, 0],
        Math.ceil(QWEN35_VOCAB_SIZE / 4),
      );
    }

    const pacingMode = pacingOptions?.pacingMode ?? 'balanced';
    const burstState = pacingOptions?.burstState ?? { gpuBurstMs: 0 };

    device.queue.writeBuffer(this.prefillUniformBuf, 0, uniformTable);
    let stepCursor = 0;
    for (let sIdx = 0; sIdx < layerSplitEnds.length; sIdx++) {
      if (pacingOptions?.abortSignal?.aborted) {
        throw new Error('Generation aborted');
      }
      const endIdx = layerSplitEnds[sIdx];
      const isLastSlice = sIdx === layerSplitEnds.length - 1;
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (; stepCursor < endIdx; stepCursor++) {
        const st = steps[stepCursor];
        pass.setPipeline(st.pipeline);
        pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
        pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
      }
      if (isLastSlice) {
        for (; stepCursor < steps.length; stepCursor++) {
          const st = steps[stepCursor];
          pass.setPipeline(st.pipeline);
          pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
          pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
        }
        if (computeLogits && greedy) {
          pass.setPipeline(this.arg1Pipe);
          pass.setBindGroup(0, this.bgArg1);
          pass.dispatchWorkgroups(Math.ceil(QWEN35_VOCAB_SIZE / 256), 1, 1);
          pass.setPipeline(this.arg2Pipe);
          pass.setBindGroup(0, this.bgArg2);
          pass.dispatchWorkgroups(1, 1, 1);
        }
      }
      pass.end();
      if (isLastSlice && computeLogits && greedy) {
        enc.copyBufferToBuffer(this.tokOutBuf, 0, this.staging4B, 0, 4);
        const tSlice = performance.now();
        device.queue.submit([enc.finish()]);
        await this.staging4B.mapAsync(GPU_MAP_READ);
        burstState.gpuBurstMs += performance.now() - tSlice;
        const tokId = new Uint32Array(this.staging4B.getMappedRange())[0];
        this.staging4B.unmap();
        this.seqPos += M;
        return tokId;
      }
      const tSlice = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      burstState.gpuBurstMs += performance.now() - tSlice;
      const { breatherMs, nextBurstMs } = computePrefillSliceBreatherMs(
        pacingMode,
        burstState.gpuBurstMs,
      );
      burstState.gpuBurstMs = nextBurstMs;
      if (breatherMs > 0) {
        await sleepMs(breatherMs);
      }
    }

    this.seqPos += M;
    if (!computeLogits) {
      return null;
    }
    const logits = await this.readFullLogits();
    return sampleNextToken(logits, sampling);
  }

  async prefill(
    tokenIds: number[],
    options: {
      chunkSize: number;
      sampling: SamplingOptions;
      pacingMode?: GpuPacingMode;
      yieldFn: () => Promise<void>;
      abortSignal?: AbortSignal;
      onChunk?: (done: number, total: number) => void;
    },
  ): Promise<{ nextTokenId: number; prefillMs: number }> {
    const startMs = performance.now();
    const pacingMode = options.pacingMode ?? 'balanced';
    const burstState = { gpuBurstMs: 0 };
    let lastTok: number | null = null;
    let offset = 0;
    let chunkIdx = 0;

    while (offset < tokenIds.length) {
      if (options.abortSignal?.aborted) {
        throw new Error('Generation aborted');
      }
      const stepSize = selectPrefillChunkSize(pacingMode, this.seqPos, options.chunkSize);
      const end = Math.min(tokenIds.length, offset + stepSize);
      const isLast = end === tokenIds.length;
      lastTok = await this.prefillChunk(tokenIds.slice(offset, end), isLast, options.sampling, {
        pacingMode,
        abortSignal: options.abortSignal,
        burstState,
      });
      options.onChunk?.(end, tokenIds.length);
      if (!isLast) {
        await options.yieldFn();
        if (pacingMode === 'max-speed' && chunkIdx % 4 === 3) {
          await sleepMs(0);
        }
      }
      offset = end;
      chunkIdx++;
    }

    if (lastTok === null) {
      throw new Error('Cannot prefill empty tokenIds.');
    }
    return {
      nextTokenId: lastTok,
      prefillMs: Math.max(1, performance.now() - startMs),
    };
  }

  async decodeStep(
    tokenId: number,
    sampling: SamplingOptions,
    pacingMode: GpuPacingMode = 'balanced',
  ): Promise<{ nextToken: number; gpuMs: number; breatherMs: number }> {
    const device = this.device!;
    if (this.seqPos + 1 > this.maxSeqCapacity) {
      throw new Error(`Context overflow at seqPos=${this.seqPos} (maxSeqCapacity=${this.maxSeqCapacity})`);
    }
    device.queue.writeBuffer(this.tokenIdsBuf, 0, new Uint32Array([tokenId]));
    const posArr = new Uint32Array([this.seqPos]);
    for (const ropeOff of this.decodeRopePosByteOffsets) {
      device.queue.writeBuffer(this.decodeUniformBuf, ropeOff + 4, posArr);
    }
    for (const gqaOff of this.decodeGqaPosByteOffsets) {
      device.queue.writeBuffer(this.decodeUniformBuf, gqaOff + 8, posArr);
    }

    const greedy = sampling.temperature <= 0 || sampling.topK === 1;
    const numSlices = selectDecodeSliceCount(pacingMode, this.seqPos, this.lastDecodeGpuMs);
    const bounds = computeDecodeSliceBounds(this.decodeLayerEndStepIndices, numSlices);
    if (bounds.length === 0) {
      throw new Error('Decode dispatch plan is empty.');
    }
    const policy = getGpuPacingPolicy(pacingMode);
    let gpuMs = 0;
    let breatherMs = 0;
    let greedyTokId: number | null = null;

    for (let s = 0; s < bounds.length; s++) {
      const slice = bounds[s];
      const isLastSlice = s === bounds.length - 1;
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let stepIdx = slice.start; stepIdx < slice.end; stepIdx++) {
        const st = this.decodeSteps[stepIdx];
        pass.setPipeline(st.pipeline);
        pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
        pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
      }
      if (isLastSlice && greedy) {
        pass.setPipeline(this.arg1Pipe);
        pass.setBindGroup(0, this.bgArg1);
        pass.dispatchWorkgroups(Math.ceil(QWEN35_VOCAB_SIZE / 256), 1, 1);
        pass.setPipeline(this.arg2Pipe);
        pass.setBindGroup(0, this.bgArg2);
        pass.dispatchWorkgroups(1, 1, 1);
      }
      pass.end();

      if (isLastSlice && greedy) {
        enc.copyBufferToBuffer(this.tokOutBuf, 0, this.staging4B, 0, 4);
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await this.staging4B.mapAsync(GPU_MAP_READ);
        gpuMs += performance.now() - t0;
        greedyTokId = new Uint32Array(this.staging4B.getMappedRange())[0];
        this.staging4B.unmap();
      } else {
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        gpuMs += performance.now() - t0;
      }

      if (s < bounds.length - 1 && policy.decodeSliceBreatherMs > 0) {
        const b0 = performance.now();
        await sleepMs(policy.decodeSliceBreatherMs);
        breatherMs += performance.now() - b0;
      }
    }

    this.seqPos += 1;
    this.lastDecodeGpuMs = gpuMs;

    const nextToken =
      greedyTokId !== null ? greedyTokId : sampleNextToken(await this.readFullLogits(), sampling);

    const postMs = computeDecodePostStepBreatherMs(pacingMode, gpuMs);
    if (postMs > 0) {
      const b0 = performance.now();
      await sleepMs(postMs);
      breatherMs += performance.now() - b0;
    }

    return { nextToken, gpuMs, breatherMs };
  }

  private async verifyRealKernels(
    device: GpuDeviceLike,
    dynStorage4Bgl: unknown,
    gdnRecBgl: unknown,
    qkRopeBgl: unknown,
    dynRwStorageBgl: unknown,
  ): Promise<void> {
    const wasm = this.wasm!;
    const tempBufs: GpuBufferLike[] = [];
    const mk = (size: number, usage: number) => {
      const b = device.createBuffer({ size, usage });
      tempBufs.push(b);
      return b;
    };

    try {
      // 1. Verify batched WGSL_Q4_GEMM_F16 (M=7, N=40, K=96 -> non-32-aligned M & N!)
      {
        const M = 7;
        const N = 40;
        const K = 96;
        const blocks = K / 32;
        const wU32Len = N * blocks * 5;
        const scratchBase = wasm.wasm_scratch_ptr();
        const wPtr = scratchBase;
        const xPtr = wPtr + wU32Len * 4;
        const refOutPtr = xPtr + K * 4;
        const wCpu = new Uint32Array(wasm.memory.buffer, wPtr, wU32Len);
        for (let b = 0; b < N * blocks; b++) {
          const i = b * 5;
          wCpu[i] = (0x12345678 + b * 17) >>> 0;
          wCpu[i + 1] = (0x87654321 ^ (b * 29)) >>> 0;
          wCpu[i + 2] = (0x0f1e2d3c + b * 11) >>> 0;
          wCpu[i + 3] = (0x5a6b7c8d ^ (b * 19)) >>> 0;
          wCpu[i + 4] = 0xc666399a; // f16 scale ~0.015 (0x21ec?) -> let's use exact bits 0x2b80 (~0.0117) and -0.09375 (0xae00)
          wCpu[i + 4] = (0x2b80 | (0xae00 << 16)) >>> 0;
        }
        const xFull = new Float32Array(M * K);
        for (let t = 0; t < M; t++) {
          for (let k = 0; k < K; k++) xFull[t * K + k] = Math.sin(t * 0.4 + k * 0.11) * 0.6;
        }

        const wGpu = mk(wU32Len * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const xGpu = mk(xFull.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const yGpu = mk(M * N * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uGpu = mk(256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
        device.queue.writeBuffer(wGpu, 0, new Uint32Array(wasm.memory.buffer, wPtr, wU32Len));
        device.queue.writeBuffer(xGpu, 0, xFull);
        device.queue.writeBuffer(uGpu, 0, new Uint32Array([M, N, K, 0, 0, 0, 0, 0]));
        const bg = device.createBindGroup({
          layout: dynStorage4Bgl,
          entries: [
            { binding: 0, resource: { buffer: uGpu, size: 32 } },
            { binding: 1, resource: { buffer: wGpu } },
            { binding: 2, resource: { buffer: xGpu } },
            { binding: 3, resource: { buffer: yGpu } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.gemmPipe);
        pass.setBindGroup(0, bg, [0]);
        pass.dispatchWorkgroups(Math.ceil(N / 32), Math.ceil(M / 32), 1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const gpuOut = await readGpuF32Slice(device, yGpu, 0, M * N * 4);

        let maxErr = 0;
        for (let t = 0; t < M; t++) {
          new Float32Array(wasm.memory.buffer, xPtr, K).set(xFull.subarray(t * K, (t + 1) * K));
          wasm.cpu_ref_q4_gemv(N, K, wPtr, xPtr, refOutPtr);
          const refRow = new Float32Array(wasm.memory.buffer, refOutPtr, N);
          for (let r = 0; r < N; r++) {
            maxErr = Math.max(maxErr, Math.abs(gpuOut[t * N + r] - refRow[r]));
          }
        }
        if (maxErr > 0.05) {
          throw new Error(`WGSL_Q4_GEMM_F16 verification failed: maxAbsError=${maxErr}`);
        }
      }

      // 2. Verify WGSL_GDN_RECURRENT (M=2, L2 norm + Gated RMSNorm + softplus decay) vs TS CPU ref
      {
        const M = 2;
        const workCpu = new Float32Array(M * 12352);
        for (let t = 0; t < M; t++) {
          for (let i = 0; i < 128; i++) {
            workCpu[t * 8192 + i] = Math.sin(t + i * 0.07) * 0.8; // q head 0
            workCpu[t * 8192 + 2048 + i] = Math.cos(t * 0.5 + i * 0.05) * 0.7; // k head 0
            workCpu[t * 8192 + 4096 + i] = Math.sin(t * 0.9 + i * 0.03) * 0.6; // v head 0
            workCpu[M * 8192 + t * 4096 + i] = Math.cos(i * 0.04 - t) * 0.5; // z head 0
          }
          workCpu[M * 12288 + t * 64 + 0] = -0.3 + t * 0.2; // b
          workCpu[M * 12288 + t * 64 + 32] = 0.4 - t * 0.3; // a
        }
        const auxCpu = new Float32Array(192);
        auxCpu[0] = -0.15; // A_neg_exp[0]
        auxCpu[32] = 0.25; // dt_bias[0]
        for (let i = 0; i < 128; i++) auxCpu[64 + i] = 0.85 + 0.1 * Math.sin(i);

        // Exact TS reference for head 0
        const stateRef = new Float32Array(128 * 128);
        const outRef = new Float32Array(128);
        for (let t = 0; t < M; t++) {
          let qSq = 0;
          let kSq = 0;
          for (let i = 0; i < 128; i++) {
            const qv = workCpu[t * 8192 + i];
            const kv = workCpu[t * 8192 + 2048 + i];
            qSq += qv * qv;
            kSq += kv * kv;
          }
          const invQ = (1 / Math.sqrt(qSq + 1e-6)) * (1 / Math.sqrt(128));
          const invK = 1 / Math.sqrt(kSq + 1e-6);
          const qn = new Float32Array(128);
          const kn = new Float32Array(128);
          for (let i = 0; i < 128; i++) {
            qn[i] = workCpu[t * 8192 + i] * invQ;
            kn[i] = workCpu[t * 8192 + 2048 + i] * invK;
          }
          const beta = 1 / (1 + Math.exp(-workCpu[M * 12288 + t * 64]));
          const u = workCpu[M * 12288 + t * 64 + 32] + auxCpu[32];
          const sp = u > 20 ? u : Math.log(1 + Math.exp(u));
          const decay = Math.exp(auxCpu[0] * sp);

          const rawOut = new Float32Array(128);
          for (let j = 0; j < 128; j++) {
            let sk = 0;
            for (let i = 0; i < 128; i++) sk += stateRef[i * 128 + j] * decay * kn[i];
            const delta = (workCpu[t * 8192 + 4096 + j] - sk) * beta;
            let o = 0;
            for (let i = 0; i < 128; i++) {
              const sn = stateRef[i * 128 + j] * decay + kn[i] * delta;
              stateRef[i * 128 + j] = sn;
              o += sn * qn[i];
            }
            rawOut[j] = o;
          }
          let oSq = 0;
          for (let j = 0; j < 128; j++) oSq += rawOut[j] * rawOut[j];
          const invRms = 1 / Math.sqrt(oSq / 128 + 1e-6);
          for (let j = 0; j < 128; j++) {
            const zv = workCpu[M * 8192 + t * 4096 + j];
            outRef[j] = rawOut[j] * invRms * (zv / (1 + Math.exp(-zv))) * auxCpu[64 + j];
          }
        }

        const auxG = mk(auxCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const workG = mk(workCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const stG = mk(32 * 128 * 128 * 4, GPU_USAGE.STORAGE);
        const outG = mk(M * 4096 * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uG = mk(256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
        device.queue.writeBuffer(auxG, 0, auxCpu);
        device.queue.writeBuffer(workG, 0, workCpu);
        device.queue.writeBuffer(uG, 0, new Uint32Array([M, 0, 0, 0, M * 8192, M * 12288, 0, 0]));
        const bg = device.createBindGroup({
          layout: gdnRecBgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: auxG } },
            { binding: 2, resource: { buffer: workG } },
            { binding: 3, resource: { buffer: stG } },
            { binding: 4, resource: { buffer: outG } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.gdnRecPipe);
        pass.setBindGroup(0, bg, [0]);
        pass.dispatchWorkgroups(32, 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const gpuOut = await readGpuF32Slice(device, outG, 4096 * 4, 128 * 4);
        let maxErr = 0;
        for (let i = 0; i < 128; i++) maxErr = Math.max(maxErr, Math.abs(gpuOut[i] - outRef[i]));
        if (maxErr > 1e-4) {
          throw new Error(`WGSL_GDN_RECURRENT verification failed: maxAbsError=${maxErr}`);
        }
      }

      // 3. Verify WGSL_QK_NORM_ROPE + 8-Query GQA-Tiled WGSL_GQA_*_PREFILL (odd M=31, both tok=29 & tok=30, qHead=0 & qHead=3)
      {
        const M = 31;
        const startPos = 0;
        const maxSeq = 64;
        const invFreq = new Float32Array(32);
        for (let d = 0; d < 32; d++) invFreq[d] = Math.pow(10000000, (-2 * d) / 64);
        const qNorm = new Float32Array(256).fill(1.2);
        const kNorm = new Float32Array(256).fill(0.9);
        const auxCpu = new Float32Array(32 + 256 + 256);
        auxCpu.set(invFreq, 0);
        auxCpu.set(qNorm, 32);
        auxCpu.set(kNorm, 288);

        const workCpu = new Float32Array(M * 10240);
        for (let i = 0; i < workCpu.length; i++) workCpu[i] = Math.sin(i * 0.031) * 0.4;

        const applyNormRope = (vec: Float32Array, wNorm: Float32Array, posVal: number): Float32Array => {
          let sq = 0;
          for (let d = 0; d < 256; d++) sq += vec[d] * vec[d];
          const inv = 1 / Math.sqrt(sq / 256 + 1e-6);
          const n = new Float32Array(256);
          for (let d = 0; d < 256; d++) n[d] = vec[d] * inv * wNorm[d];
          const out = new Float32Array(256);
          for (let d = 0; d < 256; d++) {
            if (d < 32) {
              const ang = posVal * invFreq[d];
              out[d] = n[d] * Math.cos(ang) - n[d + 32] * Math.sin(ang);
            } else if (d < 64) {
              const ang = posVal * invFreq[d - 32];
              out[d] = n[d] * Math.cos(ang) + n[d - 32] * Math.sin(ang);
            } else {
              out[d] = n[d];
            }
          }
          return out;
        };

        const kRoped: Float32Array[] = [];
        for (let t = 0; t < M; t++) {
          kRoped.push(applyNormRope(workCpu.slice(M * 8192 + t * 1024, M * 8192 + t * 1024 + 256), kNorm, startPos + t));
        }

        const computeHeadRef = (tokCheck: number, qHead: number) => {
          const qBase = tokCheck * 8192 + qHead * 512;
          const qRef = applyNormRope(workCpu.slice(qBase, qBase + 256), qNorm, startPos + tokCheck);
          const scores = new Float32Array(tokCheck + 1);
          let maxS = -Infinity;
          for (let t = 0; t <= tokCheck; t++) {
            let dot = 0;
            for (let d = 0; d < 256; d++) dot += qRef[d] * kRoped[t][d];
            scores[t] = dot / 16;
            if (scores[t] > maxS) maxS = scores[t];
          }
          let lSum = 0;
          const probs = new Float32Array(tokCheck + 1);
          for (let t = 0; t <= tokCheck; t++) {
            probs[t] = Math.exp(scores[t] - maxS);
            lSum += probs[t];
          }
          const attnRef = new Float32Array(256);
          const gateOff = qBase + 256;
          for (let d = 0; d < 256; d++) {
            let acc = 0;
            for (let t = 0; t <= tokCheck; t++) acc += (probs[t] / lSum) * workCpu[M * 9216 + t * 1024 + d];
            attnRef[d] = acc / (1 + Math.exp(-workCpu[gateOff + d]));
          }
          return { qRef, attnRef };
        };

        const numSplits = 4;
        const auxG = mk(auxCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const workG = mk(workCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC);
        const kvBytes = Number(wasm.full_attn_layer_kv_bytes(maxSeq, this.kvMode === 'int8' ? 1 : 0));
        const kvG = mk(kvBytes, GPU_USAGE.STORAGE);
        const splitG = mk(M * 16 * numSplits * 260 * 4, GPU_USAGE.STORAGE);
        const outG = mk(M * 4096 * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uG = mk(4 * 256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);

        device.queue.writeBuffer(auxG, 0, auxCpu);
        device.queue.writeBuffer(workG, 0, workCpu);
        device.queue.writeBuffer(uG, 0, new Uint32Array([M, startPos, 0, M * 8192, 32, 288, 0, 0]));
        device.queue.writeBuffer(uG, 256, new Uint32Array([0, 0, maxSeq, startPos, M, M * 8192, M * 9216, 0]));
        device.queue.writeBuffer(uG, 512, new Uint32Array([1, 0, maxSeq, startPos, M, 0, 0, numSplits]));
        device.queue.writeBuffer(uG, 768, new Uint32Array([numSplits, 256, 0, 512, 0, 0, 0, 0]));

        const bgRope = device.createBindGroup({
          layout: qkRopeBgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: auxG } },
            { binding: 2, resource: { buffer: workG } },
          ],
        });
        const bgPref = device.createBindGroup({
          layout: dynRwStorageBgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: workG } },
            { binding: 2, resource: { buffer: kvG } },
            { binding: 3, resource: { buffer: splitG } },
          ],
        });
        const bgRed = device.createBindGroup({
          layout: dynStorage4Bgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: workG } },
            { binding: 2, resource: { buffer: splitG } },
            { binding: 3, resource: { buffer: outG } },
          ],
        });

        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.qkRopePipe);
        pass.setBindGroup(0, bgRope, [0]);
        pass.dispatchWorkgroups(20, M, 1);
        pass.setPipeline(this.gqaPrefillPipe);
        pass.setBindGroup(0, bgPref, [256]);
        pass.dispatchWorkgroups(4, M, 1);
        pass.setPipeline(this.gqaPrefillPipe);
        pass.setBindGroup(0, bgPref, [512]);
        pass.dispatchWorkgroups(4, Math.ceil(M / 2), numSplits);
        pass.setPipeline(this.gqaReducePipe);
        pass.setBindGroup(0, bgRed, [768]);
        pass.dispatchWorkgroups(16, M, 1);
        pass.end();
        device.queue.submit([enc.finish()]);

        const aTol = this.kvMode === 'int8' ? 0.05 : 0.01;
        for (const tokCheck of [29, 30]) {
          for (const qHead of [0, 3]) {
            const { qRef, attnRef } = computeHeadRef(tokCheck, qHead);
            const gpuQ = await readGpuF32Slice(device, workG, (tokCheck * 8192 + qHead * 512) * 4, 256 * 4);
            let qErr = 0;
            for (let d = 0; d < 256; d++) qErr = Math.max(qErr, Math.abs(gpuQ[d] - qRef[d]));
            if (qErr > 1e-4) {
              throw new Error(`WGSL_QK_NORM_ROPE verification failed (tok=${tokCheck}, head=${qHead}): maxAbsError=${qErr}`);
            }
            const gpuAttn = await readGpuF32Slice(device, outG, (tokCheck * 4096 + qHead * 256) * 4, 256 * 4);
            let aErr = 0;
            for (let d = 0; d < 256; d++) aErr = Math.max(aErr, Math.abs(gpuAttn[d] - attnRef[d]));
            if (aErr > aTol) {
              throw new Error(`WGSL_GQA_${this.kvMode}_PREFILL verification failed (tok=${tokCheck}, head=${qHead}): maxAbsError=${aErr}`);
            }
          }
        }
      }
    } finally {
      for (const b of tempBufs) b.destroy();
    }
  }

  dispose(): void {
    for (const buf of this.allocatedBuffers) {
      try {
        buf.destroy();
      } catch {
        // Ignore already destroyed buffers
      }
    }
    this.allocatedBuffers = [];
    if (this.device) {
      try {
        this.device.destroy();
      } catch {
        // Ignore
      }
      this.device = null;
    }
    this.wasm = null;
    this.decodeSteps = [];
    this.decodeLayerEndStepIndices = [];
    this.seqPos = 0;
    this.lastDecodeGpuMs = 0;
  }
}
