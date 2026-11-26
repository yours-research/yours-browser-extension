import { selectComputeTier } from '@yours/models/hardware';
import type { GpuPacingMode, LoadSource, WgslKvMode } from '../shared/protocol.ts';
import {
  calibratePacingForHardware,
  computeDecodeBurstBreatherMs,
  computeDecodeSliceBounds,
  computePrefillSliceBreatherMs,
  getGpuPacingPolicy,
  resolveExtensionAssetUrl,
  selectDecodeSliceCount,
  selectPrefillChunkSize,
  sleepMs,
} from './gpuPacing.ts';
import {
  computeKvSliceByteOffset,
  computeLayerWeightByteSizes,
  computeTieredKvPlan,
  getDeltaNetConvDim,
  getDeltaNetLayerCount,
  getDeltaNetQkvzAbRows,
  getFullAttnKvHeadU32s,
  getFullAttnLayerCount,
  getFullAttnLayerSet,
  getFullAttnQkvRows,
  partitionVocabRowsIntoSlabs,
  partitionWeightLayersIntoSlabs,
  resolveModelArch,
  type ModelArchConfig,
  type TieredKvPlan,
  type VocabSlabSlice,
  type WeightSlabPartition,
} from './modelConfig.ts';
import {
  computeAuxLayout,
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
  streamUrlToGpuSegments,
  type GpuBufferLike,
  type GpuBufferWriteSegment,
  type GpuDeviceLike,
  type WasmEngineExports,
  type WgslComputeTier,
} from './wgslWebgpu.ts';

export type { WgslComputeTier } from './wgslWebgpu.ts';

const GQA_FP16_NUM_SPLITS = 32;
const GQA_INT8_NUM_SPLITS = 64;
const PREFILL_NUM_SPLITS = 16;
const PREFILL_MAX_CHUNK = 128;
export const MAX_SPEC_DRAFT = 7;
const MAX_SPEC_TOKENS = 1 + MAX_SPEC_DRAFT;
const UNIFORM_MAX_SLOTS = 1024;
const MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024;
const MAX_GPU_PREFIX_VRAM_BYTES = 512 * 1024 * 1024;
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
  allocatedKvCapacity: number;
  maxSafeKvCapacity: number;
  computeTier: WgslComputeTier;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  hasDp4a: boolean;
  usingDp4a: boolean;
  maxBufferSize: number;
}

export interface WgslCpuSnapshot {
  seqPos: number;
  kvMode: WgslKvMode;
  dnState: Uint8Array;
  dnConv: Uint8Array;
  kvSlices0: Uint8Array[];
  kvSlices1: Uint8Array[];
}

interface GpuPrefixSlot {
  slotId: string;
  seqPos: number;
  byteSize: number;
  dnState: GpuBufferLike;
  dnConv: GpuBufferLike;
  kvPacked0: GpuBufferLike | null;
  kvPacked1: GpuBufferLike | null;
}

interface DispatchStep {
  pipeline: unknown;
  bindGroup: unknown;
  uniformByteOffset: number;
  wx: number;
  wy: number;
  wz: number;
}

interface VerifyFaPatchTarget {
  qkUIdx: number;
  gqa0UIdx: number;
  gqa1UIdx: number;
  gqa1Step: DispatchStep;
}

interface VerifyGraphPlan {
  M: number;
  uniformBuf: GpuBufferLike;
  dynCpu: ArrayBuffer;
  dynU32: Uint32Array;
  dynByteStart: number;
  steps: DispatchStep[];
  faPatches: VerifyFaPatchTarget[];
  bgArgBatch1: unknown;
  bgArgBatch2: unknown;
  bgSpecAccept: unknown;
  ownedBuffers: GpuBufferLike[];
}

interface TierShaderSelection {
  embed: number;
  rmsNorm: number;
  swiglu: number;
  gemv: number;
  gemvWide: number;
  gateUpSwiglu: number;
  gemvBatch: number;
  gemm: number;
  conv: number;
  gdnRec: number;
  gdnChunkPrep: number;
  gdnChunkApply: number;
  qkRope: number;
  gqaSplitPlain: number;
  gqaReduce: number;
  gqaPrefill: number;
  arg1: number;
  arg2: number;
}

function getTierShaderIds(tier: WgslComputeTier, kvMode: WgslKvMode): TierShaderSelection {
  const isInt8 = kvMode === 'int8';
  if (tier === 'tier1-f16-subgroups') {
    return {
      embed: 10,
      rmsNorm: 11,
      swiglu: 12,
      gemv: 2,
      gemvWide: 22,
      gateUpSwiglu: 21,
      gemvBatch: 24,
      gemm: 13,
      conv: 14,
      gdnRec: 15,
      gdnChunkPrep: 19,
      gdnChunkApply: 20,
      qkRope: 16,
      gqaSplitPlain: isInt8 ? 5 : 4,
      gqaReduce: 9,
      gqaPrefill: isInt8 ? 18 : 17,
      arg1: 7,
      arg2: 8,
    };
  }
  if (tier === 'tier2-f16-shared') {
    return {
      embed: 10,
      rmsNorm: 34,
      swiglu: 12,
      gemv: 25,
      gemvWide: 29,
      gateUpSwiglu: 27,
      gemvBatch: 31,
      gemm: 13,
      conv: 14,
      gdnRec: 15,
      gdnChunkPrep: 41,
      gdnChunkApply: 42,
      qkRope: 16,
      gqaSplitPlain: isInt8 ? 37 : 35,
      gqaReduce: 9,
      gqaPrefill: isInt8 ? 40 : 38,
      arg1: 7,
      arg2: 8,
    };
  }
  return {
    embed: 10,
    rmsNorm: 34,
    swiglu: 12,
    gemv: 26,
    gemvWide: 30,
    gateUpSwiglu: 28,
    gemvBatch: 32,
    gemm: 33,
    conv: 14,
    gdnRec: 15,
    gdnChunkPrep: 41,
    gdnChunkApply: 42,
    qkRope: 16,
    gqaSplitPlain: isInt8 ? 37 : 36,
    gqaReduce: 9,
    gqaPrefill: isInt8 ? 40 : 39,
    arg1: 7,
    arg2: 8,
  };
}

function getCandidateComputeTiers(
  hasShaderF16: boolean,
  hasSubgroups: boolean,
  forceComputeTier?: WgslComputeTier,
): WgslComputeTier[] {
  const features = [
    ...(hasShaderF16 ? ['shader-f16'] : []),
    ...(hasSubgroups ? ['subgroups'] : []),
  ];
  const primary = forceComputeTier ?? selectComputeTier(features);
  if (primary === 'tier1-f16-subgroups') {
    return ['tier1-f16-subgroups', 'tier2-f16-shared', 'tier3-f32-universal'];
  }
  if (primary === 'tier2-f16-shared') {
    return ['tier2-f16-shared', 'tier3-f32-universal'];
  }
  return ['tier3-f32-universal'];
}

export class WgslQwen35Engine {
  private device: GpuDeviceLike | null = null;
  private wasm: WasmEngineExports | null = null;
  private allocatedBuffers: GpuBufferLike[] = [];
  private prefillUniformScratch: ArrayBuffer | null = null;

  private arch: ModelArchConfig = resolveModelArch();
  private computeTier: WgslComputeTier = 'tier1-f16-subgroups';
  private kvMode: WgslKvMode = 'int8';
  private kvPlan!: TieredKvPlan;
  private allocatedKvCapacity = 33280;
  private maxSafeKvCapacity = 131584;
  private maxSeqCapacity = 33280;
  private maxBufferSize = 0;
  private seqPos = 0;
  private auxLayout!: AuxLayout;

  private weightPartitions: WeightSlabPartition[] = [];
  private wSlabs: GpuBufferLike[] = [];
  private vocabSlices: VocabSlabSlice[] = [];
  private lmHeadSlabs: GpuBufferLike[] = [];

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
  private dnStateBackupBuf!: GpuBufferLike;
  private dnConvBackupBuf!: GpuBufferLike;
  private verifyLogitsBuf!: GpuBufferLike;
  private verifyTokOutBuf!: GpuBufferLike;
  private verifyAcceptOutBuf!: GpuBufferLike;
  private verifyStaging64B!: GpuBufferLike;
  private partValBuf!: GpuBufferLike;
  private partIdxBuf!: GpuBufferLike;

  private embedPipe!: unknown;
  private rmsNormPipe!: unknown;
  private swigluPipe!: unknown;
  private gemvPipe!: unknown;
  private gemvWidePipe!: unknown;
  private gateUpSwigluPipe!: unknown;
  private gemvBatchPipe!: unknown;
  private gemmPipe!: unknown;
  private convPipe!: unknown;
  private gdnRecPipe!: unknown;
  private gdnChunkPrepPipe!: unknown;
  private gdnChunkApplyPipe!: unknown;
  private qkRopePipe!: unknown;
  private gqaSplitPipe!: unknown;
  private gqaSplitPlainPipe!: unknown;
  private gqaReducePipe!: unknown;
  private gqaPrefillPipe!: unknown;
  private arg1Pipe!: unknown;
  private arg2Pipe!: unknown;
  private argmaxBatch1Pipe!: unknown;
  private argmaxBatch2Pipe!: unknown;
  private specAcceptPipe!: unknown;
  private gemvBatchWidePipe: unknown = null;
  private gateUpSwigluBatchPipe: unknown = null;
  private usingDp4a = false;

  // Retained bind group layouts for dynamic KV cache reallocation & verify plan cache
  private dynStorage4Bgl!: unknown;
  private gdnApplyBgl!: unknown;
  private qkRopeBgl!: unknown;
  private gqaSplitBgl!: unknown;
  private dynRwStorageBgl!: unknown;
  private staticBgl!: unknown;
  private staticRead2Bgl!: unknown;
  private verifyPlans = new Map<number, VerifyGraphPlan>();
  private rbUniformScratch = new ArrayBuffer(256 * 256);

  private bgEmbedPerSlice: unknown[] = [];
  private bgRmsNorm!: unknown;
  private bgWSlabNormToWork: unknown[] = [];
  private bgWSlabNormToAttn: unknown[] = [];
  private bgWSlabAttnToHidden: unknown[] = [];
  private bgLmHeadPerSlice: unknown[] = [];
  private bgConv!: unknown;
  private bgGdnRec!: unknown;
  private bgQkRope!: unknown;
  private bgGqaSplit0!: unknown;
  private bgGqaSplit1!: unknown;
  private bgGqaReduce!: unknown;
  private bgGqaPrefill0!: unknown;
  private bgGqaPrefill1!: unknown;

  // Separate prefill bind groups (using prefillUniformBuf) so static decodeUniformBuf stays untouched
  private pBgEmbedPerSlice: unknown[] = [];
  private pBgRmsNorm!: unknown;
  private pBgSwiGlu!: unknown;
  private pBgWSlabNormToWork: unknown[] = [];
  private pBgWSlabAttnToHidden: unknown[] = [];
  private pBgConv!: unknown;
  private pBgGdnRec!: unknown;
  private pBgGdnChunkPrep!: unknown;
  private pBgGdnChunkApply!: unknown;
  private pBgQkRope!: unknown;
  private pBgGqaReduce!: unknown;
  private pBgLmHeadPerSlice: unknown[] = [];
  private bgArg1!: unknown;
  private bgArg2!: unknown;

  private decodeSteps: DispatchStep[] = [];
  private decodeLayerEndStepIndices: number[] = [];
  private lastDecodeGpuMs = 0;
  private emaDecodeStepMs = 0;
  private displayHz = 60;
  private decodeBurstMs = 0;
  private layerOffsets: Array<{
    slabIdx: number;
    isFullAttn: boolean;
    dnIdx: number;
    faIdx: number;
    w: number[];
  }> = [];

  private gpuPrefixSlots = new Map<string, GpuPrefixSlot>();

  private weightMb = 0;
  private numDnLayers = 24;
  private numFaLayers = 8;
  private fa0Count = 4;
  private fa1Count = 4;
  private dnConvDim = 8192;
  private dnValDim = 4096;
  private dnQkvzAbRows = 12352;
  private attnQkvRows = 10240;
  private dnConvPerLayerF32s = 8192 * 3;
  private dnStatePerLayerF32s = 32 * 128 * 128;
  private dnSingleBytes = 32 * 128 * 128 * 4;
  private dnConvSingleBytes = 8192 * 3 * 4;
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

  getComputeTier(): WgslComputeTier {
    return this.computeTier;
  }

  getAllocatedKvCapacity(): number {
    return this.allocatedKvCapacity;
  }

  getMaxSafeKvCapacity(): number {
    return this.maxSafeKvCapacity;
  }

  getArchConfig(): ModelArchConfig {
    return { ...this.arch, fullAttnLayers: [...this.arch.fullAttnLayers] };
  }

  setDisplayHz(hz: number): void {
    if (Number.isFinite(hz) && hz > 0) {
      this.displayHz = Math.max(30, Math.min(240, hz));
    }
  }

  getMemoryStats(): {
    weightMb: number;
    kvCacheMb: number;
    allocatedKvMb: number;
    deltaNetStateMb: number;
    estVramMb: number;
  } {
    const rowBytes = this.kvRowU32s * 4;
    const kvActiveBytes = this.numFaLayers * 2 * this.arch.attnKvHeads * this.seqPos * rowBytes;
    const kvCacheMb = kvActiveBytes / (1024 * 1024);
    const allocatedKvBytes = (this.fa0Count + this.fa1Count) * this.kvSingleBytes;
    const allocatedKvMb = allocatedKvBytes / (1024 * 1024);
    const deltaNetStateMb =
      (this.numDnLayers * (this.dnSingleBytes + this.dnConvSingleBytes)) / (1024 * 1024);
    return {
      weightMb: this.weightMb,
      kvCacheMb,
      allocatedKvMb,
      deltaNetStateMb,
      estVramMb: this.weightMb + kvCacheMb + deltaNetStateMb,
    };
  }

  private track(buf: GpuBufferLike): GpuBufferLike {
    this.allocatedBuffers.push(buf);
    return buf;
  }

  private untrackAndDestroy(buf: GpuBufferLike): void {
    const idx = this.allocatedBuffers.indexOf(buf);
    if (idx >= 0) {
      this.allocatedBuffers.splice(idx, 1);
    }
    try {
      buf.destroy();
    } catch {
      // Ignore already destroyed buffers
    }
  }

  private recordDecodeGpuStepMs(stepMs: number): void {
    this.lastDecodeGpuMs = stepMs;
    if (!Number.isFinite(stepMs) || stepMs <= 0) {
      return;
    }
    if (this.emaDecodeStepMs <= 0) {
      this.emaDecodeStepMs = stepMs;
    } else {
      this.emaDecodeStepMs = this.emaDecodeStepMs * 0.75 + stepMs * 0.25;
    }
  }

  private resolveBundleFileUrl(
    fileName: string,
    assetBaseUrl?: string,
    bundleBaseUrl?: string | null,
  ): string {
    if (bundleBaseUrl && bundleBaseUrl.trim().length > 0) {
      return `${bundleBaseUrl.trim().replace(/\/+$/, '')}/${fileName}`;
    }
    return resolveExtensionAssetUrl(`models/qwen35-4b/${fileName}`, assetBaseUrl);
  }

  private async fetchBundledManifest(
    assetBaseUrl?: string,
    bundleBaseUrl?: string | null,
  ): Promise<WgslSlabManifest> {
    const manifestUrl = this.resolveBundleFileUrl('manifest.json', assetBaseUrl, bundleBaseUrl);
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
      (parsed.version !== 1 && parsed.version !== 2) ||
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
      bundleBaseUrl?: string | null;
      modelId: string;
      kvMode?: WgslKvMode;
      maxContext?: number;
      pacingMode?: GpuPacingMode;
      forceComputeTier?: WgslComputeTier;
      deviceMemoryGb?: number;
      onProgress?: (phase: string, pct: number) => void;
    },
  ): Promise<WgslLoadStats> {
    this.dispose();
    const loadStart = performance.now();

    try {
      const wasm = await loadWgslWasm(options.assetBaseUrl);
      this.wasm = wasm;
      const {
        device,
        hasShaderF16,
        hasSubgroups,
        hasDp4a,
        maxBufferSize,
        maxStorageBufferBindingSize,
      } = await acquireWgslDevice();
      this.device = device;
      this.maxBufferSize = maxBufferSize;

      const manifest = await this.fetchBundledManifest(
        options.assetBaseUrl,
        options.bundleBaseUrl,
      );
      this.arch = resolveModelArch(manifest);
      this.kvMode = options.kvMode ?? 'int8';
      this.auxLayout = computeAuxLayout(this.arch);

      this.numFaLayers = getFullAttnLayerCount(this.arch);
      this.numDnLayers = getDeltaNetLayerCount(this.arch);
      this.fa0Count = Math.ceil(this.numFaLayers / 2);
      this.fa1Count = Math.max(0, this.numFaLayers - this.fa0Count);
      this.dnConvDim = getDeltaNetConvDim(this.arch);
      this.dnValDim = this.arch.dnValHeads * this.arch.dnHeadDim;
      this.dnQkvzAbRows = getDeltaNetQkvzAbRows(this.arch);
      this.attnQkvRows = getFullAttnQkvRows(this.arch);
      this.dnConvPerLayerF32s = this.dnConvDim * (this.arch.dnConvKernel - 1);
      this.dnStatePerLayerF32s = this.arch.dnValHeads * this.arch.dnHeadDim * this.arch.dnHeadDim;
      this.dnSingleBytes = this.dnStatePerLayerF32s * 4;
      this.dnConvSingleBytes = this.dnConvPerLayerF32s * 4;

      const inferredDeviceMemGb =
        options.deviceMemoryGb ??
        (typeof navigator !== 'undefined' && 'deviceMemory' in navigator
          ? Number((navigator as unknown as { deviceMemory?: number }).deviceMemory) || undefined
          : undefined);

      const requestedContext = Math.min(
        options.maxContext ?? 131072,
        this.kvMode === 'fp16' ? 65536 : 131072,
      );
      this.kvPlan = computeTieredKvPlan(
        requestedContext,
        this.arch,
        this.kvMode,
        { maxBufferSize, maxStorageBufferBindingSize },
        inferredDeviceMemGb,
      );
      this.allocatedKvCapacity = this.kvPlan.initialCapacity;
      this.maxSafeKvCapacity = this.kvPlan.maxSafeCapacity;
      this.maxSeqCapacity = this.allocatedKvCapacity;

      const policy = getGpuPacingPolicy(options.pacingMode ?? 'balanced');
      const layerByteSizes = computeLayerWeightByteSizes(this.arch);
      this.weightPartitions = partitionWeightLayersIntoSlabs(
        layerByteSizes,
        maxStorageBufferBindingSize,
      );
      this.vocabSlices = partitionVocabRowsIntoSlabs(
        this.arch.vocabSize,
        this.arch.hiddenSize,
        maxStorageBufferBindingSize,
      );

      const totalLayerBytesPlanned = layerByteSizes.reduce((s, b) => s + b, 0);
      if (totalLayerBytesPlanned !== manifest.slab0Bytes + manifest.slab1Bytes) {
        throw new Error(
          `Layer weight byte mismatch: planned ${totalLayerBytesPlanned} vs manifest ${
            manifest.slab0Bytes + manifest.slab1Bytes
          }.`,
        );
      }
      const totalVocabBytesPlanned = this.vocabSlices.reduce((s, v) => s + v.byteSize, 0);
      if (totalVocabBytesPlanned !== manifest.slab2Bytes) {
        throw new Error(
          `LM head byte mismatch: planned ${totalVocabBytesPlanned} vs manifest ${manifest.slab2Bytes}.`,
        );
      }
      if (this.auxLayout.totalF32s * 4 !== manifest.auxBytes) {
        throw new Error(
          `Aux size mismatch: planned (${this.auxLayout.totalF32s * 4}) vs manifest (${manifest.auxBytes})`,
        );
      }

      const gpuStreamStart = performance.now();
      options.onProgress?.('Streaming bundled Q4 slabs -> WebGPU VRAM...', 0);

      this.wSlabs = this.weightPartitions.map((part) =>
        this.track(
          device.createBuffer({
            size: part.totalBytes,
            usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
          }),
        ),
      );
      this.lmHeadSlabs = this.vocabSlices.map((slice) =>
        this.track(
          device.createBuffer({
            size: slice.byteSize,
            usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
          }),
        ),
      );
      this.auxBuf = this.track(
        device.createBuffer({
          size: manifest.auxBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
        }),
      );

      const layerTargets: Array<{ buffer: GpuBufferLike; start: number; end: number }> = [];
      {
        let cursor = 0;
        for (let i = 0; i < this.weightPartitions.length; i++) {
          const bytes = this.weightPartitions[i]!.totalBytes;
          layerTargets.push({ buffer: this.wSlabs[i]!, start: cursor, end: cursor + bytes });
          cursor += bytes;
        }
      }
      const lmTargets: Array<{ buffer: GpuBufferLike; start: number; end: number }> = [];
      {
        let cursor = 0;
        for (let i = 0; i < this.vocabSlices.length; i++) {
          const bytes = this.vocabSlices[i]!.byteSize;
          lmTargets.push({ buffer: this.lmHeadSlabs[i]!, start: cursor, end: cursor + bytes });
          cursor += bytes;
        }
      }
      const auxTargets: Array<{ buffer: GpuBufferLike; start: number; end: number }> = [
        { buffer: this.auxBuf, start: 0, end: manifest.auxBytes },
      ];

      const mapSpanToSegments = (
        targets: Array<{ buffer: GpuBufferLike; start: number; end: number }>,
        spanStart: number,
        spanBytes: number,
      ): GpuBufferWriteSegment[] => {
        const spanEnd = spanStart + spanBytes;
        const segs: GpuBufferWriteSegment[] = [];
        for (const t of targets) {
          const overlapStart = Math.max(spanStart, t.start);
          const overlapEnd = Math.min(spanEnd, t.end);
          if (overlapEnd > overlapStart) {
            segs.push({
              buffer: t.buffer,
              dstByteOffset: overlapStart - t.start,
              byteLength: overlapEnd - overlapStart,
            });
          }
        }
        return segs;
      };

      const slab0Shards =
        manifest.shards?.slab0 ?? [{ file: 'slab0.bin', bytes: manifest.slab0Bytes }];
      const slab1Shards =
        manifest.shards?.slab1 ?? [{ file: 'slab1.bin', bytes: manifest.slab1Bytes }];
      const slab2Shards =
        manifest.shards?.slab2 ?? [{ file: 'slab2.bin', bytes: manifest.slab2Bytes }];
      const auxShards =
        manifest.shards?.aux ?? [{ file: 'aux_params.bin', bytes: manifest.auxBytes }];

      const validateShardSum = (name: string, list: Array<{ bytes: number }>, expected: number) => {
        const sum = list.reduce((s, sh) => s + sh.bytes, 0);
        if (sum !== expected) {
          throw new Error(`Manifest shard sum mismatch for ${name}: got ${sum}, expected ${expected}.`);
        }
      };
      validateShardSum('slab0', slab0Shards, manifest.slab0Bytes);
      validateShardSum('slab1', slab1Shards, manifest.slab1Bytes);
      validateShardSum('slab2', slab2Shards, manifest.slab2Bytes);
      validateShardSum('aux', auxShards, manifest.auxBytes);

      const streamGroups = [
        { shards: [...slab0Shards, ...slab1Shards], targets: layerTargets },
        { shards: slab2Shards, targets: lmTargets },
        { shards: auxShards, targets: auxTargets },
      ];
      const totalBundleBytes =
        manifest.slab0Bytes + manifest.slab1Bytes + manifest.slab2Bytes + manifest.auxBytes;
      let completedBytes = 0;

      for (const group of streamGroups) {
        let logicalOffset = 0;
        for (const shard of group.shards) {
          const segs = mapSpanToSegments(group.targets, logicalOffset, shard.bytes);
          const url = this.resolveBundleFileUrl(
            shard.file,
            options.assetBaseUrl,
            options.bundleBaseUrl,
          );
          await streamUrlToGpuSegments(device, url, segs, shard.bytes, {
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
          await device.queue.onSubmittedWorkDone();
          logicalOffset += shard.bytes;
          completedBytes += shard.bytes;
        }
      }
      const opfsReadMs = performance.now() - gpuStreamStart;

      options.onProgress?.('Compiling Qwen3.5 WGSL compute pipelines...', 93);
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
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      });
      const gqaSplitBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
          { binding: 4, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      });
      const gdnApplyBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
          { binding: 5, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
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

      this.dynStorage4Bgl = dynStorage4Bgl;
      this.gdnApplyBgl = gdnApplyBgl;
      this.qkRopeBgl = qkRopeBgl;
      this.gqaSplitBgl = gqaSplitBgl;
      this.dynRwStorageBgl = dynRwStorageBgl;
      this.staticBgl = staticBgl;
      this.staticRead2Bgl = staticRead2Bgl;

      const dyn4Layout = device.createPipelineLayout({ bindGroupLayouts: [dynStorage4Bgl] });
      const dynRwLayout = device.createPipelineLayout({ bindGroupLayouts: [dynRwStorageBgl] });
      const gdnLayout = device.createPipelineLayout({ bindGroupLayouts: [gdnRecBgl] });
      const gdnApplyLayout = device.createPipelineLayout({ bindGroupLayouts: [gdnApplyBgl] });
      const qkRopeLayout = device.createPipelineLayout({ bindGroupLayouts: [qkRopeBgl] });
      const gqaSplitLayout = device.createPipelineLayout({ bindGroupLayouts: [gqaSplitBgl] });
      const staticArg1Layout = device.createPipelineLayout({ bindGroupLayouts: [staticBgl] });
      const staticArg2Layout = device.createPipelineLayout({ bindGroupLayouts: [staticRead2Bgl] });

      // Self-healing tier fallback loop: compile pipelines and verify kernels on highest viable tier.
      const candidates = getCandidateComputeTiers(
        hasShaderF16,
        hasSubgroups,
        options.forceComputeTier,
      );
      let activeTier: WgslComputeTier | null = null;
      let lastTierError: unknown = null;

      for (const tier of candidates) {
        try {
          const ids = getTierShaderIds(tier, this.kvMode);
          [
            this.embedPipe,
            this.rmsNormPipe,
            this.swigluPipe,
            this.gemvPipe,
            this.gemvWidePipe,
            this.gateUpSwigluPipe,
            this.gemvBatchPipe,
            this.gemmPipe,
            this.convPipe,
            this.gdnRecPipe,
            this.gdnChunkPrepPipe,
            this.gdnChunkApplyPipe,
            this.qkRopePipe,
            this.gqaSplitPlainPipe,
            this.gqaReducePipe,
            this.gqaPrefillPipe,
            this.arg1Pipe,
            this.arg2Pipe,
            this.argmaxBatch1Pipe,
            this.argmaxBatch2Pipe,
            this.specAcceptPipe,
          ] = await Promise.all([
            createValidatedPipeline(device, `${tier}-embed-lookup`, getWgslShaderSource(wasm, ids.embed), dyn4Layout),
            createValidatedPipeline(device, `${tier}-rmsnorm-batch`, getWgslShaderSource(wasm, ids.rmsNorm), dyn4Layout),
            createValidatedPipeline(device, `${tier}-swiglu-batch`, getWgslShaderSource(wasm, ids.swiglu), dyn4Layout),
            createValidatedPipeline(device, `${tier}-q4-gemv`, getWgslShaderSource(wasm, ids.gemv), dyn4Layout),
            createValidatedPipeline(device, `${tier}-q4-gemv-wide-k`, getWgslShaderSource(wasm, ids.gemvWide), dyn4Layout),
            createValidatedPipeline(device, `${tier}-q4-gate-up-swiglu`, getWgslShaderSource(wasm, ids.gateUpSwiglu), dyn4Layout),
            createValidatedPipeline(device, `${tier}-q4-gemv-batch`, getWgslShaderSource(wasm, ids.gemvBatch), dyn4Layout),
            createValidatedPipeline(device, `${tier}-q4-gemm`, getWgslShaderSource(wasm, ids.gemm), dyn4Layout),
            createValidatedPipeline(device, `${tier}-gdn-conv1d`, getWgslShaderSource(wasm, ids.conv), dynRwLayout),
            createValidatedPipeline(device, `${tier}-gdn-recurrent`, getWgslShaderSource(wasm, ids.gdnRec), gdnLayout),
            createValidatedPipeline(device, `${tier}-gdn-chunk-prep`, getWgslShaderSource(wasm, ids.gdnChunkPrep), dyn4Layout),
            createValidatedPipeline(device, `${tier}-gdn-chunk-apply`, getWgslShaderSource(wasm, ids.gdnChunkApply), gdnApplyLayout),
            createValidatedPipeline(device, `${tier}-qk-norm-rope`, getWgslShaderSource(wasm, ids.qkRope), qkRopeLayout),
            createValidatedPipeline(device, `${tier}-gqa-${this.kvMode}-split`, getWgslShaderSource(wasm, ids.gqaSplitPlain), gqaSplitLayout),
            createValidatedPipeline(device, `${tier}-gqa-reduce`, getWgslShaderSource(wasm, ids.gqaReduce), dyn4Layout),
            createValidatedPipeline(device, `${tier}-gqa-${this.kvMode}-prefill`, getWgslShaderSource(wasm, ids.gqaPrefill), dynRwLayout),
            createValidatedPipeline(device, `${tier}-argmax-pass1`, getWgslShaderSource(wasm, ids.arg1), staticArg1Layout),
            createValidatedPipeline(device, `${tier}-argmax-pass2`, getWgslShaderSource(wasm, ids.arg2), staticArg2Layout),
            createValidatedPipeline(device, `${tier}-argmax-batch-pass1`, getWgslShaderSource(wasm, 132), staticArg1Layout),
            createValidatedPipeline(device, `${tier}-argmax-batch-pass2`, getWgslShaderSource(wasm, 133), staticArg2Layout),
            createValidatedPipeline(device, `${tier}-spec-accept-greedy`, getWgslShaderSource(wasm, 43), staticArg2Layout),
          ]);

          this.gemvBatchWidePipe = null;
          this.gateUpSwigluBatchPipe = null;
          if (tier === 'tier1-f16-subgroups') {
            try {
              this.gemvBatchWidePipe = await createValidatedPipeline(
                device,
                'tier1-q4-gemv-batch-wide',
                getWgslShaderSource(wasm, 44),
                dyn4Layout,
              );
            } catch {
              this.gemvBatchWidePipe = null;
            }
            try {
              this.gateUpSwigluBatchPipe = await createValidatedPipeline(
                device,
                'tier1-q4-gate-up-swiglu-batch-v2',
                getWgslShaderSource(wasm, 45),
                dyn4Layout,
              );
            } catch {
              this.gateUpSwigluBatchPipe = null;
            }
          }

          this.gqaSplitPipe = this.gqaSplitPlainPipe;
          this.usingDp4a = false;
          if (tier === 'tier1-f16-subgroups' && this.kvMode === 'int8' && hasDp4a) {
            try {
              this.gqaSplitPipe = await createValidatedPipeline(
                device,
                'real-gqa-int8-split-dp4a',
                getWgslShaderSource(wasm, 23),
                gqaSplitLayout,
              );
              this.usingDp4a = true;
            } catch {
              this.gqaSplitPipe = this.gqaSplitPlainPipe;
              this.usingDp4a = false;
            }
          }

          await this.verifyRealKernels(
            device,
            dynStorage4Bgl,
            gdnRecBgl,
            gdnApplyBgl,
            qkRopeBgl,
            dynRwStorageBgl,
            gqaSplitBgl,
          );
          activeTier = tier;
          break;
        } catch (err) {
          lastTierError = err;
          console.warn(
            `[WGSL] Compute tier "${tier}" failed pipeline compile/verify; stepping down:`,
            err,
          );
        }
      }

      if (!activeTier) {
        throw lastTierError ?? new Error('All WGSL compute tiers failed initialization.');
      }
      this.computeTier = activeTier;
      const pipelineCompileMs = performance.now() - compileStart;

      // Allocate hybrid state buffers
      const dnLayerAllocCount = Math.max(1, this.numDnLayers);
      this.dnStateBuf = this.track(
        device.createBuffer({
          size: dnLayerAllocCount * this.dnSingleBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );
      this.dnConvBuf = this.track(
        device.createBuffer({
          size: dnLayerAllocCount * this.dnConvSingleBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );

      this.kvRowU32s = getFullAttnKvHeadU32s(this.arch, this.kvMode);
      this.kvPlaneU32s = this.arch.attnKvHeads * this.allocatedKvCapacity * this.kvRowU32s;
      this.kvSingleBytes = 2 * this.kvPlaneU32s * 4;
      this.kvSlab0 = this.track(
        device.createBuffer({
          size: Math.max(1, this.fa0Count) * this.kvSingleBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );
      this.kvSlab1 = this.track(
        device.createBuffer({
          size: Math.max(1, this.fa1Count) * this.kvSingleBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );

      // Allocate non-aliasing activation buffers sized for PREFILL_MAX_CHUNK & arch
      const H = this.arch.hiddenSize;
      const I = this.arch.intermediateSize;
      const V = this.arch.vocabSize;
      const workCols = Math.max(I * 2, this.dnQkvzAbRows, this.attnQkvRows);
      const attnCols = Math.max(I, this.dnValDim, this.arch.attnQHeads * this.arch.attnHeadDim);
      const gqaSplitFloats = Math.max(
        this.arch.attnQHeads * GQA_INT8_NUM_SPLITS * 260,
        PREFILL_MAX_CHUNK * this.arch.attnQHeads * PREFILL_NUM_SPLITS * 260,
        558080,
      );

      this.tokenIdsBuf = this.track(
        device.createBuffer({ size: PREFILL_MAX_CHUNK * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }),
      );
      this.hiddenBuf = this.track(
        device.createBuffer({ size: PREFILL_MAX_CHUNK * H * 4, usage: GPU_USAGE.STORAGE }),
      );
      const normRows = Math.max(PREFILL_MAX_CHUNK, dnLayerAllocCount * MAX_SPEC_TOKENS);
      this.normBuf = this.track(
        device.createBuffer({ size: normRows * H * 4, usage: GPU_USAGE.STORAGE }),
      );
      this.workBuf = this.track(
        device.createBuffer({
          size: PREFILL_MAX_CHUNK * workCols * 4,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );
      this.attnOutBuf = this.track(
        device.createBuffer({ size: PREFILL_MAX_CHUNK * attnCols * 4, usage: GPU_USAGE.STORAGE }),
      );
      this.gqaSplitBuf = this.track(
        device.createBuffer({
          size: gqaSplitFloats * 4,
          usage: GPU_USAGE.STORAGE,
        }),
      );
      this.logitsBuf = this.track(
        device.createBuffer({
          size: V * 4,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );

      // Speculative verification & DeltaNet rollback scratch buffers
      this.dnStateBackupBuf = this.track(
        device.createBuffer({
          size: dnLayerAllocCount * this.dnSingleBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );
      this.dnConvBackupBuf = this.track(
        device.createBuffer({
          size: dnLayerAllocCount * this.dnConvSingleBytes,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );
      this.verifyLogitsBuf = this.track(
        device.createBuffer({
          size: MAX_SPEC_TOKENS * V * 4,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC,
        }),
      );
      this.verifyTokOutBuf = this.track(
        device.createBuffer({
          size: 32,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
        }),
      );
      this.verifyAcceptOutBuf = this.track(
        device.createBuffer({
          size: 64,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC,
        }),
      );
      this.verifyStaging64B = this.track(
        device.createBuffer({
          size: 64,
          usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST,
        }),
      );

      const argmaxWgCount = Math.ceil(V / 256);
      const partValBuf = this.track(
        device.createBuffer({ size: MAX_SPEC_TOKENS * argmaxWgCount * 4, usage: GPU_USAGE.STORAGE }),
      );
      const partIdxBuf = this.track(
        device.createBuffer({ size: MAX_SPEC_TOKENS * argmaxWgCount * 4, usage: GPU_USAGE.STORAGE }),
      );
      this.partValBuf = partValBuf;
      this.partIdxBuf = partIdxBuf;
      this.tokOutBuf = this.track(
        device.createBuffer({ size: 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC }),
      );
      this.staging4B = this.track(
        device.createBuffer({ size: 4, usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST }),
      );

      const uArg1 = this.track(device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
      const uArg2 = this.track(device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
      device.queue.writeBuffer(uArg1, 0, new Uint32Array([V, 0, 0, 0]));
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
      this.bgEmbedPerSlice = this.lmHeadSlabs.map((slab) =>
        makeDyn4(this.decodeUniformBuf, this.tokenIdsBuf, slab, this.hiddenBuf),
      );
      this.bgRmsNorm = makeDyn4(this.decodeUniformBuf, this.auxBuf, this.hiddenBuf, this.normBuf);
      this.bgWSlabNormToWork = this.wSlabs.map((slab) =>
        makeDyn4(this.decodeUniformBuf, slab, this.normBuf, this.workBuf),
      );
      this.bgWSlabNormToAttn = this.wSlabs.map((slab) =>
        makeDyn4(this.decodeUniformBuf, slab, this.normBuf, this.attnOutBuf),
      );
      this.bgWSlabAttnToHidden = this.wSlabs.map((slab) =>
        makeDyn4(this.decodeUniformBuf, slab, this.attnOutBuf, this.hiddenBuf),
      );
      this.bgLmHeadPerSlice = this.lmHeadSlabs.map((slab) =>
        makeDyn4(this.decodeUniformBuf, slab, this.normBuf, this.logitsBuf),
      );
      this.bgGqaReduce = makeDyn4(this.decodeUniformBuf, this.workBuf, this.gqaSplitBuf, this.attnOutBuf);
      this.bgConv = makeDynRw(this.decodeUniformBuf, this.auxBuf, this.dnConvBuf, this.workBuf);
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
          { binding: 3, resource: { buffer: this.tokenIdsBuf } },
        ],
      });

      // Prefill bind groups bound to prefillUniformBuf
      this.pBgEmbedPerSlice = this.lmHeadSlabs.map((slab) =>
        makeDyn4(this.prefillUniformBuf, this.tokenIdsBuf, slab, this.hiddenBuf),
      );
      this.pBgRmsNorm = makeDyn4(this.prefillUniformBuf, this.auxBuf, this.hiddenBuf, this.normBuf);
      this.pBgSwiGlu = makeDyn4(this.prefillUniformBuf, this.auxBuf, this.workBuf, this.attnOutBuf);
      this.pBgWSlabNormToWork = this.wSlabs.map((slab) =>
        makeDyn4(this.prefillUniformBuf, slab, this.normBuf, this.workBuf),
      );
      this.pBgWSlabAttnToHidden = this.wSlabs.map((slab) =>
        makeDyn4(this.prefillUniformBuf, slab, this.attnOutBuf, this.hiddenBuf),
      );
      this.pBgLmHeadPerSlice = this.lmHeadSlabs.map((slab) =>
        makeDyn4(this.prefillUniformBuf, slab, this.normBuf, this.logitsBuf),
      );
      this.pBgConv = makeDynRw(this.prefillUniformBuf, this.auxBuf, this.dnConvBuf, this.workBuf);
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
      this.pBgGdnChunkPrep = makeDyn4(this.prefillUniformBuf, this.auxBuf, this.workBuf, this.gqaSplitBuf);
      this.pBgGdnChunkApply = device.createBindGroup({
        layout: gdnApplyBgl,
        entries: [
          { binding: 0, resource: { buffer: this.prefillUniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: this.auxBuf } },
          { binding: 2, resource: { buffer: this.workBuf } },
          { binding: 3, resource: { buffer: this.gqaSplitBuf } },
          { binding: 4, resource: { buffer: this.dnStateBuf } },
          { binding: 5, resource: { buffer: this.attnOutBuf } },
        ],
      });
      this.pBgQkRope = device.createBindGroup({
        layout: qkRopeBgl,
        entries: [
          { binding: 0, resource: { buffer: this.prefillUniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: this.auxBuf } },
          { binding: 2, resource: { buffer: this.workBuf } },
          { binding: 3, resource: { buffer: this.tokenIdsBuf } },
        ],
      });

      this.rebuildKvBindGroups();

      // Compute exact Q4 weight u32 offsets per GPU partition
      const fullAttnSet = getFullAttnLayerSet(this.arch);
      const attnQRows = this.arch.attnQHeads * this.arch.attnHeadDim * 2;
      const attnKvRows = this.arch.attnKvHeads * this.arch.attnHeadDim;
      const attnORows = this.arch.attnQHeads * this.arch.attnHeadDim;

      this.layerOffsets = new Array(this.arch.numLayers);
      let dnIdx = 0;
      let faIdx = 0;
      for (let sIdx = 0; sIdx < this.weightPartitions.length; sIdx++) {
        const part = this.weightPartitions[sIdx]!;
        let wOff = 0;
        const takeW = (rows: number, cols: number): number => {
          const u32s = (rows * wasm.q4_row_u32s(cols)) >>> 0;
          const cur = wOff;
          wOff += u32s;
          return cur;
        };
        for (let l = part.startLayer; l < part.endLayer; l++) {
          const isFullAttn = fullAttnSet.has(l);
          if (!isFullAttn) {
            const w = [
              takeW(this.dnConvDim, H),
              takeW(this.dnValDim, H),
              takeW(this.arch.dnValHeads * 2, H),
              takeW(H, this.dnValDim),
              takeW(I, H),
              takeW(I, H),
              takeW(H, I),
            ];
            this.layerOffsets[l] = { slabIdx: sIdx, isFullAttn: false, dnIdx: dnIdx++, faIdx: 0, w };
          } else {
            const w = [
              takeW(attnQRows, H),
              takeW(attnKvRows, H),
              takeW(attnKvRows, H),
              takeW(H, attnORows),
              takeW(I, H),
              takeW(I, H),
              takeW(H, I),
            ];
            this.layerOffsets[l] = { slabIdx: sIdx, isFullAttn: true, dnIdx: 0, faIdx: faIdx++, w };
          }
        }
        if (wOff * 4 !== part.totalBytes) {
          throw new Error(
            `GPU slab ${sIdx} byte mismatch: planned ${wOff * 4} vs partition ${part.totalBytes}.`,
          );
        }
      }

      const hiddenRowU32s = wasm.q4_row_u32s(H);
      for (const lo of this.layerOffsets) {
        if (!lo.isFullAttn) {
          const [wQkv, wZ, wAb] = lo.w;
          if (
            wZ !== wQkv + this.dnConvDim * hiddenRowU32s ||
            wAb !== wZ + this.dnValDim * hiddenRowU32s
          ) {
            throw new Error('DeltaNet weight offsets are not contiguous for QKV+Z+AB fusion.');
          }
        } else {
          const [wQg, wK, wV] = lo.w;
          if (
            wK !== wQg + attnQRows * hiddenRowU32s ||
            wV !== wK + attnKvRows * hiddenRowU32s
          ) {
            throw new Error('Full-Attn weight offsets are not contiguous for QG+K+V fusion.');
          }
        }
      }

      this.buildStaticDecodePlan();
      this.weightMb = totalBundleBytes / (1024 * 1024);

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
        allocatedKvCapacity: this.allocatedKvCapacity,
        maxSafeKvCapacity: this.maxSafeKvCapacity,
        computeTier: this.computeTier,
        hasShaderF16,
        hasSubgroups,
        hasDp4a,
        usingDp4a: this.usingDp4a,
        maxBufferSize,
      };
    } catch (err) {
      this.dispose();
      throw err;
    }
  }

  private invalidateVerifyPlans(): void {
    for (const plan of this.verifyPlans.values()) {
      for (const buf of plan.ownedBuffers) {
        try {
          buf.destroy();
        } catch {
          // Ignore already destroyed buffers
        }
      }
    }
    this.verifyPlans.clear();
  }

  private rebuildKvBindGroups(): void {
    this.invalidateVerifyPlans();
    const device = this.device!;
    this.bgGqaSplit0 = device.createBindGroup({
      layout: this.gqaSplitBgl,
      entries: [
        { binding: 0, resource: { buffer: this.decodeUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.workBuf } },
        { binding: 2, resource: { buffer: this.kvSlab0 } },
        { binding: 3, resource: { buffer: this.gqaSplitBuf } },
        { binding: 4, resource: { buffer: this.tokenIdsBuf } },
      ],
    });
    this.bgGqaSplit1 = device.createBindGroup({
      layout: this.gqaSplitBgl,
      entries: [
        { binding: 0, resource: { buffer: this.decodeUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.workBuf } },
        { binding: 2, resource: { buffer: this.kvSlab1 } },
        { binding: 3, resource: { buffer: this.gqaSplitBuf } },
        { binding: 4, resource: { buffer: this.tokenIdsBuf } },
      ],
    });
    this.bgGqaPrefill0 = device.createBindGroup({
      layout: this.dynRwStorageBgl,
      entries: [
        { binding: 0, resource: { buffer: this.prefillUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.workBuf } },
        { binding: 2, resource: { buffer: this.kvSlab0 } },
        { binding: 3, resource: { buffer: this.gqaSplitBuf } },
      ],
    });
    this.bgGqaPrefill1 = device.createBindGroup({
      layout: this.dynRwStorageBgl,
      entries: [
        { binding: 0, resource: { buffer: this.prefillUniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.workBuf } },
        { binding: 2, resource: { buffer: this.kvSlab1 } },
        { binding: 3, resource: { buffer: this.gqaSplitBuf } },
      ],
    });
  }

  /**
   * Dynamically grows `kvSlab0` / `kvSlab1` along the tiered KV plan when `neededSeqLen`
   * approaches `allocatedKvCapacity`, copying only active `[0..seqPos)` spans GPU-to-GPU.
   */
  async ensureKvCapacity(neededSeqLen: number): Promise<void> {
    if (!this.device || !this.wasm) {
      throw new Error('WGSL engine not loaded.');
    }
    if (neededSeqLen + 8 <= this.allocatedKvCapacity) {
      return;
    }
    if (neededSeqLen + 8 > this.maxSafeKvCapacity) {
      throw new Error(
        `Context overflow: needed=${neededSeqLen + 8} exceeds maxSafeKvCapacity=${this.maxSafeKvCapacity}.`,
      );
    }

    const targetNeeded = Math.min(this.maxSafeKvCapacity, neededSeqLen + 512);
    let nextCap = this.maxSafeKvCapacity;
    for (const tierCap of this.kvPlan.tiers) {
      if (tierCap >= targetNeeded) {
        nextCap = tierCap;
        break;
      }
    }
    if (nextCap <= this.allocatedKvCapacity) {
      return;
    }

    const device = this.device;
    const oldCap = this.allocatedKvCapacity;
    const oldKvSlab0 = this.kvSlab0;
    const oldKvSlab1 = this.kvSlab1;

    const newKvPlaneU32s = this.arch.attnKvHeads * nextCap * this.kvRowU32s;
    const newKvSingleBytes = 2 * newKvPlaneU32s * 4;
    const newKvSlab0 = this.track(
      device.createBuffer({
        size: Math.max(1, this.fa0Count) * newKvSingleBytes,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
      }),
    );
    const newKvSlab1 = this.track(
      device.createBuffer({
        size: Math.max(1, this.fa1Count) * newKvSingleBytes,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
      }),
    );

    const rowBytes = this.kvRowU32s * 4;
    const copyBytes = this.seqPos * rowBytes;
    if (copyBytes > 0) {
      const enc = device.createCommandEncoder();
      const copySlab = (
        srcSlab: GpuBufferLike,
        dstSlab: GpuBufferLike,
        layerCount: number,
      ) => {
        for (let l = 0; l < layerCount; l++) {
          for (let plane = 0 as 0 | 1; plane < 2; plane = (plane + 1) as 0 | 1) {
            for (let h = 0; h < this.arch.attnKvHeads; h++) {
              const srcOff = computeKvSliceByteOffset(
                l,
                plane,
                h,
                oldCap,
                this.arch,
                this.kvMode,
              );
              const dstOff = computeKvSliceByteOffset(
                l,
                plane,
                h,
                nextCap,
                this.arch,
                this.kvMode,
              );
              enc.copyBufferToBuffer(srcSlab, srcOff, dstSlab, dstOff, copyBytes);
            }
          }
        }
      };
      copySlab(oldKvSlab0, newKvSlab0, this.fa0Count);
      copySlab(oldKvSlab1, newKvSlab1, this.fa1Count);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    this.kvSlab0 = newKvSlab0;
    this.kvSlab1 = newKvSlab1;
    this.allocatedKvCapacity = nextCap;
    this.maxSeqCapacity = nextCap;
    this.kvPlaneU32s = newKvPlaneU32s;
    this.kvSingleBytes = newKvSingleBytes;

    this.untrackAndDestroy(oldKvSlab0);
    this.untrackAndDestroy(oldKvSlab1);

    this.rebuildKvBindGroups();
    this.buildStaticDecodePlan();
  }

  private packGqaSplitsParam(splits: number): number {
    const gqaRatio = Math.max(1, Math.floor(this.arch.attnQHeads / this.arch.attnKvHeads));
    return (splits & 0xffff) | ((this.arch.attnKvHeads & 0xff) << 16) | ((gqaRatio & 0xff) << 24);
  }

  private packGdnHeadsParam(): number {
    return (this.arch.dnValHeads & 0xffff) | ((this.arch.dnKeyHeads & 0xffff) << 16);
  }

  private packQkRopeParam(): number {
    return (
      (this.arch.attnQHeads & 0xff) |
      ((this.arch.attnKvHeads & 0xff) << 8) |
      (((this.arch.ropeDim / 2) & 0xffff) << 16)
    );
  }

  private buildStaticDecodePlan(): void {
    const device = this.device!;
    const uniformTable = new ArrayBuffer(UNIFORM_MAX_SLOTS * 256);
    const u32View = new Uint32Array(uniformTable);
    const f32View = new Float32Array(uniformTable);

    this.decodeSteps = [];
    this.decodeLayerEndStepIndices = [];
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

    const H = this.arch.hiddenSize;
    const I = this.arch.intermediateSize;
    const L = this.arch.numLayers;
    const gemvWx = (rows: number) => Math.ceil(rows / 16);
    const gemvWideWx = (rows: number) => Math.ceil(rows / 4);
    const gateUpWx = (rows: number) => Math.ceil(rows / 8);
    const gqaSplits = this.kvMode === 'int8' ? GQA_INT8_NUM_SPLITS : GQA_FP16_NUM_SPLITS;
    const gqaSplitsPacked = this.packGqaSplitsParam(gqaSplits);
    const gdnHeadsPacked = this.packGdnHeadsParam();
    const qkRopePacked = this.packQkRopeParam();
    const convWgs = Math.ceil(this.dnConvDim / 128);
    const embedBlocksY = Math.ceil(H / 32);
    const attnQDim = this.arch.attnQHeads * this.arch.attnHeadDim * 2;
    const attnKvDim = this.arch.attnKvHeads * this.arch.attnHeadDim;
    const attnODim = this.arch.attnQHeads * this.arch.attnHeadDim;
    const qkRopeHeadCount = this.arch.attnQHeads + this.arch.attnKvHeads;

    // 0. Embedding lookup for single token across vocab slice(s)
    for (let sIdx = 0; sIdx < this.vocabSlices.length; sIdx++) {
      const slice = this.vocabSlices[sIdx]!;
      const rowEnd = this.vocabSlices.length === 1 ? 0 : slice.endRow;
      push(
        this.embedPipe,
        this.bgEmbedPerSlice[sIdx]!,
        [1, H, slice.startRow, rowEnd, 0, 0, 0, 0],
        1,
        embedBlocksY,
      );
    }

    for (let l = 0; l < L; l++) {
      const la = this.auxLayout.layers[l]!;
      const lo = this.layerOffsets[l]!;
      const bgNormToWork = this.bgWSlabNormToWork[lo.slabIdx]!;
      const bgNormToAttn = this.bgWSlabNormToAttn[lo.slabIdx]!;
      const bgAttnToHidden = this.bgWSlabAttnToHidden[lo.slabIdx]!;

      // Pre-attn RMSNorm
      push(this.rmsNormPipe, this.bgRmsNorm, [1, H, la.inputNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);

      if (!lo.isFullAttn) {
        const [wQkv, , , wOut, wGate, wUp, wDown] = lo.w;
        // Fused QKV+Z+AB projection contiguous in weightSlab
        push(this.gemvPipe, bgNormToWork, [this.dnQkvzAbRows, H, wQkv, 0, 0, 0, 0, 0], gemvWx(this.dnQkvzAbRows));
        push(
          this.convPipe,
          this.bgConv,
          [1, la.convWOff, lo.dnIdx * this.dnConvPerLayerF32s, 0, this.dnConvDim, 0, 0, 0],
          convWgs,
        );
        push(
          this.gdnRecPipe,
          this.bgGdnRec,
          [
            1,
            lo.dnIdx * this.dnStatePerLayerF32s,
            la.dnAuxOff,
            0,
            this.dnConvDim,
            this.dnConvDim + this.dnValDim,
            0,
            gdnHeadsPacked,
          ],
          this.arch.dnValHeads,
        );
        push(this.gemvWidePipe, bgAttnToHidden, [H, this.dnValDim, wOut, 0, 0, 1, 0, 0], gemvWideWx(H));

        push(this.rmsNormPipe, this.bgRmsNorm, [1, H, la.postNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);
        push(
          this.gateUpSwigluPipe,
          bgNormToAttn,
          [I, H, wGate, 0, 0, 0, wUp, 0],
          gateUpWx(I),
        );
        push(this.gemvWidePipe, bgAttnToHidden, [H, I, wDown, 0, 0, 1, 0, 0], gemvWideWx(H));
      } else {
        const [wQg, , , wO, wGate, wUp, wDown] = lo.w;
        push(this.gemvPipe, bgNormToWork, [this.attnQkvRows, H, wQg, 0, 0, 0, 0, 0], gemvWx(this.attnQkvRows));

        push(
          this.qkRopePipe,
          this.bgQkRope,
          [
            1,
            0xffffffff,
            0,
            attnQDim,
            la.qNormOff,
            la.kNormOff,
            this.auxLayout.invFreqOff,
            qkRopePacked,
          ],
          qkRopeHeadCount,
          1,
        );

        const inKv0 = lo.faIdx < this.fa0Count;
        const localKv = inKv0 ? lo.faIdx : lo.faIdx - this.fa0Count;
        const kvBaseU32 = localKv * (this.kvSingleBytes / 4);
        push(
          this.gqaSplitPipe,
          inKv0 ? this.bgGqaSplit0 : this.bgGqaSplit1,
          [
            kvBaseU32,
            this.maxSeqCapacity,
            0xffffffff,
            0,
            attnQDim,
            attnQDim + attnKvDim,
            gqaSplitsPacked,
            this.arch.attnHeadDim * 2,
          ],
          this.arch.attnKvHeads,
          gqaSplits,
        );

        push(
          this.gqaReducePipe,
          this.bgGqaReduce,
          [
            gqaSplits,
            this.arch.attnHeadDim,
            0,
            this.arch.attnHeadDim * 2,
            this.arch.attnQHeads,
            0,
            0,
            0,
          ],
          this.arch.attnQHeads,
        );
        push(this.gemvWidePipe, bgAttnToHidden, [H, attnODim, wO, 0, 0, 1, 0, 0], gemvWideWx(H));

        push(this.rmsNormPipe, this.bgRmsNorm, [1, H, la.postNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);
        push(
          this.gateUpSwigluPipe,
          bgNormToAttn,
          [I, H, wGate, 0, 0, 0, wUp, 0],
          gateUpWx(I),
        );
        push(this.gemvWidePipe, bgAttnToHidden, [H, I, wDown, 0, 0, 1, 0, 0], gemvWideWx(H));
      }
      if (l < L - 1) {
        this.decodeLayerEndStepIndices.push(this.decodeSteps.length);
      }
    }

    // Final RMSNorm + LM Head (across 1 or more vocab slices)
    push(this.rmsNormPipe, this.bgRmsNorm, [1, H, this.auxLayout.finalNormOff, 0, 0, 1e-6, 0, 0], 1, 1, true);
    for (let sIdx = 0; sIdx < this.vocabSlices.length; sIdx++) {
      const slice = this.vocabSlices[sIdx]!;
      push(
        this.gemvPipe,
        this.bgLmHeadPerSlice[sIdx]!,
        [slice.rowCount, H, 0, 0, slice.startRow, 0, 0, 0],
        gemvWx(slice.rowCount),
      );
    }
    this.decodeLayerEndStepIndices.push(this.decodeSteps.length);
    if (this.decodeLayerEndStepIndices.length !== L) {
      throw new Error(
        `Invalid decode layer boundary table length: expected ${L}, got ${this.decodeLayerEndStepIndices.length}.`,
      );
    }

    device.queue.writeBuffer(this.decodeUniformBuf, 0, uniformTable);
  }

  resetDecodeBurst(): void {
    this.decodeBurstMs = 0;
  }

  async resetState(): Promise<void> {
    this.lastDecodeGpuMs = 0;
    this.emaDecodeStepMs = 0;
    this.decodeBurstMs = 0;
    if (!this.device) {
      this.seqPos = 0;
      return;
    }
    const enc = this.device.createCommandEncoder();
    const dnCount = Math.max(1, this.numDnLayers);
    if (typeof enc.clearBuffer === 'function') {
      enc.clearBuffer(this.dnStateBuf);
      enc.clearBuffer(this.dnConvBuf);
    } else {
      const zeros = new Uint8Array(this.dnSingleBytes);
      for (let i = 0; i < dnCount; i++) {
        this.device.queue.writeBuffer(this.dnStateBuf, i * this.dnSingleBytes, zeros);
        this.device.queue.writeBuffer(
          this.dnConvBuf,
          i * this.dnConvSingleBytes,
          zeros.subarray(0, this.dnConvSingleBytes),
        );
      }
    }
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this.seqPos = 0;
  }

  private getGpuPrefixVramCapBytes(): number {
    if (this.maxBufferSize > 0) {
      return Math.min(MAX_GPU_PREFIX_VRAM_BYTES, Math.floor(this.maxBufferSize * 0.25));
    }
    return MAX_GPU_PREFIX_VRAM_BYTES;
  }

  private destroyPrefixSlotBuffers(slot: GpuPrefixSlot): void {
    this.untrackAndDestroy(slot.dnState);
    this.untrackAndDestroy(slot.dnConv);
    if (slot.kvPacked0) this.untrackAndDestroy(slot.kvPacked0);
    if (slot.kvPacked1) this.untrackAndDestroy(slot.kvPacked1);
  }

  hasGpuPrefixSlot(slotId: string): boolean {
    return this.gpuPrefixSlots.has(slotId);
  }

  /**
   * Snapshots active DeltaNet state and `[0..seqPos)` KV cache slices directly
   * into dedicated GPU buffers inside a single command encoder (zero `mapAsync`).
   */
  async saveGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; byteSize: number }> {
    if (!this.device) {
      throw new Error('WGSL engine not loaded.');
    }
    const existing = this.gpuPrefixSlots.get(slotId);
    if (existing) {
      this.destroyPrefixSlotBuffers(existing);
      this.gpuPrefixSlots.delete(slotId);
    }

    const dnStateBytes = Math.max(1, this.numDnLayers) * this.dnSingleBytes;
    const dnConvBytes = Math.max(1, this.numDnLayers) * this.dnConvSingleBytes;
    const rowBytes = this.kvRowU32s * 4;
    const sliceBytes = this.seqPos * rowBytes;
    const kv0Bytes = this.fa0Count * 2 * this.arch.attnKvHeads * sliceBytes;
    const kv1Bytes = this.fa1Count * 2 * this.arch.attnKvHeads * sliceBytes;
    const totalBytes = dnStateBytes + dnConvBytes + kv0Bytes + kv1Bytes;

    const vramCap = this.getGpuPrefixVramCapBytes();
    if (totalBytes > vramCap) {
      throw new Error(
        `GPU prefix slot "${slotId}" (${(totalBytes / (1024 * 1024)).toFixed(1)} MiB) exceeds VRAM cap (${(vramCap / (1024 * 1024)).toFixed(1)} MiB).`,
      );
    }
    let usedVram = 0;
    for (const slot of this.gpuPrefixSlots.values()) {
      usedVram += slot.byteSize;
    }
    if (usedVram + totalBytes > vramCap) {
      // Auto-evict oldest volatile slots (keeping 'base-system' pinned) before rejecting new caches
      for (const candidate of Array.from(this.gpuPrefixSlots.values())) {
        if (candidate.slotId === 'base-system') {
          continue;
        }
        this.destroyPrefixSlotBuffers(candidate);
        this.gpuPrefixSlots.delete(candidate.slotId);
        usedVram -= candidate.byteSize;
        if (usedVram + totalBytes <= vramCap) {
          break;
        }
      }
    }
    if (usedVram + totalBytes > vramCap) {
      throw new Error(
        `GPU prefix slot "${slotId}" (${(totalBytes / (1024 * 1024)).toFixed(1)} MiB) would exceed VRAM cap (${(vramCap / (1024 * 1024)).toFixed(1)} MiB).`,
      );
    }

    const device = this.device;
    const dnState = this.track(
      device.createBuffer({
        size: dnStateBytes,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
      }),
    );
    const dnConv = this.track(
      device.createBuffer({
        size: dnConvBytes,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
      }),
    );
    const kvPacked0 =
      kv0Bytes > 0
        ? this.track(
            device.createBuffer({
              size: kv0Bytes,
              usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
            }),
          )
        : null;
    const kvPacked1 =
      kv1Bytes > 0
        ? this.track(
            device.createBuffer({
              size: kv1Bytes,
              usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST,
            }),
          )
        : null;

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.dnStateBuf, 0, dnState, 0, dnStateBytes);
    enc.copyBufferToBuffer(this.dnConvBuf, 0, dnConv, 0, dnConvBytes);

    const copyKvToPacked = (
      srcSlab: GpuBufferLike,
      dstPacked: GpuBufferLike | null,
      layerCount: number,
    ) => {
      if (!dstPacked || sliceBytes === 0) return;
      let dstCursor = 0;
      for (let l = 0; l < layerCount; l++) {
        for (let plane = 0 as 0 | 1; plane < 2; plane = (plane + 1) as 0 | 1) {
          for (let h = 0; h < this.arch.attnKvHeads; h++) {
            const srcOff = computeKvSliceByteOffset(
              l,
              plane,
              h,
              this.allocatedKvCapacity,
              this.arch,
              this.kvMode,
            );
            enc.copyBufferToBuffer(srcSlab, srcOff, dstPacked, dstCursor, sliceBytes);
            dstCursor += sliceBytes;
          }
        }
      }
    };
    copyKvToPacked(this.kvSlab0, kvPacked0, this.fa0Count);
    copyKvToPacked(this.kvSlab1, kvPacked1, this.fa1Count);

    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    this.gpuPrefixSlots.set(slotId, {
      slotId,
      seqPos: this.seqPos,
      byteSize: totalBytes,
      dnState,
      dnConv,
      kvPacked0,
      kvPacked1,
    });
    return { seqPos: this.seqPos, byteSize: totalBytes };
  }

  /**
   * Restores a GPU-resident prefix slot back into live DeltaNet and KV buffers
   * in a single command submission, growing KV capacity automatically if needed.
   */
  async restoreGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; restoreMs: number }> {
    if (!this.device) {
      throw new Error('WGSL engine not loaded.');
    }
    const slot = this.gpuPrefixSlots.get(slotId);
    if (!slot) {
      throw new Error(`GPU prefix slot "${slotId}" not found.`);
    }

    const t0 = performance.now();
    await this.ensureKvCapacity(slot.seqPos + 64);

    const device = this.device;
    const dnStateBytes = Math.max(1, this.numDnLayers) * this.dnSingleBytes;
    const dnConvBytes = Math.max(1, this.numDnLayers) * this.dnConvSingleBytes;
    const rowBytes = this.kvRowU32s * 4;
    const sliceBytes = slot.seqPos * rowBytes;

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(slot.dnState, 0, this.dnStateBuf, 0, dnStateBytes);
    enc.copyBufferToBuffer(slot.dnConv, 0, this.dnConvBuf, 0, dnConvBytes);

    const copyPackedToKv = (
      srcPacked: GpuBufferLike | null,
      dstSlab: GpuBufferLike,
      layerCount: number,
    ) => {
      if (!srcPacked || sliceBytes === 0) return;
      let srcCursor = 0;
      for (let l = 0; l < layerCount; l++) {
        for (let plane = 0 as 0 | 1; plane < 2; plane = (plane + 1) as 0 | 1) {
          for (let h = 0; h < this.arch.attnKvHeads; h++) {
            const dstOff = computeKvSliceByteOffset(
              l,
              plane,
              h,
              this.allocatedKvCapacity,
              this.arch,
              this.kvMode,
            );
            enc.copyBufferToBuffer(srcPacked, srcCursor, dstSlab, dstOff, sliceBytes);
            srcCursor += sliceBytes;
          }
        }
      }
    };
    copyPackedToKv(slot.kvPacked0, this.kvSlab0, this.fa0Count);
    copyPackedToKv(slot.kvPacked1, this.kvSlab1, this.fa1Count);

    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    this.seqPos = slot.seqPos;
    this.lastDecodeGpuMs = 0;
    this.decodeBurstMs = 0;
    return { seqPos: slot.seqPos, restoreMs: performance.now() - t0 };
  }

  deleteGpuPrefixSlot(slotId: string): boolean {
    const slot = this.gpuPrefixSlots.get(slotId);
    if (!slot) {
      return false;
    }
    this.destroyPrefixSlotBuffers(slot);
    this.gpuPrefixSlots.delete(slotId);
    return true;
  }

  listGpuPrefixSlots(): Array<{ slotId: string; seqPos: number; byteSize: number }> {
    return Array.from(this.gpuPrefixSlots.values()).map((s) => ({
      slotId: s.slotId,
      seqPos: s.seqPos,
      byteSize: s.byteSize,
    }));
  }

  clearGpuPrefixSlots(): void {
    for (const slot of this.gpuPrefixSlots.values()) {
      this.destroyPrefixSlotBuffers(slot);
    }
    this.gpuPrefixSlots.clear();
  }

  async snapshotState(): Promise<WgslCpuSnapshot> {
    if (!this.device) {
      throw new Error('WGSL engine not loaded.');
    }
    const rowBytes = this.kvRowU32s * 4;
    const perSeqKvBytes = this.numFaLayers * 2 * this.arch.attnKvHeads * this.seqPos * rowBytes;
    const dnCount = Math.max(1, this.numDnLayers);
    const totalBytes = dnCount * (this.dnSingleBytes + this.dnConvSingleBytes) + perSeqKvBytes;
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

    const dnState = await readRaw(this.dnStateBuf, 0, dnCount * this.dnSingleBytes);
    const dnConv = await readRaw(this.dnConvBuf, 0, dnCount * this.dnConvSingleBytes);
    const kvSlices0: Uint8Array[] = [];
    const kvSlices1: Uint8Array[] = [];
    const copyBytes = this.seqPos * rowBytes;

    for (let idx = 0; idx < this.fa0Count; idx++) {
      for (let h = 0; h < this.arch.attnKvHeads; h++) {
        const kOff = computeKvSliceByteOffset(idx, 0, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        const vOff = computeKvSliceByteOffset(idx, 1, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        kvSlices0.push(await readRaw(this.kvSlab0, kOff, copyBytes));
        kvSlices0.push(await readRaw(this.kvSlab0, vOff, copyBytes));
      }
    }
    for (let idx = 0; idx < this.fa1Count; idx++) {
      for (let h = 0; h < this.arch.attnKvHeads; h++) {
        const kOff = computeKvSliceByteOffset(idx, 0, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        const vOff = computeKvSliceByteOffset(idx, 1, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        kvSlices1.push(await readRaw(this.kvSlab1, kOff, copyBytes));
        kvSlices1.push(await readRaw(this.kvSlab1, vOff, copyBytes));
      }
    }

    return { seqPos: this.seqPos, kvMode: this.kvMode, dnState, dnConv, kvSlices0, kvSlices1 };
  }

  async restoreState(snap: WgslCpuSnapshot): Promise<void> {
    if (!this.device || snap.kvMode !== this.kvMode) {
      throw new Error('Incompatible WGSL state snapshot.');
    }
    await this.ensureKvCapacity(snap.seqPos + 64);
    await this.resetState();
    this.device.queue.writeBuffer(this.dnStateBuf, 0, snap.dnState as BufferSource);
    this.device.queue.writeBuffer(this.dnConvBuf, 0, snap.dnConv as BufferSource);

    let cursor0 = 0;
    for (let idx = 0; idx < this.fa0Count; idx++) {
      for (let h = 0; h < this.arch.attnKvHeads; h++) {
        const kOff = computeKvSliceByteOffset(idx, 0, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        const vOff = computeKvSliceByteOffset(idx, 1, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        const k0 = snap.kvSlices0[cursor0];
        const v0 = snap.kvSlices0[cursor0 + 1];
        if (k0 && k0.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab0, kOff, k0 as BufferSource);
        if (v0 && v0.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab0, vOff, v0 as BufferSource);
        cursor0 += 2;
      }
    }
    let cursor1 = 0;
    for (let idx = 0; idx < this.fa1Count; idx++) {
      for (let h = 0; h < this.arch.attnKvHeads; h++) {
        const kOff = computeKvSliceByteOffset(idx, 0, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        const vOff = computeKvSliceByteOffset(idx, 1, h, this.allocatedKvCapacity, this.arch, this.kvMode);
        const k1 = snap.kvSlices1[cursor1];
        const v1 = snap.kvSlices1[cursor1 + 1];
        if (k1 && k1.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab1, kOff, k1 as BufferSource);
        if (v1 && v1.byteLength > 0) this.device.queue.writeBuffer(this.kvSlab1, vOff, v1 as BufferSource);
        cursor1 += 2;
      }
    }
    await this.device.queue.onSubmittedWorkDone();
    this.seqPos = snap.seqPos;
  }

  private async readFullLogits(): Promise<Float32Array> {
    return readGpuF32Slice(this.device!, this.logitsBuf, 0, this.arch.vocabSize * 4);
  }

  private async prefillChunk(
    chunkTokens: number[],
    computeLogits: boolean,
    sampling: SamplingOptions,
    pacingOptions?: {
      pacingMode?: GpuPacingMode;
      calibratedPolicy?: ReturnType<typeof calibratePacingForHardware>;
      abortSignal?: AbortSignal;
      burstState?: { gpuBurstMs: number };
      deferSync?: boolean;
    },
  ): Promise<number | null> {
    const device = this.device!;
    const M = chunkTokens.length;
    if (M === 0) return null;
    if (this.seqPos + M > this.maxSeqCapacity) {
      throw new Error(`Context overflow: seqPos=${this.seqPos} + ${M} > maxSeqCapacity=${this.maxSeqCapacity}`);
    }

    const startPos = this.seqPos;
    const pacingMode = pacingOptions?.pacingMode ?? 'balanced';
    const calibratedPolicy =
      pacingOptions?.calibratedPolicy ??
      calibratePacingForHardware(pacingMode, this.displayHz, this.emaDecodeStepMs);
    const fastPrefillBatch = pacingMode !== 'smooth-ui' && this.emaDecodeStepMs < 22;
    const layersPerSlice = pacingMode === 'max-speed' || fastPrefillBatch ? this.arch.numLayers : 4;
    device.queue.writeBuffer(this.tokenIdsBuf, 0, new Uint32Array(chunkTokens));

    if (!this.prefillUniformScratch) {
      this.prefillUniformScratch = new ArrayBuffer(UNIFORM_MAX_SLOTS * 256);
    }
    const uniformTable = this.prefillUniformScratch;
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

    const H = this.arch.hiddenSize;
    const I = this.arch.intermediateSize;
    const L = this.arch.numLayers;
    const V = this.arch.vocabSize;
    const mTiles = Math.ceil(M / 64);
    const prefillSplits = startPos + M <= 512 ? 1 : startPos + M <= 4096 ? 4 : PREFILL_NUM_SPLITS;
    const prefillSplitsPacked = this.packGqaSplitsParam(prefillSplits);
    const gdnHeadsPacked = this.packGdnHeadsParam();
    const qkRopePacked = this.packQkRopeParam();
    const convWgs = Math.ceil(this.dnConvDim / 128);
    const embedBlocksY = Math.ceil(H / 32);
    const attnQDim = this.arch.attnQHeads * this.arch.attnHeadDim * 2;
    const attnKvDim = this.arch.attnKvHeads * this.arch.attnHeadDim;
    const attnODim = this.arch.attnQHeads * this.arch.attnHeadDim;
    const qkRopeHeadCount = this.arch.attnQHeads + this.arch.attnKvHeads;

    for (let sIdx = 0; sIdx < this.vocabSlices.length; sIdx++) {
      const slice = this.vocabSlices[sIdx]!;
      const rowEnd = this.vocabSlices.length === 1 ? 0 : slice.endRow;
      push(
        this.embedPipe,
        this.pBgEmbedPerSlice[sIdx]!,
        [M, H, slice.startRow, rowEnd, 0, 0, 0, 0],
        M,
        embedBlocksY,
      );
    }

    for (let l = 0; l < L; l++) {
      const la = this.auxLayout.layers[l]!;
      const lo = this.layerOffsets[l]!;
      const bgNormToWork = this.pBgWSlabNormToWork[lo.slabIdx]!;
      const bgAttnToHidden = this.pBgWSlabAttnToHidden[lo.slabIdx]!;

      push(this.rmsNormPipe, this.pBgRmsNorm, [M, H, la.inputNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);

      if (!lo.isFullAttn) {
        const [wQkv, wZ, wAb, wOut, wGate, wUp, wDown] = lo.w;
        push(this.gemmPipe, bgNormToWork, [M, this.dnConvDim, H, wQkv, 0, 0, 0, 0], Math.ceil(this.dnConvDim / 64), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, this.dnValDim, H, wZ, 0, M * this.dnConvDim, 0, 0], Math.ceil(this.dnValDim / 64), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, this.arch.dnValHeads * 2, H, wAb, 0, M * (this.dnConvDim + this.dnValDim), 0, 0], Math.ceil((this.arch.dnValHeads * 2) / 64), mTiles);
        push(
          this.convPipe,
          this.pBgConv,
          [M, la.convWOff, lo.dnIdx * this.dnConvPerLayerF32s, 0, this.dnConvDim, 0, 0, 0],
          convWgs,
        );
        if (M === 1) {
          push(
            this.gdnRecPipe,
            this.pBgGdnRec,
            [
              1,
              lo.dnIdx * this.dnStatePerLayerF32s,
              la.dnAuxOff,
              0,
              M * this.dnConvDim,
              M * (this.dnConvDim + this.dnValDim),
              0,
              gdnHeadsPacked,
            ],
            this.arch.dnValHeads,
          );
        } else {
          for (let tokStart = 0; tokStart < M; tokStart += 32) {
            const C = Math.min(32, M - tokStart);
            push(
              this.gdnChunkPrepPipe,
              this.pBgGdnChunkPrep,
              [C, tokStart, la.dnAuxOff, 0, M * (this.dnConvDim + this.dnValDim), 0, gdnHeadsPacked, 0],
              this.arch.dnValHeads,
            );
            push(
              this.gdnChunkApplyPipe,
              this.pBgGdnChunkApply,
              [0, C, tokStart, lo.dnIdx * this.dnStatePerLayerF32s, la.dnAuxOff, M * this.dnConvDim, 0, gdnHeadsPacked],
              this.arch.dnValHeads,
              8,
            );
          }
          push(
            this.gdnChunkApplyPipe,
            this.pBgGdnChunkApply,
            [1, M, 0, lo.dnIdx * this.dnStatePerLayerF32s, la.dnAuxOff, M * this.dnConvDim, 0, gdnHeadsPacked],
            this.arch.dnValHeads,
            M,
          );
        }
        push(this.gemmPipe, bgAttnToHidden, [M, H, this.dnValDim, wOut, 0, 0, 1, 0], Math.ceil(H / 64), mTiles);

        push(this.rmsNormPipe, this.pBgRmsNorm, [M, H, la.postNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        push(this.gemmPipe, bgNormToWork, [M, I, H, wGate, 0, 0, 0, 0], Math.ceil(I / 64), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, I, H, wUp, 0, M * I, 0, 0], Math.ceil(I / 64), mTiles);
        push(this.swigluPipe, this.pBgSwiGlu, [M * I, 0, M * I, 0, 0, 0, 0, 0], Math.ceil((M * I) / 256));
        push(this.gemmPipe, bgAttnToHidden, [M, H, I, wDown, 0, 0, 1, 0], Math.ceil(H / 64), mTiles);
      } else {
        const [wQg, wK, wV, wO, wGate, wUp, wDown] = lo.w;
        push(this.gemmPipe, bgNormToWork, [M, attnQDim, H, wQg, 0, 0, 0, 0], Math.ceil(attnQDim / 64), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, attnKvDim, H, wK, 0, M * attnQDim, 0, 0], Math.ceil(attnKvDim / 64), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, attnKvDim, H, wV, 0, M * (attnQDim + attnKvDim), 0, 0], Math.ceil(attnKvDim / 64), mTiles);
        push(
          this.qkRopePipe,
          this.pBgQkRope,
          [
            M,
            startPos,
            0,
            M * attnQDim,
            la.qNormOff,
            la.kNormOff,
            this.auxLayout.invFreqOff,
            qkRopePacked,
          ],
          qkRopeHeadCount,
          M,
        );

        const inKv0 = lo.faIdx < this.fa0Count;
        const localKv = inKv0 ? lo.faIdx : lo.faIdx - this.fa0Count;
        const kvBaseU32 = localKv * (this.kvSingleBytes / 4);
        const bgGqaPrefill = inKv0 ? this.bgGqaPrefill0 : this.bgGqaPrefill1;
        push(
          this.gqaPrefillPipe,
          bgGqaPrefill,
          [0, kvBaseU32, this.maxSeqCapacity, startPos, M, M * attnQDim, M * (attnQDim + attnKvDim), prefillSplitsPacked],
          this.arch.attnKvHeads,
          M,
        );
        push(
          this.gqaPrefillPipe,
          bgGqaPrefill,
          [1, kvBaseU32, this.maxSeqCapacity, startPos, M, 0, 0, prefillSplitsPacked],
          this.arch.attnKvHeads,
          Math.ceil(M / 2),
          prefillSplits,
        );
        push(
          this.gqaReducePipe,
          this.pBgGqaReduce,
          [prefillSplits, this.arch.attnHeadDim, 0, this.arch.attnHeadDim * 2, this.arch.attnQHeads, 0, 0, 0],
          this.arch.attnQHeads,
          M,
        );
        push(this.gemmPipe, bgAttnToHidden, [M, H, attnODim, wO, 0, 0, 1, 0], Math.ceil(H / 64), mTiles);

        push(this.rmsNormPipe, this.pBgRmsNorm, [M, H, la.postNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        push(this.gemmPipe, bgNormToWork, [M, I, H, wGate, 0, 0, 0, 0], Math.ceil(I / 64), mTiles);
        push(this.gemmPipe, bgNormToWork, [M, I, H, wUp, 0, M * I, 0, 0], Math.ceil(I / 64), mTiles);
        push(this.swigluPipe, this.pBgSwiGlu, [M * I, 0, M * I, 0, 0, 0, 0, 0], Math.ceil((M * I) / 256));
        push(this.gemmPipe, bgAttnToHidden, [M, H, I, wDown, 0, 0, 1, 0], Math.ceil(H / 64), mTiles);
      }
      if ((l + 1) % layersPerSlice === 0 || l === L - 1) {
        layerSplitEnds.push(steps.length);
      }
    }

    const greedy = sampling.temperature <= 0 || sampling.topK === 1;
    if (computeLogits) {
      push(
        this.rmsNormPipe,
        this.pBgRmsNorm,
        [1, H, this.auxLayout.finalNormOff, (M - 1) * H, 0, 1e-6, 0, 0],
        1,
        1,
        1,
        true,
      );
      for (let sIdx = 0; sIdx < this.vocabSlices.length; sIdx++) {
        const slice = this.vocabSlices[sIdx]!;
        push(
          this.gemvPipe,
          this.pBgLmHeadPerSlice[sIdx]!,
          [slice.rowCount, H, 0, 0, slice.startRow, 0, 0, 0],
          Math.ceil(slice.rowCount / 16),
        );
      }
    }

    const burstState = pacingOptions?.burstState ?? { gpuBurstMs: 0 };

    device.queue.writeBuffer(this.prefillUniformBuf, 0, uniformTable, 0, nextSlot * 256);
    let stepCursor = 0;
    for (let sIdx = 0; sIdx < layerSplitEnds.length; sIdx++) {
      if (pacingOptions?.abortSignal?.aborted) {
        throw new Error('Generation aborted');
      }
      const endIdx = layerSplitEnds[sIdx]!;
      const isLastSlice = sIdx === layerSplitEnds.length - 1;
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (; stepCursor < endIdx; stepCursor++) {
        const st = steps[stepCursor]!;
        pass.setPipeline(st.pipeline);
        pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
        pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
      }
      if (isLastSlice) {
        for (; stepCursor < steps.length; stepCursor++) {
          const st = steps[stepCursor]!;
          pass.setPipeline(st.pipeline);
          pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
          pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
        }
        if (computeLogits && greedy) {
          pass.setPipeline(this.arg1Pipe);
          pass.setBindGroup(0, this.bgArg1);
          pass.dispatchWorkgroups(Math.ceil(V / 256), 1, 1);
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
      if (!pacingOptions?.deferSync) {
        await device.queue.onSubmittedWorkDone();
        burstState.gpuBurstMs += performance.now() - tSlice;
        const { breatherMs, nextBurstMs } = computePrefillSliceBreatherMs(
          pacingMode,
          burstState.gpuBurstMs,
          calibratedPolicy,
        );
        burstState.gpuBurstMs = nextBurstMs;
        if (breatherMs > 0) {
          await sleepMs(breatherMs);
        }
      } else {
        burstState.gpuBurstMs += 8;
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
      skipFinalLogits?: boolean;
    },
  ): Promise<{ nextTokenId: number; prefillMs: number }> {
    const startMs = performance.now();
    await this.ensureKvCapacity(this.seqPos + tokenIds.length);
    const pacingMode = options.pacingMode ?? 'balanced';
    const calibratedPolicy = calibratePacingForHardware(
      pacingMode,
      this.displayHz,
      this.emaDecodeStepMs,
    );
    const burstState = { gpuBurstMs: 0 };
    let lastTok: number | null = options.skipFinalLogits ? -1 : null;
    let offset = 0;
    let chunkIdx = 0;

    const fastPrefillPipeline =
      pacingMode === 'max-speed' ||
      (pacingMode !== 'smooth-ui' && this.emaDecodeStepMs < 22);

    while (offset < tokenIds.length) {
      if (options.abortSignal?.aborted) {
        throw new Error('Generation aborted');
      }
      const remaining = tokenIds.length - offset;
      const baseStepSize = selectPrefillChunkSize(
        pacingMode,
        this.seqPos,
        options.chunkSize,
        calibratedPolicy,
      );
      const stepSize =
        fastPrefillPipeline &&
        remaining > 64 &&
        this.computeTier !== 'tier3-f32-universal'
          ? Math.min(PREFILL_MAX_CHUNK, Math.max(baseStepSize, 128))
          : Math.min(PREFILL_MAX_CHUNK, baseStepSize);
      const end = Math.min(tokenIds.length, offset + stepSize);
      const isLast = end === tokenIds.length;
      const computeLogits = isLast && !options.skipFinalLogits;
      const deferSync = fastPrefillPipeline && !isLast && chunkIdx % 4 !== 3;
      const chunkTok = await this.prefillChunk(tokenIds.slice(offset, end), computeLogits, options.sampling, {
        pacingMode,
        calibratedPolicy,
        abortSignal: options.abortSignal,
        burstState,
        deferSync,
      });
      if (computeLogits) {
        lastTok = chunkTok;
      }
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
    await this.ensureKvCapacity(this.seqPos + 1);
    const device = this.device!;
    if (this.seqPos + 1 > this.maxSeqCapacity) {
      throw new Error(`Context overflow at seqPos=${this.seqPos} (maxSeqCapacity=${this.maxSeqCapacity})`);
    }
    device.queue.writeBuffer(this.tokenIdsBuf, 0, new Uint32Array([tokenId, this.seqPos]));

    const greedy = sampling.temperature <= 0 || sampling.topK === 1;
    const policy = calibratePacingForHardware(pacingMode, this.displayHz, this.emaDecodeStepMs);
    const numSlices = selectDecodeSliceCount(pacingMode, this.seqPos, this.lastDecodeGpuMs, policy);
    const bounds = computeDecodeSliceBounds(this.decodeLayerEndStepIndices, numSlices);
    if (bounds.length === 0) {
      throw new Error('Decode dispatch plan is empty.');
    }
    let gpuMs = 0;
    let breatherMs = 0;
    let greedyTokId: number | null = null;
    const argmaxWgCount = Math.ceil(this.arch.vocabSize / 256);

    for (let s = 0; s < bounds.length; s++) {
      const slice = bounds[s]!;
      const isLastSlice = s === bounds.length - 1;
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let stepIdx = slice.start; stepIdx < slice.end; stepIdx++) {
        const st = this.decodeSteps[stepIdx]!;
        pass.setPipeline(st.pipeline);
        pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
        pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
      }
      if (isLastSlice && greedy) {
        pass.setPipeline(this.arg1Pipe);
        pass.setBindGroup(0, this.bgArg1);
        pass.dispatchWorkgroups(argmaxWgCount, 1, 1);
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
    this.recordDecodeGpuStepMs(gpuMs);

    const nextToken =
      greedyTokId !== null ? greedyTokId : sampleNextToken(await this.readFullLogits(), sampling);

    const burst = computeDecodeBurstBreatherMs(pacingMode, gpuMs, this.decodeBurstMs, policy);
    this.decodeBurstMs = burst.nextBurstMs;
    if (burst.breatherMs > 0) {
      const b0 = performance.now();
      await sleepMs(burst.breatherMs);
      breatherMs += performance.now() - b0;
    }

    return { nextToken, gpuMs, breatherMs };
  }

  private buildOrGetVerifyPlan(M: number): VerifyGraphPlan {
    const existing = this.verifyPlans.get(M);
    if (existing) {
      return existing;
    }
    const device = this.device!;
    const H = this.arch.hiddenSize;
    const I = this.arch.intermediateSize;
    const L = this.arch.numLayers;
    const V = this.arch.vocabSize;
    const argmaxWgCount = Math.ceil(V / 256);
    const batchWx = (rows: number) => Math.ceil(rows / 16);
    const widePipe = this.gemvBatchWidePipe ?? this.gemvBatchPipe;
    const wideWx = (rows: number) =>
      this.gemvBatchWidePipe ? Math.ceil(rows / 8) : Math.ceil(rows / 16);
    const gdnHeadsPacked = this.packGdnHeadsParam();
    const qkRopePacked = this.packQkRopeParam();
    const convWgs = Math.ceil(this.dnConvDim / 128);
    const embedBlocksY = Math.ceil(H / 32);
    const attnQDim = this.arch.attnQHeads * this.arch.attnHeadDim * 2;
    const attnKvDim = this.arch.attnKvHeads * this.arch.attnHeadDim;
    const attnODim = this.arch.attnQHeads * this.arch.attnHeadDim;
    const qkRopeHeadCount = this.arch.attnQHeads + this.arch.attnKvHeads;

    const faCount = this.fa0Count + this.fa1Count;
    const dynSlotCount = Math.max(1, faCount * 4);
    const dynCpu = new ArrayBuffer(dynSlotCount * 256);
    const dynU32 = new Uint32Array(dynCpu);

    const tempTable = new ArrayBuffer(UNIFORM_MAX_SLOTS * 256);
    const u32View = new Uint32Array(tempTable);
    const f32View = new Float32Array(tempTable);
    let nextSlot = dynSlotCount;

    const allocStaticSlot = (
      vals: [number, number, number, number, number, number, number, number],
      isFloat5 = false,
    ): number => {
      if (nextSlot >= UNIFORM_MAX_SLOTS) {
        throw new Error(`Verify plan exceeded UNIFORM_MAX_SLOTS (${UNIFORM_MAX_SLOTS})`);
      }
      const slot = nextSlot++;
      const byteOff = slot * 256;
      const uIdx = byteOff / 4;
      for (let i = 0; i < 8; i++) {
        if (isFloat5 && i === 5) f32View[uIdx + i] = vals[i];
        else u32View[uIdx + i] = vals[i];
      }
      return byteOff;
    };

    const ownedBuffers: GpuBufferLike[] = [];
    const uniformBuf = device.createBuffer({
      size: UNIFORM_MAX_SLOTS * 256,
      usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST,
    });
    ownedBuffers.push(uniformBuf);

    const makeDyn4 = (b1: GpuBufferLike, b2: GpuBufferLike, b3: GpuBufferLike) =>
      device.createBindGroup({
        layout: this.dynStorage4Bgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
        ],
      });
    const makeDynRw = (b1: GpuBufferLike, b2: GpuBufferLike, b3: GpuBufferLike) =>
      device.createBindGroup({
        layout: this.dynRwStorageBgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
        ],
      });

    const vBgEmbedPerSlice = this.lmHeadSlabs.map((slab) =>
      makeDyn4(this.tokenIdsBuf, slab, this.hiddenBuf),
    );
    const vBgRmsNorm = makeDyn4(this.auxBuf, this.hiddenBuf, this.normBuf);
    const vBgSwiGlu = makeDyn4(this.auxBuf, this.workBuf, this.attnOutBuf);
    const vBgWSlabNormToWork = this.wSlabs.map((slab) => makeDyn4(slab, this.normBuf, this.workBuf));
    const vBgWSlabNormToAttn = this.wSlabs.map((slab) => makeDyn4(slab, this.normBuf, this.attnOutBuf));
    const vBgWSlabAttnToHidden = this.wSlabs.map((slab) => makeDyn4(slab, this.attnOutBuf, this.hiddenBuf));
    const vBgLmHead = this.lmHeadSlabs.map((slab) => makeDyn4(slab, this.normBuf, this.verifyLogitsBuf));
    const vBgConv = makeDynRw(this.auxBuf, this.dnConvBuf, this.workBuf);
    const vBgGqaReduce = makeDyn4(this.workBuf, this.gqaSplitBuf, this.attnOutBuf);
    const vBgGdnChunkPrep = makeDyn4(this.auxBuf, this.workBuf, this.gqaSplitBuf);
    const vBgGdnChunkApply = device.createBindGroup({
      layout: this.gdnApplyBgl,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.auxBuf } },
        { binding: 2, resource: { buffer: this.workBuf } },
        { binding: 3, resource: { buffer: this.gqaSplitBuf } },
        { binding: 4, resource: { buffer: this.dnStateBuf } },
        { binding: 5, resource: { buffer: this.attnOutBuf } },
      ],
    });
    const vBgQkRope = device.createBindGroup({
      layout: this.qkRopeBgl,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
        { binding: 1, resource: { buffer: this.auxBuf } },
        { binding: 2, resource: { buffer: this.workBuf } },
        { binding: 3, resource: { buffer: this.tokenIdsBuf } },
      ],
    });
    const vBgGqaPrefill0 = makeDynRw(this.workBuf, this.kvSlab0, this.gqaSplitBuf);
    const vBgGqaPrefill1 = makeDynRw(this.workBuf, this.kvSlab1, this.gqaSplitBuf);

    const uArgBatchBuf = device.createBuffer({
      size: 16,
      usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST,
    });
    const uSpecAcceptBuf = device.createBuffer({
      size: 16,
      usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST,
    });
    ownedBuffers.push(uArgBatchBuf, uSpecAcceptBuf);
    device.queue.writeBuffer(uArgBatchBuf, 0, new Uint32Array([V, M, argmaxWgCount, 0]));
    device.queue.writeBuffer(uSpecAcceptBuf, 0, new Uint32Array([M - 1, 0, 0, 0]));

    const bgArgBatch1 = device.createBindGroup({
      layout: this.staticBgl,
      entries: [
        { binding: 0, resource: { buffer: uArgBatchBuf } },
        { binding: 1, resource: { buffer: this.verifyLogitsBuf } },
        { binding: 2, resource: { buffer: this.partValBuf } },
        { binding: 3, resource: { buffer: this.partIdxBuf } },
      ],
    });
    const bgArgBatch2 = device.createBindGroup({
      layout: this.staticRead2Bgl,
      entries: [
        { binding: 0, resource: { buffer: uArgBatchBuf } },
        { binding: 1, resource: { buffer: this.partValBuf } },
        { binding: 2, resource: { buffer: this.partIdxBuf } },
        { binding: 3, resource: { buffer: this.verifyTokOutBuf } },
      ],
    });
    const bgSpecAccept = device.createBindGroup({
      layout: this.staticRead2Bgl,
      entries: [
        { binding: 0, resource: { buffer: uSpecAcceptBuf } },
        { binding: 1, resource: { buffer: this.verifyTokOutBuf } },
        { binding: 2, resource: { buffer: this.tokenIdsBuf } },
        { binding: 3, resource: { buffer: this.verifyAcceptOutBuf } },
      ],
    });

    const steps: DispatchStep[] = [];
    const faPatches: VerifyFaPatchTarget[] = [];

    const pushStatic = (
      pipeline: unknown,
      bindGroup: unknown,
      vals: [number, number, number, number, number, number, number, number],
      wx: number,
      wy = 1,
      wz = 1,
      isFloat5 = false,
    ) => {
      steps.push({
        pipeline,
        bindGroup,
        uniformByteOffset: allocStaticSlot(vals, isFloat5),
        wx,
        wy,
        wz,
      });
    };

    for (let sIdx = 0; sIdx < this.vocabSlices.length; sIdx++) {
      const slice = this.vocabSlices[sIdx]!;
      const rowEnd = this.vocabSlices.length === 1 ? 0 : slice.endRow;
      pushStatic(this.embedPipe, vBgEmbedPerSlice[sIdx]!, [M, H, slice.startRow, rowEnd, 0, 0, 0, 0], M, embedBlocksY);
    }

    for (let l = 0; l < L; l++) {
      const la = this.auxLayout.layers[l]!;
      const lo = this.layerOffsets[l]!;
      const bgNormToWork = vBgWSlabNormToWork[lo.slabIdx]!;
      const bgAttnToHidden = vBgWSlabAttnToHidden[lo.slabIdx]!;

      if (!lo.isFullAttn) {
        const [wQkv, wZ, wAb, wOut, wGate, wUp, wDown] = lo.w;
        const dnNormOff = lo.dnIdx * MAX_SPEC_TOKENS * H;
        pushStatic(this.rmsNormPipe, vBgRmsNorm, [M, H, la.inputNormOff, 0, dnNormOff, 1e-6, 0, 0], M, 1, 1, true);
        pushStatic(this.gemvBatchPipe, bgNormToWork, [this.dnConvDim, H, wQkv, dnNormOff, 0, 0, M, 0], batchWx(this.dnConvDim));
        pushStatic(this.gemvBatchPipe, bgNormToWork, [this.dnValDim, H, wZ, dnNormOff, M * this.dnConvDim, 0, M, 0], batchWx(this.dnValDim));
        pushStatic(this.gemvBatchPipe, bgNormToWork, [this.arch.dnValHeads * 2, H, wAb, dnNormOff, M * (this.dnConvDim + this.dnValDim), 0, M, 0], batchWx(this.arch.dnValHeads * 2));

        pushStatic(this.convPipe, vBgConv, [M, la.convWOff, lo.dnIdx * this.dnConvPerLayerF32s, 0, this.dnConvDim, 0, 0, 0], convWgs);
        pushStatic(
          this.gdnChunkPrepPipe,
          vBgGdnChunkPrep,
          [M, 0, la.dnAuxOff, 0, M * (this.dnConvDim + this.dnValDim), 0, gdnHeadsPacked, 0],
          this.arch.dnValHeads,
        );
        pushStatic(
          this.gdnChunkApplyPipe,
          vBgGdnChunkApply,
          [0, M, 0, lo.dnIdx * this.dnStatePerLayerF32s, la.dnAuxOff, M * this.dnConvDim, 0, gdnHeadsPacked],
          this.arch.dnValHeads,
          8,
        );
        pushStatic(
          this.gdnChunkApplyPipe,
          vBgGdnChunkApply,
          [1, M, 0, lo.dnIdx * this.dnStatePerLayerF32s, la.dnAuxOff, M * this.dnConvDim, 0, gdnHeadsPacked],
          this.arch.dnValHeads,
          M,
        );
        pushStatic(widePipe, bgAttnToHidden, [H, this.dnValDim, wOut, 0, 0, 1, M, 0], wideWx(H));
        pushStatic(this.rmsNormPipe, vBgRmsNorm, [M, H, la.postNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        if (this.gateUpSwigluBatchPipe) {
          pushStatic(
            this.gateUpSwigluBatchPipe,
            vBgWSlabNormToAttn[lo.slabIdx]!,
            [I, H, wGate, 0, 0, 0, M, wUp],
            Math.ceil(I / 8),
          );
        } else {
          pushStatic(this.gemvBatchPipe, bgNormToWork, [I, H, wGate, 0, 0, 0, M, 0], batchWx(I));
          pushStatic(this.gemvBatchPipe, bgNormToWork, [I, H, wUp, 0, M * I, 0, M, 0], batchWx(I));
          pushStatic(this.swigluPipe, vBgSwiGlu, [M * I, 0, M * I, 0, 0, 0, 0, 0], Math.ceil((M * I) / 256));
        }
        pushStatic(widePipe, bgAttnToHidden, [H, I, wDown, 0, 0, 1, M, 0], wideWx(H));
      } else {
        const [wQg, wK, wV, wO, wGate, wUp, wDown] = lo.w;
        pushStatic(this.rmsNormPipe, vBgRmsNorm, [M, H, la.inputNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        pushStatic(this.gemvBatchPipe, bgNormToWork, [attnQDim, H, wQg, 0, 0, 0, M, 0], batchWx(attnQDim));
        pushStatic(this.gemvBatchPipe, bgNormToWork, [attnKvDim, H, wK, 0, M * attnQDim, 0, M, 0], batchWx(attnKvDim));
        pushStatic(this.gemvBatchPipe, bgNormToWork, [attnKvDim, H, wV, 0, M * (attnQDim + attnKvDim), 0, M, 0], batchWx(attnKvDim));

        const qkSlot = lo.faIdx * 4;
        const gqa0Slot = qkSlot + 1;
        const gqa1Slot = qkSlot + 2;
        const redSlot = qkSlot + 3;
        const qkUIdx = (qkSlot * 256) / 4;
        const gqa0UIdx = (gqa0Slot * 256) / 4;
        const gqa1UIdx = (gqa1Slot * 256) / 4;
        const redUIdx = (redSlot * 256) / 4;

        const inKv0 = lo.faIdx < this.fa0Count;
        const localKv = inKv0 ? lo.faIdx : lo.faIdx - this.fa0Count;
        const kvBaseU32 = localKv * (this.kvSingleBytes / 4);
        const bgGqaPrefill = inKv0 ? vBgGqaPrefill0 : vBgGqaPrefill1;

        dynU32.set([M, 0, 0, M * attnQDim, la.qNormOff, la.kNormOff, this.auxLayout.invFreqOff, qkRopePacked], qkUIdx);
        dynU32.set([0, kvBaseU32, this.maxSeqCapacity, 0, M, M * attnQDim, M * (attnQDim + attnKvDim), 0], gqa0UIdx);
        dynU32.set([1, kvBaseU32, this.maxSeqCapacity, 0, M, 0, 0, 0], gqa1UIdx);
        dynU32.set([1, this.arch.attnHeadDim, 0, this.arch.attnHeadDim * 2, this.arch.attnQHeads, 0, 0, 0], redUIdx);

        steps.push({ pipeline: this.qkRopePipe, bindGroup: vBgQkRope, uniformByteOffset: qkSlot * 256, wx: qkRopeHeadCount, wy: M, wz: 1 });
        steps.push({ pipeline: this.gqaPrefillPipe, bindGroup: bgGqaPrefill, uniformByteOffset: gqa0Slot * 256, wx: this.arch.attnKvHeads, wy: M, wz: 1 });
        const gqa1Step: DispatchStep = { pipeline: this.gqaPrefillPipe, bindGroup: bgGqaPrefill, uniformByteOffset: gqa1Slot * 256, wx: this.arch.attnKvHeads, wy: Math.ceil(M / 2), wz: 1 };
        steps.push(gqa1Step);
        steps.push({ pipeline: this.gqaReducePipe, bindGroup: vBgGqaReduce, uniformByteOffset: redSlot * 256, wx: this.arch.attnQHeads, wy: M, wz: 1 });
        faPatches.push({ qkUIdx, gqa0UIdx, gqa1UIdx, redUIdx, gqa1Step } as VerifyFaPatchTarget & { redUIdx: number });

        pushStatic(widePipe, bgAttnToHidden, [H, attnODim, wO, 0, 0, 1, M, 0], wideWx(H));
        pushStatic(this.rmsNormPipe, vBgRmsNorm, [M, H, la.postNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
        if (this.gateUpSwigluBatchPipe) {
          pushStatic(
            this.gateUpSwigluBatchPipe,
            vBgWSlabNormToAttn[lo.slabIdx]!,
            [I, H, wGate, 0, 0, 0, M, wUp],
            Math.ceil(I / 8),
          );
        } else {
          pushStatic(this.gemvBatchPipe, bgNormToWork, [I, H, wGate, 0, 0, 0, M, 0], batchWx(I));
          pushStatic(this.gemvBatchPipe, bgNormToWork, [I, H, wUp, 0, M * I, 0, M, 0], batchWx(I));
          pushStatic(this.swigluPipe, vBgSwiGlu, [M * I, 0, M * I, 0, 0, 0, 0, 0], Math.ceil((M * I) / 256));
        }
        pushStatic(widePipe, bgAttnToHidden, [H, I, wDown, 0, 0, 1, M, 0], wideWx(H));
      }
    }

    pushStatic(this.rmsNormPipe, vBgRmsNorm, [M, H, this.auxLayout.finalNormOff, 0, 0, 1e-6, 0, 0], M, 1, 1, true);
    for (let sIdx = 0; sIdx < this.vocabSlices.length; sIdx++) {
      const slice = this.vocabSlices[sIdx]!;
      pushStatic(this.gemvBatchPipe, vBgLmHead[sIdx]!, [slice.rowCount, H, 0, 0, slice.startRow, 0, M, V], batchWx(slice.rowCount));
    }

    const dynByteStart = 0;
    const staticStartByte = dynSlotCount * 256;
    const staticByteLen = (nextSlot - dynSlotCount) * 256;
    if (staticByteLen > 0) {
      device.queue.writeBuffer(uniformBuf, staticStartByte, tempTable, staticStartByte, staticByteLen);
    }

    const plan: VerifyGraphPlan = {
      M,
      uniformBuf,
      dynCpu,
      dynU32,
      dynByteStart,
      steps,
      faPatches,
      bgArgBatch1,
      bgArgBatch2,
      bgSpecAccept,
      ownedBuffers,
    };
    this.verifyPlans.set(M, plan);
    return plan;
  }

  /**
   * Verifies up to `MAX_SPEC_DRAFT` (7) drafted continuation tokens in a single batched forward pass
   * (`M = 1 + draftTokens.length <= 8`) with cached per-M uniform/bind-group tables, batched GPU
   * argmax, and GPU-side greedy prefix accept (`64B` readback), rolling back only DeltaNet state on divergence.
   */
  async decodeSpeculativeStep(
    currentTokenId: number,
    draftTokens: number[],
    sampling: SamplingOptions,
    pacingMode: GpuPacingMode = 'balanced',
  ): Promise<{
    tokens: number[];
    acceptedCount: number;
    draftedCount: number;
    gpuMs: number;
    breatherMs: number;
  }> {
    if (sampling.temperature > 0 && sampling.topK !== 1) {
      throw new Error('decodeSpeculativeStep requires greedy sampling.');
    }
    if (draftTokens.length === 0 || draftTokens.length > MAX_SPEC_DRAFT) {
      throw new Error(`Invalid draftTokens length (${draftTokens.length}); expected 1..${MAX_SPEC_DRAFT}.`);
    }
    const tokens = [currentTokenId, ...draftTokens];
    const M = tokens.length;
    await this.ensureKvCapacity(this.seqPos + M);
    const device = this.device!;
    if (this.seqPos + M > this.maxSeqCapacity) {
      throw new Error(`Context overflow at seqPos=${this.seqPos} + ${M} > maxSeqCapacity=${this.maxSeqCapacity}`);
    }

    const startPos = this.seqPos;
    const V = this.arch.vocabSize;
    const L = this.arch.numLayers;
    const argmaxWgCount = Math.ceil(V / 256);
    const verifySplits = startPos + M <= 512 ? 1 : startPos + M <= 4096 ? 4 : PREFILL_NUM_SPLITS;
    const verifySplitsPacked = this.packGqaSplitsParam(verifySplits);
    const gdnHeadsPacked = this.packGdnHeadsParam();
    const convWgs = Math.ceil(this.dnConvDim / 128);
    const dnCount = Math.max(1, this.numDnLayers);

    const plan = this.buildOrGetVerifyPlan(M);
    for (const fp of plan.faPatches) {
      const target = fp as VerifyFaPatchTarget & { redUIdx: number };
      plan.dynU32[target.qkUIdx + 1] = startPos;
      plan.dynU32[target.gqa0UIdx + 3] = startPos;
      plan.dynU32[target.gqa0UIdx + 7] = verifySplitsPacked;
      plan.dynU32[target.gqa1UIdx + 3] = startPos;
      plan.dynU32[target.gqa1UIdx + 7] = verifySplitsPacked;
      plan.dynU32[target.redUIdx] = verifySplits;
      target.gqa1Step.wz = verifySplits;
    }

    device.queue.writeBuffer(this.tokenIdsBuf, 0, new Uint32Array(tokens));
    device.queue.writeBuffer(plan.uniformBuf, plan.dynByteStart, plan.dynCpu);

    const enc = device.createCommandEncoder();
    // Snapshot DeltaNet state before running speculative M-token pass
    enc.copyBufferToBuffer(this.dnStateBuf, 0, this.dnStateBackupBuf, 0, dnCount * this.dnSingleBytes);
    enc.copyBufferToBuffer(this.dnConvBuf, 0, this.dnConvBackupBuf, 0, dnCount * this.dnConvSingleBytes);

    const pass = enc.beginComputePass();
    for (let i = 0; i < plan.steps.length; i++) {
      const st = plan.steps[i]!;
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
      pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
    }
    pass.setPipeline(this.argmaxBatch1Pipe);
    pass.setBindGroup(0, plan.bgArgBatch1);
    pass.dispatchWorkgroups(argmaxWgCount, M, 1);
    pass.setPipeline(this.argmaxBatch2Pipe);
    pass.setBindGroup(0, plan.bgArgBatch2);
    pass.dispatchWorkgroups(M, 1, 1);
    pass.setPipeline(this.specAcceptPipe);
    pass.setBindGroup(0, plan.bgSpecAccept);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    enc.copyBufferToBuffer(this.verifyAcceptOutBuf, 0, this.verifyStaging64B, 0, 64);

    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await this.verifyStaging64B.mapAsync(GPU_MAP_READ);
    let gpuMs = performance.now() - t0;
    const acceptView = new Uint32Array(this.verifyStaging64B.getMappedRange());
    const acceptedCount = Math.min(draftTokens.length, acceptView[0]!);
    const predicted = Array.from(acceptView.subarray(2, 2 + M));
    this.verifyStaging64B.unmap();

    if (acceptedCount < draftTokens.length) {
      const keepTokens = acceptedCount + 1;
      const H = this.arch.hiddenSize;
      const batchWx = (rows: number) => Math.ceil(rows / 16);
      const rbU32 = new Uint32Array(this.rbUniformScratch);
      let rbSlot = 0;
      const pushRb = (vals: [number, number, number, number, number, number, number, number]) => {
        const off = rbSlot++ * 256;
        const uIdx = off / 4;
        for (let i = 0; i < 8; i++) rbU32[uIdx + i] = vals[i];
        return off;
      };

      const rbEnc = device.createCommandEncoder();
      rbEnc.copyBufferToBuffer(this.dnStateBackupBuf, 0, this.dnStateBuf, 0, dnCount * this.dnSingleBytes);
      rbEnc.copyBufferToBuffer(this.dnConvBackupBuf, 0, this.dnConvBuf, 0, dnCount * this.dnConvSingleBytes);

      const rPass = rbEnc.beginComputePass();
      for (let l = 0; l < L; l++) {
        const lo = this.layerOffsets[l]!;
        if (lo.isFullAttn) continue;
        const la = this.auxLayout.layers[l]!;
        const [wQkv, wZ, wAb] = lo.w;
        const dnNormOff = lo.dnIdx * MAX_SPEC_TOKENS * H;
        const pBgNormToWork = this.pBgWSlabNormToWork[lo.slabIdx]!;

        const qOff = pushRb([this.dnConvDim, H, wQkv, dnNormOff, 0, 0, keepTokens, 0]);
        rPass.setPipeline(this.gemvBatchPipe);
        rPass.setBindGroup(0, pBgNormToWork, [qOff]);
        rPass.dispatchWorkgroups(batchWx(this.dnConvDim), 1, 1);

        const zOff = pushRb([this.dnValDim, H, wZ, dnNormOff, keepTokens * this.dnConvDim, 0, keepTokens, 0]);
        rPass.setPipeline(this.gemvBatchPipe);
        rPass.setBindGroup(0, pBgNormToWork, [zOff]);
        rPass.dispatchWorkgroups(batchWx(this.dnValDim), 1, 1);

        const abOff = pushRb([this.arch.dnValHeads * 2, H, wAb, dnNormOff, keepTokens * (this.dnConvDim + this.dnValDim), 0, keepTokens, 0]);
        rPass.setPipeline(this.gemvBatchPipe);
        rPass.setBindGroup(0, pBgNormToWork, [abOff]);
        rPass.dispatchWorkgroups(batchWx(this.arch.dnValHeads * 2), 1, 1);

        const cOff = pushRb([keepTokens, la.convWOff, lo.dnIdx * this.dnConvPerLayerF32s, 0, this.dnConvDim, 0, 0, 0]);
        rPass.setPipeline(this.convPipe);
        rPass.setBindGroup(0, this.pBgConv, [cOff]);
        rPass.dispatchWorkgroups(convWgs, 1, 1);

        const gOff = pushRb([
          keepTokens,
          lo.dnIdx * this.dnStatePerLayerF32s,
          la.dnAuxOff,
          0,
          keepTokens * this.dnConvDim,
          keepTokens * (this.dnConvDim + this.dnValDim),
          0,
          gdnHeadsPacked,
        ]);
        rPass.setPipeline(this.gdnRecPipe);
        rPass.setBindGroup(0, this.pBgGdnRec, [gOff]);
        rPass.dispatchWorkgroups(this.arch.dnValHeads, 1, 1);
      }
      rPass.end();

      device.queue.writeBuffer(this.prefillUniformBuf, 0, this.rbUniformScratch, 0, rbSlot * 256);
      const tRb = performance.now();
      device.queue.submit([rbEnc.finish()]);
      await device.queue.onSubmittedWorkDone();
      gpuMs += performance.now() - tRb;
    }

    this.seqPos = startPos + acceptedCount + 1;
    this.recordDecodeGpuStepMs(gpuMs / (acceptedCount + 1));

    let breatherMs = 0;
    const policy = calibratePacingForHardware(pacingMode, this.displayHz, this.emaDecodeStepMs);
    const burst = computeDecodeBurstBreatherMs(pacingMode, gpuMs, this.decodeBurstMs, policy);
    this.decodeBurstMs = burst.nextBurstMs;
    if (burst.breatherMs > 0) {
      const b0 = performance.now();
      await sleepMs(burst.breatherMs);
      breatherMs += performance.now() - b0;
    }

    return {
      tokens: predicted.slice(0, acceptedCount + 1),
      acceptedCount,
      draftedCount: draftTokens.length,
      gpuMs,
      breatherMs,
    };
  }

  private async verifyRealKernels(
    device: GpuDeviceLike,
    dynStorage4Bgl: unknown,
    gdnRecBgl: unknown,
    gdnApplyBgl: unknown,
    qkRopeBgl: unknown,
    dynRwStorageBgl: unknown,
    gqaSplitBgl: unknown,
  ): Promise<void> {
    const wasm = this.wasm!;
    const tempBufs: GpuBufferLike[] = [];
    const mk = (size: number, usage: number) => {
      const b = device.createBuffer({ size, usage });
      tempBufs.push(b);
      return b;
    };

    try {
      // 0. Verify Q4 GEMV and Wide-K GEMV on (64x1024), (64x512), and odd (37x96)
      for (const [rows, cols] of [
        [64, 1024],
        [64, 512],
        [37, 96],
      ] as const) {
        const blocks = cols / 32;
        const wU32Len = rows * blocks * 5;
        const scratchBase = wasm.wasm_scratch_ptr();
        const wPtr = scratchBase;
        const xPtr = wPtr + wU32Len * 4;
        const refOutPtr = xPtr + cols * 4;
        const wCpu = new Uint32Array(wasm.memory.buffer, wPtr, wU32Len);
        for (let b = 0; b < rows * blocks; b++) {
          const i = b * 5;
          wCpu[i] = (0x13572468 + b * 13) >>> 0;
          wCpu[i + 1] = (0x24681357 ^ (b * 23)) >>> 0;
          wCpu[i + 2] = (0x0a1b2c3d + b * 7) >>> 0;
          wCpu[i + 3] = (0x55aa33cc ^ (b * 31)) >>> 0;
          wCpu[i + 4] = (0x2b80 | (0xae00 << 16)) >>> 0;
        }
        const xCpu = new Float32Array(wasm.memory.buffer, xPtr, cols);
        for (let k = 0; k < cols; k++) xCpu[k] = Math.cos(k * 0.13) * 0.7;

        const wGpu = mk(wU32Len * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const xGpu = mk(cols * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const yGpu = mk(rows * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uGpu = mk(256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
        device.queue.writeBuffer(wGpu, 0, new Uint32Array(wasm.memory.buffer, wPtr, wU32Len));
        device.queue.writeBuffer(xGpu, 0, xCpu.slice());
        device.queue.writeBuffer(uGpu, 0, new Uint32Array([rows, cols, 0, 0, 0, 0, 0, 0]));
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
        pass.setPipeline(this.gemvPipe);
        pass.setBindGroup(0, bg, [0]);
        pass.dispatchWorkgroups(Math.ceil(rows / 16), 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const gpuOut = await readGpuF32Slice(device, yGpu, 0, rows * 4);

        wasm.cpu_ref_q4_gemv(rows, cols, wPtr, xPtr, refOutPtr);
        const refRow = new Float32Array(wasm.memory.buffer, refOutPtr, rows);
        let maxErr = 0;
        for (let r = 0; r < rows; r++) {
          maxErr = Math.max(maxErr, Math.abs(gpuOut[r] - refRow[r]));
        }
        if (maxErr >= 0.05) {
          throw new Error(`Q4 GEMV (${rows}x${cols}) verification failed: maxAbsError=${maxErr}`);
        }

        const encWide = device.createCommandEncoder();
        const passWide = encWide.beginComputePass();
        passWide.setPipeline(this.gemvWidePipe);
        passWide.setBindGroup(0, bg, [0]);
        passWide.dispatchWorkgroups(Math.ceil(rows / 4), 1, 1);
        passWide.end();
        device.queue.submit([encWide.finish()]);
        const gpuWideOut = await readGpuF32Slice(device, yGpu, 0, rows * 4);
        let maxWideErr = 0;
        for (let r = 0; r < rows; r++) {
          maxWideErr = Math.max(maxWideErr, Math.abs(gpuWideOut[r] - refRow[r]));
        }
        if (maxWideErr >= 0.05) {
          throw new Error(`Q4 GEMV WIDE_K (${rows}x${cols}) verification failed: maxAbsError=${maxWideErr}`);
        }
      }

      // 0b. Verify fused Q4 Gate + Up + SwiGLU
      {
        const rows = 64;
        const cols = 512;
        const blocks = cols / 32;
        const matU32Len = rows * blocks * 5;
        const scratchBase = wasm.wasm_scratch_ptr();
        const wGatePtr = scratchBase;
        const wUpPtr = wGatePtr + matU32Len * 4;
        const xPtr = wUpPtr + matU32Len * 4;
        const refGatePtr = xPtr + cols * 4;
        const refUpPtr = refGatePtr + rows * 4;

        const wCpu = new Uint32Array(wasm.memory.buffer, wGatePtr, matU32Len * 2);
        for (let b = 0; b < rows * blocks * 2; b++) {
          const i = b * 5;
          wCpu[i] = (0x11223344 + b * 13) >>> 0;
          wCpu[i + 1] = (0x44332211 ^ (b * 19)) >>> 0;
          wCpu[i + 2] = (0x01020304 + b * 7) >>> 0;
          wCpu[i + 3] = (0x55aa33cc ^ (b * 29)) >>> 0;
          wCpu[i + 4] = (0x2b80 | (0xae00 << 16)) >>> 0;
        }
        const xCpu = new Float32Array(wasm.memory.buffer, xPtr, cols);
        for (let k = 0; k < cols; k++) xCpu[k] = Math.sin(k * 0.17) * 0.8;

        const wGpu = mk(matU32Len * 2 * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const xGpu = mk(cols * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const yGpu = mk(rows * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uGpu = mk(256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
        device.queue.writeBuffer(wGpu, 0, new Uint32Array(wasm.memory.buffer, wGatePtr, matU32Len * 2));
        device.queue.writeBuffer(xGpu, 0, xCpu.slice());
        device.queue.writeBuffer(uGpu, 0, new Uint32Array([rows, cols, 0, 0, 0, 0, matU32Len, 0]));
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
        pass.setPipeline(this.gateUpSwigluPipe);
        pass.setBindGroup(0, bg, [0]);
        pass.dispatchWorkgroups(Math.ceil(rows / 8), 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const gpuOut = await readGpuF32Slice(device, yGpu, 0, rows * 4);

        wasm.cpu_ref_q4_gemv(rows, cols, wGatePtr, xPtr, refGatePtr);
        wasm.cpu_ref_q4_gemv(rows, cols, wUpPtr, xPtr, refUpPtr);
        const refG = new Float32Array(wasm.memory.buffer, refGatePtr, rows);
        const refU = new Float32Array(wasm.memory.buffer, refUpPtr, rows);
        let maxErr = 0;
        for (let r = 0; r < rows; r++) {
          const expected = (refG[r] / (1 + Math.exp(-refG[r]))) * refU[r];
          maxErr = Math.max(maxErr, Math.abs(gpuOut[r] - expected));
        }
        if (maxErr >= 0.05) {
          throw new Error(`Q4_GATE_UP_SWIGLU verification failed: maxAbsError=${maxErr}`);
        }
      }

      // 0c. Verify Q4_GEMV_BATCH on (M=5, rows=64, cols=512)
      {
        const M = 5;
        const rows = 64;
        const cols = 512;
        const blocks = cols / 32;
        const wU32Len = rows * blocks * 5;
        const scratchBase = wasm.wasm_scratch_ptr();
        const wPtr = scratchBase;
        const xPtr = wPtr + wU32Len * 4;
        const refOutPtr = xPtr + cols * 4;
        const wCpu = new Uint32Array(wasm.memory.buffer, wPtr, wU32Len);
        for (let b = 0; b < rows * blocks; b++) {
          const i = b * 5;
          wCpu[i] = (0x12345678 + b * 17) >>> 0;
          wCpu[i + 1] = (0x87654321 ^ (b * 29)) >>> 0;
          wCpu[i + 2] = (0x0f1e2d3c + b * 11) >>> 0;
          wCpu[i + 3] = (0x5a6b7c8d ^ (b * 19)) >>> 0;
          wCpu[i + 4] = (0x2b80 | (0xae00 << 16)) >>> 0;
        }
        const xFull = new Float32Array(M * cols);
        for (let t = 0; t < M; t++) {
          for (let k = 0; k < cols; k++) xFull[t * cols + k] = Math.cos(t * 0.7 + k * 0.11) * 0.65;
        }
        const wGpu = mk(wU32Len * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const xGpu = mk(xFull.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const yGpu = mk(M * rows * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uGpu = mk(256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
        device.queue.writeBuffer(wGpu, 0, new Uint32Array(wasm.memory.buffer, wPtr, wU32Len));
        device.queue.writeBuffer(xGpu, 0, xFull);
        device.queue.writeBuffer(uGpu, 0, new Uint32Array([rows, cols, 0, 0, 0, 0, M, 0]));
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
        pass.setPipeline(this.gemvBatchPipe);
        pass.setBindGroup(0, bg, [0]);
        pass.dispatchWorkgroups(Math.ceil(rows / 16), 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const gpuOut = await readGpuF32Slice(device, yGpu, 0, M * rows * 4);

        let maxErr = 0;
        for (let t = 0; t < M; t++) {
          new Float32Array(wasm.memory.buffer, xPtr, cols).set(xFull.subarray(t * cols, (t + 1) * cols));
          wasm.cpu_ref_q4_gemv(rows, cols, wPtr, xPtr, refOutPtr);
          const refRow = new Float32Array(wasm.memory.buffer, refOutPtr, rows);
          for (let r = 0; r < rows; r++) {
            maxErr = Math.max(maxErr, Math.abs(gpuOut[t * rows + r] - refRow[r]));
          }
        }
        if (maxErr >= 0.05) {
          throw new Error(`Q4_GEMV_BATCH verification failed: maxAbsError=${maxErr}`);
        }
      }

      // 0d. Verify RMSNORM_BATCH
      {
        const M = 2;
        const len = 512;
        const auxCpu = new Float32Array(len);
        const srcCpu = new Float32Array(M * len);
        for (let i = 0; i < len; i++) auxCpu[i] = 0.8 + 0.3 * Math.sin(i * 0.1);
        for (let i = 0; i < srcCpu.length; i++) srcCpu[i] = Math.cos(i * 0.07) * 1.5;

        const auxG = mk(auxCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const srcG = mk(srcCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const dstG = mk(srcCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const uG = mk(256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
        device.queue.writeBuffer(auxG, 0, auxCpu);
        device.queue.writeBuffer(srcG, 0, srcCpu);
        const uBuf = new ArrayBuffer(32);
        new Uint32Array(uBuf).set([M, len, 0, 0, 0, 0, 0, 0]);
        new Float32Array(uBuf)[5] = 1e-6;
        device.queue.writeBuffer(uG, 0, uBuf);
        const bg = device.createBindGroup({
          layout: dynStorage4Bgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: auxG } },
            { binding: 2, resource: { buffer: srcG } },
            { binding: 3, resource: { buffer: dstG } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.rmsNormPipe);
        pass.setBindGroup(0, bg, [0]);
        pass.dispatchWorkgroups(M, 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const gpuOut = await readGpuF32Slice(device, dstG, 0, M * len * 4);
        let maxErr = 0;
        for (let t = 0; t < M; t++) {
          let sq = 0;
          for (let i = 0; i < len; i++) sq += srcCpu[t * len + i]! ** 2;
          const inv = 1 / Math.sqrt(sq / len + 1e-6);
          for (let i = 0; i < len; i++) {
            const ref = srcCpu[t * len + i]! * inv * auxCpu[i]!;
            maxErr = Math.max(maxErr, Math.abs(gpuOut[t * len + i]! - ref));
          }
        }
        if (maxErr >= 1e-4) {
          throw new Error(`RMSNORM_BATCH verification failed: maxAbsError=${maxErr}`);
        }
      }

      // 1. Verify batched Q4 GEMM (64x64 tiles) on (M=7, N=40, K=96) and (M=64, N=128, K=256)
      for (const [M, N, K] of [
        [7, 40, 96],
        [64, 128, 256],
      ] as const) {
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
        pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(M / 64), 1);
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
        if (maxErr >= 0.05) {
          throw new Error(`Q4_GEMM (${M}x${N}x${K}) verification failed: maxAbsError=${maxErr}`);
        }
      }

      // 2. Verify GDN_RECURRENT (M=2) AND Chunkwise WY (M=37 -> 32+5, non-zero initial state) vs TS CPU ref
      {
        const runGdnCpuRef = (
          M: number,
          workCpu: Float32Array,
          auxCpu: Float32Array,
          initStateHead0: Float32Array,
        ): { outputs: Float32Array[]; finalState: Float32Array } => {
          const stateRef = new Float32Array(initStateHead0);
          const outputs: Float32Array[] = [];
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
            const tokOut = new Float32Array(128);
            for (let j = 0; j < 128; j++) {
              const zv = workCpu[M * 8192 + t * 4096 + j];
              tokOut[j] = rawOut[j] * invRms * (zv / (1 + Math.exp(-zv))) * auxCpu[64 + j];
            }
            outputs.push(tokOut);
          }
          return { outputs, finalState: stateRef };
        };

        // 2a. Recurrent (M=2)
        {
          const M = 2;
          const workCpu = new Float32Array(M * 12352);
          for (let t = 0; t < M; t++) {
            for (let i = 0; i < 128; i++) {
              workCpu[t * 8192 + i] = Math.sin(t + i * 0.07) * 0.8;
              workCpu[t * 8192 + 2048 + i] = Math.cos(t * 0.5 + i * 0.05) * 0.7;
              workCpu[t * 8192 + 4096 + i] = Math.sin(t * 0.9 + i * 0.03) * 0.6;
              workCpu[M * 8192 + t * 4096 + i] = Math.cos(i * 0.04 - t) * 0.5;
            }
            workCpu[M * 12288 + t * 64 + 0] = -0.3 + t * 0.2;
            workCpu[M * 12288 + t * 64 + 32] = 0.4 - t * 0.3;
          }
          const auxCpu = new Float32Array(192);
          auxCpu[0] = -0.15;
          auxCpu[32] = 0.25;
          for (let i = 0; i < 128; i++) auxCpu[64 + i] = 0.85 + 0.1 * Math.sin(i);
          const { outputs } = runGdnCpuRef(M, workCpu, auxCpu, new Float32Array(128 * 128));
          const outRef = outputs[1];

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
            throw new Error(`GDN_RECURRENT verification failed: maxAbsError=${maxErr}`);
          }
        }

        // 2b. Chunkwise Parallel WY (M=37 -> 32 + 5) with non-zero initial dn_state
        {
          const M = 37;
          const workCpu = new Float32Array(M * 12352);
          for (let t = 0; t < M; t++) {
            for (let i = 0; i < 128; i++) {
              workCpu[t * 8192 + i] = Math.sin(t * 0.31 + i * 0.07) * 0.8;
              workCpu[t * 8192 + 2048 + i] = Math.cos(t * 0.23 + i * 0.05) * 0.7;
              workCpu[t * 8192 + 4096 + i] = Math.sin(t * 0.47 + i * 0.03) * 0.6;
              workCpu[M * 8192 + t * 4096 + i] = Math.cos(i * 0.04 - t * 0.19) * 0.5;
            }
            workCpu[M * 12288 + t * 64 + 0] = -0.3 + Math.sin(t * 0.3) * 0.4;
            workCpu[M * 12288 + t * 64 + 32] = 0.2 + Math.cos(t * 0.2) * 0.5;
          }
          const auxCpu = new Float32Array(192);
          auxCpu[0] = -0.15;
          auxCpu[32] = 0.25;
          for (let i = 0; i < 128; i++) auxCpu[64 + i] = 0.85 + 0.1 * Math.sin(i);

          const initStateHead0 = new Float32Array(128 * 128);
          for (let idx = 0; idx < initStateHead0.length; idx++) {
            initStateHead0[idx] = Math.sin(idx * 0.017) * 0.05;
          }
          const { outputs: refOutputs, finalState: refState } = runGdnCpuRef(
            M,
            workCpu,
            auxCpu,
            initStateHead0,
          );

          const auxG = mk(auxCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
          const workG = mk(workCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
          const scratchG = mk(558080 * 4, GPU_USAGE.STORAGE);
          const stG = mk(32 * 128 * 128 * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC);
          const outG = mk(M * 4096 * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
          const uG = mk(6 * 256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);

          device.queue.writeBuffer(auxG, 0, auxCpu);
          device.queue.writeBuffer(workG, 0, workCpu);
          device.queue.writeBuffer(stG, 0, initStateHead0);

          device.queue.writeBuffer(uG, 0, new Uint32Array([32, 0, 0, 0, M * 12288, 0, 0, 0]));
          device.queue.writeBuffer(uG, 256, new Uint32Array([0, 32, 0, 0, 0, M * 8192, 0, 0]));
          device.queue.writeBuffer(uG, 512, new Uint32Array([5, 32, 0, 0, M * 12288, 0, 0, 0]));
          device.queue.writeBuffer(uG, 768, new Uint32Array([0, 5, 32, 0, 0, M * 8192, 0, 0]));
          device.queue.writeBuffer(uG, 1024, new Uint32Array([1, M, 0, 0, 0, M * 8192, 0, 0]));

          const bgPrep = device.createBindGroup({
            layout: dynStorage4Bgl,
            entries: [
              { binding: 0, resource: { buffer: uG, size: 32 } },
              { binding: 1, resource: { buffer: auxG } },
              { binding: 2, resource: { buffer: workG } },
              { binding: 3, resource: { buffer: scratchG } },
            ],
          });
          const bgApply = device.createBindGroup({
            layout: gdnApplyBgl,
            entries: [
              { binding: 0, resource: { buffer: uG, size: 32 } },
              { binding: 1, resource: { buffer: auxG } },
              { binding: 2, resource: { buffer: workG } },
              { binding: 3, resource: { buffer: scratchG } },
              { binding: 4, resource: { buffer: stG } },
              { binding: 5, resource: { buffer: outG } },
            ],
          });

          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(this.gdnChunkPrepPipe);
          pass.setBindGroup(0, bgPrep, [0]);
          pass.dispatchWorkgroups(32, 1, 1);
          pass.setPipeline(this.gdnChunkApplyPipe);
          pass.setBindGroup(0, bgApply, [256]);
          pass.dispatchWorkgroups(32, 8, 1);
          pass.setPipeline(this.gdnChunkPrepPipe);
          pass.setBindGroup(0, bgPrep, [512]);
          pass.dispatchWorkgroups(32, 1, 1);
          pass.setPipeline(this.gdnChunkApplyPipe);
          pass.setBindGroup(0, bgApply, [768]);
          pass.dispatchWorkgroups(32, 8, 1);
          pass.setPipeline(this.gdnChunkApplyPipe);
          pass.setBindGroup(0, bgApply, [1024]);
          pass.dispatchWorkgroups(32, M, 1);
          pass.end();
          device.queue.submit([enc.finish()]);

          for (const tokCheck of [0, 31, 32, 36]) {
            const gpuOut = await readGpuF32Slice(device, outG, tokCheck * 4096 * 4, 128 * 4);
            let outErr = 0;
            for (let j = 0; j < 128; j++) {
              outErr = Math.max(outErr, Math.abs(gpuOut[j] - refOutputs[tokCheck][j]));
            }
            if (outErr >= 1e-4) {
              throw new Error(
                `GDN_CHUNK_WY output verification failed (tok=${tokCheck}): maxAbsError=${outErr}`,
              );
            }
          }
          const gpuState = await readGpuF32Slice(device, stG, 0, 128 * 128 * 4);
          let stErr = 0;
          for (let idx = 0; idx < 128 * 128; idx++) {
            stErr = Math.max(stErr, Math.abs(gpuState[idx] - refState[idx]));
          }
          if (stErr >= 1e-4) {
            throw new Error(`GDN_CHUNK_WY state verification failed: maxAbsError=${stErr}`);
          }
        }
      }

      // 3. Verify QK_NORM_ROPE + 8-Query GQA-Tiled GQA_PREFILL
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
            { binding: 3, resource: { buffer: auxG } },
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
              throw new Error(`QK_NORM_ROPE verification failed (tok=${tokCheck}, head=${qHead}): maxAbsError=${qErr}`);
            }
            const gpuAttn = await readGpuF32Slice(device, outG, (tokCheck * 4096 + qHead * 256) * 4, 256 * 4);
            let aErr = 0;
            for (let d = 0; d < 256; d++) aErr = Math.max(aErr, Math.abs(gpuAttn[d] - attnRef[d]));
            if (aErr > aTol) {
              throw new Error(`GQA_${this.kvMode}_PREFILL verification failed (tok=${tokCheck}, head=${qHead}): maxAbsError=${aErr}`);
            }
          }
        }
      }

      // 4. Verify 4-Q-Head Grouped GQA INT8 Decode (and optional DP4a) at seqPos=96, num_splits=8
      if (this.kvMode === 'int8') {
        const seqPos = 96;
        const maxSeq = 128;
        const numSplits = 8;
        const actCpu = new Float32Array(10240);
        for (let i = 0; i < actCpu.length; i++) {
          actCpu[i] = Math.sin(i * 0.029) * 0.45;
        }
        const kHist: Float32Array[] = [];
        const vHist: Float32Array[] = [];
        for (let t = 0; t <= seqPos; t++) {
          const kv = new Float32Array(256);
          const vv = new Float32Array(256);
          if (t < seqPos) {
            for (let d = 0; d < 256; d++) {
              kv[d] = Math.cos(t * 0.17 + d * 0.041) * 0.5;
              vv[d] = Math.sin(t * 0.23 - d * 0.037) * 0.5;
            }
          } else {
            kv.set(actCpu.subarray(8192, 8192 + 256));
            vv.set(actCpu.subarray(9216, 9216 + 256));
          }
          kHist.push(kv);
          vHist.push(vv);
        }

        const f32ToF16Bits = (v: number): number => {
          const f32 = new Float32Array([v]);
          const u32 = new Uint32Array(f32.buffer)[0];
          const sign = (u32 >>> 16) & 0x8000;
          const exp = ((u32 >>> 23) & 0xff) - 127 + 15;
          const frac = u32 & 0x7fffff;
          if (exp <= 0) return sign;
          if (exp >= 31) return sign | 0x7c00;
          return sign | (exp << 10) | (frac >>> 13);
        };

        const kvBytes = Number(wasm.full_attn_layer_kv_bytes(maxSeq, 1));
        const kvInit = new Uint32Array(kvBytes / 4);
        const headStride = maxSeq * 65;
        const vPlaneOff = 4 * headStride;
        for (let t = 0; t < seqPos; t++) {
          let km = 0;
          let vm = 0;
          for (let d = 0; d < 256; d++) {
            km = Math.max(km, Math.abs(kHist[t][d]));
            vm = Math.max(vm, Math.abs(vHist[t][d]));
          }
          const ks = Math.max(km / 127, 1e-6);
          const vs = Math.max(vm / 127, 1e-6);
          const kBase = t * 65;
          const vBase = vPlaneOff + t * 65;
          for (let i = 0; i < 64; i++) {
            const pk =
              (Math.max(-127, Math.min(127, Math.round(kHist[t][i * 4] / ks))) & 0xff) |
              ((Math.max(-127, Math.min(127, Math.round(kHist[t][i * 4 + 1] / ks))) & 0xff) << 8) |
              ((Math.max(-127, Math.min(127, Math.round(kHist[t][i * 4 + 2] / ks))) & 0xff) << 16) |
              ((Math.max(-127, Math.min(127, Math.round(kHist[t][i * 4 + 3] / ks))) & 0xff) << 24);
            const pv =
              (Math.max(-127, Math.min(127, Math.round(vHist[t][i * 4] / vs))) & 0xff) |
              ((Math.max(-127, Math.min(127, Math.round(vHist[t][i * 4 + 1] / vs))) & 0xff) << 8) |
              ((Math.max(-127, Math.min(127, Math.round(vHist[t][i * 4 + 2] / vs))) & 0xff) << 16) |
              ((Math.max(-127, Math.min(127, Math.round(vHist[t][i * 4 + 3] / vs))) & 0xff) << 24);
            kvInit[kBase + i] = pk >>> 0;
            kvInit[vBase + i] = pv >>> 0;
          }
          kvInit[kBase + 64] = (f32ToF16Bits(ks) | (f32ToF16Bits(vs) << 16)) >>> 0;
        }

        const refHeads: Float32Array[] = [];
        for (let qHead = 0; qHead < 4; qHead++) {
          const qBase = qHead * 512;
          const gateOff = qBase + 256;
          const scores = new Float32Array(seqPos + 1);
          let maxS = -Infinity;
          for (let t = 0; t <= seqPos; t++) {
            let dot = 0;
            for (let d = 0; d < 256; d++) dot += actCpu[qBase + d] * kHist[t][d];
            scores[t] = dot / 16;
            if (scores[t] > maxS) maxS = scores[t];
          }
          let lSum = 0;
          const probs = new Float32Array(seqPos + 1);
          for (let t = 0; t <= seqPos; t++) {
            probs[t] = Math.exp(scores[t] - maxS);
            lSum += probs[t];
          }
          const outH = new Float32Array(256);
          for (let d = 0; d < 256; d++) {
            let acc = 0;
            for (let t = 0; t <= seqPos; t++) acc += (probs[t] / lSum) * vHist[t][d];
            outH[d] = acc / (1 + Math.exp(-actCpu[gateOff + d]));
          }
          refHeads.push(outH);
        }

        const actG = mk(actCpu.byteLength, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const kvG = mk(kvBytes, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const splitG = mk(16 * numSplits * 260 * 4, GPU_USAGE.STORAGE);
        const outG = mk(4096 * 4, GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC);
        const posG = mk(16, GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST);
        const uG = mk(2 * 256, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);

        device.queue.writeBuffer(actG, 0, actCpu);
        device.queue.writeBuffer(posG, 0, new Uint32Array([0, seqPos, 0, 0]));
        device.queue.writeBuffer(uG, 0, new Uint32Array([0, maxSeq, 0xffffffff, 0, 8192, 9216, numSplits, 512]));
        device.queue.writeBuffer(uG, 256, new Uint32Array([numSplits, 256, 0, 512, 0, 0, 0, 0]));

        const bgSplit = device.createBindGroup({
          layout: gqaSplitBgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: actG } },
            { binding: 2, resource: { buffer: kvG } },
            { binding: 3, resource: { buffer: splitG } },
            { binding: 4, resource: { buffer: posG } },
          ],
        });
        const bgRed = device.createBindGroup({
          layout: dynStorage4Bgl,
          entries: [
            { binding: 0, resource: { buffer: uG, size: 32 } },
            { binding: 1, resource: { buffer: actG } },
            { binding: 2, resource: { buffer: splitG } },
            { binding: 3, resource: { buffer: outG } },
          ],
        });

        const gqaPipesToVerify: Array<{ name: string; pipe: unknown }> = [
          { name: 'GQA_INT8_SPLIT', pipe: this.gqaSplitPlainPipe },
        ];
        if (this.usingDp4a) {
          gqaPipesToVerify.push({ name: 'GQA_INT8_SPLIT_DP4A', pipe: this.gqaSplitPipe });
        }

        for (const entry of gqaPipesToVerify) {
          device.queue.writeBuffer(kvG, 0, kvInit);
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(entry.pipe);
          pass.setBindGroup(0, bgSplit, [0]);
          pass.dispatchWorkgroups(4, numSplits, 1);
          pass.setPipeline(this.gqaReducePipe);
          pass.setBindGroup(0, bgRed, [256]);
          pass.dispatchWorkgroups(16, 1, 1);
          pass.end();
          device.queue.submit([enc.finish()]);

          for (let qHead = 0; qHead < 4; qHead++) {
            const gpuOut = await readGpuF32Slice(device, outG, qHead * 256 * 4, 256 * 4);
            let maxErr = 0;
            for (let d = 0; d < 256; d++) {
              maxErr = Math.max(maxErr, Math.abs(gpuOut[d] - refHeads[qHead][d]));
            }
            if (maxErr >= 0.05) {
              throw new Error(
                `${entry.name} verification failed (seqPos=${seqPos}, numSplits=${numSplits}, head=${qHead}): maxAbsError=${maxErr}`,
              );
            }
          }
        }
      }
    } finally {
      for (const b of tempBufs) b.destroy();
    }
  }

  dispose(): void {
    this.invalidateVerifyPlans();
    this.clearGpuPrefixSlots();
    for (const buf of this.allocatedBuffers) {
      try {
        buf.destroy();
      } catch {
        // Ignore already destroyed buffers
      }
    }
    this.allocatedBuffers = [];
    this.wSlabs = [];
    this.lmHeadSlabs = [];
    this.weightPartitions = [];
    this.vocabSlices = [];
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
    this.emaDecodeStepMs = 0;
    this.decodeBurstMs = 0;
    this.usingDp4a = false;
  }
}
