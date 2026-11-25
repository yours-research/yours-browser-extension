import type {
  WgslBenchConfig,
  WgslBenchPoint,
  WgslBenchResult,
  WgslGemvVariant,
  WgslKernelCorrectnessEntry,
  WgslOpfsLoadResult,
  WgslOpfsStageTiming,
} from '../shared/protocol.ts';
import { computeElevatedLimits } from './webgpuLimits.ts';
import { loadWgslWasm } from './wgslWebgpu.ts';

// Avoid needing @webgpu/types by declaring minimal WebGPU DOM interfaces used here.
interface GpuBufferLike {
  size: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GpuComputePassLike {
  setPipeline(pipeline: unknown): void;
  setBindGroup(index: number, bindGroup: unknown, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GpuCommandEncoderLike {
  beginComputePass(): GpuComputePassLike;
  copyBufferToBuffer(
    src: GpuBufferLike,
    srcOffset: number,
    dst: GpuBufferLike,
    dstOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

interface GpuQueueLike {
  writeBuffer(buffer: GpuBufferLike, bufferOffset: number, data: BufferSource | SharedArrayBuffer): void;
  submit(commandBuffers: unknown[]): void;
  onSubmittedWorkDone(): Promise<void>;
}

interface GpuShaderCompilationMessage {
  type: 'error' | 'warning' | 'info' | string;
  lineNum: number;
  linePos: number;
  message: string;
}

interface GpuShaderModuleLike {
  getCompilationInfo?(): Promise<{ messages: GpuShaderCompilationMessage[] }>;
}

interface GpuDeviceLike {
  limits: Record<string, number>;
  features: Set<string>;
  queue: GpuQueueLike;
  createBuffer(desc: { size: number; usage: number; mappedAtCreation?: boolean }): GpuBufferLike;
  createShaderModule(desc: { label?: string; code: string }): GpuShaderModuleLike;
  createBindGroupLayout(desc: unknown): unknown;
  createPipelineLayout(desc: { bindGroupLayouts: unknown[] }): unknown;
  createComputePipelineAsync(desc: {
    label?: string;
    layout: unknown;
    compute: { module: GpuShaderModuleLike; entryPoint: string };
  }): Promise<unknown>;
  createBindGroup(desc: unknown): unknown;
  createCommandEncoder(): GpuCommandEncoderLike;
  pushErrorScope?(filter: 'validation' | 'out-of-memory' | 'internal'): void;
  popErrorScope?(): Promise<{ message: string } | null>;
  destroy(): void;
}

interface GpuAdapterLike {
  limits: Record<string, number>;
  features: Iterable<string>;
  requestDevice(desc?: { requiredFeatures?: string[]; requiredLimits?: Record<string, number> }): Promise<GpuDeviceLike>;
}

const GPU_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

const GPU_MAP_READ = 0x0001;
const GPU_SHADER_COMPUTE = 0x0004;

// Architectural constants matching crates/tinygpt-wgsl-engine/src/lib.rs
const HIDDEN = 2560;
const INTERMEDIATE = 9216;
const NUM_LAYERS = 32;
const VOCAB_SIZE = 248320;
const FULL_ATTN_LAYERS = new Set([3, 7, 11, 15, 19, 23, 27, 31]);

const WEIGHT_SUBDIR = 'tinygpt-wgsl-weights';

interface WasmExports {
  memory: WebAssembly.Memory;
  wgsl_get_shader_ptr(): number;
  wgsl_get_shader_len(): number;
  wgsl_build_shader(shaderId: number): number;
  wasm_scratch_ptr(): number;
  wasm_scratch_len(): number;
  q4_row_u32s(k: number): number;
  q4_matrix_bytes(rows: number, cols: number): bigint;
  deltanet_layer_state_bytes(): number;
  full_attn_layer_kv_bytes(maxSeq: number, int8Kv: number): bigint;
  cpu_ref_q4_gemv(
    rows: number,
    cols: number,
    wPtr: number,
    xPtr: number,
    outPtr: number,
  ): void;
  cpu_ref_deltanet_head_step(
    qPtr: number,
    kPtr: number,
    vPtr: number,
    zPtr: number,
    gateA: number,
    gateB: number,
    convPtr: number,
    statePtr: number,
    outPtr: number,
  ): void;
}

function getShaderSource(wasm: WasmExports, shaderId: number): string {
  const len = wasm.wgsl_build_shader(shaderId);
  const ptr = wasm.wgsl_get_shader_ptr();
  const view = new Uint8Array(wasm.memory.buffer, ptr, len);
  return new TextDecoder().decode(view);
}

async function createValidatedPipeline(
  device: GpuDeviceLike,
  label: string,
  code: string,
  layout: unknown,
): Promise<unknown> {
  const mod = device.createShaderModule({ label, code });
  if (typeof mod.getCompilationInfo === 'function') {
    const info = await mod.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      const detail = errors
        .map((m) => `line ${m.lineNum}:${m.linePos}: ${m.message}`)
        .join(' | ');
      throw new Error(`WGSL shader [${label}] failed compilation: ${detail}`);
    }
  }
  device.pushErrorScope?.('validation');
  try {
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout,
      compute: { module: mod, entryPoint: 'main' },
    });
    const scopeErr = await device.popErrorScope?.();
    if (scopeErr) {
      throw new Error(`Pipeline [${label}] validation failed: ${scopeErr.message}`);
    }
    return pipeline;
  } catch (err) {
    const scopeErr = await device.popErrorScope?.().catch(() => null);
    if (scopeErr) {
      throw new Error(`Pipeline [${label}] validation error: ${scopeErr.message}`);
    }
    throw err;
  }
}

async function acquireWgslDevice(): Promise<{
  device: GpuDeviceLike;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  maxBufferSize: number;
}> {
  const nav = navigator as unknown as {
    gpu?: { requestAdapter(opts?: { powerPreference?: string }): Promise<GpuAdapterLike | null> };
  };
  if (!nav.gpu) {
    throw new Error('WebGPU not supported in Offscreen Document.');
  }
  const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter returned.');
  }
  const features = new Set(Array.from(adapter.features));
  const requiredFeatures: string[] = [];
  if (features.has('shader-f16')) requiredFeatures.push('shader-f16');
  if (features.has('subgroups')) requiredFeatures.push('subgroups');

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: computeElevatedLimits(adapter.limits),
  });

  return {
    device,
    hasShaderF16: features.has('shader-f16'),
    hasSubgroups: features.has('subgroups'),
    maxBufferSize: device.limits.maxBufferSize ?? 0,
  };
}

function f32ToF16Bits(val: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = (x >>> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

function packF16Pair(a: number, b: number): number {
  return (f32ToF16Bits(a) & 0xffff) | ((f32ToF16Bits(b) & 0xffff) << 16);
}

function fillQ4WeightPattern(u32Out: Uint32Array, scale: number, zero: number): void {
  const metaWord = packF16Pair(scale, zero);
  const blocks = Math.floor(u32Out.length / 5);
  for (let b = 0; b < blocks; b++) {
    const base = b * 5;
    // Alternating small nibbles centered near 8 so zero = -8*scale yields balanced weights
    u32Out[base] = 0x7986a597;
    u32Out[base + 1] = 0x6a8597a6;
    u32Out[base + 2] = 0x976a8598;
    u32Out[base + 3] = 0x8597a679;
    u32Out[base + 4] = metaWord;
  }
}

function writeBufferChunkedQ4(
  device: GpuDeviceLike,
  buffer: GpuBufferLike,
  totalBytes: number,
): void {
  const chunkBytes = 16 * 1024 * 1024;
  // Ensure chunk is a multiple of 20 bytes (1 Q4 block = 5 u32s)
  const alignedChunkBytes = Math.floor(chunkBytes / 20) * 20;
  const staging = new Uint32Array(alignedChunkBytes / 4);
  fillQ4WeightPattern(staging, 0.0025, -0.02);

  let offset = 0;
  while (offset < totalBytes) {
    const remaining = totalBytes - offset;
    const writeLen = Math.min(alignedChunkBytes, remaining);
    if (writeLen === alignedChunkBytes) {
      device.queue.writeBuffer(buffer, offset, staging);
    } else {
      device.queue.writeBuffer(buffer, offset, staging.subarray(0, writeLen / 4));
    }
    offset += writeLen;
  }
}

async function readGpuF32Slice(
  device: GpuDeviceLike,
  src: GpuBufferLike,
  byteOffset: number,
  byteLength: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, byteOffset, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPU_MAP_READ);
  const copy = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return copy;
}

async function readGpuU32Scalar(device: GpuDeviceLike, src: GpuBufferLike): Promise<number> {
  const staging = device.createBuffer({
    size: 4,
    usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPU_MAP_READ);
  const val = new Uint32Array(staging.getMappedRange())[0];
  staging.unmap();
  staging.destroy();
  return val;
}

interface DispatchStep {
  pipeline: unknown;
  bindGroup: unknown;
  uniformByteOffset: number;
  wx: number;
  wy: number;
  wz: number;
}

export class WgslEngineRunner {
  private readonly assetBaseUrl?: string;

  constructor(assetBaseUrl?: string) {
    this.assetBaseUrl = assetBaseUrl;
  }

  async runOpfsLoadBench(
    action: 'prepare' | 'load' | 'clear',
    totalMb = 2560,
    chunkMibList = [16, 32, 64],
  ): Promise<WgslOpfsLoadResult> {
    const root = await navigator.storage.getDirectory();

    if (action === 'clear') {
      try {
        await root.removeEntry(WEIGHT_SUBDIR, { recursive: true });
      } catch {
        // Ignore if absent
      }
      return {
        action: 'clear',
        ok: true,
        totalBytes: 0,
        timings: [],
        details: 'Cleared Track 2 WGSL flat weight slabs from OPFS.',
      };
    }

    const totalBytes = Math.round(totalMb) * 1024 * 1024;
    const slab0Bytes = Math.floor(totalBytes / 2);
    const slab1Bytes = totalBytes - slab0Bytes;

    if (action === 'prepare') {
      const start = performance.now();
      try {
        await root.removeEntry(WEIGHT_SUBDIR, { recursive: true });
      } catch {
        // Ignore
      }
      const dir = await root.getDirectoryHandle(WEIGHT_SUBDIR, { create: true });
      const writeChunk = new Uint8Array(8 * 1024 * 1024);
      writeChunk.fill(0x5a);

      for (const [name, size] of [
        ['wgsl-slab-0.bin', slab0Bytes],
        ['wgsl-slab-1.bin', slab1Bytes],
      ] as const) {
        const fh = await dir.getFileHandle(name, { create: true });
        const writable = await fh.createWritable();
        let written = 0;
        while (written < size) {
          const len = Math.min(writeChunk.byteLength, size - written);
          await writable.write(len === writeChunk.byteLength ? writeChunk : writeChunk.subarray(0, len));
          written += len;
        }
        await writable.close();
      }
      const elapsed = performance.now() - start;
      return {
        action: 'prepare',
        ok: true,
        totalBytes,
        timings: [
          {
            stagingChunkMib: 8,
            durationMs: elapsed,
            throughputMBs: totalBytes / (1024 * 1024) / (elapsed / 1000),
            totalBytes,
          },
        ],
        details: `Prepared ${(totalBytes / (1024 * 1024 * 1024)).toFixed(2)} GiB WGSL binary slabs in ${(elapsed / 1000).toFixed(2)}s.`,
      };
    }

    // action === 'load': Stream directly from OPFS File.slice() into WebGPU GPUBuffers
    const dir = await root.getDirectoryHandle(WEIGHT_SUBDIR, { create: false });
    const f0 = await (await dir.getFileHandle('wgsl-slab-0.bin')).getFile();
    const f1 = await (await dir.getFileHandle('wgsl-slab-1.bin')).getFile();
    const actualTotal = f0.size + f1.size;

    const { device } = await acquireWgslDevice();
    const timings: WgslOpfsStageTiming[] = [];

    try {
      for (const chunkMib of chunkMibList) {
        const chunkBytes = chunkMib * 1024 * 1024;
        const buf0 = device.createBuffer({
          size: f0.size,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
        });
        const buf1 = device.createBuffer({
          size: f1.size,
          usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
        });

        const t0 = performance.now();
        for (const [file, gpuBuf] of [
          [f0, buf0],
          [f1, buf1],
        ] as const) {
          let offset = 0;
          while (offset < file.size) {
            const end = Math.min(offset + chunkBytes, file.size);
            const sliceBuf = await file.slice(offset, end).arrayBuffer();
            device.queue.writeBuffer(gpuBuf, offset, sliceBuf);
            offset = end;
          }
        }
        await device.queue.onSubmittedWorkDone();
        const durationMs = Math.max(0.1, performance.now() - t0);
        timings.push({
          stagingChunkMib: chunkMib,
          durationMs,
          throughputMBs: actualTotal / (1024 * 1024) / (durationMs / 1000),
          totalBytes: actualTotal,
        });

        buf0.destroy();
        buf1.destroy();
      }
    } finally {
      device.destroy();
    }

    return {
      action: 'load',
      ok: true,
      totalBytes: actualTotal,
      timings,
      details: `Streamed ${(actualTotal / (1024 * 1024 * 1024)).toFixed(2)} GiB OPFS -> GPUQueue.writeBuffer across chunk sizes [${chunkMibList.join(', ')}] MiB.`,
    };
  }

  private async verifyKernels(
    device: GpuDeviceLike,
    wasm: WasmExports,
    hasShaderF16: boolean,
    hasSubgroups: boolean,
  ): Promise<WgslKernelCorrectnessEntry[]> {
    const entries: WgslKernelCorrectnessEntry[] = [];

    // Helper: create dynamic uniform BGL + 3 storage buffers
    const gemvBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const gemvLayout = device.createPipelineLayout({ bindGroupLayouts: [gemvBgl] });

    // 1. Verify Q4 GEMV variants against Rust WASM CPU reference
    const rows = 64;
    const cols = 256;
    const rowU32 = wasm.q4_row_u32s(cols);
    const wU32Len = rows * rowU32;
    const scratchBase = wasm.wasm_scratch_ptr();
    const wPtr = scratchBase;
    const xPtr = wPtr + wU32Len * 4;
    const refOutPtr = xPtr + cols * 4;

    const wCpu = new Uint32Array(wasm.memory.buffer, wPtr, wU32Len);
    for (let b = 0; b < rows * (cols / 32); b++) {
      const idx = b * 5;
      wCpu[idx] = (0x12345678 + b * 17) >>> 0;
      wCpu[idx + 1] = (0x87654321 ^ (b * 31)) >>> 0;
      wCpu[idx + 2] = (0x0f1e2d3c + b * 7) >>> 0;
      wCpu[idx + 3] = (0x5a6b7c8d ^ (b * 13)) >>> 0;
      wCpu[idx + 4] = packF16Pair(0.015, -0.11);
    }
    const xCpu = new Float32Array(wasm.memory.buffer, xPtr, cols);
    for (let i = 0; i < cols; i++) {
      xCpu[i] = Math.sin(i * 0.13) * 0.75;
    }

    wasm.cpu_ref_q4_gemv(rows, cols, wPtr, xPtr, refOutPtr);
    const refGemv = new Float32Array(wasm.memory.buffer, refOutPtr, rows).slice();

    const wGpu = device.createBuffer({
      size: wU32Len * 4,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
    });
    const xGpu = device.createBuffer({
      size: cols * 4,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
    });
    const yGpu = device.createBuffer({
      size: rows * 4,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC,
    });
    const uGpu = device.createBuffer({
      size: 32,
      usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST,
    });
    device.queue.writeBuffer(wGpu, 0, new Uint32Array(wasm.memory.buffer, wPtr, wU32Len));
    device.queue.writeBuffer(xGpu, 0, new Float32Array(wasm.memory.buffer, xPtr, cols));
    device.queue.writeBuffer(uGpu, 0, new Uint32Array([rows, cols, 0, 0, 0, 0, 0, 0]));

    const gemvBg = device.createBindGroup({
      layout: gemvBgl,
      entries: [
        { binding: 0, resource: { buffer: uGpu } },
        { binding: 1, resource: { buffer: wGpu } },
        { binding: 2, resource: { buffer: xGpu } },
        { binding: 3, resource: { buffer: yGpu } },
      ],
    });

    const gemvTests: Array<{ name: string; shaderId: number; wx: number; ok: boolean; tol: number }> = [
      { name: 'q4-scalar', shaderId: 0, wx: Math.ceil(rows / 64), ok: true, tol: 1e-3 },
      { name: 'q4-vec4-f16', shaderId: 1, wx: Math.ceil(rows / 64), ok: hasShaderF16, tol: 2e-2 },
      {
        name: 'q4-subgroups-f16',
        shaderId: 2,
        wx: Math.ceil(rows / 4),
        ok: hasShaderF16 && hasSubgroups,
        tol: 2e-2,
      },
    ];

    for (const gt of gemvTests) {
      if (!gt.ok) {
        entries.push({
          kernel: gt.name,
          passed: false,
          maxAbsError: -1,
          details: 'Skipped (GPU missing required shader-f16 or subgroups feature).',
        });
        continue;
      }
      const pipe = await createValidatedPipeline(
        device,
        `verify-${gt.name}`,
        getShaderSource(wasm, gt.shaderId),
        gemvLayout,
      );
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, gemvBg);
      pass.dispatchWorkgroups(gt.wx, 1, 1);
      pass.end();
      device.queue.submit([enc.finish()]);
      const gpuOut = await readGpuF32Slice(device, yGpu, 0, rows * 4);
      let maxErr = 0;
      for (let i = 0; i < rows; i++) {
        maxErr = Math.max(maxErr, Math.abs(gpuOut[i] - refGemv[i]));
      }
      entries.push({
        kernel: gt.name,
        passed: maxErr <= gt.tol,
        maxAbsError: maxErr,
        details: `maxAbsError=${maxErr.toExponential(2)} (tol=${gt.tol})`,
      });
    }

    wGpu.destroy();
    xGpu.destroy();
    yGpu.destroy();
    uGpu.destroy();

    // 2. Verify DeltaNet recurrent kernel vs Rust WASM CPU reference
    {
      const actCpu = new Float32Array(16500);
      for (let i = 0; i < 128; i++) {
        actCpu[i] = Math.sin(i * 0.05) * 0.4; // q
        actCpu[2048 + i] = Math.cos(i * 0.07) * 0.4; // k
        actCpu[4096 + i] = Math.sin(i * 0.11) * 0.5; // v
        actCpu[8192 + i] = Math.cos(i * 0.09) * 0.5; // z
      }
      actCpu[12288] = 0.25; // gate a for head 0
      actCpu[12320] = -0.35; // gate b for head 0

      // Run 2 steps in Rust WASM for head 0
      const pBase = wasm.wasm_scratch_ptr();
      const qP = pBase;
      const kP = qP + 128 * 4;
      const vP = kP + 128 * 4;
      const zP = vP + 128 * 4;
      const convP = zP + 128 * 4;
      const stateP = convP + 128 * 4 * 4;
      const outP = stateP + 128 * 128 * 4;

      new Float32Array(wasm.memory.buffer, qP, 128).set(actCpu.subarray(0, 128));
      new Float32Array(wasm.memory.buffer, kP, 128).set(actCpu.subarray(2048, 2048 + 128));
      new Float32Array(wasm.memory.buffer, vP, 128).set(actCpu.subarray(4096, 4096 + 128));
      new Float32Array(wasm.memory.buffer, zP, 128).set(actCpu.subarray(8192, 8192 + 128));
      new Float32Array(wasm.memory.buffer, convP, 128 * 4).fill(0);
      new Float32Array(wasm.memory.buffer, stateP, 128 * 128).fill(0);

      wasm.cpu_ref_deltanet_head_step(qP, kP, vP, zP, 0.25, -0.35, convP, stateP, outP);
      wasm.cpu_ref_deltanet_head_step(qP, kP, vP, zP, 0.25, -0.35, convP, stateP, outP);
      const refDn = new Float32Array(wasm.memory.buffer, outP, 128).slice();

      const dnBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const dnPipe = await createValidatedPipeline(
        device,
        'verify-deltanet_step',
        getShaderSource(wasm, 3),
        device.createPipelineLayout({ bindGroupLayouts: [dnBgl] }),
      );
      const stateBytes = wasm.deltanet_layer_state_bytes();
      const actBuf = device.createBuffer({ size: actCpu.byteLength, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST });
      const stBuf = device.createBuffer({ size: stateBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST });
      const outBuf = device.createBuffer({ size: 4096 * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC });
      const uBuf = device.createBuffer({ size: 32, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST });
      device.queue.writeBuffer(actBuf, 0, actCpu);
      // State starts at f32 offset 8192*4 = 32768; conv starts at 0
      device.queue.writeBuffer(uBuf, 0, new Uint32Array([8192 * 4, 0, 0, 2048, 4096, 8192, 12288, 0]));
      const dnBg = device.createBindGroup({
        layout: dnBgl,
        entries: [
          { binding: 0, resource: { buffer: uBuf } },
          { binding: 1, resource: { buffer: actBuf } },
          { binding: 2, resource: { buffer: stBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(dnPipe);
      pass.setBindGroup(0, dnBg);
      pass.dispatchWorkgroups(32, 1, 1);
      pass.dispatchWorkgroups(32, 1, 1);
      pass.end();
      device.queue.submit([enc.finish()]);
      const gpuDn = await readGpuF32Slice(device, outBuf, 0, 128 * 4);
      let maxErr = 0;
      for (let i = 0; i < 128; i++) {
        maxErr = Math.max(maxErr, Math.abs(gpuDn[i] - refDn[i]));
      }
      entries.push({
        kernel: 'deltanet_step',
        passed: maxErr < 1e-3,
        maxAbsError: maxErr,
        details: `2-step recurrent maxAbsError=${maxErr.toExponential(2)}`,
      });
      actBuf.destroy();
      stBuf.destroy();
      outBuf.destroy();
      uBuf.destroy();
    }

    // 3. Verify Split-K FlashDecoding GQA FP16 + INT8 KV Attention kernels against exact TS reference
    for (const int8Mode of [false, true]) {
      const kName = int8Mode ? 'gqa_int8_step' : 'gqa_fp16_step';
      if (!hasSubgroups || (!int8Mode && !hasShaderF16)) {
        entries.push({
          kernel: kName,
          passed: false,
          maxAbsError: -1,
          details: 'Skipped (GPU missing required shader-f16 or subgroups feature).',
        });
        continue;
      }
      const seqLen = 40;
      const maxSeq = 48;
      const numSplits = 8;
      const actCpu = new Float32Array(12288);
      const qSteps: Float32Array[] = [];
      const kSteps: Float32Array[] = [];
      const vSteps: Float32Array[] = [];
      const gSteps: Float32Array[] = [];
      for (let t = 0; t < seqLen; t++) {
        const q = new Float32Array(4096);
        const k = new Float32Array(1024);
        const v = new Float32Array(1024);
        const g = new Float32Array(4096);
        for (let i = 0; i < 4096; i++) {
          q[i] = Math.sin((t + 1) * 0.2 + i * 0.01) * 0.5;
          g[i] = Math.cos(i * 0.03) * 0.4;
        }
        for (let i = 0; i < 1024; i++) {
          k[i] = Math.cos((t + 1) * 0.17 + i * 0.02) * 0.5;
          v[i] = Math.sin((t + 1) * 0.13 + i * 0.015) * 0.6;
        }
        qSteps.push(q);
        kSteps.push(k);
        vSteps.push(v);
        gSteps.push(g);
      }

      // Compute exact attention output at last step (t = seqLen - 1, head 0)
      const qLast = qSteps[seqLen - 1];
      const gLast = gSteps[seqLen - 1];
      const scores = new Float32Array(seqLen);
      let maxS = -Infinity;
      for (let t = 0; t < seqLen; t++) {
        let dot = 0;
        for (let d = 0; d < 256; d++) dot += qLast[d] * kSteps[t][d];
        scores[t] = dot / Math.sqrt(256);
        if (scores[t] > maxS) maxS = scores[t];
      }
      let lSum = 0;
      const probs = new Float32Array(seqLen);
      for (let t = 0; t < seqLen; t++) {
        probs[t] = Math.exp(scores[t] - maxS);
        lSum += probs[t];
      }
      const refOut = new Float32Array(256);
      for (let d = 0; d < 256; d++) {
        let acc = 0;
        for (let t = 0; t < seqLen; t++) acc += (probs[t] / lSum) * vSteps[t][d];
        refOut[d] = acc * (1 / (1 + Math.exp(-gLast[d])));
      }

      const splitBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const reduceBgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const gqaSplitPipe = await createValidatedPipeline(
        device,
        `verify-${kName}_split`,
        getShaderSource(wasm, int8Mode ? 5 : 4),
        device.createPipelineLayout({ bindGroupLayouts: [splitBgl] }),
      );
      const gqaReducePipe = await createValidatedPipeline(
        device,
        `verify-${kName}_reduce`,
        getShaderSource(wasm, 9),
        device.createPipelineLayout({ bindGroupLayouts: [reduceBgl] }),
      );

      const kvBytes = Number(wasm.full_attn_layer_kv_bytes(maxSeq, int8Mode ? 1 : 0));
      const actBuf = device.createBuffer({ size: actCpu.byteLength, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST });
      const kvBuf = device.createBuffer({ size: Math.max(kvBytes, 4096), usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST });
      const splitBuf = device.createBuffer({
        size: 16 * numSplits * 260 * 4,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
      });
      const outBuf = device.createBuffer({ size: 4096 * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC });
      const uSplitBuf = device.createBuffer({ size: 32, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST });
      const uReduceBuf = device.createBuffer({ size: 32, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST });
      device.queue.writeBuffer(uReduceBuf, 0, new Uint32Array([numSplits, 6144, 0, 0, 0, 0, 0, 0]));

      const splitBg = device.createBindGroup({
        layout: splitBgl,
        entries: [
          { binding: 0, resource: { buffer: uSplitBuf } },
          { binding: 1, resource: { buffer: actBuf } },
          { binding: 2, resource: { buffer: kvBuf } },
          { binding: 3, resource: { buffer: splitBuf } },
        ],
      });
      const reduceBg = device.createBindGroup({
        layout: reduceBgl,
        entries: [
          { binding: 0, resource: { buffer: uReduceBuf } },
          { binding: 1, resource: { buffer: actBuf } },
          { binding: 2, resource: { buffer: splitBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });

      for (let t = 0; t < seqLen; t++) {
        actCpu.set(qSteps[t], 0);
        actCpu.set(kSteps[t], 4096);
        actCpu.set(vSteps[t], 5120);
        actCpu.set(gSteps[t], 6144);
        device.queue.writeBuffer(actBuf, 0, actCpu);
        device.queue.writeBuffer(uSplitBuf, 0, new Uint32Array([0, maxSeq, t, 0, 4096, 5120, numSplits, 0]));
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(gqaSplitPipe);
        pass.setBindGroup(0, splitBg);
        pass.dispatchWorkgroups(16, numSplits, 1);
        if (t === seqLen - 1) {
          pass.setPipeline(gqaReducePipe);
          pass.setBindGroup(0, reduceBg);
          pass.dispatchWorkgroups(16, 1, 1);
        }
        pass.end();
        device.queue.submit([enc.finish()]);
      }

      const gpuOut = await readGpuF32Slice(device, outBuf, 0, 256 * 4);
      let maxErr = 0;
      for (let d = 0; d < 256; d++) {
        maxErr = Math.max(maxErr, Math.abs(gpuOut[d] - refOut[d]));
      }
      const tol = int8Mode ? 0.05 : 0.01;
      entries.push({
        kernel: kName,
        passed: maxErr < tol,
        maxAbsError: maxErr,
        details: `${seqLen}-step Split-${numSplits} GQA maxAbsError=${maxErr.toExponential(2)} (tol=${tol})`,
      });
      actBuf.destroy();
      kvBuf.destroy();
      splitBuf.destroy();
      outBuf.destroy();
      uSplitBuf.destroy();
      uReduceBuf.destroy();
    }

    // 4. Verify 2-pass GPU Argmax over 248,320 logits
    {
      const logitsCpu = new Float32Array(VOCAB_SIZE);
      for (let i = 0; i < VOCAB_SIZE; i++) {
        logitsCpu[i] = Math.sin(i * 0.013) * 2.0;
      }
      const expectedIdx = 173421;
      logitsCpu[expectedIdx] = 42.5;

      const wgCount = Math.ceil(VOCAB_SIZE / 256);
      const bgl1 = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const bgl2 = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPU_SHADER_COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const p1 = await createValidatedPipeline(
        device,
        'verify-argmax_pass1',
        getShaderSource(wasm, 7),
        device.createPipelineLayout({ bindGroupLayouts: [bgl1] }),
      );
      const p2 = await createValidatedPipeline(
        device,
        'verify-argmax_pass2',
        getShaderSource(wasm, 8),
        device.createPipelineLayout({ bindGroupLayouts: [bgl2] }),
      );

      const logitsBuf = device.createBuffer({ size: VOCAB_SIZE * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST });
      const partVal = device.createBuffer({ size: wgCount * 4, usage: GPU_USAGE.STORAGE });
      const partIdx = device.createBuffer({ size: wgCount * 4, usage: GPU_USAGE.STORAGE });
      const tokOut = device.createBuffer({ size: 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC });
      const u1 = device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST });
      const u2 = device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST });
      device.queue.writeBuffer(logitsBuf, 0, logitsCpu);
      device.queue.writeBuffer(u1, 0, new Uint32Array([VOCAB_SIZE, 0, 0, 0]));
      device.queue.writeBuffer(u2, 0, new Uint32Array([wgCount, 0, 0, 0]));

      const bg1 = device.createBindGroup({
        layout: bgl1,
        entries: [
          { binding: 0, resource: { buffer: u1 } },
          { binding: 1, resource: { buffer: logitsBuf } },
          { binding: 2, resource: { buffer: partVal } },
          { binding: 3, resource: { buffer: partIdx } },
        ],
      });
      const bg2 = device.createBindGroup({
        layout: bgl2,
        entries: [
          { binding: 0, resource: { buffer: u2 } },
          { binding: 1, resource: { buffer: partVal } },
          { binding: 2, resource: { buffer: partIdx } },
          { binding: 3, resource: { buffer: tokOut } },
        ],
      });

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(p1);
      pass.setBindGroup(0, bg1);
      pass.dispatchWorkgroups(wgCount, 1, 1);
      pass.setPipeline(p2);
      pass.setBindGroup(0, bg2);
      pass.dispatchWorkgroups(1, 1, 1);
      pass.end();
      device.queue.submit([enc.finish()]);
      const gotIdx = await readGpuU32Scalar(device, tokOut);
      entries.push({
        kernel: 'argmax_vocab_248k',
        passed: gotIdx === expectedIdx,
        maxAbsError: Math.abs(gotIdx - expectedIdx),
        details: `expected=${expectedIdx}, got=${gotIdx}`,
      });

      logitsBuf.destroy();
      partVal.destroy();
      partIdx.destroy();
      tokOut.destroy();
      u1.destroy();
      u2.destroy();
    }

    return entries;
  }

  async runFullModelDecodeBench(
    config: WgslBenchConfig,
    onProgress?: (msg: string) => void,
  ): Promise<WgslBenchResult> {
    const startedAt = new Date().toISOString();
    const gpuBandwidthGBs = config.gpuBandwidthGBs ?? 936;
    const wasm = await loadWgslWasm(this.assetBaseUrl);
    const { device, hasShaderF16, hasSubgroups, maxBufferSize } = await acquireWgslDevice();

    const allocatedBuffers: GpuBufferLike[] = [];
    const track = (b: GpuBufferLike): GpuBufferLike => {
      allocatedBuffers.push(b);
      return b;
    };

    try {
      onProgress?.(`Verifying WGSL kernels against Rust WASM CPU reference (${config.gemvVariant}, ${config.kvMode})...`);
      const correctness = await this.verifyKernels(device, wasm, hasShaderF16, hasSubgroups);

      const gemvShaderId: Record<WgslGemvVariant, number> = {
        'q4-scalar': 0,
        'q4-vec4-f16': 1,
        'q4-subgroups-f16': 2,
      };
      if ((config.gemvVariant === 'q4-vec4-f16' || config.gemvVariant === 'q4-subgroups-f16') && !hasShaderF16) {
        throw new Error('GPU lacks shader-f16 required for selected GEMV variant.');
      }
      if ((config.gemvVariant === 'q4-subgroups-f16' || !hasSubgroups) && !hasSubgroups) {
        throw new Error('GPU lacks subgroups feature required for Track 2 WGSL Split-K FlashDecoding.');
      }

      const GQA_NUM_SPLITS = 32;
      const compileStart = performance.now();

      // Create dynamic-offset bind group layouts (all distinct non-aliasing storage bindings)
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

      const dynStorage4Layout = device.createPipelineLayout({ bindGroupLayouts: [dynStorage4Bgl] });
      const dynRwStorageLayout = device.createPipelineLayout({ bindGroupLayouts: [dynRwStorageBgl] });

      const [gemvPipe, dnPipe, gqaSplitPipe, gqaReducePipe, elemPipe, arg1Pipe, arg2Pipe] = await Promise.all([
        createValidatedPipeline(
          device,
          `gemv-${config.gemvVariant}`,
          getShaderSource(wasm, gemvShaderId[config.gemvVariant]),
          dynStorage4Layout,
        ),
        createValidatedPipeline(
          device,
          'deltanet-step',
          getShaderSource(wasm, 3),
          dynRwStorageLayout,
        ),
        createValidatedPipeline(
          device,
          `gqa-${config.kvMode}-split`,
          getShaderSource(wasm, config.kvMode === 'int8' ? 5 : 4),
          dynRwStorageLayout,
        ),
        createValidatedPipeline(
          device,
          'gqa-reduce',
          getShaderSource(wasm, 9),
          dynStorage4Layout,
        ),
        createValidatedPipeline(
          device,
          'rmsnorm-swiglu',
          getShaderSource(wasm, 6),
          dynStorage4Layout,
        ),
        createValidatedPipeline(
          device,
          'argmax-pass1',
          getShaderSource(wasm, 7),
          device.createPipelineLayout({ bindGroupLayouts: [staticBgl] }),
        ),
        createValidatedPipeline(
          device,
          'argmax-pass2',
          getShaderSource(wasm, 8),
          device.createPipelineLayout({ bindGroupLayouts: [staticRead2Bgl] }),
        ),
      ]);
      const pipelineCompileMs = performance.now() - compileStart;

      // Plan Qwen3.5-4B weight offsets across 3 slabs (Layers 0..15, Layers 16..31, LM Head)
      const q4Bytes = (r: number, c: number) => Number(wasm.q4_matrix_bytes(r, c));
      const dnLayerBytes =
        q4Bytes(8192, HIDDEN) + // qkv
        q4Bytes(4096, HIDDEN) + // z
        q4Bytes(64, HIDDEN) + // a+b gates
        q4Bytes(HIDDEN, 4096) + // out_proj
        q4Bytes(INTERMEDIATE, HIDDEN) * 2 + // gate, up
        q4Bytes(HIDDEN, INTERMEDIATE); // down
      const faLayerBytes =
        q4Bytes(8192, HIDDEN) + // q+gate
        q4Bytes(1024, HIDDEN) * 2 + // k, v
        q4Bytes(HIDDEN, 4096) + // o_proj
        q4Bytes(INTERMEDIATE, HIDDEN) * 2 + // gate, up
        q4Bytes(HIDDEN, INTERMEDIATE); // down

      let slab0Bytes = 0;
      let slab1Bytes = 0;
      for (let l = 0; l < NUM_LAYERS; l++) {
        const b = FULL_ATTN_LAYERS.has(l) ? faLayerBytes : dnLayerBytes;
        if (l < 16) slab0Bytes += b;
        else slab1Bytes += b;
      }
      const lmHeadBytes = q4Bytes(VOCAB_SIZE, HIDDEN);
      const totalWeightBytes = slab0Bytes + slab1Bytes + lmHeadBytes;

      onProgress?.(
        `Allocating Qwen3.5-4B Q4 slabs (${(totalWeightBytes / (1024 * 1024)).toFixed(0)} MiB) + hybrid states...`,
      );

      const weightSlab0 = track(device.createBuffer({ size: slab0Bytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }));
      const weightSlab1 = track(device.createBuffer({ size: slab1Bytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }));
      const weightSlab2 = track(device.createBuffer({ size: lmHeadBytes, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }));
      writeBufferChunkedQ4(device, weightSlab0, slab0Bytes);
      writeBufferChunkedQ4(device, weightSlab1, slab1Bytes);
      writeBufferChunkedQ4(device, weightSlab2, lmHeadBytes);

      // DeltaNet state buffer (24 layers)
      const dnSingleBytes = wasm.deltanet_layer_state_bytes();
      const dnTotalBytes = 24 * dnSingleBytes;
      const dnStateBuf = track(device.createBuffer({ size: dnTotalBytes, usage: GPU_USAGE.STORAGE }));

      // Static KV cache slabs (4 layers per slab so each slab <= 1.04 GiB even at 128K!)
      const maxContext = Math.max(...config.contextLengths, 1024);
      const maxSeqCapacity = maxContext + config.decodeTokens + 8;
      const kvSingleBytes = Number(wasm.full_attn_layer_kv_bytes(maxSeqCapacity, config.kvMode === 'int8' ? 1 : 0));
      const kvSlabBytes = 4 * kvSingleBytes;
      const kvSlab0 = track(device.createBuffer({ size: kvSlabBytes, usage: GPU_USAGE.STORAGE }));
      const kvSlab1 = track(device.createBuffer({ size: kvSlabBytes, usage: GPU_USAGE.STORAGE }));

      // Distinct non-aliasing scratch buffers for WebGPU validation rules
      const hiddenInit = new Float32Array(HIDDEN);
      for (let i = 0; i < HIDDEN; i++) hiddenInit[i] = Math.sin(i * 0.01) * 0.1;
      const hiddenBuf = track(device.createBuffer({ size: HIDDEN * 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }));
      device.queue.writeBuffer(hiddenBuf, 0, hiddenInit);
      const normBuf = track(device.createBuffer({ size: HIDDEN * 4, usage: GPU_USAGE.STORAGE }));
      const workBuf = track(device.createBuffer({ size: 24576 * 4, usage: GPU_USAGE.STORAGE }));
      const attnOutBuf = track(device.createBuffer({ size: INTERMEDIATE * 4, usage: GPU_USAGE.STORAGE }));
      const gqaSplitBuf = track(device.createBuffer({ size: 16 * GQA_NUM_SPLITS * 260 * 4, usage: GPU_USAGE.STORAGE }));

      const gammaCpu = new Float32Array(HIDDEN * 2).fill(1.0);
      const gammaBuf = track(device.createBuffer({ size: gammaCpu.byteLength, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST }));
      device.queue.writeBuffer(gammaBuf, 0, gammaCpu);

      const logitsBuf = track(device.createBuffer({ size: VOCAB_SIZE * 4, usage: GPU_USAGE.STORAGE }));
      const argmaxWgCount = Math.ceil(VOCAB_SIZE / 256);
      const partValBuf = track(device.createBuffer({ size: argmaxWgCount * 4, usage: GPU_USAGE.STORAGE }));
      const partIdxBuf = track(device.createBuffer({ size: argmaxWgCount * 4, usage: GPU_USAGE.STORAGE }));
      const tokOutBuf = track(device.createBuffer({ size: 4, usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC }));
      const staging4B = track(device.createBuffer({ size: 4, usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST }));

      // Build uniform table (256-byte slots) & static non-aliasing bind groups for the entire 32-layer decode graph
      const maxSlots = 512;
      const uniformBytes = maxSlots * 256;
      const uniformBuf = track(device.createBuffer({ size: uniformBytes, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
      const uniformTable = new ArrayBuffer(uniformBytes);
      const u32View = new Uint32Array(uniformTable);
      const f32View = new Float32Array(uniformTable);

      const makeDyn4Bg = (b1: GpuBufferLike, b2: GpuBufferLike, b3: GpuBufferLike) =>
        device.createBindGroup({
          layout: dynStorage4Bgl,
          entries: [
            { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
            { binding: 1, resource: { buffer: b1 } },
            { binding: 2, resource: { buffer: b2 } },
            { binding: 3, resource: { buffer: b3 } },
          ],
        });

      const bgGemv0NormToWork = makeDyn4Bg(weightSlab0, normBuf, workBuf);
      const bgGemv1NormToWork = makeDyn4Bg(weightSlab1, normBuf, workBuf);
      const bgGemv0AttnToHidden = makeDyn4Bg(weightSlab0, attnOutBuf, hiddenBuf);
      const bgGemv1AttnToHidden = makeDyn4Bg(weightSlab1, attnOutBuf, hiddenBuf);
      const bgLmHead = makeDyn4Bg(weightSlab2, normBuf, logitsBuf);
      const bgRmsNorm = makeDyn4Bg(gammaBuf, hiddenBuf, normBuf);
      const bgSwiGlu = makeDyn4Bg(gammaBuf, workBuf, attnOutBuf);
      const bgGqaReduce = makeDyn4Bg(workBuf, gqaSplitBuf, attnOutBuf);

      const bgDn = device.createBindGroup({
        layout: dynRwStorageBgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: workBuf } },
          { binding: 2, resource: { buffer: dnStateBuf } },
          { binding: 3, resource: { buffer: attnOutBuf } },
        ],
      });
      const bgGqaSplit0 = device.createBindGroup({
        layout: dynRwStorageBgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: workBuf } },
          { binding: 2, resource: { buffer: kvSlab0 } },
          { binding: 3, resource: { buffer: gqaSplitBuf } },
        ],
      });
      const bgGqaSplit1 = device.createBindGroup({
        layout: dynRwStorageBgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf, size: 32 } },
          { binding: 1, resource: { buffer: workBuf } },
          { binding: 2, resource: { buffer: kvSlab1 } },
          { binding: 3, resource: { buffer: gqaSplitBuf } },
        ],
      });

      const uArg1 = track(device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
      const uArg2 = track(device.createBuffer({ size: 16, usage: GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST }));
      device.queue.writeBuffer(uArg1, 0, new Uint32Array([VOCAB_SIZE, 0, 0, 0]));
      device.queue.writeBuffer(uArg2, 0, new Uint32Array([argmaxWgCount, 0, 0, 0]));
      const bgArg1 = device.createBindGroup({
        layout: staticBgl,
        entries: [
          { binding: 0, resource: { buffer: uArg1 } },
          { binding: 1, resource: { buffer: logitsBuf } },
          { binding: 2, resource: { buffer: partValBuf } },
          { binding: 3, resource: { buffer: partIdxBuf } },
        ],
      });
      const bgArg2 = device.createBindGroup({
        layout: staticRead2Bgl,
        entries: [
          { binding: 0, resource: { buffer: uArg2 } },
          { binding: 1, resource: { buffer: partValBuf } },
          { binding: 2, resource: { buffer: partIdxBuf } },
          { binding: 3, resource: { buffer: tokOutBuf } },
        ],
      });

      const steps: DispatchStep[] = [];
      const gqaSlotByteOffsets: number[] = [];
      let nextSlot = 0;

      const pushSlot = (
        pipeline: unknown,
        bindGroup: unknown,
        vals: [number, number, number, number, number, number, number, number],
        wx: number,
        isFloat5 = false,
        wy = 1,
      ): number => {
        const slot = nextSlot++;
        const byteOff = slot * 256;
        const uIdx = byteOff / 4;
        for (let i = 0; i < 8; i++) {
          if (isFloat5 && i === 5) {
            f32View[uIdx + i] = vals[i];
          } else {
            u32View[uIdx + i] = vals[i];
          }
        }
        steps.push({ pipeline, bindGroup, uniformByteOffset: byteOff, wx, wy, wz: 1 });
        return byteOff;
      };

      const gemvWx = (rows: number) =>
        config.gemvVariant === 'q4-subgroups-f16' ? Math.ceil(rows / 4) : Math.ceil(rows / 64);

      const OFF_Z = 8192;
      const OFF_AB = 12288;
      const OFF_MLP_GATE = 0;
      const OFF_MLP_UP = 9216;

      let wOff0 = 0;
      let wOff1 = 0;
      let dnIdx = 0;
      let faIdx = 0;

      for (let layer = 0; layer < NUM_LAYERS; layer++) {
        const inSlab0 = layer < 16;
        const bgNormToWork = inSlab0 ? bgGemv0NormToWork : bgGemv1NormToWork;
        const bgAttnToHidden = inSlab0 ? bgGemv0AttnToHidden : bgGemv1AttnToHidden;
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

        // 1. Pre-attn RMSNorm (hiddenBuf -> normBuf)
        pushSlot(elemPipe, bgRmsNorm, [0, HIDDEN, 0, 0, 0, 1e-6, 0, 0], 1, true);

        if (!FULL_ATTN_LAYERS.has(layer)) {
          // Gated DeltaNet Layer
          const wQkv = takeW(8192, HIDDEN);
          const wZ = takeW(4096, HIDDEN);
          const wAb = takeW(64, HIDDEN);
          const wOut = takeW(HIDDEN, 4096);

          pushSlot(gemvPipe, bgNormToWork, [8192, HIDDEN, wQkv, 0, 0, 0, 0, 0], gemvWx(8192));
          pushSlot(gemvPipe, bgNormToWork, [4096, HIDDEN, wZ, 0, OFF_Z, 0, 0, 0], gemvWx(4096));
          pushSlot(gemvPipe, bgNormToWork, [64, HIDDEN, wAb, 0, OFF_AB, 0, 0, 0], gemvWx(64));

          const dnBaseF32 = dnIdx * (dnSingleBytes / 4);
          dnIdx++;
          pushSlot(
            dnPipe,
            bgDn,
            [dnBaseF32 + 8192 * 4, dnBaseF32, 0, 2048, 4096, OFF_Z, OFF_AB, 0],
            32,
          );
          pushSlot(gemvPipe, bgAttnToHidden, [HIDDEN, 4096, wOut, 0, 0, 1, 0, 0], gemvWx(HIDDEN));
        } else {
          // Full Attention GQA Layer (Split-K FlashDecoding)
          const wQg = takeW(8192, HIDDEN);
          const wK = takeW(1024, HIDDEN);
          const wV = takeW(1024, HIDDEN);
          const wO = takeW(HIDDEN, 4096);

          pushSlot(gemvPipe, bgNormToWork, [8192, HIDDEN, wQg, 0, 0, 0, 0, 0], gemvWx(8192));
          pushSlot(gemvPipe, bgNormToWork, [1024, HIDDEN, wK, 0, 8192, 0, 0, 0], gemvWx(1024));
          pushSlot(gemvPipe, bgNormToWork, [1024, HIDDEN, wV, 0, 9216, 0, 0, 0], gemvWx(1024));

          const inKv0 = faIdx < 4;
          const localKvIdx = inKv0 ? faIdx : faIdx - 4;
          const kvBaseU32 = localKvIdx * (kvSingleBytes / 4);
          faIdx++;

          const slotOff = pushSlot(
            gqaSplitPipe,
            inKv0 ? bgGqaSplit0 : bgGqaSplit1,
            [kvBaseU32, maxSeqCapacity, 0, 0, 8192, 9216, GQA_NUM_SPLITS, 0],
            16,
            false,
            GQA_NUM_SPLITS,
          );
          gqaSlotByteOffsets.push(slotOff);
          pushSlot(gqaReducePipe, bgGqaReduce, [GQA_NUM_SPLITS, 4096, 0, 0, 0, 0, 0, 0], 16);
          pushSlot(gemvPipe, bgAttnToHidden, [HIDDEN, 4096, wO, 0, 0, 1, 0, 0], gemvWx(HIDDEN));
        }

        // 2. Post-attn RMSNorm + SwiGLU MLP + residual add
        const wGate = takeW(INTERMEDIATE, HIDDEN);
        const wUp = takeW(INTERMEDIATE, HIDDEN);
        const wDown = takeW(HIDDEN, INTERMEDIATE);
        pushSlot(elemPipe, bgRmsNorm, [0, HIDDEN, 0, HIDDEN, 0, 1e-6, 0, 0], 1, true);
        pushSlot(gemvPipe, bgNormToWork, [INTERMEDIATE, HIDDEN, wGate, 0, OFF_MLP_GATE, 0, 0, 0], gemvWx(INTERMEDIATE));
        pushSlot(gemvPipe, bgNormToWork, [INTERMEDIATE, HIDDEN, wUp, 0, OFF_MLP_UP, 0, 0, 0], gemvWx(INTERMEDIATE));
        pushSlot(elemPipe, bgSwiGlu, [1, INTERMEDIATE, OFF_MLP_GATE, OFF_MLP_UP, 0, 0, 0, 0], Math.ceil(INTERMEDIATE / 256));
        pushSlot(gemvPipe, bgAttnToHidden, [HIDDEN, INTERMEDIATE, wDown, 0, 0, 1, 0, 0], gemvWx(HIDDEN));
      }

      // Final RMSNorm + LM Head (248,320 x 2,560)
      pushSlot(elemPipe, bgRmsNorm, [0, HIDDEN, 0, 0, 0, 1e-6, 0, 0], 1, true);
      pushSlot(gemvPipe, bgLmHead, [VOCAB_SIZE, HIDDEN, 0, 0, 0, 0, 0, 0], gemvWx(VOCAB_SIZE));

      device.queue.writeBuffer(uniformBuf, 0, uniformTable);
      await device.queue.onSubmittedWorkDone();

      const dispatchOneToken = async (seqPos: number): Promise<number> => {
        const seqArr = new Uint32Array([seqPos]);
        for (const gqaByteOff of gqaSlotByteOffsets) {
          device.queue.writeBuffer(uniformBuf, gqaByteOff + 8, seqArr);
        }
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (const st of steps) {
          pass.setPipeline(st.pipeline);
          pass.setBindGroup(0, st.bindGroup, [st.uniformByteOffset]);
          pass.dispatchWorkgroups(st.wx, st.wy, st.wz);
        }
        pass.setPipeline(arg1Pipe);
        pass.setBindGroup(0, bgArg1);
        pass.dispatchWorkgroups(argmaxWgCount, 1, 1);
        pass.setPipeline(arg2Pipe);
        pass.setBindGroup(0, bgArg2);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
        enc.copyBufferToBuffer(tokOutBuf, 0, staging4B, 0, 4);
        device.queue.submit([enc.finish()]);
        await staging4B.mapAsync(GPU_MAP_READ);
        const tokId = new Uint32Array(staging4B.getMappedRange())[0];
        staging4B.unmap();
        return tokId;
      };

      // Warmup with WebGPU validation error check on first token dispatch
      const warmup = config.warmupTokens ?? 2;
      for (let w = 0; w < warmup; w++) {
        if (w === 0) device.pushErrorScope?.('validation');
        await dispatchOneToken(w);
        if (w === 0) {
          const valErr = await device.popErrorScope?.();
          if (valErr) {
            throw new Error(`WebGPU validation error on decode pass: ${valErr.message}`);
          }
        }
      }

      const points: WgslBenchPoint[] = [];
      const weightMb = totalWeightBytes / (1024 * 1024);
      const deltaNetStateMb = dnTotalBytes / (1024 * 1024);
      const weightGB = totalWeightBytes / 1e9;

      for (const ctxLen of config.contextLengths) {
        onProgress?.(
          `[WGSL ${config.gemvVariant} / ${config.kvMode} KV] Benchmarking ${ctxLen.toLocaleString()} ctx (${config.decodeTokens} decode tokens)...`,
        );
        try {
          // One warmup dispatch at ctxLen so GPU instruction/L2 caches settle
          await dispatchOneToken(Math.max(0, ctxLen - 1));
          const t0 = performance.now();
          for (let i = 0; i < config.decodeTokens; i++) {
            await dispatchOneToken(ctxLen + i);
          }
          const decodeMs = Math.max(0.1, performance.now() - t0);
          const decodeTokPerSec = config.decodeTokens / (decodeMs / 1000);

          const activeKvBytes = Number(
            wasm.full_attn_layer_kv_bytes(ctxLen + config.decodeTokens, config.kvMode === 'int8' ? 1 : 0),
          ) * 8;
          const kvCacheMb = activeKvBytes / (1024 * 1024);
          const achievedBandwidthGBs = decodeTokPerSec * weightGB;
          const bandwidthUtilizationPct =
            gpuBandwidthGBs > 0 ? (achievedBandwidthGBs / gpuBandwidthGBs) * 100 : 0;

          points.push({
            contextLength: ctxLen,
            decodeTokens: config.decodeTokens,
            decodeMs,
            decodeTokPerSec,
            kvCacheMb,
            deltaNetStateMb,
            weightMb,
            estVramMb: weightMb + kvCacheMb + deltaNetStateMb,
            achievedBandwidthGBs,
            bandwidthUtilizationPct,
            status: 'ok',
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          points.push({
            contextLength: ctxLen,
            decodeTokens: 0,
            decodeMs: 0,
            decodeTokPerSec: 0,
            kvCacheMb: 0,
            deltaNetStateMb,
            weightMb,
            estVramMb: weightMb + deltaNetStateMb,
            achievedBandwidthGBs: 0,
            bandwidthUtilizationPct: 0,
            status: /lost/i.test(msg) ? 'device_lost' : /oom|memory|buffer/i.test(msg) ? 'oom' : 'error',
            error: msg,
          });
          break;
        }
      }

      return {
        startedAt,
        finishedAt: new Date().toISOString(),
        gemvVariant: config.gemvVariant,
        kvMode: config.kvMode,
        gpuBandwidthGBs,
        hasShaderF16,
        hasSubgroups,
        maxBufferSize,
        allocatedWeightMb: weightMb,
        allocatedKvMb: (kvSlabBytes * 2) / (1024 * 1024),
        allocatedDeltaNetMb: deltaNetStateMb,
        pipelineCompileMs,
        correctness,
        points,
      };
    } finally {
      for (const buf of allocatedBuffers) {
        try {
          buf.destroy();
        } catch {
          // Ignore already destroyed buffers
        }
      }
      try {
        device.destroy();
      } catch {
        // Ignore
      }
    }
  }
}
