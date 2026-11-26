import { resolveExtensionAssetUrl, sleepMs } from './gpuPacing.ts';
import { computeElevatedLimits } from './webgpuLimits.ts';

export type WgslComputeTier = 'tier1-f16-subgroups' | 'tier2-f16-shared' | 'tier3-f32-universal';

export interface GpuBufferLike {
  size: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface GpuComputePassLike {
  setPipeline(pipeline: unknown): void;
  setBindGroup(index: number, bindGroup: unknown, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface GpuCommandEncoderLike {
  beginComputePass(): GpuComputePassLike;
  copyBufferToBuffer(
    src: GpuBufferLike,
    srcOffset: number,
    dst: GpuBufferLike,
    dstOffset: number,
    size: number,
  ): void;
  clearBuffer?(buffer: GpuBufferLike, offset?: number, size?: number): void;
  finish(): unknown;
}

export interface GpuQueueLike {
  writeBuffer(
    buffer: GpuBufferLike,
    bufferOffset: number,
    data: BufferSource | SharedArrayBuffer,
    dataOffset?: number,
    size?: number,
  ): void;
  submit(commandBuffers: unknown[]): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface GpuShaderCompilationMessage {
  type: 'error' | 'warning' | 'info' | string;
  lineNum: number;
  linePos: number;
  message: string;
}

export interface GpuShaderModuleLike {
  getCompilationInfo?(): Promise<{ messages: GpuShaderCompilationMessage[] }>;
}

export interface GpuDeviceLike {
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
  requestDevice(desc?: {
    requiredFeatures?: string[];
    requiredLimits?: Record<string, number>;
  }): Promise<GpuDeviceLike>;
}

export const GPU_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

export const GPU_MAP_READ = 0x0001;
export const GPU_SHADER_COMPUTE = 0x0004;

export interface WasmEngineExports {
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
  repack_q4_rows(
    rows: number,
    blocksPerRow: number,
    quantPtr: number,
    scalesPtr: number,
    zpPtr: number,
    outU32Ptr: number,
  ): void;
  f16_slice_to_f32(count: number, srcU16Ptr: number, dstF32Ptr: number): void;
  cpu_ref_q4_gemv(rows: number, cols: number, wPtr: number, xPtr: number, outPtr: number): void;
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

export async function loadWgslWasm(assetBaseUrl?: string): Promise<WasmEngineExports> {
  const url = resolveExtensionAssetUrl('wasm/tinygpt_wgsl_engine.wasm', assetBaseUrl);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const bytes = await res.arrayBuffer();
  const instantiated = await WebAssembly.instantiate(bytes, {});
  return instantiated.instance.exports as unknown as WasmEngineExports;
}

export function getWgslShaderSource(wasm: WasmEngineExports, shaderId: number): string {
  const len = wasm.wgsl_build_shader(shaderId);
  const ptr = wasm.wgsl_get_shader_ptr();
  const view = new Uint8Array(wasm.memory.buffer, ptr, len);
  return new TextDecoder().decode(view);
}

export async function acquireWgslDevice(): Promise<{
  device: GpuDeviceLike;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  hasDp4a: boolean;
  subgroupMinSize: number;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}> {
  const nav = navigator as unknown as {
    gpu?: {
      requestAdapter(opts?: { powerPreference?: string }): Promise<GpuAdapterLike | null>;
      wgslLanguageFeatures?: { has?(feature: string): boolean };
    };
  };
  if (!nav.gpu) {
    throw new Error('WebGPU not supported in Offscreen Document.');
  }
  const hasDp4a = Boolean(nav.gpu.wgslLanguageFeatures?.has?.('packed_4x8_integer_dot_product'));
  const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter returned.');
  }
  const adapterExtra = adapter as unknown as {
    info?: { subgroupMinSize?: number };
    limits?: { minSubgroupSize?: number };
  };
  const subgroupMinSize = Number(
    adapterExtra.info?.subgroupMinSize ?? adapterExtra.limits?.minSubgroupSize ?? 32,
  );
  const features = new Set(Array.from(adapter.features));
  const wantF16 = features.has('shader-f16');
  const wantSubgroups = features.has('subgroups');
  const requiredLimits = computeElevatedLimits(adapter.limits);

  const featureCandidates: string[][] = [];
  if (wantF16 && wantSubgroups) {
    featureCandidates.push(['shader-f16', 'subgroups']);
  }
  if (wantF16) {
    featureCandidates.push(['shader-f16']);
  }
  featureCandidates.push([]);

  let device: GpuDeviceLike | null = null;
  let lastError: unknown = null;
  for (const reqFeatures of featureCandidates) {
    try {
      device = await adapter.requestDevice({
        requiredFeatures: reqFeatures,
        requiredLimits,
      });
      break;
    } catch (err) {
      lastError = err;
    }
  }
  if (!device) {
    try {
      device = await adapter.requestDevice();
    } catch (err) {
      throw lastError ?? err;
    }
  }

  const grantedFeatures = new Set(Array.from(device.features ?? []));
  const maxBufferSize = device.limits.maxBufferSize ?? 0;
  const maxStorageBufferBindingSize =
    device.limits.maxStorageBufferBindingSize ?? maxBufferSize;

  return {
    device,
    hasShaderF16: grantedFeatures.has('shader-f16'),
    hasSubgroups: grantedFeatures.has('subgroups'),
    hasDp4a,
    subgroupMinSize,
    maxBufferSize,
    maxStorageBufferBindingSize,
  };
}

export async function createValidatedPipeline(
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
      const detail = errors.map((m) => `line ${m.lineNum}:${m.linePos}: ${m.message}`).join(' | ');
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

export async function streamFileToGpuBuffer(
  device: GpuDeviceLike,
  file: File,
  gpuBuf: GpuBufferLike,
  chunkBytes = 16 * 1024 * 1024,
  onProgress?: (loadedBytes: number) => void,
): Promise<void> {
  let offset = 0;
  while (offset < file.size) {
    const end = Math.min(offset + chunkBytes, file.size);
    const sliceBuf = await file.slice(offset, end).arrayBuffer();
    device.queue.writeBuffer(gpuBuf, offset, sliceBuf);
    offset = end;
    onProgress?.(offset);
  }
}

const DEFAULT_URL_STREAM_STAGING_BYTES = 16 * 1024 * 1024;

export interface GpuBufferWriteSegment {
  buffer: GpuBufferLike;
  dstByteOffset: number;
  byteLength: number;
}

function writeToGpuSegments(
  device: GpuDeviceLike,
  segments: GpuBufferWriteSegment[],
  segIndex: number,
  segPos: number,
  staging: Uint8Array<ArrayBuffer>,
  byteCount: number,
): { segIndex: number; segPos: number } {
  let srcOff = 0;
  let idx = segIndex;
  let pos = segPos;
  while (srcOff < byteCount) {
    const seg = segments[idx];
    if (!seg) {
      throw new Error('Stream exceeded target GPU segment capacity.');
    }
    const avail = seg.byteLength - pos;
    const take = Math.min(avail, byteCount - srcOff);
    if (take > 0) {
      device.queue.writeBuffer(seg.buffer, seg.dstByteOffset + pos, staging, srcOff, take);
      srcOff += take;
      pos += take;
    }
    if (pos >= seg.byteLength) {
      idx++;
      pos = 0;
    }
  }
  return { segIndex: idx, segPos: pos };
}

async function flushStagingToSegments(
  device: GpuDeviceLike,
  segments: GpuBufferWriteSegment[],
  state: { segIndex: number; segPos: number; totalWritten: number },
  staging: Uint8Array<ArrayBuffer>,
  byteCount: number,
  yieldMs: number,
  onProgress?: (loadedBytes: number) => void,
): Promise<void> {
  const next = writeToGpuSegments(device, segments, state.segIndex, state.segPos, staging, byteCount);
  state.segIndex = next.segIndex;
  state.segPos = next.segPos;
  state.totalWritten += byteCount;
  onProgress?.(state.totalWritten);
  if (yieldMs > 0) {
    await sleepMs(yieldMs);
  }
}

async function streamReaderToGpuSegments(
  device: GpuDeviceLike,
  url: string,
  body: ReadableStream<Uint8Array>,
  segments: GpuBufferWriteSegment[],
  staging: Uint8Array<ArrayBuffer>,
  yieldMs: number,
  onProgress?: (loadedBytes: number) => void,
): Promise<number> {
  const stagingBytes = staging.byteLength;
  const reader = body.getReader();
  let stagingPos = 0;
  const state = { segIndex: 0, segPos: 0, totalWritten: 0 };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (value && value.byteLength > 0) {
        let chunkOffset = 0;
        while (chunkOffset < value.byteLength) {
          const space = stagingBytes - stagingPos;
          const copyLen = Math.min(space, value.byteLength - chunkOffset);
          const slice =
            chunkOffset === 0 && copyLen === value.byteLength
              ? value
              : value.subarray(chunkOffset, chunkOffset + copyLen);
          staging.set(slice, stagingPos);
          stagingPos += copyLen;
          chunkOffset += copyLen;

          if (stagingPos === stagingBytes) {
            await flushStagingToSegments(
              device,
              segments,
              state,
              staging,
              stagingBytes,
              yieldMs,
              onProgress,
            );
            stagingPos = 0;
          }
        }
      }
      if (done) {
        break;
      }
    }
  } finally {
    reader.releaseLock?.();
  }

  if (stagingPos > 0) {
    if (stagingPos % 4 !== 0) {
      throw new Error(
        `Trailing streamed bytes (${stagingPos}) for ${url} are not 4-byte aligned.`,
      );
    }
    await flushStagingToSegments(
      device,
      segments,
      state,
      staging,
      stagingPos,
      0,
      onProgress,
    );
  }
  return state.totalWritten;
}

async function streamArrayBufferToGpuSegments(
  device: GpuDeviceLike,
  url: string,
  arrayBuffer: ArrayBuffer,
  segments: GpuBufferWriteSegment[],
  staging: Uint8Array<ArrayBuffer>,
  yieldMs: number,
  onProgress?: (loadedBytes: number) => void,
): Promise<number> {
  const stagingBytes = staging.byteLength;
  const bytes = new Uint8Array(arrayBuffer);
  let srcOffset = 0;
  const state = { segIndex: 0, segPos: 0, totalWritten: 0 };

  while (srcOffset < bytes.byteLength) {
    const copyLen = Math.min(stagingBytes, bytes.byteLength - srcOffset);
    if (copyLen % 4 !== 0) {
      throw new Error(
        `Trailing streamed bytes (${copyLen}) for ${url} are not 4-byte aligned.`,
      );
    }
    staging.set(bytes.subarray(srcOffset, srcOffset + copyLen), 0);
    srcOffset += copyLen;
    await flushStagingToSegments(
      device,
      segments,
      state,
      staging,
      copyLen,
      copyLen === stagingBytes ? yieldMs : 0,
      onProgress,
    );
  }
  return state.totalWritten;
}

/**
 * Streams a fetched URL across one or more ordered GPU buffer segments via a fixed-size
 * reusable staging `Uint8Array`, allowing on-disk shards to span dynamic GPU slab boundaries.
 */
export async function streamUrlToGpuSegments(
  device: GpuDeviceLike,
  url: string,
  segments: GpuBufferWriteSegment[],
  expectedBytes: number,
  options?: {
    stagingBytes?: number;
    yieldMs?: number;
    onProgress?: (loadedBytes: number) => void;
  },
): Promise<void> {
  const stagingBytes = options?.stagingBytes ?? DEFAULT_URL_STREAM_STAGING_BYTES;
  const yieldMs = options?.yieldMs ?? 0;
  const onProgress = options?.onProgress;

  for (const seg of segments) {
    if (!Number.isInteger(seg.dstByteOffset) || seg.dstByteOffset < 0 || seg.dstByteOffset % 4 !== 0) {
      throw new Error(
        `Invalid dstByteOffset (${seg.dstByteOffset}) for ${url}: must be a non-negative multiple of 4.`,
      );
    }
    if (!Number.isInteger(seg.byteLength) || seg.byteLength < 0 || seg.byteLength % 4 !== 0) {
      throw new Error(
        `Invalid segment byteLength (${seg.byteLength}) for ${url}: is not 4-byte aligned.`,
      );
    }
  }
  if (!Number.isInteger(expectedBytes) || expectedBytes < 0) {
    throw new Error(`Invalid expectedBytes (${expectedBytes}) for ${url}.`);
  }
  const segTotal = segments.reduce((sum, s) => sum + s.byteLength, 0);
  if (segTotal !== expectedBytes) {
    throw new Error(
      `Segment byte sum (${segTotal}) does not match expectedBytes (${expectedBytes}) for ${url}.`,
    );
  }
  if (!Number.isInteger(stagingBytes) || stagingBytes < 4 || stagingBytes % 4 !== 0) {
    throw new Error(`Invalid stagingBytes (${stagingBytes}): must be a positive multiple of 4.`);
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }

  const staging = new Uint8Array(stagingBytes);
  const bytesWritten = res.body
    ? await streamReaderToGpuSegments(
        device,
        url,
        res.body,
        segments,
        staging,
        yieldMs,
        onProgress,
      )
    : await streamArrayBufferToGpuSegments(
        device,
        url,
        await res.arrayBuffer(),
        segments,
        staging,
        yieldMs,
        onProgress,
      );

  if (bytesWritten !== expectedBytes) {
    throw new Error(
      `Streamed byte count mismatch for ${url}: expected ${expectedBytes} bytes, got ${bytesWritten} bytes.`,
    );
  }
}

/**
 * Streams a fetched URL directly into a GPU buffer via a fixed-size reusable staging
 * `Uint8Array`, optionally yielding between full staging flushes to preserve UI smoothness.
 */
export async function streamUrlToGpuBuffer(
  device: GpuDeviceLike,
  url: string,
  gpuBuf: GpuBufferLike,
  expectedBytes: number,
  options?: {
    dstByteOffset?: number;
    stagingBytes?: number;
    yieldMs?: number;
    onProgress?: (loadedBytes: number) => void;
  },
): Promise<void> {
  const dstByteOffset = options?.dstByteOffset ?? 0;
  return streamUrlToGpuSegments(
    device,
    url,
    [{ buffer: gpuBuf, dstByteOffset, byteLength: expectedBytes }],
    expectedBytes,
    options,
  );
}

export async function readGpuF32Slice(
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

export async function readGpuU32Scalar(device: GpuDeviceLike, src: GpuBufferLike): Promise<number> {
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
