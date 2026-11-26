import { selectComputeTier } from '@yours/models/hardware';
import type { GpuPacingMode, WgslKvMode } from '../shared/protocol.ts';
import {
  calibratePacingForHardware,
  computeDecodeBurstBreatherMs,
  computeDecodeSliceBounds,
  computePrefillSliceBreatherMs,
  getGpuPacingPolicy,
  selectDecodeSliceCount,
  selectPrefillChunkSize,
  sleepMs,
} from './gpuPacing.ts';
import { SuffixSpeculator } from './suffixSpeculator.ts';
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
  type GpuDeviceLike,
  type WasmEngineExports,
  type WgslComputeTier,
} from './wgslWebgpu.ts';

export const GEMMA4_HIDDEN = 2560;
export const GEMMA4_INTERMEDIATE = 10240;
export const GEMMA4_NUM_LAYERS = 42;
export const GEMMA4_VOCAB_SIZE = 262144;
export const GEMMA4_ATTN_HEADS = 8;
export const GEMMA4_KV_HEADS = 2;
export const GEMMA4_SLIDING_HEAD_DIM = 256;
export const GEMMA4_GLOBAL_HEAD_DIM = 512;
export const GEMMA4_SLIDING_WINDOW = 512;
export const GEMMA4_SLIDING_RING_SLOTS = 1024;
export const GEMMA4_KV_SHARED_START_LAYER = 24;
export const GEMMA4_SLIDING_DONOR_LAYER = 22;
export const GEMMA4_GLOBAL_DONOR_LAYER = 23;
export const GEMMA4_PLE_DIM = 256;
export const GEMMA4_PLE_Q_BYTES = 2688;
export const GEMMA4_PLE_S_BYTES = 84;
export const GEMMA4_PLE_ROW_BYTES = GEMMA4_PLE_Q_BYTES + GEMMA4_PLE_S_BYTES; // 2772
export const GEMMA4_SOFTCAP = 30.0;
export const GEMMA4_GLOBAL_SPLITS = 64;
export const GEMMA4_SLIDING_SPLITS = 16;
export const GEMMA4_PREFILL_MAX_CHUNK = 64;
export const GEMMA4_MAX_SPEC_DRAFT = 4;
export const GEMMA4_MAX_SPEC_TOKENS = 1 + GEMMA4_MAX_SPEC_DRAFT;
export const GEMMA4_FULL_ATTN_LAYERS = new Set([5, 11, 17, 23, 29, 35, 41]);

const UNIFORM_SLOT_BYTES = 256;
const UNIFORM_SLOT_U32S = UNIFORM_SLOT_BYTES / 4; // 64
const UNIFORM_BANK_SLOTS = 2048;
const UNIFORM_MAX_SLOTS = UNIFORM_BANK_SLOTS * 2;
const PLE_LRU_MAX_ENTRIES = 8192;

interface Gemma4WasmExports extends WasmEngineExports {
  g4_dequant_ple_rows(numTokens: number, rawPtr: number, outF32Ptr: number): void;
}

export interface Gemma4TensorSlice {
  offset: number;
  bytes: number;
  dims?: number[];
  dataType?: number;
}

export interface Gemma4QuantPair {
  qOffset: number;
  qBytes: number;
  sOffset: number;
  sBytes: number;
  qDims?: number[];
}

export interface Gemma4ShardPart {
  file: string;
  bytes: number;
}

export interface Gemma4AuxLayerLayout {
  layer: number;
  isFullAttn: boolean;
  isKvShared: boolean;
  input_layernorm: Gemma4TensorSlice;
  q_norm: Gemma4TensorSlice;
  k_norm?: Gemma4TensorSlice;
  post_attention_layernorm: Gemma4TensorSlice;
  pre_feedforward_layernorm: Gemma4TensorSlice;
  post_feedforward_layernorm: Gemma4TensorSlice;
  post_per_layer_input_norm: Gemma4TensorSlice;
  layer_scalar: Gemma4TensorSlice;
}

export interface Gemma4LayerWeightsLayout {
  layer: number;
  slab: number;
  isFullAttn: boolean;
  isKvShared: boolean;
  qkv_proj?: Gemma4QuantPair;
  q_proj: Gemma4QuantPair;
  k_proj?: Gemma4QuantPair;
  v_proj?: Gemma4QuantPair;
  o_proj: Gemma4QuantPair;
  gate_up_proj: Gemma4QuantPair;
  down_proj: Gemma4QuantPair;
  ple_gate: Gemma4QuantPair;
  ple_proj: Gemma4QuantPair;
}

export interface Gemma4Manifest {
  version: number;
  arch: string;
  modelId: string;
  complete: boolean;
  textConfig: {
    hiddenSize: number;
    intermediateSize: number;
    numLayers: number;
    vocabSize: number;
    numAttentionHeads: number;
    numKeyValueHeads: number;
    headDim: number;
    globalHeadDim: number;
    slidingWindow: number;
    numKvSharedLayers: number;
    hiddenSizePerLayerInput: number;
    finalLogitSoftcapping: number;
    rmsNormEps: number;
    layerTypes: string[];
  };
  slab0Bytes: number;
  slab1Bytes: number;
  slab2Bytes: number;
  auxBytes: number;
  shards: {
    slab0: Gemma4ShardPart[];
    slab1: Gemma4ShardPart[];
    slab2: Gemma4ShardPart[];
    aux: Gemma4ShardPart[];
  };
  lmHeadLayout: Gemma4QuantPair;
  auxLayout: {
    per_layer_projection_norm: Gemma4TensorSlice;
    per_layer_projection_weight: Gemma4TensorSlice;
    per_layer_projection_q8?: Gemma4QuantPair & { rows: number; cols: number };
    final_norm: Gemma4TensorSlice;
    layers: Gemma4AuxLayerLayout[];
  };
  layerOffsets: Gemma4LayerWeightsLayout[];
  cpuEmbeddings?: {
    embedTokens?: { file: string; bytes: number };
    pleTokens?: { file: string; bytes: number; qOffset: number; qBytes: number; sOffset: number; sBytes: number };
  };
  pleShards?: {
    dir: string;
    shardCount: number;
    tokensPerShard: number;
    rowBytes: number;
  };
  pleHot?: {
    dataFile: string;
    dataBytes: number;
    mapFile: string;
    mapBytes: number;
    entryCount: number;
  };
}

export interface Gemma4LayerDonorRoute {
  layer: number;
  isFullAttn: boolean;
  isKvShared: boolean;
  donorLayer: number;
  localDonorIdx: number;
  globalBufIdx: 0 | 1 | -1;
}

export interface Gemma4KvMemoryPlan {
  maxContext: number;
  kvMode: WgslKvMode;
  slidingKvBytes: number;
  globalKvPerBufBytes: number;
  globalKvTotalBytes: number;
  kvTotalBytes: number;
  kvTotalMb: number;
  scratchBytes: number;
  weightBytes: number;
  totalVramBytes: number;
  totalVramMb: number;
}

export interface Gemma4LoadOptions {
  baseUrl?: string;
  wasmUrl?: string;
  kvMode?: WgslKvMode;
  maxContext?: number;
  pacingMode?: GpuPacingMode;
  displayHz?: number;
  pleRowFetcher: (tokenIds: readonly number[]) => Promise<Uint8Array>;
  onProgress?: (stage: string, pct: number) => void;
  enableFusion?: boolean;
  enableSmallBatchSpec?: boolean;
  enableVec4Gemv?: boolean;
  enableDp4a?: boolean;
  enableFlashPrefill?: boolean;
  enableGpuHotPle?: boolean;
  enableQ8PleProj?: boolean;
  enablePipelinedDecode?: boolean;
  enableW4A8Dp4a?: boolean;
}

export interface Gemma4LoadStats {
  coldLoadMs: number;
  streamWeightsMs: number;
  pipelineCompileMs: number;
  computeTier: WgslComputeTier;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  hasDp4a: boolean;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  kvMode: WgslKvMode;
  maxContext: number;
  vramPlan: Gemma4KvMemoryPlan;
}

export interface Gemma4TopLogit {
  id: number;
  val: number;
}

export interface Gemma4PrefillResult {
  prefillMs: number;
  prefillTokPerSec: number;
  pleFetchMs: number;
  pleDequantMs: number;
  nextTokenId: number;
  top5Logits?: Gemma4TopLogit[];
}

export interface Gemma4DecodeStepResult {
  nextTokenId: number;
  forwardMs: number;
  top5Logits?: Gemma4TopLogit[];
}

export interface Gemma4CpuSnapshot {
  seqPos: number;
  contextTokenIds: number[];
  kvMode: WgslKvMode;
  maxContext: number;
  slidingSeqCount: number;
  slidingBytes: Uint8Array<ArrayBuffer>;
  global0Bytes: Uint8Array<ArrayBuffer>;
  global1Bytes: Uint8Array<ArrayBuffer>;
}

export interface Gemma4GenerateOptions {
  maxNewTokens?: number;
  eosIds?: Set<number>;
  enableSpeculative?: boolean;
  draftSkipLayers?: number;
  abortSignal?: AbortSignal;
  chunkSize?: number;
  onToken?: (tokenId: number, stepIdx: number) => void;
}

export interface Gemma4GenerateResult {
  generatedIds: number[];
  promptTokens: number;
  prefixReusedTokens: number;
  prefillMs: number;
  ttftMs: number;
  prefillTokPerSec: number;
  decodeMs: number;
  decodeTokPerSec: number;
  step0Top5Logits?: Gemma4TopLogit[];
  aborted: boolean;
  speculativeStats: {
    enabled: boolean;
    drafted: number;
    accepted: number;
    steps: number;
    acceptanceRatePct: number;
  } | null;
}

interface DispatchStep {
  pipeline: unknown;
  bindGroup: unknown;
  uniformByteOffset: number;
  wx: number;
  wy: number;
  wz: number;
}

/** Clamps token ID to `0` if out-of-vocab or reserved image/audio sentinel (`258880`/`258881`), matching ONNX `Where`. */
export function normalizeGemma4PleTokenId(rawId: number): number {
  if (!Number.isInteger(rawId) || rawId < 0 || rawId >= GEMMA4_VOCAB_SIZE) {
    return 0;
  }
  if (rawId === 258880 || rawId === 258881) {
    return 0;
  }
  return rawId;
}

/** Builds exact per-layer donor mappings for Gemma 4 E4B (`0..23` own KV; `24..41` share `22` sliding / `23` full). */
export function buildGemma4DonorRoutes(
  numLayers = GEMMA4_NUM_LAYERS,
  fullAttnLayers = GEMMA4_FULL_ATTN_LAYERS,
  kvSharedStart = GEMMA4_KV_SHARED_START_LAYER,
): Gemma4LayerDonorRoute[] {
  const layerToSlidingIdx = new Map<number, number>();
  const layerToGlobalIdx = new Map<number, number>();
  let slideCount = 0;
  let globalCount = 0;

  for (let l = 0; l < kvSharedStart; l++) {
    if (fullAttnLayers.has(l)) {
      layerToGlobalIdx.set(l, globalCount);
      globalCount += 1;
    } else {
      layerToSlidingIdx.set(l, slideCount);
      slideCount += 1;
    }
  }

  const routes: Gemma4LayerDonorRoute[] = [];
  for (let l = 0; l < numLayers; l++) {
    const isFullAttn = fullAttnLayers.has(l);
    const isKvShared = l >= kvSharedStart;
    if (isFullAttn) {
      const donorLayer = isKvShared ? GEMMA4_GLOBAL_DONOR_LAYER : l;
      const gIdx = layerToGlobalIdx.get(donorLayer) ?? 3;
      const globalBufIdx: 0 | 1 = gIdx < 2 ? 0 : 1;
      routes.push({
        layer: l,
        isFullAttn: true,
        isKvShared,
        donorLayer,
        localDonorIdx: gIdx & 1,
        globalBufIdx,
      });
    } else {
      const donorLayer = isKvShared ? GEMMA4_SLIDING_DONOR_LAYER : l;
      const sIdx = layerToSlidingIdx.get(donorLayer) ?? 19;
      routes.push({
        layer: l,
        isFullAttn: false,
        isKvShared,
        donorLayer,
        localDonorIdx: sIdx,
        globalBufIdx: -1,
      });
    }
  }
  return routes;
}

/** Computes exact VRAM allocation plan for weights, fixed 40 MiB sliding ring KV, global FP16/INT8/INT4 KV, and scratch. */
export function computeGemma4KvMemoryPlan(
  maxContext: number,
  kvMode: WgslKvMode,
  weightBytes: number,
  scratchBytes = 25_200_000,
): Gemma4KvMemoryPlan {
  // 20 sliding donors * 2 (K,V) * 2 heads * 1024 slots * 128 u32s * 4B = 41,943,040 B (40.0 MiB)
  const slidingKvBytes = 20 * 4 * GEMMA4_SLIDING_RING_SLOTS * 128 * 4;
  // Each global buffer holds 2 donors * 2 (K,V) * 2 heads = 8 head-planes
  const globalU32sPerTokHead = kvMode === 'int4' ? 68 : kvMode === 'int8' ? 129 : 256;
  const globalKvPerBufBytes = 2 * 4 * maxContext * globalU32sPerTokHead * 4;
  const globalKvTotalBytes = globalKvPerBufBytes * 2;
  const kvTotalBytes = slidingKvBytes + globalKvTotalBytes;
  const totalVramBytes = weightBytes + kvTotalBytes + scratchBytes;

  return {
    maxContext,
    kvMode,
    slidingKvBytes,
    globalKvPerBufBytes,
    globalKvTotalBytes,
    kvTotalBytes,
    kvTotalMb: Number((kvTotalBytes / (1024 * 1024)).toFixed(2)),
    scratchBytes,
    weightBytes,
    totalVramBytes,
    totalVramMb: Number((totalVramBytes / (1024 * 1024)).toFixed(2)),
  };
}

/** Extracts exact top-K entries from a Float32 logit slice. */
export function extractTopKLogits(logits: Float32Array, k = 5): Gemma4TopLogit[] {
  const top: Gemma4TopLogit[] = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (top.length < k || v > top[top.length - 1].val) {
      top.push({ id: i, val: Number(v.toFixed(4)) });
      top.sort((a, b) => b.val - a.val);
      if (top.length > k) top.length = k;
    }
  }
  return top;
}

export class Gemma4WgslEngine {
  private device: GpuDeviceLike | null = null;
  private wasm: Gemma4WasmExports | null = null;
  private manifest: Gemma4Manifest | null = null;
  private routes: Gemma4LayerDonorRoute[] = [];
  private computeTier: WgslComputeTier = 'tier3-f32-universal';
  private hasSubgroups = false;
  private hasDp4a = false;
  private subgroupMinSize = 32;
  private useFusedQkv = false;
  private useSmallBatchSpec = false;
  private useVec4Gemv = false;
  private useDp4a = false;
  private useFlashPrefill = false;
  private useGpuHotPle = false;
  private useQ8PleProj = false;
  private usePipelinedDecode = false;
  private useW4A8Dp4a = false;
  private pleHotEntryCount = 0;
  private pleHotMapCpu: Uint32Array | null = null;
  private kvMode: WgslKvMode = 'int8';
  private maxContext = 131072;
  private pacingMode: GpuPacingMode = 'max-speed';
  private displayHz = 60;
  private pleRowFetcher: ((tokenIds: readonly number[]) => Promise<Uint8Array>) | null = null;

  // Model weight & hot PLE buffers
  private slab0: GpuBufferLike | null = null;
  private slab1: GpuBufferLike | null = null;
  private slab2: GpuBufferLike | null = null;
  private auxBuf: GpuBufferLike | null = null;
  private pleHotDataBuf: GpuBufferLike | null = null;
  private pleHotMapBuf: GpuBufferLike | null = null;

  // KV cache buffers
  private kvSliding: GpuBufferLike | null = null;
  private kvGlobal0: GpuBufferLike | null = null;
  private kvGlobal1: GpuBufferLike | null = null;

  // Reusable activation buffers
  private tokenIdsBuf: GpuBufferLike | null = null;
  private hiddenBuf: GpuBufferLike | null = null;
  private normBuf: GpuBufferLike | null = null;
  private qkvBuf: GpuBufferLike | null = null;
  private qBuf: GpuBufferLike | null = null;
  private kBuf: GpuBufferLike | null = null;
  private vBuf: GpuBufferLike | null = null;
  private attnOutBuf: GpuBufferLike | null = null;
  private mlpGateUpBuf: GpuBufferLike | null = null;
  private mlpActBuf: GpuBufferLike | null = null;
  private pleRawBuf: GpuBufferLike | null = null;
  private pleProjBuf: GpuBufferLike | null = null;
  private pleCombinedBuf: GpuBufferLike | null = null;
  private pleActBuf: GpuBufferLike | null = null;
  private logitsBuf: GpuBufferLike | null = null;
  private logitsSingleBuf: GpuBufferLike | null = null;
  private argmaxPartValBuf: GpuBufferLike | null = null;
  private argmaxPartIdxBuf: GpuBufferLike | null = null;
  private argmaxSingleOutBuf: GpuBufferLike | null = null;
  private argmaxMultiOutBuf: GpuBufferLike | null = null;
  private argmaxReadBuf: GpuBufferLike | null = null;
  private argmaxReadBufB: GpuBufferLike | null = null;
  private partialAttnBuf: GpuBufferLike | null = null;
  private partialPrefillBuf: GpuBufferLike | null = null;
  private dummyBuf: GpuBufferLike | null = null;
  private uniformBuf: GpuBufferLike | null = null;
  private uniformCpu = new Uint32Array(UNIFORM_MAX_SLOTS * UNIFORM_SLOT_U32S);
  private _uniformBaseSlot = 0;

  // Layouts & Pipelines
  private pipelines: Record<string, unknown> = {};
  private bindGroups: Record<string, unknown> = {};

  // Execution & Context state
  private vramPlan: Gemma4KvMemoryPlan | null = null;
  private seqPos = 0;
  private contextTokenIds: number[] = [];
  private speculator = new SuffixSpeculator();
  private pleLru = new Map<number, Uint8Array>();
  private gpuPrefixSlots = new Map<
    string,
    {
      seqPos: number;
      contextTokenIds: number[];
      slidingSeqCount: number;
      slideBuf: GpuBufferLike;
      global0Buf: GpuBufferLike;
      global1Buf: GpuBufferLike;
      byteSize: number;
    }
  >();
  private lastDecodeStepGpuMs = 0;

  async load(options: Gemma4LoadOptions): Promise<Gemma4LoadStats> {
    const t0 = performance.now();
    this.kvMode = options.kvMode ?? 'int8';
    this.maxContext = options.maxContext ?? 131072;
    this.pacingMode = options.pacingMode ?? 'max-speed';
    this.displayHz = options.displayHz ?? 60;
    this.pleRowFetcher = options.pleRowFetcher;

    const baseUrl = options.baseUrl ?? '/models/gemma4-e4b/';
    const normBaseRaw = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    const originBase =
      typeof self !== 'undefined' && self.location?.href ? self.location.href : 'http://127.0.0.1/';
    const normBaseUrl = new URL(normBaseRaw, originBase).href;
    const manifestUrl = new URL('manifest.json', normBaseUrl).href;

    options.onProgress?.('Downloading manifest & WASM kernels', 0.05);
    const loadWasmDirect = async (): Promise<WasmEngineExports> => {
      if (options.wasmUrl) {
        const res = await fetch(options.wasmUrl);
        if (!res.ok) throw new Error(`Failed to fetch ${options.wasmUrl}: ${res.status}`);
        const inst = await WebAssembly.instantiate(await res.arrayBuffer(), {});
        return inst.instance.exports as unknown as WasmEngineExports;
      }
      return loadWgslWasm();
    };

    const [manifestRes, wasmExports, gpu] = await Promise.all([
      fetch(manifestUrl),
      loadWasmDirect(),
      acquireWgslDevice(),
    ]);
    if (!manifestRes.ok) {
      throw new Error(`Failed to fetch ${manifestUrl}: ${manifestRes.status}`);
    }
    const manifest = (await manifestRes.json()) as Gemma4Manifest;
    this.manifest = manifest;
    this.routes = buildGemma4DonorRoutes(
      manifest.textConfig.numLayers,
      new Set(
        manifest.textConfig.layerTypes
          .map((t, idx) => (t === 'full_attention' ? idx : -1))
          .filter((idx) => idx >= 0),
      ),
      manifest.textConfig.numLayers - manifest.textConfig.numKvSharedLayers,
    );
    this.device = gpu.device;

    // If options.wasmUrl was provided and default path worked differently, prefer options.wasmUrl
    let resolvedWasm = wasmExports as Gemma4WasmExports;
    if (options.wasmUrl && typeof resolvedWasm.g4_dequant_ple_rows !== 'function') {
      const res = await fetch(options.wasmUrl);
      const inst = await WebAssembly.instantiate(await res.arrayBuffer(), {});
      resolvedWasm = inst.instance.exports as unknown as Gemma4WasmExports;
    }
    if (typeof resolvedWasm.g4_dequant_ple_rows !== 'function') {
      throw new Error('WASM artifact missing g4_dequant_ple_rows; rebuild WASM with npm run build:wasm.');
    }
    this.wasm = resolvedWasm;

    this.hasSubgroups = gpu.hasSubgroups;
    this.hasDp4a = gpu.hasDp4a;
    this.subgroupMinSize = gpu.subgroupMinSize;
    this.computeTier = selectComputeTier([
      ...(gpu.hasShaderF16 ? ['shader-f16'] : []),
      ...(gpu.hasSubgroups ? ['subgroups'] : []),
    ]);
    this.useFusedQkv =
      (options.enableFusion ?? true) &&
      this.hasSubgroups &&
      Boolean(manifest.layerOffsets[0]?.qkv_proj);
    this.useSmallBatchSpec =
      (options.enableSmallBatchSpec ?? true) &&
      this.computeTier === 'tier1-f16-subgroups';
    const all16Aligned =
      manifest.slab0Bytes % 16 === 0 &&
      manifest.slab1Bytes % 16 === 0 &&
      manifest.slab2Bytes % 16 === 0 &&
      manifest.lmHeadLayout.qOffset % 16 === 0 &&
      manifest.lmHeadLayout.sOffset % 16 === 0;
    this.useVec4Gemv =
      (options.enableVec4Gemv ?? true) &&
      this.computeTier === 'tier1-f16-subgroups' &&
      all16Aligned;
    this.useDp4a = (options.enableDp4a ?? true) && this.hasDp4a && this.hasSubgroups;
    this.useFlashPrefill =
      (options.enableFlashPrefill ?? true) && this.computeTier === 'tier1-f16-subgroups';
    this.useGpuHotPle = (options.enableGpuHotPle ?? true) && Boolean(manifest.pleHot);
    this.useQ8PleProj =
      (options.enableQ8PleProj ?? true) &&
      this.useDp4a &&
      Boolean(manifest.auxLayout.per_layer_projection_q8);
    this.usePipelinedDecode = (options.enablePipelinedDecode ?? true) && this.useGpuHotPle;
    this.useW4A8Dp4a =
      (options.enableW4A8Dp4a ?? false) && this.useDp4a && this.useVec4Gemv;

    const pleHotBytes =
      this.useGpuHotPle && manifest.pleHot
        ? manifest.pleHot.dataBytes + manifest.pleHot.mapBytes
        : 0;
    const baseWeightBytes =
      manifest.slab0Bytes + manifest.slab1Bytes + manifest.slab2Bytes + manifest.auxBytes;
    const weightBytes = baseWeightBytes + pleHotBytes;
    const scratchBytes = this.allocateScratchAndUniformBuffers();
    const vramPlan = computeGemma4KvMemoryPlan(
      this.maxContext,
      this.kvMode,
      weightBytes,
      scratchBytes,
    );

    if (
      vramPlan.globalKvPerBufBytes > gpu.maxStorageBufferBindingSize ||
      manifest.slab0Bytes > gpu.maxStorageBufferBindingSize ||
      (this.useGpuHotPle &&
        manifest.pleHot &&
        manifest.pleHot.dataBytes > gpu.maxStorageBufferBindingSize)
    ) {
      throw new Error(
        `GPU maxStorageBufferBindingSize (${gpu.maxStorageBufferBindingSize}) insufficient for Gemma 4 buffer plan (${vramPlan.globalKvPerBufBytes} / ${manifest.slab0Bytes}).`,
      );
    }

    // Allocate weight & KV GPU buffers
    this.slab0 = gpu.device.createBuffer({
      size: manifest.slab0Bytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
    });
    this.slab1 = gpu.device.createBuffer({
      size: manifest.slab1Bytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
    });
    this.slab2 = gpu.device.createBuffer({
      size: manifest.slab2Bytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
    });
    this.auxBuf = gpu.device.createBuffer({
      size: manifest.auxBytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
    });
    if (this.useGpuHotPle && manifest.pleHot) {
      this.pleHotEntryCount = manifest.pleHot.entryCount;
      this.pleHotDataBuf = gpu.device.createBuffer({
        size: manifest.pleHot.dataBytes,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
      });
      this.pleHotMapBuf = gpu.device.createBuffer({
        size: manifest.pleHot.mapBytes,
        usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST,
      });
    } else {
      this.pleHotEntryCount = 0;
      this.pleHotDataBuf = null;
      this.pleHotMapBuf = null;
      this.pleHotMapCpu = null;
    }
    this.kvSliding = gpu.device.createBuffer({
      size: vramPlan.slidingKvBytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC,
    });
    this.kvGlobal0 = gpu.device.createBuffer({
      size: vramPlan.globalKvPerBufBytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC,
    });
    this.kvGlobal1 = gpu.device.createBuffer({
      size: vramPlan.globalKvPerBufBytes,
      usage: GPU_USAGE.STORAGE | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC,
    });

    // Stream shards into GPU weight buffers & compile pipelines concurrently where possible
    const tStream0 = performance.now();
    const pacingPolicy = getGpuPacingPolicy(this.pacingMode);
    const totalWeightBytes =
      baseWeightBytes +
      (this.useGpuHotPle && manifest.pleHot ? manifest.pleHot.dataBytes + manifest.pleHot.mapBytes : 0);
    let streamedSoFar = 0;

    const streamShards = async (parts: Gemma4ShardPart[], target: GpuBufferLike) => {
      let dstOff = 0;
      for (const part of parts) {
        const url = new URL(part.file, normBaseUrl).href;
        const baseStreamed = streamedSoFar;
        await streamUrlToGpuSegments(
          gpu.device,
          url,
          [{ buffer: target, dstByteOffset: dstOff, byteLength: part.bytes }],
          part.bytes,
          {
            yieldMs: pacingPolicy.loadChunkYieldMs,
            onProgress: (loaded) => {
              options.onProgress?.(
                `Streaming ${part.file}`,
                0.1 + 0.75 * ((baseStreamed + loaded) / totalWeightBytes),
              );
            },
          },
        );
        dstOff += part.bytes;
        streamedSoFar += part.bytes;
      }
    };

    const tPipe0 = performance.now();
    const compilePromise = this.compilePipelinesAndBindGroups();
    await streamShards(manifest.shards.aux, this.auxBuf);
    await streamShards(manifest.shards.slab0, this.slab0);
    await streamShards(manifest.shards.slab1, this.slab1);
    await streamShards(manifest.shards.slab2, this.slab2);
    if (this.useGpuHotPle && manifest.pleHot && this.pleHotDataBuf && this.pleHotMapBuf) {
      await streamShards(
        [{ file: manifest.pleHot.dataFile, bytes: manifest.pleHot.dataBytes }],
        this.pleHotDataBuf,
      );
      const mapUrl = new URL(manifest.pleHot.mapFile, normBaseUrl).href;
      const mapRes = await fetch(mapUrl);
      if (!mapRes.ok) {
        throw new Error(`Failed to fetch ${mapUrl}: ${mapRes.status}`);
      }
      const mapBuf = await mapRes.arrayBuffer();
      this.pleHotMapCpu = new Uint32Array(mapBuf);
      gpu.device.queue.writeBuffer(this.pleHotMapBuf, 0, mapBuf);
      streamedSoFar += manifest.pleHot.mapBytes;
    }
    const streamWeightsMs = Math.round(performance.now() - tStream0);

    await compilePromise;
    const pipelineCompileMs = Math.round(performance.now() - tPipe0);
    this.vramPlan = vramPlan;
    options.onProgress?.('Gemma 4 WGSL ready', 1.0);

    return {
      coldLoadMs: Math.round(performance.now() - t0),
      streamWeightsMs,
      pipelineCompileMs,
      computeTier: this.computeTier,
      hasShaderF16: gpu.hasShaderF16,
      hasSubgroups: gpu.hasSubgroups,
      hasDp4a: gpu.hasDp4a,
      maxBufferSize: gpu.maxBufferSize,
      maxStorageBufferBindingSize: gpu.maxStorageBufferBindingSize,
      kvMode: this.kvMode,
      maxContext: this.maxContext,
      vramPlan,
    };
  }

  private allocateScratchAndUniformBuffers(): number {
    const dev = this.device!;
    let total = 0;
    const mk = (bytes: number, usage: number): GpuBufferLike => {
      const aligned = Math.max(256, Math.ceil(bytes / 4) * 4);
      total += aligned;
      return dev.createBuffer({ size: aligned, usage });
    };

    const S = GPU_USAGE.STORAGE;
    this.tokenIdsBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * 4, S | GPU_USAGE.COPY_DST);
    this.hiddenBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_HIDDEN * 4, S | GPU_USAGE.COPY_SRC);
    this.normBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_HIDDEN * 4, S);
    this.qkvBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * 6144 * 4, S);
    this.qBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_ATTN_HEADS * GEMMA4_GLOBAL_HEAD_DIM * 4, S);
    this.kBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM * 4, S);
    this.vBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM * 4, S);
    this.attnOutBuf = mk(
      GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_ATTN_HEADS * GEMMA4_GLOBAL_HEAD_DIM * 4,
      S,
    );
    this.mlpGateUpBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_INTERMEDIATE * 2 * 4, S);
    this.mlpActBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_INTERMEDIATE * 4, S);
    this.pleRawBuf = mk(
      GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * 4,
      S | GPU_USAGE.COPY_DST,
    );
    this.pleProjBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * 4, S);
    this.pleCombinedBuf = mk(
      GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * 4,
      S,
    );
    this.pleActBuf = mk(GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_PLE_DIM * 4, S);
    this.logitsBuf = mk(
      GEMMA4_MAX_SPEC_TOKENS * GEMMA4_VOCAB_SIZE * 4,
      S | GPU_USAGE.COPY_SRC,
    );
    this.logitsSingleBuf = mk(
      GEMMA4_VOCAB_SIZE * 4,
      S | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC,
    );
    this.argmaxPartValBuf = mk(8 * 1024 * 4, S);
    this.argmaxPartIdxBuf = mk(8 * 1024 * 4, S);
    this.argmaxSingleOutBuf = mk(256, S | GPU_USAGE.COPY_SRC);
    this.argmaxMultiOutBuf = mk(256, S | GPU_USAGE.COPY_DST | GPU_USAGE.COPY_SRC);
    this.argmaxReadBuf = mk(256, GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST);
    this.argmaxReadBufB = mk(256, GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST);
    this.partialAttnBuf = mk(
      GEMMA4_ATTN_HEADS * GEMMA4_GLOBAL_SPLITS * 516 * 4,
      S,
    );
    this.partialPrefillBuf = mk(
      GEMMA4_PREFILL_MAX_CHUNK * GEMMA4_ATTN_HEADS * 8 * 516 * 4,
      S,
    );
    this.dummyBuf = mk(256, S);
    this.uniformBuf = mk(UNIFORM_MAX_SLOTS * UNIFORM_SLOT_BYTES, GPU_USAGE.UNIFORM | GPU_USAGE.COPY_DST);
    return total;
  }

  private async compilePipelinesAndBindGroups(): Promise<void> {
    const dev = this.device!;
    const wasm = this.wasm!;
    const tier1 = this.computeTier === 'tier1-f16-subgroups';
    const tierF16 = this.computeTier === 'tier1-f16-subgroups' || this.computeTier === 'tier2-f16-shared';

    const makeLayout = (entries: Array<{ binding: number; type: 'uniform' | 'read-only-storage' | 'storage' }>) => {
      const bgl = dev.createBindGroupLayout({
        entries: entries.map((e) => ({
          binding: e.binding,
          visibility: GPU_SHADER_COMPUTE,
          buffer:
            e.type === 'uniform'
              ? { type: 'uniform', hasDynamicOffset: true }
              : { type: e.type },
        })),
      });
      return { bgl, pl: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }) };
    };

    const lay4 = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
    ]);
    const lay5 = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
      { binding: 4, type: 'read-only-storage' },
    ]);
    const layPleNorm = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'storage' },
    ]);
    const layElem = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
    ]);
    const layRope = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'read-only-storage' },
      { binding: 5, type: 'storage' },
    ]);
    const layFusedNorm = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
      { binding: 3, type: 'storage' },
    ]);
    const layRopeSg = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'storage' },
    ]);
    const layArg1 = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
      { binding: 3, type: 'storage' },
    ]);
    const layArg2 = makeLayout([
      { binding: 0, type: 'uniform' },
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
    ]);

    const useSg = this.hasSubgroups;
    const gemvQ4Id = tier1 ? 100 : 101;
    const gemmQ4Id = tierF16 ? 102 : 103;
    const gegluQ4Id = tier1 ? 104 : 105;
    const lmHeadQ2Id = tier1 ? 107 : 108;

    const specs: Array<[string, number, unknown]> = [
      ['q4Gemv', gemvQ4Id, lay4.pl],
      ['q4Gemm', gemmQ4Id, lay4.pl],
      ['q4Geglu', gegluQ4Id, lay4.pl],
      ['q8Gemv', useSg ? 126 : 106, lay5.pl],
      ['q2LmHead', lmHeadQ2Id, lay4.pl],
      ['q2Embed', 109, lay4.pl],
      ['pleProj', 110, lay4.pl],
      ['pleNorm', 111, layPleNorm.pl],
      ['rmsNorm', useSg ? 125 : 112, lay4.pl],
      ...(useSg ? ([['rmsFused', 127, layFusedNorm.pl]] as Array<[string, number, unknown]>) : []),
      ...(this.useFusedQkv ? ([['ropeKvSg', 128, layRopeSg.pl]] as Array<[string, number, unknown]>) : []),
      ...(this.useSmallBatchSpec
        ? ([
            ['q4GemvBatch', 129, lay4.pl],
            ['q4GegluBatch', 130, lay4.pl],
            ['q2LmHeadBatch', 131, lay4.pl],
            ['argmaxBatch1', 132, layArg1.pl],
            ['argmaxBatch2', 133, layArg2.pl],
          ] as Array<[string, number, unknown]>)
        : []),
      ['gegluBatch', 113, layElem.pl],
      ['ropeKv', 114, layRope.pl],
      ['gqaDecSlide', useSg ? 120 : 115, lay4.pl],
      ...(useSg ? ([['gqaDecSlideReduce', 121, layElem.pl]] as Array<[string, number, unknown]>) : []),
      ['gqaDecGlobal', useSg ? 122 : 116, lay4.pl],
      ['gqaDecReduce', 117, layElem.pl],
      ['gqaPrefSlide', useSg ? 123 : 118, lay4.pl],
      ['gqaPrefGlobal', useSg ? 124 : 119, lay4.pl],
      ...(this.useVec4Gemv
        ? ([
            ['q4GemvVec4', 134, lay4.pl],
            ['q4GegluVec4', 135, lay4.pl],
            ['q4GemvWideKVec4', 136, lay4.pl],
          ] as Array<[string, number, unknown]>)
        : []),
      ...(this.hasSubgroups && this.subgroupMinSize >= 32
        ? ([
            ['gqaDecSlideWarp', 148, lay4.pl],
            ...(this.kvMode === 'int4'
              ? ([['gqaDecGlobalInt4', 146, lay4.pl]] as Array<[string, number, unknown]>)
              : []),
            ...(this.useDp4a && this.kvMode === 'int8'
              ? ([['gqaDecGlobalWarp', 149, lay4.pl]] as Array<[string, number, unknown]>)
              : []),
          ] as Array<[string, number, unknown]>)
        : []),
      ...(this.useDp4a
        ? ([
            ['q8GemvDp4a', 139, lay5.pl],
            ...(this.useW4A8Dp4a
              ? ([
                  ['q4GemvDp4a', 144, lay4.pl],
                  ['q2LmHeadDp4a', 145, lay4.pl],
                ] as Array<[string, number, unknown]>)
              : []),
            ...(this.kvMode === 'int8'
              ? ([['gqaDecGlobalDp4a', 138, lay4.pl]] as Array<[string, number, unknown]>)
              : []),
          ] as Array<[string, number, unknown]>)
        : []),
      ...(this.useGpuHotPle && this.pleHotDataBuf && this.pleHotMapBuf
        ? ([['pleHotDequant', 143, lay5.pl]] as Array<[string, number, unknown]>)
        : []),
      ...(this.useFlashPrefill
        ? ([
            ['gqaPrefSlideFlash', 141, lay4.pl],
            ...(this.kvMode === 'int4'
              ? ([
                  ['gqaPrefGlobalFlashInt4', 147, lay4.pl],
                  ['gqaPrefReduce', 142, layElem.pl],
                ] as Array<[string, number, unknown]>)
              : this.useDp4a && this.kvMode === 'int8'
                ? ([
                    ['gqaPrefGlobalFlash', 140, lay4.pl],
                    ['gqaPrefReduce', 142, layElem.pl],
                  ] as Array<[string, number, unknown]>)
                : []),
          ] as Array<[string, number, unknown]>)
        : []),
      ['argmax1', 7, layArg1.pl],
      ['argmax2', 8, layArg2.pl],
    ];

    const compiled = await Promise.all(
      specs.map(async ([name, sid, pl]) => {
        const src = getWgslShaderSource(wasm, sid);
        const pipeline = await createValidatedPipeline(dev, `g4_${name}_${sid}`, src, pl);
        return [name, pipeline] as const;
      }),
    );
    for (const [name, p] of compiled) {
      this.pipelines[name] = p;
    }

    const bg4 = (bgl: unknown, b1: GpuBufferLike, b2: GpuBufferLike, b3: GpuBufferLike, uSize = 64) =>
      dev.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: uSize } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
        ],
      });

    const bg5 = (
      bgl: unknown,
      b1: GpuBufferLike,
      b2: GpuBufferLike,
      b3: GpuBufferLike,
      b4: GpuBufferLike,
      uSize = 64,
    ) =>
      dev.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: uSize } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
          { binding: 4, resource: { buffer: b4 } },
        ],
      });

    const bg3 = (bgl: unknown, b1: GpuBufferLike, b2: GpuBufferLike, uSize = 64) =>
      dev.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: uSize } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
        ],
      });

    const bgRope = (kv: GpuBufferLike) =>
      dev.createBindGroup({
        layout: layRope.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: 64 } },
          { binding: 1, resource: { buffer: this.auxBuf! } },
          { binding: 2, resource: { buffer: this.qBuf! } },
          { binding: 3, resource: { buffer: this.kBuf! } },
          { binding: 4, resource: { buffer: this.vBuf! } },
          { binding: 5, resource: { buffer: kv } },
        ],
      });

    const bgRopeSg = (kv: GpuBufferLike) =>
      dev.createBindGroup({
        layout: layRopeSg.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: 64 } },
          { binding: 1, resource: { buffer: this.auxBuf! } },
          { binding: 2, resource: { buffer: this.qBuf! } },
          { binding: 3, resource: { buffer: this.qkvBuf! } },
          { binding: 4, resource: { buffer: kv } },
        ],
      });

    // Embed / PLE / Norm / LM Head / Argmax bindGroups
    this.bindGroups = {
      embed: bg4(lay4.bgl, this.slab2!, this.tokenIdsBuf!, this.hiddenBuf!),
      pleProj: bg4(lay4.bgl, this.auxBuf!, this.hiddenBuf!, this.pleProjBuf!),
      ...(this.useQ8PleProj
        ? { pleProjQ8: bg5(lay5.bgl, this.auxBuf!, this.hiddenBuf!, this.pleProjBuf!, this.dummyBuf!) }
        : {}),
      ...(this.pleHotDataBuf && this.pleHotMapBuf
        ? {
            pleHotDequant: bg5(
              lay5.bgl,
              this.tokenIdsBuf!,
              this.pleHotMapBuf,
              this.pleRawBuf!,
              this.pleHotDataBuf,
            ),
          }
        : {}),
      pleNorm: bg5(layPleNorm.bgl, this.auxBuf!, this.pleProjBuf!, this.pleRawBuf!, this.pleCombinedBuf!),
      rmsHiddenToNorm: bg4(lay4.bgl, this.auxBuf!, this.hiddenBuf!, this.normBuf!),
      rmsNormToHidden: bg4(lay4.bgl, this.auxBuf!, this.normBuf!, this.hiddenBuf!),
      rmsFused: bg4(layFusedNorm.bgl, this.auxBuf!, this.hiddenBuf!, this.normBuf!),
      gegluBatch: bg3(layElem.bgl, this.mlpGateUpBuf!, this.mlpActBuf!),
      gqaReduce: bg3(layElem.bgl, this.partialAttnBuf!, this.attnOutBuf!),
      lmHead: bg4(lay4.bgl, this.slab2!, this.normBuf!, this.logitsBuf!),
      argmax1Main: bg4(layArg1.bgl, this.logitsBuf!, this.argmaxPartValBuf!, this.argmaxPartIdxBuf!),
      argmax1Single: bg4(layArg1.bgl, this.logitsSingleBuf!, this.argmaxPartValBuf!, this.argmaxPartIdxBuf!),
      argmax2: dev.createBindGroup({
        layout: layArg2.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: 64 } },
          { binding: 1, resource: { buffer: this.argmaxPartValBuf! } },
          { binding: 2, resource: { buffer: this.argmaxPartIdxBuf! } },
          { binding: 3, resource: { buffer: this.argmaxSingleOutBuf! } },
        ],
      }),
      argmaxBatch2: dev.createBindGroup({
        layout: layArg2.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf!, offset: 0, size: 64 } },
          { binding: 1, resource: { buffer: this.argmaxPartValBuf! } },
          { binding: 2, resource: { buffer: this.argmaxPartIdxBuf! } },
          { binding: 3, resource: { buffer: this.argmaxMultiOutBuf! } },
        ],
      }),
      ropeSlide: bgRope(this.kvSliding!),
      ropeGlobal0: bgRope(this.kvGlobal0!),
      ropeGlobal1: bgRope(this.kvGlobal1!),
      ropeSgSlide: bgRopeSg(this.kvSliding!),
      ropeSgGlobal0: bgRopeSg(this.kvGlobal0!),
      ropeSgGlobal1: bgRopeSg(this.kvGlobal1!),
      gqaSlide: bg4(lay4.bgl, this.qBuf!, this.kvSliding!, this.attnOutBuf!),
      gqaSlideSplit: bg4(lay4.bgl, this.qBuf!, this.kvSliding!, this.partialAttnBuf!),
      gqaGlobal0: bg4(lay4.bgl, this.qBuf!, this.kvGlobal0!, this.partialAttnBuf!),
      gqaGlobal1: bg4(lay4.bgl, this.qBuf!, this.kvGlobal1!, this.partialAttnBuf!),
      gqaPrefGlobal0: bg4(lay4.bgl, this.qBuf!, this.kvGlobal0!, this.attnOutBuf!),
      gqaPrefGlobal1: bg4(lay4.bgl, this.qBuf!, this.kvGlobal1!, this.attnOutBuf!),
      gqaPrefFlashGlobal0: bg4(lay4.bgl, this.qBuf!, this.kvGlobal0!, this.partialPrefillBuf!),
      gqaPrefFlashGlobal1: bg4(lay4.bgl, this.qBuf!, this.kvGlobal1!, this.partialPrefillBuf!),
      gqaPrefReduce: bg3(layElem.bgl, this.partialPrefillBuf!, this.attnOutBuf!),
    };

    // Per-slab bindGroups (`slab0` and `slab1`) for every projection buffer target
    for (let sIdx = 0; sIdx < 2; sIdx++) {
      const slab = sIdx === 0 ? this.slab0! : this.slab1!;
      this.bindGroups[`normToQkv_${sIdx}`] = bg4(lay4.bgl, slab, this.normBuf!, this.qkvBuf!);
      this.bindGroups[`normToQ_${sIdx}`] = bg4(lay4.bgl, slab, this.normBuf!, this.qBuf!);
      this.bindGroups[`normToK_${sIdx}`] = bg4(lay4.bgl, slab, this.normBuf!, this.kBuf!);
      this.bindGroups[`normToV_${sIdx}`] = bg4(lay4.bgl, slab, this.normBuf!, this.vBuf!);
      this.bindGroups[`attnToNorm_${sIdx}`] = bg4(lay4.bgl, slab, this.attnOutBuf!, this.normBuf!);
      this.bindGroups[`normToMlpAct_${sIdx}`] = bg4(lay4.bgl, slab, this.normBuf!, this.mlpActBuf!);
      this.bindGroups[`normToGateUp_${sIdx}`] = bg4(lay4.bgl, slab, this.normBuf!, this.mlpGateUpBuf!);
      this.bindGroups[`mlpActToNorm_${sIdx}`] = bg4(lay4.bgl, slab, this.mlpActBuf!, this.normBuf!);
      this.bindGroups[`q8PleGate_${sIdx}`] = bg5(
        lay5.bgl,
        slab,
        this.hiddenBuf!,
        this.pleActBuf!,
        this.pleCombinedBuf!,
      );
      this.bindGroups[`q8PleProj_${sIdx}`] = bg5(
        lay5.bgl,
        slab,
        this.pleActBuf!,
        this.normBuf!,
        this.dummyBuf!,
      );
    }
  }

  resetContext(): void {
    this.seqPos = 0;
    this.contextTokenIds = [];
    this.speculator.reset();
    this.lastDecodeStepGpuMs = 0;
  }

  getSeqPos(): number {
    return this.seqPos;
  }

  getMaxContext(): number {
    return this.maxContext;
  }

  getKvMode(): WgslKvMode {
    return this.kvMode;
  }

  getComputeTier(): WgslComputeTier {
    return this.computeTier;
  }

  getVramPlan(): Gemma4KvMemoryPlan | null {
    return this.vramPlan;
  }

  getLastDecodeGpuMs(): number {
    return this.lastDecodeStepGpuMs;
  }

  setPacingMode(mode: GpuPacingMode): void {
    this.pacingMode = mode;
  }

  setDisplayHz(hz: number): void {
    const safe = Number.isFinite(hz) && hz >= 30 && hz <= 360 ? Math.round(hz) : 60;
    this.displayHz = safe;
  }

  getHasDp4a(): boolean {
    return this.hasDp4a;
  }

  async readLastLogits(): Promise<Float32Array> {
    if (!this.device || !this.logitsBuf) {
      throw new Error('Gemma 4 engine is not loaded');
    }
    return readGpuF32Slice(this.device, this.logitsBuf, 0, GEMMA4_VOCAB_SIZE * 4);
  }

  commitSpeculativeTokens(acceptedSeq: readonly number[]): void {
    for (const id of acceptedSeq) {
      this.contextTokenIds.push(id);
      this.speculator.pushToken(id);
    }
    this.seqPos += acceptedSeq.length;
  }

  private getGlobalU32sPerTokHead(): number {
    return this.kvMode === 'int4' ? 68 : this.kvMode === 'int8' ? 129 : 256;
  }

  async saveGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; byteSize: number }> {
    const dev = this.device!;
    if (!dev || !this.kvSliding || !this.kvGlobal0 || !this.kvGlobal1) {
      throw new Error('Gemma 4 GPU buffers not initialized');
    }
    this.deleteGpuPrefixSlot(slotId);

    const pos = this.seqPos;
    const slidingPlaneStride = GEMMA4_SLIDING_RING_SLOTS * 128 * 4;
    const fullSliding = pos >= GEMMA4_SLIDING_RING_SLOTS;
    const slidingSeqCount = fullSliding ? GEMMA4_SLIDING_RING_SLOTS : pos;
    const slidingPlaneBytes = slidingSeqCount * 128 * 4;
    const slideTotalBytes = Math.max(256, fullSliding ? 80 * slidingPlaneStride : 80 * slidingPlaneBytes);

    const globalU32s = this.getGlobalU32sPerTokHead();
    const globalPlaneStride = this.maxContext * globalU32s * 4;
    const globalPlaneBytes = pos * globalU32s * 4;
    const globalTotalBytes = Math.max(256, 8 * globalPlaneBytes);

    const usage = GPU_USAGE.STORAGE | GPU_USAGE.COPY_SRC | GPU_USAGE.COPY_DST;
    const slideBuf = dev.createBuffer({ size: slideTotalBytes, usage });
    const global0Buf = dev.createBuffer({ size: globalTotalBytes, usage });
    const global1Buf = dev.createBuffer({ size: globalTotalBytes, usage });

    if (pos > 0) {
      const enc = dev.createCommandEncoder();
      if (fullSliding) {
        enc.copyBufferToBuffer(this.kvSliding, 0, slideBuf, 0, slideTotalBytes);
      } else {
        for (let p = 0; p < 80; p++) {
          enc.copyBufferToBuffer(
            this.kvSliding,
            p * slidingPlaneStride,
            slideBuf,
            p * slidingPlaneBytes,
            slidingPlaneBytes,
          );
        }
      }
      for (let p = 0; p < 8; p++) {
        enc.copyBufferToBuffer(
          this.kvGlobal0,
          p * globalPlaneStride,
          global0Buf,
          p * globalPlaneBytes,
          globalPlaneBytes,
        );
        enc.copyBufferToBuffer(
          this.kvGlobal1,
          p * globalPlaneStride,
          global1Buf,
          p * globalPlaneBytes,
          globalPlaneBytes,
        );
      }
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
    }

    const byteSize = slideTotalBytes + globalTotalBytes * 2;
    this.gpuPrefixSlots.set(slotId, {
      seqPos: pos,
      contextTokenIds: [...this.contextTokenIds],
      slidingSeqCount,
      slideBuf,
      global0Buf,
      global1Buf,
      byteSize,
    });
    return { seqPos: pos, byteSize };
  }

  async restoreGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; restoreMs: number }> {
    const slot = this.gpuPrefixSlots.get(slotId);
    if (!slot) {
      throw new Error(`Unknown Gemma 4 GPU prefix slot "${slotId}"`);
    }
    const dev = this.device!;
    if (!dev || !this.kvSliding || !this.kvGlobal0 || !this.kvGlobal1) {
      throw new Error('Gemma 4 GPU buffers not initialized');
    }
    const t0 = performance.now();
    if (slot.seqPos > 0) {
      const enc = dev.createCommandEncoder();
      const slidingPlaneStride = GEMMA4_SLIDING_RING_SLOTS * 128 * 4;
      const fullSliding = slot.slidingSeqCount >= GEMMA4_SLIDING_RING_SLOTS;
      const slidingPlaneBytes = slot.slidingSeqCount * 128 * 4;
      if (fullSliding) {
        enc.copyBufferToBuffer(slot.slideBuf, 0, this.kvSliding, 0, 80 * slidingPlaneStride);
      } else {
        for (let p = 0; p < 80; p++) {
          enc.copyBufferToBuffer(
            slot.slideBuf,
            p * slidingPlaneBytes,
            this.kvSliding,
            p * slidingPlaneStride,
            slidingPlaneBytes,
          );
        }
      }
      const globalU32s = this.getGlobalU32sPerTokHead();
      const globalPlaneStride = this.maxContext * globalU32s * 4;
      const globalPlaneBytes = slot.seqPos * globalU32s * 4;
      for (let p = 0; p < 8; p++) {
        enc.copyBufferToBuffer(
          slot.global0Buf,
          p * globalPlaneBytes,
          this.kvGlobal0,
          p * globalPlaneStride,
          globalPlaneBytes,
        );
        enc.copyBufferToBuffer(
          slot.global1Buf,
          p * globalPlaneBytes,
          this.kvGlobal1,
          p * globalPlaneStride,
          globalPlaneBytes,
        );
      }
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
    }
    this.seqPos = slot.seqPos;
    this.contextTokenIds = [...slot.contextTokenIds];
    this.speculator = new SuffixSpeculator(this.contextTokenIds);
    this.lastDecodeStepGpuMs = 0;
    return { seqPos: slot.seqPos, restoreMs: performance.now() - t0 };
  }

  deleteGpuPrefixSlot(slotId: string): boolean {
    const slot = this.gpuPrefixSlots.get(slotId);
    if (!slot) {
      return false;
    }
    this.gpuPrefixSlots.delete(slotId);
    for (const buf of [slot.slideBuf, slot.global0Buf, slot.global1Buf]) {
      try {
        buf.destroy();
      } catch {}
    }
    return true;
  }

  private async readBufferRangeBytes(
    src: GpuBufferLike,
    offset: number,
    bytes: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    if (bytes <= 0) {
      return new Uint8Array(0);
    }
    const dev = this.device!;
    const staging = dev.createBuffer({
      size: bytes,
      usage: GPU_USAGE.MAP_READ | GPU_USAGE.COPY_DST,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, offset, staging, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPU_MAP_READ, 0, bytes);
    const out = new Uint8Array(staging.getMappedRange(0, bytes)).slice();
    staging.unmap();
    try {
      staging.destroy();
    } catch {}
    return out;
  }

  async snapshotState(): Promise<Gemma4CpuSnapshot> {
    if (!this.device || !this.kvSliding || !this.kvGlobal0 || !this.kvGlobal1) {
      throw new Error('Gemma 4 engine is not loaded');
    }
    const pos = this.seqPos;
    const slidingPlaneStride = GEMMA4_SLIDING_RING_SLOTS * 128 * 4;
    const fullSliding = pos >= GEMMA4_SLIDING_RING_SLOTS;
    const slidingSeqCount = fullSliding ? GEMMA4_SLIDING_RING_SLOTS : pos;
    const slidingPlaneBytes = slidingSeqCount * 128 * 4;
    const slidingBytes = new Uint8Array(80 * slidingPlaneBytes);

    if (pos > 0) {
      if (fullSliding) {
        slidingBytes.set(await this.readBufferRangeBytes(this.kvSliding, 0, slidingBytes.byteLength));
      } else {
        for (let p = 0; p < 80; p++) {
          const part = await this.readBufferRangeBytes(
            this.kvSliding,
            p * slidingPlaneStride,
            slidingPlaneBytes,
          );
          slidingBytes.set(part, p * slidingPlaneBytes);
        }
      }
    }

    const globalU32s = this.getGlobalU32sPerTokHead();
    const globalPlaneStride = this.maxContext * globalU32s * 4;
    const globalPlaneBytes = pos * globalU32s * 4;
    const global0Bytes = new Uint8Array(8 * globalPlaneBytes);
    const global1Bytes = new Uint8Array(8 * globalPlaneBytes);
    if (pos > 0) {
      for (let p = 0; p < 8; p++) {
        global0Bytes.set(
          await this.readBufferRangeBytes(this.kvGlobal0, p * globalPlaneStride, globalPlaneBytes),
          p * globalPlaneBytes,
        );
        global1Bytes.set(
          await this.readBufferRangeBytes(this.kvGlobal1, p * globalPlaneStride, globalPlaneBytes),
          p * globalPlaneBytes,
        );
      }
    }

    return {
      seqPos: pos,
      contextTokenIds: [...this.contextTokenIds],
      kvMode: this.kvMode,
      maxContext: this.maxContext,
      slidingSeqCount,
      slidingBytes,
      global0Bytes,
      global1Bytes,
    };
  }

  async restoreState(snap: Gemma4CpuSnapshot): Promise<void> {
    if (!this.device || !this.kvSliding || !this.kvGlobal0 || !this.kvGlobal1) {
      throw new Error('Gemma 4 engine is not loaded');
    }
    if (snap.kvMode !== this.kvMode || snap.maxContext !== this.maxContext) {
      throw new Error('Snapshot geometry mismatch with active Gemma 4 session.');
    }
    const dev = this.device;
    if (snap.seqPos > 0) {
      const slidingPlaneStride = GEMMA4_SLIDING_RING_SLOTS * 128 * 4;
      const slidingPlaneBytes = snap.slidingSeqCount * 128 * 4;
      if (snap.slidingSeqCount >= GEMMA4_SLIDING_RING_SLOTS) {
        dev.queue.writeBuffer(this.kvSliding, 0, snap.slidingBytes);
      } else {
        for (let p = 0; p < 80; p++) {
          dev.queue.writeBuffer(
            this.kvSliding,
            p * slidingPlaneStride,
            snap.slidingBytes,
            p * slidingPlaneBytes,
            slidingPlaneBytes,
          );
        }
      }
      const globalU32s = this.getGlobalU32sPerTokHead();
      const globalPlaneStride = this.maxContext * globalU32s * 4;
      const globalPlaneBytes = snap.seqPos * globalU32s * 4;
      for (let p = 0; p < 8; p++) {
        dev.queue.writeBuffer(
          this.kvGlobal0,
          p * globalPlaneStride,
          snap.global0Bytes,
          p * globalPlaneBytes,
          globalPlaneBytes,
        );
        dev.queue.writeBuffer(
          this.kvGlobal1,
          p * globalPlaneStride,
          snap.global1Bytes,
          p * globalPlaneBytes,
          globalPlaneBytes,
        );
      }
      await dev.queue.onSubmittedWorkDone();
    }
    this.seqPos = snap.seqPos;
    this.contextTokenIds = [...snap.contextTokenIds];
    this.speculator = new SuffixSpeculator(this.contextTokenIds);
    this.lastDecodeStepGpuMs = 0;
  }

  private async preparePleToGpu(tokenIds: readonly number[]): Promise<{ fetchMs: number; dequantMs: number }> {
    const T = tokenIds.length;
    const tFetch0 = performance.now();
    const safeIds = new Array<number>(T);
    const missing: number[] = [];
    const seenMissing = new Set<number>();

    for (let i = 0; i < T; i++) {
      const sid = normalizeGemma4PleTokenId(tokenIds[i]);
      safeIds[i] = sid;
      if (!this.pleLru.has(sid) && !seenMissing.has(sid)) {
        seenMissing.add(sid);
        missing.push(sid);
      }
    }

    if (missing.length > 0) {
      if (!this.pleRowFetcher) {
        throw new Error('pleRowFetcher is required for Gemma4WgslEngine');
      }
      const packed = await this.pleRowFetcher(missing);
      if (packed.byteLength !== missing.length * GEMMA4_PLE_ROW_BYTES) {
        throw new Error(
          `pleRowFetcher returned ${packed.byteLength} bytes, expected ${missing.length * GEMMA4_PLE_ROW_BYTES}`,
        );
      }
      for (let i = 0; i < missing.length; i++) {
        const rowCopy = packed.slice(i * GEMMA4_PLE_ROW_BYTES, (i + 1) * GEMMA4_PLE_ROW_BYTES);
        if (this.pleLru.size >= PLE_LRU_MAX_ENTRIES) {
          const oldestKey = this.pleLru.keys().next().value;
          if (oldestKey !== undefined) this.pleLru.delete(oldestKey);
        }
        this.pleLru.set(missing[i], rowCopy);
      }
    }
    const fetchMs = performance.now() - tFetch0;

    const tDq0 = performance.now();
    const wasm = this.wasm!;
    const scratchPtr = wasm.wasm_scratch_ptr();
    const rawPtr = scratchPtr;
    const outPtr = scratchPtr + 262144; // 256 KiB room for up to 94 raw rows; 64 rows = 177 KB
    const memU8 = new Uint8Array(wasm.memory.buffer, rawPtr, T * GEMMA4_PLE_ROW_BYTES);
    for (let i = 0; i < T; i++) {
      const sid = safeIds[i];
      const row = this.pleLru.get(sid)!;
      // Touch LRU order
      this.pleLru.delete(sid);
      this.pleLru.set(sid, row);
      memU8.set(row, i * GEMMA4_PLE_ROW_BYTES);
    }

    wasm.g4_dequant_ple_rows(T, rawPtr, outPtr);
    const f32Len = T * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
    const f32View = new Float32Array(wasm.memory.buffer, outPtr, f32Len);
    this.device!.queue.writeBuffer(this.pleRawBuf!, 0, f32View);
    const dequantMs = performance.now() - tDq0;

    return { fetchMs, dequantMs };
  }

  private isHotPleToken(rawId: number): boolean {
    if (!this.useGpuHotPle || !this.pleHotMapCpu || this.pleHotEntryCount <= 0) {
      return false;
    }
    const sid = normalizeGemma4PleTokenId(rawId);
    return (this.pleHotMapCpu[sid] ?? 0xffffffff) < this.pleHotEntryCount;
  }

  private areAllHotPleTokens(tokenIds: readonly number[]): boolean {
    if (!this.useGpuHotPle || !this.pipelines.pleHotDequant || !this.pleHotMapCpu || tokenIds.length === 0) {
      return false;
    }
    for (let i = 0; i < tokenIds.length; i++) {
      if (!this.isHotPleToken(tokenIds[i]!)) {
        return false;
      }
    }
    return true;
  }

  private async preparePleOrUseGpuHot(
    tokenIds: readonly number[],
  ): Promise<{ usedGpuHot: boolean; fetchMs: number; dequantMs: number }> {
    if (this.areAllHotPleTokens(tokenIds)) {
      return { usedGpuHot: true, fetchMs: 0, dequantMs: 0 };
    }
    const res = await this.preparePleToGpu(tokenIds);
    return { usedGpuHot: false, ...res };
  }

  private pushUniformSlot(values: readonly number[]): number {
    if (this._uniformCursor >= UNIFORM_BANK_SLOTS) {
      throw new Error(`Exceeded UNIFORM_BANK_SLOTS (${UNIFORM_BANK_SLOTS})`);
    }
    const slot = this._uniformBaseSlot + this._uniformCursor;
    const base = slot * UNIFORM_SLOT_U32S;
    for (let i = 0; i < values.length; i++) {
      this.uniformCpu[base + i] = values[i] >>> 0;
    }
    this._uniformCursor += 1;
    return slot * UNIFORM_SLOT_BYTES;
  }

  private writeUniformBankToGpu(bank: 0 | 1 = 0): void {
    const startU32 = bank * UNIFORM_BANK_SLOTS * UNIFORM_SLOT_U32S;
    const countU32 = this._uniformCursor * UNIFORM_SLOT_U32S;
    this.device!.queue.writeBuffer(
      this.uniformBuf!,
      bank * UNIFORM_BANK_SLOTS * UNIFORM_SLOT_BYTES,
      this.uniformCpu,
      startU32,
      countU32,
    );
  }
  private _uniformCursor = 0;

  private buildForwardSteps(
    tokens: number,
    startPos: number,
    logitsTokens = 1,
    opts?: {
      useGpuHotPle?: boolean;
      maxLayerCount?: number;
      uniformBank?: 0 | 1;
    },
  ): {
    steps: DispatchStep[];
    layerEndIndices: number[];
    logitsTokenOffset: number;
  } {
    this._uniformBaseSlot = (opts?.uniformBank ?? 0) * UNIFORM_BANK_SLOTS;
    this._uniformCursor = 0;
    const activeLayers = Math.min(
      GEMMA4_NUM_LAYERS,
      Math.max(1, opts?.maxLayerCount ?? GEMMA4_NUM_LAYERS),
    );
    const m = this.manifest!;
    const steps: DispatchStep[] = [];
    const layerEndIndices: number[] = [];
    const isDecode = tokens === 1;

    const add = (pipeline: unknown, bindGroup: unknown, uniforms: number[], wx: number, wy = 1, wz = 1) => {
      const uniformByteOffset = this.pushUniformSlot(uniforms);
      steps.push({ pipeline, bindGroup, uniformByteOffset, wx, wy, wz });
    };

    const addQ4 = (
      bindKey: string,
      pair: Gemma4QuantPair,
      rows: number,
      cols: number,
      tokCount: number,
      mode = 0,
    ) => {
      const qU32 = pair.qOffset >>> 2;
      const sU32 = pair.sOffset >>> 2;
      if (tokCount === 1) {
        if (this.useW4A8Dp4a && this.pipelines.q4GemvDp4a) {
          add(
            this.pipelines.q4GemvDp4a,
            this.bindGroups[bindKey],
            [rows, cols, pair.qOffset >>> 4, pair.sOffset >>> 4, 0, 0, mode, 0],
            Math.ceil(rows / 4),
            1,
            1,
          );
        } else if (this.useVec4Gemv && this.pipelines.q4GemvVec4) {
          const qVec4 = pair.qOffset >>> 4;
          const sVec4 = pair.sOffset >>> 4;
          const wideK = cols > 2560;
          add(
            wideK ? this.pipelines.q4GemvWideKVec4 : this.pipelines.q4GemvVec4,
            this.bindGroups[bindKey],
            [rows, cols, qVec4, sVec4, 0, 0, mode, 0],
            Math.ceil(rows / 4),
            1,
            1,
          );
        } else {
          add(
            this.pipelines.q4Gemv,
            this.bindGroups[bindKey],
            [rows, cols, qU32, sU32, 0, 0, mode, 0],
            Math.ceil(rows / 16),
            1,
            1,
          );
        }
      } else if (tokCount <= 8 && this.useSmallBatchSpec) {
        add(
          this.pipelines.q4GemvBatch,
          this.bindGroups[bindKey],
          [rows, cols, qU32, sU32, 0, 0, mode, tokCount],
          Math.ceil(rows / 16),
          1,
          1,
        );
      } else {
        add(
          this.pipelines.q4Gemm,
          this.bindGroups[bindKey],
          [tokCount, rows, cols, qU32, sU32, 0, 0, mode, 0, 0, 0, 0],
          Math.ceil(rows / 64),
          Math.ceil(tokCount / 64),
          1,
        );
      }
    };

    const fusedNormsActive = this.hasSubgroups && Boolean(this.pipelines.rmsFused);

    // 1. Embedding + Hot PLE GPU dequant (if hot) + PLE projection + combine
    add(
      this.pipelines.q2Embed,
      this.bindGroups.embed,
      [tokens, GEMMA4_HIDDEN, m.lmHeadLayout.qOffset >>> 2, m.lmHeadLayout.sOffset >>> 2],
      tokens,
      1,
      1,
    );
    if (opts?.useGpuHotPle && this.pipelines.pleHotDequant && this.bindGroups.pleHotDequant) {
      add(
        this.pipelines.pleHotDequant,
        this.bindGroups.pleHotDequant,
        [tokens, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM, this.pleHotEntryCount],
        GEMMA4_NUM_LAYERS,
        tokens,
        1,
      );
    }
    if (
      this.useQ8PleProj &&
      this.pipelines.q8GemvDp4a &&
      this.bindGroups.pleProjQ8 &&
      m.auxLayout.per_layer_projection_q8
    ) {
      const pq8 = m.auxLayout.per_layer_projection_q8;
      add(
        this.pipelines.q8GemvDp4a,
        this.bindGroups.pleProjQ8,
        [pq8.rows, pq8.cols, pq8.qOffset >>> 2, pq8.sOffset >>> 2, 0, 0, 2, 0],
        Math.ceil(pq8.rows / 8),
        tokens,
        1,
      );
    } else {
      add(
        this.pipelines.pleProj,
        this.bindGroups.pleProj,
        [
          tokens,
          GEMMA4_HIDDEN,
          GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM,
          m.auxLayout.per_layer_projection_weight.offset >>> 2,
        ],
        84, // 10752 / 128
        tokens,
        1,
      );
    }
    add(
      this.pipelines.pleNorm,
      this.bindGroups.pleNorm,
      [
        tokens,
        GEMMA4_NUM_LAYERS,
        GEMMA4_PLE_DIM,
        m.auxLayout.per_layer_projection_norm.offset >>> 2,
      ],
      GEMMA4_NUM_LAYERS,
      tokens,
      1,
    );

    // Layer 0 input_layernorm (when fusedNormsActive, subsequent layers receive input_layernorm from previous layer's mode 4!)
    if (fusedNormsActive) {
      add(
        this.pipelines.rmsFused,
        this.bindGroups.rmsFused,
        [tokens, GEMMA4_HIDDEN, m.auxLayout.layers[0].input_layernorm.offset >>> 2, 0, 0, 0, 0, 0],
        tokens,
      );
    }

    const gKvMode = this.kvMode === 'int4' ? 2 : this.kvMode === 'int8' ? 1 : 0;

    // 2. Transformer layers (full 42, or activeLayers when self-speculating with layer skip)
    for (let l = 0; l < activeLayers; l++) {
      const lw = m.layerOffsets[l];
      const auxL = m.auxLayout.layers[l];
      const route = this.routes[l];
      const sIdx = lw.slab;
      const headDim = route.isFullAttn ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
      const qRows = GEMMA4_ATTN_HEADS * headDim;
      const kvRows = GEMMA4_KV_HEADS * headDim;

      if (!fusedNormsActive) {
        add(
          this.pipelines.rmsNorm,
          this.bindGroups.rmsHiddenToNorm,
          [tokens, GEMMA4_HIDDEN, auxL.input_layernorm.offset >>> 2, 0, 0, 0, 0, 0],
          tokens,
        );
      }

      // Q / K / V projections (fused into 1 dispatch when useFusedQkv!)
      if (this.useFusedQkv) {
        if (!route.isKvShared && lw.qkv_proj) {
          addQ4(`normToQkv_${sIdx}`, lw.qkv_proj, qRows + 2 * kvRows, GEMMA4_HIDDEN, tokens);
        } else {
          addQ4(`normToQkv_${sIdx}`, lw.q_proj, qRows, GEMMA4_HIDDEN, tokens);
        }
        const ropeBgKey = !route.isFullAttn
          ? 'ropeSgSlide'
          : route.globalBufIdx === 0
            ? 'ropeSgGlobal0'
            : 'ropeSgGlobal1';
        add(
          this.pipelines.ropeKvSg,
          this.bindGroups[ropeBgKey],
          [
            tokens,
            startPos,
            headDim,
            GEMMA4_ATTN_HEADS,
            GEMMA4_KV_HEADS,
            route.isFullAttn ? 1 : 0,
            route.isKvShared ? 1 : 0,
            gKvMode,
            route.localDonorIdx,
            this.maxContext,
            auxL.q_norm.offset >>> 2,
            auxL.k_norm ? auxL.k_norm.offset >>> 2 : 0,
          ],
          route.isKvShared ? 8 : 12,
          tokens,
          1,
        );
      } else {
        addQ4(`normToQ_${sIdx}`, lw.q_proj, qRows, GEMMA4_HIDDEN, tokens);
        if (!route.isKvShared) {
          addQ4(`normToK_${sIdx}`, lw.k_proj!, kvRows, GEMMA4_HIDDEN, tokens);
          addQ4(`normToV_${sIdx}`, lw.v_proj!, kvRows, GEMMA4_HIDDEN, tokens);
        }

        const ropeBgKey = !route.isFullAttn
          ? 'ropeSlide'
          : route.globalBufIdx === 0
            ? 'ropeGlobal0'
            : 'ropeGlobal1';
        add(
          this.pipelines.ropeKv,
          this.bindGroups[ropeBgKey],
          [
            tokens,
            startPos,
            headDim,
            GEMMA4_ATTN_HEADS,
            GEMMA4_KV_HEADS,
            route.isFullAttn ? 1 : 0,
            route.isKvShared ? 1 : 0,
            gKvMode,
            route.localDonorIdx,
            this.maxContext,
            auxL.q_norm.offset >>> 2,
            auxL.k_norm ? auxL.k_norm.offset >>> 2 : 0,
          ],
          route.isKvShared ? 8 : 12,
          tokens,
          1,
        );
      }

      // GQA Attention (with Adaptive Split-K!)
      if (!route.isFullAttn) {
        if (isDecode) {
          if (this.pipelines.gqaDecSlideWarp) {
            const sSplits = startPos < 64 ? 1 : 16;
            add(
              this.pipelines.gqaDecSlideWarp,
              sSplits === 1 ? this.bindGroups.gqaSlide : this.bindGroups.gqaSlideSplit,
              [
                route.localDonorIdx,
                startPos,
                headDim,
                this.maxContext,
                0,
                sSplits,
                GEMMA4_SLIDING_WINDOW,
                0,
              ],
              2,
              sSplits,
              1,
            );
            if (sSplits > 1) {
              add(
                this.pipelines.gqaDecSlideReduce,
                this.bindGroups.gqaReduce,
                [
                  route.localDonorIdx,
                  startPos,
                  headDim,
                  this.maxContext,
                  0,
                  sSplits,
                  GEMMA4_SLIDING_WINDOW,
                  0,
                ],
                GEMMA4_ATTN_HEADS,
                1,
                1,
              );
            }
          } else if (this.hasSubgroups) {
            const sSplits = startPos < 128 ? 1 : 8;
            add(
              this.pipelines.gqaDecSlide,
              sSplits === 1 ? this.bindGroups.gqaSlide : this.bindGroups.gqaSlideSplit,
              [
                route.localDonorIdx,
                startPos,
                headDim,
                this.maxContext,
                0,
                sSplits,
                GEMMA4_SLIDING_WINDOW,
                0,
              ],
              2,
              sSplits,
              1,
            );
            if (sSplits > 1) {
              add(
                this.pipelines.gqaDecSlideReduce,
                this.bindGroups.gqaReduce,
                [
                  route.localDonorIdx,
                  startPos,
                  headDim,
                  this.maxContext,
                  0,
                  sSplits,
                  GEMMA4_SLIDING_WINDOW,
                  0,
                ],
                GEMMA4_ATTN_HEADS,
                1,
                1,
              );
            }
          } else {
            add(
              this.pipelines.gqaDecSlide,
              this.bindGroups.gqaSlide,
              [
                route.localDonorIdx,
                startPos,
                headDim,
                this.maxContext,
                0,
                1,
                GEMMA4_SLIDING_WINDOW,
                0,
              ],
              2,
              1,
              1,
            );
          }
        } else if (this.useFlashPrefill && this.pipelines.gqaPrefSlideFlash) {
          add(
            this.pipelines.gqaPrefSlideFlash,
            this.bindGroups.gqaSlide,
            [
              route.localDonorIdx,
              startPos,
              headDim,
              this.maxContext,
              0,
              1,
              GEMMA4_SLIDING_WINDOW,
              tokens,
            ],
            4,
            Math.ceil(tokens / 4),
            1,
          );
        } else {
          add(
            this.pipelines.gqaPrefSlide,
            this.bindGroups.gqaSlide,
            [
              route.localDonorIdx,
              startPos,
              headDim,
              this.maxContext,
              0,
              1,
              GEMMA4_SLIDING_WINDOW,
              0,
            ],
            2,
            tokens,
            1,
          );
        }
      } else {
        if (isDecode) {
          const useGlobalInt4 = Boolean(this.kvMode === 'int4' && this.pipelines.gqaDecGlobalInt4);
          const useGlobalWarp = Boolean(this.kvMode === 'int8' && this.pipelines.gqaDecGlobalWarp);
          const gSplits =
            startPos < 128
              ? 1
              : startPos < 2048
                ? useGlobalWarp || useGlobalInt4
                  ? 16
                  : 8
                : startPos < 16384
                  ? 32
                  : GEMMA4_GLOBAL_SPLITS;
          const gKey =
            gSplits === 1
              ? route.globalBufIdx === 0
                ? 'gqaPrefGlobal0'
                : 'gqaPrefGlobal1'
              : route.globalBufIdx === 0
                ? 'gqaGlobal0'
                : 'gqaGlobal1';
          const gDecPipe = useGlobalInt4
            ? this.pipelines.gqaDecGlobalInt4
            : useGlobalWarp
              ? this.pipelines.gqaDecGlobalWarp
              : this.useDp4a && this.kvMode === 'int8' && this.pipelines.gqaDecGlobalDp4a
                ? this.pipelines.gqaDecGlobalDp4a
                : this.pipelines.gqaDecGlobal;
          add(
            gDecPipe,
            this.bindGroups[gKey],
            [
              route.localDonorIdx,
              startPos,
              headDim,
              this.maxContext,
              gKvMode,
              gSplits,
              0,
              0,
            ],
            2,
            gSplits,
            1,
          );
          if (gSplits > 1) {
            add(
              this.pipelines.gqaDecReduce,
              this.bindGroups.gqaReduce,
              [route.localDonorIdx, startPos, headDim, this.maxContext, gKvMode, gSplits, 0, 0],
              GEMMA4_ATTN_HEADS,
              1,
              1,
            );
          }
        } else if (
          this.useFlashPrefill &&
          ((this.kvMode === 'int4' && this.pipelines.gqaPrefGlobalFlashInt4) ||
            (this.kvMode === 'int8' && this.pipelines.gqaPrefGlobalFlash))
        ) {
          const prefillSplits = startPos + tokens >= 1024 ? 8 : 1;
          const gKey =
            prefillSplits === 1
              ? route.globalBufIdx === 0
                ? 'gqaPrefGlobal0'
                : 'gqaPrefGlobal1'
              : route.globalBufIdx === 0
                ? 'gqaPrefFlashGlobal0'
                : 'gqaPrefFlashGlobal1';
          const gFlashPipe =
            this.kvMode === 'int4'
              ? this.pipelines.gqaPrefGlobalFlashInt4
              : this.pipelines.gqaPrefGlobalFlash;
          add(
            gFlashPipe,
            this.bindGroups[gKey],
            [
              route.localDonorIdx,
              startPos,
              headDim,
              this.maxContext,
              gKvMode,
              prefillSplits,
              0,
              tokens,
            ],
            4,
            Math.ceil(tokens / 4),
            prefillSplits,
          );
          if (prefillSplits > 1) {
            add(
              this.pipelines.gqaPrefReduce,
              this.bindGroups.gqaPrefReduce,
              [
                route.localDonorIdx,
                startPos,
                headDim,
                this.maxContext,
                gKvMode,
                prefillSplits,
                0,
                tokens,
              ],
              GEMMA4_ATTN_HEADS,
              tokens,
              1,
            );
          }
        } else {
          const gKey = route.globalBufIdx === 0 ? 'gqaPrefGlobal0' : 'gqaPrefGlobal1';
          add(
            this.pipelines.gqaPrefGlobal,
            this.bindGroups[gKey],
            [route.localDonorIdx, startPos, headDim, this.maxContext, gKvMode, 1, 0, 0],
            2,
            tokens,
            1,
          );
        }
      }

      // o_proj: attnOut -> norm
      addQ4(`attnToNorm_${sIdx}`, lw.o_proj, GEMMA4_HIDDEN, qRows, tokens);

      // post_attention_layernorm + pre_feedforward_layernorm (fused in 1 pass when fusedNormsActive!)
      if (fusedNormsActive) {
        add(
          this.pipelines.rmsFused,
          this.bindGroups.rmsFused,
          [
            tokens,
            GEMMA4_HIDDEN,
            auxL.post_attention_layernorm.offset >>> 2,
            auxL.pre_feedforward_layernorm.offset >>> 2,
            0,
            0,
            3,
            0,
          ],
          tokens,
        );
      } else {
        add(
          this.pipelines.rmsNorm,
          this.bindGroups.rmsNormToHidden,
          [tokens, GEMMA4_HIDDEN, auxL.post_attention_layernorm.offset >>> 2, 0, 0, 0, 1, 0],
          tokens,
        );
        add(
          this.pipelines.rmsNorm,
          this.bindGroups.rmsHiddenToNorm,
          [tokens, GEMMA4_HIDDEN, auxL.pre_feedforward_layernorm.offset >>> 2, 0, 0, 0, 0, 0],
          tokens,
        );
      }

      // MLP gate_up + GeGLU + down_proj
      if (isDecode) {
        if (this.useVec4Gemv && this.pipelines.q4GegluVec4) {
          add(
            this.pipelines.q4GegluVec4,
            this.bindGroups[`normToMlpAct_${sIdx}`],
            [
              GEMMA4_INTERMEDIATE,
              GEMMA4_HIDDEN,
              lw.gate_up_proj.qOffset >>> 4,
              lw.gate_up_proj.sOffset >>> 4,
              0,
              0,
              0,
              0,
            ],
            Math.ceil(GEMMA4_INTERMEDIATE / 4),
            1,
            1,
          );
        } else {
          add(
            this.pipelines.q4Geglu,
            this.bindGroups[`normToMlpAct_${sIdx}`],
            [
              GEMMA4_INTERMEDIATE,
              GEMMA4_HIDDEN,
              lw.gate_up_proj.qOffset >>> 2,
              lw.gate_up_proj.sOffset >>> 2,
              0,
              0,
              0,
              0,
            ],
            Math.ceil(GEMMA4_INTERMEDIATE / 8),
            1,
            1,
          );
        }
      } else if (tokens <= 8 && this.useSmallBatchSpec) {
        add(
          this.pipelines.q4GegluBatch,
          this.bindGroups[`normToMlpAct_${sIdx}`],
          [
            GEMMA4_INTERMEDIATE,
            GEMMA4_HIDDEN,
            lw.gate_up_proj.qOffset >>> 2,
            lw.gate_up_proj.sOffset >>> 2,
            0,
            0,
            0,
            tokens,
          ],
          Math.ceil(GEMMA4_INTERMEDIATE / 8),
          1,
          1,
        );
      } else {
        addQ4(`normToGateUp_${sIdx}`, lw.gate_up_proj, GEMMA4_INTERMEDIATE * 2, GEMMA4_HIDDEN, tokens);
        add(
          this.pipelines.gegluBatch,
          this.bindGroups.gegluBatch,
          [tokens, GEMMA4_INTERMEDIATE, 0, 0],
          Math.ceil(GEMMA4_INTERMEDIATE / 256),
          tokens,
          1,
        );
      }
      addQ4(`mlpActToNorm_${sIdx}`, lw.down_proj, GEMMA4_HIDDEN, GEMMA4_INTERMEDIATE, tokens);

      // post_feedforward_layernorm residual -> hidden
      if (fusedNormsActive) {
        add(
          this.pipelines.rmsFused,
          this.bindGroups.rmsFused,
          [tokens, GEMMA4_HIDDEN, auxL.post_feedforward_layernorm.offset >>> 2, 0, 0, 0, 1, 0],
          tokens,
        );
      } else {
        add(
          this.pipelines.rmsNorm,
          this.bindGroups.rmsNormToHidden,
          [tokens, GEMMA4_HIDDEN, auxL.post_feedforward_layernorm.offset >>> 2, 0, 0, 0, 1, 0],
          tokens,
        );
      }

      // Per-Layer Embeddings: ple_gate -> ple_proj -> post_ple_norm * layer_scalar (+ next layer input_norm or final_norm!)
      const q8Pipe = this.pipelines.q8GemvDp4a ?? this.pipelines.q8Gemv;
      add(
        q8Pipe,
        this.bindGroups[`q8PleGate_${sIdx}`],
        [
          GEMMA4_PLE_DIM,
          GEMMA4_HIDDEN,
          lw.ple_gate.qOffset >>> 2,
          lw.ple_gate.sOffset >>> 2,
          0,
          0,
          1,
          l * GEMMA4_PLE_DIM,
        ],
        Math.ceil(GEMMA4_PLE_DIM / 8),
        tokens,
        1,
      );
      add(
        q8Pipe,
        this.bindGroups[`q8PleProj_${sIdx}`],
        [
          GEMMA4_HIDDEN,
          GEMMA4_PLE_DIM,
          lw.ple_proj.qOffset >>> 2,
          lw.ple_proj.sOffset >>> 2,
          0,
          0,
          0,
          0,
        ],
        Math.ceil(GEMMA4_HIDDEN / 8),
        tokens,
        1,
      );

      const isLastActiveLayer = l === activeLayers - 1;
      if (fusedNormsActive && (!isLastActiveLayer || logitsTokens > 0)) {
        const nextW1Off = !isLastActiveLayer
          ? m.auxLayout.layers[l + 1].input_layernorm.offset >>> 2
          : m.auxLayout.final_norm.offset >>> 2;
        add(
          this.pipelines.rmsFused,
          this.bindGroups.rmsFused,
          [
            tokens,
            GEMMA4_HIDDEN,
            auxL.post_per_layer_input_norm.offset >>> 2,
            nextW1Off,
            0,
            0,
            4,
            auxL.layer_scalar.offset >>> 2,
          ],
          tokens,
        );
      } else {
        add(
          this.pipelines.rmsNorm,
          this.bindGroups.rmsNormToHidden,
          [
            tokens,
            GEMMA4_HIDDEN,
            auxL.post_per_layer_input_norm.offset >>> 2,
            0,
            0,
            0,
            2,
            auxL.layer_scalar.offset >>> 2,
          ],
          tokens,
        );
      }

      layerEndIndices.push(steps.length);
    }

    // 3. Final Norm (only if not already fused into final active layer mode 4!) & LM Head
    const logitsTokenOffset = tokens - logitsTokens;
    if (logitsTokens > 0) {
      if (!fusedNormsActive) {
        add(
          this.pipelines.rmsNorm,
          this.bindGroups.rmsHiddenToNorm,
          [
            logitsTokens,
            GEMMA4_HIDDEN,
            m.auxLayout.final_norm.offset >>> 2,
            logitsTokenOffset * GEMMA4_HIDDEN,
            0,
            0,
            0,
            0,
          ],
          logitsTokens,
        );
      }
      const lmInOff = fusedNormsActive ? logitsTokenOffset * GEMMA4_HIDDEN : 0;
      if (logitsTokens >= 2 && logitsTokens <= 8 && this.useSmallBatchSpec) {
        add(
          this.pipelines.q2LmHeadBatch,
          this.bindGroups.lmHead,
          [
            GEMMA4_VOCAB_SIZE,
            GEMMA4_HIDDEN,
            m.lmHeadLayout.qOffset >>> 2,
            m.lmHeadLayout.sOffset >>> 2,
            lmInOff,
            0,
            1,
            logitsTokens,
          ],
          Math.ceil(GEMMA4_VOCAB_SIZE / 16),
          1,
          1,
        );
      } else if (logitsTokens === 1 && this.useW4A8Dp4a && this.pipelines.q2LmHeadDp4a) {
        add(
          this.pipelines.q2LmHeadDp4a,
          this.bindGroups.lmHead,
          [
            GEMMA4_VOCAB_SIZE,
            GEMMA4_HIDDEN,
            m.lmHeadLayout.qOffset >>> 2,
            m.lmHeadLayout.sOffset >>> 2,
            lmInOff,
            0,
            1,
            0,
          ],
          Math.ceil(GEMMA4_VOCAB_SIZE / 16),
          1,
          1,
        );
      } else {
        add(
          this.pipelines.q2LmHead,
          this.bindGroups.lmHead,
          [
            GEMMA4_VOCAB_SIZE,
            GEMMA4_HIDDEN,
            m.lmHeadLayout.qOffset >>> 2,
            m.lmHeadLayout.sOffset >>> 2,
            lmInOff,
            0,
            1,
            0,
          ],
          Math.ceil(GEMMA4_VOCAB_SIZE / 16),
          logitsTokens,
          1,
        );
      }
    }

    return { steps, layerEndIndices, logitsTokenOffset };
  }

  private encodeStepsRange(steps: DispatchStep[], start: number, end: number, pass?: ReturnType<GpuDeviceLike['createCommandEncoder']>['beginComputePass'] extends () => infer P ? P : never): void {
    const dev = this.device!;
    const enc = pass ? null : dev.createCommandEncoder();
    const cp = pass ?? enc!.beginComputePass();
    for (let i = start; i < end; i++) {
      const s = steps[i];
      cp.setPipeline(s.pipeline);
      cp.setBindGroup(0, s.bindGroup, [s.uniformByteOffset]);
      cp.dispatchWorkgroups(s.wx, s.wy, s.wz);
    }
    if (!pass) {
      cp.end();
      dev.queue.submit([enc!.finish()]);
    }
  }

  private async readArgmaxArray(count: number): Promise<number[]> {
    const dev = this.device!;
    const src = count === 1 ? this.argmaxSingleOutBuf! : this.argmaxMultiOutBuf!;
    const bytes = Math.max(4, count * 4);
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, this.argmaxReadBuf!, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await this.argmaxReadBuf!.mapAsync(GPU_MAP_READ, 0, bytes);
    const out = Array.from(new Uint32Array(this.argmaxReadBuf!.getMappedRange(0, bytes)));
    this.argmaxReadBuf!.unmap();
    return out;
  }

  async prefillTokens(
    tokenIds: readonly number[],
    options?: {
      chunkSize?: number;
      returnLogits?: boolean;
      abortSignal?: AbortSignal;
      onChunk?: (processed: number, total: number) => void;
    },
  ): Promise<Gemma4PrefillResult> {
    const total = tokenIds.length;
    if (total === 0) {
      throw new Error('prefillTokens requires at least 1 token');
    }
    const t0 = performance.now();
    let totalFetchMs = 0;
    let totalDqMs = 0;
    let burstMs = 0;

    let offset = 0;
    let nextTokenId = 0;
    let top5Logits: Gemma4TopLogit[] | undefined;

    while (offset < total) {
      if (options?.abortSignal?.aborted) {
        break;
      }
      let chunkLim = selectPrefillChunkSize(
        this.pacingMode,
        this.seqPos,
        options?.chunkSize,
        calibratePacingForHardware(this.pacingMode, this.displayHz, this.lastDecodeStepGpuMs),
      );
      if (this.pacingMode === 'balanced') {
        if (this.seqPos >= 49152) {
          chunkLim = Math.min(chunkLim, 16);
        } else if (this.seqPos >= 24576) {
          chunkLim = Math.min(chunkLim, 24);
        } else if (this.seqPos >= 4096) {
          chunkLim = Math.min(chunkLim, 32);
        }
      }
      const len = Math.min(chunkLim, total - offset);
      const isLastChunk = offset + len >= total;
      const chunkSlice = tokenIds.slice(offset, offset + len);

      const { usedGpuHot, fetchMs, dequantMs } = await this.preparePleOrUseGpuHot(chunkSlice);
      totalFetchMs += fetchMs;
      totalDqMs += dequantMs;

      const tChunkGpu0 = performance.now();
      const idsU32 = Uint32Array.from(chunkSlice, (x) => (x < 0 ? 0 : x >>> 0));
      this.device!.queue.writeBuffer(this.tokenIdsBuf!, 0, idsU32);

      const { steps, layerEndIndices } = this.buildForwardSteps(
        len,
        this.seqPos,
        isLastChunk ? 1 : 0,
        { useGpuHotPle: usedGpuHot },
      );
      const wg1 = Math.ceil(GEMMA4_VOCAB_SIZE / 256);
      const arg1Off = isLastChunk ? this.pushUniformSlot([GEMMA4_VOCAB_SIZE, 0, 0, 0]) : 0;
      const arg2Off = isLastChunk ? this.pushUniformSlot([wg1, 0, 0, 0]) : 0;

      this.writeUniformBankToGpu(0);

      if (this.pacingMode === 'max-speed') {
        const enc = this.device!.createCommandEncoder();
        const pass = enc.beginComputePass();
        this.encodeStepsRange(steps, 0, steps.length, pass);
        if (isLastChunk) {
          pass.setPipeline(this.pipelines.argmax1);
          pass.setBindGroup(0, this.bindGroups.argmax1Main, [arg1Off]);
          pass.dispatchWorkgroups(wg1, 1, 1);
          pass.setPipeline(this.pipelines.argmax2);
          pass.setBindGroup(0, this.bindGroups.argmax2, [arg2Off]);
          pass.dispatchWorkgroups(1, 1, 1);
        }
        pass.end();
        this.device!.queue.submit([enc.finish()]);
        await this.device!.queue.onSubmittedWorkDone();
      } else {
        // Slice prefill across 14-layer boundaries in balanced mode so GPU frames stay ~144Hz smooth
        const sliceEnds: number[] = [];
        for (let l = 13; l < layerEndIndices.length - 1; l += 14) {
          sliceEnds.push(layerEndIndices[l]!);
        }
        sliceEnds.push(steps.length);

        let cursor = 0;
        for (let s = 0; s < sliceEnds.length; s++) {
          if (options?.abortSignal?.aborted) {
            break;
          }
          const endIdx = sliceEnds[s]!;
          const isFinalSlice = s === sliceEnds.length - 1;
          const tSlice0 = performance.now();
          const enc = this.device!.createCommandEncoder();
          const pass = enc.beginComputePass();
          this.encodeStepsRange(steps, cursor, endIdx, pass);
          if (isFinalSlice && isLastChunk) {
            pass.setPipeline(this.pipelines.argmax1);
            pass.setBindGroup(0, this.bindGroups.argmax1Main, [arg1Off]);
            pass.dispatchWorkgroups(wg1, 1, 1);
            pass.setPipeline(this.pipelines.argmax2);
            pass.setBindGroup(0, this.bindGroups.argmax2, [arg2Off]);
            pass.dispatchWorkgroups(1, 1, 1);
          }
          pass.end();
          this.device!.queue.submit([enc.finish()]);
          await this.device!.queue.onSubmittedWorkDone();
          cursor = endIdx;

          burstMs += performance.now() - tSlice0;
          if (!(isFinalSlice && isLastChunk)) {
            const { breatherMs, nextBurstMs } = computePrefillSliceBreatherMs(this.pacingMode, burstMs);
            burstMs = nextBurstMs;
            if (breatherMs > 0) {
              await sleepMs(breatherMs);
            }
          }
        }
      }

      if (isLastChunk) {
        [nextTokenId] = await this.readArgmaxArray(1);
        if (options?.returnLogits) {
          const logits = await readGpuF32Slice(
            this.device!,
            this.logitsBuf!,
            0,
            GEMMA4_VOCAB_SIZE * 4,
          );
          top5Logits = extractTopKLogits(logits, 5);
        }
      }

      if (this.pacingMode === 'max-speed') {
        const chunkGpuMs = performance.now() - tChunkGpu0;
        burstMs += chunkGpuMs;
        const { breatherMs, nextBurstMs } = computePrefillSliceBreatherMs(this.pacingMode, burstMs);
        burstMs = nextBurstMs;
        if (breatherMs > 0 && !isLastChunk) {
          await sleepMs(breatherMs);
        }
      }

      for (let i = 0; i < len; i++) {
        this.contextTokenIds.push(chunkSlice[i]);
        this.speculator.pushToken(chunkSlice[i]);
      }
      this.seqPos += len;
      offset += len;
      options?.onChunk?.(offset, total);
    }

    const prefillMs = Number((performance.now() - t0).toFixed(2));
    return {
      prefillMs,
      prefillTokPerSec: Number(((total * 1000) / Math.max(0.1, prefillMs)).toFixed(2)),
      pleFetchMs: Number(totalFetchMs.toFixed(2)),
      pleDequantMs: Number(totalDqMs.toFixed(2)),
      nextTokenId,
      top5Logits,
    };
  }

  async decodeStep(
    tokenId: number,
    options?: { returnLogits?: boolean; maxLayerCount?: number; skipContextPush?: boolean; explicitPos?: number },
  ): Promise<Gemma4DecodeStepResult> {
    const t0 = performance.now();
    const { usedGpuHot } = await this.preparePleOrUseGpuHot([tokenId]);

    const tGpu0 = performance.now();
    const policy = calibratePacingForHardware(this.pacingMode, this.displayHz, this.lastDecodeStepGpuMs);
    const activePos = options?.explicitPos ?? this.seqPos;
    const numSlices = options?.maxLayerCount
      ? 1
      : selectDecodeSliceCount(
          this.pacingMode,
          activePos,
          this.lastDecodeStepGpuMs,
          policy,
        );

    const idsU32 = new Uint32Array([tokenId >>> 0]);
    this.device!.queue.writeBuffer(this.tokenIdsBuf!, 0, idsU32);

    const { steps, layerEndIndices } = this.buildForwardSteps(1, activePos, 1, {
      useGpuHotPle: usedGpuHot,
      maxLayerCount: options?.maxLayerCount,
    });

    // Pre-reserve argmax uniform slots before uploading the full uniform buffer
    const argmax1UOff = this.pushUniformSlot([GEMMA4_VOCAB_SIZE, 0, 0, 0]);
    const wg1 = Math.ceil(GEMMA4_VOCAB_SIZE / 256);
    const argmax2UOff = this.pushUniformSlot([wg1, 0, 0, 0]);

    this.writeUniformBankToGpu(0);

    if (numSlices === 1) {
      const enc = this.device!.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.encodeStepsRange(steps, 0, steps.length, pass);
      pass.setPipeline(this.pipelines.argmax1);
      pass.setBindGroup(0, this.bindGroups.argmax1Main, [argmax1UOff]);
      pass.dispatchWorkgroups(wg1, 1, 1);
      pass.setPipeline(this.pipelines.argmax2);
      pass.setBindGroup(0, this.bindGroups.argmax2, [argmax2UOff]);
      pass.dispatchWorkgroups(1, 1, 1);
      pass.end();
      this.device!.queue.submit([enc.finish()]);
    } else {
      const bounds = computeDecodeSliceBounds(layerEndIndices, numSlices);
      for (let b = 0; b < bounds.length; b++) {
        const enc = this.device!.createCommandEncoder();
        const pass = enc.beginComputePass();
        this.encodeStepsRange(steps, bounds[b].start, bounds[b].end, pass);
        if (b === bounds.length - 1) {
          this.encodeStepsRange(steps, bounds[b].end, steps.length, pass);
          pass.setPipeline(this.pipelines.argmax1);
          pass.setBindGroup(0, this.bindGroups.argmax1Main, [argmax1UOff]);
          pass.dispatchWorkgroups(wg1, 1, 1);
          pass.setPipeline(this.pipelines.argmax2);
          pass.setBindGroup(0, this.bindGroups.argmax2, [argmax2UOff]);
          pass.dispatchWorkgroups(1, 1, 1);
        }
        pass.end();
        this.device!.queue.submit([enc.finish()]);
        if (b < bounds.length - 1) {
          await this.device!.queue.onSubmittedWorkDone();
          if (policy.decodeSliceBreatherMs > 0) {
            await sleepMs(policy.decodeSliceBreatherMs);
          }
        }
      }
    }

    const [nextTokenId] = await this.readArgmaxArray(1);
    let top5Logits: Gemma4TopLogit[] | undefined;
    if (options?.returnLogits) {
      const logits = await readGpuF32Slice(
        this.device!,
        this.logitsBuf!,
        0,
        GEMMA4_VOCAB_SIZE * 4,
      );
      top5Logits = extractTopKLogits(logits, 5);
    }

    this.lastDecodeStepGpuMs = performance.now() - tGpu0;
    if (!options?.skipContextPush) {
      this.contextTokenIds.push(tokenId);
      this.speculator.pushToken(tokenId);
      this.seqPos += 1;
    }

    return {
      nextTokenId,
      top5Logits,
      forwardMs: Number((performance.now() - t0).toFixed(2)),
    };
  }

  /** Launches one asynchronous single-token GPU decode step into `uniformBank` (0 or 1), reading `argmaxReadBuf` / `argmaxReadBufB`. */
  private launchPipelinedSlot(
    bank: 0 | 1,
    pos: number,
    knownTokenId?: number,
  ): {
    bank: 0 | 1;
    readBuf: GpuBufferLike;
    pos: number;
    tSubmitMs: number;
    mapPromise: Promise<void>;
  } {
    const dev = this.device!;
    if (knownTokenId !== undefined) {
      dev.queue.writeBuffer(this.tokenIdsBuf!, 0, Uint32Array.of(knownTokenId >>> 0));
    }
    const { steps } = this.buildForwardSteps(1, pos, 1, {
      useGpuHotPle: true,
      uniformBank: bank,
    });
    const wg1 = Math.ceil(GEMMA4_VOCAB_SIZE / 256);
    const argmax1UOff = this.pushUniformSlot([GEMMA4_VOCAB_SIZE, 0, 0, 0]);
    const argmax2UOff = this.pushUniformSlot([wg1, 0, 0, 0]);
    this.writeUniformBankToGpu(bank);

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeStepsRange(steps, 0, steps.length, pass);
    pass.setPipeline(this.pipelines.argmax1);
    pass.setBindGroup(0, this.bindGroups.argmax1Main, [argmax1UOff]);
    pass.dispatchWorkgroups(wg1, 1, 1);
    pass.setPipeline(this.pipelines.argmax2);
    pass.setBindGroup(0, this.bindGroups.argmax2, [argmax2UOff]);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();

    const readBuf = bank === 0 ? this.argmaxReadBuf! : this.argmaxReadBufB!;
    enc.copyBufferToBuffer(this.argmaxSingleOutBuf!, 0, readBuf, 0, 4);
    enc.copyBufferToBuffer(this.argmaxSingleOutBuf!, 0, this.tokenIdsBuf!, 0, 4);
    const tSubmitMs = performance.now();
    dev.queue.submit([enc.finish()]);
    const mapPromise = readBuf.mapAsync(GPU_MAP_READ, 0, 4);
    return { bank, readBuf, pos, tSubmitMs, mapPromise };
  }

  /** Executes up to `maxSteps` decode steps using 2-in-flight GPU/CPU pipelined overlap when hot PLE is active. */
  private async decodePipelined(
    initialTokenId: number,
    maxSteps: number,
    options?: {
      eosIds?: Set<number>;
      abortSignal?: AbortSignal;
      onToken?: (tokenId: number) => void;
    },
  ): Promise<{ tokens: number[]; aborted: boolean }> {
    const produced: number[] = [];
    let aborted = false;
    let curInputTok = initialTokenId;

    while (produced.length < maxSteps) {
      if (options?.abortSignal?.aborted) {
        aborted = true;
        break;
      }

      if (!this.isHotPleToken(curInputTok)) {
        const d = await this.decodeStep(curInputTok);
        const predTok = d.nextTokenId;
        if (options?.eosIds?.has(predTok)) {
          break;
        }
        produced.push(predTok);
        options?.onToken?.(predTok);
        curInputTok = predTok;
        continue;
      }

      const remaining = maxSteps - produced.length;
      const inFlight = [this.launchPipelinedSlot(0, this.seqPos, curInputTok)];
      if (remaining > 1) {
        inFlight.push(this.launchPipelinedSlot(1, this.seqPos + 1));
      }

      let coldMissedToken: number | null = null;
      let prevDoneMs = performance.now();

      while (inFlight.length > 0) {
        const cur = inFlight.shift()!;
        await cur.mapPromise;
        const predTok = new Uint32Array(cur.readBuf.getMappedRange(0, 4))[0];
        cur.readBuf.unmap();

        const now = performance.now();
        this.lastDecodeStepGpuMs = Math.max(0.5, now - Math.max(prevDoneMs, cur.tSubmitMs));
        prevDoneMs = now;

        this.contextTokenIds.push(curInputTok);
        this.speculator.pushToken(curInputTok);
        this.seqPos += 1;

        if (options?.eosIds?.has(predTok)) {
          while (inFlight.length > 0) {
            const rem = inFlight.shift()!;
            await rem.mapPromise;
            rem.readBuf.unmap();
          }
          return { tokens: produced, aborted };
        }

        produced.push(predTok);
        options?.onToken?.(predTok);

        if (produced.length >= maxSteps || options?.abortSignal?.aborted) {
          aborted = Boolean(options?.abortSignal?.aborted);
          while (inFlight.length > 0) {
            const rem = inFlight.shift()!;
            await rem.mapPromise;
            rem.readBuf.unmap();
          }
          return { tokens: produced, aborted };
        }

        if (this.isHotPleToken(predTok)) {
          curInputTok = predTok;
          if (produced.length + inFlight.length < maxSteps) {
            inFlight.push(this.launchPipelinedSlot(cur.bank, this.seqPos + inFlight.length));
          }
        } else {
          // Cold token fallback: drain lookahead slot (computed with placeholder slot 0) and continue from predTok
          while (inFlight.length > 0) {
            const rem = inFlight.shift()!;
            await rem.mapPromise;
            rem.readBuf.unmap();
          }
          coldMissedToken = predTok;
          break;
        }
      }

      if (coldMissedToken !== null) {
        curInputTok = coldMissedToken;
        continue;
      }
      break;
    }

    return { tokens: produced, aborted };
  }

  /** Benchmarks or executes `count` fixed decode steps starting from `firstTokenId`. */
  async decodeSequence(
    firstTokenId: number,
    count: number,
  ): Promise<{ tokens: number[]; decodeMs: number; decodeTokPerSec: number }> {
    const t0 = performance.now();
    let tokens: number[];
    if (this.pacingMode === 'max-speed' && this.usePipelinedDecode && this.useGpuHotPle && count > 1) {
      const res = await this.decodePipelined(firstTokenId, count);
      tokens = res.tokens;
    } else {
      tokens = [];
      let cur = firstTokenId;
      for (let i = 0; i < count; i++) {
        const d = await this.decodeStep(cur);
        cur = d.nextTokenId;
        tokens.push(cur);
      }
    }
    const decodeMs = Math.max(0.1, performance.now() - t0);
    return {
      tokens,
      decodeMs: Number(decodeMs.toFixed(2)),
      decodeTokPerSec: Number(((tokens.length * 1000) / decodeMs).toFixed(2)),
    };
  }

  private async draftTokensWithLayerSkip(
    startTokenId: number,
    draftCount: number,
    skipLayers: number,
  ): Promise<number[]> {
    const maxLayerCount = Math.max(6, GEMMA4_NUM_LAYERS - skipLayers);
    const drafts: number[] = [];
    let cur = startTokenId;
    for (let k = 0; k < draftCount; k++) {
      const d = await this.decodeStep(cur, {
        maxLayerCount,
        skipContextPush: true,
        explicitPos: this.seqPos + k,
      });
      cur = d.nextTokenId;
      drafts.push(cur);
    }
    return drafts;
  }

  /** Verifies `tokens` (`T <= 5`: `[curToken, ...draftTokens]`) in a single batched forward pass, returning greedy predictions `[pred_0 .. pred_{T-1}]`. */
  async verifySpeculativeTokens(tokens: readonly number[]): Promise<{ predictions: number[]; gpuMs: number }> {
    const T = tokens.length;
    if (T < 1 || T > GEMMA4_MAX_SPEC_TOKENS) {
      throw new Error(`verifySpeculativeTokens requires 1..${GEMMA4_MAX_SPEC_TOKENS} tokens, got ${T}`);
    }
    if (T === 1) {
      const r = await this.decodeStep(tokens[0]);
      return { predictions: [r.nextTokenId], gpuMs: r.forwardMs };
    }

    const { usedGpuHot } = await this.preparePleOrUseGpuHot(tokens);
    const tGpu0 = performance.now();
    const idsU32 = Uint32Array.from(tokens, (x) => x >>> 0);
    this.device!.queue.writeBuffer(this.tokenIdsBuf!, 0, idsU32);

    const { steps } = this.buildForwardSteps(T, this.seqPos, T, { useGpuHotPle: usedGpuHot });
    const wg1 = Math.ceil(GEMMA4_VOCAB_SIZE / 256);

    if (this.useSmallBatchSpec && this.pipelines.argmaxBatch1) {
      const batchUOff = this.pushUniformSlot([GEMMA4_VOCAB_SIZE, T, wg1, 0]);
      this.device!.queue.writeBuffer(
        this.uniformBuf!,
        0,
        this.uniformCpu,
        0,
        this._uniformCursor * UNIFORM_SLOT_U32S,
      );

      const enc = this.device!.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.encodeStepsRange(steps, 0, steps.length, pass);
      pass.setPipeline(this.pipelines.argmaxBatch1);
      pass.setBindGroup(0, this.bindGroups.argmax1Main, [batchUOff]);
      pass.dispatchWorkgroups(wg1, T, 1);
      pass.setPipeline(this.pipelines.argmaxBatch2);
      pass.setBindGroup(0, this.bindGroups.argmaxBatch2, [batchUOff]);
      pass.dispatchWorkgroups(T, 1, 1);
      pass.end();

      this.device!.queue.submit([enc.finish()]);
      const predictions = await this.readArgmaxArray(T);
      this.lastDecodeStepGpuMs = performance.now() - tGpu0;

      return { predictions, gpuMs: this.lastDecodeStepGpuMs };
    }

    const argmaxUniforms: Array<[number, number]> = [];
    for (let t = 0; t < T; t++) {
      argmaxUniforms.push([
        this.pushUniformSlot([GEMMA4_VOCAB_SIZE, 0, 0, 0]),
        this.pushUniformSlot([wg1, 0, 0, 0]),
      ]);
    }

    this.writeUniformBankToGpu(0);

    const enc = this.device!.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeStepsRange(steps, 0, steps.length, pass);
    pass.setPipeline(this.pipelines.argmax1);
    pass.setBindGroup(0, this.bindGroups.argmax1Main, [argmaxUniforms[0][0]]);
    pass.dispatchWorkgroups(wg1, 1, 1);
    pass.setPipeline(this.pipelines.argmax2);
    pass.setBindGroup(0, this.bindGroups.argmax2, [argmaxUniforms[0][1]]);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    enc.copyBufferToBuffer(this.argmaxSingleOutBuf!, 0, this.argmaxMultiOutBuf!, 0, 4);

    for (let t = 1; t < T; t++) {
      enc.copyBufferToBuffer(
        this.logitsBuf!,
        t * GEMMA4_VOCAB_SIZE * 4,
        this.logitsSingleBuf!,
        0,
        GEMMA4_VOCAB_SIZE * 4,
      );
      const p = enc.beginComputePass();
      p.setPipeline(this.pipelines.argmax1);
      p.setBindGroup(0, this.bindGroups.argmax1Single, [argmaxUniforms[t][0]]);
      p.dispatchWorkgroups(wg1, 1, 1);
      p.setPipeline(this.pipelines.argmax2);
      p.setBindGroup(0, this.bindGroups.argmax2, [argmaxUniforms[t][1]]);
      p.dispatchWorkgroups(1, 1, 1);
      p.end();
      enc.copyBufferToBuffer(this.argmaxSingleOutBuf!, 0, this.argmaxMultiOutBuf!, t * 4, 4);
    }

    this.device!.queue.submit([enc.finish()]);
    const predictions = await this.readArgmaxArray(T);
    this.lastDecodeStepGpuMs = performance.now() - tGpu0;

    return { predictions, gpuMs: this.lastDecodeStepGpuMs };
  }

  /** Full prompt-to-EOS greedy/speculative generation supporting zero-recompute prefix reuse and clean abort. */
  async generate(
    promptIds: readonly number[],
    options?: Gemma4GenerateOptions,
  ): Promise<Gemma4GenerateResult> {
    const maxNewTokens = options?.maxNewTokens ?? 128;
    const eosIds = options?.eosIds ?? new Set([1, 106]);
    const draftSkipLayers = Math.max(0, options?.draftSkipLayers ?? 0);
    const enableSpec = Boolean(options?.enableSpeculative) || draftSkipLayers > 0;

    // Check longest exact prefix reuse
    let common = 0;
    const maxCommon = Math.min(this.contextTokenIds.length, promptIds.length);
    while (common < maxCommon && this.contextTokenIds[common] === promptIds[common]) {
      common += 1;
    }

    // Notice if the entire prompt already matched up to promptIds.length - 1, keep at least 1 token to sample fresh logits
    if (common > 0 && common === promptIds.length) {
      common = Math.max(0, promptIds.length - 1);
    }

    // Because sliding/full KV cannot be rolled back mid-context without full prefill unless exact prefix matches seqPos
    if (common !== this.contextTokenIds.length) {
      this.resetContext();
      common = 0;
    }

    const suffixToPrefill = promptIds.slice(common);
    const tPref0 = performance.now();
    const prefRes = await this.prefillTokens(suffixToPrefill, {
      chunkSize: options?.chunkSize,
      returnLogits: true,
      abortSignal: options?.abortSignal,
    });
    const ttftMs = Number((performance.now() - tPref0).toFixed(2));

    const generatedIds: number[] = [];
    let aborted = Boolean(options?.abortSignal?.aborted);
    let nextTok = prefRes.nextTokenId;
    let drafted = 0;
    let accepted = 0;
    let specSteps = 0;
    let burstMs = 0;

    const tDec0 = performance.now();

    if (
      !enableSpec &&
      this.pacingMode === 'max-speed' &&
      this.usePipelinedDecode &&
      this.useGpuHotPle &&
      maxNewTokens > 1 &&
      !aborted &&
      !eosIds.has(nextTok)
    ) {
      generatedIds.push(nextTok);
      options?.onToken?.(nextTok, 0);
      const remaining = maxNewTokens - generatedIds.length;
      if (remaining > 0) {
        const pipeRes = await this.decodePipelined(nextTok, remaining, {
          eosIds,
          abortSignal: options?.abortSignal,
          onToken: (tokId) => {
            generatedIds.push(tokId);
            options?.onToken?.(tokId, generatedIds.length - 1);
          },
        });
        aborted = pipeRes.aborted;
      }
    } else {
      while (generatedIds.length < maxNewTokens) {
        if (options?.abortSignal?.aborted) {
          aborted = true;
          break;
        }
        if (eosIds.has(nextTok)) {
          break;
        }

        generatedIds.push(nextTok);
        options?.onToken?.(nextTok, generatedIds.length - 1);
        if (generatedIds.length >= maxNewTokens) {
          break;
        }

        let draft: number[] = [];
        const maxDraftAllowed = Math.min(GEMMA4_MAX_SPEC_DRAFT, maxNewTokens - generatedIds.length - 1);
        if (draftSkipLayers > 0 && maxDraftAllowed >= 2) {
          draft = await this.draftTokensWithLayerSkip(nextTok, Math.min(3, maxDraftAllowed), draftSkipLayers);
        } else if (enableSpec) {
          draft = this.speculator.propose([...this.contextTokenIds, nextTok], {
            minMatch: 3,
            maxMatch: 8,
            maxDraft: maxDraftAllowed,
          });
        }

        if (draft.length > 0) {
          const batch = [nextTok, ...draft];
          const ver = await this.verifySpeculativeTokens(batch);
          specSteps += 1;
          drafted += draft.length;

          let matchCount = 0;
          while (matchCount < draft.length && ver.predictions[matchCount] === draft[matchCount]) {
            matchCount += 1;
          }
          accepted += matchCount;

          // Commit [nextTok, draft[0..matchCount-1]] permanently into KV state; discard rejected draft tail
          // Overwriting rejected positions simply resets seqPos to seqPos + 1 + matchCount!
          const acceptedSeq = [nextTok, ...draft.slice(0, matchCount)];
          for (const id of acceptedSeq) {
            this.contextTokenIds.push(id);
            this.speculator.pushToken(id);
          }
          this.seqPos += 1 + matchCount;

          for (let k = 0; k < matchCount; k++) {
            if (eosIds.has(draft[k])) {
              nextTok = draft[k];
              break;
            }
            generatedIds.push(draft[k]);
            options?.onToken?.(draft[k], generatedIds.length - 1);
            if (generatedIds.length >= maxNewTokens) {
              nextTok = ver.predictions[matchCount];
              break;
            }
          }
          if (!eosIds.has(nextTok)) {
            nextTok = ver.predictions[matchCount];
          }
          burstMs += ver.gpuMs;
        } else {
          const stepRes = await this.decodeStep(nextTok);
          nextTok = stepRes.nextTokenId;
          burstMs += this.lastDecodeStepGpuMs;
        }

        const policy = calibratePacingForHardware(
          this.pacingMode,
          this.displayHz,
          this.lastDecodeStepGpuMs,
        );
        const { breatherMs, nextBurstMs } = computeDecodeBurstBreatherMs(
          this.pacingMode,
          0,
          burstMs,
          policy,
        );
        burstMs = nextBurstMs;
        if (breatherMs > 0) {
          await sleepMs(breatherMs);
        }
      }
    }

    const decodeMs = Number(Math.max(0.1, performance.now() - tDec0).toFixed(2));
    const decodeTokPerSec =
      generatedIds.length > 0 ? Number(((generatedIds.length * 1000) / decodeMs).toFixed(2)) : 0;

    return {
      generatedIds,
      promptTokens: promptIds.length,
      prefixReusedTokens: common,
      prefillMs: prefRes.prefillMs,
      ttftMs,
      prefillTokPerSec: prefRes.prefillTokPerSec,
      decodeMs,
      decodeTokPerSec,
      step0Top5Logits: prefRes.top5Logits,
      aborted,
      speculativeStats: enableSpec
        ? {
            enabled: true,
            drafted,
            accepted,
            steps: specSteps,
            acceptanceRatePct: drafted > 0 ? Number(((accepted / drafted) * 100).toFixed(2)) : 0,
          }
        : null,
    };
  }

  dispose(): void {
    const bufs = [
      this.slab0,
      this.slab1,
      this.slab2,
      this.auxBuf,
      this.pleHotDataBuf,
      this.pleHotMapBuf,
      this.kvSliding,
      this.kvGlobal0,
      this.kvGlobal1,
      this.tokenIdsBuf,
      this.hiddenBuf,
      this.normBuf,
      this.qkvBuf,
      this.qBuf,
      this.kBuf,
      this.vBuf,
      this.attnOutBuf,
      this.mlpGateUpBuf,
      this.mlpActBuf,
      this.pleRawBuf,
      this.pleProjBuf,
      this.pleCombinedBuf,
      this.pleActBuf,
      this.logitsBuf,
      this.logitsSingleBuf,
      this.argmaxPartValBuf,
      this.argmaxPartIdxBuf,
      this.argmaxSingleOutBuf,
      this.argmaxMultiOutBuf,
      this.argmaxReadBuf,
      this.argmaxReadBufB,
      this.partialAttnBuf,
      this.partialPrefillBuf,
      this.dummyBuf,
      this.uniformBuf,
    ];
    for (const b of bufs) {
      try {
        b?.destroy();
      } catch {}
    }
    this.device?.destroy();
    for (const key of [...this.gpuPrefixSlots.keys()]) {
      this.deleteGpuPrefixSlot(key);
    }
    this.device = null;
    this.pipelines = {};
    this.bindGroups = {};
    this.pleLru.clear();
  }
}
