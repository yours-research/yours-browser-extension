export const ENGINE_PORT_NAME = 'tinygpt-engine';

export type ModelDtype = 'q4f16' | 'q4' | 'fp16';

export interface ModelPreset {
  id: string;
  label: string;
  defaultDtype: ModelDtype;
  supportedDtypes: ModelDtype[];
  /** Approximate text-only model weights (embed_tokens + decoder_model_merged) in GB */
  estimatedWeightGB: Record<ModelDtype, number>;
  notes: string;
}

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
    label: 'Qwen3.5-4B (Custom WGSL)',
    defaultDtype: 'q4f16',
    supportedDtypes: ['q4f16', 'q4'],
    estimatedWeightGB: {
      q4f16: 2.67,
      q4: 2.96,
      fp16: 8.5,
    },
    notes: 'Primary Qwen3.5 hybrid baseline. 24 Gated DeltaNet + 8 Full Attention layers.',
  },
  {
    id: 'jtaylorexe/pragma-exp004-gspo-rl',
    label: 'Pragma EXP-004 GSPO-RL (v81920, -254 MiB)',
    defaultDtype: 'q4f16',
    supportedDtypes: ['q4f16', 'q4'],
    estimatedWeightGB: {
      q4f16: 2.2,
      q4: 2.4,
      fp16: 7.5,
    },
    notes: 'Promoted Qwen3.5-4B GSPO-RL bundle with pruned 81,920-token vocab (-254 MiB LM head VRAM vs stock).',
  },
  {
    id: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
    label: 'Gemma 4 E4B QAT (Custom WGSL)',
    defaultDtype: 'q4f16',
    supportedDtypes: ['q4f16', 'q4'],
    estimatedWeightGB: {
      q4f16: 2.1,
      q4: 2.3,
      fp16: 8.0,
    },
    notes: 'Gemma 4 E4B QAT offline WGSL runtime. 35 sliding + 7 global layers, 18 KV-shared, PLE hot cache.',
  },
  {
    id: 'onnx-community/Qwen3.5-0.8B-ONNX',
    label: 'Qwen3.5-0.8B (Fast Hybrid Sanity)',
    defaultDtype: 'q4f16',
    supportedDtypes: ['q4f16', 'q4'],
    estimatedWeightGB: {
      q4f16: 0.65,
      q4: 0.72,
      fp16: 1.7,
    },
    notes: 'Lightweight Qwen3.5 hybrid baseline for quick smoke testing.',
  },
  {
    id: 'onnx-community/Qwen3-4B-ONNX',
    label: 'Qwen3-4B (Pure Attention Fallback)',
    defaultDtype: 'q4f16',
    supportedDtypes: ['q4f16', 'q4'],
    estimatedWeightGB: {
      q4f16: 2.5,
      q4: 2.8,
      fp16: 8.0,
    },
    notes: 'Fallback pure-GQA baseline for comparing against Qwen3.5 hybrid.',
  },
];

export const BUNDLED_MODEL_IDS = [
  'onnx-community/Qwen3.5-4B-ONNX-OPT',
  'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
] as const;

export type BundledModelId = (typeof BUNDLED_MODEL_IDS)[number];

export function isGemma4ModelId(modelId: string): boolean {
  return modelId.toLowerCase().includes('gemma-4') || modelId.toLowerCase().includes('gemma4');
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  toolName?: string;
  toolCallId?: string;
  toolArgs?: Record<string, unknown>;
}

export interface HardwareProfileSnapshot {
  adapterName: string;
  vendor: string;
  architecture: string;
  deviceMemoryGb: number | null;
  hasF16: boolean;
  hasSubgroups: boolean;
  recommendedTier: WgslComputeTier;
  maxBufferSizeMb: number;
  maxStorageBindingMb: number;
  recommendedModelId: string;
  probedAt: number;
}

export interface ModelCatalogChoice {
  id: string;
  label: string;
  family: string;
  paramsB: number;
  estimatedSizeGb: number;
  contextLength: number;
  source: 'bundled' | 'verified-manifest' | 'custom';
  cachedInOpfs: boolean;
  bundled: boolean;
  manifestVerified: boolean;
  notes?: string;
}

export interface ModelDownloadProgressInfo {
  modelId: string;
  phase:
    | 'probing'
    | 'verifying-manifest'
    | 'downloading'
    | 'verifying-sha'
    | 'writing-opfs'
    | 'complete'
    | 'error';
  bytesDownloaded: number;
  totalBytes: number;
  pct: number;
  mbPerSec: number;
  error?: string;
}

export type EngineBackend = 'ort' | 'wgsl';
export type WgslComputeTier = 'tier1-f16-subgroups' | 'tier2-f16-shared' | 'tier3-f32-universal';
export type WasmVariant = 'asyncify' | 'jspi';
export type YieldMode = 'timeout' | 'message-channel' | 'none';
export type GraphOptimizationLevel = 'all' | 'extended' | 'basic' | 'disabled';
export type WebGpuValidationMode = 'basic' | 'wgpuOnly' | 'disabled';
export type BufferCacheMode = 'default' | 'disabled' | 'lazyRelease' | 'simple' | 'bucket';
export type GpuPacingMode = 'max-speed' | 'balanced' | 'smooth-ui';
export type ExecutionHost = 'worker' | 'offscreen-main';
export type LoadSource = 'bundled' | 'opfs' | 'network';

export interface UiSmoothnessMetrics {
  avgFps: number;
  p50FrameMs: number;
  p95FrameMs: number;
  p99FrameMs: number;
  maxFrameMs: number;
  jankPctOver33ms: number;
  freezePctOver100ms: number;
  avgInputLatencyMs: number;
  frameCount: number;
}

export interface EngineConfig {
  modelId: string;
  dtype: ModelDtype;
  /** Tokens per prefill slice to prevent GPU TDR/watchdog timeouts; 0 disables chunking */
  chunkSize: number;
  maxNewTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  /** False skips <think> chain-of-thought and enables multi-turn prefix cache reuse */
  enableThinking: boolean;
  /** Theoretical GPU memory bandwidth in GB/s for Decision Gate utilization metric (RTX 3090 = 936) */
  gpuBandwidthGBs: number;
  /** ONNX Runtime Web WASM flavor ('asyncify' default vs 'jspi') */
  wasmVariant: WasmVariant;
  /** Cooperative yield strategy between prefill chunks / decode steps */
  yieldMode: YieldMode;
  /** Enables fast greedy float16 LUT argmax, pooled attention masks, and throttled telemetry */
  fastDecodePath: boolean;
  /** Omits progress_callback when loading warm OPFS cache to avoid JS stream chunking */
  fastOpfsLoad: boolean;
  /** ORT session graph optimization level */
  graphOptimizationLevel: GraphOptimizationLevel;
  /** ORT WebGPU EP validation mode */
  validationMode: WebGpuValidationMode;
  /** ORT WebGPU EP storage buffer cache mode */
  storageBufferCacheMode: BufferCacheMode;
  /** ORT WebGPU EP FP32 matmul accumulation override (null = ORT default) */
  enableMatmulFp32Accumulation: boolean | null;
  /** Inference engine backend ('ort' = Transformers.js/ORT WebGPU, 'wgsl' = custom Rust/WGSL engine) */
  backend: EngineBackend;
  /** KV cache precision used by the custom WGSL engine */
  wgslKvMode: WgslKvMode;
  /** Maximum context capacity allocated for the custom WGSL engine */
  wgslMaxContext: number;
  /** GPU pacing profile controlling chunk size, layer splits, and UI breathers */
  gpuPacingMode: GpuPacingMode;
  /** Host environment executing the inference engine */
  executionHost: ExecutionHost;
  /** Enables zero-weight suffix speculative decoding for greedy generation */
  enableSpeculative: boolean;
  /** Optional compute tier override ('tier1-f16-subgroups' | 'tier2-f16-shared' | 'tier3-f32-universal') */
  forceComputeTier?: WgslComputeTier | null;
  /** Optional external/local HTTP base URL serving a WGSL slab bundle (used for model benchmarking) */
  bundleBaseUrl?: string | null;
}

export const DEFAULT_ENGINE_CONFIG: EngineConfig = {
  modelId: MODEL_PRESETS[0].id,
  dtype: 'q4f16',
  chunkSize: 64,
  maxNewTokens: 1024,
  temperature: 0.0,
  topP: 0.95,
  topK: 20,
  enableThinking: false,
  gpuBandwidthGBs: 936,
  wasmVariant: 'asyncify',
  yieldMode: 'none',
  fastDecodePath: true,
  fastOpfsLoad: true,
  graphOptimizationLevel: 'all',
  validationMode: 'basic',
  storageBufferCacheMode: 'default',
  enableMatmulFp32Accumulation: null,
  backend: 'wgsl',
  wgslKvMode: 'int8',
  wgslMaxContext: 131072,
  gpuPacingMode: 'balanced',
  executionHost: 'worker',
  enableSpeculative: true,
  forceComputeTier: null,
  bundleBaseUrl: null,
};

export interface LoadBreakdown {
  opfsReadMs: number;
  tokenizerMs: number;
  modelLoadMs: number;
  totalMs: number;
}

export type ModelLoadStatus =
  | 'unloaded'
  | 'downloading'
  | 'compiling'
  | 'ready'
  | 'generating'
  | 'benchmarking'
  | 'error';

export interface TelemetryState {
  status: ModelLoadStatus;
  modelId: string;
  dtype: ModelDtype;
  loadProgressPct: number;
  loadPhaseLabel: string;
  coldLoadMs: number | null;
  loadedFromCache: boolean | null;
  loadSource: LoadSource | null;
  executionHost: ExecutionHost;
  loadBreakdown: LoadBreakdown | null;
  tokensPerSec: number | null;
  prefillTokPerSec: number | null;
  ttftMs: number | null;
  activeContextTokens: number;
  estVramMb: number;
  kvCacheMb: number;
  deltaNetStateMb: number;
  prefillChunkProgress: { done: number; total: number } | null;
  prefixReused: boolean | null;
  hasCpuSnapshot: boolean;
  cpuSnapshotTokens: number | null;
  lastError: string | null;
  deviceLostReason: string | null;
  maxContextReached: number;
  speculativeStats: {
    drafted: number;
    accepted: number;
    steps: number;
    acceptanceRatePct: number;
    tau?: number;
    alpha?: number;
    avgDraftLen?: number;
    skippedSteps?: number;
    speedupMultiplier?: number;
  } | null;
  baselineDecodeTokPerSec: number | null;
  vramDeltaMb: number | null;
  computeTier: WgslComputeTier | null;
  allocatedKvCapacity: number;
  maxKvCapacity: number;
  basePromptCached: boolean;
  basePromptTokens: number;
  lastSlotRestoreMs: number | null;
  displayHz: number;
  backgroundPrefillState: 'idle' | 'prefilling' | 'warm';
  activePrefixSlot: string | null;
}

export const INITIAL_TELEMETRY_STATE: TelemetryState = {
  status: 'unloaded',
  modelId: DEFAULT_ENGINE_CONFIG.modelId,
  dtype: DEFAULT_ENGINE_CONFIG.dtype,
  loadProgressPct: 0,
  loadPhaseLabel: 'Idle',
  coldLoadMs: null,
  loadedFromCache: null,
  loadSource: null,
  executionHost: 'worker',
  loadBreakdown: null,
  tokensPerSec: null,
  prefillTokPerSec: null,
  ttftMs: null,
  activeContextTokens: 0,
  estVramMb: 0,
  kvCacheMb: 0,
  deltaNetStateMb: 0,
  prefillChunkProgress: null,
  prefixReused: null,
  hasCpuSnapshot: false,
  cpuSnapshotTokens: null,
  lastError: null,
  deviceLostReason: null,
  maxContextReached: 0,
  speculativeStats: null,
  baselineDecodeTokPerSec: null,
  vramDeltaMb: null,
  computeTier: null,
  allocatedKvCapacity: 0,
  maxKvCapacity: 0,
  basePromptCached: false,
  basePromptTokens: 0,
  lastSlotRestoreMs: null,
  displayHz: 60,
  backgroundPrefillState: 'idle',
  activePrefixSlot: null,
};

export interface WebGpuProbeInfo {
  supported: boolean;
  adapterInfo: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  } | null;
  features: string[];
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  adapterLimits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupStorageSize: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxStorageBuffersPerShaderStage: number;
  } | null;
  grantedLimits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
  } | null;
  error: string | null;
  probedAt: string;
}

export interface OpfsFileSummary {
  name: string;
  sizeBytes: number;
}

export interface OpfsStatusInfo {
  supported: boolean;
  persisted: boolean | null;
  quotaBytes: number | null;
  usageBytes: number | null;
  modelFilesCount: number;
  modelCacheBytes: number;
  spikeFiles: OpfsFileSummary[];
  error: string | null;
}

export interface OpfsSpikeResult {
  action: 'write' | 'verify' | 'clear';
  ok: boolean;
  totalBytes: number;
  shardCount: number;
  durationMs: number;
  throughputMBs: number;
  details: string;
}

export interface DecodeStepBreakdown {
  avgForwardMs: number;
  avgLogitsMs: number;
  avgSampleMs: number;
  avgYieldMs: number;
}

export interface BenchmarkPoint {
  targetTokens: number;
  actualPromptTokens: number;
  decodeTokens: number;
  chunkSize: number;
  prefillMs: number;
  ttftMs: number;
  prefillTokPerSec: number;
  decodeTokPerSec: number;
  kvCacheMb: number;
  deltaNetStateMb: number;
  estVramMb: number;
  achievedBandwidthGBs: number;
  bandwidthUtilizationPct: number;
  breakdown?: DecodeStepBreakdown;
  status: 'ok' | 'oom' | 'device_lost' | 'aborted' | 'error';
  error?: string;
}

export interface BenchmarkSuiteResult {
  startedAt: string;
  finishedAt: string;
  modelId: string;
  dtype: ModelDtype;
  gpuBandwidthGBs: number;
  points: BenchmarkPoint[];
  maxSafeTokens: number;
  oomTokenCount: number | null;
  recommendation: string;
}

export interface NiahResult {
  targetTokens: number;
  actualTokens: number;
  needleDepthPct: number;
  passkey: string;
  responseText: string;
  passed: boolean;
  ttftMs: number;
  prefillTokPerSec: number;
  decodeTokPerSec: number;
  error?: string;
}

export interface LogEntry {
  ts: string;
  level: 'info' | 'warn' | 'error';
  source: 'offscreen' | 'sidepanel' | 'webgpu' | 'opfs';
  message: string;
}

export type BackgroundMessage = {
  type: 'ENSURE_OFFSCREEN';
};

export type BackgroundResponse =
  | { ok: true }
  | { ok: false; error: string };

export type WgslGemvVariant = 'q4-scalar' | 'q4-vec4-f16' | 'q4-subgroups-f16';
export type WgslKvMode = 'fp16' | 'int8' | 'int4';

export interface WgslBenchConfig {
  gemvVariant: WgslGemvVariant;
  kvMode: WgslKvMode;
  contextLengths: number[];
  decodeTokens: number;
  warmupTokens?: number;
  gpuBandwidthGBs?: number;
}

export interface WgslKernelCorrectnessEntry {
  kernel: string;
  passed: boolean;
  maxAbsError: number;
  details: string;
}

export interface WgslBenchPoint {
  contextLength: number;
  decodeTokens: number;
  decodeMs: number;
  decodeTokPerSec: number;
  kvCacheMb: number;
  deltaNetStateMb: number;
  weightMb: number;
  estVramMb: number;
  achievedBandwidthGBs: number;
  bandwidthUtilizationPct: number;
  status: 'ok' | 'oom' | 'device_lost' | 'error';
  error?: string;
}

export interface WgslBenchResult {
  startedAt: string;
  finishedAt: string;
  gemvVariant: WgslGemvVariant;
  kvMode: WgslKvMode;
  gpuBandwidthGBs: number;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
  maxBufferSize: number;
  allocatedWeightMb: number;
  allocatedKvMb: number;
  allocatedDeltaNetMb: number;
  pipelineCompileMs: number;
  correctness: WgslKernelCorrectnessEntry[];
  points: WgslBenchPoint[];
  error?: string;
}

export interface WgslOpfsStageTiming {
  stagingChunkMib: number;
  durationMs: number;
  throughputMBs: number;
  totalBytes: number;
}

export interface WgslOpfsLoadResult {
  action: 'prepare' | 'load' | 'clear';
  ok: boolean;
  totalBytes: number;
  timings: WgslOpfsStageTiming[];
  details: string;
}

export type ClientToEngineMessage =
  | { type: 'GET_STATE' }
  | { type: 'UPDATE_CONFIG'; config: Partial<EngineConfig> }
  | { type: 'PROBE_WEBGPU' }
  | { type: 'CHECK_OPFS' }
  | { type: 'RUN_OPFS_SPIKE'; totalMiB: number; shardMiB: number }
  | { type: 'VERIFY_OPFS_SPIKE' }
  | { type: 'CLEAR_OPFS_SPIKE' }
  | { type: 'CLEAR_MODEL_CACHE' }
  | { type: 'LOAD_MODEL'; config?: Partial<EngineConfig> }
  | { type: 'UNLOAD_MODEL' }
  | { type: 'RESET_CONTEXT'; restoreBasePrompt?: boolean }
  | { type: 'SNAPSHOT_CONTEXT' }
  | { type: 'RESTORE_CONTEXT' }
  | { type: 'SET_DISPLAY_HZ'; hz: number }
  | {
      type: 'PREFILL_BACKGROUND';
      requestId?: string;
      slotKey: string;
      messages: ChatMessage[];
      isDraft?: boolean;
      url?: string;
      contentHash?: string;
    }
  | { type: 'ABORT_BACKGROUND_PREFILL'; requestId?: string; slotKey?: string }
  | { type: 'EVICT_SLOT'; requestId: string; slotKey: string }
  | {
      type: 'GENERATE';
      requestId: string;
      messages: ChatMessage[];
      config?: Partial<EngineConfig>;
      groundingText?: string;
      stopOnToolCall?: boolean;
      prefixSlotKey?: string;
      tools?: unknown;
    }
  | { type: 'PROBE_HARDWARE_CATALOG' }
  | {
      type: 'DOWNLOAD_VERIFIED_MODEL';
      modelId: string;
      manifestUrl?: string;
    }
  | { type: 'CANCEL_MODEL_DOWNLOAD'; modelId?: string }
  | { type: 'ABORT'; requestId: string }
  | {
      type: 'RUN_BENCHMARK_SUITE';
      contextLengths: number[];
      decodeTokens: number;
      chunkSize?: number;
    }
  | {
      type: 'RUN_NIAH';
      targetTokens: number;
      depthRatio: number;
      chunkSize?: number;
    }
  | { type: 'RUN_WGSL_BENCH'; config: WgslBenchConfig }
  | { type: 'RUN_SPEC_MICROBENCH' }
  | {
      type: 'RUN_WGSL_OPFS_BENCH';
      action: 'prepare' | 'load' | 'clear';
      totalMb?: number;
      chunkMibList?: number[];
    };

export interface GenerationTokenTrace {
  modelId: string;
  promptIds: number[];
  generatedIds: number[];
  promptText: string;
  generatedText: string;
  speculativeStats: TelemetryState['speculativeStats'];
}

export interface SpecMicrobenchResult {
  modelId: string;
  t1Ms: number;
  t2Ms: number;
  t3Ms: number;
  t5Ms: number;
  t8Ms: number;
  t5RejectMs: number;
  ratioT5ToT1: number;
  ratioT8ToT1: number;
  gatePass: boolean;
  tree?: {
    t2Ms: number;
    t3Ms: number;
    t5Ms: number;
    t8Ms: number;
    t12Ms: number;
    t16Ms: number;
    t5RejectMs: number;
    ratioT5ToT1: number;
    ratioT8ToT1: number;
    ratioT16ToT1: number;
    rejectOverheadMs: number;
    chainRejectOverheadMs: number;
  };
  error?: string;
}

export type EngineToClientMessage =
  | {
      type: 'STATE_SNAPSHOT';
      telemetry: TelemetryState;
      config: EngineConfig;
      probeInfo: WebGpuProbeInfo | null;
      opfsStatus: OpfsStatusInfo | null;
      lastBenchmark: BenchmarkSuiteResult | null;
      lastNiah: NiahResult | null;
      lastWgslBench: WgslBenchResult | null;
      lastWgslOpfs: WgslOpfsLoadResult | null;
      logs: LogEntry[];
    }
  | { type: 'TELEMETRY_UPDATE'; requestId?: string; telemetry: TelemetryState }
  | { type: 'CONFIG_UPDATE'; config: EngineConfig }
  | { type: 'HARDWARE_PROFILE_UPDATE'; hardwareProfile: HardwareProfileSnapshot; catalog: ModelCatalogChoice[] }
  | { type: 'MODEL_DOWNLOAD_PROGRESS'; progress: ModelDownloadProgressInfo }
  | { type: 'MODEL_DOWNLOAD_COMPLETE'; modelId: string; cachedInOpfs: boolean; catalog: ModelCatalogChoice[] }
  | { type: 'MODEL_DOWNLOAD_ERROR'; modelId: string; error: string }
  | { type: 'WEBGPU_PROBE_RESULT'; probeInfo: WebGpuProbeInfo }
  | { type: 'OPFS_STATUS'; opfsStatus: OpfsStatusInfo }
  | { type: 'OPFS_SPIKE_PROGRESS'; message: string; pct: number }
  | { type: 'OPFS_SPIKE_RESULT'; result: OpfsSpikeResult }
  | { type: 'BENCHMARK_PROGRESS'; message: string; point?: BenchmarkPoint }
  | { type: 'BENCHMARK_RESULT'; result: BenchmarkSuiteResult }
  | { type: 'NIAH_PROGRESS'; message: string }
  | { type: 'NIAH_RESULT'; result: NiahResult }
  | { type: 'WGSL_BENCH_PROGRESS'; message: string }
  | { type: 'WGSL_BENCH_RESULT'; result: WgslBenchResult }
  | { type: 'SPEC_MICROBENCH_RESULT'; result: SpecMicrobenchResult }
  | { type: 'WGSL_OPFS_BENCH_RESULT'; result: WgslOpfsLoadResult }
  | { type: 'LOG'; entry: LogEntry }
  | { type: 'PREFILL_COMPLETE'; requestId: string; slotKey: string; tokens: number; elapsedMs: number }
  | { type: 'PREFILL_ERROR'; requestId: string; slotKey: string; error: string }
  | { type: 'EVICT_COMPLETE'; requestId: string; slotKey: string }
  | { type: 'TOKEN_DELTA'; requestId: string; delta: string; tokenCount: number; isReasoning?: boolean }
  | {
      type: 'GENERATION_COMPLETE';
      requestId: string;
      finishReason: 'stop' | 'length' | 'abort';
      trace?: GenerationTokenTrace;
    }
  | { type: 'GENERATION_ERROR'; requestId: string; error: string };
