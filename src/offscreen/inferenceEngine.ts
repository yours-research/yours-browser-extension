import { AutoTokenizer, PreTrainedTokenizer, env } from '@huggingface/transformers';
import { buildSlot0Envelope } from '@yours/harness/compaction';
import { CANONICAL_TOOL_DEFINITIONS } from '@yours/tools/schemas';
import {
  BUNDLED_MODEL_IDS,
  DEFAULT_ENGINE_CONFIG,
  INITIAL_TELEMETRY_STATE,
  MODEL_PRESETS,
  isGemma4ModelId,
  type BenchmarkPoint,
  type BenchmarkSuiteResult,
  type ChatMessage,
  type EngineConfig,
  type ExecutionHost,
  type GenerationPhaseStep,
  type GenerationPhaseTrace,
  type GenerationTokenTrace,
  type LogEntry,
  type NiahResult,
  type OpfsSpikeResult,
  type OpfsStatusInfo,
  type SpecMicrobenchResult,
  type TelemetryState,
  type WebGpuProbeInfo,
  type YieldMode,
} from '../shared/protocol.ts';
import { buildHarnessSystemPrompt } from '../shared/systemPrompt.ts';
import { buildNeedleInHaystackPrompt, buildSyntheticTokenIds, verifyNeedleAnswer } from './benchmarks.ts';
import { resolveExtensionAssetUrl } from './gpuPacing.ts';
import { Gemma4ExtensionAdapter, type WgslRuntimeEngine } from './gemma4ExtensionAdapter.ts';
import { clearWgslWeightCache } from './onnxWeightLoader.ts';
import { OpfsModelCache } from './opfsCache.ts';
import {
  BASE_SYSTEM_SLOT_ID,
  PrefixCacheManager,
  isExactTokenPrefix,
} from './prefixCacheManager.ts';
import {
  AdaptiveVerifyCostModel,
  EngineASpeculator,
  type ScoredDraftChain,
  type ToolSchema,
  type ParamType,
} from './adaptiveSpeculator.ts';
import { computeBandwidthStats, isMessagePrefix, type SamplingOptions } from './sampling.ts';
import { installWebGpuLimitPatch, probeWebGpu } from './webgpuLimits.ts';
import { WgslQwen35Engine, MAX_SPEC_DRAFT, MAX_TREE_DRAFT } from './wgslRealEngine.ts';
import { OffscreenSpecAdapter } from './cycle17/rc1/wasmEngineAdapter.ts';
import { GdnDraftRuntime, shaderAssets } from './cycle17/draft/gdnEagle3Runtime.ts';
import { loadGdnBundle } from './cycle17/draft/gdnEagle3Loader.ts';
import {
  runEngineBBench,
  toAdapterOptions,
  type EngineBBenchReport,
} from './cycle17/bench/engineBHeadBench.ts';
import {
  IncrementalTokenDecoder,
  loadSpecCalibration,
  saveSpecCalibration,
  shouldEnableEngineB,
} from './specCalibration.ts';

const EXP007B_DRAFT_HEAD_SHA256 =
  'b456e39bb8e541809971527ea01092ba6f3f76efba47bc9e75b8b3612dbef3e1';
const RC1_CHAIN_STEP_MS = [10.33, 12.63, 15.64, 18.695, 21.75, 24.37333, 26.99667, 29.62];

const CONTROL_TOKENS_TO_STRIP_RE =
  /<\|im_end\|>|<\|endoftext\|>|<end_of_turn>|<turn\|>|<\|turn>model\r?\n?|<\|im_start\|>assistant\r?\n?|<eos>|<bos>|<pad>/g;

/**
 * Decodes generated token IDs while preserving reasoning delimiters (including Gemma 4's
 * special `<|channel>thought` / `<channel|>` tokens) and stripping only turn/EOS control tokens.
 * Prepends `<think>\n` when the prompt template already pre-opened `<think>` before token 0.
 */
export function decodeGeneratedText(
  tokenizer: Pick<PreTrainedTokenizer, 'decode'>,
  tokenIds: number[],
  options?: { preOpenedThink?: boolean },
): string {
  const raw = tokenizer.decode(tokenIds, { skip_special_tokens: false });
  let cleaned = raw.replace(CONTROL_TOKENS_TO_STRIP_RE, '');
  if (
    options?.preOpenedThink &&
    cleaned.length > 0 &&
    !/^\s*(?:<think>|<thought>|<\|channel>thought)/.test(cleaned)
  ) {
    cleaned = `<think>\n${cleaned}`;
  }
  return cleaned;
}

export function buildIncrementalTurnPrompt(
  chatTemplate: string,
  suffixMessages: readonly ChatMessage[],
): string | null {
  if (chatTemplate.includes('<|im_start|>')) {
    let out = '<|im_end|>\n';
    for (const msg of suffixMessages) {
      out += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    }
    out += '<|im_start|>assistant\n<think>\n\n</think>\n\n';
    return out;
  }
  if (chatTemplate.includes('<|turn>')) {
    let out = '<turn|>\n';
    for (const msg of suffixMessages) {
      const role = msg.role === 'assistant' ? 'model' : msg.role;
      out += `<|turn>${role}\n${msg.content.trim()}<turn|>\n`;
    }
    out += '<|turn>model\n';
    return out;
  }
  return null;
}

export interface EngineCallbacks {
  onTelemetry: (telemetry: TelemetryState) => void;
  onConfig: (config: EngineConfig) => void;
  onLog: (entry: LogEntry) => void;
  onProbeResult: (info: WebGpuProbeInfo) => void;
  onOpfsStatus: (status: OpfsStatusInfo) => void;
  onSpikeProgress: (message: string, pct: number) => void;
  onSpikeResult: (result: OpfsSpikeResult) => void;
  onBenchmarkProgress: (message: string, point?: BenchmarkPoint) => void;
  onBenchmarkResult: (result: BenchmarkSuiteResult) => void;
  onSpecMicrobenchResult?: (result: SpecMicrobenchResult) => void;
  onNiahProgress: (message: string) => void;
  onNiahResult: (result: NiahResult) => void;
  onTokenDelta: (requestId: string, delta: string, tokenCount: number) => void;
  onGenerationComplete: (
    requestId: string,
    finishReason: 'stop' | 'length' | 'abort',
    trace?: GenerationTokenTrace,
  ) => void;
  onGenerationError: (requestId: string, error: string) => void;
  onPrefillComplete?: (
    requestId: string,
    slotKey: string,
    tokens: number,
    elapsedMs: number,
  ) => void;
  onPrefillError?: (requestId: string, slotKey: string, error: string) => void;
  onEvictComplete?: (requestId: string, slotKey: string) => void;
}

let yieldChannel: MessageChannel | null = null;
const yieldWaiters: Array<() => void> = [];

function yieldFast(mode: YieldMode): Promise<void> {
  if (mode === 'none') {
    return Promise.resolve();
  }
  if (mode === 'message-channel' && typeof MessageChannel !== 'undefined') {
    if (!yieldChannel) {
      yieldChannel = new MessageChannel();
      yieldChannel.port1.onmessage = () => {
        const next = yieldWaiters.shift();
        next?.();
      };
    }
    return new Promise((resolve) => {
      yieldWaiters.push(resolve);
      yieldChannel!.port2.postMessage(null);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class InferenceEngine {
  readonly opfsCache = new OpfsModelCache();
  private readonly callbacks: EngineCallbacks;
  private readonly assetBaseUrl?: string;
  private readonly executionHost: ExecutionHost;

  private config: EngineConfig;
  private telemetry: TelemetryState;
  private probeInfo: WebGpuProbeInfo | null = null;
  private opfsStatus: OpfsStatusInfo | null = null;
  private lastBenchmark: BenchmarkSuiteResult | null = null;
  private lastNiah: NiahResult | null = null;
  private logs: LogEntry[] = [];

  private wgslEngine: WgslRuntimeEngine | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private eosTokenIds = new Set<number>();

  private cachedMessages: ChatMessage[] = [];
  private cachedTokenCount = 0;
  private cachedTokenIds: number[] = [];
  private pendingLiveTokenIds: number[] = [];
  private cachedThinkingMode = false;
  private activeBgSignature: string | null = null;
  private wgslSnapshot: { seqPos: number } | null = null;
  private wgslSnapshotMeta: {
    cachedMessages: ChatMessage[];
    cachedThinkingMode: boolean;
    cachedTokenIds: number[];
  } | null = null;

  private activeRequestId: string | null = null;
  private activeAbortController: AbortController | null = null;
  private abortedRequests = new Set<string>();
  private loadingPromise: Promise<void> | null = null;
  private taskQueue: Promise<void> = Promise.resolve();
  private envConfigured = false;
  private prefixCache: PrefixCacheManager | null = null;
  private displayHz = 60;
  private bgAbortController: AbortController | null = null;
  private activeBgSlotKey: string | null = null;
  private activeBgRequestId: string | null = null;
  private bgPrefillSeq = 0;
  private engineASpeculator: EngineASpeculator | null = null;
  private engineATokenizerRef: PreTrainedTokenizer | null = null;
  private offscreenSpecAdapter: OffscreenSpecAdapter | null = null;
  private offscreenSpecAdapterPromise: Promise<OffscreenSpecAdapter | null> | null = null;
  private offscreenSpecTokenizerRef: PreTrainedTokenizer | null = null;
  private offscreenSpecDeviceRef: GPUDevice | null = null;
  private adapterName = 'webgpu-default';
  private engineBActiveForAdapter = false;
  private measuredChainStepMs: number[] | null = null;
  private measuredEngineBReport: EngineBBenchReport | null = null;
  private measuredEngineBHeadMs: number | null = null;
  private measuredRejectPenaltyMs: number | null = null;
  private pendingPhaseEnqueue = new Map<
    string,
    { enqueue: number; bgPrefillActiveAtEnqueue: boolean }
  >();
  private engineACostModel = new AdaptiveVerifyCostModel({
    maxM: 1 + MAX_SPEC_DRAFT,
    initialT1Ms: 9.1,
    initialSlopeMs: 3.1,
    initialRejectPenaltyMs: 1.5,
  });
  private engineATreeCostModel = new AdaptiveVerifyCostModel({
    maxM: 1 + MAX_TREE_DRAFT,
    initialT1Ms: 14.5,
    initialSlopeMs: 4.8,
    initialRejectPenaltyMs: 0,
  });

  constructor(
    callbacks: EngineCallbacks,
    options?: { assetBaseUrl?: string; executionHost?: ExecutionHost },
  ) {
    this.callbacks = callbacks;
    this.assetBaseUrl = options?.assetBaseUrl;
    this.executionHost = options?.executionHost ?? 'worker';
    this.config = {
      ...DEFAULT_ENGINE_CONFIG,
      backend: 'wgsl',
      wgslKvMode: 'int8',
      wgslMaxContext: 131072,
      modelId: DEFAULT_ENGINE_CONFIG.modelId,
      dtype: 'q4f16',
      executionHost: this.executionHost,
    };
    this.telemetry = {
      ...INITIAL_TELEMETRY_STATE,
      executionHost: this.executionHost,
    };
    installWebGpuLimitPatch((info) => {
      this.log('error', 'webgpu', `WebGPU device lost (${info.reason}): ${info.message}`);
      this.updateTelemetry({
        status: 'error',
        deviceLostReason: info.reason,
        lastError: `WebGPU device lost at ${this.telemetry.activeContextTokens} tokens: ${info.message}`,
      });
    });
  }

  private log(level: LogEntry['level'], source: LogEntry['source'], message: string): void {
    const entry: LogEntry = {
      ts: new Date().toISOString(),
      level,
      source,
      message,
    };
    this.logs.push(entry);
    if (this.logs.length > 200) {
      this.logs.splice(0, this.logs.length - 200);
    }
    this.callbacks.onLog(entry);
  }

  private updateTelemetry(patch: Partial<TelemetryState>): void {
    this.telemetry = { ...this.telemetry, ...patch };
    this.callbacks.onTelemetry(this.telemetry);
  }

  private configureTransformersEnv(): void {
    if (this.envConfigured) {
      return;
    }
    this.envConfigured = true;

    env.useWasmCache = false;
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.useBrowserCache = false;
    env.useCustomCache = false;

    this.log(
      'info',
      'offscreen',
      `Configured offline local Transformers.js tokenizer (host=${this.executionHost}).`,
    );
  }

  getStateSnapshot() {
    return {
      telemetry: this.telemetry,
      config: this.config,
      probeInfo: this.probeInfo,
      opfsStatus: this.opfsStatus,
      lastBenchmark: this.lastBenchmark,
      lastNiah: this.lastNiah,
      lastWgslBench: null,
      lastWgslOpfs: null,
      logs: this.logs,
    };
  }

  async updateConfig(patch: Partial<EngineConfig>): Promise<void> {
    const nextMaxContext =
      typeof patch.wgslMaxContext === 'number' &&
      Number.isFinite(patch.wgslMaxContext) &&
      patch.wgslMaxContext > 0
        ? Math.min(131072, Math.round(patch.wgslMaxContext))
        : this.config.wgslMaxContext;
    const contextCapChanged = nextMaxContext !== this.config.wgslMaxContext;
    const nextForceTier =
      patch.forceComputeTier !== undefined
        ? patch.forceComputeTier
        : (this.config.forceComputeTier ?? null);
    const tierChanged = nextForceTier !== (this.config.forceComputeTier ?? null);

    let normalizedPatchModelId =
      typeof patch.modelId === 'string' ? patch.modelId.trim() : undefined;
    let derivedHfBundleUrl: string | null = null;
    if (normalizedPatchModelId && normalizedPatchModelId.startsWith('hf://')) {
      const repoId = normalizedPatchModelId.slice('hf://'.length).replace(/^\/+|\/+$/g, '');
      normalizedPatchModelId = repoId;
      derivedHfBundleUrl = `https://huggingface.co/${repoId}/resolve/main`;
    }

    const explicitBundleBaseUrl =
      patch.bundleBaseUrl !== undefined
        ? (typeof patch.bundleBaseUrl === 'string' && patch.bundleBaseUrl.trim().length > 0
            ? patch.bundleBaseUrl.trim().replace(/\/+$/, '')
            : null)
        : (this.config.bundleBaseUrl ?? null);

    const isKnownPreset =
      typeof normalizedPatchModelId === 'string' &&
      MODEL_PRESETS.some((p) => p.id === normalizedPatchModelId);
    const isBundledPatchModel =
      typeof normalizedPatchModelId === 'string' &&
      (BUNDLED_MODEL_IDS as readonly string[]).includes(normalizedPatchModelId);

    const nextBundleBaseUrl =
      patch.bundleBaseUrl !== undefined
        ? explicitBundleBaseUrl
        : derivedHfBundleUrl ??
          (normalizedPatchModelId && !isBundledPatchModel && isKnownPreset
            ? `https://huggingface.co/${normalizedPatchModelId}/resolve/main`
            : isBundledPatchModel
              ? null
              : explicitBundleBaseUrl);
    const bundleChanged = nextBundleBaseUrl !== (this.config.bundleBaseUrl ?? null);
    const nextModelId =
      typeof normalizedPatchModelId === 'string' && normalizedPatchModelId.length > 0
        ? (isBundledPatchModel || Boolean(nextBundleBaseUrl)
            ? normalizedPatchModelId
            : DEFAULT_ENGINE_CONFIG.modelId)
        : this.config.modelId;
    const modelChanged = nextModelId !== this.config.modelId;

    this.config = {
      ...this.config,
      ...patch,
      backend: 'wgsl',
      wgslKvMode: 'int8',
      wgslMaxContext: nextMaxContext,
      forceComputeTier: nextForceTier,
      bundleBaseUrl: nextBundleBaseUrl,
      modelId: nextModelId,
      dtype: 'q4f16',
    };
    this.callbacks.onConfig(this.config);

    if ((modelChanged || bundleChanged || contextCapChanged || tierChanged) && this.wgslEngine) {
      this.log('info', 'offscreen', 'WGSL engine model/geometry/tier config changed; unloading active session.');
      await this.unloadModel();
    } else {
      this.updateTelemetry({
        modelId: this.config.modelId,
        dtype: this.config.dtype,
        executionHost: this.executionHost,
      });
    }
  }

  async runProbeWebGpu(): Promise<WebGpuProbeInfo> {
    const info = await probeWebGpu();
    this.probeInfo = info;
    if (info.supported) {
      this.log(
        'info',
        'webgpu',
        `WebGPU OK: ${info.adapterInfo?.vendor ?? 'unknown'} | shader-f16=${info.hasShaderF16}, subgroups=${info.hasSubgroups}, maxBuffer=${((info.grantedLimits?.maxBufferSize ?? 0) / (1024 * 1024)).toFixed(0)} MiB`,
      );
    } else {
      this.log('error', 'webgpu', `WebGPU probe failed: ${info.error}`);
    }
    this.callbacks.onProbeResult(info);
    return info;
  }

  async refreshOpfsStatus(): Promise<OpfsStatusInfo> {
    const status = await this.opfsCache.getOpfsStatus();
    this.opfsStatus = status;
    this.callbacks.onOpfsStatus(status);
    return status;
  }

  async runOpfsSpike(totalMiB: number, shardMiB: number): Promise<void> {
    try {
      this.log('info', 'opfs', `Starting Phase 0 OPFS spike: ${totalMiB} MiB in ${shardMiB} MiB shards...`);
      const result = await this.opfsCache.runOpfsSpike(totalMiB, shardMiB, (msg, pct) => {
        this.callbacks.onSpikeProgress(msg, pct);
      });
      this.log('info', 'opfs', result.details);
      this.callbacks.onSpikeResult(result);
      await this.refreshOpfsStatus();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log('error', 'opfs', `OPFS spike failed: ${msg}`);
      this.callbacks.onSpikeResult({
        action: 'write',
        ok: false,
        totalBytes: 0,
        shardCount: 0,
        durationMs: 0,
        throughputMBs: 0,
        details: msg,
      });
    }
  }

  async verifyOpfsSpike(): Promise<void> {
    const result = await this.opfsCache.verifyOpfsSpike((msg, pct) => {
      this.callbacks.onSpikeProgress(msg, pct);
    });
    this.log(result.ok ? 'info' : 'error', 'opfs', result.details);
    this.callbacks.onSpikeResult(result);
    await this.refreshOpfsStatus();
  }

  async clearOpfsSpike(): Promise<void> {
    const result = await this.opfsCache.clearOpfsSpike();
    this.log('info', 'opfs', result.details);
    this.callbacks.onSpikeResult(result);
    await this.refreshOpfsStatus();
  }

  async clearModelCache(): Promise<void> {
    await this.unloadModel();
    await this.opfsCache.clearModelCache().catch(() => {});
    await clearWgslWeightCache(this.opfsCache).catch(() => {});
    this.log('info', 'opfs', 'Purged OPFS model cache.');
    await this.refreshOpfsStatus().catch(() => {});
  }

  private isEngineReady(): boolean {
    return Boolean(this.tokenizer && this.wgslEngine);
  }

  async loadModel(overrideConfig?: Partial<EngineConfig>): Promise<void> {
    if (overrideConfig) {
      await this.updateConfig(overrideConfig);
    }

    if (this.isEngineReady()) {
      return;
    }

    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    this.loadingPromise = (async () => {
      this.configureTransformersEnv();

      const startMs = performance.now();
      const isGemma4 = isGemma4ModelId(this.config.modelId);
      const modelLabel = isGemma4 ? 'Gemma 4 E4B QAT' : 'Pragma-1-4B RC1';
      const assetSubdir = isGemma4 ? 'models/gemma4-e4b' : 'models/pragma-1-4b-rc1';

      this.updateTelemetry({
        status: 'downloading',
        modelId: this.config.modelId,
        dtype: this.config.dtype,
        executionHost: this.executionHost,
        loadProgressPct: 0,
        loadPhaseLabel: `Loading bundled ${modelLabel} tokenizer & weights...`,
        loadBreakdown: null,
        lastError: null,
        deviceLostReason: null,
      });

      this.log(
        'info',
        'offscreen',
        `Loading bundled ${this.config.modelId} on WGSL backend (kv=int8, maxContext=${this.config.wgslMaxContext}, pacing=${this.config.gpuPacingMode}, host=${this.executionHost})...`,
      );

      const tokStartMs = performance.now();
      const customBundleUrl =
        typeof this.config.bundleBaseUrl === 'string' && this.config.bundleBaseUrl.trim().length > 0
          ? this.config.bundleBaseUrl.trim().replace(/\/+$/, '')
          : null;
      const tokenizerUrl = customBundleUrl ?? resolveExtensionAssetUrl(assetSubdir, this.assetBaseUrl);
      let extraEosIds: number[] = [];
      let tokenizer: PreTrainedTokenizer;
      try {
        tokenizer = await AutoTokenizer.from_pretrained(tokenizerUrl, {
          local_files_only: !customBundleUrl,
        });
      } catch (tokErr) {
        if (!customBundleUrl) {
          throw tokErr;
        }
        const tokJsonRes = await fetch(`${tokenizerUrl}/tokenizer.json`);
        if (!tokJsonRes.ok) {
          throw new Error(`Failed loading tokenizer.json from ${tokenizerUrl} (${tokJsonRes.status}).`);
        }
        const tokJson = (await tokJsonRes.json()) as Record<string, unknown>;
        let tokCfg: Record<string, unknown> = {
          tokenizer_class: 'Qwen2Tokenizer',
          eos_token: '<|im_end|>',
          pad_token: '<|endoftext|>',
        };
        try {
          const fallbackCfgRes = await fetch(
            resolveExtensionAssetUrl('models/pragma-1-4b-rc1/tokenizer_config.json', this.assetBaseUrl),
          );
          if (fallbackCfgRes.ok) {
            const qwenCfg = (await fallbackCfgRes.json()) as Record<string, unknown>;
            if (typeof qwenCfg.chat_template === 'string') {
              tokCfg.chat_template = qwenCfg.chat_template;
            }
          }
        } catch {
          // Ignore fallback chat_template fetch failure
        }
        try {
          const customCfgRes = await fetch(`${tokenizerUrl}/tokenizer_config.json`);
          if (customCfgRes.ok) {
            const parsedCustomCfg = (await customCfgRes.json()) as Record<string, unknown>;
            tokCfg = { ...tokCfg, ...parsedCustomCfg };
          }
        } catch {
          // tokenizer_config.json is optional on standalone WGSL bundles
        }
        tokenizer = new PreTrainedTokenizer(
          tokJson as unknown as ConstructorParameters<typeof PreTrainedTokenizer>[0],
          tokCfg as unknown as ConstructorParameters<typeof PreTrainedTokenizer>[1],
        );
      }

      try {
        const genCfgRes = await fetch(`${tokenizerUrl}/generation_config.json`);
        if (genCfgRes.ok) {
          const genCfg = (await genCfgRes.json()) as { eos_token_id?: number | number[] };
          if (typeof genCfg.eos_token_id === 'number') {
            extraEosIds.push(genCfg.eos_token_id);
          } else if (Array.isArray(genCfg.eos_token_id)) {
            for (const id of genCfg.eos_token_id) {
              if (typeof id === 'number' && Number.isFinite(id)) extraEosIds.push(id);
            }
          }
        }
      } catch {
        // Ignore optional generation_config.json error
      }
      if (isGemma4 || !tokenizer.chat_template) {
        try {
          const jinjaUrl = `${tokenizerUrl.replace(/\/+$/, '')}/chat_template.jinja`;
          const jinjaRes = await fetch(jinjaUrl);
          if (jinjaRes.ok) {
            tokenizer.chat_template = await jinjaRes.text();
          }
        } catch {
          // Ignore optional chat_template.jinja fetch error
        }
      }
      const tokenizerMs = performance.now() - tokStartMs;

      const rawDevMem =
        typeof navigator !== 'undefined' && 'deviceMemory' in navigator
          ? Number((navigator as unknown as { deviceMemory?: number }).deviceMemory)
          : undefined;
      const deviceMemoryGb =
        typeof rawDevMem === 'number' && Number.isFinite(rawDevMem) && rawDevMem > 0
          ? rawDevMem
          : undefined;

      const wgslEngine: WgslRuntimeEngine = isGemma4
        ? new Gemma4ExtensionAdapter()
        : (new WgslQwen35Engine() as unknown as WgslRuntimeEngine);
      wgslEngine.setDisplayHz(this.displayHz);
      const stats = await wgslEngine.load(null, {
        assetBaseUrl: this.assetBaseUrl,
        bundleBaseUrl: customBundleUrl,
        modelId: this.config.modelId,
        kvMode: 'int8',
        maxContext: this.config.wgslMaxContext,
        pacingMode: this.config.gpuPacingMode,
        forceComputeTier: this.config.forceComputeTier ?? undefined,
        deviceMemoryGb,
        onProgress: (phase, pct) => {
          this.updateTelemetry({
            status: pct >= 90 ? 'compiling' : 'downloading',
            loadProgressPct: Math.min(100, pct),
            loadPhaseLabel: phase,
          });
        },
      });
      wgslEngine.setDisplayHz(this.displayHz);

      this.tokenizer = tokenizer;
      this.wgslEngine = wgslEngine;
      this.adapterName = stats.adapterName ?? 'webgpu-default';
      this.eosTokenIds = this.collectEosTokenIds(tokenizer);
      for (const extraId of extraEosIds) {
        this.eosTokenIds.add(extraId);
      }
      if (isGemma4) {
        this.eosTokenIds.add(1);
        this.eosTokenIds.add(106);
      } else {
        for (const stopTok of ['<|im_end|>', '<|endoftext|>']) {
          try {
            const encoded = tokenizer.encode(stopTok, { add_special_tokens: false });
            if (encoded.length === 1 && Number.isFinite(encoded[0])) {
              this.eosTokenIds.add(encoded[0]!);
            }
          } catch {
            // Ignore unmapped stop token
          }
        }
        if (this.eosTokenIds.size === 0) {
          this.eosTokenIds.add(248044);
          this.eosTokenIds.add(248046);
        }
      }
      this.prefixCache = new PrefixCacheManager(wgslEngine, { maxTabSlots: 8 });

      await this.prewarmBaseSystemPrompt();
      if (!isGemma4) {
        await this.ensureZeroConfigSpecCalibration();
      }

      const coldLoadMs = performance.now() - startMs;
      const memStats = wgslEngine.getMemoryStats();
      const baseInfo = this.prefixCache.getBasePromptInfo();
      this.log(
        'info',
        'offscreen',
        `WGSL ${modelLabel} ready in ${(coldLoadMs / 1000).toFixed(2)}s (tier=${stats.computeTier}, source=${stats.loadSource}, host=${this.executionHost}, stream=${stats.opfsReadMs.toFixed(0)}ms, compile+verify=${stats.pipelineCompileMs.toFixed(0)}ms, kv=int8 [${stats.allocatedKvCapacity}/${stats.maxSafeKvCapacity}], dp4a=${stats.usingDp4a}, baseSlot=${baseInfo.tokens}t, hz=${this.displayHz}).`,
      );

      this.updateTelemetry({
        status: 'ready',
        loadProgressPct: 100,
        loadPhaseLabel: `Ready (${stats.computeTier})`,
        coldLoadMs,
        loadedFromCache: true,
        loadSource: stats.loadSource,
        executionHost: this.executionHost,
        loadBreakdown: {
          opfsReadMs: stats.opfsReadMs,
          tokenizerMs,
          modelLoadMs: stats.repackMs + stats.pipelineCompileMs,
          totalMs: coldLoadMs,
        },
        estVramMb: memStats.estVramMb,
        kvCacheMb: memStats.kvCacheMb,
        deltaNetStateMb: memStats.deltaNetStateMb,
        vramDeltaMb: memStats.vramDeltaMb ?? null,
        activeContextTokens: wgslEngine.activeSeqPos,
        computeTier: stats.computeTier,
        allocatedKvCapacity: stats.allocatedKvCapacity,
        maxKvCapacity: stats.maxSafeKvCapacity,
        basePromptCached: baseInfo.cached,
        basePromptTokens: baseInfo.tokens,
        displayHz: this.displayHz,
        backgroundPrefillState: 'idle',
        activePrefixSlot: baseInfo.cached ? BASE_SYSTEM_SLOT_ID : null,
      });

      void this.refreshOpfsStatus().catch(() => {});
    })()
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        this.log('error', 'offscreen', `Model load failed: ${msg}`);
        this.updateTelemetry({
          status: 'error',
          lastError: msg,
          loadPhaseLabel: 'Load failed',
        });
        throw err;
      })
      .finally(() => {
        this.loadingPromise = null;
      });

    return this.loadingPromise;
  }

  async unloadModel(): Promise<void> {
    this.abortBackgroundPrefill();
    await this.resetContext(false);
    this.prefixCache?.clearAll();
    this.prefixCache = null;
    this.wgslSnapshot = null;
    this.wgslSnapshotMeta = null;
    if (this.offscreenSpecAdapter) {
      this.offscreenSpecAdapter.dispose();
      this.offscreenSpecAdapter = null;
    }
    this.offscreenSpecAdapterPromise = null;
    this.engineBActiveForAdapter = false;
    this.offscreenSpecTokenizerRef = null;
    this.offscreenSpecDeviceRef = null;
    if (this.wgslEngine) {
      this.wgslEngine.dispose();
      this.wgslEngine = null;
    }
    this.tokenizer = null;
    this.updateTelemetry({
      status: 'unloaded',
      loadProgressPct: 0,
      loadPhaseLabel: 'Idle',
      coldLoadMs: null,
      loadedFromCache: null,
      loadSource: null,
      loadBreakdown: null,
      estVramMb: 0,
      kvCacheMb: 0,
      deltaNetStateMb: 0,
      vramDeltaMb: null,
      baselineDecodeTokPerSec: null,
      activeContextTokens: 0,
      hasCpuSnapshot: false,
      cpuSnapshotTokens: null,
      prefixReused: null,
      computeTier: null,
      allocatedKvCapacity: 0,
      maxKvCapacity: 0,
      basePromptCached: false,
      basePromptTokens: 0,
      lastSlotRestoreMs: null,
      backgroundPrefillState: 'idle',
      activePrefixSlot: null,
    });
  }

  async resetContext(restoreBasePrompt = false): Promise<void> {
    this.abortBackgroundPrefill();
    if (restoreBasePrompt && this.prefixCache?.hasBasePrompt() && this.wgslEngine) {
      const restored = await this.prefixCache.resetToBasePrompt();
      if (restored) {
        this.cachedMessages = restored.entry.messages.map((m) => ({ ...m }));
        this.cachedTokenCount = restored.seqPos;
        this.cachedTokenIds = [...restored.entry.tokenIds];
        this.pendingLiveTokenIds = [];
        this.cachedThinkingMode = this.config.enableThinking;
        const wgslMem = this.wgslEngine.getMemoryStats();
        this.updateTelemetry({
          activeContextTokens: restored.seqPos,
          kvCacheMb: wgslMem.kvCacheMb,
          deltaNetStateMb: wgslMem.deltaNetStateMb,
          estVramMb: wgslMem.estVramMb,
          allocatedKvCapacity: this.wgslEngine.getAllocatedKvCapacity(),
          maxKvCapacity: this.wgslEngine.getMaxSafeKvCapacity(),
          prefixReused: true,
          lastSlotRestoreMs: restored.restoreMs,
          activePrefixSlot: BASE_SYSTEM_SLOT_ID,
          backgroundPrefillState: 'idle',
          prefillChunkProgress: null,
          speculativeStats: null,
        });
        return;
      }
    }

    if (this.wgslEngine) {
      await this.wgslEngine.resetState();
    }
    this.cachedMessages = [];
    this.cachedTokenCount = 0;
    this.cachedTokenIds = [];
    this.pendingLiveTokenIds = [];
    const wgslMem = this.wgslEngine ? this.wgslEngine.getMemoryStats() : null;
    this.updateTelemetry({
      activeContextTokens: 0,
      kvCacheMb: 0,
      deltaNetStateMb: wgslMem ? wgslMem.deltaNetStateMb : 0,
      estVramMb: wgslMem ? wgslMem.estVramMb : 0,
      allocatedKvCapacity: this.wgslEngine ? this.wgslEngine.getAllocatedKvCapacity() : 0,
      maxKvCapacity: this.wgslEngine ? this.wgslEngine.getMaxSafeKvCapacity() : 0,
      prefixReused: null,
      lastSlotRestoreMs: null,
      activePrefixSlot: null,
      backgroundPrefillState: 'idle',
      prefillChunkProgress: null,
      speculativeStats: null,
    });
  }

  setDisplayHz(hz: number): void {
    const safeHz = Number.isFinite(hz) && hz >= 30 && hz <= 360 ? Math.round(hz) : 60;
    this.displayHz = safeHz;
    this.wgslEngine?.setDisplayHz(safeHz);
    this.updateTelemetry({ displayHz: safeHz });
  }

  abortBackgroundPrefill(): void {
    this.bgPrefillSeq += 1;
    const prevReqId = this.activeBgRequestId;
    const prevSlotKey = this.activeBgSlotKey ?? '';
    this.activeBgRequestId = null;
    if (this.bgAbortController) {
      this.bgAbortController.abort();
      this.bgAbortController = null;
    }
    if (this.telemetry.backgroundPrefillState === 'prefilling') {
      this.updateTelemetry({ backgroundPrefillState: 'idle' });
    }
    if (prevReqId) {
      this.callbacks.onPrefillComplete?.(prevReqId, prevSlotKey, 0, 0);
    }
  }

  evictSlot(slotKey: string, requestId?: string): void {
    this.prefixCache?.deleteSlot(slotKey);
    if (requestId) {
      this.callbacks.onEvictComplete?.(requestId, slotKey);
    }
  }

  private normalizeMessagesForChatTemplate(messages: readonly ChatMessage[]): ChatMessage[] {
    return messages.map((msg, idx) => {
      if (msg.role === 'tool') {
        return {
          role: 'user',
          content: msg.toolName ? `[Tool Result: ${msg.toolName}]\n${msg.content}` : msg.content,
        };
      }
      if (idx > 0 && msg.role === 'system') {
        return { role: 'user', content: msg.content };
      }
      return { role: msg.role, content: msg.content };
    });
  }

  private formatChatMessages(
    messages: readonly ChatMessage[],
    options: { addGenerationPrompt: boolean },
  ): string {
    const normalized = this.normalizeMessagesForChatTemplate(messages);
    return this.tokenizer!.apply_chat_template(normalized, {
      tokenize: false,
      add_generation_prompt: options.addGenerationPrompt,
      enable_thinking: this.config.enableThinking,
    } as Parameters<PreTrainedTokenizer['apply_chat_template']>[1]) as unknown as string;
  }

  private encodePrefixMessages(messages: readonly ChatMessage[], isDraft: boolean): number[] {
    if (!this.tokenizer || messages.length === 0) {
      return [];
    }
    if (isDraft) {
      const hasUserMessage = messages.some((m) => m.role === 'user');
      const normalized = this.normalizeMessagesForChatTemplate(messages);
      if (hasUserMessage) {
        let formatted = this.formatChatMessages(normalized, { addGenerationPrompt: false });
        for (const suffix of ['<|im_end|>\n', '<|im_end|>', '<turn|>\n', '<turn|>']) {
          if (formatted.endsWith(suffix)) {
            formatted = formatted.slice(0, -suffix.length);
            break;
          }
        }
        return this.tokenizer.encode(formatted, { add_special_tokens: false });
      }
    }

    const marker = '__TINYGPT_PREFIX_MARKER__';
    const withMarker = this.formatChatMessages(
      [...messages, { role: 'user', content: marker }],
      { addGenerationPrompt: false },
    );
    let markerIdx = withMarker.lastIndexOf(`<|im_start|>user\n${marker}`);
    if (markerIdx < 0) {
      markerIdx = withMarker.lastIndexOf(`<|turn>user\n${marker}`);
    }
    const prefixText = markerIdx >= 0 ? withMarker.slice(0, markerIdx) : withMarker;
    return this.tokenizer.encode(prefixText, { add_special_tokens: false });
  }

  private async prewarmBaseSystemPrompt(): Promise<void> {
    if (!this.wgslEngine || !this.tokenizer || !this.prefixCache) {
      return;
    }
    try {
      const defaultTools = CANONICAL_TOOL_DEFINITIONS.filter(
        (t) =>
          t.name === 'read_active_tab' ||
          t.name === 'list_tabs' ||
          t.name === 'read_tab' ||
          t.name === 'run_sandbox',
      );
      const slot0 = await buildSlot0Envelope({
        systemPrompt: buildHarnessSystemPrompt({
          pageEnabled: true,
          pageAttached: false,
          researchEnabled: false,
        }),
        tools: defaultTools as unknown as NonNullable<
          Parameters<typeof buildSlot0Envelope>[0]
        >['tools'],
      });
      const baseMsg: ChatMessage = {
        role: 'system',
        content: slot0.messages[0]!.content,
      };
      const baseTokenIds = this.encodePrefixMessages([baseMsg], false);
      if (baseTokenIds.length === 0 || baseTokenIds.length >= this.wgslEngine.maxContextTokens) {
        return;
      }
      await this.wgslEngine.resetState();
      await this.wgslEngine.prefill(baseTokenIds, {
        chunkSize: 64,
        sampling: { temperature: 0, topK: 1, topP: 1 },
        pacingMode: this.config.gpuPacingMode,
        yieldFn: async () => yieldFast('message-channel'),
      });
      await this.prefixCache.saveBasePromptSlot(baseTokenIds, [baseMsg]);
      this.cachedMessages = [baseMsg];
      this.cachedTokenIds = [...baseTokenIds];
      this.pendingLiveTokenIds = [];
      this.cachedTokenCount = this.wgslEngine.activeSeqPos;
      this.cachedThinkingMode = this.config.enableThinking;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log('warn', 'offscreen', `Base prompt prewarm skipped: ${msg}`);
    }
  }

  enqueueBackgroundPrefill(
    slotKey: string,
    messages: ChatMessage[],
    options?: { isDraft?: boolean; url?: string; contentHash?: string; requestId?: string },
  ): void {
    // Never let composer typing or base-system warmup abort an in-progress active-tab or thread prefill.
    if (
      (slotKey.startsWith('draft:') || slotKey.startsWith('base-system')) &&
      this.bgAbortController !== null &&
      (this.activeBgSlotKey?.startsWith('tab:') || this.activeBgSlotKey?.startsWith('thread:'))
    ) {
      if (options?.requestId) {
        this.callbacks.onPrefillComplete?.(options.requestId, slotKey, 0, 0);
      }
      return;
    }
    const sig = `${slotKey}::${this.config.enableThinking ? 1 : 0}::${options?.isDraft ? 1 : 0}::${messages.map((m) => `${m.role}:${m.content.length}:${m.content.slice(0, 24)}`).join('|')}`;
    if (this.bgAbortController !== null && this.activeBgSignature === sig) {
      if (options?.requestId) {
        this.callbacks.onPrefillComplete?.(options.requestId, slotKey, 0, 0);
      }
      return;
    }
    this.abortBackgroundPrefill();
    const mySeq = this.bgPrefillSeq;
    this.activeBgSignature = sig;
    if (options?.requestId) {
      this.activeBgRequestId = options.requestId;
    }
    this.taskQueue = this.taskQueue
      .then(() => this.runBackgroundPrefill(mySeq, slotKey, messages, options))
      .catch((err) => {
        if (this.telemetry.backgroundPrefillState === 'prefilling') {
          this.updateTelemetry({ backgroundPrefillState: 'idle' });
        }
        if (options?.requestId && this.activeBgRequestId === options.requestId) {
          this.activeBgRequestId = null;
          const errMsg = err instanceof Error ? err.message : String(err);
          this.callbacks.onPrefillError?.(options.requestId, slotKey, errMsg);
        }
      });
  }

  private getEvaluatedLiveTokenIds(): number[] {
    if (!this.wgslEngine) {
      return [];
    }
    return this.pendingLiveTokenIds.length === 0
      ? this.cachedTokenIds
      : this.cachedTokenIds.slice(0, this.wgslEngine.activeSeqPos);
  }

  private async runBackgroundPrefill(
    seq: number,
    slotKey: string,
    messages: ChatMessage[],
    options?: { isDraft?: boolean; url?: string; contentHash?: string; requestId?: string },
  ): Promise<void> {
    const startMs = performance.now();
    const reqId = options?.requestId;
    let reqSettled = false;
    const resolvePrefill = (tokens: number) => {
      if (reqId && !reqSettled) {
        reqSettled = true;
        if (this.activeBgRequestId === reqId) {
          this.activeBgRequestId = null;
        }
        this.callbacks.onPrefillComplete?.(reqId, slotKey, tokens, performance.now() - startMs);
      }
    };
    const rejectPrefill = (errMsg: string) => {
      if (reqId && !reqSettled) {
        reqSettled = true;
        if (this.activeBgRequestId === reqId) {
          this.activeBgRequestId = null;
        }
        this.callbacks.onPrefillError?.(reqId, slotKey, errMsg);
      }
    };

    if (
      seq !== this.bgPrefillSeq ||
      !this.isEngineReady() ||
      !this.wgslEngine ||
      !this.tokenizer ||
      !this.prefixCache ||
      this.activeRequestId !== null ||
      this.telemetry.status !== 'ready'
    ) {
      resolvePrefill(0);
      return;
    }

    const targetTokenIds = this.encodePrefixMessages(messages, Boolean(options?.isDraft));
    if (targetTokenIds.length === 0 || targetTokenIds.length >= this.wgslEngine.getMaxSafeKvCapacity() - 64) {
      resolvePrefill(0);
      return;
    }

    const match = this.prefixCache.findBestMatch(targetTokenIds, this.getEvaluatedLiveTokenIds());
    if (match.matchedTokens === targetTokenIds.length) {
      if (slotKey === BASE_SYSTEM_SLOT_ID) {
        await this.prefixCache.saveBasePromptSlot(targetTokenIds, messages);
      } else if (match.source === 'live' || slotKey.startsWith('base-system:')) {
        await this.prefixCache.saveTabSlot(slotKey, targetTokenIds, messages, {
          url: options?.url,
          contentHash: options?.contentHash,
        });
      }
      this.updateTelemetry({
        backgroundPrefillState: 'warm',
        activePrefixSlot: match.slotId ?? slotKey,
      });
      resolvePrefill(0);
      return;
    }

    // Never wipe an active multi-turn live conversation KV cache for a speculative background prefill
    if (
      match.source !== 'live' &&
      this.cachedMessages.some((m) => m.role === 'assistant')
    ) {
      resolvePrefill(0);
      return;
    }

    const controller = new AbortController();
    this.bgAbortController = controller;
    this.activeBgSlotKey = slotKey;
    this.updateTelemetry({ backgroundPrefillState: 'prefilling' });
    let partialPrefixSaved = false;

    try {
      if (match.source === 'slot' && match.slotId) {
        const restored = await this.prefixCache.restoreSlot(match.slotId);
        this.cachedTokenIds = [...restored.entry.tokenIds];
        this.cachedMessages = restored.entry.messages.map((m) => ({ ...m }));
        this.pendingLiveTokenIds = [];
        this.cachedTokenCount = restored.seqPos;
      } else if (match.source === 'none') {
        await this.wgslEngine.resetState();
        this.cachedTokenIds = [];
        this.cachedMessages = [];
        this.pendingLiveTokenIds = [];
        this.cachedTokenCount = 0;
      }

      if (seq !== this.bgPrefillSeq || controller.signal.aborted) {
        throw new Error('__BG_ABORTED__');
      }

      const deltaTokens = targetTokenIds.slice(match.matchedTokens);
      if (deltaTokens.length > 0) {
        await this.wgslEngine.prefill(deltaTokens, {
          chunkSize: Math.min(64, this.config.chunkSize || 64),
          sampling: { temperature: 0, topK: 1, topP: 1 },
          pacingMode: this.config.gpuPacingMode,
          abortSignal: controller.signal,
          skipFinalLogits: true,
          yieldFn: async () => {
            if (seq !== this.bgPrefillSeq || controller.signal.aborted) {
              controller.abort();
              throw new Error('__BG_ABORTED__');
            }
            await yieldFast('message-channel');
          },
          onChunk: () => {
            if (seq !== this.bgPrefillSeq || controller.signal.aborted) {
              controller.abort();
              throw new Error('__BG_ABORTED__');
            }
            const pos = this.wgslEngine!.activeSeqPos;
            this.cachedTokenCount = pos;
            if (pos > 0 && pos <= targetTokenIds.length) {
              this.cachedTokenIds = targetTokenIds.slice(0, pos);
              this.pendingLiveTokenIds = [];
              this.cachedMessages = [];
              partialPrefixSaved = true;
            }
          },
        });
      }

      if (seq !== this.bgPrefillSeq || controller.signal.aborted) {
        throw new Error('__BG_ABORTED__');
      }

      if (slotKey === BASE_SYSTEM_SLOT_ID) {
        await this.prefixCache.saveBasePromptSlot(targetTokenIds, messages);
      } else {
        await this.prefixCache.saveTabSlot(slotKey, targetTokenIds, messages, {
          url: options?.url,
          contentHash: options?.contentHash,
        });
      }
      this.cachedTokenIds = [...targetTokenIds];
      this.cachedMessages = messages.map((m) => ({ ...m }));
      this.pendingLiveTokenIds = [];
      this.cachedTokenCount = this.wgslEngine.activeSeqPos;
      this.cachedThinkingMode = this.config.enableThinking;
      this.refreshCacheTelemetry();
      this.updateTelemetry({
        backgroundPrefillState: 'warm',
        activePrefixSlot: slotKey,
        prefillChunkProgress: null,
      });
      resolvePrefill(deltaTokens.length);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === '__BG_ABORTED__' || msg === 'Generation aborted' || controller.signal.aborted) {
        const preservedSeqPos = this.wgslEngine?.activeSeqPos ?? 0;
        if (partialPrefixSaved && preservedSeqPos > 0 && this.cachedTokenIds.length === preservedSeqPos) {
          // Keep evaluated partial prefix chunks live so the interrupting turn can continue incrementally.
          this.cachedTokenCount = preservedSeqPos;
          this.cachedThinkingMode = this.config.enableThinking;
        } else {
          this.cachedTokenIds = [];
          this.cachedMessages = [];
          this.pendingLiveTokenIds = [];
          this.cachedTokenCount = 0;
          if (this.prefixCache.hasBasePrompt()) {
            await this.prefixCache.resetToBasePrompt().catch(() => {});
          } else {
            await this.wgslEngine.resetState().catch(() => {});
          }
        }
        this.updateTelemetry({
          backgroundPrefillState: 'idle',
          prefillChunkProgress: null,
        });
        resolvePrefill(0);
        return;
      }
      this.log('warn', 'offscreen', `Background speculative prefill stopped: ${msg}`);
      this.updateTelemetry({ backgroundPrefillState: 'idle', prefillChunkProgress: null });
      rejectPrefill(msg);
    } finally {
      if (this.bgAbortController === controller) {
        this.bgAbortController = null;
        this.activeBgSlotKey = null;
        this.activeBgSignature = null;
      }
    }
  }

  async snapshotContext(): Promise<void> {
    if (!this.wgslEngine || this.wgslEngine.activeSeqPos === 0) {
      this.log('warn', 'offscreen', 'No active WGSL hybrid state to snapshot.');
      return;
    }
    try {
      const snap = await this.wgslEngine.snapshotState();
      this.wgslSnapshot = snap;
      this.wgslSnapshotMeta = {
        cachedMessages: this.cachedMessages.map((m) => ({ ...m })),
        cachedThinkingMode: this.cachedThinkingMode,
        cachedTokenIds: [...this.cachedTokenIds],
      };
      this.updateTelemetry({
        hasCpuSnapshot: true,
        cpuSnapshotTokens: snap.seqPos,
      });
      const mem = this.wgslEngine.getMemoryStats();
      this.log(
        'info',
        'offscreen',
        `Snapshotted WGSL hybrid DeltaNet + KV state (${snap.seqPos} tokens, ${(mem.kvCacheMb + mem.deltaNetStateMb).toFixed(1)} MiB) to CPU RAM.`,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log('warn', 'offscreen', msg);
      this.updateTelemetry({ lastError: msg });
    }
  }

  async restoreContext(): Promise<void> {
    if (!this.wgslEngine || !this.wgslSnapshot || !this.wgslSnapshotMeta) {
      this.log('warn', 'offscreen', 'No CPU state snapshot available to restore.');
      return;
    }
    await this.wgslEngine.restoreState(this.wgslSnapshot);
    this.cachedMessages = this.wgslSnapshotMeta.cachedMessages.map((m) => ({ ...m }));
    this.cachedTokenCount = this.wgslSnapshot.seqPos;
    this.cachedThinkingMode = this.wgslSnapshotMeta.cachedThinkingMode;
    this.cachedTokenIds = [...this.wgslSnapshotMeta.cachedTokenIds];
    this.pendingLiveTokenIds = [];
    this.refreshCacheTelemetry();
    this.log('info', 'offscreen', `Restored WGSL hybrid state (${this.cachedTokenCount} tokens).`);
  }

  abort(requestId: string): void {
    if (requestId) {
      this.abortedRequests.add(requestId);
    }
    if (requestId === this.activeRequestId || !this.activeRequestId || !requestId) {
      this.activeAbortController?.abort();
      if (this.activeRequestId) {
        this.abortedRequests.add(this.activeRequestId);
        this.log('info', 'offscreen', `Abort requested for ${this.activeRequestId}.`);
      }
    }
  }

  enqueueGenerate(
    requestId: string,
    messages: ChatMessage[],
    overrideConfig?: Partial<EngineConfig>,
    groundingText?: string,
    stopOnToolCall = false,
  ): void {
    this.pendingPhaseEnqueue.set(requestId, {
      enqueue: performance.timeOrigin + performance.now(),
      bgPrefillActiveAtEnqueue: this.bgAbortController !== null,
    });
    this.abortBackgroundPrefill();
    this.taskQueue = this.taskQueue
      .then(() =>
        this.runGenerate(requestId, messages, overrideConfig, groundingText, stopOnToolCall),
      )
      .catch((err) => {
        this.pendingPhaseEnqueue.delete(requestId);
        const msg = err instanceof Error ? err.message : String(err);
        this.callbacks.onGenerationError(requestId, msg);
      });
  }

  enqueueBenchmarkSuite(contextLengths: number[], decodeTokens: number, chunkSize?: number): void {
    this.abortBackgroundPrefill();
    this.taskQueue = this.taskQueue
      .then(() => this.runBenchmarkSuite(contextLengths, decodeTokens, chunkSize))
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        this.log('error', 'offscreen', `Benchmark suite error: ${msg}`);
      });
  }

  enqueueSpecMicrobench(): void {
    this.abortBackgroundPrefill();
    this.taskQueue = this.taskQueue
      .then(() => this.runSpeculativeMicrobench())
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        this.log('error', 'offscreen', `Speculative microbench error: ${msg}`);
      });
  }

  enqueueNeedleInHaystack(targetTokens: number, depthRatio: number, chunkSize?: number): void {
    this.abortBackgroundPrefill();
    this.taskQueue = this.taskQueue
      .then(() => this.runNeedleInHaystack(targetTokens, depthRatio, chunkSize))
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        this.log('error', 'offscreen', `NIAH error: ${msg}`);
      });
  }

  private collectEosTokenIds(tokenizer: PreTrainedTokenizer): Set<number> {
    const ids = new Set<number>();
    const addVal = (v: unknown) => {
      if (typeof v === 'number' && Number.isFinite(v)) {
        ids.add(v);
      } else if (Array.isArray(v)) {
        for (const item of v) {
          addVal(item);
        }
      }
    };
    addVal(tokenizer.eos_token_id);
    return ids;
  }

  private buildIncrementalChatMlPrompt(suffixMessages: ChatMessage[]): string | null {
    if (!this.tokenizer) {
      return null;
    }
    const template =
      typeof this.tokenizer.chat_template === 'string' ? this.tokenizer.chat_template : '';
    return buildIncrementalTurnPrompt(template, suffixMessages);
  }

  private refreshCacheTelemetry(): void {
    if (!this.wgslEngine) {
      return;
    }
    const stats = this.wgslEngine.getMemoryStats();
    const totalContext = Math.max(this.cachedTokenCount, this.wgslEngine.activeSeqPos);
    this.updateTelemetry({
      activeContextTokens: totalContext,
      kvCacheMb: stats.kvCacheMb,
      deltaNetStateMb: stats.deltaNetStateMb,
      estVramMb: stats.estVramMb,
      allocatedKvCapacity: this.wgslEngine.getAllocatedKvCapacity(),
      maxKvCapacity: this.wgslEngine.getMaxSafeKvCapacity(),
      maxContextReached: Math.max(this.telemetry.maxContextReached, totalContext),
    });
  }

  private async prefillAndPickFirstToken(
    tokenIds: number[],
    chunkSize: number,
    sampleOpts: SamplingOptions,
    requestId?: string,
    skipFinalLogits = false,
  ): Promise<{ nextToken: number; prefillMs: number }> {
    if (!this.wgslEngine) {
      throw new Error('Model is not loaded.');
    }
    const controller = new AbortController();
    this.activeAbortController = controller;
    try {
      const wgslChunk = chunkSize > 0 ? chunkSize : 64;
      const { nextTokenId, prefillMs } = await this.wgslEngine.prefill(tokenIds, {
        chunkSize: wgslChunk,
        sampling: sampleOpts,
        pacingMode: this.config.gpuPacingMode,
        abortSignal: controller.signal,
        skipFinalLogits,
        yieldFn: async () => {
          if ((requestId && this.abortedRequests.has(requestId)) || controller.signal.aborted) {
            controller.abort();
            throw new Error('__ABORTED__');
          }
          await yieldFast(
            this.config.yieldMode === 'none' ? 'message-channel' : this.config.yieldMode,
          );
        },
        onChunk: (done, total) => {
          if ((requestId && this.abortedRequests.has(requestId)) || controller.signal.aborted) {
            controller.abort();
            throw new Error('__ABORTED__');
          }
          this.cachedTokenCount = this.wgslEngine!.activeSeqPos;
          const isFinalChunk = done >= total;
          if (isFinalChunk || !this.config.fastDecodePath || done % (wgslChunk * 4) === 0) {
            this.updateTelemetry({
              prefillChunkProgress: isFinalChunk ? null : { done, total },
            });
          }
          if (isFinalChunk || !this.config.fastDecodePath) {
            this.refreshCacheTelemetry();
          }
        },
      });
      this.cachedTokenCount = this.wgslEngine.activeSeqPos;
      this.refreshCacheTelemetry();
      return { nextToken: nextTokenId, prefillMs };
    } finally {
      if (this.activeAbortController === controller) {
        this.activeAbortController = null;
      }
    }
  }

  private async decodeNextToken(
    tokenId: number,
    sampleOpts: SamplingOptions,
    stepIdx: number,
  ): Promise<{
    nextToken: number;
    forwardMs: number;
    logitsMs: number;
    sampleMs: number;
    breatherMs: number;
    encodeMs?: number;
    mapMs?: number;
  }> {
    if (!this.wgslEngine) {
      throw new Error('Model is not loaded.');
    }
    const res = await this.wgslEngine.decodeStep(tokenId, sampleOpts, this.config.gpuPacingMode);
    this.cachedTokenCount = this.wgslEngine.activeSeqPos;
    if (!this.config.fastDecodePath || stepIdx % 8 === 0) {
      this.refreshCacheTelemetry();
    }
    return {
      nextToken: res.nextToken,
      forwardMs: res.gpuMs,
      logitsMs: 0,
      sampleMs: 0,
      breatherMs: res.breatherMs,
      encodeMs: res.encodeMs,
      mapMs: res.mapMs,
    };
  }

  private async ensureZeroConfigSpecCalibration(): Promise<void> {
    const engine = this.wgslEngine;
    const device = engine?.getDevice?.() ?? null;
    const slices = engine?.getTiedTableSlices?.() ?? [];
    const totalVocab = slices.reduce((acc, s) => Math.max(acc, s.endRow), 0);
    if (
      !engine ||
      !device ||
      slices.length === 0 ||
      totalVocab !== 81920 ||
      typeof engine.measureSpecStep !== 'function' ||
      typeof fetch !== 'function'
    ) {
      return;
    }
    try {
      const cached = await loadSpecCalibration(this.adapterName, EXP007B_DRAFT_HEAD_SHA256);
      if (cached) {
        this.applyMeasuredSpecCalibration(
          cached.t1Ms,
          cached.stepMs,
          cached.headMs,
          cached.rejectPenaltyMs,
        );
        this.log(
          'info',
          'offscreen',
          `Loaded cached spec calibration (${this.adapterName}): t1=${cached.t1Ms.toFixed(2)}ms, headMs=${cached.headMs.toFixed(2)}ms, rejectP=${cached.rejectPenaltyMs.toFixed(2)}ms, engineB=${shouldEnableEngineB(cached.headMs, cached.t1Ms) ? 'enabled' : 'bypassed (Engine A)'}.`,
        );
        return;
      }
      await this.runFastSpecCalibration();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log('warn', 'offscreen', `Silent spec calibration skipped: ${msg}`);
    }
  }

  private applyMeasuredSpecCalibration(
    t1Ms: number,
    stepMs: number[],
    headMs: number,
    rejectPenaltyMs: number,
  ): void {
    this.measuredChainStepMs = [...stepMs];
    this.measuredEngineBHeadMs = headMs;
    this.measuredRejectPenaltyMs = rejectPenaltyMs;
    const t5Ms = stepMs[4] ?? t1Ms * 2.0;
    this.engineACostModel.t1Ms = t1Ms;
    this.engineACostModel.slopePerTokenMs = Math.max(0.15 * t1Ms, (t5Ms - t1Ms) / 4);
    this.engineACostModel.rejectPenaltyMs = Math.max(0.2, rejectPenaltyMs);
  }

  private async runFastSpecCalibration(): Promise<void> {
    const engine = this.wgslEngine;
    const device = engine?.getDevice?.() ?? null;
    const slices = engine?.getTiedTableSlices?.() ?? [];
    if (!engine || !device || slices.length === 0 || typeof engine.measureSpecStep !== 'function') {
      return;
    }
    const calSlot = '__spec_calib';
    const greedy: SamplingOptions = { temperature: 0, topK: 1, topP: 1 };
    const tCal0 = performance.now();
    try {
      engine.setCaptureDraftTaps?.(false);
      // Warmup & time T(1) across 4 scalar steps from the current/minimal context.
      if (engine.activeSeqPos === 0) {
        await engine.prefill([1, 2, 3, 4], {
          chunkSize: 64,
          sampling: greedy,
          pacingMode: 'max-speed',
          yieldFn: async () => {},
        });
      }
      await engine.saveGpuPrefixSlot(calSlot);
      const t1Samples: number[] = [];
      let tok = 10;
      for (let i = 0; i < 4; i++) {
        const res = await engine.decodeStep(tok, greedy, 'max-speed');
        tok = res.nextToken;
        if (i >= 1) t1Samples.push(res.gpuMs);
      }
      t1Samples.sort((a, b) => a - b);
      const t1Ms = Number((t1Samples[Math.floor(t1Samples.length / 2)] ?? 10.5).toFixed(3));

      // Warmup & measure M=5 verify latency + P9a-P9c reject surcharge via paired steps.
      await engine.measureSpecStep(5, false);
      const acceptSamples: number[] = [];
      const rejectSamples: number[] = [];
      for (let i = 0; i < 2; i++) {
        acceptSamples.push(await engine.measureSpecStep(5, false));
        rejectSamples.push(await engine.measureSpecStep(5, true));
      }
      const t5Ms = Number(
        (acceptSamples.reduce((a, b) => a + b, 0) / acceptSamples.length).toFixed(3),
      );
      const t5RejectMs = Number(
        (rejectSamples.reduce((a, b) => a + b, 0) / rejectSamples.length).toFixed(3),
      );
      const rejectPenaltyMs = Number(Math.max(0.2, t5RejectMs - t5Ms).toFixed(3));

      // Time Engine B headMs(k=5) via single-submit chain (1 warmup + 2 reps).
      const baseStr = resolveExtensionAssetUrl('cycle17/', this.assetBaseUrl);
      const originFallback =
        typeof globalThis.location?.href === 'string'
          ? globalThis.location.href
          : 'https://extension.invalid/';
      const cycle17Base = new URL(baseStr.endsWith('/') ? baseStr : `${baseStr}/`, originFallback);
      const bundle = await loadGdnBundle(new URL('assets/exp007/manifest.json', cycle17Base));
      const tapsResp = await fetch(new URL('fixtures/trained-exp007/taps.f32', cycle17Base));
      let headMs = 999;
      if (bundle && tapsResp.ok) {
        if (bundle.sha256 !== EXP007B_DRAFT_HEAD_SHA256) {
          throw new Error('EXP-007b calibration manifest SHA-256 mismatch');
        }
        const taps = new Float32Array(await tapsResp.arrayBuffer());
        const runtime = await GdnDraftRuntime.create(
          device,
          bundle,
          { slices },
          shaderAssets(new URL('wgsl/', cycle17Base)),
        );
        try {
          const anchor = { taps, rootToken: 271 };
          await runtime.proposeChain(anchor, 5);
          const headSamples: number[] = [];
          for (let i = 0; i < 2; i++) {
            const h0 = performance.now();
            await runtime.proposeChain(anchor, 5);
            headSamples.push(performance.now() - h0);
          }
          headSamples.sort((a, b) => a - b);
          headMs = Number((headSamples[0] ?? 15).toFixed(3));
        } finally {
          runtime.dispose();
        }
      }

      const slope = Math.max(0.15 * t1Ms, (t5Ms - t1Ms) / 4);
      const stepMs = [
        t1Ms,
        Number((t1Ms + slope).toFixed(3)),
        Number((t1Ms + 2 * slope).toFixed(3)),
        Number((t1Ms + 3 * slope).toFixed(3)),
        t5Ms,
        Number((t5Ms + slope).toFixed(3)),
        Number((t5Ms + 2 * slope).toFixed(3)),
        Number((t5Ms + 3 * slope).toFixed(3)),
      ];

      this.applyMeasuredSpecCalibration(t1Ms, stepMs, headMs, rejectPenaltyMs);
      await saveSpecCalibration({
        adapterName: this.adapterName,
        dropinSha256: EXP007B_DRAFT_HEAD_SHA256,
        t1Ms,
        stepMs,
        headMs,
        rejectPenaltyMs,
        measuredAtIso: new Date().toISOString(),
      });
      const calMs = performance.now() - tCal0;
      this.log(
        'info',
        'offscreen',
        `Silent spec calibration (${calMs.toFixed(0)}ms, ${this.adapterName}): t1=${t1Ms.toFixed(2)}ms, t5=${t5Ms.toFixed(2)}ms, headMs=${headMs.toFixed(2)}ms, rejectP=${rejectPenaltyMs.toFixed(2)}ms, engineB=${shouldEnableEngineB(headMs, t1Ms) ? 'enabled' : 'bypassed (Engine A)'}.`,
      );
    } finally {
      engine.deleteGpuPrefixSlot(calSlot);
      engine.deleteGpuPrefixSlot('__eb_probe__');
      await this.resetContext(this.prefixCache?.hasBasePrompt() ?? false);
    }
  }

  private async getOrCreateOffscreenSpecAdapter(): Promise<OffscreenSpecAdapter | null> {
    const tok = this.tokenizer;
    const engine = this.wgslEngine;
    const device = engine?.getDevice?.() ?? null;
    if (!tok || !engine || !device || typeof fetch !== 'function') {
      return null;
    }
    const slices = engine.getTiedTableSlices?.() ?? [];
    const totalVocab = slices.reduce((acc, s) => Math.max(acc, s.endRow), 0);
    // Certified EXP-007b draft head targets EXP-006 / Pragma-1-4B RC1 (81,920 tied vocab rows)
    if (slices.length === 0 || totalVocab !== 81920) {
      return null;
    }
    if (
      this.offscreenSpecAdapter &&
      this.offscreenSpecTokenizerRef === tok &&
      this.offscreenSpecDeviceRef === device
    ) {
      return this.offscreenSpecAdapter;
    }
    if (this.offscreenSpecAdapterPromise) {
      return this.offscreenSpecAdapterPromise;
    }
    this.offscreenSpecAdapterPromise = (async () => {
      try {
        const baseStr = resolveExtensionAssetUrl('cycle17/', this.assetBaseUrl);
        const originFallback =
          typeof globalThis.location?.href === 'string'
            ? globalThis.location.href
            : 'https://extension.invalid/';
        const cycle17Base = new URL(
          baseStr.endsWith('/') ? baseStr : `${baseStr}/`,
          originFallback,
        );
        const wasmResp = await fetch(new URL('wasm/yours_spec_core.wasm', cycle17Base));
        if (!wasmResp.ok) {
          return null;
        }
        const wasmBytes = await wasmResp.arrayBuffer();
        if (this.offscreenSpecAdapter) {
          this.offscreenSpecAdapter.dispose();
        }
        const effectiveStepMs = this.measuredChainStepMs ?? RC1_CHAIN_STEP_MS;
        const effectiveT1Ms = effectiveStepMs[0] ?? this.engineACostModel.t1Ms;
        const effectiveHeadMs =
          this.measuredEngineBReport?.headMs ?? this.measuredEngineBHeadMs;
        const effectiveRejectMs =
          this.measuredEngineBReport?.reject?.rejectPenaltyMs ??
          this.measuredRejectPenaltyMs ??
          this.engineACostModel.rejectPenaltyMs;
        const enableEngineB = shouldEnableEngineB(effectiveHeadMs, effectiveT1Ms);
        engine.setCaptureDraftTaps?.(enableEngineB);

        const measuredOpts = this.measuredEngineBReport
          ? (() => {
              const base = toAdapterOptions(
                this.measuredEngineBReport,
                effectiveStepMs,
                this.engineACostModel.rejectPenaltyMs,
              );
              return {
                profile: base.profile,
                cutoff: { ...base.cutoff, valueGate: true },
              };
            })()
          : {
              profile: {
                stepMs: effectiveStepMs,
                rejectPenaltyMs: effectiveRejectMs,
              },
              ...(effectiveHeadMs !== null
                ? { cutoff: { headMs: effectiveHeadMs, valueGate: true } }
                : { cutoff: { valueGate: true } }),
            };
        const adapter = await OffscreenSpecAdapter.create({
          wasm: wasmBytes,
          tokenizer: {
            encode: (text: string) => tok.encode(text, { add_special_tokens: false }),
            decode: (tokens: readonly number[]) =>
              tokens.length === 0
                ? ''
                : tok.decode(Array.from(tokens, (t) => Number(t)), {
                    skip_special_tokens: false,
                  }),
          },
          ...measuredOpts,
          ...(enableEngineB
            ? {
                engineC: {
                  manifestUrl: new URL('assets/exp007/manifest.json', cycle17Base),
                  expectedSha256: EXP007B_DRAFT_HEAD_SHA256,
                  device,
                  table: { slices },
                  readShader: shaderAssets(new URL('wgsl/', cycle17Base)),
                },
              }
            : {}),
        });
        this.offscreenSpecAdapter = adapter;
        this.offscreenSpecTokenizerRef = tok;
        this.offscreenSpecDeviceRef = device;
        this.engineBActiveForAdapter = enableEngineB;
        return adapter;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        this.log('warn', 'offscreen', `OffscreenSpecAdapter unavailable (falling back to Engine A): ${msg}`);
        return null;
      } finally {
        this.offscreenSpecAdapterPromise = null;
      }
    })();
    return this.offscreenSpecAdapterPromise;
  }

  private getOrCreateEngineASpeculator(): EngineASpeculator {
    const tok = this.tokenizer!;
    if (this.engineASpeculator && this.engineATokenizerRef === tok) {
      return this.engineASpeculator;
    }
    const extraTools: ToolSchema[] = CANONICAL_TOOL_DEFINITIONS.map((def) => {
      const schemaObj = (def.parameters ?? {}) as {
        properties?: Record<string, { type?: string; enum?: string[] }>;
        required?: string[];
      };
      const req = new Set(schemaObj.required ?? []);
      const params = Object.entries(schemaObj.properties ?? {}).map(([name, meta]) => ({
        name,
        type: ((meta.type as ParamType | undefined) ?? 'string'),
        enum: meta.enum,
        required: req.has(name),
      }));
      params.sort((a, b) => Number(b.required) - Number(a.required));
      return { name: def.name, params };
    });
    this.engineASpeculator = new EngineASpeculator({
      tokenizer: {
        encode: (text: string) => tok.encode(text, { add_special_tokens: false }),
        decode: (tokens: readonly number[]) =>
          tokens.length === 0
            ? ''
            : tok.decode(Array.from(tokens, (t) => Number(t)), { skip_special_tokens: false }),
      },
      extraTools,
      adaptiveProfile: this.engineACostModel,
      minProb: 0.02,
    });
    this.engineATokenizerRef = tok;
    return this.engineASpeculator;
  }

  private async runGenerate(
    requestId: string,
    messages: ChatMessage[],
    overrideConfig?: Partial<EngineConfig>,
    groundingText?: string,
    stopOnToolCall = false,
  ): Promise<void> {
    if (this.abortedRequests.has(requestId)) {
      this.abortedRequests.delete(requestId);
      this.callbacks.onGenerationComplete(requestId, 'abort');
      return;
    }

    if (overrideConfig) {
      await this.updateConfig(overrideConfig);
    }

    if (!this.isEngineReady()) {
      await this.loadModel();
    }
    if (this.abortedRequests.has(requestId)) {
      this.abortedRequests.delete(requestId);
      this.callbacks.onGenerationComplete(requestId, 'abort');
      return;
    }
    if (!this.isEngineReady() || !this.tokenizer) {
      throw new Error('Model failed to initialize.');
    }

    const epochNow = () => performance.timeOrigin + performance.now();
    const enqueueMeta = this.pendingPhaseEnqueue.get(requestId);
    this.pendingPhaseEnqueue.delete(requestId);
    const phaseMarks: Record<string, number> = {
      ...(enqueueMeta ? { enqueue: enqueueMeta.enqueue } : {}),
      dequeue: epochNow(),
    };
    const phaseSteps: GenerationPhaseStep[] = [];

    this.activeRequestId = requestId;
    const runStartMs = performance.now();
    phaseMarks.runStart = performance.timeOrigin + runStartMs;

    try {
      this.wgslEngine?.resetDecodeBurst();
      this.updateTelemetry({
        status: 'generating',
        lastError: null,
        tokensPerSec: null,
        ttftMs: null,
        prefillChunkProgress: null,
        speculativeStats: null,
        backgroundPrefillState: 'idle',
      });

      let promptTokenIds: number[] | null = null;
      let fullTokenIds: number[] | null = null;
      let prefixTokenIds: number[] = [];
      let reusedPrefix = false;
      let restoredSlotId: string | null = null;
      let lastSlotRestoreMs: number | null = null;
      let promptOpensThink = this.config.enableThinking && !isGemma4ModelId(this.config.modelId);
      const hasActiveState = this.wgslEngine !== null && this.wgslEngine.activeSeqPos > 0;

      const normMessages = this.normalizeMessagesForChatTemplate(messages);
      const normCached = this.normalizeMessagesForChatTemplate(this.cachedMessages);
      const evaluatedLiveIds = this.getEvaluatedLiveTokenIds();
      const pendingMatchesCached =
        this.wgslEngine !== null &&
        this.cachedTokenIds.length ===
          this.wgslEngine.activeSeqPos + this.pendingLiveTokenIds.length &&
        this.pendingLiveTokenIds.length <= 2;

      if (
        !this.config.enableThinking &&
        !this.cachedThinkingMode &&
        hasActiveState &&
        normCached.length > 0 &&
        normCached[normCached.length - 1]?.role !== 'system' &&
        pendingMatchesCached &&
        isMessagePrefix(normCached, normMessages)
      ) {
        const suffixMessages = normMessages.slice(normCached.length);
        const incrementalText = this.buildIncrementalChatMlPrompt(suffixMessages);
        if (incrementalText !== null) {
          promptOpensThink = /<think>\s*$/.test(incrementalText);
          promptTokenIds = [
            ...this.pendingLiveTokenIds,
            ...this.tokenizer.encode(incrementalText, { add_special_tokens: false }),
          ];
          prefixTokenIds = evaluatedLiveIds;
          this.pendingLiveTokenIds = [];
          reusedPrefix = true;
          restoredSlotId = 'live';
        }
      }

      if (promptTokenIds === null) {
        const formattedPrompt = this.formatChatMessages(normMessages, {
          addGenerationPrompt: true,
        });
        promptOpensThink = /<think>\s*$/.test(formattedPrompt);
        fullTokenIds = this.tokenizer.encode(formattedPrompt, {
          add_special_tokens: false,
        });

        const match = this.prefixCache
          ? this.prefixCache.findBestMatch(fullTokenIds, evaluatedLiveIds)
          : { source: 'none' as const, slotId: null, matchedTokens: 0 };

        if (match.source === 'live' && match.matchedTokens < fullTokenIds.length) {
          promptTokenIds = fullTokenIds.slice(match.matchedTokens);
          prefixTokenIds = evaluatedLiveIds;
          this.pendingLiveTokenIds = [];
          reusedPrefix = true;
          restoredSlotId = 'live';
        } else if (
          match.source === 'slot' &&
          match.slotId &&
          match.matchedTokens < fullTokenIds.length
        ) {
          const restored = await this.prefixCache!.restoreSlot(match.slotId);
          this.cachedTokenIds = [...restored.entry.tokenIds];
          this.cachedMessages = restored.entry.messages.map((m) => ({ ...m }));
          this.pendingLiveTokenIds = [];
          this.cachedTokenCount = restored.seqPos;
          promptTokenIds = fullTokenIds.slice(match.matchedTokens);
          prefixTokenIds = [...restored.entry.tokenIds];
          reusedPrefix = true;
          restoredSlotId = match.slotId;
          lastSlotRestoreMs = restored.restoreMs;
        } else {
          await this.resetContext(false);
          promptTokenIds = fullTokenIds;
          prefixTokenIds = [];
          this.pendingLiveTokenIds = [];
          reusedPrefix = false;
        }
      }

      this.updateTelemetry({
        prefixReused: reusedPrefix,
        activePrefixSlot: restoredSlotId,
        ...(lastSlotRestoreMs !== null ? { lastSlotRestoreMs } : {}),
      });
      phaseMarks.cacheResolved = epochNow();
      let ctxPrefillTokens = 0;
      this.log(
        'info',
        'offscreen',
        `Prefilling ${promptTokenIds.length} tokens (backend=wgsl, tier=${this.wgslEngine?.getComputeTier() ?? 'unknown'}, pacing=${this.config.gpuPacingMode}, host=${this.executionHost}, prefixReused=${reusedPrefix}${restoredSlotId ? ` [${restoredSlotId}]` : ''}, totalContext=${this.cachedTokenCount + promptTokenIds.length}, chunkSize=${this.config.chunkSize}).`,
      );

      const sampleOpts: SamplingOptions = {
        temperature: this.config.temperature,
        topK: this.config.topK,
        topP: this.config.topP,
      };
      const totalPromptTokenCount = promptTokenIds.length;
      let totalPrefillMs = 0;

      // Checkpoint [baseSystem, activePageContext] when cold-prefilling attached page context so
      // follow-ups after multi-step tool calls can restore the evaluated page KV state in ~3ms.
      if (
        fullTokenIds &&
        restoredSlotId !== 'live' &&
        this.prefixCache &&
        this.wgslEngine &&
        normMessages.length >= 3 &&
        normMessages[1]?.role === 'system'
      ) {
        const contextMsgs = normMessages.slice(0, 2);
        const contextTokenIds = this.encodePrefixMessages(contextMsgs, false);
        const contextDeltaLen = contextTokenIds.length - prefixTokenIds.length;
        if (
          contextDeltaLen > 16 &&
          promptTokenIds.length > contextDeltaLen + 8 &&
          isExactTokenPrefix(contextTokenIds, fullTokenIds)
        ) {
          const contextDelta = promptTokenIds.slice(0, contextDeltaLen);
          const remainingPrompt = promptTokenIds.slice(contextDeltaLen);
          ctxPrefillTokens = contextDelta.length;
          phaseMarks.ctxPrefillStart = epochNow();
          const ctxRes = await this.prefillAndPickFirstToken(
            contextDelta,
            this.config.chunkSize,
            sampleOpts,
            requestId,
            true,
          );
          phaseMarks.ctxPrefillEnd = epochNow();
          totalPrefillMs += ctxRes.prefillMs;
          if (this.abortedRequests.has(requestId)) {
            throw new Error('__ABORTED__');
          }
          try {
            await this.prefixCache.saveTurnCheckpoint(contextTokenIds, contextMsgs);
            this.cachedTokenIds = [...contextTokenIds];
            this.cachedMessages = contextMsgs.map((m) => ({ ...m }));
            this.cachedTokenCount = this.wgslEngine.activeSeqPos;
            this.refreshCacheTelemetry();
          } catch {
            // Ignore checkpoint failures when volatile prefix VRAM budget is full
          }
          phaseMarks.ctxCheckpointEnd = epochNow();
          prefixTokenIds = [...contextTokenIds];
          promptTokenIds = remainingPrompt;
        }
      }

      phaseMarks.prefillStart = epochNow();
      const { nextToken: firstToken, prefillMs } = await this.prefillAndPickFirstToken(
        promptTokenIds,
        this.config.chunkSize,
        sampleOpts,
        requestId,
      );
      phaseMarks.prefillEnd = epochNow();
      totalPrefillMs += prefillMs;

      const prefillTokPerSec = totalPromptTokenCount / (Math.max(0.1, totalPrefillMs) / 1000);
      const ttftMs = performance.now() - runStartMs;
      phaseMarks.firstToken = performance.timeOrigin + runStartMs + ttftMs;
      this.updateTelemetry({ ttftMs, prefillTokPerSec });

      let nextToken = firstToken;

      const generatedIds: number[] = [];
      const sessionTokenIds = [...prefixTokenIds, ...promptTokenIds];
      const useSpeculative = this.config.enableSpeculative && sampleOpts.temperature <= 0;
      const groundingTokens =
        useSpeculative && groundingText && groundingText.trim().length > 0
          ? this.tokenizer.encode(groundingText, { add_special_tokens: false })
          : [];
      const promptScaffold = messages
        .filter((m) => m.role === 'system' || m.role === 'user')
        .map((m) => m.content)
        .join('\n')
        .slice(-12000);
      const specAdapter = useSpeculative ? await this.getOrCreateOffscreenSpecAdapter() : null;
      const speculator =
        useSpeculative && !specAdapter ? this.getOrCreateEngineASpeculator() : null;
      this.wgslEngine?.setCaptureDraftTaps?.(
        Boolean(useSpeculative && specAdapter && this.engineBActiveForAdapter),
      );
      if (useSpeculative) {
        const seedIds = [...groundingTokens, ...sessionTokenIds];
        // Cap initial SAM ingest at trailing 8192 tokens so turn startup stays <3ms on 32K+ prompts
        const initialSpecSeed = seedIds.length > 8192 ? seedIds.slice(-8192) : seedIds;
        if (specAdapter) {
          specAdapter.beginTurn(initialSpecSeed, promptScaffold, this.config.enableThinking);
        } else if (speculator) {
          speculator.beginSample(promptScaffold, this.config.enableThinking);
          speculator.observe(initialSpecSeed, false);
        }
      }
      phaseMarks.speculatorReady = epochNow();

      let specDrafted = 0;
      let specAccepted = 0;
      let specSteps = 0;
      let specSkippedSteps = 0;
      let learnedSteps = 0;
      let headInvocations = 0;
      let totalSpecGpuMs = 0;

      let emittedText = '';
      let finishReason: 'stop' | 'length' | 'abort' = 'length';
      const incrementalDecoder = new IncrementalTokenDecoder(this.tokenizer!, promptOpensThink);
      const decodeStartMs = performance.now();
      phaseMarks.decodeStart = performance.timeOrigin + decodeStartMs;

      const buildSpecStats = (): TelemetryState['speculativeStats'] => {
        if (specSteps === 0 && specSkippedSteps === 0) return null;
        const tau = specSteps > 0 ? 1 + specAccepted / specSteps : 1;
        const meanSpecStepMs = specSteps > 0 && totalSpecGpuMs > 0 ? totalSpecGpuMs / specSteps : 0;
        const speedupMultiplier =
          specSteps > 0 && meanSpecStepMs > 0
            ? Math.max(1, Number(((tau * this.engineACostModel.t1Ms) / meanSpecStepMs).toFixed(2)))
            : undefined;
        const draftRoute: 'learned' | 'copy-grammar' | 'hybrid' | undefined = specAdapter
          ? learnedSteps > 0
            ? learnedSteps === specSteps
              ? 'learned'
              : 'hybrid'
            : 'copy-grammar'
          : specSteps > 0
            ? 'copy-grammar'
            : undefined;
        return {
          drafted: specDrafted,
          accepted: specAccepted,
          steps: specSteps,
          acceptanceRatePct: specDrafted > 0 ? (specAccepted / specDrafted) * 100 : 0,
          tau,
          alpha: specDrafted > 0 ? specAccepted / specDrafted : 0,
          avgDraftLen: specSteps > 0 ? specDrafted / specSteps : 0,
          skippedSteps: specSkippedSteps,
          speedupMultiplier,
          draftRoute,
          learnedSteps,
          headInvocations,
        };
      };

      const emitTokenAndCheckDone = (tok: number): boolean => {
        if (this.abortedRequests.has(requestId)) {
          finishReason = 'abort';
          return true;
        }
        if (this.eosTokenIds.has(tok)) {
          finishReason = 'stop';
          return true;
        }
        generatedIds.push(tok);
        sessionTokenIds.push(tok);
        if (specAdapter) {
          specAdapter.observeEmitted([tok]);
        } else if (speculator) {
          speculator.observe([tok], true);
        }

        incrementalDecoder.appendToken(tok);
        const safeText = incrementalDecoder.getSafeEmittedText();
        if (safeText.length > emittedText.length) {
          const delta = safeText.slice(emittedText.length);
          emittedText = safeText;
          if (phaseMarks.firstDelta === undefined) {
            phaseMarks.firstDelta = epochNow();
          }
          this.callbacks.onTokenDelta(requestId, delta, generatedIds.length);
        }

        if (
          stopOnToolCall &&
          (emittedText.includes('</tool_call>') || emittedText.includes('<tool_call|>'))
        ) {
          finishReason = 'stop';
          return true;
        }

        if (generatedIds.length >= this.config.maxNewTokens) {
          finishReason = 'length';
          return true;
        }
        return false;
      };

      let iter = 0;
      let lastTelemetryEmitMs = decodeStartMs;
      while (generatedIds.length < this.config.maxNewTokens) {
        if (emitTokenAndCheckDone(nextToken)) {
          break;
        }

        const nowMs = performance.now();
        if (!this.config.fastDecodePath || nowMs - lastTelemetryEmitMs >= 150) {
          lastTelemetryEmitMs = nowMs;
          const decodeElapsedSec = Math.max(0.001, (nowMs - decodeStartMs) / 1000);
          this.updateTelemetry({
            tokensPerSec: generatedIds.length / decodeElapsedSec,
            baselineDecodeTokPerSec: Number((1000 / Math.max(0.5, this.engineACostModel.t1Ms)).toFixed(2)),
            speculativeStats: buildSpecStats(),
          });
        }

        const tYield0 = performance.now();
        if (this.config.yieldMode !== 'none') {
          await yieldFast(this.config.yieldMode);
        } else if (iter % 8 === 7) {
          await yieldFast('message-channel');
        }
        const yieldMs = performance.now() - tYield0;
        if (this.abortedRequests.has(requestId)) {
          finishReason = 'abort';
          break;
        }

        const remainingToMax = this.config.maxNewTokens - generatedIds.length;
        const canSpeculate =
          useSpeculative &&
          this.wgslEngine !== null &&
          remainingToMax > 1;

        let stepDraftMs = 0;
        if (canSpeculate) {
          const inToolCallBlock =
            stopOnToolCall && (emittedText.includes('<tool_call') || emittedText.includes('<|call>'));
          const supressNearClose =
            inToolCallBlock && (emittedText.endsWith('</tool_call') || emittedText.endsWith('<tool_call|'));

          const tDraft0 = performance.now();
          let cleanDraft: number[] = [];
          let routedProposal: Awaited<ReturnType<OffscreenSpecAdapter['propose']>> | null = null;
          let fallbackChain: ScoredDraftChain | null = null;

          if (specAdapter) {
            const tapInfo = this.wgslEngine!.getDraftAnchorTaps?.() ?? { taps: undefined, valid: false };
            const budget = Math.min(5, MAX_SPEC_DRAFT, remainingToMax - 1);
            const anchor =
              tapInfo.valid && tapInfo.taps ? { taps: tapInfo.taps, rootToken: nextToken } : undefined;
            if (budget >= 1 && !supressNearClose) {
              routedProposal = await specAdapter.propose(budget, anchor, false);
              if (routedProposal.headInvoked) {
                headInvocations += 1;
              }
              let draftSuffixText = emittedText;
              for (const dt of routedProposal.tree.tokens) {
                if (this.eosTokenIds.has(dt)) break;
                cleanDraft.push(dt);
                if (stopOnToolCall) {
                  draftSuffixText += this.tokenizer.decode([dt], { skip_special_tokens: false });
                  if (
                    draftSuffixText.includes('</tool_call>') ||
                    draftSuffixText.includes('<tool_call|>')
                  ) {
                    break;
                  }
                }
              }
            }
          } else if (speculator) {
            const maxDraftChain = Math.min(MAX_SPEC_DRAFT, remainingToMax - 1);
            fallbackChain = supressNearClose
              ? { tokens: [], parents: [], probs: [], keys: [] }
              : speculator.proposeChain(maxDraftChain);
            let draftSuffixText = emittedText;
            for (const dt of fallbackChain.tokens) {
              if (this.eosTokenIds.has(dt)) break;
              cleanDraft.push(dt);
              if (stopOnToolCall) {
                draftSuffixText += this.tokenizer.decode([dt], { skip_special_tokens: false });
                if (
                  draftSuffixText.includes('</tool_call>') ||
                  draftSuffixText.includes('<tool_call|>')
                ) {
                  break;
                }
              }
            }
          }
          const draftMs = performance.now() - tDraft0;
          stepDraftMs = draftMs;

          if (cleanDraft.length === 0) {
            specSkippedSteps += 1;
          } else if (
            this.wgslEngine!.activeSeqPos + 1 + cleanDraft.length <= this.wgslEngine!.maxContextTokens
          ) {
            const specRes = await this.wgslEngine!.decodeSpeculativeStep(
              nextToken,
              cleanDraft,
              sampleOpts,
              this.config.gpuPacingMode,
            );
            this.engineACostModel.recordStepTiming(
              1 + cleanDraft.length,
              specRes.gpuMs,
              specRes.acceptedCount < specRes.draftedCount,
            );
            if (specAdapter && routedProposal) {
              if (routedProposal.source === 'C') {
                learnedSteps += 1;
              }
              const acceptedIndices = Array.from(
                { length: specRes.acceptedCount },
                (_, idx) => idx,
              );
              specAdapter.commit(routedProposal, acceptedIndices, []);
            } else if (speculator && fallbackChain) {
              const verifiedSlice: ScoredDraftChain = {
                tokens: fallbackChain.tokens.slice(0, cleanDraft.length),
                parents: fallbackChain.parents.slice(0, cleanDraft.length),
                probs: fallbackChain.probs.slice(0, cleanDraft.length),
                keys: fallbackChain.keys.slice(0, cleanDraft.length),
              };
              speculator.feedback(verifiedSlice, specRes.acceptedCount);
            }
            this.cachedTokenCount = this.wgslEngine!.activeSeqPos;
            specDrafted += specRes.draftedCount;
            specAccepted += specRes.acceptedCount;
            specSteps += 1;
            totalSpecGpuMs += specRes.gpuMs;
            if (!this.config.fastDecodePath || iter % 4 === 0) {
              this.refreshCacheTelemetry();
            }

            const tEmit0 = performance.now();
            let stoppedMidBatch = false;
            for (let i = 0; i < specRes.acceptedCount; i++) {
              if (emitTokenAndCheckDone(specRes.tokens[i]!)) {
                stoppedMidBatch = true;
                break;
              }
            }
            const emitMs = performance.now() - tEmit0;
            phaseSteps.push({
              kind: 'chain',
              M: 1 + cleanDraft.length,
              kept: 1 + specRes.acceptedCount,
              draftMs,
              planMs: specRes.planMs ?? 0,
              encodeMs: specRes.encodeMs ?? 0,
              mapMs: specRes.mapMs ?? specRes.gpuMs,
              rollbackEncodeMs: specRes.rollbackEncodeMs ?? 0,
              rollbackWaitMs: specRes.rollbackWaitMs ?? 0,
              emitMs,
              yieldMs,
              planBuilt: specRes.planBuilt ?? false,
            });
            if (stoppedMidBatch) {
              break;
            }
            nextToken = specRes.tokens[specRes.acceptedCount]!;
            iter++;
            continue;
          }
        }

        const stepRes = await this.decodeNextToken(nextToken, sampleOpts, iter);
        this.engineACostModel.recordStepTiming(1, stepRes.forwardMs, false);
        phaseSteps.push({
          kind: 'scalar',
          M: 1,
          kept: 1,
          draftMs: stepDraftMs,
          planMs: 0,
          encodeMs: stepRes.encodeMs ?? 0,
          mapMs: stepRes.mapMs ?? stepRes.forwardMs,
          rollbackEncodeMs: 0,
          rollbackWaitMs: 0,
          emitMs: 0,
          yieldMs,
          planBuilt: false,
        });
        nextToken = stepRes.nextToken;
        iter++;
      }

      phaseMarks.decodeEnd = epochNow();
      this.wgslEngine?.flushPendingRollback?.('end-of-turn');
      if (specAdapter) {
        specAdapter.endTurn();
      } else if (speculator) {
        speculator.endSample();
      }

      if (generatedIds.length > 0) {
        const totalDecodeSec = Math.max(0.001, (performance.now() - decodeStartMs) / 1000);
        this.updateTelemetry({
          tokensPerSec: generatedIds.length / totalDecodeSec,
          baselineDecodeTokPerSec: Number((1000 / Math.max(0.5, this.engineACostModel.t1Ms)).toFixed(2)),
          speculativeStats: buildSpecStats(),
        });
        this.refreshCacheTelemetry();
      }

      if (finishReason !== 'abort') {
        const finalDecoded = decodeGeneratedText(this.tokenizer, generatedIds, {
          preOpenedThink: promptOpensThink,
        });
        if (finalDecoded.length > emittedText.length) {
          this.callbacks.onTokenDelta(
            requestId,
            finalDecoded.slice(emittedText.length),
            generatedIds.length,
          );
          emittedText = finalDecoded;
        }
      }

      const evalSeqPos = this.wgslEngine?.activeSeqPos ?? -1;
      if (finishReason !== 'abort' && evalSeqPos >= 0 && evalSeqPos <= sessionTokenIds.length) {
        this.pendingLiveTokenIds = sessionTokenIds.slice(evalSeqPos);
        this.cachedMessages = [...messages, { role: 'assistant', content: emittedText }];
        this.cachedThinkingMode = this.config.enableThinking;
        this.cachedTokenIds = sessionTokenIds;
      } else {
        await this.resetContext(this.prefixCache?.hasBasePrompt() ?? false);
      }

      phaseMarks.complete = epochNow();
      const phases: GenerationPhaseTrace = {
        marks: phaseMarks,
        info: {
          promptTokens: totalPromptTokenCount,
          ctxPrefillTokens,
          prefixReused: reusedPrefix,
          slot: restoredSlotId,
          slotRestoreMs: lastSlotRestoreMs,
          bgPrefillActiveAtEnqueue: enqueueMeta?.bgPrefillActiveAtEnqueue ?? false,
        },
        steps: phaseSteps,
        tools: [],
      };
      this.updateTelemetry({ status: 'ready', prefillChunkProgress: null });
      const trace: GenerationTokenTrace = {
        modelId: this.config.modelId,
        promptIds: sessionTokenIds.slice(0, Math.max(0, sessionTokenIds.length - generatedIds.length)),
        generatedIds: [...generatedIds],
        promptText: promptScaffold,
        generatedText: emittedText,
        speculativeStats: buildSpecStats(),
        phases,
      };
      this.callbacks.onGenerationComplete(requestId, finishReason, trace);
    } catch (err) {
      this.wgslEngine?.flushPendingRollback?.('end-of-turn');
      this.wgslEngine?.invalidateDraftAnchor?.();
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === '__ABORTED__' || msg === 'Generation aborted') {
        await this.resetContext();
        this.updateTelemetry({ status: 'ready', prefillChunkProgress: null });
        this.callbacks.onGenerationComplete(requestId, 'abort');
        return;
      }
      const errMsg = `Inference failed at ${this.cachedTokenCount} tokens: ${msg}`;
      this.log('error', 'offscreen', errMsg);
      this.updateTelemetry({
        status: 'ready',
        lastError: errMsg,
        prefillChunkProgress: null,
      });
      this.callbacks.onGenerationError(requestId, errMsg);
    } finally {
      this.activeAbortController = null;
      this.abortedRequests.delete(requestId);
      if (this.activeRequestId === requestId) {
        this.activeRequestId = null;
      }
    }
  }

  async runBenchmarkSuite(
    contextLengths: number[],
    decodeTokens: number,
    chunkSizeOverride?: number,
  ): Promise<void> {
    if (!this.isEngineReady()) {
      await this.loadModel();
    }
    if (!this.isEngineReady() || !this.tokenizer || !this.wgslEngine) {
      throw new Error('Model is not loaded.');
    }

    const startedAt = new Date().toISOString();
    const chunkSize = chunkSizeOverride ?? this.config.chunkSize;
    const weightGB = this.wgslEngine.getMemoryStats().weightMb / 1024;
    const points: BenchmarkPoint[] = [];
    let maxSafeTokens = 0;
    let oomTokenCount: number | null = null;

    this.updateTelemetry({ status: 'benchmarking', lastError: null });

    try {
      for (const targetTokens of contextLengths) {
        await this.resetContext();
        this.callbacks.onBenchmarkProgress(
          `Benchmarking ${targetTokens.toLocaleString()} context tokens (chunk=${chunkSize}, backend=wgsl, pacing=${this.config.gpuPacingMode})...`,
        );

        const promptIds = buildSyntheticTokenIds(this.tokenizer, targetTokens);
        const startRunMs = performance.now();

        try {
          const greedyOpts: SamplingOptions = { temperature: 0, topK: 1, topP: 1 };
          const { nextToken: firstToken, prefillMs } = await this.prefillAndPickFirstToken(
            promptIds,
            chunkSize,
            greedyOpts,
          );
          const ttftMs = performance.now() - startRunMs;
          const prefillTokPerSec = promptIds.length / (prefillMs / 1000);

          let nextToken = firstToken;

          let forwardMsSum = 0;
          let logitsMsSum = 0;
          let sampleMsSum = 0;
          let yieldMsSum = 0;

          const decodeStartMs = performance.now();
          for (let i = 0; i < decodeTokens; i++) {
            const stepRes = await this.decodeNextToken(nextToken, greedyOpts, i);
            nextToken = stepRes.nextToken;
            forwardMsSum += stepRes.forwardMs;
            logitsMsSum += stepRes.logitsMs;
            sampleMsSum += stepRes.sampleMs;
            yieldMsSum += stepRes.breatherMs;
            if (this.config.yieldMode !== 'none') {
              const tY = performance.now();
              await yieldFast(this.config.yieldMode);
              yieldMsSum += performance.now() - tY;
            }
          }
          const decodeMs = Math.max(1, performance.now() - decodeStartMs);
          const decodeTokPerSec = decodeTokens / (decodeMs / 1000);
          this.refreshCacheTelemetry();

          const mem = this.wgslEngine.getMemoryStats();
          const kvCacheMb = mem.kvCacheMb;
          const deltaNetStateMb = mem.deltaNetStateMb;
          const estVramMb = mem.estVramMb;
          const bw = computeBandwidthStats(
            decodeTokPerSec,
            weightGB,
            this.config.gpuBandwidthGBs,
          );

          const point: BenchmarkPoint = {
            targetTokens,
            actualPromptTokens: promptIds.length,
            decodeTokens,
            chunkSize,
            prefillMs,
            ttftMs,
            prefillTokPerSec,
            decodeTokPerSec,
            kvCacheMb,
            deltaNetStateMb,
            estVramMb,
            achievedBandwidthGBs: bw.achievedGBs,
            bandwidthUtilizationPct: bw.utilizationPct,
            breakdown: {
              avgForwardMs: forwardMsSum / Math.max(1, decodeTokens),
              avgLogitsMs: logitsMsSum / Math.max(1, decodeTokens),
              avgSampleMs: sampleMsSum / Math.max(1, decodeTokens),
              avgYieldMs: yieldMsSum / Math.max(1, decodeTokens),
            },
            status: 'ok',
          };
          points.push(point);
          maxSafeTokens = Math.max(maxSafeTokens, promptIds.length + decodeTokens);
          this.callbacks.onBenchmarkProgress(
            `Done ${targetTokens.toLocaleString()}t: prefill ${prefillTokPerSec.toFixed(0)} t/s, decode ${decodeTokPerSec.toFixed(1)} t/s (${bw.utilizationPct.toFixed(1)}% BW)`,
            point,
          );
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          const isDeviceLost = /device.*lost|webgpu/i.test(msg);
          const isOom = /oom|memory|allocation|buffer/i.test(msg);
          oomTokenCount = this.cachedTokenCount || targetTokens;
          const point: BenchmarkPoint = {
            targetTokens,
            actualPromptTokens: promptIds.length,
            decodeTokens: 0,
            chunkSize,
            prefillMs: 0,
            ttftMs: 0,
            prefillTokPerSec: 0,
            decodeTokPerSec: 0,
            kvCacheMb: this.telemetry.kvCacheMb,
            deltaNetStateMb: this.telemetry.deltaNetStateMb,
            estVramMb: this.telemetry.estVramMb,
            achievedBandwidthGBs: 0,
            bandwidthUtilizationPct: 0,
            status: isDeviceLost ? 'device_lost' : isOom ? 'oom' : 'error',
            error: msg,
          };
          points.push(point);
          this.log('error', 'offscreen', `Benchmark failed at ${oomTokenCount} tokens: ${msg}`);
          break;
        }
      }
    } finally {
      await this.resetContext();
      this.updateTelemetry({ status: 'ready', prefillChunkProgress: null });
    }

    const okPoints = points.filter((p) => p.status === 'ok');
    const peakDecode = okPoints.reduce((m, p) => Math.max(m, p.decodeTokPerSec), 0);
    const decision = computeBandwidthStats(peakDecode, weightGB, this.config.gpuBandwidthGBs);

    const suite: BenchmarkSuiteResult = {
      startedAt,
      finishedAt: new Date().toISOString(),
      modelId: this.config.modelId,
      dtype: this.config.dtype,
      gpuBandwidthGBs: this.config.gpuBandwidthGBs,
      points,
      maxSafeTokens,
      oomTokenCount,
      recommendation: decision.recommendation,
    };

    this.lastBenchmark = suite;
    this.callbacks.onBenchmarkResult(suite);
  }

  async runNeedleInHaystack(
    targetTokens: number,
    depthRatio: number,
    chunkSizeOverride?: number,
  ): Promise<void> {
    if (!this.isEngineReady()) {
      await this.loadModel();
    }
    if (!this.isEngineReady() || !this.tokenizer) {
      throw new Error('Model is not loaded.');
    }

    const chunkSize = chunkSizeOverride ?? this.config.chunkSize;
    this.updateTelemetry({ status: 'benchmarking', lastError: null });
    this.callbacks.onNiahProgress(
      `Building ${targetTokens.toLocaleString()}-token Needle-in-a-Haystack prompt (${(depthRatio * 100).toFixed(0)}% depth)...`,
    );

    const passkey = 'VELVET-ORBIT-7492';
    const built = buildNeedleInHaystackPrompt(this.tokenizer, targetTokens, depthRatio, passkey);

    try {
      await this.resetContext();
      this.callbacks.onNiahProgress(
        `Prefilling ${built.actualTokens.toLocaleString()} NIAH tokens in ${chunkSize}-token chunks...`,
      );

      const startMs = performance.now();
      const greedyOpts: SamplingOptions = { temperature: 0, topK: 1, topP: 1 };
      const { nextToken: firstToken, prefillMs } = await this.prefillAndPickFirstToken(
        built.tokenIds,
        chunkSize,
        greedyOpts,
      );
      const ttftMs = performance.now() - startMs;
      const prefillTokPerSec = built.actualTokens / (prefillMs / 1000);

      let nextToken = firstToken;
      const generatedIds: number[] = [];
      const decodeStartMs = performance.now();

      for (let i = 0; i < 48; i++) {
        if (this.eosTokenIds.has(nextToken)) {
          break;
        }
        generatedIds.push(nextToken);
        if (this.config.yieldMode !== 'none') {
          await yieldFast(this.config.yieldMode);
        }
        const stepRes = await this.decodeNextToken(nextToken, greedyOpts, i);
        nextToken = stepRes.nextToken;
      }

      const decodeMs = Math.max(1, performance.now() - decodeStartMs);
      const decodeTokPerSec = generatedIds.length / (decodeMs / 1000);
      const responseText = this.tokenizer.decode(generatedIds, { skip_special_tokens: true }).trim();
      const passed = verifyNeedleAnswer(responseText, passkey);

      const result: NiahResult = {
        targetTokens,
        actualTokens: built.actualTokens,
        needleDepthPct: depthRatio,
        passkey,
        responseText,
        passed,
        ttftMs,
        prefillTokPerSec,
        decodeTokPerSec,
      };
      this.lastNiah = result;
      this.log(
        passed ? 'info' : 'warn',
        'offscreen',
        `NIAH ${built.actualTokens} tokens @ ${(depthRatio * 100).toFixed(0)}%: ${passed ? 'PASS' : 'FAIL'} ("${responseText}")`,
      );
      this.callbacks.onNiahResult(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const result: NiahResult = {
        targetTokens,
        actualTokens: built.actualTokens,
        needleDepthPct: depthRatio,
        passkey,
        responseText: '',
        passed: false,
        ttftMs: 0,
        prefillTokPerSec: 0,
        decodeTokPerSec: 0,
        error: msg,
      };
      this.lastNiah = result;
      this.log('error', 'offscreen', `NIAH failed: ${msg}`);
      this.callbacks.onNiahResult(result);
    } finally {
      this.updateTelemetry({ status: 'ready', prefillChunkProgress: null });
    }
  }

  async runSpeculativeMicrobench(): Promise<void> {
    if (this.loadingPromise) {
      await this.loadingPromise.catch(() => {});
    }
    if (!this.isEngineReady()) {
      await this.loadModel();
    }
    if (!this.isEngineReady() || !this.tokenizer || !this.wgslEngine) {
      throw new Error('WGSL Qwen3.5 engine not active for speculative microbench.');
    }

    const engine = this.wgslEngine;
    const greedy: SamplingOptions = { temperature: 0, topK: 1, topP: 1 };
    const slotId = '__spec_microbench';

    this.updateTelemetry({ status: 'benchmarking', lastError: null });
    try {
      await this.resetContext(false);
      const encodedBenchPrompt = this.tokenizer.encode(
        'You are a helpful browser assistant. Summarize the exact cluster metrics step by step: Alpha throughput is 415 req/s, Beta throughput is 635 req/s, combined total is 1050 req/s in region us-east-webgpu-1. '.repeat(3),
        { add_special_tokens: false },
      );
      const promptIds =
        encodedBenchPrompt.length >= 96
          ? encodedBenchPrompt.slice(0, 96)
          : [...encodedBenchPrompt, ...buildSyntheticTokenIds(this.tokenizer, 96 - encodedBenchPrompt.length)];
      const pref = await engine.prefill(promptIds, {
        chunkSize: 64,
        sampling: greedy,
        pacingMode: 'max-speed',
        yieldFn: async () => {},
      });

      // Generate 12 exact greedy reference tokens from seqPos=96
      const refTokens: number[] = [pref.nextTokenId];
      let cur = pref.nextTokenId;
      for (let i = 0; i < 12; i++) {
        const d = await engine.decodeStep(cur, greedy, 'max-speed');
        cur = d.nextToken;
        refTokens.push(cur);
      }

      // Reset to 96 + prefill and snapshot right before refTokens[0]
      await this.resetContext(false);
      await engine.prefill(promptIds, {
        chunkSize: 64,
        sampling: greedy,
        pacingMode: 'max-speed',
        yieldFn: async () => {},
      });
      await engine.saveGpuPrefixSlot(slotId);

      // Measure T(1) over 12 steps (after 2 warmup steps)
      let t1Sum = 0;
      const t1Iters = 12;
      let t1Tok = refTokens[0]!;
      for (let i = 0; i < 2 + t1Iters; i++) {
        const res = await engine.decodeStep(t1Tok, greedy, 'max-speed');
        t1Tok = res.nextToken;
        if (i >= 2) t1Sum += res.gpuMs;
      }
      const t1Ms = t1Sum / t1Iters;

      const measureM = async (M: number, acceptAll: boolean): Promise<number> => {
        const draftLen = M - 1;
        const acceptDraft = refTokens.slice(1, 1 + draftLen);
        const wrongDraft = new Array<number>(draftLen).fill((refTokens[1]! + 17) % 1000);
        const step2Draft = refTokens.slice(2, 2 + draftLen);
        // Warmup builds cached VerifyGraphPlan for this M
        await engine.restoreGpuPrefixSlot(slotId);
        await engine.decodeSpeculativeStep(refTokens[0]!, acceptDraft, greedy, 'max-speed');

        let sum = 0;
        const iters = 10;
        for (let i = 0; i < iters; i++) {
          await engine.restoreGpuPrefixSlot(slotId);
          if (acceptAll) {
            const r = await engine.decodeSpeculativeStep(refTokens[0]!, acceptDraft, greedy, 'max-speed');
            sum += r.gpuMs;
          } else {
            // Pair reject step (defers rollback) with next verify step (merges rollback via P9b submitWith)
            await engine.decodeSpeculativeStep(refTokens[0]!, wrongDraft, greedy, 'max-speed');
            const r2 = await engine.decodeSpeculativeStep(refTokens[1]!, step2Draft, greedy, 'max-speed');
            sum += r2.gpuMs;
          }
        }
        return sum / iters;
      };

      const t2Ms = await measureM(2, true);
      const t3Ms = await measureM(3, true);
      const t5Ms = await measureM(5, true);
      const t8Ms = await measureM(8, true);
      const t5RejectMs = await measureM(5, false);

      let tree: SpecMicrobenchResult['tree'] | undefined;
      if (engine.isTreeSpecSupported()) {
        const measureTreeM = async (M: number, acceptAll: boolean): Promise<number> => {
          const draftLen = M - 1;
          const draftTokens = acceptAll
            ? refTokens.slice(1, 1 + Math.min(draftLen, 8))
            : new Array<number>(Math.min(draftLen, 8)).fill((refTokens[1]! + 17) % 1000);
          while (draftTokens.length < draftLen) {
            draftTokens.push((refTokens[1]! + 19 + draftTokens.length) % 1000);
          }
          const draftParents = draftTokens.map((_, idx) => (idx < 8 ? idx : idx % 4));
          const depths = [0];
          const ancestorMasks = [1];
          for (let i = 0; i < draftTokens.length; i++) {
            const p = draftParents[i]!;
            depths.push(depths[p]! + 1);
            ancestorMasks.push(ancestorMasks[p]! | (1 << (i + 1)));
          }
          const treeInput = { draftTokens, draftParents, ancestorMasks, depths };
          await engine.restoreGpuPrefixSlot(slotId);
          await engine.decodeSpeculativeTreeStep!(refTokens[0]!, treeInput, greedy, 'max-speed');

          let sum = 0;
          const iters = 10;
          for (let i = 0; i < iters; i++) {
            await engine.restoreGpuPrefixSlot(slotId);
            const r = await engine.decodeSpeculativeTreeStep!(
              refTokens[0]!,
              treeInput,
              greedy,
              'max-speed',
            );
            sum += r.gpuMs;
          }
          return sum / iters;
        };

        const tr2 = await measureTreeM(2, true);
        const tr3 = await measureTreeM(3, true);
        const tr5 = await measureTreeM(5, true);
        const tr8 = await measureTreeM(8, true);
        const tr12 = await measureTreeM(12, true);
        const tr16 = await measureTreeM(16, true);
        const tr5Reject = await measureTreeM(5, false);

        // Strict non-linear tree parity check (src != dst compaction + M > 8 split)
        const wrong = (refTokens[1]! + 17) % 1000;
        const verifyBranchParity = async (
          label: string,
          draftTokens: number[],
          draftParents: number[],
          expectedSlots: number[],
        ): Promise<void> => {
          const depths = [0];
          const ancestorMasks = [1];
          for (let i = 0; i < draftTokens.length; i++) {
            const p = draftParents[i]!;
            depths.push(depths[p]! + 1);
            ancestorMasks.push(ancestorMasks[p]! | (1 << (i + 1)));
          }
          await engine.restoreGpuPrefixSlot(slotId);
          const r = await engine.decodeSpeculativeTreeStep!(
            refTokens[0]!,
            { draftTokens, draftParents, ancestorMasks, depths },
            greedy,
            'max-speed',
          );
          const gotFollow = (await engine.decodeStep(r.tokens[r.acceptedCount]!, greedy, 'max-speed')).nextToken;
          const ok =
            r.acceptedCount === expectedSlots.length &&
            expectedSlots.every((s, idx) => r.acceptedSlotIds[idx] === s) &&
            r.tokens[r.acceptedCount] === refTokens[1 + expectedSlots.length] &&
            gotFollow === refTokens[2 + expectedSlots.length];
          if (!ok) {
            throw new Error(
              `Tree parity [${label}] failed: got=${JSON.stringify({ accepted: r.acceptedCount, slots: r.acceptedSlotIds, tokens: r.tokens, gotFollow })} expectedSlots=${JSON.stringify(expectedSlots)} ref=${JSON.stringify(refTokens.slice(0, 6))}`,
            );
          }
        };

        await verifyBranchParity(
          'M=4 zero-accept rollback',
          [wrong, wrong, wrong],
          [0, 1, 0],
          [],
        );
        await verifyBranchParity(
          'M=6 non-linear src!=dst',
          [wrong, refTokens[1]!, wrong, refTokens[2]!, wrong],
          [0, 0, 2, 2, 4],
          [2, 4],
        );
        await verifyBranchParity(
          'M=12 split-batch src!=dst',
          [wrong, wrong, wrong, wrong, wrong, wrong, wrong, wrong, refTokens[1]!, refTokens[2]!, wrong],
          [0, 0, 1, 2, 3, 4, 5, 6, 0, 9, 10],
          [9, 10],
        );

        // Consecutive multi-step tree verification (step 1 accepts [2, 4], step 2 accepts [3, 6] across M=9 split)
        await engine.restoreGpuPrefixSlot(slotId);
        const step1Input = {
          draftTokens: [wrong, refTokens[1]!, wrong, refTokens[2]!, wrong],
          draftParents: [0, 0, 2, 2, 4],
          ancestorMasks: [1, 3, 5, 13, 21, 53],
          depths: [0, 1, 1, 2, 2, 3],
        };
        const s1 = await engine.decodeSpeculativeTreeStep!(refTokens[0]!, step1Input, greedy, 'max-speed');
        const step2Tok0 = s1.tokens[s1.acceptedCount]!; // == refTokens[3]
        const step2Tokens = [wrong, wrong, refTokens[4]!, wrong, wrong, refTokens[5]!, wrong, wrong];
        const step2Parents = [0, 0, 0, 1, 2, 3, 3, 4];
        const step2Depths = [0];
        const step2Masks = [1];
        for (let i = 0; i < step2Tokens.length; i++) {
          const p = step2Parents[i]!;
          step2Depths.push(step2Depths[p]! + 1);
          step2Masks.push(step2Masks[p]! | (1 << (i + 1)));
        }
        const s2 = await engine.decodeSpeculativeTreeStep!(
          step2Tok0,
          { draftTokens: step2Tokens, draftParents: step2Parents, ancestorMasks: step2Masks, depths: step2Depths },
          greedy,
          'max-speed',
        );
        const s2Follow = (await engine.decodeStep(s2.tokens[s2.acceptedCount]!, greedy, 'max-speed')).nextToken;
        if (
          s1.acceptedCount !== 2 ||
          s2.acceptedCount !== 2 ||
          s2.acceptedSlotIds[0] !== 3 ||
          s2.acceptedSlotIds[1] !== 6 ||
          s2.tokens[s2.acceptedCount] !== refTokens[6] ||
          s2Follow !== refTokens[7]
        ) {
          throw new Error(
            `Consecutive tree parity failed: s1=${JSON.stringify(s1)} s2=${JSON.stringify(s2)} s2Follow=${s2Follow} ref6_7=${JSON.stringify(refTokens.slice(6, 8))}`,
          );
        }

        tree = {
          t2Ms: Number(tr2.toFixed(2)),
          t3Ms: Number(tr3.toFixed(2)),
          t5Ms: Number(tr5.toFixed(2)),
          t8Ms: Number(tr8.toFixed(2)),
          t12Ms: Number(tr12.toFixed(2)),
          t16Ms: Number(tr16.toFixed(2)),
          t5RejectMs: Number(tr5Reject.toFixed(2)),
          ratioT5ToT1: Number((tr5 / Math.max(0.01, t1Ms)).toFixed(3)),
          ratioT8ToT1: Number((tr8 / Math.max(0.01, t1Ms)).toFixed(3)),
          ratioT16ToT1: Number((tr16 / Math.max(0.01, t1Ms)).toFixed(3)),
          rejectOverheadMs: Number((tr5Reject - tr5).toFixed(2)),
          chainRejectOverheadMs: Number((t5RejectMs - t5Ms).toFixed(2)),
        };
      }

      let engineB: SpecMicrobenchResult['engineB'];
      let measuredEbReport: EngineBBenchReport | null = null;
      const ebDevice = engine.getDevice?.() ?? null;
      const ebSlices = engine.getTiedTableSlices?.() ?? [];
      const ebVocab = ebSlices.reduce((acc, s) => Math.max(acc, s.endRow), 0);
      if (
        ebDevice &&
        ebSlices.length > 0 &&
        ebVocab === 81920 &&
        typeof engine.measureSpecStep === 'function' &&
        typeof fetch === 'function'
      ) {
        try {
          const baseStr = resolveExtensionAssetUrl('cycle17/', this.assetBaseUrl);
          const originFallback =
            typeof globalThis.location?.href === 'string'
              ? globalThis.location.href
              : 'https://extension.invalid/';
          const cycle17Base = new URL(
            baseStr.endsWith('/') ? baseStr : `${baseStr}/`,
            originFallback,
          );
          const bundle = await loadGdnBundle(new URL('assets/exp007/manifest.json', cycle17Base));
          const tapsResp = await fetch(new URL('fixtures/trained-exp007/taps.f32', cycle17Base));
          if (bundle && tapsResp.ok) {
            const taps = new Float32Array(await tapsResp.arrayBuffer());
            const runtime = await GdnDraftRuntime.create(
              ebDevice,
              bundle,
              { slices: ebSlices },
              shaderAssets(new URL('wgsl/', cycle17Base)),
            );
            try {
              const ebReport = await runEngineBBench(
                runtime,
                { taps, rootToken: 271 },
                { measure: (M, reject) => engine.measureSpecStep!(M, reject) },
                { budgetMs: 10_000 },
              );
              measuredEbReport = ebReport;
              engineB = {
                headSha256: bundle.sha256,
                headMs: Number(ebReport.headMs.toFixed(3)),
                head: ebReport.head.map((h) => ({
                  ...h,
                  medianMs: Number(h.medianMs.toFixed(3)),
                  p10Ms: Number(h.p10Ms.toFixed(3)),
                  p90Ms: Number(h.p90Ms.toFixed(3)),
                })),
                ...(ebReport.reject
                  ? {
                      reject: {
                        M: ebReport.reject.M,
                        acceptMs: Number(ebReport.reject.acceptMs.toFixed(3)),
                        rejectMs: Number(ebReport.reject.rejectMs.toFixed(3)),
                        rejectPenaltyMs: Number(ebReport.reject.rejectPenaltyMs.toFixed(3)),
                        samples: ebReport.reject.samples,
                      },
                    }
                  : {}),
                elapsedMs: Number(ebReport.elapsedMs.toFixed(1)),
              };
            } finally {
              runtime.dispose();
            }
          }
        } catch (err) {
          engineB = {
            headSha256: EXP007B_DRAFT_HEAD_SHA256,
            headMs: 0,
            head: [],
            elapsedMs: 0,
            error: err instanceof Error ? err.message : String(err),
          };
        }
      }

      engine.deleteGpuPrefixSlot(slotId);
      engine.deleteGpuPrefixSlot('__eb_probe__');
      await this.resetContext(this.prefixCache?.hasBasePrompt() ?? false);

      const ratioT5ToT1 = Number((t5Ms / Math.max(0.01, t1Ms)).toFixed(3));
      const ratioT8ToT1 = Number((t8Ms / Math.max(0.01, t1Ms)).toFixed(3));
      const stepMs = [
        t1Ms,
        t2Ms,
        t3Ms,
        (t3Ms + t5Ms) / 2,
        t5Ms,
        t5Ms + (t8Ms - t5Ms) / 3,
        t5Ms + (2 * (t8Ms - t5Ms)) / 3,
        t8Ms,
      ];
      const rejectP = Math.max(
        0.2,
        measuredEbReport?.reject?.rejectPenaltyMs ?? t5RejectMs - t5Ms,
      );
      this.applyMeasuredSpecCalibration(
        t1Ms,
        stepMs,
        measuredEbReport?.headMs ?? this.measuredEngineBHeadMs ?? 999,
        rejectP,
      );
      if (measuredEbReport) {
        this.measuredEngineBReport = measuredEbReport;
        await saveSpecCalibration({
          adapterName: this.adapterName,
          dropinSha256: EXP007B_DRAFT_HEAD_SHA256,
          t1Ms,
          stepMs,
          headMs: measuredEbReport.headMs,
          rejectPenaltyMs: rejectP,
          measuredAtIso: new Date().toISOString(),
        });
      }
      if (this.offscreenSpecAdapter) {
        this.offscreenSpecAdapter.dispose();
        this.offscreenSpecAdapter = null;
      }
      this.offscreenSpecAdapterPromise = null;
      this.engineBActiveForAdapter = false;
      if (tree) {
        const treeSlope = Math.max(0.15 * t1Ms, (tree.t5Ms - tree.t2Ms) / 3);
        this.engineATreeCostModel.t1Ms = Math.max(t1Ms, tree.t2Ms - treeSlope);
        this.engineATreeCostModel.slopePerTokenMs = treeSlope;
        this.engineATreeCostModel.rejectPenaltyMs = Math.max(0, tree.t5RejectMs - tree.t5Ms);
      }
      const result: SpecMicrobenchResult = {
        modelId: this.config.modelId,
        t1Ms: Number(t1Ms.toFixed(2)),
        t2Ms: Number(t2Ms.toFixed(2)),
        t3Ms: Number(t3Ms.toFixed(2)),
        t5Ms: Number(t5Ms.toFixed(2)),
        t8Ms: Number(t8Ms.toFixed(2)),
        t5RejectMs: Number(t5RejectMs.toFixed(2)),
        ratioT5ToT1,
        ratioT8ToT1,
        gatePass: ratioT5ToT1 <= 2.7,
        ...(tree ? { tree } : {}),
        ...(engineB ? { engineB } : {}),
      };
      this.callbacks.onSpecMicrobenchResult?.(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.callbacks.onSpecMicrobenchResult?.({
        modelId: this.config.modelId,
        t1Ms: 0,
        t2Ms: 0,
        t3Ms: 0,
        t5Ms: 0,
        t8Ms: 0,
        t5RejectMs: 0,
        ratioT5ToT1: 0,
        ratioT8ToT1: 0,
        gatePass: false,
        error: msg,
      });
    } finally {
      this.updateTelemetry({ status: 'ready', prefillChunkProgress: null });
    }
  }
}
