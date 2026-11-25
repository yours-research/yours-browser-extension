import { AutoTokenizer, env, type PreTrainedTokenizer } from '@huggingface/transformers';
import {
  DEFAULT_ENGINE_CONFIG,
  INITIAL_TELEMETRY_STATE,
  type BenchmarkPoint,
  type BenchmarkSuiteResult,
  type ChatMessage,
  type EngineConfig,
  type ExecutionHost,
  type LogEntry,
  type NiahResult,
  type OpfsSpikeResult,
  type OpfsStatusInfo,
  type TelemetryState,
  type WebGpuProbeInfo,
  type YieldMode,
} from '../shared/protocol.ts';
import { buildBaseSystemMessage } from '../shared/systemPrompt.ts';
import { buildNeedleInHaystackPrompt, buildSyntheticTokenIds, verifyNeedleAnswer } from './benchmarks.ts';
import { resolveExtensionAssetUrl } from './gpuPacing.ts';
import { clearWgslWeightCache } from './onnxWeightLoader.ts';
import { OpfsModelCache } from './opfsCache.ts';
import { BASE_SYSTEM_SLOT_ID, PrefixCacheManager } from './prefixCacheManager.ts';
import { computeBandwidthStats, isMessagePrefix, type SamplingOptions } from './sampling.ts';
import { SuffixSpeculator } from './suffixSpeculator.ts';
import { installWebGpuLimitPatch, probeWebGpu } from './webgpuLimits.ts';
import { WgslQwen35Engine, type WgslCpuSnapshot } from './wgslRealEngine.ts';

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
  onNiahProgress: (message: string) => void;
  onNiahResult: (result: NiahResult) => void;
  onTokenDelta: (requestId: string, delta: string, tokenCount: number) => void;
  onGenerationComplete: (requestId: string, finishReason: 'stop' | 'length' | 'abort') => void;
  onGenerationError: (requestId: string, error: string) => void;
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

  private wgslEngine: WgslQwen35Engine | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private eosTokenIds = new Set<number>();

  private cachedMessages: ChatMessage[] = [];
  private cachedTokenCount = 0;
  private cachedTokenIds: number[] = [];
  private cachedThinkingMode = false;
  private wgslSnapshot: WgslCpuSnapshot | null = null;
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
  private bgPrefillSeq = 0;

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

    this.config = {
      ...this.config,
      ...patch,
      backend: 'wgsl',
      wgslKvMode: 'int8',
      wgslMaxContext: nextMaxContext,
      forceComputeTier: nextForceTier,
      modelId: DEFAULT_ENGINE_CONFIG.modelId,
      dtype: 'q4f16',
    };
    this.callbacks.onConfig(this.config);

    if ((contextCapChanged || tierChanged) && this.wgslEngine) {
      this.log('info', 'offscreen', 'WGSL engine geometry/tier config changed; unloading active session.');
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
      this.updateTelemetry({
        status: 'downloading',
        modelId: this.config.modelId,
        dtype: this.config.dtype,
        executionHost: this.executionHost,
        loadProgressPct: 0,
        loadPhaseLabel: 'Loading bundled Qwen3.5-4B tokenizer & weights...',
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
      const tokenizerUrl = resolveExtensionAssetUrl('models/qwen35-4b', this.assetBaseUrl);
      const tokenizer = await AutoTokenizer.from_pretrained(tokenizerUrl, {
        local_files_only: true,
      });
      const tokenizerMs = performance.now() - tokStartMs;

      const rawDevMem =
        typeof navigator !== 'undefined' && 'deviceMemory' in navigator
          ? Number((navigator as unknown as { deviceMemory?: number }).deviceMemory)
          : undefined;
      const deviceMemoryGb =
        typeof rawDevMem === 'number' && Number.isFinite(rawDevMem) && rawDevMem > 0
          ? rawDevMem
          : undefined;

      const wgslEngine = new WgslQwen35Engine();
      const stats = await wgslEngine.load(null, {
        assetBaseUrl: this.assetBaseUrl,
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
      this.eosTokenIds = this.collectEosTokenIds(tokenizer);
      this.eosTokenIds.add(248044);
      this.eosTokenIds.add(248046);
      this.prefixCache = new PrefixCacheManager(wgslEngine, { maxTabSlots: 3 });

      await this.prewarmBaseSystemPrompt();

      const coldLoadMs = performance.now() - startMs;
      const memStats = wgslEngine.getMemoryStats();
      const baseInfo = this.prefixCache.getBasePromptInfo();
      this.log(
        'info',
        'offscreen',
        `WGSL Qwen3.5-4B ready in ${(coldLoadMs / 1000).toFixed(2)}s (tier=${stats.computeTier}, source=${stats.loadSource}, host=${this.executionHost}, stream=${stats.opfsReadMs.toFixed(0)}ms, compile+verify=${stats.pipelineCompileMs.toFixed(0)}ms, kv=int8 [${stats.allocatedKvCapacity}/${stats.maxSafeKvCapacity}], dp4a=${stats.usingDp4a}, baseSlot=${baseInfo.tokens}t, hz=${this.displayHz}).`,
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
    if (this.bgAbortController) {
      this.bgAbortController.abort();
      this.bgAbortController = null;
    }
    if (this.telemetry.backgroundPrefillState === 'prefilling') {
      this.updateTelemetry({ backgroundPrefillState: 'idle' });
    }
  }

  private normalizeMessagesForChatTemplate(messages: readonly ChatMessage[]): ChatMessage[] {
    return messages.map((msg, idx) =>
      idx > 0 && msg.role === 'system'
        ? { role: 'user', content: msg.content }
        : { role: msg.role, content: msg.content },
    );
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
        if (formatted.endsWith('<|im_end|>\n')) {
          formatted = formatted.slice(0, -'<|im_end|>\n'.length);
        } else if (formatted.endsWith('<|im_end|>')) {
          formatted = formatted.slice(0, -'<|im_end|>'.length);
        }
        return this.tokenizer.encode(formatted, { add_special_tokens: false });
      }
    }

    const marker = '__TINYGPT_PREFIX_MARKER__';
    const withMarker = this.formatChatMessages(
      [...messages, { role: 'user', content: marker }],
      { addGenerationPrompt: false },
    );
    const markerIdx = withMarker.lastIndexOf(`<|im_start|>user\n${marker}`);
    const prefixText = markerIdx >= 0 ? withMarker.slice(0, markerIdx) : withMarker;
    return this.tokenizer.encode(prefixText, { add_special_tokens: false });
  }

  private async prewarmBaseSystemPrompt(): Promise<void> {
    if (!this.wgslEngine || !this.tokenizer || !this.prefixCache) {
      return;
    }
    try {
      const baseMsg = buildBaseSystemMessage();
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
    options?: { isDraft?: boolean; url?: string; contentHash?: string },
  ): void {
    this.abortBackgroundPrefill();
    const mySeq = this.bgPrefillSeq;
    this.taskQueue = this.taskQueue
      .then(() => this.runBackgroundPrefill(mySeq, slotKey, messages, options))
      .catch(() => {
        if (this.telemetry.backgroundPrefillState === 'prefilling') {
          this.updateTelemetry({ backgroundPrefillState: 'idle' });
        }
      });
  }

  private async runBackgroundPrefill(
    seq: number,
    slotKey: string,
    messages: ChatMessage[],
    options?: { isDraft?: boolean; url?: string; contentHash?: string },
  ): Promise<void> {
    if (
      seq !== this.bgPrefillSeq ||
      !this.isEngineReady() ||
      !this.wgslEngine ||
      !this.tokenizer ||
      !this.prefixCache ||
      this.activeRequestId !== null ||
      this.telemetry.status !== 'ready'
    ) {
      return;
    }

    const targetTokenIds = this.encodePrefixMessages(messages, Boolean(options?.isDraft));
    if (targetTokenIds.length === 0 || targetTokenIds.length >= this.wgslEngine.getMaxSafeKvCapacity() - 64) {
      return;
    }

    const match = this.prefixCache.findBestMatch(targetTokenIds, this.cachedTokenIds);
    if (match.matchedTokens === targetTokenIds.length) {
      if (match.source === 'live') {
        await this.prefixCache.saveTabSlot(slotKey, targetTokenIds, messages, {
          url: options?.url,
          contentHash: options?.contentHash,
        });
      }
      this.updateTelemetry({
        backgroundPrefillState: 'warm',
        activePrefixSlot: match.slotId ?? slotKey,
      });
      return;
    }

    const controller = new AbortController();
    this.bgAbortController = controller;
    this.updateTelemetry({ backgroundPrefillState: 'prefilling' });

    try {
      if (match.source === 'slot' && match.slotId) {
        const restored = await this.prefixCache.restoreSlot(match.slotId);
        this.cachedTokenIds = [...restored.entry.tokenIds];
        this.cachedMessages = restored.entry.messages.map((m) => ({ ...m }));
        this.cachedTokenCount = restored.seqPos;
      } else if (match.source === 'none') {
        await this.wgslEngine.resetState();
        this.cachedTokenIds = [];
        this.cachedMessages = [];
        this.cachedTokenCount = 0;
      }

      if (seq !== this.bgPrefillSeq || controller.signal.aborted) {
        throw new Error('__BG_ABORTED__');
      }

      const deltaTokens = targetTokenIds.slice(match.matchedTokens);
      if (deltaTokens.length > 0) {
        await this.wgslEngine.prefill(deltaTokens, {
          chunkSize: Math.min(32, this.config.chunkSize || 32),
          sampling: { temperature: 0, topK: 1, topP: 1 },
          pacingMode: 'smooth-ui',
          abortSignal: controller.signal,
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
            this.cachedTokenCount = this.wgslEngine!.activeSeqPos;
          },
        });
      }

      if (seq !== this.bgPrefillSeq || controller.signal.aborted) {
        throw new Error('__BG_ABORTED__');
      }

      await this.prefixCache.saveTabSlot(slotKey, targetTokenIds, messages, {
        url: options?.url,
        contentHash: options?.contentHash,
      });
      this.cachedTokenIds = [...targetTokenIds];
      this.cachedMessages = messages.map((m) => ({ ...m }));
      this.cachedTokenCount = this.wgslEngine.activeSeqPos;
      this.cachedThinkingMode = this.config.enableThinking;
      this.refreshCacheTelemetry();
      this.updateTelemetry({
        backgroundPrefillState: 'warm',
        activePrefixSlot: slotKey,
        prefillChunkProgress: null,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === '__BG_ABORTED__' || msg === 'Generation aborted' || controller.signal.aborted) {
        this.cachedTokenIds = [];
        this.cachedMessages = [];
        this.cachedTokenCount = 0;
        if (this.prefixCache.hasBasePrompt()) {
          await this.prefixCache.resetToBasePrompt().catch(() => {});
        } else {
          await this.wgslEngine.resetState().catch(() => {});
        }
        this.updateTelemetry({
          backgroundPrefillState: 'idle',
          prefillChunkProgress: null,
        });
        return;
      }
      this.log('warn', 'offscreen', `Background speculative prefill stopped: ${msg}`);
      this.updateTelemetry({ backgroundPrefillState: 'idle', prefillChunkProgress: null });
    } finally {
      if (this.bgAbortController === controller) {
        this.bgAbortController = null;
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
  ): void {
    this.abortBackgroundPrefill();
    this.taskQueue = this.taskQueue
      .then(() => this.runGenerate(requestId, messages, overrideConfig, groundingText))
      .catch((err) => {
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
    if (!template.includes('<|im_start|>')) {
      return null;
    }
    let out = '<|im_end|>\n';
    for (const msg of suffixMessages) {
      out += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    }
    out += '<|im_start|>assistant\n<think>\n\n</think>\n\n';
    return out;
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
          this.updateTelemetry({
            prefillChunkProgress: isFinalChunk ? null : { done, total },
          });
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
    };
  }

  private async runGenerate(
    requestId: string,
    messages: ChatMessage[],
    overrideConfig?: Partial<EngineConfig>,
    groundingText?: string,
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

    this.activeRequestId = requestId;
    const runStartMs = performance.now();

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
      let reusedPrefix = false;
      let restoredSlotId: string | null = null;
      let lastSlotRestoreMs: number | null = null;
      const hasActiveState = this.wgslEngine !== null && this.wgslEngine.activeSeqPos > 0;

      const normMessages = this.normalizeMessagesForChatTemplate(messages);
      const normCached = this.normalizeMessagesForChatTemplate(this.cachedMessages);

      if (
        !this.config.enableThinking &&
        !this.cachedThinkingMode &&
        hasActiveState &&
        normCached.length > 0 &&
        this.cachedTokenIds.length === this.wgslEngine!.activeSeqPos &&
        isMessagePrefix(normCached, normMessages)
      ) {
        const suffixMessages = normMessages.slice(normCached.length);
        const incrementalText = this.buildIncrementalChatMlPrompt(suffixMessages);
        if (incrementalText !== null) {
          promptTokenIds = this.tokenizer.encode(incrementalText, { add_special_tokens: false });
          reusedPrefix = true;
          restoredSlotId = 'live';
        }
      }

      if (promptTokenIds === null) {
        const formattedPrompt = this.formatChatMessages(normMessages, {
          addGenerationPrompt: true,
        });
        const fullTokenIds = this.tokenizer.encode(formattedPrompt, {
          add_special_tokens: false,
        });

        const match = this.prefixCache
          ? this.prefixCache.findBestMatch(fullTokenIds, this.cachedTokenIds)
          : { source: 'none' as const, slotId: null, matchedTokens: 0 };

        if (match.source === 'live' && match.matchedTokens < fullTokenIds.length) {
          promptTokenIds = fullTokenIds.slice(match.matchedTokens);
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
          this.cachedTokenCount = restored.seqPos;
          promptTokenIds = fullTokenIds.slice(match.matchedTokens);
          reusedPrefix = true;
          restoredSlotId = match.slotId;
          lastSlotRestoreMs = restored.restoreMs;
        } else {
          await this.resetContext(false);
          promptTokenIds = fullTokenIds;
          reusedPrefix = false;
        }
      }

      this.updateTelemetry({
        prefixReused: reusedPrefix,
        activePrefixSlot: restoredSlotId,
        ...(lastSlotRestoreMs !== null ? { lastSlotRestoreMs } : {}),
      });
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
      const { nextToken: firstToken, prefillMs } = await this.prefillAndPickFirstToken(
        promptTokenIds,
        this.config.chunkSize,
        sampleOpts,
        requestId,
      );

      const prefillTokPerSec = promptTokenIds.length / (prefillMs / 1000);
      const ttftMs = performance.now() - runStartMs;
      this.updateTelemetry({ ttftMs, prefillTokPerSec });

      let nextToken = firstToken;

      const generatedIds: number[] = [];
      const sessionTokenIds = reusedPrefix
        ? [...this.cachedTokenIds, ...promptTokenIds]
        : [...promptTokenIds];
      const groundingTokens =
        groundingText && groundingText.trim().length > 0
          ? this.tokenizer.encode(groundingText, { add_special_tokens: false })
          : [];
      const speculator = new SuffixSpeculator([...groundingTokens, ...sessionTokenIds]);
      let specDrafted = 0;
      let specAccepted = 0;
      let specSteps = 0;

      let emittedText = '';
      let finishReason: 'stop' | 'length' | 'abort' = 'length';
      const decodeStartMs = performance.now();

      const buildSpecStats = () =>
        specSteps > 0
          ? {
              drafted: specDrafted,
              accepted: specAccepted,
              steps: specSteps,
              acceptanceRatePct: specDrafted > 0 ? (specAccepted / specDrafted) * 100 : 0,
            }
          : null;

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
        speculator.pushToken(tok);

        const decodedFull = this.tokenizer!.decode(generatedIds, {
          skip_special_tokens: true,
        });
        // Hold back trailing replacement char U+FFFD if a multi-byte UTF-8 sequence spans tokens.
        const safeEnd = decodedFull.endsWith('\uFFFD') ? decodedFull.length - 1 : decodedFull.length;
        if (safeEnd > emittedText.length) {
          const delta = decodedFull.slice(emittedText.length, safeEnd);
          emittedText = decodedFull.slice(0, safeEnd);
          this.callbacks.onTokenDelta(requestId, delta, generatedIds.length);
        }

        if (generatedIds.length >= this.config.maxNewTokens) {
          finishReason = 'length';
          return true;
        }
        return false;
      };

      let iter = 0;
      while (generatedIds.length < this.config.maxNewTokens) {
        if (emitTokenAndCheckDone(nextToken)) {
          break;
        }

        if (!this.config.fastDecodePath || iter % 4 === 0) {
          const decodeElapsedSec = Math.max(0.001, (performance.now() - decodeStartMs) / 1000);
          this.updateTelemetry({
            tokensPerSec: generatedIds.length / decodeElapsedSec,
            speculativeStats: buildSpecStats(),
          });
        }

        if (this.config.yieldMode !== 'none') {
          await yieldFast(this.config.yieldMode);
        } else if (iter % 8 === 7) {
          await yieldFast('message-channel');
        }
        if (this.abortedRequests.has(requestId)) {
          finishReason = 'abort';
          break;
        }

        const remainingToMax = this.config.maxNewTokens - generatedIds.length;
        const canSpeculate =
          this.config.enableSpeculative &&
          sampleOpts.temperature <= 0 &&
          this.wgslEngine !== null &&
          remainingToMax > 1;

        if (canSpeculate) {
          const maxDraft = Math.min(4, remainingToMax - 1);
          const rawDraft = speculator.propose({ maxDraft });
          const cleanDraft: number[] = [];
          for (const dt of rawDraft) {
            if (this.eosTokenIds.has(dt)) break;
            cleanDraft.push(dt);
          }

          if (
            cleanDraft.length > 0 &&
            this.wgslEngine!.activeSeqPos + 1 + cleanDraft.length <= this.wgslEngine!.maxContextTokens
          ) {
            const specRes = await this.wgslEngine!.decodeSpeculativeStep(
              nextToken,
              cleanDraft,
              sampleOpts,
              this.config.gpuPacingMode,
            );
            this.cachedTokenCount = this.wgslEngine!.activeSeqPos;
            specDrafted += specRes.draftedCount;
            specAccepted += specRes.acceptedCount;
            specSteps += 1;
            if (!this.config.fastDecodePath || iter % 4 === 0) {
              this.refreshCacheTelemetry();
            }

            let stoppedMidBatch = false;
            for (let i = 0; i < specRes.acceptedCount; i++) {
              if (emitTokenAndCheckDone(specRes.tokens[i])) {
                stoppedMidBatch = true;
                break;
              }
            }
            if (stoppedMidBatch) {
              break;
            }
            nextToken = specRes.tokens[specRes.acceptedCount];
            iter++;
            continue;
          }
        }

        const stepRes = await this.decodeNextToken(nextToken, sampleOpts, iter);
        nextToken = stepRes.nextToken;
        iter++;
      }

      if (generatedIds.length > 0) {
        const totalDecodeSec = Math.max(0.001, (performance.now() - decodeStartMs) / 1000);
        this.updateTelemetry({
          tokensPerSec: generatedIds.length / totalDecodeSec,
          speculativeStats: buildSpecStats(),
        });
        this.refreshCacheTelemetry();
      }

      if (finishReason !== 'abort') {
        const finalDecoded = this.tokenizer.decode(generatedIds, {
          skip_special_tokens: true,
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

      this.cachedMessages = [...messages, { role: 'assistant', content: emittedText }];
      this.cachedThinkingMode = this.config.enableThinking;
      this.cachedTokenIds = sessionTokenIds;

      this.updateTelemetry({ status: 'ready', prefillChunkProgress: null });
      this.callbacks.onGenerationComplete(requestId, finishReason);
    } catch (err) {
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
}
