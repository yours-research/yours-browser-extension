import type { WorkerPortTransport } from '@yours/harness/adapters';
import {
  DEFAULT_ENGINE_CONFIG,
  ENGINE_PORT_NAME,
  INITIAL_TELEMETRY_STATE,
  type BenchmarkSuiteResult,
  type ChatMessage,
  type ClientToEngineMessage,
  type EngineConfig,
  type EngineToClientMessage,
  type HardwareProfileSnapshot,
  type LogEntry,
  type ModelCatalogChoice,
  type ModelDownloadProgressInfo,
  type NiahResult,
  type OpfsSpikeResult,
  type OpfsStatusInfo,
  type TelemetryState,
  type WebGpuProbeInfo,
  type WgslBenchResult,
  type WgslOpfsLoadResult,
} from '../shared/protocol.ts';
import type { ExtractedPageContent, ToolExecutionResult } from '../tools/types.ts';

export interface EngineClientSnapshot {
  connected: boolean;
  telemetry: TelemetryState;
  config: EngineConfig;
  probeInfo: WebGpuProbeInfo | null;
  opfsStatus: OpfsStatusInfo | null;
  hardwareProfile: HardwareProfileSnapshot | null;
  catalogModels: ModelCatalogChoice[];
  downloadProgress: ModelDownloadProgressInfo | null;
  downloadError: string | null;
  spikeProgress: { message: string; pct: number } | null;
  spikeResult: OpfsSpikeResult | null;
  benchmarkProgress: string | null;
  lastBenchmark: BenchmarkSuiteResult | null;
  niahProgress: string | null;
  lastNiah: NiahResult | null;
  wgslProgress: string | null;
  lastWgslBench: WgslBenchResult | null;
  lastWgslOpfs: WgslOpfsLoadResult | null;
  logs: LogEntry[];
}

export interface StreamCallbacks {
  onDelta: (delta: string, tokenCount: number) => void;
  onComplete: (finishReason: 'stop' | 'length' | 'abort') => void;
  onError: (error: string) => void;
}

export interface EnginePortLike {
  postMessage: (msg: ClientToEngineMessage) => void;
  disconnect: () => void;
  onMessage: { addListener: (cb: (msg: EngineToClientMessage) => void) => void };
  onDisconnect: { addListener: (cb: () => void) => void };
}

export interface ChromeTransportLike {
  ensureOffscreen: () => Promise<void>;
  connect: () => EnginePortLike;
}

function defaultChromeTransport(): ChromeTransportLike {
  return {
    async ensureOffscreen() {
      const response = (await chrome.runtime.sendMessage({ type: 'ENSURE_OFFSCREEN' })) as
        | { ok: true }
        | { ok: false; error: string }
        | undefined;
      if (response && !response.ok) {
        throw new Error(response.error);
      }
    },
    connect() {
      return chrome.runtime.connect({ name: ENGINE_PORT_NAME }) as unknown as EnginePortLike;
    },
  };
}

export class OffscreenEngineClient {
  private readonly transport: ChromeTransportLike;
  private port: EnginePortLike | null = null;
  private connectPromise: Promise<EnginePortLike> | null = null;
  private readonly listeners = new Set<() => void>();
  private readonly rawMessageListeners = new Set<(message: unknown) => void>();
  private readonly activeStreams = new Map<string, StreamCallbacks>();
  private snapshot: EngineClientSnapshot = {
    connected: false,
    telemetry: { ...INITIAL_TELEMETRY_STATE },
    config: { ...DEFAULT_ENGINE_CONFIG },
    probeInfo: null,
    opfsStatus: null,
    hardwareProfile: null,
    catalogModels: [],
    downloadProgress: null,
    downloadError: null,
    spikeProgress: null,
    spikeResult: null,
    benchmarkProgress: null,
    lastBenchmark: null,
    niahProgress: null,
    lastNiah: null,
    wgslProgress: null,
    lastWgslBench: null,
    lastWgslOpfs: null,
    logs: [],
  };

  constructor(transport: ChromeTransportLike = defaultChromeTransport()) {
    this.transport = transport;
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): EngineClientSnapshot => {
    return this.snapshot;
  };

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }

  private updateSnapshot(patch: Partial<EngineClientSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    this.emit();
  }

  private appendLog(entry: LogEntry): void {
    const nextLogs = [...this.snapshot.logs, entry];
    if (nextLogs.length > 200) {
      nextLogs.splice(0, nextLogs.length - 200);
    }
    this.updateSnapshot({ logs: nextLogs });
  }

  async connect(): Promise<EnginePortLike> {
    if (this.port) {
      return this.port;
    }
    if (this.connectPromise) {
      return this.connectPromise;
    }

    this.connectPromise = (async () => {
      await this.transport.ensureOffscreen();
      const port = this.transport.connect();
      port.onMessage.addListener(this.handleEngineMessage);
      port.onDisconnect.addListener(() => {
        if (this.port === port) {
          this.port = null;
        }
        this.updateSnapshot({ connected: false });
        for (const [reqId, callbacks] of this.activeStreams.entries()) {
          callbacks.onError('Disconnected from Offscreen WebGPU engine.');
          this.activeStreams.delete(reqId);
        }
      });
      this.port = port;
      this.updateSnapshot({ connected: true });
      port.postMessage({ type: 'GET_STATE' });
      return port;
    })().finally(() => {
      this.connectPromise = null;
    });

    return this.connectPromise;
  }

  private handleEngineMessage = (msg: EngineToClientMessage): void => {
    for (const rawListener of this.rawMessageListeners) {
      rawListener(msg);
    }

    switch (msg.type) {
      case 'STATE_SNAPSHOT':
        this.updateSnapshot({
          connected: true,
          telemetry: msg.telemetry,
          config: msg.config,
          probeInfo: msg.probeInfo,
          opfsStatus: msg.opfsStatus,
          lastBenchmark: msg.lastBenchmark,
          lastNiah: msg.lastNiah,
          lastWgslBench: msg.lastWgslBench,
          lastWgslOpfs: msg.lastWgslOpfs,
          logs: msg.logs,
        });
        break;
      case 'TELEMETRY_UPDATE':
        this.updateSnapshot({ telemetry: msg.telemetry });
        break;
      case 'CONFIG_UPDATE':
        this.updateSnapshot({ config: msg.config });
        break;
      case 'HARDWARE_PROFILE_UPDATE':
        this.updateSnapshot({
          hardwareProfile: msg.hardwareProfile,
          catalogModels: msg.catalog,
        });
        break;
      case 'MODEL_DOWNLOAD_PROGRESS':
        this.updateSnapshot({
          downloadProgress: msg.progress,
          downloadError: null,
        });
        break;
      case 'MODEL_DOWNLOAD_COMPLETE':
        this.updateSnapshot({
          downloadProgress: null,
          downloadError: null,
          catalogModels: msg.catalog,
        });
        break;
      case 'MODEL_DOWNLOAD_ERROR':
        this.updateSnapshot({
          downloadProgress: null,
          downloadError: msg.error,
        });
        break;
      case 'WEBGPU_PROBE_RESULT':
        this.updateSnapshot({ probeInfo: msg.probeInfo });
        break;
      case 'OPFS_STATUS':
        this.updateSnapshot({ opfsStatus: msg.opfsStatus });
        break;
      case 'OPFS_SPIKE_PROGRESS':
        this.updateSnapshot({ spikeProgress: { message: msg.message, pct: msg.pct } });
        break;
      case 'OPFS_SPIKE_RESULT':
        this.updateSnapshot({ spikeProgress: null, spikeResult: msg.result });
        break;
      case 'BENCHMARK_PROGRESS':
        this.updateSnapshot({ benchmarkProgress: msg.message });
        break;
      case 'BENCHMARK_RESULT':
        this.updateSnapshot({ benchmarkProgress: null, lastBenchmark: msg.result });
        break;
      case 'NIAH_PROGRESS':
        this.updateSnapshot({ niahProgress: msg.message });
        break;
      case 'NIAH_RESULT':
        this.updateSnapshot({ niahProgress: null, lastNiah: msg.result });
        break;
      case 'WGSL_BENCH_PROGRESS':
        this.updateSnapshot({ wgslProgress: msg.message });
        break;
      case 'WGSL_BENCH_RESULT':
        this.updateSnapshot({ wgslProgress: null, lastWgslBench: msg.result });
        break;
      case 'WGSL_OPFS_BENCH_RESULT':
        this.updateSnapshot({ lastWgslOpfs: msg.result });
        break;
      case 'LOG':
        this.appendLog(msg.entry);
        break;
      case 'TOKEN_DELTA': {
        const stream = this.activeStreams.get(msg.requestId);
        stream?.onDelta(msg.delta, msg.tokenCount);
        break;
      }
      case 'GENERATION_COMPLETE': {
        const stream = this.activeStreams.get(msg.requestId);
        if (stream) {
          this.activeStreams.delete(msg.requestId);
          stream.onComplete(msg.finishReason);
        }
        break;
      }
      case 'GENERATION_ERROR': {
        const stream = this.activeStreams.get(msg.requestId);
        if (stream) {
          this.activeStreams.delete(msg.requestId);
          stream.onError(msg.error);
        }
        break;
      }
    }
  };

  async send(msg: ClientToEngineMessage): Promise<void> {
    const port = await this.connect();
    port.postMessage(msg);
  }

  createWgslPortTransport(): WorkerPortTransport {
    return {
      postMessage: (message: unknown) => {
        const msg = message as ClientToEngineMessage;
        if (this.port) {
          this.port.postMessage(msg);
          return;
        }
        void this.send(msg).catch((err) => {
          const errMsg = err instanceof Error ? err.message : String(err);
          const reqId = (msg as { requestId?: string }).requestId;
          if (reqId && msg.type === 'GENERATE') {
            this.handleEngineMessage({
              type: 'GENERATION_ERROR',
              requestId: reqId,
              error: errMsg,
            });
          } else if (reqId && msg.type === 'PREFILL_BACKGROUND') {
            this.handleEngineMessage({
              type: 'PREFILL_ERROR',
              requestId: reqId,
              slotKey: msg.slotKey,
              error: errMsg,
            });
          } else if (reqId && msg.type === 'EVICT_SLOT') {
            this.handleEngineMessage({
              type: 'EVICT_COMPLETE',
              requestId: reqId,
              slotKey: msg.slotKey,
            });
          }
        });
      },
      addMessageListener: (listener: (message: unknown) => void) => {
        this.rawMessageListeners.add(listener);
        return () => {
          this.rawMessageListeners.delete(listener);
        };
      },
    };
  }

  probeHardwareCatalog(): void {
    void this.send({ type: 'PROBE_HARDWARE_CATALOG' }).catch(() => {});
  }

  downloadVerifiedModel(modelId: string, manifestUrl?: string): void {
    void this.send({ type: 'DOWNLOAD_VERIFIED_MODEL', modelId, manifestUrl }).catch(() => {});
  }

  cancelModelDownload(modelId?: string): void {
    void this.send({ type: 'CANCEL_MODEL_DOWNLOAD', modelId }).catch(() => {});
  }

  updateConfig(config: Partial<EngineConfig>): Promise<void> {
    return this.send({ type: 'UPDATE_CONFIG', config });
  }

  loadModel(config?: Partial<EngineConfig>): Promise<void> {
    return this.send({ type: 'LOAD_MODEL', ...(config ? { config } : {}) });
  }

  unloadModel(): Promise<void> {
    return this.send({ type: 'UNLOAD_MODEL' });
  }

  async startGeneration(
    requestId: string,
    messages: ChatMessage[],
    callbacks: StreamCallbacks,
    config?: Partial<EngineConfig>,
    groundingText?: string,
    stopOnToolCall?: boolean,
  ): Promise<void> {
    this.activeStreams.set(requestId, callbacks);
    try {
      const port = await this.connect();
      port.postMessage({
        type: 'GENERATE',
        requestId,
        messages,
        config,
        ...(groundingText ? { groundingText } : {}),
        ...(stopOnToolCall ? { stopOnToolCall: true } : {}),
      });
    } catch (err) {
      this.activeStreams.delete(requestId);
      callbacks.onError(err instanceof Error ? err.message : String(err));
    }
  }

  setDisplayHz(hz: number): void {
    void this.send({ type: 'SET_DISPLAY_HZ', hz }).catch(() => {});
  }

  prefillBackground(
    slotKey: string,
    messages: ChatMessage[],
    meta?: { isDraft?: boolean; url?: string; contentHash?: string },
  ): void {
    if (!this.port) {
      return;
    }
    this.port.postMessage({
      type: 'PREFILL_BACKGROUND',
      slotKey,
      messages,
      isDraft: meta?.isDraft,
      url: meta?.url,
      contentHash: meta?.contentHash,
    });
  }

  abortBackgroundPrefill(): void {
    if (!this.port) {
      return;
    }
    this.port.postMessage({ type: 'ABORT_BACKGROUND_PREFILL' });
  }

  resetContext(restoreBasePrompt = false): void {
    void this.send({ type: 'RESET_CONTEXT', restoreBasePrompt }).catch(() => {});
  }

  async execTool(
    call: {
      name: string;
      arguments?: Record<string, unknown>;
    },
    options?: { skipSpotlight?: boolean },
  ): Promise<ToolExecutionResult> {
    if (typeof chrome === 'undefined' || !chrome.runtime?.sendMessage) {
      return {
        ok: false,
        toolName: call.name as ToolExecutionResult['toolName'],
        error: 'Chrome extension runtime messaging is unavailable.',
      };
    }
    return (await chrome.runtime.sendMessage({
      type: 'EXEC_TOOL',
      call,
      ...(options?.skipSpotlight ? { skipSpotlight: true } : {}),
    })) as ToolExecutionResult;
  }

  async checkResearchPermission(): Promise<boolean> {
    if (typeof chrome !== 'undefined' && chrome.permissions?.contains) {
      try {
        return await chrome.permissions.contains({ origins: ['https://*/*'] });
      } catch {
        // Fallback to service worker
      }
    }
    if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
      const res = (await chrome.runtime.sendMessage({ type: 'CHECK_RESEARCH_PERMISSION' })) as
        | { ok: boolean; granted?: boolean }
        | undefined;
      return Boolean(res?.granted);
    }
    return false;
  }

  async requestResearchPermission(): Promise<boolean> {
    if (typeof chrome !== 'undefined' && chrome.permissions?.request) {
      try {
        return await chrome.permissions.request({ origins: ['https://*/*'] });
      } catch {
        return false;
      }
    }
    if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
      const res = (await chrome.runtime.sendMessage({ type: 'REQUEST_RESEARCH_PERMISSION' })) as
        | { ok: boolean; granted?: boolean }
        | undefined;
      return Boolean(res?.granted);
    }
    return false;
  }

  async extractActiveTabPreview(): Promise<ExtractedPageContent | null> {
    if (typeof chrome === 'undefined' || !chrome.runtime?.sendMessage) {
      return null;
    }
    try {
      const res = (await chrome.runtime.sendMessage({ type: 'EXTRACT_ACTIVE_TAB_PREVIEW' })) as
        | { ok: boolean; page?: ExtractedPageContent }
        | undefined;
      return res?.ok && res.page ? res.page : null;
    } catch {
      return null;
    }
  }

  abortGeneration(requestId: string): void {
    const stream = this.activeStreams.get(requestId);
    if (this.port) {
      this.port.postMessage({ type: 'ABORT', requestId });
    }
    if (stream) {
      this.activeStreams.delete(requestId);
      stream.onComplete('abort');
    }
  }

  disconnect(): void {
    if (this.port) {
      this.port.disconnect();
      this.port = null;
    }
    this.updateSnapshot({ connected: false });
  }
}
