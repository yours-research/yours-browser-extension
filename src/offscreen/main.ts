import {
  DEFAULT_ENGINE_CONFIG,
  ENGINE_PORT_NAME,
  INITIAL_TELEMETRY_STATE,
  type BenchmarkSuiteResult,
  type ClientToEngineMessage,
  type EngineConfig,
  type EngineToClientMessage,
  type ExecutionHost,
  type LogEntry,
  type NiahResult,
  type OpfsStatusInfo,
  type TelemetryState,
  type WebGpuProbeInfo,
  type WgslBenchResult,
  type WgslOpfsLoadResult,
} from '../shared/protocol.ts';
import type { WorkerInboundMessage } from './engineWorker.ts';
import { InferenceEngine, type EngineCallbacks } from './inferenceEngine.ts';
import { installWebGpuLimitPatch } from './webgpuLimits.ts';
import { WgslEngineRunner } from './wgslEngine.ts';

// Install WebGPU elevated-limits wrapper immediately on offscreen startup.
installWebGpuLimitPatch();
if (typeof navigator !== 'undefined') {
  void navigator.storage?.persist?.().catch(() => {});
}

interface MirrorState {
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

const mirrorState: MirrorState = {
  telemetry: { ...INITIAL_TELEMETRY_STATE },
  config: { ...DEFAULT_ENGINE_CONFIG },
  probeInfo: null,
  opfsStatus: null,
  lastBenchmark: null,
  lastNiah: null,
  lastWgslBench: null,
  lastWgslOpfs: null,
  logs: [],
};

const ports = new Set<chrome.runtime.Port>();

let activeHost: ExecutionHost = 'worker';
let hostGeneration = 0;
let activeWorker: Worker | null = null;
let activeLocalEngine: InferenceEngine | null = null;
let activeLocalRunner: WgslEngineRunner | null = null;
let messageQueue: Promise<void> = Promise.resolve();

function getAssetBaseUrl(): string {
  if (typeof chrome !== 'undefined' && typeof chrome.runtime?.getURL === 'function') {
    return chrome.runtime.getURL('');
  }
  if (typeof self !== 'undefined' && self.location?.origin && self.location.origin !== 'null') {
    return `${self.location.origin}/`;
  }
  return '/';
}

function appendMirrorLog(entry: LogEntry): void {
  mirrorState.logs.push(entry);
  if (mirrorState.logs.length > 200) {
    mirrorState.logs.splice(0, mirrorState.logs.length - 200);
  }
}

function broadcast(message: EngineToClientMessage): void {
  for (const port of ports) {
    try {
      port.postMessage(message);
    } catch {
      ports.delete(port);
    }
  }
}

function buildStateSnapshot(): EngineToClientMessage {
  return {
    type: 'STATE_SNAPSHOT',
    telemetry: mirrorState.telemetry,
    config: mirrorState.config,
    probeInfo: mirrorState.probeInfo,
    opfsStatus: mirrorState.opfsStatus,
    lastBenchmark: mirrorState.lastBenchmark,
    lastNiah: mirrorState.lastNiah,
    lastWgslBench: mirrorState.lastWgslBench,
    lastWgslOpfs: mirrorState.lastWgslOpfs,
    logs: mirrorState.logs,
  };
}

function recordErrorAndBroadcast(errorMsg: string): void {
  const entry: LogEntry = {
    ts: new Date().toISOString(),
    level: 'error',
    source: 'offscreen',
    message: errorMsg,
  };
  appendMirrorLog(entry);
  broadcast({ type: 'LOG', entry });
  mirrorState.telemetry = {
    ...mirrorState.telemetry,
    status: 'error',
    lastError: errorMsg,
    prefillChunkProgress: null,
  };
  broadcast({ type: 'TELEMETRY_UPDATE', telemetry: mirrorState.telemetry });
}

function handleHostMessage(genId: number, msg: EngineToClientMessage): void {
  if (genId !== hostGeneration) {
    return;
  }

  switch (msg.type) {
    case 'STATE_SNAPSHOT':
      mirrorState.telemetry = msg.telemetry;
      mirrorState.config = msg.config;
      if (msg.probeInfo) mirrorState.probeInfo = msg.probeInfo;
      if (msg.opfsStatus) mirrorState.opfsStatus = msg.opfsStatus;
      if (msg.lastBenchmark) mirrorState.lastBenchmark = msg.lastBenchmark;
      if (msg.lastNiah) mirrorState.lastNiah = msg.lastNiah;
      if (msg.lastWgslBench) mirrorState.lastWgslBench = msg.lastWgslBench;
      if (msg.lastWgslOpfs) mirrorState.lastWgslOpfs = msg.lastWgslOpfs;
      if (mirrorState.logs.length === 0 && msg.logs.length > 0) {
        mirrorState.logs = [...msg.logs];
      }
      return;
    case 'TELEMETRY_UPDATE':
      mirrorState.telemetry = msg.telemetry;
      break;
    case 'CONFIG_UPDATE':
      mirrorState.config = msg.config;
      break;
    case 'WEBGPU_PROBE_RESULT':
      mirrorState.probeInfo = msg.probeInfo;
      break;
    case 'OPFS_STATUS':
      mirrorState.opfsStatus = msg.opfsStatus;
      break;
    case 'BENCHMARK_RESULT':
      mirrorState.lastBenchmark = msg.result;
      break;
    case 'NIAH_RESULT':
      mirrorState.lastNiah = msg.result;
      break;
    case 'WGSL_BENCH_RESULT':
      mirrorState.lastWgslBench = msg.result;
      break;
    case 'WGSL_OPFS_BENCH_RESULT':
      mirrorState.lastWgslOpfs = msg.result;
      break;
    case 'LOG':
      appendMirrorLog(msg.entry);
      break;
    default:
      break;
  }

  broadcast(msg);
}

function createLocalCallbacks(genId: number): EngineCallbacks {
  return {
    onTelemetry: (telemetry) => handleHostMessage(genId, { type: 'TELEMETRY_UPDATE', telemetry }),
    onConfig: (config) => handleHostMessage(genId, { type: 'CONFIG_UPDATE', config }),
    onLog: (entry) => handleHostMessage(genId, { type: 'LOG', entry }),
    onProbeResult: (probeInfo) =>
      handleHostMessage(genId, { type: 'WEBGPU_PROBE_RESULT', probeInfo }),
    onOpfsStatus: (opfsStatus) => handleHostMessage(genId, { type: 'OPFS_STATUS', opfsStatus }),
    onSpikeProgress: (message, pct) =>
      handleHostMessage(genId, { type: 'OPFS_SPIKE_PROGRESS', message, pct }),
    onSpikeResult: (result) => handleHostMessage(genId, { type: 'OPFS_SPIKE_RESULT', result }),
    onBenchmarkProgress: (message, point) =>
      handleHostMessage(genId, { type: 'BENCHMARK_PROGRESS', message, point }),
    onBenchmarkResult: (result) =>
      handleHostMessage(genId, { type: 'BENCHMARK_RESULT', result }),
    onNiahProgress: (message) => handleHostMessage(genId, { type: 'NIAH_PROGRESS', message }),
    onNiahResult: (result) => handleHostMessage(genId, { type: 'NIAH_RESULT', result }),
    onTokenDelta: (requestId, delta, tokenCount) =>
      handleHostMessage(genId, { type: 'TOKEN_DELTA', requestId, delta, tokenCount }),
    onGenerationComplete: (requestId, finishReason) =>
      handleHostMessage(genId, { type: 'GENERATION_COMPLETE', requestId, finishReason }),
    onGenerationError: (requestId, error) =>
      handleHostMessage(genId, { type: 'GENERATION_ERROR', requestId, error }),
  };
}

function bootWorkerHost(genId: number, syncConfig: boolean): void {
  const worker = new Worker(new URL('./engineWorker.ts', import.meta.url), { type: 'module' });
  activeWorker = worker;

  worker.onmessage = (event: MessageEvent<EngineToClientMessage>) => {
    if (event.data && typeof event.data === 'object') {
      handleHostMessage(genId, event.data);
    }
  };

  worker.onerror = (event: ErrorEvent) => {
    if (genId !== hostGeneration) {
      return;
    }
    const detail =
      event.message ||
      (event.error instanceof Error ? event.error.message : 'Unknown dedicated worker error.');
    recordErrorAndBroadcast(`Engine worker error: ${detail}`);
  };

  worker.onmessageerror = () => {
    if (genId !== hostGeneration) {
      return;
    }
    recordErrorAndBroadcast('Engine worker message deserialization failed.');
  };

  const initMsg: WorkerInboundMessage = {
    type: 'WORKER_INIT',
    assetBaseUrl: getAssetBaseUrl(),
  };
  worker.postMessage(initMsg);

  if (syncConfig) {
    worker.postMessage({
      type: 'UPDATE_CONFIG',
      config: mirrorState.config,
    } satisfies WorkerInboundMessage);
  }
}

async function bootLocalHost(genId: number): Promise<void> {
  const assetBaseUrl = getAssetBaseUrl();
  const engine = new InferenceEngine(createLocalCallbacks(genId), {
    assetBaseUrl,
    executionHost: 'offscreen-main',
  });
  const runner = new WgslEngineRunner(assetBaseUrl);
  activeLocalEngine = engine;
  activeLocalRunner = runner;

  await engine.updateConfig(mirrorState.config);
  void engine.runProbeWebGpu().catch(() => {});
  void engine.refreshOpfsStatus().catch(() => {});
}

async function switchHost(targetHost: ExecutionHost): Promise<void> {
  if (targetHost === activeHost && (activeWorker || activeLocalEngine)) {
    return;
  }

  hostGeneration += 1;
  const genId = hostGeneration;

  if (activeWorker) {
    activeWorker.onmessage = null;
    activeWorker.onerror = null;
    activeWorker.onmessageerror = null;
    activeWorker.terminate();
    activeWorker = null;
  }

  if (activeLocalEngine) {
    const prevEngine = activeLocalEngine;
    activeLocalEngine = null;
    activeLocalRunner = null;
    prevEngine.abort('');
    await prevEngine.unloadModel().catch(() => {});
  }

  if (genId !== hostGeneration) {
    return;
  }

  activeHost = targetHost;
  mirrorState.config = {
    ...mirrorState.config,
    backend: 'wgsl',
    wgslKvMode: 'int8',
    modelId: DEFAULT_ENGINE_CONFIG.modelId,
    dtype: 'q4f16',
    executionHost: targetHost,
  };
  mirrorState.telemetry = {
    ...INITIAL_TELEMETRY_STATE,
    modelId: mirrorState.config.modelId,
    dtype: mirrorState.config.dtype,
    executionHost: targetHost,
  };
  broadcast({ type: 'CONFIG_UPDATE', config: mirrorState.config });
  broadcast({ type: 'TELEMETRY_UPDATE', telemetry: mirrorState.telemetry });

  const switchLog: LogEntry = {
    ts: new Date().toISOString(),
    level: 'info',
    source: 'offscreen',
    message: `Switched inference execution host to '${targetHost}'.`,
  };
  appendMirrorLog(switchLog);
  broadcast({ type: 'LOG', entry: switchLog });

  if (targetHost === 'worker') {
    bootWorkerHost(genId, true);
  } else {
    await bootLocalHost(genId);
  }
}

function applyConfigPatchToMirror(patch: Partial<EngineConfig>): void {
  const nextMaxContext =
    typeof patch.wgslMaxContext === 'number' &&
    Number.isFinite(patch.wgslMaxContext) &&
    patch.wgslMaxContext > 0
      ? Math.min(131072, Math.round(patch.wgslMaxContext))
      : mirrorState.config.wgslMaxContext;

  mirrorState.config = {
    ...mirrorState.config,
    ...patch,
    backend: 'wgsl',
    wgslKvMode: 'int8',
    wgslMaxContext: nextMaxContext,
    modelId: DEFAULT_ENGINE_CONFIG.modelId,
    dtype: 'q4f16',
  };
}

function extractConfigPatch(msg: ClientToEngineMessage): Partial<EngineConfig> | undefined {
  switch (msg.type) {
    case 'UPDATE_CONFIG':
      return msg.config;
    case 'LOAD_MODEL':
    case 'GENERATE':
      return msg.config;
    default:
      return undefined;
  }
}

function dispatchToLocalHost(genId: number, msg: ClientToEngineMessage): void {
  const engine = activeLocalEngine;
  const runner = activeLocalRunner;
  if (!engine || !runner) {
    throw new Error('Local offscreen-main engine is not initialized.');
  }

  switch (msg.type) {
    case 'GET_STATE':
      break;
    case 'UPDATE_CONFIG':
      void engine.updateConfig(msg.config);
      break;
    case 'PROBE_WEBGPU':
      void engine.runProbeWebGpu();
      break;
    case 'CHECK_OPFS':
      void engine.refreshOpfsStatus();
      break;
    case 'RUN_OPFS_SPIKE':
      void engine.runOpfsSpike(msg.totalMiB, msg.shardMiB);
      break;
    case 'VERIFY_OPFS_SPIKE':
      void engine.verifyOpfsSpike();
      break;
    case 'CLEAR_OPFS_SPIKE':
      void engine.clearOpfsSpike();
      break;
    case 'CLEAR_MODEL_CACHE':
      void engine.clearModelCache();
      break;
    case 'LOAD_MODEL':
      void engine.loadModel(msg.config).catch(() => {
        // Errors are already surfaced via telemetry & logs.
      });
      break;
    case 'UNLOAD_MODEL':
      void engine.unloadModel();
      break;
    case 'RESET_CONTEXT':
      void engine.resetContext(Boolean(msg.restoreBasePrompt));
      break;
    case 'SNAPSHOT_CONTEXT':
      void engine.snapshotContext();
      break;
    case 'RESTORE_CONTEXT':
      void engine.restoreContext();
      break;
    case 'SET_DISPLAY_HZ':
      engine.setDisplayHz(msg.hz);
      break;
    case 'PREFILL_BACKGROUND':
      engine.enqueueBackgroundPrefill(msg.slotKey, msg.messages, {
        isDraft: msg.isDraft,
        url: msg.url,
        contentHash: msg.contentHash,
      });
      break;
    case 'ABORT_BACKGROUND_PREFILL':
      engine.abortBackgroundPrefill();
      break;
    case 'GENERATE':
      engine.enqueueGenerate(msg.requestId, msg.messages, msg.config, msg.groundingText);
      break;
    case 'ABORT':
      engine.abort(msg.requestId);
      break;
    case 'RUN_BENCHMARK_SUITE':
      engine.enqueueBenchmarkSuite(msg.contextLengths, msg.decodeTokens, msg.chunkSize);
      break;
    case 'RUN_NIAH':
      engine.enqueueNeedleInHaystack(msg.targetTokens, msg.depthRatio, msg.chunkSize);
      break;
    case 'RUN_WGSL_BENCH':
      void runner
        .runFullModelDecodeBench(msg.config, (progressMsg) =>
          handleHostMessage(genId, { type: 'WGSL_BENCH_PROGRESS', message: progressMsg }),
        )
        .then((result) => handleHostMessage(genId, { type: 'WGSL_BENCH_RESULT', result }))
        .catch((err) => {
          const errMsg = err instanceof Error ? err.message : String(err);
          const result: WgslBenchResult = {
            startedAt: new Date().toISOString(),
            finishedAt: new Date().toISOString(),
            gemvVariant: msg.config.gemvVariant,
            kvMode: msg.config.kvMode,
            gpuBandwidthGBs: msg.config.gpuBandwidthGBs ?? 936,
            hasShaderF16: false,
            hasSubgroups: false,
            maxBufferSize: 0,
            allocatedWeightMb: 0,
            allocatedKvMb: 0,
            allocatedDeltaNetMb: 0,
            pipelineCompileMs: 0,
            correctness: [],
            points: [],
            error: errMsg,
          };
          handleHostMessage(genId, { type: 'WGSL_BENCH_RESULT', result });
        });
      break;
    case 'RUN_WGSL_OPFS_BENCH':
      void runner
        .runOpfsLoadBench(msg.action, msg.totalMb, msg.chunkMibList)
        .then((result) => handleHostMessage(genId, { type: 'WGSL_OPFS_BENCH_RESULT', result }))
        .catch((err) => {
          const result: WgslOpfsLoadResult = {
            action: msg.action,
            ok: false,
            totalBytes: 0,
            timings: [],
            details: err instanceof Error ? err.message : String(err),
          };
          handleHostMessage(genId, { type: 'WGSL_OPFS_BENCH_RESULT', result });
        });
      break;
  }
}

function handleClientMessage(port: chrome.runtime.Port, msg: ClientToEngineMessage): void {
  if (msg.type === 'GET_STATE') {
    port.postMessage(buildStateSnapshot());
    return;
  }

  messageQueue = messageQueue
    .then(async () => {
      const patch = extractConfigPatch(msg);
      if (patch) {
        applyConfigPatchToMirror(patch);
      }
      const requestedHost = patch?.executionHost;
      const didSwitch = Boolean(requestedHost && requestedHost !== activeHost);
      if (didSwitch && requestedHost) {
        await switchHost(requestedHost);
      }

      if (didSwitch && msg.type === 'UPDATE_CONFIG') {
        return;
      }

      if (activeHost === 'worker') {
        if (!activeWorker) {
          throw new Error('Engine worker is not active.');
        }
        activeWorker.postMessage(msg satisfies WorkerInboundMessage);
      } else {
        dispatchToLocalHost(hostGeneration, msg);
      }
    })
    .catch((err) => {
      const errMsg = err instanceof Error ? err.message : String(err);
      recordErrorAndBroadcast(`Engine host dispatch failed: ${errMsg}`);
      if (msg.type === 'GENERATE') {
        broadcast({ type: 'GENERATION_ERROR', requestId: msg.requestId, error: errMsg });
      }
    });
}

// Register port listener synchronously at top-level startup.
if (typeof chrome !== 'undefined' && chrome.runtime?.onConnect) {
  chrome.runtime.onConnect.addListener((port) => {
    if (port.name !== ENGINE_PORT_NAME) {
      return;
    }
    ports.add(port);

    port.postMessage(buildStateSnapshot());

    port.onMessage.addListener((rawMsg) => {
      handleClientMessage(port, rawMsg as ClientToEngineMessage);
    });

    port.onDisconnect.addListener(() => {
      ports.delete(port);
    });
  });
}

// Boot immediately in default 'worker' mode.
bootWorkerHost(hostGeneration, false);
