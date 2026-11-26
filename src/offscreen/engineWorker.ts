import type {
  ClientToEngineMessage,
  EngineToClientMessage,
  WgslBenchResult,
  WgslOpfsLoadResult,
} from '../shared/protocol.ts';
import { InferenceEngine, type EngineCallbacks } from './inferenceEngine.ts';
import { installWebGpuLimitPatch } from './webgpuLimits.ts';
import { WgslEngineRunner } from './wgslEngine.ts';

export interface WorkerInitMessage {
  type: 'WORKER_INIT';
  assetBaseUrl: string;
}

export type WorkerInboundMessage = WorkerInitMessage | ClientToEngineMessage;

interface DedicatedWorkerScopeLike {
  postMessage(message: EngineToClientMessage): void;
  onmessage: ((event: MessageEvent<WorkerInboundMessage>) => void) | null;
}

// Install WebGPU elevated-limits wrapper immediately before any adapter/device calls.
installWebGpuLimitPatch();

const workerSelf = self as unknown as DedicatedWorkerScopeLike;
const queuedMessages: ClientToEngineMessage[] = [];

let engine: InferenceEngine | null = null;
let wgslRunner: WgslEngineRunner | null = null;
let lastWgslBench: WgslBenchResult | null = null;
let lastWgslOpfs: WgslOpfsLoadResult | null = null;
let lifecycleQueue: Promise<void> = Promise.resolve();

function postToHost(message: EngineToClientMessage): void {
  workerSelf.postMessage(message);
}

function enqueueLifecycle(fn: () => Promise<void>): void {
  lifecycleQueue = lifecycleQueue.then(fn).catch(() => {
    // Errors are surfaced via engine telemetry and logs.
  });
}

function createCallbacks(): EngineCallbacks {
  return {
    onTelemetry: (telemetry) => postToHost({ type: 'TELEMETRY_UPDATE', telemetry }),
    onConfig: (config) => postToHost({ type: 'CONFIG_UPDATE', config }),
    onLog: (entry) => postToHost({ type: 'LOG', entry }),
    onProbeResult: (probeInfo) => postToHost({ type: 'WEBGPU_PROBE_RESULT', probeInfo }),
    onOpfsStatus: (opfsStatus) => postToHost({ type: 'OPFS_STATUS', opfsStatus }),
    onSpikeProgress: (message, pct) => postToHost({ type: 'OPFS_SPIKE_PROGRESS', message, pct }),
    onSpikeResult: (result) => postToHost({ type: 'OPFS_SPIKE_RESULT', result }),
    onBenchmarkProgress: (message, point) =>
      postToHost({ type: 'BENCHMARK_PROGRESS', message, point }),
    onBenchmarkResult: (result) => postToHost({ type: 'BENCHMARK_RESULT', result }),
    onNiahProgress: (message) => postToHost({ type: 'NIAH_PROGRESS', message }),
    onNiahResult: (result) => postToHost({ type: 'NIAH_RESULT', result }),
    onTokenDelta: (requestId, delta, tokenCount) =>
      postToHost({ type: 'TOKEN_DELTA', requestId, delta, tokenCount }),
    onGenerationComplete: (requestId, finishReason) =>
      postToHost({ type: 'GENERATION_COMPLETE', requestId, finishReason }),
    onGenerationError: (requestId, error) =>
      postToHost({ type: 'GENERATION_ERROR', requestId, error }),
    onPrefillComplete: (requestId, slotKey, tokens, elapsedMs) =>
      postToHost({ type: 'PREFILL_COMPLETE', requestId, slotKey, tokens, elapsedMs }),
    onPrefillError: (requestId, slotKey, error) =>
      postToHost({ type: 'PREFILL_ERROR', requestId, slotKey, error }),
    onEvictComplete: (requestId, slotKey) =>
      postToHost({ type: 'EVICT_COMPLETE', requestId, slotKey }),
  };
}

function handleClientMessage(msg: ClientToEngineMessage): void {
  if (!engine || !wgslRunner) {
    queuedMessages.push(msg);
    return;
  }
  const activeEngine = engine;
  const activeRunner = wgslRunner;

  switch (msg.type) {
    case 'GET_STATE':
      postToHost({
        type: 'STATE_SNAPSHOT',
        ...activeEngine.getStateSnapshot(),
        lastWgslBench,
        lastWgslOpfs,
      });
      break;
    case 'UPDATE_CONFIG':
      enqueueLifecycle(() => activeEngine.updateConfig(msg.config));
      break;
    case 'PROBE_WEBGPU':
      void activeEngine.runProbeWebGpu();
      break;
    case 'CHECK_OPFS':
      void activeEngine.refreshOpfsStatus();
      break;
    case 'RUN_OPFS_SPIKE':
      void activeEngine.runOpfsSpike(msg.totalMiB, msg.shardMiB);
      break;
    case 'VERIFY_OPFS_SPIKE':
      void activeEngine.verifyOpfsSpike();
      break;
    case 'CLEAR_OPFS_SPIKE':
      void activeEngine.clearOpfsSpike();
      break;
    case 'CLEAR_MODEL_CACHE':
      enqueueLifecycle(() => activeEngine.clearModelCache());
      break;
    case 'LOAD_MODEL':
      enqueueLifecycle(() => activeEngine.loadModel(msg.config));
      break;
    case 'UNLOAD_MODEL':
      enqueueLifecycle(() => activeEngine.unloadModel());
      break;
    case 'RESET_CONTEXT':
      enqueueLifecycle(() => activeEngine.resetContext(Boolean(msg.restoreBasePrompt)));
      break;
    case 'SNAPSHOT_CONTEXT':
      void activeEngine.snapshotContext();
      break;
    case 'RESTORE_CONTEXT':
      void activeEngine.restoreContext();
      break;
    case 'SET_DISPLAY_HZ':
      activeEngine.setDisplayHz(msg.hz);
      break;
    case 'PREFILL_BACKGROUND':
      activeEngine.enqueueBackgroundPrefill(msg.slotKey, msg.messages, {
        isDraft: msg.isDraft,
        url: msg.url,
        contentHash: msg.contentHash,
        requestId: msg.requestId,
      });
      break;
    case 'ABORT_BACKGROUND_PREFILL':
      activeEngine.abortBackgroundPrefill();
      break;
    case 'EVICT_SLOT':
      activeEngine.evictSlot(msg.slotKey, msg.requestId);
      break;
    case 'PROBE_HARDWARE_CATALOG':
    case 'DOWNLOAD_VERIFIED_MODEL':
    case 'CANCEL_MODEL_DOWNLOAD':
      // Handled in the offscreen document host.
      break;
    case 'GENERATE':
      activeEngine.enqueueGenerate(
        msg.requestId,
        msg.messages,
        msg.config,
        msg.groundingText,
        msg.stopOnToolCall,
      );
      break;
    case 'ABORT':
      activeEngine.abort(msg.requestId);
      break;
    case 'RUN_BENCHMARK_SUITE':
      activeEngine.enqueueBenchmarkSuite(msg.contextLengths, msg.decodeTokens, msg.chunkSize);
      break;
    case 'RUN_NIAH':
      activeEngine.enqueueNeedleInHaystack(msg.targetTokens, msg.depthRatio, msg.chunkSize);
      break;
    case 'RUN_WGSL_BENCH':
      void activeRunner
        .runFullModelDecodeBench(msg.config, (progressMsg) =>
          postToHost({ type: 'WGSL_BENCH_PROGRESS', message: progressMsg }),
        )
        .then((result) => {
          lastWgslBench = result;
          postToHost({ type: 'WGSL_BENCH_RESULT', result });
        })
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
          lastWgslBench = result;
          postToHost({ type: 'WGSL_BENCH_RESULT', result });
        });
      break;
    case 'RUN_WGSL_OPFS_BENCH':
      void activeRunner
        .runOpfsLoadBench(msg.action, msg.totalMb, msg.chunkMibList)
        .then((result) => {
          lastWgslOpfs = result;
          postToHost({ type: 'WGSL_OPFS_BENCH_RESULT', result });
        })
        .catch((err) => {
          const result: WgslOpfsLoadResult = {
            action: msg.action,
            ok: false,
            totalBytes: 0,
            timings: [],
            details: err instanceof Error ? err.message : String(err),
          };
          lastWgslOpfs = result;
          postToHost({ type: 'WGSL_OPFS_BENCH_RESULT', result });
        });
      break;
  }
}

function initializeWorker(assetBaseUrl: string): void {
  if (engine) {
    return;
  }
  engine = new InferenceEngine(createCallbacks(), {
    assetBaseUrl,
    executionHost: 'worker',
  });
  wgslRunner = new WgslEngineRunner(assetBaseUrl);

  const pending = queuedMessages.splice(0, queuedMessages.length);
  for (const msg of pending) {
    handleClientMessage(msg);
  }

  void engine.runProbeWebGpu().catch(() => {});
  void engine.refreshOpfsStatus().catch(() => {});
}

workerSelf.onmessage = (event: MessageEvent<WorkerInboundMessage>) => {
  const msg = event.data;
  if (!msg || typeof msg !== 'object') {
    return;
  }
  if (msg.type === 'WORKER_INIT') {
    initializeWorker(msg.assetBaseUrl);
    return;
  }
  handleClientMessage(msg);
};
