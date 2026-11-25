import { useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { createTinyGptChatAdapter } from './chatAdapter.ts';
import { ChatThread } from './components/ChatThread.tsx';
import { DiagnosticsDrawer } from './components/DiagnosticsDrawer.tsx';
import { TelemetryBar } from './components/TelemetryBar.tsx';
import { OffscreenEngineClient } from './engineClient.ts';

const engineClient = new OffscreenEngineClient();
(window as unknown as { __TINYGPT_CLIENT__?: OffscreenEngineClient }).__TINYGPT_CLIENT__ =
  engineClient;

export function App() {
  const snapshot = useSyncExternalStore(engineClient.subscribe, engineClient.getSnapshot);
  const [labOpen, setLabOpen] = useState(false);

  useEffect(() => {
    void engineClient.connect().catch(() => {
      // Connection errors are surfaced through snapshot/logs.
    });
  }, []);

  const adapter = useMemo(() => createTinyGptChatAdapter(engineClient), []);
  const modelReady =
    snapshot.telemetry.status === 'ready' || snapshot.telemetry.status === 'generating';

  return (
    <div className="app-shell">
      <TelemetryBar
        connected={snapshot.connected}
        telemetry={snapshot.telemetry}
        labOpen={labOpen}
        onToggleLab={() => setLabOpen((prev) => !prev)}
        onLoadModel={() => void engineClient.send({ type: 'LOAD_MODEL' })}
        onUnloadModel={() => void engineClient.send({ type: 'UNLOAD_MODEL' })}
        onResetContext={() => void engineClient.send({ type: 'RESET_CONTEXT' })}
      />

      {labOpen && (
        <DiagnosticsDrawer
          config={snapshot.config}
          telemetry={snapshot.telemetry}
          probeInfo={snapshot.probeInfo}
          opfsStatus={snapshot.opfsStatus}
          spikeProgress={snapshot.spikeProgress}
          spikeResult={snapshot.spikeResult}
          benchmarkProgress={snapshot.benchmarkProgress}
          lastBenchmark={snapshot.lastBenchmark}
          niahProgress={snapshot.niahProgress}
          lastNiah={snapshot.lastNiah}
          logs={snapshot.logs}
          onUpdateConfig={(patch) =>
            void engineClient.send({ type: 'UPDATE_CONFIG', config: patch })
          }
          onProbeWebGpu={() => void engineClient.send({ type: 'PROBE_WEBGPU' })}
          onCheckOpfs={() => void engineClient.send({ type: 'CHECK_OPFS' })}
          onRunOpfsSpike={(totalMiB, shardMiB) =>
            void engineClient.send({ type: 'RUN_OPFS_SPIKE', totalMiB, shardMiB })
          }
          onVerifyOpfsSpike={() => void engineClient.send({ type: 'VERIFY_OPFS_SPIKE' })}
          onClearOpfsSpike={() => void engineClient.send({ type: 'CLEAR_OPFS_SPIKE' })}
          onClearModelCache={() => void engineClient.send({ type: 'CLEAR_MODEL_CACHE' })}
          onSnapshotContext={() => void engineClient.send({ type: 'SNAPSHOT_CONTEXT' })}
          onRestoreContext={() => void engineClient.send({ type: 'RESTORE_CONTEXT' })}
          onResetContext={() => void engineClient.send({ type: 'RESET_CONTEXT' })}
          onRunBenchmarkSuite={(contextLengths, decodeTokens) =>
            void engineClient.send({
              type: 'RUN_BENCHMARK_SUITE',
              contextLengths,
              decodeTokens,
            })
          }
          onRunNiah={(targetTokens, depthRatio) =>
            void engineClient.send({
              type: 'RUN_NIAH',
              targetTokens,
              depthRatio,
            })
          }
        />
      )}

      <main className="chat-main">
        <ChatThread adapter={adapter} modelReady={modelReady} />
      </main>
    </div>
  );
}
