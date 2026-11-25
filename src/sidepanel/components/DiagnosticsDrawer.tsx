import { useState } from 'react';
import type {
  BenchmarkSuiteResult,
  EngineConfig,
  GpuPacingMode,
  LogEntry,
  NiahResult,
  OpfsSpikeResult,
  OpfsStatusInfo,
  TelemetryState,
  WebGpuProbeInfo,
} from '../../shared/protocol.ts';

interface DiagnosticsDrawerProps {
  config: EngineConfig;
  telemetry: TelemetryState;
  probeInfo: WebGpuProbeInfo | null;
  opfsStatus: OpfsStatusInfo | null;
  spikeProgress: { message: string; pct: number } | null;
  spikeResult: OpfsSpikeResult | null;
  benchmarkProgress: string | null;
  lastBenchmark: BenchmarkSuiteResult | null;
  niahProgress: string | null;
  lastNiah: NiahResult | null;
  logs: LogEntry[];
  onUpdateConfig: (patch: Partial<EngineConfig>) => void;
  onProbeWebGpu: () => void;
  onCheckOpfs: () => void;
  onRunOpfsSpike: (totalMiB: number, shardMiB: number) => void;
  onVerifyOpfsSpike: () => void;
  onClearOpfsSpike: () => void;
  onClearModelCache: () => void;
  onSnapshotContext: () => void;
  onRestoreContext: () => void;
  onResetContext: () => void;
  onRunBenchmarkSuite: (contextLengths: number[], decodeTokens: number) => void;
  onRunNiah: (targetTokens: number, depthRatio: number) => void;
}

type LabTab = 'config' | 'spike' | 'bench';

function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  return `${bytes} B`;
}

export function DiagnosticsDrawer({
  config,
  telemetry,
  probeInfo,
  opfsStatus,
  spikeProgress,
  spikeResult,
  benchmarkProgress,
  lastBenchmark,
  niahProgress,
  lastNiah,
  logs,
  onUpdateConfig,
  onProbeWebGpu,
  onCheckOpfs,
  onRunOpfsSpike,
  onVerifyOpfsSpike,
  onClearOpfsSpike,
  onSnapshotContext,
  onRestoreContext,
  onResetContext,
  onRunBenchmarkSuite,
  onRunNiah,
}: DiagnosticsDrawerProps) {
  const [tab, setTab] = useState<LabTab>('config');
  const [niahTarget, setNiahTarget] = useState(32768);
  const [niahDepth, setNiahDepth] = useState(0.5);

  const isModelReady = telemetry.status === 'ready';

  return (
    <section className="lab-drawer">
      <div className="lab-tabs">
        <button
          type="button"
          className={tab === 'config' ? 'tab-btn tab-btn-active' : 'tab-btn'}
          onClick={() => setTab('config')}
        >
          Engine &amp; State
        </button>
        <button
          type="button"
          className={tab === 'spike' ? 'tab-btn tab-btn-active' : 'tab-btn'}
          onClick={() => setTab('spike')}
        >
          Phase 0 Spike
        </button>
        <button
          type="button"
          className={tab === 'bench' ? 'tab-btn tab-btn-active' : 'tab-btn'}
          onClick={() => setTab('bench')}
        >
          Bench &amp; NIAH
        </button>
      </div>

      {tab === 'config' && (
        <div className="lab-body">
          <div className="info-card">
            <div className="info-title">Model: Qwen3.5-4B (q4f16, Bundled Offline)</div>
            <div className="info-row">
              Engine: Native Rust/WGSL Track 5 · INT8 KV + Suffix Speculation
            </div>
            <div className="info-row">
              Execution Host:{' '}
              {telemetry.executionHost === 'worker'
                ? 'Dedicated Web Worker'
                : 'Offscreen Main Thread'}
            </div>
          </div>

          <div className="form-grid">
            <label className="field">
              <span>GPU Pacing Mode</span>
              <select
                value={config.gpuPacingMode}
                onChange={(e) => onUpdateConfig({ gpuPacingMode: e.target.value as GpuPacingMode })}
              >
                <option value="balanced">Balanced — Smooth UI + Fast (Default)</option>
                <option value="smooth-ui">Smooth UI — 60 FPS Priority</option>
                <option value="max-speed">Max Speed — Unpaced Benchmark</option>
              </select>
            </label>

            <label className="field">
              <span>Max New Tokens</span>
              <input
                type="number"
                min={16}
                max={4096}
                step={64}
                value={config.maxNewTokens}
                onChange={(e) => onUpdateConfig({ maxNewTokens: Number(e.target.value) || 256 })}
              />
            </label>

            <label className="field">
              <span>Temperature (0 = greedy)</span>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={config.temperature}
                onChange={(e) => onUpdateConfig({ temperature: Number(e.target.value) || 0 })}
              />
            </label>

            <label className="field">
              <span>GPU Bandwidth (GB/s)</span>
              <input
                type="number"
                min={50}
                max={4000}
                step={10}
                value={config.gpuBandwidthGBs}
                onChange={(e) => onUpdateConfig({ gpuBandwidthGBs: Number(e.target.value) || 936 })}
              />
            </label>
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={config.enableThinking}
              onChange={(e) => onUpdateConfig({ enableThinking: e.target.checked })}
            />
            <span>Enable Qwen3.5 &lt;think&gt; mode (disables warm multi-turn prefix reuse)</span>
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={config.enableSpeculative}
              onChange={(e) => onUpdateConfig({ enableSpeculative: e.target.checked })}
            />
            <span>Enable Zero-Weight Suffix Speculation (greedy)</span>
          </label>

          <div className="btn-row">
            <button
              type="button"
              className="btn"
              disabled={!isModelReady || telemetry.activeContextTokens === 0}
              onClick={onSnapshotContext}
            >
              Snapshot Hybrid State
            </button>
            <button
              type="button"
              className="btn"
              disabled={!isModelReady || !telemetry.hasCpuSnapshot}
              onClick={onRestoreContext}
            >
              Restore Snapshot
              {telemetry.cpuSnapshotTokens !== null ? ` (${telemetry.cpuSnapshotTokens}t)` : ''}
            </button>
            <button type="button" className="btn" onClick={onResetContext}>
              Reset State
            </button>
          </div>

          <div className="kv-breakdown">
            <span>KV Attn: {telemetry.kvCacheMb.toFixed(1)} MiB</span>
            <span>DeltaNet Recurrent: {telemetry.deltaNetStateMb.toFixed(1)} MiB</span>
            <span>Max Reached: {telemetry.maxContextReached.toLocaleString()} tok</span>
            {telemetry.speculativeStats && (
              <span>
                Speculative: {telemetry.speculativeStats.accepted}/{telemetry.speculativeStats.drafted}{' '}
                ({telemetry.speculativeStats.acceptanceRatePct.toFixed(1)}%)
              </span>
            )}
          </div>

          <div className="log-box">
            {logs.length === 0 ? (
              <div className="muted">No engine logs yet.</div>
            ) : (
              logs.slice(-40).map((entry, idx) => (
                <div key={`${entry.ts}-${idx}`} className={`log-line log-${entry.level}`}>
                  <span className="log-meta">
                    [{entry.source}] {entry.ts.slice(11, 19)}
                  </span>{' '}
                  {entry.message}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {tab === 'spike' && (
        <div className="lab-body">
          <div className="btn-row">
            <button type="button" className="btn btn-primary" onClick={onProbeWebGpu}>
              Probe WebGPU Limits
            </button>
            <button type="button" className="btn" onClick={onCheckOpfs}>
              Refresh OPFS Status
            </button>
          </div>

          {probeInfo && (
            <div className="info-card">
              <div className="info-title">
                WebGPU Adapter: {probeInfo.supported ? 'OK' : 'UNAVAILABLE'}
              </div>
              {probeInfo.adapterInfo && (
                <div className="info-row">
                  {probeInfo.adapterInfo.vendor} / {probeInfo.adapterInfo.architecture}{' '}
                  {probeInfo.adapterInfo.description !== 'unknown'
                    ? `(${probeInfo.adapterInfo.description})`
                    : ''}
                </div>
              )}
              <div className="info-row">
                Features: shader-f16={String(probeInfo.hasShaderF16)}, subgroups=
                {String(probeInfo.hasSubgroups)} ({probeInfo.features.length} total)
              </div>
              {probeInfo.adapterLimits && (
                <div className="info-row">
                  Adapter maxBuffer: {formatBytes(probeInfo.adapterLimits.maxBufferSize)} |
                  maxStorageBinding: {formatBytes(probeInfo.adapterLimits.maxStorageBufferBindingSize)}
                </div>
              )}
              {probeInfo.grantedLimits && (
                <div className="info-row">
                  Elevated Device maxBuffer: {formatBytes(probeInfo.grantedLimits.maxBufferSize)} |
                  maxStorageBinding: {formatBytes(probeInfo.grantedLimits.maxStorageBufferBindingSize)}
                </div>
              )}
              {probeInfo.error && <div className="error-text">{probeInfo.error}</div>}
            </div>
          )}

          {opfsStatus && (
            <div className="info-card">
              <div className="info-title">OPFS Storage Status</div>
              <div className="info-row">
                Persisted: {String(opfsStatus.persisted)} | Usage:{' '}
                {formatBytes(opfsStatus.usageBytes)} / Quota: {formatBytes(opfsStatus.quotaBytes)}
              </div>
              <div className="info-row">
                Model cache: {opfsStatus.modelFilesCount} files (
                {formatBytes(opfsStatus.modelCacheBytes)}) | Spike files:{' '}
                {opfsStatus.spikeFiles.length}
              </div>
              {opfsStatus.error && <div className="error-text">{opfsStatus.error}</div>}
            </div>
          )}

          <div className="btn-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onRunOpfsSpike(2560, 256)}
            >
              Run 2.5 GiB OPFS Spike
            </button>
            <button type="button" className="btn" onClick={() => onRunOpfsSpike(64, 16)}>
              Quick 64 MiB Spike
            </button>
            <button type="button" className="btn" onClick={onVerifyOpfsSpike}>
              Verify After Restart
            </button>
            <button type="button" className="btn" onClick={onClearOpfsSpike}>
              Clear Spike Files
            </button>
          </div>

          {spikeProgress && (
            <div className="progress-banner">
              {spikeProgress.message} ({spikeProgress.pct.toFixed(0)}%)
            </div>
          )}

          {spikeResult && (
            <div className={spikeResult.ok ? 'info-card' : 'info-card info-card-error'}>
              <div className="info-title">
                OPFS Spike ({spikeResult.action}): {spikeResult.ok ? 'PASS' : 'FAIL'}
              </div>
              <div className="info-row">{spikeResult.details}</div>
              <div className="info-row">
                Throughput: {spikeResult.throughputMBs.toFixed(1)} MB/s | Time:{' '}
                {(spikeResult.durationMs / 1000).toFixed(2)}s
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'bench' && (
        <div className="lab-body">
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-primary"
              disabled={!isModelReady}
              onClick={() => onRunBenchmarkSuite([1024, 8192, 32768, 65536], 32)}
            >
              Run 1K / 8K / 32K / 64K Sweep
            </button>
            <button
              type="button"
              className="btn"
              disabled={!isModelReady}
              onClick={() => onRunBenchmarkSuite([1024, 8192], 16)}
            >
              Quick 1K / 8K
            </button>
          </div>

          <div className="niah-controls">
            <label className="field">
              <span>NIAH Context</span>
              <select
                value={niahTarget}
                onChange={(e) => setNiahTarget(Number(e.target.value))}
              >
                <option value={4096}>4K tokens</option>
                <option value={16384}>16K tokens</option>
                <option value={32768}>32K tokens (POC target)</option>
                <option value={65536}>64K tokens</option>
              </select>
            </label>
            <label className="field">
              <span>Needle Depth</span>
              <select
                value={niahDepth}
                onChange={(e) => setNiahDepth(Number(e.target.value))}
              >
                <option value={0.1}>10% (Start)</option>
                <option value={0.5}>50% (Middle)</option>
                <option value={0.9}>90% (End)</option>
              </select>
            </label>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!isModelReady}
              onClick={() => onRunNiah(niahTarget, niahDepth)}
            >
              Run NIAH Test
            </button>
          </div>

          {benchmarkProgress && <div className="progress-banner">{benchmarkProgress}</div>}
          {niahProgress && <div className="progress-banner">{niahProgress}</div>}

          {lastBenchmark && (
            <div className="info-card">
              <div className="info-title">
                Benchmark Sweep ({lastBenchmark.modelId.split('/').pop()} {lastBenchmark.dtype})
              </div>
              <div className="info-row">
                Max Safe Context: {lastBenchmark.maxSafeTokens.toLocaleString()} tok | OOM At:{' '}
                {lastBenchmark.oomTokenCount !== null
                  ? lastBenchmark.oomTokenCount.toLocaleString()
                  : 'None'}
              </div>
              <div className="info-row">{lastBenchmark.recommendation}</div>
              <table className="bench-table">
                <thead>
                  <tr>
                    <th>Ctx</th>
                    <th>Prefill</th>
                    <th>TTFT</th>
                    <th>Decode</th>
                    <th>BW%</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {lastBenchmark.points.map((pt) => (
                    <tr key={pt.targetTokens}>
                      <td>{pt.actualPromptTokens.toLocaleString()}</td>
                      <td>{pt.prefillTokPerSec.toFixed(0)} t/s</td>
                      <td>{pt.ttftMs.toFixed(0)} ms</td>
                      <td>{pt.decodeTokPerSec.toFixed(1)} t/s</td>
                      <td>{pt.bandwidthUtilizationPct.toFixed(1)}%</td>
                      <td>{pt.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {lastNiah && (
            <div className={lastNiah.passed ? 'info-card' : 'info-card info-card-error'}>
              <div className="info-title">
                NIAH ({lastNiah.actualTokens.toLocaleString()} tok @{' '}
                {(lastNiah.needleDepthPct * 100).toFixed(0)}%):{' '}
                {lastNiah.passed ? 'PASS' : 'FAIL'}
              </div>
              <div className="info-row">
                Expected: <code>{lastNiah.passkey}</code> | TTFT: {lastNiah.ttftMs.toFixed(0)} ms |
                Prefill: {lastNiah.prefillTokPerSec.toFixed(0)} t/s | Decode:{' '}
                {lastNiah.decodeTokPerSec.toFixed(1)} t/s
              </div>
              <div className="info-row">Response: {lastNiah.responseText || lastNiah.error}</div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
