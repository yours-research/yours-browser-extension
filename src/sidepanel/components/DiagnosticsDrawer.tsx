import { useEffect, useState } from 'react';
import {
  ContextBudgetMeter,
  DataTableCard,
  HardwareStatusStrip,
  ShortcutSheet,
  formatCompactTokens,
  type DataTableColumn,
  type DataTableRow,
  type ShortcutGroup,
} from '@yours/ui';
import {
  isGemma4ModelId,
  type BenchmarkSuiteResult,
  type EngineConfig,
  type GpuPacingMode,
  type LogEntry,
  type NiahResult,
  type OpfsSpikeResult,
  type OpfsStatusInfo,
  type TelemetryState,
  type WebGpuProbeInfo,
  type WgslComputeTier,
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
  requestedTab?: LabTab;
  requestedTabNonce?: number;
}

type LabTab = 'config' | 'spike' | 'bench' | 'shortcuts';

const SHORTCUT_GROUPS: ShortcutGroup[] = [
  {
    id: 'composer',
    title: 'Composer & Thread',
    items: [
      { id: 'send', label: 'Send message', description: 'Submit prompt to local WebGPU runtime', keys: 'Enter' },
      { id: 'newline', label: 'New line', description: 'Insert line break in composer', keys: ['Shift', 'Enter'] },
      { id: 'stop', label: 'Stop generation', description: 'Abort active prefill or decode stream', keys: 'Esc' },
    ],
  },
  {
    id: 'drawers',
    title: 'Drawers & Inspection',
    items: [
      { id: 'shortcuts', label: 'Keyboard shortcuts', description: 'Open this reference sheet', keys: ['Alt', '/'] },
      { id: 'page-chip', label: 'Toggle active page grounding', description: 'Attach or detach current browser tab', keys: 'Page' },
      { id: 'think-chip', label: 'Toggle extended thinking', description: 'Enable or skip <think> reasoning trace', keys: 'Think' },
    ],
  },
];

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
  requestedTab,
  requestedTabNonce,
}: DiagnosticsDrawerProps) {
  const [tab, setTab] = useState<LabTab>(requestedTab ?? 'config');
  useEffect(() => {
    if (requestedTab) {
      setTab(requestedTab);
    }
  }, [requestedTab, requestedTabNonce]);
  const [niahTarget, setNiahTarget] = useState(32768);
  const [niahDepth, setNiahDepth] = useState(0.5);

  const isModelReady = telemetry.status === 'ready';
  const isGemma4 = isGemma4ModelId(config.modelId);

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
        <button
          type="button"
          className={tab === 'shortcuts' ? 'tab-btn tab-btn-active' : 'tab-btn'}
          onClick={() => setTab('shortcuts')}
        >
          Shortcuts
        </button>
      </div>

      {tab === 'config' && (
        <div className="lab-body">
          <ContextBudgetMeter
            title="Live KV Context Budget"
            usedTokens={telemetry.activeContextTokens}
            maxTokens={telemetry.allocatedKvCapacity || config.wgslMaxContext}
            summaryLabel={`${formatCompactTokens(telemetry.activeContextTokens)} / ${formatCompactTokens(telemetry.allocatedKvCapacity || config.wgslMaxContext)} tok`}
            segments={[
              {
                id: 'kv-used',
                label: 'Active KV tokens',
                tokensLabel: `${telemetry.activeContextTokens.toLocaleString()} tok`,
              },
              {
                id: 'kv-mb',
                label: 'KV + DeltaNet VRAM',
                tokensLabel: `${(telemetry.kvCacheMb + telemetry.deltaNetStateMb).toFixed(0)} MB`,
              },
            ]}
          />
          <HardwareStatusStrip
            tierLabel={
              telemetry.computeTier ??
              (probeInfo?.hasSubgroups && probeInfo?.hasShaderF16
                ? 'tier1-f16-subgroups'
                : probeInfo?.hasShaderF16
                  ? 'tier2-f16-shared'
                  : 'probing')
            }
            adapterLabel={
              probeInfo?.adapterInfo?.device ||
              probeInfo?.adapterInfo?.description ||
              probeInfo?.adapterInfo?.vendor ||
              'WebGPU adapter'
            }
            memoryLabel={
              telemetry.allocatedKvCapacity > 0
                ? `${telemetry.activeContextTokens.toLocaleString()}/${telemetry.allocatedKvCapacity.toLocaleString()} KV`
                : probeInfo?.grantedLimits?.maxBufferSize
                  ? `${Math.round(probeInfo.grantedLimits.maxBufferSize / (1024 * 1024))} MB max buf`
                  : undefined
            }
            status={
              telemetry.status === 'error'
                ? 'unsupported'
                : telemetry.computeTier
                  ? 'ready'
                  : 'probing'
            }
            sha256Verified={
              opfsStatus ? opfsStatus.modelFilesCount > 0 || Boolean(spikeResult?.ok) : undefined
            }
          />

          <div className="info-card">
            <div className="info-title">
              Model: {isGemma4 ? 'Gemma 4 E4B QAT (q4f16/q2, Bundled Offline)' : 'Qwen3.5-4B (q4f16, Bundled Offline)'}
            </div>
            <div className="info-row">
              Engine:{' '}
              {isGemma4
                ? 'Native Rust/WGSL Gemma 4 · INT8 KV + PLE Hot Cache + Suffix Spec'
                : 'Native Rust/WGSL Track 5 · INT8 KV + Suffix Speculation'}
            </div>
            <div className="info-row">
              Execution Host:{' '}
              {telemetry.executionHost === 'worker'
                ? 'Dedicated Web Worker'
                : 'Offscreen Main Thread'}{' '}
              · Tier: {telemetry.computeTier ?? 'auto'} · Display: {telemetry.displayHz}Hz
            </div>
            <div className="info-row">
              Dynamic KV Cap: {telemetry.allocatedKvCapacity.toLocaleString()} /{' '}
              {telemetry.maxKvCapacity.toLocaleString()} tok · Base Slot:{' '}
              {telemetry.basePromptCached ? `${telemetry.basePromptTokens}t cached` : 'unwarmed'}
            </div>
          </div>

          <div className="form-grid">
            <label className="field">
              <span>Compute Tier (Cross-GPU)</span>
              <select
                value={config.forceComputeTier ?? ''}
                onChange={(e) =>
                  onUpdateConfig({
                    forceComputeTier: (e.target.value || null) as WgslComputeTier | null,
                  })
                }
              >
                <option value="">Auto (T1 → T2 → T3 fallback)</option>
                <option value="tier1-f16-subgroups">Tier 1 — f16 + subgroups</option>
                <option value="tier2-f16-shared">Tier 2 — f16 + workgroup shared</option>
                <option value="tier3-f32-universal">Tier 3 — Universal f32</option>
              </select>
            </label>

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

          {(() => {
            const liveMetricColumns: DataTableColumn[] = [
              { key: 'metric', label: 'Metric', sortable: true },
              { key: 'value', label: 'Value', align: 'right', numeric: true, sortable: true },
              { key: 'target', label: 'Gate', align: 'right', sortable: true },
            ];
            const liveMetricRows: DataTableRow[] = [
              {
                metric: 'Last TTFT',
                value: telemetry.ttftMs !== null ? `${telemetry.ttftMs.toFixed(0)} ms` : '0 ms',
                target: '<= 1500 ms',
              },
              {
                metric: 'Decode speed',
                value:
                  telemetry.tokensPerSec !== null ? `${telemetry.tokensPerSec.toFixed(1)} tok/s` : '0.0 tok/s',
                target: '>= 45 tok/s',
              },
              {
                metric: 'KV tokens active',
                value: telemetry.activeContextTokens,
                target: `${telemetry.allocatedKvCapacity || 0} cap`,
              },
              {
                metric: 'Est. VRAM footprint',
                value: `${telemetry.estVramMb.toFixed(0)} MB`,
                target: telemetry.computeTier ?? 'tier1',
              },
            ];
            return (
              <DataTableCard
                title="Live Runtime Telemetry"
                subtitle="Sortable client-side snapshot of active WebGPU & KV state"
                badge={`${telemetry.displayHz} Hz`}
                columns={liveMetricColumns}
                rows={liveMetricRows}
                defaultSort={{ key: 'metric', direction: 'asc' }}
                copyFormat="markdown"
                onCopy={(formatted) => {
                  void navigator.clipboard?.writeText(formatted).catch(() => {});
                }}
              />
            );
          })()}

          {lastBenchmark && (() => {
            const benchColumns: DataTableColumn[] = [
              { key: 'ctx', label: 'Ctx', align: 'right', numeric: true, sortable: true },
              { key: 'prefill', label: 'Prefill', align: 'right', numeric: true, sortable: true },
              { key: 'ttft', label: 'TTFT', align: 'right', numeric: true, sortable: true },
              { key: 'decode', label: 'Decode', align: 'right', numeric: true, sortable: true },
              { key: 'bw', label: 'BW%', align: 'right', numeric: true, sortable: true },
              { key: 'status', label: 'Status', sortable: true },
            ];
            const benchRows: DataTableRow[] = lastBenchmark.points.map((pt) => ({
              ctx: pt.actualPromptTokens.toLocaleString(),
              prefill: `${pt.prefillTokPerSec.toFixed(0)} t/s`,
              ttft: `${pt.ttftMs.toFixed(0)} ms`,
              decode: `${pt.decodeTokPerSec.toFixed(1)} t/s`,
              bw: `${pt.bandwidthUtilizationPct.toFixed(1)}%`,
              status: pt.status,
            }));
            return (
              <DataTableCard
                title={`Benchmark Sweep (${lastBenchmark.modelId.split('/').pop()} ${lastBenchmark.dtype})`}
                subtitle={lastBenchmark.recommendation}
                badge={`Max ${lastBenchmark.maxSafeTokens.toLocaleString()} tok`}
                columns={benchColumns}
                rows={benchRows}
                defaultSort={{ key: 'ctx', direction: 'asc' }}
                copyFormat="markdown"
                onCopy={(formatted) => {
                  void navigator.clipboard?.writeText(formatted).catch(() => {});
                }}
                footerLabel={`OOM At: ${
                  lastBenchmark.oomTokenCount !== null
                    ? lastBenchmark.oomTokenCount.toLocaleString()
                    : 'None'
                }`}
              />
            );
          })()}

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

      {tab === 'shortcuts' && (
        <div className="lab-body">
          <ShortcutSheet
            title="Keyboard shortcuts"
            subtitle="Local side panel navigation & inference controls"
            badge="Zero cloud"
            groups={SHORTCUT_GROUPS}
          />
        </div>
      )}
    </section>
  );
}
