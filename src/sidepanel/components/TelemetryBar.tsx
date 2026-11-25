import type { TelemetryState } from '../../shared/protocol.ts';

interface TelemetryBarProps {
  connected: boolean;
  telemetry: TelemetryState;
  labOpen: boolean;
  onToggleLab: () => void;
  onLoadModel: () => void;
  onUnloadModel: () => void;
  onResetContext: () => void;
}

function statusBadgeClass(status: TelemetryState['status']): string {
  switch (status) {
    case 'ready':
      return 'badge badge-ready';
    case 'generating':
    case 'benchmarking':
      return 'badge badge-active';
    case 'downloading':
    case 'compiling':
      return 'badge badge-loading';
    case 'error':
      return 'badge badge-error';
    default:
      return 'badge badge-idle';
  }
}

export function TelemetryBar({
  connected,
  telemetry,
  labOpen,
  onToggleLab,
  onLoadModel,
  onUnloadModel,
  onResetContext,
}: TelemetryBarProps) {
  const canLoad = telemetry.status === 'unloaded' || telemetry.status === 'error';
  const isBusy =
    telemetry.status === 'downloading' ||
    telemetry.status === 'compiling' ||
    telemetry.status === 'generating' ||
    telemetry.status === 'benchmarking';

  const statusText =
    telemetry.status === 'downloading'
      ? `Loading ${telemetry.loadProgressPct.toFixed(0)}%`
      : telemetry.status === 'compiling'
        ? 'Compiling WebGPU...'
        : telemetry.status.toUpperCase();

  const loadSourceLabel =
    telemetry.loadSource === 'bundled'
      ? 'Bundled'
      : telemetry.loadedFromCache
        ? 'OPFS'
        : 'Net';

  const tierBadgeLabel =
    telemetry.computeTier === 'tier1-f16-subgroups'
      ? 'T1 f16+sg'
      : telemetry.computeTier === 'tier2-f16-shared'
        ? 'T2 f16'
        : telemetry.computeTier === 'tier3-f32-universal'
          ? 'T3 f32'
          : null;

  return (
    <header className="telemetry-bar">
      <div className="telemetry-row-top">
        <div className="telemetry-title">
          <span className={connected ? 'dot dot-online' : 'dot dot-offline'} />
          <strong>TinyGPT</strong>
          <span className={statusBadgeClass(telemetry.status)}>{statusText}</span>
          {tierBadgeLabel && (
            <span className="meta-chip" title={`Active WebGPU compute tier (${telemetry.displayHz}Hz)`}>
              {tierBadgeLabel} · {telemetry.displayHz}Hz
            </span>
          )}
          {telemetry.basePromptCached && (
            <span className="meta-chip" title={`Permanent Slot 0 cached (${telemetry.basePromptTokens}t)`}>
              Slot0
              {telemetry.lastSlotRestoreMs !== null ? ` ${telemetry.lastSlotRestoreMs.toFixed(1)}ms` : ''}
            </span>
          )}
          {telemetry.backgroundPrefillState !== 'idle' && (
            <span className="meta-chip" title="Background speculative prefix prefill state">
              BG:{telemetry.backgroundPrefillState}
            </span>
          )}
          {telemetry.coldLoadMs !== null && (
            <span className="meta-chip" title="Cold load time into WebGPU">
              {loadSourceLabel} {(telemetry.coldLoadMs / 1000).toFixed(2)}s
            </span>
          )}
          {telemetry.speculativeStats && telemetry.speculativeStats.drafted > 0 && (
            <span className="meta-chip" title="Zero-weight suffix speculative decoding acceptance">
              Spec {telemetry.speculativeStats.accepted}/{telemetry.speculativeStats.drafted} (
              {telemetry.speculativeStats.acceptanceRatePct.toFixed(0)}%)
            </span>
          )}
        </div>

        <div className="telemetry-actions">
          {canLoad ? (
            <button type="button" className="btn btn-primary" onClick={onLoadModel}>
              Load Model
            </button>
          ) : (
            <button
              type="button"
              className="btn"
              onClick={onUnloadModel}
              disabled={isBusy}
              title="Unload model and free WebGPU VRAM"
            >
              Unload
            </button>
          )}
          <button
            type="button"
            className="btn"
            onClick={onResetContext}
            disabled={telemetry.activeContextTokens === 0 || isBusy}
            title="Clear KV cache & DeltaNet recurrent state"
          >
            Reset KV
          </button>
          <button
            type="button"
            className={labOpen ? 'btn btn-active' : 'btn'}
            onClick={onToggleLab}
          >
            {labOpen ? 'Close Lab' : 'Lab'}
          </button>
        </div>
      </div>

      <div className="telemetry-metrics">
        <div className="metric">
          <span className="metric-label">Decode</span>
          <span className="metric-value">
            {telemetry.tokensPerSec !== null ? `${telemetry.tokensPerSec.toFixed(1)} tok/s` : '—'}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">TTFT</span>
          <span className="metric-value">
            {telemetry.ttftMs !== null ? `${telemetry.ttftMs.toFixed(0)} ms` : '—'}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Context</span>
          <span className="metric-value">
            {telemetry.activeContextTokens.toLocaleString()} tok
            {telemetry.prefixReused === true && ' (warm)'}
            {telemetry.prefixReused === false && telemetry.activeContextTokens > 0 && ' (reset)'}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Est. VRAM / KV</span>
          <span className="metric-value">
            {telemetry.estVramMb > 0
              ? `${(telemetry.estVramMb / 1024).toFixed(2)}G / ${(telemetry.kvCacheMb + telemetry.deltaNetStateMb).toFixed(0)}M`
              : '—'}
          </span>
        </div>
      </div>

      {telemetry.prefillChunkProgress && (
        <div className="progress-banner">
          Prefill chunk: {telemetry.prefillChunkProgress.done.toLocaleString()} /{' '}
          {telemetry.prefillChunkProgress.total.toLocaleString()} tokens
        </div>
      )}

      {telemetry.status === 'downloading' && (
        <div className="progress-banner">
          {telemetry.loadPhaseLabel} ({telemetry.loadProgressPct.toFixed(1)}%)
        </div>
      )}

      {(telemetry.lastError || telemetry.deviceLostReason) && (
        <div className="error-banner">
          {telemetry.deviceLostReason ? `GPU Lost (${telemetry.deviceLostReason}): ` : ''}
          {telemetry.lastError}
        </div>
      )}
    </header>
  );
}
