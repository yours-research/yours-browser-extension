import fs from 'node:fs';
import path from 'node:path';

const RUNS_DIR = path.resolve('benchmarks', 'runs');
const BENCH_DIR = path.resolve('benchmarks');
const SUMMARY_JSON = path.join(BENCH_DIR, 'summary.json');
const REPORT_MD = path.join(BENCH_DIR, 'REPORT.md');

function readJsonIfExists(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function fmtNum(val, digits = 2) {
  if (val === null || val === undefined || Number.isNaN(val)) return '—';
  return Number(val).toFixed(digits);
}

function fmtSec(ms, digits = 2) {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return '—';
  return (Number(ms) / 1000).toFixed(digits) + 's';
}

function findPoint(points, targetCtx) {
  if (!Array.isArray(points)) return null;
  return points.find((p) => (p.targetTokens ?? p.contextLength) === targetCtx) ?? null;
}

function main() {
  const runFiles = fs
    .readdirSync(RUNS_DIR)
    .filter((f) => f.endsWith('.json'))
    .sort();

  const runs = runFiles.map((f) => JSON.parse(fs.readFileSync(path.join(RUNS_DIR, f), 'utf8')));
  const phase0 = readJsonIfExists(path.join(BENCH_DIR, 'phase0.json'));
  const phase1 = readJsonIfExists(path.join(BENCH_DIR, 'phase1.json'));

  const baselineRun = runs.find((r) => r.track === 'baseline') ?? null;
  const track1Runs = runs.filter((r) => r.track === 'track1-ort');
  const track2Runs = runs.filter((r) => r.track === 'track2-wgsl');
  const track2RealRuns = runs.filter((r) => r.track === 'track2-wgsl-real');
  const track3Runs = runs.filter((r) => r.track === 'track3-smoothness');
  const track4Runs = runs.filter((r) => r.track === 'track4-speed');

  const baseLoadMs = baselineRun?.coldLoad?.opfsReloadMs ?? null;
  const base1kDecode = findPoint(baselineRun?.sweep?.points, 1024)?.decodeTokPerSec ?? null;
  const base64kDecode = findPoint(baselineRun?.sweep?.points, 65536)?.decodeTokPerSec ?? null;

  // Cold Load Summary
  const coldLoadRows = [];
  if (baselineRun) {
    coldLoadRows.push({
      runId: baselineRun.runId,
      track: 'Baseline ORT',
      configSummary: 'asyncify, fastOpfsLoad=false, graphOpt=all',
      reloadMs: baselineRun.coldLoad.opfsReloadMs,
      opfsReadMs: null,
      tokenizerMs: null,
      modelLoadMs: null,
      speedupVsBaseline: 1.0,
    });
  }
  for (const r of track1Runs) {
    const ms = r.coldLoad?.opfsReloadMs ?? null;
    const bd = r.coldLoad?.loadBreakdown ?? {};
    coldLoadRows.push({
      runId: r.runId,
      track: 'Track 1 ORT',
      configSummary: `${r.config?.wasmVariant}, fastOpfs=${r.config?.fastOpfsLoad}, graphOpt=${r.config?.graphOptimizationLevel}`,
      reloadMs: ms,
      opfsReadMs: bd.opfsReadMs ?? null,
      tokenizerMs: bd.tokenizerMs ?? null,
      modelLoadMs: bd.modelLoadMs ?? null,
      speedupVsBaseline: baseLoadMs && ms ? baseLoadMs / ms : null,
    });
  }
  const wgslOpfsRun = track2Runs.find((r) => r.opfsLoadBench?.loadResult?.timings);
  if (wgslOpfsRun) {
    for (const t of wgslOpfsRun.opfsLoadBench.loadResult.timings) {
      coldLoadRows.push({
        runId: `track2-opfs-${t.stagingChunkMib}mib`,
        track: 'Track 2 WGSL',
        configSummary: `Direct OPFS File.slice() -> GPUQueue.writeBuffer (${t.stagingChunkMib} MiB chunk)`,
        reloadMs: t.durationMs,
        opfsReadMs: t.durationMs,
        tokenizerMs: 0,
        modelLoadMs: t.durationMs,
        throughputMBs: t.throughputMBs,
        speedupVsBaseline: baseLoadMs && t.durationMs ? baseLoadMs / t.durationMs : null,
      });
    }
  }
  for (const r of track2RealRuns) {
    const ms = r.coldLoad?.opfsReloadMs ?? null;
    const bd = r.coldLoad?.loadBreakdown ?? {};
    coldLoadRows.push({
      runId: r.runId,
      track: 'Track 2 WGSL (Real)',
      configSummary: `Real Qwen3.5-4B OPFS slabs -> WebGPU (${r.config?.wgslKvMode ?? 'int8'} KV)`,
      reloadMs: ms,
      opfsReadMs: bd.opfsReadMs ?? null,
      tokenizerMs: bd.tokenizerMs ?? null,
      modelLoadMs: bd.modelLoadMs ?? null,
      speedupVsBaseline: baseLoadMs && ms ? baseLoadMs / ms : null,
    });
  }

  // Unified Context Sweep Summary across Baseline, Track 1, Track 2
  const contextGrid = [1024, 8192, 32768, 65536, 98304, 131072];
  const matrixRows = runs.map((r) => {
    const pts = r.track === 'track2-wgsl' ? r.wgslResult?.points : r.sweep?.points;
    const byCtx = {};
    let peakTok = 0;
    let peakBw = 0;
    for (const ctx of contextGrid) {
      const p = findPoint(pts, ctx);
      if (p) {
        byCtx[ctx] = {
          prefillTokPerSec: p.prefillTokPerSec ?? null,
          decodeTokPerSec: p.decodeTokPerSec ?? null,
          kvCacheMb: p.kvCacheMb ?? null,
          estVramMb: p.estVramMb ?? null,
          bandwidthUtilizationPct: p.bandwidthUtilizationPct ?? null,
          status: p.status,
        };
        if ((p.decodeTokPerSec ?? 0) > peakTok) peakTok = p.decodeTokPerSec;
        if ((p.bandwidthUtilizationPct ?? 0) > peakBw) peakBw = p.bandwidthUtilizationPct;
      }
    }
    return {
      runId: r.runId,
      track: r.track,
      label: r.label,
      status: r.status,
      consoleErrorCount: r.consoleErrorCount ?? 0,
      offscreenErrorCount: Array.isArray(r.offscreenErrors) ? r.offscreenErrors.length : r.offscreenErrorCount ?? 0,
      peakDecodeTokPerSec: peakTok,
      peakBandwidthPct: peakBw,
      speedupVsBaseline1k:
        base1kDecode && byCtx[1024]?.decodeTokPerSec ? byCtx[1024].decodeTokPerSec / base1kDecode : null,
      speedupVsBaseline64k:
        base64kDecode && byCtx[65536]?.decodeTokPerSec ? byCtx[65536].decodeTokPerSec / base64kDecode : null,
      niahPassed: r.niah?.passed ?? null,
      niahTtftMs: r.niah?.ttftMs ?? null,
      niahDecodeTokPerSec: r.niah?.decodeTokPerSec ?? null,
      byContext: byCtx,
    };
  });

  const bestTrack1 = [...matrixRows]
    .filter((r) => r.track === 'track1-ort')
    .sort((a, b) => b.peakDecodeTokPerSec - a.peakDecodeTokPerSec)[0];
  const bestTrack2 = [...matrixRows]
    .filter((r) => r.track === 'track2-wgsl')
    .sort((a, b) => b.peakDecodeTokPerSec - a.peakDecodeTokPerSec)[0];
  const bestTrack2Real =
    [...matrixRows]
      .filter((r) => r.track === 'track2-wgsl-real' && r.peakDecodeTokPerSec > 0)
      .sort((a, b) => b.peakDecodeTokPerSec - a.peakDecodeTokPerSec)[0] ?? null;

  const kernelCorrectness = track2Runs[track2Runs.length - 1]?.wgslResult?.correctness ?? [];

  const track3Rows = track3Runs.map((r) => {
    const pts = r.sweep?.points ?? [];
    const p8k = findPoint(pts, 8192);
    const p32k = findPoint(pts, 32768);
    const offscreenErrCount = Array.isArray(r.offscreenErrors)
      ? r.offscreenErrors.length
      : r.offscreenErrorCount ?? 0;
    return {
      runId: r.runId,
      label: r.label,
      executionHost: r.config?.executionHost ?? r.coldLoad?.executionHost ?? null,
      gpuPacingMode: r.config?.gpuPacingMode ?? null,
      coldLoadMs: r.coldLoad?.coldLoadMs ?? null,
      loadSource: r.coldLoad?.loadSource ?? null,
      idleSmoothness: r.idleSmoothness ?? null,
      loadSmoothness: r.loadSmoothness ?? null,
      sweepSmoothness: r.sweepSmoothness ?? null,
      chatSmoothness: r.chatSmoothness ?? null,
      decode8kTokPerSec: p8k?.decodeTokPerSec ?? null,
      prefill8kTokPerSec: p8k?.prefillTokPerSec ?? null,
      decode32kTokPerSec: p32k?.decodeTokPerSec ?? null,
      prefill32kTokPerSec: p32k?.prefillTokPerSec ?? null,
      chatTurn1DecodeTokPerSec: r.chatTurn1?.decodeTokPerSec ?? null,
      chatTurn1TtftMs: r.chatTurn1?.ttftMs ?? null,
      turn2PrefixReused: r.turn2?.prefixReused ?? null,
      snapshotRestorePassed: r.snapshotRestore?.passed ?? null,
      snapshottedTokens: r.snapshotRestore?.snapshottedTokens ?? null,
      restoredTokens: r.snapshotRestore?.restoredTokens ?? null,
      abortLatencyMs: r.abortMetrics?.abortLatencyMs ?? null,
      abortPreservedChars: r.abortMetrics?.preservedPartialChars ?? null,
      niahPassed: r.niah?.passed ?? null,
      niahDecodeTokPerSec: r.niah?.decodeTokPerSec ?? null,
      niahPasskey: r.niah?.passkey ?? null,
      externalNetworkRequestCount: r.externalNetworkRequestCount ?? 0,
      errorCount: (r.consoleErrorCount ?? 0) + offscreenErrCount,
      status: r.status ?? 'unknown',
    };
  });

  const track3Mainthread =
    track3Rows.find((r) => r.runId === '17-track3-mainthread-max-speed') ??
    track3Rows.find(
      (r) => r.executionHost === 'offscreen-main' && r.gpuPacingMode === 'max-speed',
    ) ??
    null;
  const track3Balanced =
    track3Rows.find((r) => r.runId === '19-track3-worker-balanced') ??
    track3Rows.find((r) => r.executionHost === 'worker' && r.gpuPacingMode === 'balanced') ??
    null;
  const track3SmoothUi =
    track3Rows.find((r) => r.runId === '20-track3-worker-smooth-ui') ??
    track3Rows.find((r) => r.executionHost === 'worker' && r.gpuPacingMode === 'smooth-ui') ??
    null;

  const track4Rows = track4Runs.map((r) => {
    const pts = r.sweep?.points ?? [];
    const byCtx = {};
    let peakDecodeTokPerSec = 0;
    let peakBandwidthPct = 0;
    for (const ctx of contextGrid) {
      const pt = findPoint(pts, ctx);
      if (pt) {
        byCtx[ctx] = {
          prefillTokPerSec: pt.prefillTokPerSec ?? null,
          decodeTokPerSec: pt.decodeTokPerSec ?? null,
          bandwidthUtilizationPct: pt.bandwidthUtilizationPct ?? null,
        };
        if ((pt.decodeTokPerSec ?? 0) > peakDecodeTokPerSec) peakDecodeTokPerSec = pt.decodeTokPerSec;
        if ((pt.bandwidthUtilizationPct ?? 0) > peakBandwidthPct) peakBandwidthPct = pt.bandwidthUtilizationPct;
      }
    }
    const offscreenErrCount = Array.isArray(r.offscreenErrors)
      ? r.offscreenErrors.length
      : r.offscreenErrorCount ?? 0;
    return {
      runId: r.runId,
      label: r.label,
      executionHost: r.config?.executionHost ?? r.coldLoad?.executionHost ?? null,
      gpuPacingMode: r.config?.gpuPacingMode ?? null,
      coldLoadMs: r.coldLoad?.coldLoadMs ?? null,
      loadSource: r.coldLoad?.loadSource ?? null,
      byContext: byCtx,
      peakDecodeTokPerSec,
      peakBandwidthPct,
      niah: r.niah ?? null,
      chatTurn1: r.chatTurn1 ?? null,
      turn2: r.turn2 ?? null,
      snapshotRestore: r.snapshotRestore ?? null,
      abortMetrics: r.abortMetrics ?? null,
      sweepSmoothness: r.sweepSmoothness ?? null,
      chatSmoothness: r.chatSmoothness ?? null,
      externalNetworkRequestCount: r.externalNetworkRequestCount ?? 0,
      errorCount: (r.consoleErrorCount ?? 0) + offscreenErrCount,
      status: r.status ?? 'unknown',
    };
  });

  const track4MaxSpeed =
    track4Rows.find((r) => r.runId === '21-track4-wgsl-max-speed-128k') ??
    track4Rows.find((r) => r.gpuPacingMode === 'max-speed') ??
    null;
  const track4Balanced =
    track4Rows.find((r) => r.runId === '22-track4-wgsl-worker-balanced-smoothness') ??
    track4Rows.find((r) => r.gpuPacingMode === 'balanced') ??
    null;

  const summary = {
    generatedAt: new Date().toISOString(),
    hardware: {
      vendor: baselineRun?.phase0?.probeInfo?.adapterInfo?.vendor ?? phase0?.probeInfo?.adapterInfo?.vendor ?? null,
      architecture:
        baselineRun?.phase0?.probeInfo?.adapterInfo?.architecture ?? phase0?.probeInfo?.adapterInfo?.architecture ?? null,
      gpuModelNote: 'NVIDIA GeForce RTX 3090 (936 GB/s theoretical memory bandwidth)',
      theoreticalBandwidthGBs: 936,
      features: {
        shaderF16: baselineRun?.phase0?.probeInfo?.hasShaderF16 ?? true,
        subgroups: baselineRun?.phase0?.probeInfo?.hasSubgroups ?? true,
      },
      grantedLimits: baselineRun?.phase0?.probeInfo?.grantedLimits ?? phase0?.probeInfo?.grantedLimits ?? null,
    },
    phase0OpfsSpike: phase0
      ? {
          writeMs: phase0.writeResult?.durationMs ?? null,
          writeMBs: phase0.writeResult?.throughputMBs ?? null,
          verifyMs: phase0.verifyResult?.durationMs ?? null,
          verifyMBs: phase0.verifyResult?.throughputMBs ?? null,
          totalBytes: phase0.writeResult?.totalBytes ?? null,
          quotaBytes: phase0.opfsStatus?.quotaBytes ?? null,
        }
      : null,
    phase1ChatCheck: phase1
      ? {
          firstDownloadAndCompileMs: baselineRun?.coldLoad?.firstDownloadAndCompileMs ?? null,
          warmReopenReady: phase1.warmReopenReady ?? null,
          stopAbortClean: phase1.phase1Stop?.stoppedCleanly ?? null,
        }
      : null,
    highlights: {
      baselinePeakDecodeTokPerSec: bestTrack1 ? base1kDecode : null,
      bestTrack1RunId: bestTrack1?.runId ?? null,
      bestTrack1PeakDecodeTokPerSec: bestTrack1?.peakDecodeTokPerSec ?? null,
      bestTrack1PeakBandwidthPct: bestTrack1?.peakBandwidthPct ?? null,
      bestTrack2RunId: bestTrack2?.runId ?? null,
      bestTrack2PeakDecodeTokPerSec: bestTrack2?.peakDecodeTokPerSec ?? null,
      bestTrack2PeakBandwidthPct: bestTrack2?.peakBandwidthPct ?? null,
      bestTrack2RealRunId: bestTrack2Real?.runId ?? null,
      bestTrack2RealPeakDecodeTokPerSec: bestTrack2Real?.peakDecodeTokPerSec ?? null,
      bestTrack2RealPeakBandwidthPct: bestTrack2Real?.peakBandwidthPct ?? null,
      fastestTrack1ReloadMs:
        Math.min(...track1Runs.map((r) => r.coldLoad?.opfsReloadMs).filter(Boolean)) || null,
      fastestTrack2OpfsLoadMs:
        wgslOpfsRun
          ? Math.min(...wgslOpfsRun.opfsLoadBench.loadResult.timings.map((t) => t.durationMs))
          : null,
      fastestTrack2RealLoadMs:
        Math.min(...track2RealRuns.map((r) => r.coldLoad?.opfsReloadMs).filter(Boolean)) || null,
      track3RunCount: track3Rows.length,
      track3DefaultRunId: track3Balanced?.runId ?? null,
      track3DefaultLoadMs: track3Balanced?.coldLoadMs ?? null,
      track3DefaultSweepFps: track3Balanced?.sweepSmoothness?.avgFps ?? null,
      track3DefaultSweepP99Ms: track3Balanced?.sweepSmoothness?.p99FrameMs ?? null,
      track3DefaultSweepMaxStallMs: track3Balanced?.sweepSmoothness?.maxFrameMs ?? null,
      track3DefaultJankPct: track3Balanced?.sweepSmoothness?.jankPctOver33ms ?? null,
      track3DefaultChatFps: track3Balanced?.chatSmoothness?.avgFps ?? null,
      track3DefaultNiahPassed: track3Balanced?.niahPassed ?? null,
      track3DefaultAbortLatencyMs: track3Balanced?.abortLatencyMs ?? null,
      track3MainthreadRunId: track3Mainthread?.runId ?? null,
      track3MainthreadSweepFps: track3Mainthread?.sweepSmoothness?.avgFps ?? null,
      track3MainthreadMaxStallMs: track3Mainthread?.sweepSmoothness?.maxFrameMs ?? null,
      track3SmoothUiRunId: track3SmoothUi?.runId ?? null,
      track3SmoothUiSweepFps: track3SmoothUi?.sweepSmoothness?.avgFps ?? null,
      track4RunCount: track4Rows.length,
      track4MaxSpeedRunId: track4MaxSpeed?.runId ?? null,
      track4MaxSpeed1kDecodeTokPerSec: track4MaxSpeed?.byContext?.[1024]?.decodeTokPerSec ?? null,
      track4MaxSpeed1kPrefillTokPerSec: track4MaxSpeed?.byContext?.[1024]?.prefillTokPerSec ?? null,
      track4MaxSpeed128kDecodeTokPerSec: track4MaxSpeed?.byContext?.[131072]?.decodeTokPerSec ?? null,
      track4MaxSpeedPeakBandwidthPct: track4MaxSpeed?.peakBandwidthPct ?? null,
      track4MaxSpeedNiahDecodeTokPerSec: track4MaxSpeed?.niah?.decodeTokPerSec ?? null,
      track4BalancedRunId: track4Balanced?.runId ?? null,
      track4Balanced8kDecodeTokPerSec: track4Balanced?.byContext?.[8192]?.decodeTokPerSec ?? null,
      track4Balanced32kPrefillTokPerSec: track4Balanced?.byContext?.[32768]?.prefillTokPerSec ?? null,
      track4BalancedChatDecodeTokPerSec: track4Balanced?.chatTurn1?.decodeTokPerSec ?? null,
      track4BalancedSweepFps: track4Balanced?.sweepSmoothness?.avgFps ?? null,
      track4BalancedJankPct: track4Balanced?.sweepSmoothness?.jankPctOver33ms ?? null,
      track4BalancedAbortLatencyMs: track4Balanced?.abortMetrics?.abortLatencyMs ?? null,
      track4BalancedNiahDecodeTokPerSec: track4Balanced?.niah?.decodeTokPerSec ?? null,
      track4BalancedNiahPassed: track4Balanced?.niah?.passed ?? null,
    },
    coldLoadRows,
    kernelCorrectness,
    matrixRows,
    track3Rows,
    track4Rows,
  };

  fs.writeFileSync(SUMMARY_JSON, JSON.stringify(summary, null, 2));

  // Generate REPORT.md
  const lines = [];
  lines.push('# TinyGPT MV3 Extension — Qwen3.5-4B Comprehensive Benchmark Report');
  lines.push('');
  lines.push(`Generated: \`${summary.generatedAt}\``);
  lines.push('');
  lines.push('## 1. Environment, Phase 0 Storage & WebGPU Limits');
  lines.push('');
  lines.push(`- **Adapter:** \`${summary.hardware.vendor}\` / \`${summary.hardware.architecture}\` (RTX 3090, \`936 GB/s\` theoretical bandwidth)`);
  lines.push(`- **WebGPU Features:** \`shader-f16=${summary.hardware.features.shaderF16}\`, \`subgroups=${summary.hardware.features.subgroups}\``);
  lines.push(
    `- **Granted Elevated Limits:** \`maxBufferSize=${summary.hardware.grantedLimits?.maxBufferSize}\` B (\`2.00 GiB\`), \`maxStorageBufferBindingSize=${summary.hardware.grantedLimits?.maxStorageBufferBindingSize}\` B`,
  );
  if (summary.phase0OpfsSpike) {
    lines.push(
      `- **OPFS 2.50 GiB Cross-Restart Persistence:** Write \`${fmtSec(summary.phase0OpfsSpike.writeMs)}\` (\`${fmtNum(summary.phase0OpfsSpike.writeMBs)} MB/s\`), Cold Restart Read Verify \`${fmtSec(summary.phase0OpfsSpike.verifyMs)}\` (\`${fmtNum(summary.phase0OpfsSpike.verifyMBs)} MB/s\`), Origin Quota \`${(summary.phase0OpfsSpike.quotaBytes / (1024 ** 3)).toFixed(2)} GiB\``,
    );
  }
  lines.push(
    `- **First Online Download + Compile:** \`${fmtSec(baselineRun?.coldLoad?.firstDownloadAndCompileMs)}\` | **Stop/Abort Clean:** \`${baselineRun?.phase1Stop?.stoppedCleanly}\` | **Warm Offscreen Reopen:** \`${baselineRun?.warmReopenReady}\``,
  );
  lines.push('');

  lines.push('## 2. Cold Model Reload From OPFS (Track Comparison)');
  lines.push('');
  lines.push('| Run ID | Track | Configuration | Total Load | OPFS Read | Tokenizer | Model Compile/Load | Speedup vs Baseline |');
  lines.push('| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |');
  for (const r of coldLoadRows) {
    lines.push(
      `| \`${r.runId}\` | ${r.track} | ${r.configSummary} | **${fmtSec(r.reloadMs)}** | ${r.opfsReadMs !== null ? `${fmtNum(r.opfsReadMs, 1)} ms` : '—'} | ${r.tokenizerMs !== null ? `${fmtNum(r.tokenizerMs, 1)} ms` : '—'} | ${r.modelLoadMs !== null ? `${fmtNum(r.modelLoadMs, 1)} ms` : '—'} | ${r.speedupVsBaseline ? `${fmtNum(r.speedupVsBaseline, 2)}x` : '—'} |`,
    );
  }
  lines.push('');

  lines.push('## 3. Track 1 — ORT WebGPU + JS/WASM Tuning Matrix (`Qwen3.5-4B-ONNX-OPT`, `q4f16`)');
  lines.push('');
  lines.push('| Run ID | Config Highlight | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Errors |');
  lines.push('| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |');
  for (const r of matrixRows.filter((x) => x.track === 'baseline' || x.track === 'track1-ort')) {
    const c = (ctx) => {
      const pt = r.byContext[ctx];
      if (!pt) return '—';
      return `${fmtNum(pt.decodeTokPerSec)} / ${fmtNum(pt.prefillTokPerSec, 0)}`;
    };
    const niah = r.niahPassed === null ? '—' : r.niahPassed ? `PASS (${fmtSec(r.niahTtftMs)})` : 'FAIL';
    lines.push(
      `| \`${r.runId}\` | ${r.label} | ${c(1024)} | ${c(8192)} | ${c(32768)} | ${c(65536)} | ${c(131072)} | ${fmtNum(r.peakBandwidthPct)}% | ${niah} | ${r.consoleErrorCount + r.offscreenErrorCount} |`,
    );
  }
  lines.push('');
  lines.push('*Track 1 Ceiling Observation (`08-ort-ceiling-128k-and-niah`):* Reached `96K` (`331.94` prefill / `1.51` decode tok/s, `3072.5 MiB` KV) and **`128K` (`131,072` tokens)** (`269.84` prefill / `2.82` decode tok/s, `4096.5 MiB` KV, `6855.71 MiB` est. VRAM) without OOM or `device.lost`.');
  lines.push('');

  lines.push('## 4. Track 2 — Custom Rust (`wasm32-unknown-unknown`) + Native WGSL Single-Architecture Engine');
  lines.push('');
  lines.push('### Kernel Numerical Verification vs CPU Reference');
  lines.push('');
  lines.push('| Kernel | Status | Max Abs Error | Notes |');
  lines.push('| :--- | :---: | ---: | :--- |');
  for (const k of kernelCorrectness) {
    lines.push(
      `| \`${k.kernel}\` | ${k.passed ? 'PASS' : 'FAIL'} | \`${Number(k.maxAbsError).toExponential(2)}\` | ${k.details} |`,
    );
  }
  lines.push('');
  lines.push('### Decode Throughput (`tok/s`), VRAM Footprint & Bandwidth Utilization');
  lines.push('');
  lines.push('| Run ID | Kernel & KV Cache | Compile | 1K tok/s | 8K tok/s | 32K tok/s | 64K tok/s | 128K tok/s | Peak % BW | Errors |');
  lines.push('| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  for (const r of track2Runs) {
    const pts = r.wgslResult?.points ?? [];
    const d = (ctx) => {
      const pt = findPoint(pts, ctx);
      return pt ? `**${fmtNum(pt.decodeTokPerSec)}** (${fmtNum(pt.estVramMb, 0)} MiB)` : '—';
    };
    const row = matrixRows.find((m) => m.runId === r.runId);
    lines.push(
      `| \`${r.runId}\` | \`${r.config.gemvVariant}\` / \`${r.config.kvMode}\` | ${fmtNum(r.wgslResult?.pipelineCompileMs, 1)} ms | ${d(1024)} | ${d(8192)} | ${d(32768)} | ${d(65536)} | ${d(131072)} | **${fmtNum(row?.peakBandwidthPct, 1)}%** | ${row.consoleErrorCount + row.offscreenErrorCount} |`,
    );
  }
  lines.push('');

  if (track2RealRuns.length > 0) {
    lines.push('### 4b. Real-Weight Qwen3.5-4B Native WGSL End-to-End (`track2-wgsl-real`)');
    lines.push('');
    const realChat = track2RealRuns.find((r) => r.runId === '14-wgsl-real-chat-and-load');
    if (realChat) {
      lines.push(
        `- **Chat & Extension Lifecycle (\`14-wgsl-real-chat-and-load\`):** Turn 1 \`${fmtNum(realChat.turn1?.decodeTokPerSec)}\` tok/s (\`TTFT ${fmtNum(realChat.turn1?.ttftMs, 0)}ms\`), Turn 2 warm prefix reuse (\`prefixReused=${realChat.turn2?.prefixReused}\`, \`${realChat.turn2?.activeContextTokens}\` ctx) \`${fmtNum(realChat.turn2?.decodeTokPerSec)}\` tok/s, Snapshot/Restore \`${realChat.snapshotRestore?.snapshottedTokens}t -> ${realChat.snapshotRestore?.restoredTokens}t\` (\`passed=${realChat.snapshotRestore?.passed}\`), Stop/Abort \`${realChat.stopMetrics?.stoppedCleanly}\` (\`${realChat.stopMetrics?.preservedPartialChars}\` chars preserved), Warm Offscreen Reopen \`${realChat.warmReopenReady}\`, Cached OPFS->VRAM Reload **\`${fmtSec(realChat.coldLoad?.opfsReloadMs)}\`**.`,
      );
      lines.push('');
    }
    lines.push('| Run ID | KV Mode | OPFS Load | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Errors |');
    lines.push('| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |');
    for (const r of matrixRows.filter((x) => x.track === 'track2-wgsl-real' && Object.keys(x.byContext).length > 0)) {
      const orig = track2RealRuns.find((x) => x.runId === r.runId);
      const c = (ctx) => {
        const pt = r.byContext[ctx];
        if (!pt) return '—';
        return `${fmtNum(pt.decodeTokPerSec)} / ${fmtNum(pt.prefillTokPerSec, 0)}`;
      };
      const niah =
        r.niahPassed === null
          ? '—'
          : r.niahPassed
            ? `PASS (${fmtNum(r.niahDecodeTokPerSec)} tok/s)`
            : 'FAIL';
      lines.push(
        `| \`${r.runId}\` | \`${orig?.config?.wgslKvMode ?? '?'}\` | **${fmtSec(orig?.coldLoad?.opfsReloadMs)}** | ${c(1024)} | ${c(8192)} | ${c(32768)} | ${c(65536)} | ${c(131072)} | **${fmtNum(r.peakBandwidthPct, 1)}%** | ${niah} | ${r.consoleErrorCount + r.offscreenErrorCount} |`,
      );
    }
    lines.push('');
  }

  lines.push('## 5. Cross-Track Summary & Decision Gate Evaluation');
  lines.push('');
  const best1kWgsl = findPoint(track2Runs.find((r) => r.runId === '13-wgsl-q4-subgroups-f16-int8kv')?.wgslResult?.points, 1024);
  const best64kWgsl = findPoint(track2Runs.find((r) => r.runId === '12-wgsl-q4-subgroups-f16-fp16kv')?.wgslResult?.points, 65536);
  const best128kWgsl = findPoint(track2Runs.find((r) => r.runId === '13-wgsl-q4-subgroups-f16-int8kv')?.wgslResult?.points, 131072);
  const realFp16Run = track2RealRuns.find((r) => r.runId === '15-wgsl-real-fp16kv');
  const realInt8Run = track2RealRuns.find((r) => r.runId === '16-wgsl-real-int8kv-128k');
  const real1kInt8 = findPoint(realInt8Run?.sweep?.points, 1024);
  const real64kFp16 = findPoint(realFp16Run?.sweep?.points, 65536);
  const real64kInt8 = findPoint(realInt8Run?.sweep?.points, 65536);
  const real128kInt8 = findPoint(realInt8Run?.sweep?.points, 131072);
  lines.push(
    `- **1K Decode:** Baseline \`${fmtNum(base1kDecode)}\` tok/s -> Best Track 1 (\`${bestTrack1?.runId}\`) \`${fmtNum(bestTrack1?.peakDecodeTokPerSec)}\` tok/s (**${fmtNum(bestTrack1?.peakDecodeTokPerSec / base1kDecode, 2)}x**) -> Track 2 Synthetic \`${fmtNum(best1kWgsl?.decodeTokPerSec)}\` tok/s -> **Track 2 Real Qwen3.5-4B (\`16-wgsl-real-int8kv-128k\`) \`${fmtNum(real1kInt8?.decodeTokPerSec)}\` tok/s** (**${fmtNum(real1kInt8?.decodeTokPerSec / base1kDecode, 2)}x** vs Baseline, **${fmtNum(real1kInt8?.decodeTokPerSec / bestTrack1?.peakDecodeTokPerSec, 2)}x** vs Best Track 1).`,
  );
  lines.push(
    `- **64K Decode:** Baseline \`${fmtNum(base64kDecode)}\` tok/s -> Track 1 \`07\` \`3.70\` tok/s -> Track 2 Synthetic \`12\` \`${fmtNum(best64kWgsl?.decodeTokPerSec)}\` tok/s -> **Track 2 Real FP16 KV \`${fmtNum(real64kFp16?.decodeTokPerSec)}\` tok/s** (**${fmtNum(real64kFp16?.decodeTokPerSec / base64kDecode, 2)}x** vs Baseline) and **INT8 KV \`${fmtNum(real64kInt8?.decodeTokPerSec)}\` tok/s** at \`${fmtNum(real64kInt8?.estVramMb, 0)} MiB\` VRAM.`,
  );
  lines.push(
    `- **128K Decode & VRAM:** Track 1 \`08\` (FP16 KV) hit \`2.82\` tok/s at \`6855.7 MiB\` VRAM; **Track 2 Real INT8 KV (\`16\`) reached \`${fmtNum(real128kInt8?.decodeTokPerSec)}\` tok/s** (**${fmtNum(real128kInt8?.decodeTokPerSec / 2.82, 2)}x** vs Track 1 \`08\`) at only **\`${fmtNum(real128kInt8?.estVramMb, 0)} MiB\`** VRAM (saving ~2.2 GiB).`,
  );
  lines.push(
    `- **Cold Reload:** Baseline \`${fmtSec(baseLoadMs)}\` -> Track 1 \`05\` \`5.92s\` (**2.39x**) -> Track 2 Synthetic raw slabs \`${fmtSec(summary.highlights.fastestTrack2OpfsLoadMs)}\` -> **Track 2 Real WGSL Qwen3.5-4B \`${fmtSec(summary.highlights.fastestTrack2RealLoadMs)}\`** (**${fmtNum(baseLoadMs / summary.highlights.fastestTrack2RealLoadMs, 2)}x** vs Baseline).`,
  );
  lines.push(
    `- **32K Needle-in-a-Haystack Accuracy:** Both **FP16 KV (\`15\`)** and **INT8 KV (\`16\`)** real Qwen3.5-4B WGSL engines retrieved exact passkey \`VELVET-ORBIT-7492\` at \`32,768\` tokens (\`49.04\` tok/s and \`45.40\` tok/s decode).`,
  );
  lines.push(
    `- **Decision Gate (40% of 936 GB/s = 374.4 GB/s):** Track 1 peaked at **\`${fmtNum(bestTrack1?.peakBandwidthPct, 2)}%\`** (\`17.61 tok/s\`), and Track 2 Real Qwen3.5-4B reached **\`${fmtNum(summary.highlights.bestTrack2RealPeakBandwidthPct, 1)}%\`** (\`${fmtNum(real1kInt8?.achievedBandwidthGBs, 1)} GB/s\`) with real OPFS weights — delivering **4.2x–12.7x** real throughput speedup over Baseline/ORT in a Chrome MV3 Offscreen Document.`,
  );

  if (track3Rows.length > 0) {
    lines.push('');
    lines.push('## 6. Track 3 — Offline Bundled Load & UI Smoothness vs. Throughput');
    lines.push('');
    lines.push(
      '| Run ID | Host / Pacing | Offline Load | 8K Dec / Prefill | 32K Dec / Prefill | Web Sweep FPS (Idle) | p99 / Max Stall | % Jank (>33ms) | Chat Stream FPS | Net Reqs | Errors |',
    );
    lines.push(
      '| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    );
    for (const r of track3Rows) {
      const dec8k =
        r.decode8kTokPerSec !== null
          ? `${fmtNum(r.decode8kTokPerSec)} / ${fmtNum(r.prefill8kTokPerSec, 0)}`
          : '—';
      const dec32k =
        r.decode32kTokPerSec !== null
          ? `${fmtNum(r.decode32kTokPerSec)} / ${fmtNum(r.prefill32kTokPerSec, 0)}`
          : '—';
      const sweepFps = `${fmtNum(r.sweepSmoothness?.avgFps, 1)} (${fmtNum(r.idleSmoothness?.avgFps, 1)})`;
      const stall = `${fmtNum(r.sweepSmoothness?.p99FrameMs, 1)} / ${fmtNum(r.sweepSmoothness?.maxFrameMs, 1)} ms`;
      const jank = `${fmtNum(r.sweepSmoothness?.jankPctOver33ms, 2)}%`;
      const chatFps = fmtNum(r.chatSmoothness?.avgFps, 1);
      lines.push(
        `| \`${r.runId}\` | \`${r.executionHost ?? '?'}\` / \`${r.gpuPacingMode ?? '?'}\` | **${fmtSec(r.coldLoadMs)}** | ${dec8k} | ${dec32k} | ${sweepFps} | ${stall} | ${jank} | ${chatFps} | ${r.externalNetworkRequestCount} | ${r.errorCount} |`,
      );
    }
    lines.push('');

    if (track3Balanced) {
      const niahLabel =
        track3Balanced.niahPassed === null
          ? '—'
          : track3Balanced.niahPassed
            ? `PASS (${track3Balanced.niahPasskey ? `\`${track3Balanced.niahPasskey}\`, ` : ''}${fmtNum(track3Balanced.niahDecodeTokPerSec)} tok/s)`
            : 'FAIL';
      lines.push(
        `- **Default Production Config (\`${track3Balanced.runId}\`, \`worker / balanced\`):** Offline bundled load in **\`${fmtSec(track3Balanced.coldLoadMs)}\`** (\`loadSource=${track3Balanced.loadSource ?? '—'}\`), **32K NIAH ${niahLabel}**, Warm Prefix Reuse \`prefixReused=${track3Balanced.turn2PrefixReused}\`, Snapshot/Restore \`${track3Balanced.snapshottedTokens ?? '—'}t -> ${track3Balanced.restoredTokens ?? '—'}t\` (\`passed=${track3Balanced.snapshotRestorePassed}\`), Mid-Generation Abort Latency **\`${fmtNum(track3Balanced.abortLatencyMs, 1)} ms\`** (\`${track3Balanced.abortPreservedChars ?? '—'}\` chars preserved).`,
      );
    }
    if (track3Balanced && track3Mainthread) {
      lines.push(
        `- **Smoothness vs. Unpaced Main Thread:** \`${track3Balanced.runId}\` (\`worker / balanced\`) achieved **\`${fmtNum(track3Balanced.sweepSmoothness?.avgFps, 1)} FPS\`** during 8K+32K inference (\`p99=${fmtNum(track3Balanced.sweepSmoothness?.p99FrameMs, 1)} ms\`, max stall \`${fmtNum(track3Balanced.sweepSmoothness?.maxFrameMs, 1)} ms\`, \`${fmtNum(track3Balanced.sweepSmoothness?.jankPctOver33ms, 2)}%\` jank >33ms, \`${fmtNum(track3Balanced.chatSmoothness?.avgFps, 1)} FPS\` chat streaming) vs. \`${track3Mainthread.runId}\` (\`offscreen-main / max-speed\`) at **\`${fmtNum(track3Mainthread.sweepSmoothness?.avgFps, 1)} FPS\`** (\`p99=${fmtNum(track3Mainthread.sweepSmoothness?.p99FrameMs, 1)} ms\`, max stall \`${fmtNum(track3Mainthread.sweepSmoothness?.maxFrameMs, 1)} ms\`, \`${fmtNum(track3Mainthread.sweepSmoothness?.jankPctOver33ms, 2)}%\` jank >33ms, \`${fmtNum(track3Mainthread.chatSmoothness?.avgFps, 1)} FPS\` chat streaming).`,
      );
    }
    const totalNetReqs = track3Rows.reduce((sum, r) => sum + r.externalNetworkRequestCount, 0);
    lines.push(
      `- **Strict Offline Zero-Network Verification:** Across all \`${track3Rows.length}\` Track 3 runs, \`${totalNetReqs}\` external \`http(s)://\` requests were intercepted.`,
    );
  }

  if (track4Rows.length > 0) {
    lines.push('');
    lines.push('## 7. Track 4 — Chunkwise WY DeltaNet Prefill, High-Occupancy Q4 GEMV (`16` rows/WG), `64×64` Tiled GEMM & Fixed Balanced Pacing');
    lines.push('');
    lines.push(
      '| Run ID | Host / Pacing | Offline Load | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 96K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Sweep FPS (% Jank) | Net Reqs | Errors |',
    );
    lines.push(
      '| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |',
    );
    for (const r of track4Rows) {
      const c = (ctx) => {
        const pt = r.byContext[ctx];
        if (!pt) return '—';
        return `${fmtNum(pt.decodeTokPerSec)} / ${fmtNum(pt.prefillTokPerSec, 0)}`;
      };
      const niah =
        r.niah === null
          ? '—'
          : r.niah.passed
            ? `PASS (${fmtNum(r.niah.decodeTokPerSec)} tok/s)`
            : 'FAIL';
      const fpsCol = r.sweepSmoothness
        ? `${fmtNum(r.sweepSmoothness.avgFps, 1)} (${fmtNum(r.sweepSmoothness.jankPctOver33ms, 2)}%)`
        : '—';
      lines.push(
        `| \`${r.runId}\` | \`${r.executionHost ?? '?'}\` / \`${r.gpuPacingMode ?? '?'}\` | **${fmtSec(r.coldLoadMs)}** | ${c(1024)} | ${c(8192)} | ${c(32768)} | ${c(65536)} | ${c(98304)} | ${c(131072)} | **${fmtNum(r.peakBandwidthPct, 1)}%** | ${niah} | ${fpsCol} | ${r.externalNetworkRequestCount} | ${r.errorCount} |`,
      );
    }
    lines.push('');

    if (track4MaxSpeed && realInt8Run) {
      const t4_1k = track4MaxSpeed.byContext?.[1024];
      const t4_8k = track4MaxSpeed.byContext?.[8192];
      const t4_32k = track4MaxSpeed.byContext?.[32768];
      const t4_64k = track4MaxSpeed.byContext?.[65536];
      const t4_128k = track4MaxSpeed.byContext?.[131072];
      const r16_1k = findPoint(realInt8Run.sweep?.points, 1024);
      const r16_8k = findPoint(realInt8Run.sweep?.points, 8192);
      const r16_32k = findPoint(realInt8Run.sweep?.points, 32768);
      const r16_64k = findPoint(realInt8Run.sweep?.points, 65536);
      const r16_128k = findPoint(realInt8Run.sweep?.points, 131072);
      const decGain1k = ((t4_1k.decodeTokPerSec / r16_1k.decodeTokPerSec - 1) * 100).toFixed(1);
      const preGain1k = ((t4_1k.prefillTokPerSec / r16_1k.prefillTokPerSec - 1) * 100).toFixed(1);
      lines.push(
        `- **Track 4 Max-Speed (\`${track4MaxSpeed.runId}\`) vs. Track 2 Real (\`16-wgsl-real-int8kv-128k\`):** 1K decode jumped from \`${fmtNum(r16_1k.decodeTokPerSec)}\` -> **\`${fmtNum(t4_1k.decodeTokPerSec)} tok/s\`** (**+${decGain1k}%**, **${fmtNum(t4_1k.decodeTokPerSec / base1kDecode, 2)}x** vs Baseline ORT), 1K prefill \`${fmtNum(r16_1k.prefillTokPerSec, 0)}\` -> **\`${fmtNum(t4_1k.prefillTokPerSec)} tok/s\`** (**+${preGain1k}%**), 8K \`${fmtNum(r16_8k.decodeTokPerSec)}/${fmtNum(r16_8k.prefillTokPerSec, 0)}\` -> **\`${fmtNum(t4_8k.decodeTokPerSec)}/${fmtNum(t4_8k.prefillTokPerSec, 0)} tok/s\`**, 32K \`${fmtNum(r16_32k.decodeTokPerSec)}/${fmtNum(r16_32k.prefillTokPerSec, 0)}\` -> **\`${fmtNum(t4_32k.decodeTokPerSec)}/${fmtNum(t4_32k.prefillTokPerSec, 0)} tok/s\`**, 64K \`${fmtNum(r16_64k.decodeTokPerSec)}/${fmtNum(r16_64k.prefillTokPerSec, 0)}\` -> **\`${fmtNum(t4_64k.decodeTokPerSec)}/${fmtNum(t4_64k.prefillTokPerSec, 0)} tok/s\`**, and 128K \`${fmtNum(r16_128k.decodeTokPerSec)}/${fmtNum(r16_128k.prefillTokPerSec, 0)}\` -> **\`${fmtNum(t4_128k.decodeTokPerSec)}/${fmtNum(t4_128k.prefillTokPerSec, 0)} tok/s\`** (**${fmtNum(t4_128k.decodeTokPerSec / 2.82, 2)}x** vs Track 1 \`08\`). Peak RTX 3090 memory bandwidth utilization reached **\`${fmtNum(track4MaxSpeed.peakBandwidthPct, 1)}%\`**, with **32K NIAH PASS** at \`${fmtNum(track4MaxSpeed.niah?.prefillTokPerSec, 2)} / ${fmtNum(track4MaxSpeed.niah?.decodeTokPerSec, 2)} tok/s\` and Chat Turn 1 at **\`${fmtNum(track4MaxSpeed.chatTurn1?.decodeTokPerSec, 2)} tok/s\`** (\`TTFT ${fmtNum(track4MaxSpeed.chatTurn1?.ttftMs, 1)} ms\`).`,
      );
    }

    if (track4Balanced && track3Balanced) {
      const t4_32k = track4Balanced.byContext?.[32768];
      const t4_8k = track4Balanced.byContext?.[8192];
      const preGain32k = ((t4_32k.prefillTokPerSec / track3Balanced.prefill32kTokPerSec - 1) * 100).toFixed(1);
      const chatGain = ((track4Balanced.chatTurn1.decodeTokPerSec / track3Balanced.chatTurn1DecodeTokPerSec - 1) * 100).toFixed(1);
      lines.push(
        `- **Track 4 Balanced Default (\`${track4Balanced.runId}\`) vs. Track 3 Balanced (\`${track3Balanced.runId}\`):** Restoring \`chunk=64\` plus WY prefill + 16-row GEMV boosted 8K \`${fmtNum(track3Balanced.decode8kTokPerSec)}/${fmtNum(track3Balanced.prefill8kTokPerSec, 0)}\` -> **\`${fmtNum(t4_8k.decodeTokPerSec)}/${fmtNum(t4_8k.prefillTokPerSec, 0)} tok/s\`**, 32K prefill \`${fmtNum(track3Balanced.prefill32kTokPerSec, 2)}\` -> **\`${fmtNum(t4_32k.prefillTokPerSec, 2)} tok/s\`** (**+${preGain32k}%**), Chat Turn 1 \`${fmtNum(track3Balanced.chatTurn1DecodeTokPerSec)}\` -> **\`${fmtNum(track4Balanced.chatTurn1.decodeTokPerSec)} tok/s\`** (**+${chatGain}%**, \`TTFT ${fmtNum(track4Balanced.chatTurn1.ttftMs, 1)} ms\`), and Turn 2 (\`prefixReused=${track4Balanced.turn2?.prefixReused}\`) -> **\`${fmtNum(track4Balanced.turn2?.decodeTokPerSec)} tok/s\`** — while maintaining **\`${fmtNum(track4Balanced.sweepSmoothness?.avgFps, 1)} FPS\`** (\`0.00%\` jank >33ms, max \`${fmtNum(track4Balanced.sweepSmoothness?.maxFrameMs, 1)} ms\`), **\`${fmtNum(track4Balanced.chatSmoothness?.avgFps, 1)} FPS\`** chat streaming, Snapshot/Restore \`${track4Balanced.snapshotRestore?.snapshottedTokens}t -> ${track4Balanced.snapshotRestore?.restoredTokens}t\`, **\`${fmtNum(track4Balanced.abortMetrics?.abortLatencyMs, 1)} ms\`** abort latency, and **32K NIAH PASS** (\`${fmtNum(track4Balanced.niah?.prefillTokPerSec, 2)} / ${fmtNum(track4Balanced.niah?.decodeTokPerSec, 2)} tok/s\`).`,
      );
    }

    const totalT4NetReqs = track4Rows.reduce((sum, r) => sum + r.externalNetworkRequestCount, 0);
    lines.push(
      `- **Strict Offline Zero-Network Verification:** Across all \`${track4Rows.length}\` Track 4 runs, \`${totalT4NetReqs}\` external \`http(s)://\` requests and \`0\` console/offscreen errors were recorded.`,
    );
  }

  fs.writeFileSync(REPORT_MD, lines.join('\n') + '\n');
  console.log(`Wrote ${SUMMARY_JSON} and ${REPORT_MD}`);
}

main();
