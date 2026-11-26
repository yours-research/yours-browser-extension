import fs from 'node:fs';
import path from 'node:path';

const RUNS_DIR = path.resolve('benchmarks', 'runs');

function readJson(fileName) {
  const full = path.join(RUNS_DIR, fileName);
  if (!fs.existsSync(full)) {
    throw new Error(`Missing benchmark run file: ${full}`);
  }
  return JSON.parse(fs.readFileSync(full, 'utf8'));
}

function getSweepPoint(sweep, targetTokens) {
  const points = sweep?.points ?? sweep ?? [];
  return points.find((p) => p.targetTokens === targetTokens) ?? null;
}

function round(v, d = 2) {
  return v == null || !Number.isFinite(v) ? null : Number(v.toFixed(d));
}

function ratio(a, b) {
  if (!a || !b || !Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Number((a / b).toFixed(2));
}

function main() {
  const qOrt = readJson('00-baseline-ort-default.json');
  const qMax = readJson('23-track5-wgsl-max-speed-128k.json');
  const qBal = readJson('24-track5-wgsl-worker-balanced-smoothness.json');

  const g0 = readJson('g00-gemma4-math-audit.json');
  const g1Ort = readJson('g01-gemma4-ort-baseline.json');
  const g2Fp16 = readJson('g02-gemma4-wgsl-fp16kv.json');
  const g3Int8 = readJson('g03-gemma4-wgsl-int8kv-128k.json');
  const g4Max = readJson('g04-gemma4-wgsl-max-speed-128k.json');
  const g5Bal = readJson('g05-gemma4-wgsl-worker-balanced-smoothness.json');
  const g10Final = readJson('g10-gemma4-final-max-speed-128k.json');
  const g11Bal = readJson('g11-gemma4-final-balanced-smoothness.json');
  const g11B = g11Bal.balanced ?? g11Bal;

  const contexts = [1024, 8192, 32768, 65536, 98304, 131072];
  const sweepComparison = contexts.map((ctx) => {
    const qOrtPt = getSweepPoint(qOrt.sweep, ctx);
    const qWgslPt = getSweepPoint(qMax.sweep, ctx);
    const gOrtPt = getSweepPoint(g1Ort.contextSweep, ctx);
    const gFp16Pt = getSweepPoint(g2Fp16.sweep, ctx);
    const gInt8Pt = getSweepPoint(g3Int8.sweep, ctx);
    const gTrackG4Pt = getSweepPoint(g4Max.sweep, ctx);
    const gFinalPt = getSweepPoint(g10Final.sweep, ctx);

    return {
      contextTokens: ctx,
      qwen35Ort: qOrtPt
        ? { prefillTokPerSec: round(qOrtPt.prefillTokPerSec), decodeTokPerSec: round(qOrtPt.decodeTokPerSec) }
        : null,
      qwen35WgslMax: qWgslPt
        ? {
            prefillTokPerSec: round(qWgslPt.prefillTokPerSec),
            decodeTokPerSec: round(qWgslPt.decodeTokPerSec),
            achievedBandwidthGBs: round(qWgslPt.achievedBandwidthGBs),
            estVramMb: round(qWgslPt.estVramMb, 1),
          }
        : null,
      gemma4Ort: gOrtPt
        ? { prefillTokPerSec: round(gOrtPt.prefillTokPerSec), decodeTokPerSec: round(gOrtPt.decodeTokPerSec) }
        : null,
      gemma4WgslFp16: gFp16Pt
        ? { prefillTokPerSec: round(gFp16Pt.prefillTokPerSec), decodeTokPerSec: round(gFp16Pt.decodeTokPerSec) }
        : null,
      gemma4WgslInt8: gInt8Pt
        ? {
            prefillTokPerSec: round(gInt8Pt.prefillTokPerSec),
            decodeTokPerSec: round(gInt8Pt.decodeTokPerSec),
            achievedBandwidthGBs: round(gInt8Pt.achievedBandwidthGBs),
          }
        : null,
      gemma4WgslTrackG4: gTrackG4Pt
        ? {
            prefillTokPerSec: round(gTrackG4Pt.prefillTokPerSec),
            decodeTokPerSec: round(gTrackG4Pt.decodeTokPerSec),
            achievedBandwidthGBs: round(gTrackG4Pt.achievedBandwidthGBs),
          }
        : null,
      gemma4WgslWorkerMax: gFinalPt
        ? {
            prefillTokPerSec: round(gFinalPt.prefillTokPerSec),
            decodeTokPerSec: round(gFinalPt.decodeTokPerSec),
            achievedBandwidthGBs: round(gFinalPt.achievedBandwidthGBs),
          }
        : null,
      gemma4WgslVsOrtDecodeSpeedup: ratio(gFinalPt?.decodeTokPerSec, gOrtPt?.decodeTokPerSec),
      gemma4VsQwen35DecodeRatio: ratio(gFinalPt?.decodeTokPerSec, qWgslPt?.decodeTokPerSec),
      gemma4VsQwen35PrefillRatio: ratio(gFinalPt?.prefillTokPerSec, qWgslPt?.prefillTokPerSec),
    };
  });

  const comparison = {
    runId: 'g06-gemma4-vs-qwen35-comparison',
    recordedAt: new Date().toISOString(),
    models: {
      qwen35_4b: {
        modelId: qMax.config.modelId,
        coldLoadMs: round(qMax.coldLoad.coldLoadMs, 1),
        baseVramMb: round(qMax.coldLoad.estVramMb, 1),
        vram128kMb: round(getSweepPoint(qMax.sweep, 131072)?.estVramMb, 1),
        kvCache128kMb: round(getSweepPoint(qMax.sweep, 131072)?.kvCacheMb, 1),
        niah32k: {
          passed: qMax.niah.passed,
          ttftMs: round(qMax.niah.ttftMs, 1),
          prefillTokPerSec: round(qMax.niah.prefillTokPerSec, 2),
          decodeTokPerSec: round(qMax.niah.decodeTokPerSec, 2),
        },
        chatTurn1: {
          ttftMs: round(qMax.chatTurn1.ttftMs, 1),
          decodeTokPerSec: round(qMax.chatTurn1.decodeTokPerSec, 2),
        },
        speculativeCheck: {
          parityPassed: qMax.speculativeCheck.parityPassed,
          baselineTokPerSec: round(qMax.speculativeCheck.baselineDecodeTokPerSec, 2),
          speculativeTokPerSec: round(qMax.speculativeCheck.speculativeDecodeTokPerSec, 2),
          speedupRatio: qMax.speculativeCheck.speedupRatio,
          acceptanceRatePct: round(qMax.speculativeCheck.speculativeStats.acceptanceRatePct, 2),
        },
        balancedSmoothness: {
          chatTurn1DecodeTokPerSec: round(qBal.chatTurn1.decodeTokPerSec, 2),
          turn2PrefixReused: qBal.turn2.prefixReused,
          abortLatencyMs: qBal.abortMetrics.abortLatencyMs,
          chatAvgFps: qBal.chatSmoothness.avgFps,
          chatP99FrameMs: qBal.chatSmoothness.p99FrameMs,
          sweepAvgFps: qBal.sweepSmoothness.avgFps,
          sweepJankPct: qBal.sweepSmoothness.jankPctOver33ms,
        },
      },
      gemma4_e4b_qat: {
        modelId: g0.modelId,
        gpuWeightsGiB: g0.footprintBreakdownBytes.gpuWeightsGiB,
        cpuOpfsPleGiB: g0.footprintBreakdownBytes.cpuOpfsEmbeddingsGiB,
        coldLoadOrtMs: g1Ort.totalColdLoadMs,
        coldLoadWgslWorkerMs: g10Final.load.coldLoadMs,
        baseVramMb: g10Final.load.vramPlan.totalVramMb,
        vram128kMb: g10Final.load.vramPlan.totalVramMb,
        kvCache128kMb: g10Final.load.vramPlan.kvTotalMb,
        ortBaseline1kDecodeTokPerSec: g1Ort.contextSweep[0]?.decodeTokPerSec ?? null,
        wgslFp16Parity: {
          goldenAText: g10Final.golden.goldenA.text,
          goldenATop0: g10Final.golden.goldenA.top0 ?? null,
          goldenBTop0: g10Final.golden.goldenB.top0 ?? null,
          goldenBDecodeTokPerSec: g10Final.golden.goldenB.decodeTokPerSec,
        },
        niah32k: {
          passed: g10Final.niah.passed,
          ttftMs: round(g10Final.niah.ttftMs, 1),
          prefillTokPerSec: round(g10Final.niah.prefillTokPerSec, 2),
          decodeTokPerSec: round(g10Final.niah.decodeTokPerSec, 2),
        },
        chatTurn1: {
          ttftMs: round(g10Final.chatTurn1.ttftMs, 1),
          decodeTokPerSec: round(g10Final.chatTurn1.decodeTokPerSec, 2),
        },
        speculativeCheck: {
          parityPassed: g10Final.speculativeCheck.parityPassed,
          baselineTokPerSec: round(g10Final.speculativeCheck.baselineDecodeTokPerSec, 2),
          speculativeTokPerSec: round(g10Final.speculativeCheck.speculativeDecodeTokPerSec, 2),
          speedupRatio: g10Final.speculativeCheck.speedupRatio,
          acceptanceRatePct: round(g10Final.speculativeCheck.speculativeStats?.acceptanceRatePct, 2),
        },
        balancedSmoothness: {
          chatTurn1DecodeTokPerSec: round(g11B.chatTurn1.decodeTokPerSec, 2),
          turn2PrefixReused: g11B.turn2.prefixReused,
          abortLatencyMs: g11B.abortMetrics.abortLatencyMs,
          chatAvgFps: g11B.chatSmoothness.avgFps,
          chatP99FrameMs: g11B.chatSmoothness.p99FrameMs,
          sweepAvgFps: g11B.sweepSmoothness.avgFps,
          sweepJankPct: g11B.sweepSmoothness.jankPctOver33ms,
          sweep8kPrefillTokPerSec: round(getSweepPoint(g11B.sweep, 8192)?.prefillTokPerSec, 2),
          sweep8kDecodeTokPerSec: round(getSweepPoint(g11B.sweep, 8192)?.decodeTokPerSec, 2),
          sweep32kPrefillTokPerSec: round(getSweepPoint(g11B.sweep, 32768)?.prefillTokPerSec, 2),
          sweep32kDecodeTokPerSec: round(getSweepPoint(g11B.sweep, 32768)?.decodeTokPerSec, 2),
        },
        trackG4G5Baseline: {
          coldLoadMs: g4Max.coldLoad.coldLoadMs,
          chatTurn1DecodeTokPerSec: round(g4Max.chatTurn1.decodeTokPerSec, 2),
          balancedSweepAvgFps: g5Bal.sweepSmoothness.avgFps,
          balancedSweepJankPct: g5Bal.sweepSmoothness.jankPctOver33ms,
        },
      },
    },
    sweepComparison,
  };

  const outPath = path.join(RUNS_DIR, 'g06-gemma4-vs-qwen35-comparison.json');
  fs.writeFileSync(outPath, JSON.stringify(comparison, null, 2));
  console.log(`\nSaved ${outPath}\n`);

  console.log('| Context | Qwen3.5 ORT dec | Qwen3.5 WGSL Max dec (prefill) | Gemma 4 ORT dec | Gemma 4 WGSL Max dec (prefill) | G4 vs ORT | G4 vs Qwen3.5 dec |');
  console.log('|---:|---:|---:|---:|---:|---:|---:|');
  for (const row of sweepComparison) {
    const qOrtStr = row.qwen35Ort ? `${row.qwen35Ort.decodeTokPerSec}` : '-';
    const qWgStr = row.qwen35WgslMax ? `${row.qwen35WgslMax.decodeTokPerSec} (${row.qwen35WgslMax.prefillTokPerSec})` : '-';
    const gOrtStr = row.gemma4Ort ? `${row.gemma4Ort.decodeTokPerSec}` : '-';
    const gWgStr = row.gemma4WgslWorkerMax ? `${row.gemma4WgslWorkerMax.decodeTokPerSec} (${row.gemma4WgslWorkerMax.prefillTokPerSec})` : '-';
    console.log(
      `| ${row.contextTokens} | ${qOrtStr} | ${qWgStr} | ${gOrtStr} | ${gWgStr} | ${row.gemma4WgslVsOrtDecodeSpeedup ?? '-'}x | ${row.gemma4VsQwen35DecodeRatio ?? '-'}x |`,
    );
  }
}

main();
