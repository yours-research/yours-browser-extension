import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

async function launchExtensionPage() {
  if (!fs.existsSync(path.join(DIST_DIR, 'manifest.json'))) {
    throw new Error('dist/manifest.json missing. Run npm run build first.');
  }
  fs.mkdirSync(PROFILE_DIR, { recursive: true });

  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    viewport: { width: 1080, height: 960 },
    args: [
      `--disable-extensions-except=${DIST_DIR}`,
      `--load-extension=${DIST_DIR}`,
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });

  let sw = context.serviceWorkers()[0];
  if (!sw) {
    sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
  }
  const extensionId = new URL(sw.url()).hostname;

  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('pageerror', (err) => {
    consoleErrors.push(err.message);
  });

  await page.goto(`chrome-extension://${extensionId}/sidepanel.html`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => Boolean(window.__TINYGPT_CLIENT__ && window.__TINYGPT_CLIENT__.getSnapshot().connected),
    { timeout: 25000 },
  );

  return { context, page, consoleErrors };
}

async function getSnapshot(page) {
  return page.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());
}

async function waitForSnapshot(page, predicate, { timeoutMs = 600000, pollMs = 500, onTick } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastLog = 0;
  while (Date.now() < deadline) {
    const snap = await getSnapshot(page);
    if (onTick && Date.now() - lastLog >= 3000) {
      onTick(snap);
      lastLog = Date.now();
    }
    if (predicate(snap)) {
      return snap;
    }
    await new Promise((r) => setTimeout(r, pollMs));
  }
  const finalSnap = await getSnapshot(page);
  throw new Error(`Timed out after ${timeoutMs}ms. Last telemetry: ${JSON.stringify(finalSnap.telemetry)}`);
}

async function loadWithConfig(page, cfgPatch) {
  await page.evaluate(
    (patch) => window.__TINYGPT_CLIENT__.send({ type: 'LOAD_MODEL', config: patch }),
    cfgPatch,
  );
  const snap = await waitForSnapshot(
    page,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`Model load failed: ${s.telemetry.lastError}`);
      }
      return s.telemetry.status === 'ready';
    },
    {
      timeoutMs: 10 * 60 * 1000,
      pollMs: 400,
      onTick: (s) => {
        console.log(`  [Load] ${s.telemetry.status} ${s.telemetry.loadProgressPct.toFixed(0)}% - ${s.telemetry.loadPhaseLabel}`);
      },
    },
  );
  return snap;
}

async function runSweep(page, contexts, decodeTokens = 32, chunkSize) {
  const before = await getSnapshot(page);
  const prevAt = before.lastBenchmark?.finishedAt ?? null;
  await page.evaluate(
    ({ contexts: ctxs, decodeTokens: dec, chunkSize: cs }) =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_BENCHMARK_SUITE',
        contextLengths: ctxs,
        decodeTokens: dec,
        chunkSize: cs,
      }),
    { contexts, decodeTokens, chunkSize },
  );

  let lastMsg = '';
  const snap = await waitForSnapshot(
    page,
    (s) => s.lastBenchmark && s.lastBenchmark.finishedAt !== prevAt,
    {
      timeoutMs: 45 * 60 * 1000,
      pollMs: 500,
      onTick: (s) => {
        const chunk = s.telemetry.prefillChunkProgress
          ? ` (${s.telemetry.prefillChunkProgress.done}/${s.telemetry.prefillChunkProgress.total})`
          : '';
        const msg = `${s.benchmarkProgress ?? 'running'}${chunk}`;
        if (msg !== lastMsg) {
          console.log(`  [Sweep] ${msg}`);
          lastMsg = msg;
        }
      },
    },
  );
  return snap;
}

async function runNiah(page, targetTokens = 32768, depthRatio = 0.5, chunkSize) {
  const before = await getSnapshot(page);
  const prevJson = JSON.stringify(before.lastNiah ?? null);
  await page.evaluate(
    ({ targetTokens: t, depthRatio: d, chunkSize: cs }) =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_NIAH',
        targetTokens: t,
        depthRatio: d,
        chunkSize: cs,
      }),
    { targetTokens, depthRatio, chunkSize },
  );

  let lastMsg = '';
  const snap = await waitForSnapshot(
    page,
    (s) => s.lastNiah && JSON.stringify(s.lastNiah) !== prevJson,
    {
      timeoutMs: 25 * 60 * 1000,
      pollMs: 500,
      onTick: (s) => {
        const chunk = s.telemetry.prefillChunkProgress
          ? ` (${s.telemetry.prefillChunkProgress.done}/${s.telemetry.prefillChunkProgress.total})`
          : '';
        const msg = `${s.niahProgress ?? 'running'}${chunk}`;
        if (msg !== lastMsg) {
          console.log(`  [NIAH] ${msg}`);
          lastMsg = msg;
        }
      },
    },
  );
  return snap;
}

function saveRun(runId, payload) {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  const filePath = path.join(RUNS_DIR, `${runId}.json`);
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  console.log(`Saved ${filePath}`);
}

async function runIsolatedConfig({
  runId,
  label,
  configPatch,
  contexts,
  decodeTokens = 32,
  includeNiah = false,
  niahTokens = 32768,
}) {
  console.log(`\n=== [${runId}] ${label} ===`);
  let session;
  try {
    session = await launchExtensionPage();
    const loadedSnap = await loadWithConfig(session.page, configPatch);
    const coldLoad = {
      opfsReloadMs: loadedSnap.telemetry.coldLoadMs,
      loadedFromCache: loadedSnap.telemetry.loadedFromCache,
      loadBreakdown: loadedSnap.telemetry.loadBreakdown,
    };
    console.log(
      `  Cold OPFS load: ${(coldLoad.opfsReloadMs / 1000).toFixed(2)}s`,
      JSON.stringify(coldLoad.loadBreakdown),
    );

    const sweptSnap = await runSweep(
      session.page,
      contexts,
      decodeTokens,
      configPatch.chunkSize ?? 512,
    );
    let niah = null;
    if (includeNiah) {
      const niahSnap = await runNiah(
        session.page,
        niahTokens,
        0.5,
        configPatch.chunkSize ?? 512,
      );
      niah = niahSnap.lastNiah;
    }

    const finalSnap = await getSnapshot(session.page);
    const offscreenErrors = finalSnap.logs.filter((l) => l.level === 'error');
    const payload = {
      runId,
      track: 'track1-ort',
      recordedAt: new Date().toISOString(),
      label,
      config: finalSnap.config,
      coldLoad,
      sweep: sweptSnap.lastBenchmark,
      niah,
      consoleErrorCount: session.consoleErrors.length,
      offscreenErrors,
      status: 'ok',
    };
    saveRun(runId, payload);
    return payload;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`  [${runId}] FAILED: ${msg}`);
    const payload = {
      runId,
      track: 'track1-ort',
      recordedAt: new Date().toISOString(),
      label,
      config: configPatch,
      status: 'error',
      error: msg,
    };
    saveRun(runId, payload);
    return payload;
  } finally {
    if (session) {
      await session.context.close();
    }
  }
}

async function main() {
  const results = [];

  // 01: JS fast decode path (pooled masks, float16 LUT argmax, no yield clamp)
  const r01 = await runIsolatedConfig({
    runId: '01-ort-js-fastpath',
    label: 'ORT WebGPU + JS Fast Decode Path (yield=none, float16 LUT argmax, pooled mask)',
    configPatch: {
      wasmVariant: 'asyncify',
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: false,
      chunkSize: 512,
      graphOptimizationLevel: 'all',
      validationMode: 'basic',
      storageBufferCacheMode: 'default',
      enableMatmulFp32Accumulation: null,
    },
    contexts: [1024, 8192, 32768, 65536],
  });
  results.push(r01);

  // 02: Fast OPFS load (skip JS stream progress wrapper on warm cache)
  const r02 = await runIsolatedConfig({
    runId: '02-ort-fast-opfs-load',
    label: 'ORT WebGPU + Fast OPFS Load (arrayBuffer direct path)',
    configPatch: {
      wasmVariant: 'asyncify',
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: true,
      chunkSize: 512,
      graphOptimizationLevel: 'all',
      validationMode: 'basic',
      storageBufferCacheMode: 'default',
      enableMatmulFp32Accumulation: null,
    },
    contexts: [1024, 8192],
  });
  results.push(r02);

  // 03: JSPI WASM build vs asyncify
  const r03 = await runIsolatedConfig({
    runId: '03-ort-wasm-jspi',
    label: 'ORT WebGPU + JSPI WASM Build + JS Fast Decode',
    configPatch: {
      wasmVariant: 'jspi',
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: true,
      chunkSize: 512,
      graphOptimizationLevel: 'all',
      validationMode: 'basic',
      storageBufferCacheMode: 'default',
      enableMatmulFp32Accumulation: null,
    },
    contexts: [1024, 8192, 32768, 65536],
  });
  results.push(r03);

  // 04: EP tuning (storageBufferCacheMode='bucket', validationMode='wgpuOnly', fp32Accum=false)
  const r04 = await runIsolatedConfig({
    runId: '04-ort-ep-bucket-wgpuonly',
    label: 'ORT WebGPU EP Tuned (bucket cache, wgpuOnly validation, fp32Accum=false)',
    configPatch: {
      wasmVariant: 'asyncify',
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: true,
      chunkSize: 512,
      graphOptimizationLevel: 'all',
      validationMode: 'wgpuOnly',
      storageBufferCacheMode: 'bucket',
      enableMatmulFp32Accumulation: false,
    },
    contexts: [1024, 8192, 32768, 65536],
  });
  results.push(r04);

  // 05: Lower graph optimization level ('basic') to measure cold-load vs decode tradeoff
  const r05 = await runIsolatedConfig({
    runId: '05-ort-graph-opt-basic',
    label: 'ORT WebGPU graphOptimizationLevel=basic (cold compile vs throughput tradeoff)',
    configPatch: {
      wasmVariant: 'asyncify',
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: true,
      chunkSize: 512,
      graphOptimizationLevel: 'basic',
      validationMode: 'wgpuOnly',
      storageBufferCacheMode: 'bucket',
      enableMatmulFp32Accumulation: false,
    },
    contexts: [1024, 8192, 32768],
  });
  results.push(r05);

  // Determine winning base session config across r01, r03, r04 at 32K (fallback to r04 if r03 failed)
  const candidates = [r01, r03, r04].filter((r) => r.status === 'ok' && r.sweep?.points?.length);
  let winner = r04.status === 'ok' ? r04 : r01;
  let best32kDecode = -1;
  for (const cand of candidates) {
    const pt = cand.sweep.points.find((p) => p.targetTokens === 32768 && p.status === 'ok');
    if (pt && pt.decodeTokPerSec > best32kDecode) {
      best32kDecode = pt.decodeTokPerSec;
      winner = cand;
    }
  }
  const winCfg = {
    wasmVariant: winner.config.wasmVariant ?? 'asyncify',
    yieldMode: 'none',
    fastDecodePath: true,
    fastOpfsLoad: true,
    graphOptimizationLevel: winner.config.graphOptimizationLevel ?? 'all',
    validationMode: winner.config.validationMode ?? 'basic',
    storageBufferCacheMode: winner.config.storageBufferCacheMode ?? 'default',
    enableMatmulFp32Accumulation: winner.config.enableMatmulFp32Accumulation ?? null,
  };
  console.log(`\n[Track 1] Selected winning session config from ${winner.runId}:`, winCfg);

  // 06 & 07 & 08: Reuse single warm browser session across chunk=1024, chunk=2048, and 64K->128K ceiling + 32K NIAH
  let warmSession;
  try {
    warmSession = await launchExtensionPage();
    const loadedSnap = await loadWithConfig(warmSession.page, { ...winCfg, chunkSize: 1024 });
    const baseColdLoad = {
      opfsReloadMs: loadedSnap.telemetry.coldLoadMs,
      loadedFromCache: loadedSnap.telemetry.loadedFromCache,
      loadBreakdown: loadedSnap.telemetry.loadBreakdown,
    };

    // 06: chunk=1024
    console.log('\n=== [06-ort-chunk-1024] Chunk Size 1024 Sweep ===');
    const s06 = await runSweep(warmSession.page, [1024, 8192, 32768, 65536], 32, 1024);
    const p06 = {
      runId: '06-ort-chunk-1024',
      track: 'track1-ort',
      recordedAt: new Date().toISOString(),
      label: `Winning ORT Config (${winner.runId}) + Chunk Size 1024`,
      config: { ...winCfg, chunkSize: 1024 },
      coldLoad: baseColdLoad,
      sweep: s06.lastBenchmark,
      status: 'ok',
    };
    saveRun('06-ort-chunk-1024', p06);
    results.push(p06);

    // 07: chunk=2048
    console.log('\n=== [07-ort-chunk-2048] Chunk Size 2048 Sweep ===');
    const s07 = await runSweep(warmSession.page, [1024, 8192, 32768, 65536], 32, 2048);
    const p07 = {
      runId: '07-ort-chunk-2048',
      track: 'track1-ort',
      recordedAt: new Date().toISOString(),
      label: `Winning ORT Config (${winner.runId}) + Chunk Size 2048`,
      config: { ...winCfg, chunkSize: 2048 },
      coldLoad: baseColdLoad,
      sweep: s07.lastBenchmark,
      status: 'ok',
    };
    saveRun('07-ort-chunk-2048', p07);
    results.push(p07);

    // Pick best chunkSize (512 vs 1024 vs 2048) for 32K prefill without errors
    let bestChunk = 1024;
    let best32kPrefill = -1;
    for (const [cs, sw] of [
      [512, winner.sweep],
      [1024, s06.lastBenchmark],
      [2048, s07.lastBenchmark],
    ]) {
      const pt = sw?.points?.find((p) => p.targetTokens === 32768 && p.status === 'ok');
      if (pt && pt.prefillTokPerSec > best32kPrefill) {
        best32kPrefill = pt.prefillTokPerSec;
        bestChunk = cs;
      }
    }

    // 08: Context ceiling sweep (65536, 98304, 131072) + 32K NIAH
    console.log(`\n=== [08-ort-ceiling-128k-and-niah] Ceiling Test (chunk=${bestChunk}) + 32K NIAH ===`);
    const s08 = await runSweep(warmSession.page, [65536, 98304, 131072], 16, bestChunk);
    let niah = null;
    try {
      const niahSnap = await runNiah(warmSession.page, 32768, 0.5, bestChunk);
      niah = niahSnap.lastNiah;
    } catch (err) {
      niah = { error: err instanceof Error ? err.message : String(err), passed: false };
    }
    const finalSnap = await getSnapshot(warmSession.page);
    const p08 = {
      runId: '08-ort-ceiling-128k-and-niah',
      track: 'track1-ort',
      recordedAt: new Date().toISOString(),
      label: `Winning ORT Config (${winner.runId}, chunk=${bestChunk}) 64K-128K Ceiling + 32K NIAH`,
      config: { ...winCfg, chunkSize: bestChunk },
      coldLoad: baseColdLoad,
      sweep: s08.lastBenchmark,
      niah,
      consoleErrorCount: warmSession.consoleErrors.length,
      offscreenErrors: finalSnap.logs.filter((l) => l.level === 'error'),
      status: 'ok',
    };
    saveRun('08-ort-ceiling-128k-and-niah', p08);
    results.push(p08);
  } finally {
    if (warmSession) {
      await warmSession.context.close();
    }
  }

  console.log('\n[Track 1 Matrix Complete] Runs saved:', results.map((r) => `${r.runId}:${r.status}`).join(', '));
}

main().catch((err) => {
  console.error('[Track 1 Matrix Fatal]', err);
  process.exit(1);
});
