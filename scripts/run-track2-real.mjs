import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

function saveRun(runId, payload) {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  const filePath = path.join(RUNS_DIR, `${runId}.json`);
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  console.log(`Saved ${filePath}`);
}

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
  const sidePanelUrl = `chrome-extension://${extensionId}/sidepanel.html`;

  const consoleErrors = [];
  const blankPage = context.pages()[0] ?? (await context.newPage());

  async function openSidePanelTab() {
    const page = await context.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(`[SidePanel] ${msg.text()}`);
      }
    });
    page.on('pageerror', (err) => {
      consoleErrors.push(`[PageError] ${err.message}`);
    });
    await page.goto(sidePanelUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => Boolean(window.__TINYGPT_CLIENT__ && window.__TINYGPT_CLIENT__.getSnapshot().connected),
      { timeout: 25000 },
    );
    return page;
  }

  const page = await openSidePanelTab();
  return { context, page, blankPage, openSidePanelTab, consoleErrors };
}

async function getSnapshot(page) {
  return page.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());
}

async function waitForSnapshot(page, predicate, { timeoutMs = 600000, pollMs = 300, onTick } = {}) {
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
  throw new Error(
    `Timed out after ${timeoutMs}ms. Last telemetry: ${JSON.stringify(finalSnap.telemetry)}`,
  );
}

async function loadModelWithConfig(page, cfg, timeoutMs = 20 * 60 * 1000) {
  await page.evaluate(
    (c) => window.__TINYGPT_CLIENT__.send({ type: 'LOAD_MODEL', config: c }),
    cfg,
  );
  let lastPhase = '';
  const snap = await waitForSnapshot(
    page,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`Model load failed: ${s.telemetry.lastError}`);
      }
      return s.telemetry.status === 'ready' && s.config.backend === 'wgsl';
    },
    {
      timeoutMs,
      pollMs: 400,
      onTick: (s) => {
        const line = `status=${s.telemetry.status} pct=${s.telemetry.loadProgressPct.toFixed(0)}% phase="${s.telemetry.loadPhaseLabel}"`;
        if (line !== lastPhase) {
          console.log(`  [Load] ${line}`);
          lastPhase = line;
        }
      },
    },
  );
  return snap;
}

async function runSweep(page, contexts, decodeTokens, chunkSize, timeoutMs = 45 * 60 * 1000) {
  const prevAt = (await getSnapshot(page)).lastBenchmark?.finishedAt ?? null;
  await page.evaluate(
    ({ contextLengths, decodeTokens: dt, chunkSize: cs }) =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_BENCHMARK_SUITE',
        contextLengths,
        decodeTokens: dt,
        chunkSize: cs,
      }),
    { contextLengths: contexts, decodeTokens, chunkSize },
  );

  let lastMsg = '';
  return waitForSnapshot(
    page,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`Benchmark failed: ${s.telemetry.lastError}`);
      }
      return s.lastBenchmark && s.lastBenchmark.finishedAt !== prevAt;
    },
    {
      timeoutMs,
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
}

async function runNiah(page, targetTokens, depthRatio, chunkSize, timeoutMs = 25 * 60 * 1000) {
  const prevJson = JSON.stringify((await getSnapshot(page)).lastNiah ?? null);
  await page.evaluate(
    ({ targetTokens: t, depthRatio: d, chunkSize: c }) =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_NIAH',
        targetTokens: t,
        depthRatio: d,
        chunkSize: c,
      }),
    { targetTokens, depthRatio, chunkSize },
  );

  let lastMsg = '';
  return waitForSnapshot(
    page,
    (s) => s.lastNiah && JSON.stringify(s.lastNiah) !== prevJson,
    {
      timeoutMs,
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
}

async function runChatTurn(page, prompt, expectedMsgCount, timeoutMs = 3 * 60 * 1000) {
  await page.locator('.composer-input').fill(prompt);
  await page.locator('.composer-actions button', { hasText: 'Send' }).click();

  await waitForSnapshot(
    page,
    (s) => s.telemetry.status === 'generating' || s.telemetry.activeContextTokens > 0,
    { timeoutMs: 60000, pollMs: 100 },
  );

  const postSnap = await waitForSnapshot(
    page,
    (s) => {
      if (s.telemetry.status === 'error' || s.telemetry.lastError) {
        throw new Error(`Chat turn failed: ${s.telemetry.lastError}`);
      }
      return s.telemetry.status === 'ready';
    },
    { timeoutMs, pollMs: 250 },
  );

  await page.waitForFunction(
    (count) => document.querySelectorAll('.msg-assistant .msg-content').length >= count,
    expectedMsgCount,
    { timeout: 10000 },
  );

  const responseText = await page.locator('.msg-assistant .msg-content').last().innerText();
  return {
    prompt,
    responseText,
    responseLength: responseText.length,
    ttftMs: postSnap.telemetry.ttftMs,
    prefillTokPerSec: postSnap.telemetry.prefillTokPerSec,
    decodeTokPerSec: postSnap.telemetry.tokensPerSec,
    activeContextTokens: postSnap.telemetry.activeContextTokens,
    prefixReused: postSnap.telemetry.prefixReused,
    estVramMb: postSnap.telemetry.estVramMb,
  };
}

async function run14ChatAndLoad() {
  const runId = '14-wgsl-real-chat-and-load';
  const cfg = {
    backend: 'wgsl',
    wgslKvMode: 'int8',
    wgslMaxContext: 131072,
    yieldMode: 'none',
    fastDecodePath: true,
    fastOpfsLoad: true,
    chunkSize: 64,
    enableThinking: false,
    temperature: 0,
    maxNewTokens: 256,
  };

  console.log(`\n=== [${runId}] Repack/Load, Multi-Turn Chat, Snapshot/Restore, Abort & Cold OPFS Reload ===`);
  let session = await launchExtensionPage();
  let initialLoad;
  let turn1;
  let turn2;
  let snapshotRestore;
  let stopMetrics;
  let warmReopenReady = false;
  let consoleErrors = [];
  let offscreenErrors = [];

  try {
    const loadSnap = await loadModelWithConfig(session.page, cfg);
    initialLoad = {
      coldLoadMs: loadSnap.telemetry.coldLoadMs,
      loadedFromCache: loadSnap.telemetry.loadedFromCache,
      loadBreakdown: loadSnap.telemetry.loadBreakdown,
      estVramMb: loadSnap.telemetry.estVramMb,
      deltaNetStateMb: loadSnap.telemetry.deltaNetStateMb,
    };
    console.log('  Initial WGSL Load:', JSON.stringify(initialLoad));

    // Turn 1: Real English streaming generation
    turn1 = await runChatTurn(
      session.page,
      'List 3 benefits of hybrid linear + full attention in 3 short bullets.',
      1,
    );
    console.log(
      `  Turn 1 (${turn1.decodeTokPerSec?.toFixed(1)} tok/s, TTFT=${turn1.ttftMs?.toFixed(0)}ms): ${JSON.stringify(turn1.responseText.slice(0, 160))}`,
    );
    assert(turn1.responseText.trim().length >= 30, `Turn 1 too short/empty: "${turn1.responseText}"`);

    // Turn 2: Warm multi-turn prefix reuse
    turn2 = await runChatTurn(session.page, 'Summarize bullet 1 in one short sentence.', 2);
    console.log(
      `  Turn 2 (prefixReused=${turn2.prefixReused}, ctx=${turn2.activeContextTokens}, ${turn2.decodeTokPerSec?.toFixed(1)} tok/s): ${JSON.stringify(turn2.responseText.slice(0, 160))}`,
    );
    assert(turn2.prefixReused === true, `Expected prefixReused=true on Turn 2, got ${turn2.prefixReused}`);
    assert(turn2.responseText.trim().length >= 15, `Turn 2 too short/empty: "${turn2.responseText}"`);

    // Snapshot & Restore test
    const ctxBeforeSnap = turn2.activeContextTokens;
    await session.page.evaluate(() => window.__TINYGPT_CLIENT__.send({ type: 'SNAPSHOT_CONTEXT' }));
    const snapped = await waitForSnapshot(
      session.page,
      (s) => s.telemetry.hasCpuSnapshot === true && s.telemetry.cpuSnapshotTokens === ctxBeforeSnap,
      { timeoutMs: 30000, pollMs: 100 },
    );
    await session.page.evaluate(() => window.__TINYGPT_CLIENT__.send({ type: 'RESET_CONTEXT' }));
    await waitForSnapshot(session.page, (s) => s.telemetry.activeContextTokens === 0, {
      timeoutMs: 15000,
      pollMs: 100,
    });
    await session.page.evaluate(() => window.__TINYGPT_CLIENT__.send({ type: 'RESTORE_CONTEXT' }));
    const restored = await waitForSnapshot(
      session.page,
      (s) => s.telemetry.activeContextTokens === snapped.telemetry.cpuSnapshotTokens,
      { timeoutMs: 30000, pollMs: 100 },
    );
    snapshotRestore = {
      snapshottedTokens: snapped.telemetry.cpuSnapshotTokens,
      restoredTokens: restored.telemetry.activeContextTokens,
      passed: restored.telemetry.activeContextTokens === ctxBeforeSnap,
    };
    console.log('  Snapshot/Restore:', JSON.stringify(snapshotRestore));
    assert(snapshotRestore.passed, 'Snapshot/Restore token count mismatch.');

    // AbortSignal / Stop test
    await session.page
      .locator('.composer-input')
      .fill('Write a very detailed 1500-word technical essay explaining WebGPU compute shaders, workgroup shared memory, and subgroup matrix operations.');
    await session.page.locator('.composer-actions button', { hasText: 'Send' }).click();

    await session.page.waitForFunction(
      () => {
        const msgs = document.querySelectorAll('.msg-assistant .msg-content');
        return msgs.length >= 3 && (msgs[msgs.length - 1].textContent?.length ?? 0) >= 24;
      },
      { timeout: 60000 },
    );
    await session.page.locator('.composer-actions button', { hasText: 'Stop' }).click();
    const postStopSnap = await waitForSnapshot(
      session.page,
      (s) => s.telemetry.status === 'ready',
      { timeoutMs: 15000, pollMs: 100 },
    );
    const partialText = await session.page.locator('.msg-assistant .msg-content').last().innerText();
    stopMetrics = {
      stoppedCleanly: postStopSnap.telemetry.status === 'ready',
      preservedPartialChars: partialText.length,
      partialPreview: partialText.slice(0, 140),
    };
    console.log('  Stop/Abort:', JSON.stringify(stopMetrics));
    assert(stopMetrics.stoppedCleanly && partialText.length >= 24, 'Abort did not stop cleanly or lost partial text.');

    // Warm Offscreen persistence across Side Panel close & reopen
    await session.page.close();
    const reopened = await session.openSidePanelTab();
    const reopenedSnap = await waitForSnapshot(
      reopened,
      (s) => s.connected && s.telemetry.status === 'ready',
      { timeoutMs: 15000, pollMs: 150 },
    );
    warmReopenReady = reopenedSnap.telemetry.status === 'ready';
    console.log(`  Warm SidePanel Reopen: ready=${warmReopenReady}`);
    assert(warmReopenReady, 'Model did not stay warm in Offscreen after SidePanel close.');

    consoleErrors = [...session.consoleErrors];
    offscreenErrors = reopenedSnap.logs.filter((l) => l.level === 'error');
  } finally {
    await session.context.close();
  }

  // Fresh browser launch measuring warm OPFS `.bin` slab load -> GPU
  console.log('  Closed Chromium completely. Relaunching to measure cached WGSL OPFS -> VRAM load...');
  session = await launchExtensionPage();
  let coldLoad;
  try {
    const reloadSnap = await loadModelWithConfig(session.page, cfg, 5 * 60 * 1000);
    coldLoad = {
      opfsReloadMs: reloadSnap.telemetry.coldLoadMs,
      loadedFromCache: reloadSnap.telemetry.loadedFromCache,
      loadBreakdown: reloadSnap.telemetry.loadBreakdown,
    };
    console.log('  Cached OPFS -> VRAM Reload:', JSON.stringify(coldLoad));
    assert(coldLoad.loadedFromCache === true, `Expected loadedFromCache=true on second launch, got ${coldLoad.loadedFromCache}`);

    consoleErrors.push(...session.consoleErrors);
    offscreenErrors.push(...reloadSnap.logs.filter((l) => l.level === 'error'));
  } finally {
    await session.context.close();
  }

  const payload = {
    runId,
    track: 'track2-wgsl-real',
    recordedAt: new Date().toISOString(),
    label: 'Real Qwen3.5-4B WGSL (INT8 KV): Repack, Warm Load, Chat, Prefix Reuse, Snapshot & Abort',
    config: cfg,
    initialLoad,
    coldLoad,
    turn1,
    turn2,
    snapshotRestore,
    stopMetrics,
    warmReopenReady,
    consoleErrorCount: consoleErrors.length,
    offscreenErrors,
    status: consoleErrors.length === 0 && offscreenErrors.length === 0 ? 'ok' : 'error',
  };
  saveRun(runId, payload);
  assert(payload.status === 'ok', `Run ${runId} had console/offscreen errors.`);
  return payload;
}

async function runSweepAndNiahConfig({ runId, label, cfg, contexts, decodeTokens }) {
  console.log(`\n=== [${runId}] ${label} ===`);
  const session = await launchExtensionPage();
  let finalPayload;
  try {
    const loadSnap = await loadModelWithConfig(session.page, cfg, 5 * 60 * 1000);
    const coldLoad = {
      opfsReloadMs: loadSnap.telemetry.coldLoadMs,
      loadedFromCache: loadSnap.telemetry.loadedFromCache,
      loadBreakdown: loadSnap.telemetry.loadBreakdown,
    };
    console.log(`  OPFS -> VRAM Load: ${(coldLoad.opfsReloadMs / 1000).toFixed(2)}s`, JSON.stringify(coldLoad.loadBreakdown));

    const sweepSnap = await runSweep(session.page, contexts, decodeTokens, cfg.chunkSize);
    const sweep = sweepSnap.lastBenchmark;
    for (const pt of sweep.points) {
      console.log(
        `  ctx=${pt.targetTokens}: prefill=${pt.prefillTokPerSec.toFixed(1)} tok/s, decode=${pt.decodeTokPerSec.toFixed(2)} tok/s (${pt.bandwidthUtilizationPct.toFixed(1)}% BW, VRAM=${pt.estVramMb.toFixed(0)} MiB) [${pt.status}]`,
      );
    }

    const niahSnap = await runNiah(session.page, 32768, 0.5, cfg.chunkSize);
    const niah = niahSnap.lastNiah;
    console.log(
      `  32K NIAH: passed=${niah.passed}, prefill=${niah.prefillTokPerSec.toFixed(1)} tok/s, decode=${niah.decodeTokPerSec.toFixed(2)} tok/s, response=${JSON.stringify(niah.responseText)}`,
    );

    const finalSnap = await getSnapshot(session.page);
    const offscreenErrors = finalSnap.logs.filter((l) => l.level === 'error');
    const allOk =
      sweep.points.every((p) => p.status === 'ok') &&
      niah.passed === true &&
      session.consoleErrors.length === 0 &&
      offscreenErrors.length === 0;

    finalPayload = {
      runId,
      track: 'track2-wgsl-real',
      recordedAt: new Date().toISOString(),
      label,
      config: finalSnap.config,
      coldLoad,
      sweep,
      niah,
      consoleErrorCount: session.consoleErrors.length,
      offscreenErrors,
      status: allOk ? 'ok' : 'error',
    };
    saveRun(runId, finalPayload);
    assert(allOk, `Run ${runId} failed assertions (sweep/NIAH/errors).`);
    return finalPayload;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`  [${runId}] FAILED: ${msg}`);
    if (!finalPayload) {
      saveRun(runId, {
        runId,
        track: 'track2-wgsl-real',
        recordedAt: new Date().toISOString(),
        label,
        config: cfg,
        consoleErrorCount: session.consoleErrors.length,
        status: 'error',
        error: msg,
      });
    }
    throw err;
  } finally {
    await session.context.close();
  }
}

async function main() {
  const r14 = await run14ChatAndLoad();

  const r15 = await runSweepAndNiahConfig({
    runId: '15-wgsl-real-fp16kv',
    label: 'Real Qwen3.5-4B WGSL (FP16 KV, 1K-64K Sweep + 32K NIAH)',
    cfg: {
      backend: 'wgsl',
      wgslKvMode: 'fp16',
      wgslMaxContext: 65536,
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: true,
      chunkSize: 64,
      enableThinking: false,
      temperature: 0,
    },
    contexts: [1024, 8192, 32768, 65536],
    decodeTokens: 32,
  });

  const r16 = await runSweepAndNiahConfig({
    runId: '16-wgsl-real-int8kv-128k',
    label: 'Real Qwen3.5-4B WGSL (INT8 KV, 1K-128K Sweep + 32K NIAH)',
    cfg: {
      backend: 'wgsl',
      wgslKvMode: 'int8',
      wgslMaxContext: 131072,
      yieldMode: 'none',
      fastDecodePath: true,
      fastOpfsLoad: true,
      chunkSize: 64,
      enableThinking: false,
      temperature: 0,
    },
    contexts: [1024, 8192, 32768, 65536, 131072],
    decodeTokens: 24,
  });

  console.log(
    '\n[Track 2 Real Complete] Statuses:',
    [r14, r15, r16].map((r) => `${r.runId}:${r.status}`).join(', '),
  );
}

main().catch((err) => {
  console.error('[Track 2 Real Fatal]', err);
  process.exit(1);
});
