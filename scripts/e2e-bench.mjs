import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile');
const BENCH_DIR = path.resolve('benchmarks');

function parseArgs(argv) {
  const opts = {
    stage: 'all',
    spikeMib: 2560,
    shardMib: 256,
    contexts: [1024, 8192, 32768, 65536],
    decodeTokens: 32,
    niahTokens: 32768,
    niahDepth: 0.5,
    cleanProfile: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--stage' && argv[i + 1]) {
      opts.stage = argv[++i];
    } else if (arg === '--spike-mib' && argv[i + 1]) {
      opts.spikeMib = Number(argv[++i]);
    } else if (arg === '--shard-mib' && argv[i + 1]) {
      opts.shardMib = Number(argv[++i]);
    } else if (arg === '--contexts' && argv[i + 1]) {
      opts.contexts = argv[++i]
        .split(',')
        .map((n) => Number(n.trim()))
        .filter((n) => Number.isFinite(n) && n > 0);
    } else if (arg === '--decode-tokens' && argv[i + 1]) {
      opts.decodeTokens = Number(argv[++i]);
    } else if (arg === '--niah-tokens' && argv[i + 1]) {
      opts.niahTokens = Number(argv[++i]);
    } else if (arg === '--niah-depth' && argv[i + 1]) {
      opts.niahDepth = Number(argv[++i]);
    } else if (arg === '--clean-profile') {
      opts.cleanProfile = true;
    }
  }
  return opts;
}

function logSection(title) {
  console.log(`\n============================================================`);
  console.log(`[TinyGPT E2E] ${title}`);
  console.log(`============================================================`);
}

function assert(cond, message) {
  if (!cond) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

async function launchExtensionBrowser() {
  if (!fs.existsSync(path.join(DIST_DIR, 'manifest.json'))) {
    throw new Error(`Missing ${path.join(DIST_DIR, 'manifest.json')}. Run "npm run build" first.`);
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

  const consoleErrors = [];
  const pageErrors = [];
  const offscreenLogs = [];

  let sw = context.serviceWorkers()[0];
  if (!sw) {
    sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
  }
  const extensionId = new URL(sw.url()).hostname;

  try {
    sw.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(`[SW] ${msg.text()}`);
      }
    });
  } catch {
    // Worker console subscription may vary by Playwright version.
  }

  // Try optional browser CDP attachment to watch offscreen.html console/exceptions.
  try {
    const browser = context.browser();
    if (browser && typeof browser.newBrowserCDPSession === 'function') {
      const cdp = await browser.newBrowserCDPSession();
      await cdp.send('Target.setDiscoverTargets', { discover: true });
      const { targetInfos } = await cdp.send('Target.getTargets');
      for (const t of targetInfos) {
        if (t.url.includes(extensionId) && t.url.includes('offscreen.html')) {
          const { sessionId } = await cdp.send('Target.attachToTarget', {
            targetId: t.targetId,
            flatten: true,
          });
          await cdp.send('Runtime.enable', {}, sessionId);
        }
      }
    }
  } catch {
    // Offscreen logs are also mirrored via __TINYGPT_CLIENT__.getSnapshot().logs.
  }

  const blankPage = context.pages()[0] ?? (await context.newPage());
  const sidePanelUrl = `chrome-extension://${extensionId}/sidepanel.html`;

  async function openSidePanelTab() {
    const page = await context.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(`[SidePanel] ${msg.text()}`);
      }
    });
    page.on('pageerror', (err) => {
      pageErrors.push(`[SidePanel PageError] ${err.message}`);
    });
    await page.goto(sidePanelUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => Boolean(window.__TINYGPT_CLIENT__ && window.__TINYGPT_CLIENT__.getSnapshot().connected),
      { timeout: 25000 },
    );
    return page;
  }

  const page = await openSidePanelTab();

  return {
    context,
    extensionId,
    blankPage,
    page,
    openSidePanelTab,
    consoleErrors,
    pageErrors,
    offscreenLogs,
  };
}

async function getSnapshot(page) {
  return page.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());
}

async function waitForSnapshot(page, predicate, { timeoutMs = 60000, pollMs = 500, onTick } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastTickLog = 0;
  while (Date.now() < deadline) {
    const snap = await getSnapshot(page);
    if (onTick && Date.now() - lastTickLog >= 3000) {
      onTick(snap);
      lastTickLog = Date.now();
    }
    if (predicate(snap)) {
      return snap;
    }
    await new Promise((r) => setTimeout(r, pollMs));
  }
  const finalSnap = await getSnapshot(page);
  throw new Error(
    `Timed out after ${timeoutMs}ms waiting for condition. Last telemetry: ${JSON.stringify(finalSnap.telemetry)}`,
  );
}

async function openLabTab(page, tabName) {
  const labBtn = page.locator('header.telemetry-bar button', { hasText: /^(Lab|Close Lab)$/ });
  const text = (await labBtn.textContent())?.trim();
  if (text === 'Lab') {
    await labBtn.click();
  }
  await page.locator('.lab-tabs button', { hasText: tabName }).click();
}

async function ensureModelReady(page, timeoutMs = 25 * 60 * 1000) {
  let snap = await getSnapshot(page);
  if (snap.telemetry.status === 'ready') {
    return snap;
  }
  if (snap.telemetry.status === 'unloaded' || snap.telemetry.status === 'error') {
    await page.locator('header.telemetry-bar button', { hasText: 'Load Model' }).click();
  }

  let lastSummary = '';
  snap = await waitForSnapshot(
    page,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`Model load failed: ${s.telemetry.lastError}`);
      }
      return s.telemetry.status === 'ready';
    },
    {
      timeoutMs,
      pollMs: 1000,
      onTick: (s) => {
        const line = `[Load] status=${s.telemetry.status} pct=${s.telemetry.loadProgressPct.toFixed(1)}% phase="${s.telemetry.loadPhaseLabel}"`;
        if (line !== lastSummary) {
          console.log(line);
          lastSummary = line;
        }
      },
    },
  );
  return snap;
}

function collectOffscreenErrors(snap) {
  return snap.logs.filter((l) => l.level === 'error');
}

async function runPhase0(opts) {
  logSection(`PHASE 0: WebGPU Limits & ${opts.spikeMib} MiB OPFS Persistence Across Restart`);
  let session = await launchExtensionBrowser();
  let probeInfo;
  let writeResult;

  try {
    await openLabTab(session.page, 'Phase 0 Spike');

    const beforeSnap = await getSnapshot(session.page);
    const prevProbedAt = beforeSnap.probeInfo?.probedAt ?? null;
    await session.page.locator('button', { hasText: 'Probe WebGPU Limits' }).click();
    const probedSnap = await waitForSnapshot(
      session.page,
      (s) => s.probeInfo && s.probeInfo.probedAt !== prevProbedAt,
      { timeoutMs: 20000 },
    );
    probeInfo = probedSnap.probeInfo;
    console.log('WebGPU Probe:', JSON.stringify(probeInfo, null, 2));
    assert(probeInfo.supported, `WebGPU unsupported: ${probeInfo.error}`);
    const adapterText = `${probeInfo.adapterInfo?.vendor} ${probeInfo.adapterInfo?.description}`;
    assert(!/swiftshader|llvmpipe|software/i.test(adapterText), `Software WebGPU adapter detected: ${adapterText}`);
    assert(
      (probeInfo.grantedLimits?.maxBufferSize ?? 0) >= 1024 * 1024 * 1024,
      `Elevated maxBufferSize too small: ${probeInfo.grantedLimits?.maxBufferSize}`,
    );

    if (opts.spikeMib === 2560 && opts.shardMib === 256) {
      await session.page.locator('button', { hasText: 'Run 2.5 GiB OPFS Spike' }).click();
    } else {
      await session.page.evaluate(
        ({ totalMiB, shardMiB }) =>
          window.__TINYGPT_CLIENT__.send({ type: 'RUN_OPFS_SPIKE', totalMiB, shardMiB }),
        { totalMiB: opts.spikeMib, shardMiB: opts.shardMib },
      );
    }

    const writeSnap = await waitForSnapshot(
      session.page,
      (s) => s.spikeResult && s.spikeResult.action === 'write',
      {
        timeoutMs: 10 * 60 * 1000,
        pollMs: 1000,
        onTick: (s) => {
          if (s.spikeProgress) {
            console.log(`[OPFS Write] ${s.spikeProgress.message} (${s.spikeProgress.pct.toFixed(0)}%)`);
          }
        },
      },
    );
    writeResult = writeSnap.spikeResult;
    console.log('OPFS Write Result:', JSON.stringify(writeResult, null, 2));
    assert(writeResult.ok, `OPFS write failed: ${writeResult.details}`);
  } finally {
    await session.context.close();
  }

  console.log('\n[Phase 0] Closed Chromium completely. Relaunching to verify OPFS across restart...');
  session = await launchExtensionBrowser();
  let verifyResult;
  let opfsStatus;
  let offscreenErrors = [];

  try {
    await openLabTab(session.page, 'Phase 0 Spike');
    await session.page.locator('button', { hasText: 'Verify After Restart' }).click();

    const verifySnap = await waitForSnapshot(
      session.page,
      (s) => s.spikeResult && s.spikeResult.action === 'verify',
      {
        timeoutMs: 120000,
        pollMs: 500,
        onTick: (s) => {
          if (s.spikeProgress) {
            console.log(`[OPFS Verify] ${s.spikeProgress.message} (${s.spikeProgress.pct.toFixed(0)}%)`);
          }
        },
      },
    );
    verifyResult = verifySnap.spikeResult;
    opfsStatus = verifySnap.opfsStatus;
    console.log('OPFS Verify Result:', JSON.stringify(verifyResult, null, 2));
    assert(verifyResult.ok, `OPFS persistence verification failed: ${verifyResult.details}`);

    await session.page.locator('button', { hasText: 'Clear Spike Files' }).click();
    const clearedSnap = await waitForSnapshot(
      session.page,
      (s) => s.spikeResult && s.spikeResult.action === 'clear',
      { timeoutMs: 20000 },
    );
    offscreenErrors = collectOffscreenErrors(clearedSnap);

    assert(session.consoleErrors.length === 0, `SidePanel console errors: ${session.consoleErrors.join('; ')}`);
    assert(session.pageErrors.length === 0, `SidePanel page errors: ${session.pageErrors.join('; ')}`);
    assert(offscreenErrors.length === 0, `Offscreen errors: ${offscreenErrors.map((e) => e.message).join('; ')}`);
  } finally {
    await session.context.close();
  }

  const out = {
    probeInfo,
    writeResult,
    verifyResult,
    opfsStatus,
    consoleErrorCount: session.consoleErrors.length + session.pageErrors.length + offscreenErrors.length,
  };
  fs.mkdirSync(BENCH_DIR, { recursive: true });
  fs.writeFileSync(path.join(BENCH_DIR, 'phase0.json'), JSON.stringify(out, null, 2));
  return out;
}

async function runPhase1() {
  logSection('PHASE 1: Load Qwen3.5-4B, Streaming Chat, Stop Cancel, Warm Offscreen, & OPFS Reload');
  let session = await launchExtensionBrowser();
  let firstLoadMs = 0;
  let firstLoadedFromCache = null;
  let chatMetrics = {};
  let stopMetrics = {};
  let warmReopenReady = false;

  try {
    const readySnap = await ensureModelReady(session.page);
    firstLoadMs = readySnap.telemetry.coldLoadMs ?? 0;
    firstLoadedFromCache = readySnap.telemetry.loadedFromCache;
    console.log(
      `[Phase 1] Model ready in ${(firstLoadMs / 1000).toFixed(2)}s (loadedFromCache=${firstLoadedFromCache})`,
    );

    // 1. Streaming chat test
    const prompt1 = 'List 3 benefits of hybrid linear + full attention in 3 short bullets.';
    await session.page.locator('.composer-input').fill(prompt1);
    await session.page.locator('.composer-actions button', { hasText: 'Send' }).click();

    await waitForSnapshot(
      session.page,
      (s) => s.telemetry.status === 'generating' || s.telemetry.activeContextTokens > 0,
      { timeoutMs: 60000, pollMs: 150 },
    );

    const postChatSnap = await waitForSnapshot(
      session.page,
      (s) => {
        if (s.telemetry.status === 'error') {
          throw new Error(`Chat failed: ${s.telemetry.lastError}`);
        }
        return s.telemetry.status === 'ready';
      },
      { timeoutMs: 180000, pollMs: 300 },
    );

    const responseText = await session.page.locator('.msg-assistant .msg-content').last().innerText();
    chatMetrics = {
      prompt: prompt1,
      responsePreview: responseText.slice(0, 300),
      responseLength: responseText.length,
      ttftMs: postChatSnap.telemetry.ttftMs,
      prefillTokPerSec: postChatSnap.telemetry.prefillTokPerSec,
      decodeTokPerSec: postChatSnap.telemetry.tokensPerSec,
      activeContextTokens: postChatSnap.telemetry.activeContextTokens,
      estVramMb: postChatSnap.telemetry.estVramMb,
    };
    console.log('[Phase 1] Streaming chat result:', JSON.stringify(chatMetrics, null, 2));
    assert(responseText.trim().length > 0, 'Assistant produced empty response.');

    // 2. Stop/Cancel (AbortSignal) test
    const prompt2 =
      'Write a very detailed 1500-word technical essay explaining WebGPU compute shaders, workgroup shared memory, and subgroup matrix operations.';
    await session.page.locator('.composer-input').fill(prompt2);
    await session.page.locator('.composer-actions button', { hasText: 'Send' }).click();

    await waitForSnapshot(
      session.page,
      (s) => s.telemetry.status === 'generating',
      { timeoutMs: 30000, pollMs: 100 },
    );

    // Wait until a few tokens stream into the second assistant message before clicking Stop.
    await session.page.waitForFunction(
      () => {
        const msgs = document.querySelectorAll('.msg-assistant .msg-content');
        return msgs.length >= 2 && (msgs[msgs.length - 1].textContent?.length ?? 0) >= 24;
      },
      { timeout: 60000 },
    );

    const stopBtn = session.page.locator('.composer-actions button', { hasText: 'Stop' });
    await stopBtn.click();

    const postStopSnap = await waitForSnapshot(
      session.page,
      (s) => s.telemetry.status === 'ready',
      { timeoutMs: 15000, pollMs: 150 },
    );
    const partialText = await session.page.locator('.msg-assistant .msg-content').last().innerText();
    stopMetrics = {
      stoppedCleanly: postStopSnap.telemetry.status === 'ready',
      preservedPartialChars: partialText.length,
      partialPreview: partialText.slice(0, 180),
    };
    console.log('[Phase 1] Stop/Abort result:', JSON.stringify(stopMetrics, null, 2));
    assert(partialText.length > 0, 'Partial text disappeared after Stop.');

    // 3. Warm Offscreen persistence across Side Panel close & reopen
    await session.page.close();
    const reopenedPage = await session.openSidePanelTab();
    const reopenedSnap = await waitForSnapshot(
      reopenedPage,
      (s) => s.connected && s.telemetry.status === 'ready',
      { timeoutMs: 10000, pollMs: 150 },
    );
    warmReopenReady = reopenedSnap.telemetry.status === 'ready';
    console.log(`[Phase 1] Side Panel reopen while Chrome kept open: status=${reopenedSnap.telemetry.status}`);
    assert(warmReopenReady, 'Model did not stay warm in Offscreen Document after closing Side Panel.');
  } finally {
    await session.context.close();
  }

  // 4. Restart Chromium completely and measure OPFS warm reload time
  console.log('\n[Phase 1] Closed Chromium completely. Measuring OPFS reload time after restart...');
  session = await launchExtensionBrowser();
  let opfsReloadMs = 0;
  let opfsLoadedFromCache = null;
  let offscreenErrors = [];

  try {
    const reloadSnap = await ensureModelReady(session.page, 5 * 60 * 1000);
    opfsReloadMs = reloadSnap.telemetry.coldLoadMs ?? 0;
    opfsLoadedFromCache = reloadSnap.telemetry.loadedFromCache;
    offscreenErrors = collectOffscreenErrors(reloadSnap);
    console.log(
      `[Phase 1] OPFS reload after restart: ${(opfsReloadMs / 1000).toFixed(2)}s (loadedFromCache=${opfsLoadedFromCache})`,
    );
    assert(opfsLoadedFromCache === true, `Expected OPFS cache hit on restart reload, got ${opfsLoadedFromCache}`);
  } finally {
    await session.context.close();
  }

  const out = {
    firstLoadMs,
    firstLoadedFromCache,
    opfsReloadMs,
    opfsLoadedFromCache,
    warmReopenReady,
    chatMetrics,
    stopMetrics,
    offscreenErrorCount: offscreenErrors.length,
  };
  fs.mkdirSync(BENCH_DIR, { recursive: true });
  fs.writeFileSync(path.join(BENCH_DIR, 'phase1.json'), JSON.stringify(out, null, 2));
  return out;
}

async function runPhase2(opts) {
  logSection(
    `PHASE 2: Context Sweep [${opts.contexts.join(', ')}] + ${opts.niahTokens}-token NIAH`,
  );
  const session = await launchExtensionBrowser();
  let benchmarkResult = null;
  let niahResult = null;
  let offscreenErrors = [];

  try {
    await ensureModelReady(session.page);
    await openLabTab(session.page, 'Bench & NIAH');

    const beforeBench = await getSnapshot(session.page);
    const prevBenchAt = beforeBench.lastBenchmark?.finishedAt ?? null;

    await session.page.evaluate(
      ({ contexts, decodeTokens }) =>
        window.__TINYGPT_CLIENT__.send({
          type: 'RUN_BENCHMARK_SUITE',
          contextLengths: contexts,
          decodeTokens,
        }),
      { contexts: opts.contexts, decodeTokens: opts.decodeTokens },
    );

    let lastBenchMsg = '';
    const benchSnap = await waitForSnapshot(
      session.page,
      (s) => s.lastBenchmark && s.lastBenchmark.finishedAt !== prevBenchAt,
      {
        timeoutMs: 45 * 60 * 1000,
        pollMs: 1000,
        onTick: (s) => {
          const chunk = s.telemetry.prefillChunkProgress
            ? ` (chunk ${s.telemetry.prefillChunkProgress.done}/${s.telemetry.prefillChunkProgress.total})`
            : '';
          const msg = `${s.benchmarkProgress ?? 'running'}${chunk}`;
          if (msg !== lastBenchMsg) {
            console.log(`[Bench] ${msg}`);
            lastBenchMsg = msg;
          }
        },
      },
    );
    benchmarkResult = benchSnap.lastBenchmark;
    console.log('\n[Phase 2] Benchmark Sweep Result:', JSON.stringify(benchmarkResult, null, 2));

    // Run 32K Needle-in-a-Haystack test
    const prevNiahJson = JSON.stringify(benchSnap.lastNiah ?? null);
    await session.page.evaluate(
      ({ targetTokens, depthRatio }) =>
        window.__TINYGPT_CLIENT__.send({
          type: 'RUN_NIAH',
          targetTokens,
          depthRatio,
        }),
      { targetTokens: opts.niahTokens, depthRatio: opts.niahDepth },
    );

    let lastNiahMsg = '';
    const niahSnap = await waitForSnapshot(
      session.page,
      (s) => s.lastNiah && JSON.stringify(s.lastNiah) !== prevNiahJson,
      {
        timeoutMs: 25 * 60 * 1000,
        pollMs: 1000,
        onTick: (s) => {
          const chunk = s.telemetry.prefillChunkProgress
            ? ` (chunk ${s.telemetry.prefillChunkProgress.done}/${s.telemetry.prefillChunkProgress.total})`
            : '';
          const msg = `${s.niahProgress ?? 'running'}${chunk}`;
          if (msg !== lastNiahMsg) {
            console.log(`[NIAH] ${msg}`);
            lastNiahMsg = msg;
          }
        },
      },
    );
    niahResult = niahSnap.lastNiah;
    offscreenErrors = collectOffscreenErrors(niahSnap);
    console.log('\n[Phase 2] NIAH Result:', JSON.stringify(niahResult, null, 2));
  } finally {
    await session.context.close();
  }

  const out = {
    benchmarkResult,
    niahResult,
    offscreenErrors,
  };
  fs.mkdirSync(BENCH_DIR, { recursive: true });
  fs.writeFileSync(path.join(BENCH_DIR, 'phase2.json'), JSON.stringify(out, null, 2));
  return out;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.cleanProfile && fs.existsSync(PROFILE_DIR)) {
    fs.rmSync(PROFILE_DIR, { recursive: true, force: true });
  }

  const summary = {
    startedAt: new Date().toISOString(),
    stage: opts.stage,
  };

  if (opts.stage === 'phase0' || opts.stage === 'all') {
    summary.phase0 = await runPhase0(opts);
  }
  if (opts.stage === 'phase1' || opts.stage === 'all') {
    summary.phase1 = await runPhase1();
  }
  if (opts.stage === 'phase2' || opts.stage === 'all') {
    summary.phase2 = await runPhase2(opts);
  }

  summary.finishedAt = new Date().toISOString();
  fs.mkdirSync(BENCH_DIR, { recursive: true });
  fs.writeFileSync(path.join(BENCH_DIR, 'latest.json'), JSON.stringify(summary, null, 2));

  logSection('E2E SUITE COMPLETE');
  console.log(`Saved full report to ${path.join(BENCH_DIR, 'latest.json')}`);
}

main().catch((err) => {
  console.error('\n[TinyGPT E2E FAILED]', err);
  process.exit(1);
});
