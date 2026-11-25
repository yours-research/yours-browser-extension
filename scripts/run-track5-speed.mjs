import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile-track5');
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

async function cleanProfileDir() {
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    try {
      if (fs.existsSync(PROFILE_DIR)) {
        fs.rmSync(PROFILE_DIR, {
          recursive: true,
          force: true,
          maxRetries: 5,
          retryDelay: 200,
        });
      }
      fs.mkdirSync(PROFILE_DIR, { recursive: true });
      return;
    } catch (err) {
      if (attempt === 10) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 300));
    }
  }
}

function installFpsProbe() {
  if (window.__FPS_PROBE__) return;
  let running = false;
  let rafId = 0;
  let lastTs = null;
  let frames = [];
  let inputLatencies = [];

  const round = (val, digits = 2) => Number(val.toFixed(digits));
  const percentile = (sorted, p) => {
    if (sorted.length === 0) return 0;
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
    return sorted[idx];
  };

  function tick() {
    if (!running) return;
    const now = performance.now();
    if (lastTs !== null) {
      frames.push(now - lastTs);
    }
    lastTs = now;
    rafId = requestAnimationFrame(tick);
  }

  window.__FPS_PROBE__ = {
    start() {
      running = true;
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      lastTs = null;
      frames = [];
      inputLatencies = [];
      rafId = requestAnimationFrame(tick);
    },
    stop() {
      running = false;
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      const avgInputLatencyMs =
        inputLatencies.length > 0
          ? round(inputLatencies.reduce((a, b) => a + b, 0) / inputLatencies.length, 2)
          : 0;
      if (frames.length === 0) {
        return {
          avgFps: 0,
          p50FrameMs: 0,
          p95FrameMs: 0,
          p99FrameMs: 0,
          maxFrameMs: 0,
          jankPctOver33ms: 0,
          freezePctOver100ms: 0,
          avgInputLatencyMs,
          frameCount: 0,
        };
      }
      const sorted = [...frames].sort((a, b) => a - b);
      const totalMs = frames.reduce((a, b) => a + b, 0);
      const avgFps = totalMs > 0 ? (frames.length * 1000) / totalMs : 0;
      const jankCount = frames.filter((dt) => dt > 33).length;
      const freezeCount = frames.filter((dt) => dt > 100).length;
      return {
        avgFps: round(avgFps, 2),
        p50FrameMs: round(percentile(sorted, 0.5), 2),
        p95FrameMs: round(percentile(sorted, 0.95), 2),
        p99FrameMs: round(percentile(sorted, 0.99), 2),
        maxFrameMs: round(sorted[sorted.length - 1], 2),
        jankPctOver33ms: round((jankCount / frames.length) * 100, 2),
        freezePctOver100ms: round((freezeCount / frames.length) * 100, 2),
        avgInputLatencyMs,
        frameCount: frames.length,
      };
    },
    measureScrollAndInput() {
      const scroller =
        document.getElementById('scroller') ?? document.querySelector('.thread-viewport');
      const btn = document.getElementById('probe-btn');
      const tInput = performance.now();
      if (scroller) {
        const maxScroll = Math.max(1, scroller.scrollHeight - scroller.clientHeight);
        scroller.scrollTop = (scroller.scrollTop + 120) % maxScroll;
      }
      if (btn) {
        btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      }
      return new Promise((resolve) => {
        requestAnimationFrame(() => {
          const tPaint = performance.now();
          if (running) {
            inputLatencies.push(Math.max(0, tPaint - tInput));
          }
          resolve();
        });
      });
    },
  };
}

function buildWebPageHtml() {
  const paragraphs = Array.from(
    { length: 60 },
    (_, i) =>
      `<p>Scroll stress paragraph ${i + 1}: verifying continuous 60Hz browser compositing, canvas rendering, and responsive main-thread scrolling during local WebGPU inference.</p>`,
  ).join('\n');

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>TinyGPT Track 5 UI Smoothness Probe</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        padding: 20px;
        font-family: system-ui, -apple-system, sans-serif;
        background: #0b0f17;
        color: #e2e8f0;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .wave-banner {
        height: 56px;
        border-radius: 10px;
        padding: 0 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
        background: linear-gradient(90deg, #0ea5e9, #6366f1, #ec4899, #0ea5e9);
        background-size: 300% 100%;
        animation: waveShift 3s linear infinite;
      }
      @keyframes waveShift {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
      }
      .stage {
        display: flex;
        align-items: center;
        gap: 16px;
      }
      #orbital-canvas {
        border: 1px solid #263248;
        border-radius: 8px;
        background: #131a26;
      }
      #probe-btn {
        background: #38bdf8;
        color: #082f49;
        border: none;
        border-radius: 8px;
        padding: 12px 18px;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
      }
      #scroller {
        height: 300px;
        overflow-y: auto;
        border: 1px solid #263248;
        border-radius: 8px;
        background: #131a26;
        padding: 12px 16px;
        line-height: 1.5;
      }
      #scroller p {
        margin: 0 0 10px;
        color: #cbd5e1;
      }
    </style>
  </head>
  <body>
    <div class="wave-banner">
      <span>Web Compositor + CSS Gradient Wave Animation</span>
      <span id="frame-badge">60Hz Probe Active</span>
    </div>
    <div class="stage">
      <canvas id="orbital-canvas" width="420" height="160"></canvas>
      <button id="probe-btn" type="button">Counter: 0</button>
    </div>
    <div id="scroller">
      ${paragraphs}
    </div>
    <script>
      (() => {
        let clickCount = 0;
        const btn = document.getElementById('probe-btn');
        btn.addEventListener('click', () => {
          clickCount += 1;
          btn.textContent = 'Counter: ' + clickCount;
        });

        const canvas = document.getElementById('orbital-canvas');
        const ctx = canvas.getContext('2d');
        let angle = 0;
        function renderLoop(now) {
          angle = (now * 0.003) % (Math.PI * 2);
          ctx.fillStyle = '#131a26';
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          const cx = canvas.width / 2;
          const cy = canvas.height / 2;
          ctx.strokeStyle = '#334155';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, 52, 0, Math.PI * 2);
          ctx.stroke();

          const ox = cx + Math.cos(angle) * 52;
          const oy = cy + Math.sin(angle) * 52;
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(ox, oy, 10, 0, Math.PI * 2);
          ctx.fill();

          const barWidth = ((Math.sin(angle * 2) + 1) / 2) * (canvas.width - 40);
          ctx.fillStyle = '#22c55e';
          ctx.fillRect(20, canvas.height - 16, barWidth, 6);

          requestAnimationFrame(renderLoop);
        }
        requestAnimationFrame(renderLoop);
      })();
      (${installFpsProbe.toString()})();
    </script>
  </body>
</html>`;
}

async function launchIsolatedSession(withWebPage = false) {
  if (!fs.existsSync(path.join(DIST_DIR, 'manifest.json'))) {
    throw new Error('dist/manifest.json missing. Run npm run build first.');
  }
  await cleanProfileDir();

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

  const externalNetworkRequests = [];
  await context.route(/^https?:\/\//, (route) => {
    externalNetworkRequests.push(route.request().url());
    void route.abort();
  });

  let sw = context.serviceWorkers()[0];
  if (!sw) {
    sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
  }
  const extensionId = new URL(sw.url()).hostname;
  const sidePanelUrl = `chrome-extension://${extensionId}/sidepanel.html`;

  const consoleErrors = [];
  const attachErrorListeners = (page, tag) => {
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(`[${tag}] ${msg.text()}`);
      }
    });
    page.on('pageerror', (err) => {
      consoleErrors.push(`[${tag}:PageError] ${err.message}`);
    });
  };

  const sidePanel = await context.newPage();
  attachErrorListeners(sidePanel, 'SidePanel');
  await sidePanel.goto(sidePanelUrl, { waitUntil: 'domcontentloaded' });
  await sidePanel.waitForFunction(
    () => Boolean(window.__TINYGPT_CLIENT__ && window.__TINYGPT_CLIENT__.getSnapshot().connected),
    { timeout: 25000 },
  );

  let webPage = null;
  if (withWebPage) {
    webPage = context.pages().find((p) => p !== sidePanel) ?? (await context.newPage());
    attachErrorListeners(webPage, 'WebPage');
    await webPage.setContent(buildWebPageHtml(), { waitUntil: 'domcontentloaded' });
  }

  return {
    context,
    sidePanel,
    webPage,
    externalNetworkRequests,
    consoleErrors,
  };
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

async function measureSmoothnessDuring(targetPage, asyncWorkFn) {
  await targetPage.bringToFront();
  await new Promise((r) => setTimeout(r, 150));
  await targetPage.evaluate(() => window.__FPS_PROBE__.start());
  const timer = setInterval(() => {
    void targetPage.evaluate(() => window.__FPS_PROBE__.measureScrollAndInput()).catch(() => {});
  }, 250);
  try {
    const result = await asyncWorkFn();
    clearInterval(timer);
    const smoothness = await targetPage.evaluate(() => window.__FPS_PROBE__.stop());
    return { result, smoothness };
  } finally {
    clearInterval(timer);
  }
}

async function loadModelWithConfig(sidePanel, cfg, timeoutMs = 10 * 60 * 1000) {
  await sidePanel.evaluate(
    (c) => window.__TINYGPT_CLIENT__.send({ type: 'LOAD_MODEL', config: c }),
    cfg,
  );
  let lastPhase = '';
  return waitForSnapshot(
    sidePanel,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`Model load failed: ${s.telemetry.lastError}`);
      }
      return (
        s.telemetry.status === 'ready' &&
        s.telemetry.coldLoadMs !== null &&
        s.config.executionHost === cfg.executionHost &&
        s.config.gpuPacingMode === cfg.gpuPacingMode
      );
    },
    {
      timeoutMs,
      pollMs: 250,
      onTick: (s) => {
        const line = `status=${s.telemetry.status} pct=${s.telemetry.loadProgressPct.toFixed(0)}% phase="${s.telemetry.loadPhaseLabel}"`;
        if (line !== lastPhase) {
          console.log(`  [Load] ${line}`);
          lastPhase = line;
        }
      },
    },
  );
}

async function updateConfig(sidePanel, patch) {
  await sidePanel.evaluate(
    (p) => window.__TINYGPT_CLIENT__.send({ type: 'UPDATE_CONFIG', config: p }),
    patch,
  );
  return waitForSnapshot(
    sidePanel,
    (s) => Object.entries(patch).every(([k, v]) => s.config[k] === v),
    { timeoutMs: 15000, pollMs: 50 },
  );
}

async function resetContext(sidePanel) {
  await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.send({ type: 'RESET_CONTEXT' }));
  return waitForSnapshot(
    sidePanel,
    (s) => s.telemetry.activeContextTokens === 0 && s.telemetry.status === 'ready',
    { timeoutMs: 15000, pollMs: 100 },
  );
}

async function runSweep(sidePanel, contexts, decodeTokens, chunkSize, timeoutMs = 45 * 60 * 1000) {
  const prevAt = (await getSnapshot(sidePanel)).lastBenchmark?.finishedAt ?? null;
  await sidePanel.evaluate(
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
    sidePanel,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`Benchmark failed: ${s.telemetry.lastError}`);
      }
      return (
        Boolean(s.lastBenchmark) &&
        s.lastBenchmark.finishedAt !== prevAt &&
        s.telemetry.status === 'ready'
      );
    },
    {
      timeoutMs,
      pollMs: 400,
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

async function runNiah(sidePanel, targetTokens, depthRatio, chunkSize, timeoutMs = 20 * 60 * 1000) {
  const prevJson = JSON.stringify((await getSnapshot(sidePanel)).lastNiah ?? null);
  await sidePanel.evaluate(
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
    sidePanel,
    (s) => {
      if (s.telemetry.status === 'error') {
        throw new Error(`NIAH failed: ${s.telemetry.lastError}`);
      }
      return (
        Boolean(s.lastNiah) &&
        JSON.stringify(s.lastNiah) !== prevJson &&
        s.telemetry.status === 'ready'
      );
    },
    {
      timeoutMs,
      pollMs: 400,
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

async function clickSendAndWaitTurn(sidePanel, prompt, expectedMsgCount, timeoutMs = 5 * 60 * 1000) {
  await sidePanel.locator('.composer-actions button', { hasText: 'Send' }).click();

  await waitForSnapshot(
    sidePanel,
    (s) => s.telemetry.status === 'generating' || s.telemetry.activeContextTokens > 0,
    { timeoutMs: 60000, pollMs: 100 },
  );

  const postSnap = await waitForSnapshot(
    sidePanel,
    (s) => {
      if (s.telemetry.status === 'error' || s.telemetry.lastError) {
        throw new Error(`Chat turn failed: ${s.telemetry.lastError}`);
      }
      return s.telemetry.status === 'ready';
    },
    { timeoutMs, pollMs: 200 },
  );

  await sidePanel.waitForFunction(
    (count) => document.querySelectorAll('.msg-assistant .msg-content').length >= count,
    expectedMsgCount,
    { timeout: 10000 },
  );
  await sidePanel.locator('.composer-actions button', { hasText: 'Stop' }).waitFor({
    state: 'detached',
    timeout: 10000,
  });

  // Wait for @assistant-ui/react streaming renderer to flush all remaining deltas to the DOM
  let responseText = '';
  let stableChecks = 0;
  const flushDeadline = Date.now() + 5000;
  while (Date.now() < flushDeadline) {
    const cur = await sidePanel.locator('.msg-assistant .msg-content').last().innerText();
    if (cur === responseText && cur.length > 0) {
      stableChecks += 1;
      if (stableChecks >= 3) break;
    } else {
      responseText = cur;
      stableChecks = 0;
    }
    await new Promise((r) => setTimeout(r, 100));
  }
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
    speculativeStats: postSnap.telemetry.speculativeStats,
  };
}

async function runChatTurn(sidePanel, prompt, expectedMsgCount, timeoutMs = 5 * 60 * 1000) {
  await sidePanel.locator('.composer-input').fill(prompt);
  return clickSendAndWaitTurn(sidePanel, prompt, expectedMsgCount, timeoutMs);
}

async function runMaxSpeed128k() {
  const runId = '23-track5-wgsl-max-speed-128k';
  const label = 'Track 5 WGSL Speedups (Zero-Barrier Factored GEMV + Fused Dispatches + 4H GQA + Speculative Parity)';
  const cfg = {
    backend: 'wgsl',
    wgslKvMode: 'int8',
    wgslMaxContext: 131072,
    executionHost: 'worker',
    gpuPacingMode: 'max-speed',
    chunkSize: 64,
    enableThinking: false,
    enableSpeculative: false,
    temperature: 0,
    maxNewTokens: 128,
  };

  console.log(`\n=== [${runId}] ${label} ===`);
  const session = await launchIsolatedSession(false);
  let finalPayload;

  try {
    const { sidePanel, externalNetworkRequests, consoleErrors } = session;

    const loadSnap = await loadModelWithConfig(sidePanel, cfg);
    const coldLoad = {
      coldLoadMs: loadSnap.telemetry.coldLoadMs,
      loadSource: loadSnap.telemetry.loadSource,
      executionHost: loadSnap.telemetry.executionHost,
      loadBreakdown: loadSnap.telemetry.loadBreakdown,
      estVramMb: loadSnap.telemetry.estVramMb,
    };
    console.log('  Offline Load:', JSON.stringify(coldLoad));
    assert(coldLoad.loadSource === 'bundled', `Expected loadSource='bundled', got ${coldLoad.loadSource}`);
    assert(coldLoad.executionHost === 'worker', `Expected executionHost='worker', got ${coldLoad.executionHost}`);

    const contexts = [1024, 8192, 32768, 65536, 98304, 131072];
    const sweepSnap = await runSweep(sidePanel, contexts, 32, cfg.chunkSize);
    const sweep = sweepSnap.lastBenchmark;
    for (const pt of sweep.points) {
      console.log(
        `  ctx=${pt.targetTokens}: prefill=${pt.prefillTokPerSec.toFixed(1)} tok/s, decode=${pt.decodeTokPerSec.toFixed(2)} tok/s (${pt.bandwidthUtilizationPct.toFixed(1)}% BW, VRAM=${pt.estVramMb.toFixed(0)} MiB) [${pt.status}]`,
      );
    }

    const niahSnap = await runNiah(sidePanel, 32768, 0.5, cfg.chunkSize);
    const niah = niahSnap.lastNiah;
    console.log(
      `  32K NIAH: passed=${niah.passed}, prefill=${niah.prefillTokPerSec.toFixed(1)} tok/s, decode=${niah.decodeTokPerSec.toFixed(2)} tok/s, response=${JSON.stringify(niah.responseText)}`,
    );
    assert(niah.passed === true, `32K NIAH failed (response="${niah.responseText}").`);

    await resetContext(sidePanel);

    const chatTurn1 = await runChatTurn(
      sidePanel,
      'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.',
      1,
    );
    console.log(
      `  Chat Turn 1 (${chatTurn1.decodeTokPerSec?.toFixed(1)} tok/s, TTFT=${chatTurn1.ttftMs?.toFixed(0)}ms): ${JSON.stringify(chatTurn1.responseText.slice(0, 140))}`,
    );
    assert(chatTurn1.responseText.trim().length >= 30, `Chat Turn 1 response too short: "${chatTurn1.responseText}"`);

    // Speculative Parity & Speedup Check
    const specPrompt =
      'Rewrite and clean up the following config bullet list verbatim in Markdown, keeping every field name and exact numeric limit unchanged:\n' +
      '- maxBufferSize: 2147483648\n' +
      '- maxStorageBufferBindingSize: 2147483644\n' +
      '- workers: DedicatedWorker (es module)\n' +
      '- kvCacheMode: int8 (131072 context)\n' +
      '- weightShards: 21 parts (<= 128 MiB each)\n' +
      '- decodePacing: balanced (144Hz smooth compositor)';

    await resetContext(sidePanel);
    await updateConfig(sidePanel, { enableSpeculative: false, maxNewTokens: 256 });
    const specBaseline = await runChatTurn(sidePanel, specPrompt, 2);

    await resetContext(sidePanel);
    await updateConfig(sidePanel, { enableSpeculative: true, maxNewTokens: 256 });
    const specEnabled = await runChatTurn(sidePanel, specPrompt, 3);

    const parityPassed = specBaseline.responseText === specEnabled.responseText;
    const accepted = specEnabled.speculativeStats?.accepted ?? 0;
    const speculativeCheck = {
      prompt: specPrompt,
      parityPassed,
      baselineDecodeTokPerSec: specBaseline.decodeTokPerSec,
      speculativeDecodeTokPerSec: specEnabled.decodeTokPerSec,
      speedupRatio:
        specBaseline.decodeTokPerSec && specEnabled.decodeTokPerSec
          ? Number((specEnabled.decodeTokPerSec / specBaseline.decodeTokPerSec).toFixed(3))
          : null,
      speculativeStats: specEnabled.speculativeStats,
      baselineLength: specBaseline.responseText.length,
      speculativeLength: specEnabled.responseText.length,
      responsePreview: specEnabled.responseText.slice(0, 200),
    };
    console.log('  Speculative Parity Check:', JSON.stringify(speculativeCheck));
    assert(
      parityPassed,
      `Speculative parity mismatch!\nBaseline: ${JSON.stringify(specBaseline.responseText)}\nSpeculative: ${JSON.stringify(specEnabled.responseText)}`,
    );
    assert(accepted > 0, `Expected speculativeStats.accepted > 0, got ${accepted}`);

    const finalSnap = await getSnapshot(sidePanel);
    const offscreenErrors = finalSnap.logs.filter((l) => l.level === 'error');
    const sweepAllOk = sweep.points.every((p) => p.status === 'ok');

    const allOk =
      externalNetworkRequests.length === 0 &&
      consoleErrors.length === 0 &&
      offscreenErrors.length === 0 &&
      sweepAllOk &&
      niah.passed === true &&
      parityPassed &&
      accepted > 0;

    finalPayload = {
      runId,
      track: 'track5-speed',
      recordedAt: new Date().toISOString(),
      label,
      config: finalSnap.config,
      coldLoad,
      sweep,
      niah,
      chatTurn1,
      speculativeCheck,
      externalNetworkRequestCount: externalNetworkRequests.length,
      externalNetworkRequests,
      consoleErrorCount: consoleErrors.length,
      offscreenErrors,
      status: allOk ? 'ok' : 'error',
    };

    saveRun(runId, finalPayload);

    assert(
      externalNetworkRequests.length === 0,
      `Detected ${externalNetworkRequests.length} external network request(s): ${externalNetworkRequests.join(', ')}`,
    );
    assert(consoleErrors.length === 0, `Detected console errors: ${consoleErrors.join(' | ')}`);
    assert(offscreenErrors.length === 0, `Detected offscreen errors: ${JSON.stringify(offscreenErrors)}`);
    assert(sweepAllOk, 'Not all sweep benchmark points succeeded.');

    return finalPayload;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`  [${runId}] FAILED: ${msg}`);
    if (!finalPayload) {
      saveRun(runId, {
        runId,
        track: 'track5-speed',
        recordedAt: new Date().toISOString(),
        label,
        config: cfg,
        externalNetworkRequestCount: session.externalNetworkRequests.length,
        externalNetworkRequests: session.externalNetworkRequests,
        consoleErrorCount: session.consoleErrors.length,
        offscreenErrors: [],
        status: 'error',
        error: msg,
      });
    }
    throw err;
  } finally {
    await session.context.close();
  }
}

async function runBalancedSmoothness() {
  const runId = '24-track5-wgsl-worker-balanced-smoothness';
  const label = 'Track 5 WGSL Speedups (Dedicated Worker, Balanced Default + Speculative + Extended Suite)';
  const cfg = {
    backend: 'wgsl',
    wgslKvMode: 'int8',
    wgslMaxContext: 131072,
    executionHost: 'worker',
    gpuPacingMode: 'balanced',
    chunkSize: 64,
    enableThinking: false,
    enableSpeculative: true,
    temperature: 0,
    maxNewTokens: 128,
  };

  console.log(`\n=== [${runId}] ${label} ===`);
  const session = await launchIsolatedSession(true);
  let finalPayload;

  try {
    const { sidePanel, webPage, externalNetworkRequests, consoleErrors } = session;

    const { smoothness: idleSmoothness } = await measureSmoothnessDuring(
      webPage,
      () => new Promise((r) => setTimeout(r, 3000)),
    );
    console.log('  Idle Smoothness (3s):', JSON.stringify(idleSmoothness));

    const { result: loadSnap, smoothness: loadSmoothness } = await measureSmoothnessDuring(
      webPage,
      () => loadModelWithConfig(sidePanel, cfg),
    );
    const coldLoad = {
      coldLoadMs: loadSnap.telemetry.coldLoadMs,
      loadSource: loadSnap.telemetry.loadSource,
      executionHost: loadSnap.telemetry.executionHost,
      loadBreakdown: loadSnap.telemetry.loadBreakdown,
      estVramMb: loadSnap.telemetry.estVramMb,
    };
    console.log('  Offline Load:', JSON.stringify(coldLoad));
    console.log('  Load Smoothness:', JSON.stringify(loadSmoothness));
    assert(coldLoad.loadSource === 'bundled', `Expected loadSource='bundled', got ${coldLoad.loadSource}`);
    assert(coldLoad.executionHost === 'worker', `Expected executionHost='worker', got ${coldLoad.executionHost}`);

    const { result: sweepSnap, smoothness: sweepSmoothness } = await measureSmoothnessDuring(
      webPage,
      () => runSweep(sidePanel, [8192, 32768], 32, cfg.chunkSize),
    );
    const sweep = sweepSnap.lastBenchmark;
    for (const pt of sweep.points) {
      console.log(
        `  ctx=${pt.targetTokens}: prefill=${pt.prefillTokPerSec.toFixed(1)} tok/s, decode=${pt.decodeTokPerSec.toFixed(2)} tok/s (${pt.bandwidthUtilizationPct.toFixed(1)}% BW, VRAM=${pt.estVramMb.toFixed(0)} MiB) [${pt.status}]`,
      );
    }
    console.log('  Sweep Smoothness:', JSON.stringify(sweepSmoothness));

    await sidePanel.bringToFront();
    await sidePanel.evaluate(installFpsProbe);
    const turn1Prompt =
      'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.';
    await sidePanel.locator('.composer-input').fill(turn1Prompt);
    const { result: chatTurn1, smoothness: chatSmoothness } = await measureSmoothnessDuring(
      sidePanel,
      () => clickSendAndWaitTurn(sidePanel, turn1Prompt, 1),
    );
    console.log(
      `  Chat Turn 1 (${chatTurn1.decodeTokPerSec?.toFixed(1)} tok/s, TTFT=${chatTurn1.ttftMs?.toFixed(0)}ms): ${JSON.stringify(chatTurn1.responseText.slice(0, 140))}`,
    );
    console.log('  Chat Smoothness:', JSON.stringify(chatSmoothness));
    assert(
      chatTurn1.responseText.trim().length >= 30,
      `Chat Turn 1 response too short: "${chatTurn1.responseText}"`,
    );

    const turn2 = await runChatTurn(sidePanel, 'Summarize bullet 1 in one short sentence.', 2);
    console.log(
      `  Turn 2 (prefixReused=${turn2.prefixReused}, ctx=${turn2.activeContextTokens}, ${turn2.decodeTokPerSec?.toFixed(1)} tok/s): ${JSON.stringify(turn2.responseText.slice(0, 140))}`,
    );
    assert(turn2.prefixReused === true, `Expected prefixReused=true on Turn 2, got ${turn2.prefixReused}`);
    assert(turn2.responseText.trim().length >= 15, `Turn 2 response too short: "${turn2.responseText}"`);

    const ctxBeforeSnap = turn2.activeContextTokens;
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.send({ type: 'SNAPSHOT_CONTEXT' }));
    const snapped = await waitForSnapshot(
      sidePanel,
      (s) => s.telemetry.hasCpuSnapshot === true && s.telemetry.cpuSnapshotTokens === ctxBeforeSnap,
      { timeoutMs: 30000, pollMs: 100 },
    );
    await resetContext(sidePanel);
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.send({ type: 'RESTORE_CONTEXT' }));
    const restored = await waitForSnapshot(
      sidePanel,
      (s) => s.telemetry.activeContextTokens === snapped.telemetry.cpuSnapshotTokens,
      { timeoutMs: 30000, pollMs: 100 },
    );
    const snapshotRestore = {
      snapshottedTokens: snapped.telemetry.cpuSnapshotTokens,
      restoredTokens: restored.telemetry.activeContextTokens,
      passed: restored.telemetry.activeContextTokens === ctxBeforeSnap,
    };
    console.log('  Snapshot/Restore:', JSON.stringify(snapshotRestore));
    assert(snapshotRestore.passed, 'Snapshot/Restore token count mismatch.');

    const prevCount = await sidePanel.locator('.msg-assistant .msg-content').count();
    await sidePanel
      .locator('.composer-input')
      .fill('Write a very detailed 1000-word technical essay on GPU compute shaders.');
    await sidePanel.locator('.composer-actions button', { hasText: 'Send' }).click();
    await sidePanel.waitForFunction(
      (minPrev) => {
        const msgs = document.querySelectorAll('.msg-assistant .msg-content');
        return msgs.length > minPrev && (msgs[msgs.length - 1].textContent?.length ?? 0) >= 20;
      },
      prevCount,
      { timeout: 60000 },
    );
    const tClick = performance.now();
    await sidePanel.locator('.composer-actions button', { hasText: 'Stop' }).click();
    const postStopSnap = await waitForSnapshot(
      sidePanel,
      (s) => s.telemetry.status === 'ready',
      { timeoutMs: 15000, pollMs: 50 },
    );
    const abortLatencyMs = Number((performance.now() - tClick).toFixed(2));
    const partialText = await sidePanel.locator('.msg-assistant .msg-content').last().innerText();
    const abortMetrics = {
      stoppedCleanly: postStopSnap.telemetry.status === 'ready',
      abortLatencyMs,
      preservedPartialChars: partialText.length,
      partialPreview: partialText.slice(0, 140),
    };
    console.log('  Abort Latency:', JSON.stringify(abortMetrics));
    assert(
      abortMetrics.stoppedCleanly && abortMetrics.preservedPartialChars >= 20,
      `Abort did not stop cleanly or lost partial text (chars=${abortMetrics.preservedPartialChars}).`,
    );
    assert(
      abortMetrics.abortLatencyMs <= 200,
      `Abort latency (${abortMetrics.abortLatencyMs} ms) exceeded 200 ms budget.`,
    );

    const niahSnap = await runNiah(sidePanel, 32768, 0.5, cfg.chunkSize);
    const niah = niahSnap.lastNiah;
    console.log(
      `  32K NIAH: passed=${niah.passed}, prefill=${niah.prefillTokPerSec.toFixed(1)} tok/s, decode=${niah.decodeTokPerSec.toFixed(2)} tok/s, response=${JSON.stringify(niah.responseText)}`,
    );
    assert(niah.passed === true, `32K NIAH failed (response="${niah.responseText}").`);

    const finalSnap = await getSnapshot(sidePanel);
    const offscreenErrors = finalSnap.logs.filter((l) => l.level === 'error');
    const sweepAllOk = sweep.points.every((p) => p.status === 'ok');

    const allOk =
      externalNetworkRequests.length === 0 &&
      consoleErrors.length === 0 &&
      offscreenErrors.length === 0 &&
      sweepAllOk &&
      niah.passed === true &&
      sweepSmoothness.jankPctOver33ms <= 1 &&
      chatSmoothness.jankPctOver33ms <= 1;

    finalPayload = {
      runId,
      track: 'track5-speed',
      recordedAt: new Date().toISOString(),
      label,
      config: finalSnap.config,
      idleSmoothness,
      coldLoad,
      loadSmoothness,
      sweep,
      sweepSmoothness,
      chatTurn1,
      chatSmoothness,
      turn2,
      snapshotRestore,
      abortMetrics,
      niah,
      externalNetworkRequestCount: externalNetworkRequests.length,
      externalNetworkRequests,
      consoleErrorCount: consoleErrors.length,
      offscreenErrors,
      status: allOk ? 'ok' : 'error',
    };

    saveRun(runId, finalPayload);

    assert(
      externalNetworkRequests.length === 0,
      `Detected ${externalNetworkRequests.length} external network request(s): ${externalNetworkRequests.join(', ')}`,
    );
    assert(consoleErrors.length === 0, `Detected console errors: ${consoleErrors.join(' | ')}`);
    assert(offscreenErrors.length === 0, `Detected offscreen errors: ${JSON.stringify(offscreenErrors)}`);
    assert(sweepAllOk, 'Not all sweep benchmark points succeeded.');
    assert(
      sweepSmoothness.jankPctOver33ms <= 1,
      `Sweep jankPctOver33ms (${sweepSmoothness.jankPctOver33ms}%) exceeded 1%.`,
    );
    assert(
      chatSmoothness.jankPctOver33ms <= 1,
      `Chat jankPctOver33ms (${chatSmoothness.jankPctOver33ms}%) exceeded 1%.`,
    );

    return finalPayload;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`  [${runId}] FAILED: ${msg}`);
    if (!finalPayload) {
      saveRun(runId, {
        runId,
        track: 'track5-speed',
        recordedAt: new Date().toISOString(),
        label,
        config: cfg,
        externalNetworkRequestCount: session.externalNetworkRequests.length,
        externalNetworkRequests: session.externalNetworkRequests,
        consoleErrorCount: session.consoleErrors.length,
        offscreenErrors: [],
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
  const onlyArg = process.argv.find((a) => a.startsWith('--only='))?.split('=')[1];
  const results = [];
  if (!onlyArg || onlyArg === '23') {
    results.push(await runMaxSpeed128k());
  }
  if (!onlyArg || onlyArg === '24') {
    results.push(await runBalancedSmoothness());
  }
  console.log(
    '\n[Track 5 Speed Complete] Statuses:',
    results.map((r) => `${r.runId}:${r.status}`).join(', '),
  );
}

main().catch((err) => {
  console.error('[Track 5 Speed Fatal]', err);
  process.exit(1);
});
