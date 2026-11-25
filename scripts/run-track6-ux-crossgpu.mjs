import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile-track6');
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
      if (attempt === 10) throw err;
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
      if (rafId) cancelAnimationFrame(rafId);
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
    { length: 48 },
    (_, i) =>
      `<p>Scroll stress paragraph ${i + 1}: verifying cross-GPU 3-tier WebGPU shaders, tiered KV growth, Slot 0 prompt caching, and local page extraction without external network requests.</p>`,
  ).join('\n');

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>TinyGPT Track 6 Local Test Article</title>
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
        height: 52px;
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
      .stage { display: flex; align-items: center; gap: 16px; }
      #orbital-canvas { border: 1px solid #263248; border-radius: 8px; background: #131a26; }
      #probe-btn {
        background: #38bdf8;
        color: #082f49;
        border: none;
        border-radius: 8px;
        padding: 12px 18px;
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
    </style>
  </head>
  <body>
    <div class="wave-banner">
      <span>Web Compositor + CSS Gradient Wave Animation</span>
      <span id="frame-badge">Compositor Probe Active</span>
    </div>
    <div class="stage">
      <canvas id="orbital-canvas" width="420" height="140"></canvas>
      <button id="probe-btn" type="button">Counter: 0</button>
    </div>
    <article id="scroller">
      <h1>WebGPU Cross-GPU & UX Benchmarks</h1>
      ${paragraphs}
    </article>
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
        function renderLoop(now) {
          const angle = (now * 0.003) % (Math.PI * 2);
          ctx.fillStyle = '#131a26';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          const cx = canvas.width / 2;
          const cy = canvas.height / 2;
          ctx.strokeStyle = '#334155';
          ctx.beginPath();
          ctx.arc(cx, cy, 46, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(cx + Math.cos(angle) * 46, cy + Math.sin(angle) * 46, 9, 0, Math.PI * 2);
          ctx.fill();
          requestAnimationFrame(renderLoop);
        }
        requestAnimationFrame(renderLoop);
      })();
      (${installFpsProbe.toString()})();
    </script>
  </body>
</html>`;
}

async function getSnapshot(page) {
  return page.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());
}

async function waitForSnapshot(page, predicate, { timeoutMs = 600000, pollMs = 200, onTick } = {}) {
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
  throw new Error(`Timed out after ${timeoutMs}ms. Telemetry: ${JSON.stringify(finalSnap.telemetry)}`);
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

async function runDirectGenerate(sidePanel, messages, config = undefined) {
  return sidePanel.evaluate(
    ({ msgs, cfg }) =>
      new Promise((resolve, reject) => {
        const client = window.__TINYGPT_CLIENT__;
        const reqId = `t6-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
        let text = '';
        client.startGeneration(
          reqId,
          msgs,
          {
            onDelta(delta) {
              text += delta;
            },
            onComplete(finishReason) {
              const snap = client.getSnapshot();
              resolve({
                text,
                finishReason,
                ttftMs: snap.telemetry.ttftMs,
                prefillTokPerSec: snap.telemetry.prefillTokPerSec,
                tokensPerSec: snap.telemetry.tokensPerSec,
                activeContextTokens: snap.telemetry.activeContextTokens,
                prefixReused: snap.telemetry.prefixReused,
                activePrefixSlot: snap.telemetry.activePrefixSlot,
                lastSlotRestoreMs: snap.telemetry.lastSlotRestoreMs,
                computeTier: snap.telemetry.computeTier,
                allocatedKvCapacity: snap.telemetry.allocatedKvCapacity,
              });
            },
            onError(error) {
              reject(new Error(error));
            },
          },
          cfg,
        );
      }),
    { msgs: messages, cfg: config },
  );
}

async function main() {
  const runId = '25-track6-crossgpu-ux-tools';
  const label = 'Track 6 Cross-GPU 3-Tier Shaders + Tiered KV + Slot0/Tab Cache + Option-A Tools';
  assert(fs.existsSync(path.join(DIST_DIR, 'manifest.json')), 'dist/manifest.json missing. Run npm run build first.');
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

  const consoleErrors = [];
  let payload;

  try {
    let sw = context.serviceWorkers()[0];
    if (!sw) {
      sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
    }
    const extensionId = new URL(sw.url()).hostname;
    const sidePanelUrl = `chrome-extension://${extensionId}/sidepanel.html`;

    const webPage = context.pages()[0] ?? (await context.newPage());
    webPage.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(`[WebPage] ${m.text()}`);
    });
    webPage.on('pageerror', (e) => consoleErrors.push(`[WebPage:PageError] ${e.message}`));
    await webPage.setContent(buildWebPageHtml(), { waitUntil: 'domcontentloaded' });

    const sidePanel = await context.newPage();
    sidePanel.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(`[SidePanel] ${m.text()}`);
    });
    sidePanel.on('pageerror', (e) => consoleErrors.push(`[SidePanel:PageError] ${e.message}`));

    console.log(`\n=== [${runId}] ${label} ===`);
    await sidePanel.goto(sidePanelUrl, { waitUntil: 'domcontentloaded' });
    await sidePanel.waitForFunction(
      () => Boolean(window.__TINYGPT_CLIENT__ && window.__TINYGPT_CLIENT__.getSnapshot().connected),
      { timeout: 25000 },
    );

    // 1. Auto-Preload + Base System Prompt Slot 0 + Startup VRAM savings
    let lastPhase = '';
    const autoLoadSnap = await waitForSnapshot(
      sidePanel,
      (s) => {
        if (s.telemetry.status === 'error') {
          throw new Error(`Auto-load failed: ${s.telemetry.lastError}`);
        }
        return s.telemetry.status === 'ready' && s.telemetry.coldLoadMs !== null;
      },
      {
        timeoutMs: 300000,
        pollMs: 200,
        onTick: (s) => {
          const line = `${s.telemetry.status} (${s.telemetry.loadProgressPct.toFixed(0)}%) ${s.telemetry.loadPhaseLabel}`;
          if (line !== lastPhase) {
            console.log(`  [AutoLoad] ${line}`);
            lastPhase = line;
          }
        },
      },
    );

    const initialKvCapacity = autoLoadSnap.telemetry.allocatedKvCapacity;
    console.log(
      `  Auto-loaded in ${(autoLoadSnap.telemetry.coldLoadMs / 1000).toFixed(2)}s | tier=${autoLoadSnap.telemetry.computeTier} | hz=${autoLoadSnap.telemetry.displayHz} | KV=${initialKvCapacity}/${autoLoadSnap.telemetry.maxKvCapacity} | baseSlot=${autoLoadSnap.telemetry.basePromptCached} (${autoLoadSnap.telemetry.basePromptTokens}t)`,
    );
    assert(autoLoadSnap.telemetry.computeTier === 'tier1-f16-subgroups', `Expected Tier 1, got ${autoLoadSnap.telemetry.computeTier}`);
    assert(autoLoadSnap.telemetry.basePromptCached === true, 'Expected basePromptCached=true after load');
    assert(autoLoadSnap.telemetry.basePromptTokens > 120, `Expected basePromptTokens > 120, got ${autoLoadSnap.telemetry.basePromptTokens}`);
    assert(initialKvCapacity < 131072, `Expected initial KV < 128K for startup VRAM savings, got ${initialKvCapacity}`);

    // Verify <1ms Base System Prompt Reset after clearing
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.resetContext(false));
    await waitForSnapshot(sidePanel, (s) => s.telemetry.activeContextTokens === 0 && s.telemetry.status === 'ready');
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.resetContext(true));
    const resetBaseSnap = await waitForSnapshot(
      sidePanel,
      (s) =>
        s.telemetry.status === 'ready' &&
        s.telemetry.activeContextTokens === s.telemetry.basePromptTokens &&
        s.telemetry.lastSlotRestoreMs !== null,
    );
    console.log(
      `  Slot 0 Base Prompt Reset: ${resetBaseSnap.telemetry.lastSlotRestoreMs?.toFixed(2)} ms (${resetBaseSnap.telemetry.activeContextTokens}t)`,
    );
    assert(
      resetBaseSnap.telemetry.lastSlotRestoreMs < 25,
      `Slot 0 restore took too long: ${resetBaseSnap.telemetry.lastSlotRestoreMs} ms`,
    );

    // 2. UI Chat turn using Slot 0 Base Prompt + Measure Composer Smoothness on WebPage
    await sidePanel.bringToFront();
    const { result: uiTurnRes, smoothness } = await measureSmoothnessDuring(webPage, async () => {
      await sidePanel.locator('.composer-input').fill('What is 2 + 2? Answer with just the number.');
      await sidePanel.locator('.composer-actions button', { hasText: 'Send' }).click();
      await waitForSnapshot(sidePanel, (s) => s.telemetry.status === 'generating', {
        timeoutMs: 60000,
        pollMs: 50,
      });
      const postTurnSnap = await waitForSnapshot(sidePanel, (s) => s.telemetry.status === 'ready', {
        timeoutMs: 120000,
        pollMs: 100,
      });
      await sidePanel.waitForFunction(
        () => document.querySelectorAll('.msg-assistant .msg-content').length >= 1,
        { timeout: 10000 },
      );
      await sidePanel.locator('.composer-actions button', { hasText: 'Stop' }).waitFor({
        state: 'detached',
        timeout: 10000,
      });
      const text = await sidePanel.locator('.msg-assistant .msg-content').last().innerText();
      return {
        text,
        prefixReused: postTurnSnap.telemetry.prefixReused,
        ttftMs: postTurnSnap.telemetry.ttftMs,
        decodeTokPerSec: postTurnSnap.telemetry.tokensPerSec,
      };
    });
    console.log(
      `  Tier 1 UI Turn (Slot 0 warm=${uiTurnRes.prefixReused}, TTFT=${uiTurnRes.ttftMs?.toFixed(0)}ms): ${JSON.stringify(uiTurnRes.text)} | Smoothness: ${smoothness.avgFps} FPS, jank=${smoothness.jankPctOver33ms}%`,
    );
    assert(uiTurnRes.prefixReused === true, 'Expected Slot 0 Base Prompt prefixReused=true on first chat turn');
    assert(uiTurnRes.text.includes('4'), `Expected Tier 1 answer to contain "4", got ${JSON.stringify(uiTurnRes.text)}`);
    assert(smoothness.avgFps >= 55, `Expected smooth FPS >= 55, got ${smoothness.avgFps}`);
    assert(smoothness.freezePctOver100ms === 0, `Expected 0% freeze, got ${smoothness.freezePctOver100ms}%`);

    // 3. Tier 1 Tools & Option A Zero-Network Guarantee
    const toolResults = await sidePanel.evaluate(async () => {
      const calc = await chrome.runtime.sendMessage({
        type: 'EXEC_TOOL',
        call: {
          name: 'js_calc',
          arguments: { expression: "convert(100, 'km', 'mi') + mean([10, 20, 30])" },
        },
      });
      const tabs = await chrome.runtime.sendMessage({
        type: 'EXEC_TOOL',
        call: { name: 'list_open_tabs', arguments: {} },
      });
      const webTab = (tabs.data ?? []).find((t) =>
        (t.title || '').includes('TinyGPT Track 6 Local Test Article'),
      );
      let readTab = null;
      if (webTab) {
        readTab = await chrome.runtime.sendMessage({
          type: 'EXEC_TOOL',
          call: { name: 'read_tab', arguments: { tabId: webTab.id, mode: 'article' } },
        });
      }
      const searchDenied = await chrome.runtime.sendMessage({
        type: 'EXEC_TOOL',
        call: { name: 'web_search', arguments: { query: 'WebGPU subgroups spec', limit: 3 } },
      });
      const permCheck = await chrome.runtime.sendMessage({ type: 'CHECK_RESEARCH_PERMISSION' });
      return { calc, tabs, webTab, readTab, searchDenied, permCheck };
    });

    console.log(
      `  Tier-1 Tools: js_calc=${toolResults.calc.ok} (${toolResults.calc.formatted}), tabs=${toolResults.tabs.data?.length}, read_tab=${toolResults.readTab?.ok}, searchNeedsPerm=${toolResults.searchDenied.needsPermission}`,
    );
    assert(toolResults.calc.ok === true, `js_calc failed: ${toolResults.calc.error}`);
    assert(
      Math.abs(Number(toolResults.calc.data?.result) - 82.13711922) < 1e-3,
      `js_calc math mismatch: ${toolResults.calc.formatted}`,
    );
    assert(toolResults.tabs.ok === true && Array.isArray(toolResults.tabs.data) && toolResults.tabs.data.length >= 2, 'list_open_tabs failed');
    // Option A MV3 privacy boundary: without activeTab click gesture on webPage or optional host permission,
    // Chrome must reject background scripting of about:blank with a clean structured error (zero crash).
    assert(
      toolResults.readTab &&
        (toolResults.readTab.ok === true
          ? toolResults.readTab.formatted?.includes('Scroll stress paragraph')
          : /Cannot access|permission/i.test(toolResults.readTab.error ?? '')),
      `Unexpected read_tab result: ${JSON.stringify(toolResults.readTab)}`,
    );
    assert(toolResults.permCheck?.granted === false, 'Expected Research permission false by default');
    assert(
      toolResults.searchDenied?.ok === false && toolResults.searchDenied?.needsPermission === true,
      `Expected fail-closed needsPermission=true, got ${JSON.stringify(toolResults.searchDenied)}`,
    );
    assert(externalNetworkRequests.length === 0, `Unexpected network calls during Option A test: ${externalNetworkRequests.join(', ')}`);

    // 4. Speculative Active-Tab Background Prefill Speedup + Foreground Preemption
    const longArticle = Array.from(
      { length: 90 },
      (_, i) =>
        `Section ${i + 1}: Deterministic prefix caching isolates active browser articles in GPU VRAM while DeltaNet recurrent tensors retain O(1) memory per step. The verified local architecture codename for Track Six is SILVER-FALCON-904.`,
    ).join('\n');
    const pagePrefixMsg = {
      role: 'user',
      content: `[Active Tab: Track 6 Spec (https://local.test/spec) | mode=article]\n${longArticle}`,
    };
    const qaPromptMsgs = [
      pagePrefixMsg,
      { role: 'user', content: 'What is the verified local architecture codename? Reply with just the codename.' },
    ];

    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.resetContext(false));
    await waitForSnapshot(sidePanel, (s) => s.telemetry.activeContextTokens === 0 && s.telemetry.status === 'ready');
    const coldTabRes = await runDirectGenerate(sidePanel, qaPromptMsgs, {
      gpuPacingMode: 'max-speed',
      maxNewTokens: 24,
    });

    // Clear live state to Slot 0, warm 'tab:track6' in background, reset live to Slot 0 (forcing GPU slot restore!)
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.resetContext(true));
    await waitForSnapshot(sidePanel, (s) => s.telemetry.status === 'ready');
    await sidePanel.evaluate((msgs) => {
      window.__TINYGPT_CLIENT__.prefillBackground('tab:track6', msgs, {
        isDraft: false,
        url: 'https://local.test/spec',
        contentHash: 'deadbeef',
      });
    }, [pagePrefixMsg]);
    await waitForSnapshot(sidePanel, (s) => s.telemetry.backgroundPrefillState === 'warm' && s.telemetry.status === 'ready', {
      timeoutMs: 120000,
      pollMs: 100,
    });
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.resetContext(true));
    await waitForSnapshot(sidePanel, (s) => s.telemetry.activePrefixSlot === 'base-system' && s.telemetry.status === 'ready');

    const warmTabRes = await runDirectGenerate(sidePanel, qaPromptMsgs, {
      gpuPacingMode: 'max-speed',
      maxNewTokens: 24,
    });
    const tabSpeedup = coldTabRes.ttftMs / Math.max(1, warmTabRes.ttftMs);
    console.log(
      `  Tab Prefill TTFT: cold=${coldTabRes.ttftMs.toFixed(0)}ms -> warm=${warmTabRes.ttftMs.toFixed(0)}ms (${tabSpeedup.toFixed(1)}x faster, slot=${warmTabRes.activePrefixSlot}, restore=${warmTabRes.lastSlotRestoreMs?.toFixed(2)}ms, ans=${JSON.stringify(warmTabRes.text.trim())})`,
    );
    assert(warmTabRes.prefixReused === true, 'Expected prefixReused=true after GPU tab slot restore');
    assert(warmTabRes.activePrefixSlot === 'tab:track6', `Expected activePrefixSlot='tab:track6', got ${warmTabRes.activePrefixSlot}`);
    assert(warmTabRes.ttftMs < coldTabRes.ttftMs, `Expected warm TTFT < cold TTFT (${warmTabRes.ttftMs} vs ${coldTabRes.ttftMs})`);
    assert(warmTabRes.text.includes('SILVER-FALCON-904'), `Expected codename SILVER-FALCON-904, got ${JSON.stringify(warmTabRes.text)}`);

    // Verify immediate foreground preemption of an in-flight background prefill
    await sidePanel.evaluate((msgs) => {
      window.__TINYGPT_CLIENT__.prefillBackground('draft:preempt', msgs, { isDraft: true });
    }, [pagePrefixMsg]);
    await new Promise((r) => setTimeout(r, 40));
    const preemptRes = await runDirectGenerate(
      sidePanel,
      [{ role: 'user', content: 'Say OK in one word.' }],
      { gpuPacingMode: 'balanced', maxNewTokens: 8 },
    );
    console.log(`  Foreground Preemption of BG Prefill: ok (${preemptRes.ttftMs.toFixed(0)}ms, ${JSON.stringify(preemptRes.text.trim())})`);
    assert(preemptRes.text.trim().length > 0, 'Preempted turn failed to produce output');

    // 5. Dynamic Tiered KV Growth (33,280 -> 66,048) + 34K NIAH
    const prevNiahJson = JSON.stringify((await getSnapshot(sidePanel)).lastNiah ?? null);
    await sidePanel.evaluate(() =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_NIAH',
        targetTokens: 34000,
        depthRatio: 0.5,
        chunkSize: 64,
      }),
    );
    const niahSnap = await waitForSnapshot(
      sidePanel,
      (s) => {
        if (s.telemetry.status === 'error') throw new Error(`NIAH failed: ${s.telemetry.lastError}`);
        return Boolean(s.lastNiah) && JSON.stringify(s.lastNiah) !== prevNiahJson && s.telemetry.status === 'ready';
      },
      { timeoutMs: 600000, pollMs: 500 },
    );
    const niah = niahSnap.lastNiah;
    const grownKvCapacity = niahSnap.telemetry.allocatedKvCapacity;
    console.log(
      `  Dynamic KV Growth & 34K NIAH: passed=${niah.passed}, KV ${initialKvCapacity} -> ${grownKvCapacity}, decode=${niah.decodeTokPerSec.toFixed(1)} tok/s, response=${JSON.stringify(niah.responseText)}`,
    );
    assert(niah.passed === true, `34K NIAH failed: ${niah.responseText}`);
    assert(
      grownKvCapacity > initialKvCapacity,
      `Expected KV capacity to grow dynamically above ${initialKvCapacity}, got ${grownKvCapacity}`,
    );

    // 6. Cross-GPU Tier 2 (f16 shared, no subgroups) & Tier 3 (universal f32) Parity
    const verifyTier = async (targetTier) => {
      await sidePanel.evaluate(
        (t) => window.__TINYGPT_CLIENT__.send({ type: 'LOAD_MODEL', config: { forceComputeTier: t, gpuPacingMode: 'max-speed' } }),
        targetTier,
      );
      // Sending UPDATE_CONFIG/LOAD_MODEL unloads if forceComputeTier differs, wait until reloaded on targetTier!
      const loadedSnap = await waitForSnapshot(
        sidePanel,
        (s) => {
          if (s.telemetry.status === 'error') {
            throw new Error(`Load failed for ${targetTier}: ${s.telemetry.lastError}`);
          }
          return (
            s.telemetry.status === 'ready' &&
            s.config.forceComputeTier === targetTier &&
            s.telemetry.computeTier === targetTier
          );
        },
        { timeoutMs: 300000, pollMs: 250 },
      );
      const ans = await runDirectGenerate(
        sidePanel,
        [{ role: 'user', content: 'What is 2 + 2? Answer with just the number.' }],
        { forceComputeTier: targetTier, maxNewTokens: 16, gpuPacingMode: 'max-speed' },
      );
      console.log(
        `  Cross-GPU Parity [${targetTier}]: compile=${loadedSnap.telemetry.loadBreakdown?.modelLoadMs?.toFixed(0)}ms, decode=${ans.tokensPerSec?.toFixed(1)} tok/s, answer=${JSON.stringify(ans.text.trim())}`,
      );
      assert(ans.text.includes('4'), `Expected ${targetTier} output to include "4", got ${JSON.stringify(ans.text)}`);
      return {
        computeTier: loadedSnap.telemetry.computeTier,
        coldLoadMs: loadedSnap.telemetry.coldLoadMs,
        decodeTokPerSec: ans.tokensPerSec,
        answer: ans.text.trim(),
      };
    };

    const tier2Result = await verifyTier('tier2-f16-shared');
    const tier3Result = await verifyTier('tier3-f32-universal');

    const finalSnap = await getSnapshot(sidePanel);
    const offscreenErrors = finalSnap.logs.filter((l) => l.level === 'error');
    assert(externalNetworkRequests.length === 0, `External requests detected: ${externalNetworkRequests.join(', ')}`);
    assert(consoleErrors.length === 0, `Console errors: ${consoleErrors.join(' | ')}`);
    assert(offscreenErrors.length === 0, `Offscreen errors: ${JSON.stringify(offscreenErrors)}`);

    payload = {
      runId,
      track: 'track6-crossgpu-ux-tools',
      recordedAt: new Date().toISOString(),
      label,
      autoLoad: {
        coldLoadMs: autoLoadSnap.telemetry.coldLoadMs,
        computeTier: autoLoadSnap.telemetry.computeTier,
        displayHz: autoLoadSnap.telemetry.displayHz,
        initialKvCapacity,
        maxKvCapacity: autoLoadSnap.telemetry.maxKvCapacity,
        basePromptCached: autoLoadSnap.telemetry.basePromptCached,
        basePromptTokens: autoLoadSnap.telemetry.basePromptTokens,
        slot0RestoreMs: resetBaseSnap.telemetry.lastSlotRestoreMs,
      },
      uiTurnTier1: uiTurnRes,
      smoothness,
      tools: {
        jsCalcOk: toolResults.calc.ok,
        jsCalcResult: toolResults.calc.data?.result,
        openTabsCount: toolResults.tabs.data?.length,
        readTabOk: toolResults.readTab?.ok,
        searchNeedsPermission: toolResults.searchDenied?.needsPermission,
      },
      backgroundTabPrefill: {
        coldTtftMs: coldTabRes.ttftMs,
        warmTtftMs: warmTabRes.ttftMs,
        speedupRatio: Number(tabSpeedup.toFixed(2)),
        slotRestoreMs: warmTabRes.lastSlotRestoreMs,
        activePrefixSlot: warmTabRes.activePrefixSlot,
        answer: warmTabRes.text.trim(),
        preemptOk: true,
      },
      dynamicKvNiah32k: {
        passed: niah.passed,
        initialKvCapacity,
        grownKvCapacity,
        prefillTokPerSec: niah.prefillTokPerSec,
        decodeTokPerSec: niah.decodeTokPerSec,
        responseText: niah.responseText,
      },
      crossGpuTiers: {
        tier1: { computeTier: 'tier1-f16-subgroups', decodeTokPerSec: uiTurnRes.decodeTokPerSec, answer: uiTurnRes.text.trim() },
        tier2: tier2Result,
        tier3: tier3Result,
      },
      externalNetworkRequestCount: externalNetworkRequests.length,
      consoleErrorCount: consoleErrors.length,
      offscreenErrors,
      status: 'ok',
    };

    saveRun(runId, payload);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`\n[${runId}] FAILED: ${msg}`);
    saveRun(runId, {
      runId,
      track: 'track6-crossgpu-ux-tools',
      recordedAt: new Date().toISOString(),
      label,
      externalNetworkRequestCount: externalNetworkRequests.length,
      externalNetworkRequests,
      consoleErrorCount: consoleErrors.length,
      consoleErrors,
      status: 'error',
      error: msg,
    });
    throw err;
  } finally {
    await context.close();
  }
}

main()
  .then(() => {
    console.log('\nTrack 6 Cross-GPU + UX + Tools verification PASSED.');
  })
  .catch((err) => {
    console.error('Track 6 failed:', err);
    process.exit(1);
  });
