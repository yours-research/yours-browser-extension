import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile-head-to-head');
const OUT_PATH = path.resolve('benchmarks', 'runs', 'g12-extension-head-to-head.json');

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

function cleanProfileDir() {
  for (let i = 0; i < 8; i++) {
    try {
      if (fs.existsSync(PROFILE_DIR)) {
        fs.rmSync(PROFILE_DIR, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
      }
      fs.mkdirSync(PROFILE_DIR, { recursive: true });
      return;
    } catch {
      // retry transient Windows file lock
    }
  }
}

function buildArticleContent() {
  const paras = Array.from(
    { length: 12 },
    (_, i) =>
      `Section ${i + 1}: TinyGPT WebGPU inference pairs local block-quantized weights with KV caching and prefix reuse. The verified benchmark key for this guide is QUANTUM-ANCHOR-8821.`,
  );
  return paras.join('\n\n');
}

async function waitReady(sidePanel, pred = (s) => s.telemetry.status === 'ready', timeoutMs = 240000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const snap = await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());
    if (snap.telemetry.status === 'error') {
      throw new Error(`Engine error: ${snap.telemetry.lastError}`);
    }
    if (pred(snap)) {
      return snap;
    }
    await new Promise((r) => setTimeout(r, 120));
  }
  throw new Error('Timed out waiting for Side Panel engine state.');
}

async function sendUiChatTurn(sidePanel, prompt, expectedMsgCount) {
  const t0 = performance.now();
  await sidePanel.locator('.composer-input').fill(prompt);
  await sidePanel.locator('.composer-actions button', { hasText: 'Send' }).click();
  await sidePanel.waitForFunction(
    (count) => document.querySelectorAll('.msg-assistant .msg-content').length >= count,
    expectedMsgCount,
    { timeout: 60000 },
  );
  await sidePanel.locator('.composer-actions button', { hasText: 'Stop' }).waitFor({
    state: 'detached',
    timeout: 120000,
  });
  const snap = await waitReady(sidePanel);
  let text = '';
  for (let i = 0; i < 6; i++) {
    const cur = await sidePanel.locator('.msg-assistant .msg-content').last().innerText();
    if (cur === text && cur.length > 0) break;
    text = cur;
    await new Promise((r) => setTimeout(r, 80));
  }
  return {
    wallMs: Math.round(performance.now() - t0),
    ttftMs: snap.telemetry.ttftMs !== null ? Math.round(snap.telemetry.ttftMs) : null,
    prefillTokPerSec:
      snap.telemetry.prefillTokPerSec !== null ? Number(snap.telemetry.prefillTokPerSec.toFixed(1)) : null,
    decodeTokPerSec:
      snap.telemetry.tokensPerSec !== null ? Number(snap.telemetry.tokensPerSec.toFixed(1)) : null,
    activeContextTokens: snap.telemetry.activeContextTokens,
    prefixReused: snap.telemetry.prefixReused,
    activePrefixSlot: snap.telemetry.activePrefixSlot,
    text: text.trim(),
  };
}

async function testModelInExtension(sidePanel, label, articleUrl, articleContent) {
  const loadSnap = await waitReady(
    sidePanel,
    (s) => s.telemetry.status === 'ready' && s.telemetry.coldLoadMs !== null && s.telemetry.basePromptCached,
    300000,
  );
  assert(loadSnap.telemetry.basePromptCached, `${label}: expected Slot0 cached`);
  console.log(
    `  [${label}] Ready in ${(loadSnap.telemetry.coldLoadMs / 1000).toFixed(2)}s (slot0=${loadSnap.telemetry.basePromptTokens}t, tier=${loadSnap.telemetry.computeTier})`,
  );

  // 1. Page OFF -> clean arithmetic sanity check
  const pageBtnText0 = await sidePanel.locator('button', { hasText: /^Page:/ }).innerText();
  if (pageBtnText0 !== 'Page: Off') {
    await sidePanel.locator('button', { hasText: /^Page:/ }).click();
  }
  const mathTurn = await sendUiChatTurn(sidePanel, 'What is 2 + 2? Answer with just the number.', 1);
  console.log(`  [${label} Math 2+2] ans=${JSON.stringify(mathTurn.text)}, ttft=${mathTurn.ttftMs}ms, dec=${mathTurn.decodeTokPerSec} tok/s`);
  assert(mathTurn.text.includes('4'), `${label} 2+2 failed: ${mathTurn.text}`);

  // New Chat (restores Slot 0)
  await sidePanel.locator('button', { hasText: 'New Chat' }).click();
  await waitReady(
    sidePanel,
    (s) => s.telemetry.status === 'ready' && s.telemetry.activeContextTokens === s.telemetry.basePromptTokens,
  );

  // 2. Page ON -> inject simulated active page, wait for warm prefill, ask Q&A + Turn 2 followup
  await sidePanel.locator('button', { hasText: /^Page:/ }).click();
  await sidePanel.evaluate(
    ({ url, content }) => {
      window.__TINYGPT_UI__.setSimulatedActivePage({
        mode: 'article',
        title: 'WebGPU Local LLM Guide',
        url,
        content,
        contentHash: 'h2h-key-8821',
        charCount: content.length,
        truncated: false,
      });
    },
    { url: articleUrl, content: articleContent },
  );

  await waitReady(
    sidePanel,
    (s) => s.telemetry.backgroundPrefillState === 'warm' && s.telemetry.status === 'ready',
    60000,
  );

  const pageTurn1 = await sendUiChatTurn(
    sidePanel,
    'What is the verified benchmark key mentioned in this guide? Reply with just the key.',
    1,
  );
  console.log(
    `  [${label} Page Q&A Turn 1] reused=${pageTurn1.prefixReused}, ttft=${pageTurn1.ttftMs}ms, dec=${pageTurn1.decodeTokPerSec} tok/s, ans=${JSON.stringify(pageTurn1.text)}`,
  );
  assert(pageTurn1.text.includes('QUANTUM-ANCHOR-8821'), `${label} Page Turn 1 failed: ${pageTurn1.text}`);

  const pageTurn2 = await sendUiChatTurn(
    sidePanel,
    'Repeat that exact passkey once more without extra commentary.',
    2,
  );
  console.log(
    `  [${label} Followup Turn 2] reused=${pageTurn2.prefixReused}, slot=${pageTurn2.activePrefixSlot}, ttft=${pageTurn2.ttftMs}ms, dec=${pageTurn2.decodeTokPerSec} tok/s, ans=${JSON.stringify(pageTurn2.text)}`,
  );
  assert(pageTurn2.prefixReused === true, `${label} expected Turn 2 prefixReused=true`);

  return {
    modelId: loadSnap.config.modelId,
    coldLoadMs: Math.round(loadSnap.telemetry.coldLoadMs),
    loadSource: loadSnap.telemetry.loadSource,
    computeTier: loadSnap.telemetry.computeTier,
    estVramMb: Math.round(loadSnap.telemetry.estVramMb),
    basePromptTokens: loadSnap.telemetry.basePromptTokens,
    mathTurn,
    pageTurn1,
    pageTurn2,
  };
}

async function main() {
  assert(fs.existsSync(path.join(DIST_DIR, 'manifest.json')), 'Missing dist/manifest.json. Run npm run build first.');
  assert(fs.existsSync(path.join(DIST_DIR, 'models', 'qwen35-4b', 'manifest.json')), 'Missing dist/models/qwen35-4b/manifest.json');
  assert(fs.existsSync(path.join(DIST_DIR, 'models', 'gemma4-e4b', 'manifest.json')), 'Missing dist/models/gemma4-e4b/manifest.json');
  assert(
    !fs.existsSync(path.join(DIST_DIR, 'models', 'gemma4-e4b', 'embed_tokens_q2.bin')),
    'Unexpected large embed_tokens_q2.bin inside dist/models/gemma4-e4b',
  );
  assert(
    !fs.existsSync(path.join(DIST_DIR, 'models', 'gemma4-e4b', 'ple_tokens_q2.bin')),
    'Unexpected large ple_tokens_q2.bin inside dist/models/gemma4-e4b',
  );

  cleanProfileDir();
  const articleContent = buildArticleContent();
  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`<!doctype html><html><head><title>WebGPU Local LLM Guide</title></head><body><p>${articleContent}</p></body></html>`);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const articleUrl = `http://127.0.0.1:${server.address().port}/guide`;

  const externalRequests = [];
  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    viewport: { width: 1120, height: 920 },
    args: [
      `--disable-extensions-except=${DIST_DIR}`,
      `--load-extension=${DIST_DIR}`,
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });

  context.on('request', (req) => {
    const u = req.url();
    if (u.startsWith('https://')) externalRequests.push(u);
  });

  const consoleErrors = [];
  try {
    let sw = context.serviceWorkers()[0];
    if (!sw) sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
    const extId = new URL(sw.url()).hostname;

    const webPage = context.pages()[0] ?? (await context.newPage());
    await webPage.goto(articleUrl, { waitUntil: 'domcontentloaded' });

    const sidePanel = await context.newPage();
    sidePanel.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    sidePanel.on('pageerror', (e) => consoleErrors.push(e.message));

    console.log('\n=== [g12-extension-head-to-head] Testing Qwen3.5-4B -> Gemma 4 E4B QAT ===');
    await sidePanel.goto(`chrome-extension://${extId}/sidepanel.html`, { waitUntil: 'domcontentloaded' });
    await sidePanel.waitForFunction(() => Boolean(window.__TINYGPT_CLIENT__?.getSnapshot().connected), { timeout: 20000 });

    const qwenResult = await testModelInExtension(sidePanel, 'Qwen3.5-4B', articleUrl, articleContent);

    // Clear simulated page before switching models via actual UI <select data-testid="model-switcher">
    await sidePanel.evaluate(() => window.__TINYGPT_UI__.setSimulatedActivePage(null));
    console.log('  Switching to Gemma 4 E4B QAT via UI Model Switcher...');
    await sidePanel.locator('[data-testid="model-switcher"]').selectOption('onnx-community/gemma-4-E4B-it-qat-mobile-ONNX');

    await waitReady(
      sidePanel,
      (s) =>
        s.config.modelId === 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX' &&
        s.telemetry.status === 'ready' &&
        s.telemetry.coldLoadMs !== null &&
        s.telemetry.basePromptCached,
      300000,
    );

    const gemma4Result = await testModelInExtension(sidePanel, 'Gemma 4 E4B QAT', articleUrl, articleContent);

    assert(externalRequests.length === 0, `Expected 0 external HTTPS requests, saw: ${externalRequests.join(', ')}`);
    assert(consoleErrors.length === 0, `Console errors observed: ${consoleErrors.join(' | ')}`);

    const out = {
      runId: 'g12-extension-head-to-head',
      recordedAt: new Date().toISOString(),
      externalHttpsRequests: externalRequests.length,
      consoleErrorCount: consoleErrors.length,
      qwen35: qwenResult,
      gemma4: gemma4Result,
    };
    fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
    fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2));
    console.log(`Saved ${OUT_PATH}`);
  } finally {
    await context.close().catch(() => {});
    server.close();
    fs.rmSync(PROFILE_DIR, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('test-extension-head-to-head failed:', err);
  process.exit(1);
});
