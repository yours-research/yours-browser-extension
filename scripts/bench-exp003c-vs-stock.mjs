import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const BENCH_EVAL_DIR = path.resolve('benchmarks', 'evals');
const PROFILE_DIR = path.resolve('.bench-exp003c-profile');
const STAGED_DIST_DIR = path.resolve('.bench-exp003c-dist');
const EXP003C_DIR = 'D:/yours-pragma-lab/05_wgsl_bundles/EXP-003c-tool-qad';

const FPS_PROBE_SCRIPT = `
(() => {
  if (window.__FPS_PROBE__) return;
  let running = false;
  let rafId = 0;
  let lastTs = null;
  let frames = [];
  const round = (v, d = 2) => Number(v.toFixed(d));
  function tick() {
    if (!running) return;
    const now = performance.now();
    if (lastTs !== null) frames.push(now - lastTs);
    lastTs = now;
    rafId = requestAnimationFrame(tick);
  }
  window.__FPS_PROBE__ = {
    start() {
      running = true;
      if (rafId) cancelAnimationFrame(rafId);
      lastTs = null;
      frames = [];
      rafId = requestAnimationFrame(tick);
    },
    stop() {
      running = false;
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      if (frames.length === 0) return { avgFps: 0, jankPctOver33ms: 0, frameCount: 0 };
      const avgDt = frames.reduce((a, b) => a + b, 0) / frames.length;
      const jank = frames.filter((dt) => dt > 33.3).length;
      return {
        avgFps: avgDt > 0 ? round(1000 / avgDt, 2) : 0,
        jankPctOver33ms: round((jank / frames.length) * 100, 2),
        frameCount: frames.length,
      };
    },
  };
})();
`;

const LEAK_PATTERNS = [
  /<\/?think\b[^>]*>/i,
  /<\/?thought\b[^>]*>/i,
  /<\/?tool_call\b[^>]*>/i,
  /<\|im_(?:start|end)\|>/i,
  /<\|endoftext\|>/i,
];

function findLeakedPatterns(text) {
  if (!text) return [];
  return LEAK_PATTERNS.filter((re) => re.test(text)).map((re) => re.source);
}

function round(val, digits = 2) {
  return Number.isFinite(val) ? Number(val.toFixed(digits)) : 0;
}

async function safeRemoveDir(dirPath) {
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    try {
      if (!fs.existsSync(dirPath)) return;
      for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
        if (entry.isSymbolicLink()) fs.rmSync(path.join(dirPath, entry.name), { force: true });
      }
      fs.rmSync(dirPath, { recursive: true, force: true, maxRetries: 4, retryDelay: 200 });
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
}

async function stageEvalExtensionDir() {
  const prodManifestPath = path.join(DIST_DIR, 'manifest.json');
  const prodManifest = JSON.parse(fs.readFileSync(prodManifestPath, 'utf8'));
  await safeRemoveDir(STAGED_DIST_DIR);
  fs.mkdirSync(STAGED_DIST_DIR, { recursive: true });
  for (const entry of fs.readdirSync(DIST_DIR, { withFileTypes: true })) {
    const srcPath = path.join(DIST_DIR, entry.name);
    const dstPath = path.join(STAGED_DIST_DIR, entry.name);
    if (entry.name === 'manifest.json') {
      const evalManifest = {
        ...prodManifest,
        host_permissions: ['http://127.0.0.1/*', 'https://*/*'],
      };
      fs.writeFileSync(dstPath, JSON.stringify(evalManifest, null, 2));
    } else if (entry.isDirectory()) {
      fs.symlinkSync(srcPath, dstPath, 'junction');
    } else {
      fs.copyFileSync(srcPath, dstPath);
    }
  }
}

function buildBenchServer() {
  const htmlHead = (title) =>
    `<!doctype html><html><head><meta charset="utf-8"><title>${title}</title></head>`;
  const routes = {
    '/home': `${htmlHead('Home Dashboard')}<body><main><h1>TinyGPT Benchmark Home</h1><p>Local GPU benchmark host page.</p></main><script>${FPS_PROBE_SCRIPT}</script></body></html>`,
    '/spec-alpha': `${htmlHead('Alpha Cluster Spec')}<body><article><h1>Alpha Cluster Telemetry</h1><p>Cluster Identifier: <strong>ALPHA-TOKEN-718</strong>.</p><p>Measured sustained request throughput: <strong>415</strong> req/s.</p></article></body></html>`,
    '/spec-beta': `${htmlHead('Beta Cluster Spec')}<body><article><h1>Beta Cluster Telemetry</h1><p>Cluster Identifier: <strong>BETA-TOKEN-924</strong>.</p><p>Measured sustained request throughput: <strong>635</strong> req/s.</p></article></body></html>`,
  };

  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url || '/', 'http://127.0.0.1').pathname;
    if (pathname.startsWith('/exp003c/')) {
      const rel = path.basename(pathname.slice('/exp003c/'.length));
      const full = path.join(EXP003C_DIR, rel);
      if (!fs.existsSync(full) || !fs.statSync(full).isFile()) {
        res.writeHead(404);
        res.end('Missing asset');
        return;
      }
      const stat = fs.statSync(full);
      const ext = path.extname(rel).toLowerCase();
      res.writeHead(200, {
        'Content-Type': ext === '.json' ? 'application/json; charset=utf-8' : 'application/octet-stream',
        'Content-Length': stat.size,
        'Access-Control-Allow-Origin': '*',
      });
      fs.createReadStream(full).pipe(res);
      return;
    }

    const html = routes[pathname];
    if (!html) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, baseUrl: `http://127.0.0.1:${addr.port}` });
    });
  });
}

async function waitForUi(sidePanel, predicate, timeoutMs = 180000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const m = await sidePanel.evaluate(() => window.__TINYGPT_UI__?.getMetrics() ?? null);
    if (m && (await predicate(m))) return m;
    await new Promise((r) => setTimeout(r, 120));
  }
  const last = await sidePanel.evaluate(() => window.__TINYGPT_UI__?.getMetrics() ?? null);
  throw new Error(`Timed out waiting for UI state (${timeoutMs}ms): ${JSON.stringify(last)}`);
}

async function runBenchTurn(sidePanel, prompt, { fpsPage = null, timeoutMs = 120000 } = {}) {
  const composer = sidePanel.locator('.yui-composer-input');
  await composer.fill(prompt);
  if (fpsPage) {
    await fpsPage.bringToFront();
    await fpsPage.evaluate(() => window.__FPS_PROBE__?.start());
  }
  const t0 = Date.now();
  await sidePanel.locator('.yui-send-btn').click();

  let sawGen = false;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const m = await sidePanel.evaluate(() => window.__TINYGPT_UI__.getMetrics());
    if (m.isGenerating || m.turnState === 'streaming' || m.turnState === 'prefilling' || m.turnState === 'tool_executing') {
      sawGen = true;
    }
    if (sawGen && !m.isGenerating && (m.turnState === 'done' || m.turnState === 'idle' || m.turnState === 'aborted')) {
      break;
    }
    await new Promise((r) => setTimeout(r, 90));
  }
  await new Promise((r) => setTimeout(r, 250));
  const elapsedMs = Date.now() - t0;
  const metrics = await sidePanel.evaluate(() => window.__TINYGPT_UI__.getMetrics());
  const traj = await sidePanel.evaluate(() => window.__TINYGPT_UI__.getTrajectory());
  const smoothness = fpsPage ? await fpsPage.evaluate(() => window.__FPS_PROBE__?.stop() ?? null) : null;
  const assistants = traj.messages.filter((m) => m.role === 'assistant');
  const lastAssistant = assistants[assistants.length - 1] ?? null;
  const toolMessages = traj.messages.filter((m) => m.role === 'tool');
  const answerText = lastAssistant?.content ?? '';
  const reasoningText = lastAssistant?.reasoning ?? '';
  return {
    elapsedMs,
    metrics,
    answerText,
    reasoningText,
    toolMessages,
    smoothness,
    leaks: findLeakedPatterns(answerText),
  };
}

async function benchmarkSingleModel({ label, sidePanel, homePage, baseUrl }) {
  await homePage.bringToFront();
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setPageEnabled(true);
    window.__TINYGPT_UI__.setEnableThinking(false);
    window.__TINYGPT_UI__.refreshActiveTab();
    await window.__TINYGPT_UI__.clearChat();
  });
  await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.activePage?.url?.includes('/home')), 12000);

  const clientSnap = await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());

  // Warmup short turn so prefix slot & WGSL command caches are equally warm for both models
  await runBenchTurn(sidePanel, 'Reply with OK.');
  await sidePanel.evaluate(() => window.__TINYGPT_UI__.clearChat());

  // 1. Direct Chat (System 1, sustained decode length for accurate lm_head tok/s)
  const direct = await runBenchTurn(
    sidePanel,
    'Name the capital of Japan and write three concise bullet points explaining why coastal cities have milder temperatures than inland cities.',
    { fpsPage: homePage },
  );

  // 2. Adaptive Thinking (System 2)
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setEnableThinking(true);
    await window.__TINYGPT_UI__.clearChat();
  });
  const thinking = await runBenchTurn(
    sidePanel,
    'Think step by step and compute (27 * 19) + 14. End with ONLY the exact number on the final line.',
    { fpsPage: homePage },
  );

  // 3. Multi-Tab Tool Chain (list_tabs + read_tab + run_sandbox)
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setEnableThinking(false);
    await window.__TINYGPT_UI__.clearChat();
  });
  const tools = await runBenchTurn(
    sidePanel,
    'Use list_tabs and read_tab to inspect "Alpha Cluster Spec" and "Beta Cluster Spec", then use run_sandbox to compute Alpha req/s + Beta req/s and state ONLY the exact sum.',
    { fpsPage: homePage },
  );

  const directOk =
    /tokyo/i.test(direct.answerText) &&
    direct.answerText.length >= 60 &&
    direct.leaks.length === 0;
  const thinkingOk = /\b527\b/.test(thinking.answerText) && thinking.reasoningText.length > 0 && thinking.leaks.length === 0;
  const toolsOk =
    tools.toolMessages.length >= 3 &&
    tools.toolMessages.some((t) => t.toolName === 'run_sandbox' || t.toolName === 'js_calc') &&
    /1,?050/.test(tools.answerText) &&
    tools.leaks.length === 0;

  return {
    label,
    modelId: clientSnap.telemetry.modelId,
    computeTier: clientSnap.telemetry.computeTier,
    coldLoadMs: round(clientSnap.telemetry.coldLoadMs ?? 0, 1),
    estVramMb: round(clientSnap.telemetry.estVramMb ?? 0, 2),
    allocatedKvCapacity: clientSnap.telemetry.allocatedKvCapacity ?? 0,
    direct: {
      passed: directOk,
      ttftMs: round(direct.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(direct.metrics.tokensPerSec ?? 0, 2),
      avgFps: direct.smoothness?.avgFps ?? 0,
      jankPct: direct.smoothness?.jankPctOver33ms ?? 0,
      leaks: direct.leaks.length,
    },
    thinking: {
      passed: thinkingOk,
      ttftMs: round(thinking.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(thinking.metrics.tokensPerSec ?? 0, 2),
      reasoningChars: thinking.reasoningText.length,
      avgFps: thinking.smoothness?.avgFps ?? 0,
      jankPct: thinking.smoothness?.jankPctOver33ms ?? 0,
      leaks: thinking.leaks.length,
    },
    tools: {
      passed: toolsOk,
      toolSteps: tools.toolMessages.length,
      ttftMs: round(tools.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(tools.metrics.tokensPerSec ?? 0, 2),
      avgFps: tools.smoothness?.avgFps ?? 0,
      jankPct: tools.smoothness?.jankPctOver33ms ?? 0,
      leaks: tools.leaks.length,
      answerSample: tools.answerText.slice(0, 120),
    },
  };
}

async function main() {
  if (!fs.existsSync(path.join(EXP003C_DIR, 'manifest.json'))) {
    throw new Error(`EXP-003c manifest not found at ${EXP003C_DIR}`);
  }
  const stockManifest = JSON.parse(
    fs.readFileSync(path.resolve('dist/models/qwen35-4b/manifest.json'), 'utf8'),
  );
  const expManifest = JSON.parse(
    fs.readFileSync(path.join(EXP003C_DIR, 'manifest.json'), 'utf8'),
  );

  await stageEvalExtensionDir();
  await safeRemoveDir(PROFILE_DIR);
  fs.mkdirSync(PROFILE_DIR, { recursive: true });

  const { server, baseUrl } = await buildBenchServer();
  const runtimeErrors = [];
  let context = null;

  try {
    context = await chromium.launchPersistentContext(PROFILE_DIR, {
      headless: false,
      viewport: { width: 1200, height: 920 },
      args: [
        `--disable-extensions-except=${STAGED_DIST_DIR}`,
        `--load-extension=${STAGED_DIST_DIR}`,
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--no-first-run',
        '--no-default-browser-check',
      ],
    });

    let sw = context.serviceWorkers()[0];
    if (!sw) sw = await context.waitForEvent('serviceworker', { timeout: 25000 });
    const extensionId = new URL(sw.url()).hostname;

    const homePage = context.pages()[0] ?? (await context.newPage());
    await homePage.goto(`${baseUrl}/home`, { waitUntil: 'domcontentloaded' });
    const alphaPage = await context.newPage();
    await alphaPage.goto(`${baseUrl}/spec-alpha`, { waitUntil: 'domcontentloaded' });
    const betaPage = await context.newPage();
    await betaPage.goto(`${baseUrl}/spec-beta`, { waitUntil: 'domcontentloaded' });

    const sidePanel = await context.newPage();
    sidePanel.on('pageerror', (err) => runtimeErrors.push(err.message));
    await sidePanel.goto(`chrome-extension://${extensionId}/sidepanel.html`, {
      waitUntil: 'domcontentloaded',
    });
    await sidePanel.waitForFunction(
      () => Boolean(window.__TINYGPT_CLIENT__?.getSnapshot().connected && window.__TINYGPT_UI__),
      { timeout: 30000 },
    );

    await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.basePromptCached), 180000);
    console.log('=== Benchmarking Stock Qwen3.5-4B (v248320) ===');
    const stockResult = await benchmarkSingleModel({
      label: 'Stock Qwen3.5-4B (v248320)',
      sidePanel,
      homePage,
      baseUrl,
    });
    console.log(JSON.stringify(stockResult, null, 2));

    console.log('\n=== Loading & Benchmarking EXP-003c Pragma-1-4B (v81920) ===');
    await sidePanel.evaluate(async (url) => {
      await window.__TINYGPT_UI__.loadModelConfig({
        modelId: 'jtaylorexe/pragma-exp003c-tool-qad',
        bundleBaseUrl: url,
      });
    }, `${baseUrl}/exp003c`);
    await waitForUi(
      sidePanel,
      (m) =>
        m.status === 'ready' &&
        m.modelId === 'jtaylorexe/pragma-exp003c-tool-qad' &&
        Boolean(m.basePromptCached),
      180000,
    );

    const expResult = await benchmarkSingleModel({
      label: 'EXP-003c Pragma-1-4B (v81920)',
      sidePanel,
      homePage,
      baseUrl,
    });
    console.log(JSON.stringify(expResult, null, 2));

    const slab2MiBStock = stockManifest.slab2Bytes / (1024 * 1024);
    const slab2MiBExp = expManifest.slab2Bytes / (1024 * 1024);
    const slab2Ratio = round(stockManifest.slab2Bytes / expManifest.slab2Bytes, 2);
    const stockMeanDecode = round(
      (stockResult.direct.decodeTokPerSec +
        stockResult.thinking.decodeTokPerSec +
        stockResult.tools.decodeTokPerSec) /
        3,
      2,
    );
    const expMeanDecode = round(
      (expResult.direct.decodeTokPerSec +
        expResult.thinking.decodeTokPerSec +
        expResult.tools.decodeTokPerSec) /
        3,
      2,
    );
    const stockMeanTtft = round(
      (stockResult.direct.ttftMs + stockResult.thinking.ttftMs + stockResult.tools.ttftMs) / 3,
      1,
    );
    const expMeanTtft = round(
      (expResult.direct.ttftMs + expResult.thinking.ttftMs + expResult.tools.ttftMs) / 3,
      1,
    );

    const report = {
      timestampIso: new Date().toISOString(),
      runtimeErrors,
      bundleComparison: {
        stockVocab: 248320,
        exp003cVocab: expManifest.architecture?.vocab_size ?? 81920,
        stockSlab2Bytes: stockManifest.slab2Bytes,
        exp003cSlab2Bytes: expManifest.slab2Bytes,
        slab2ReductionMiB: round(slab2MiBExp - slab2MiBStock, 2),
        lmHeadCompressionRatio: slab2Ratio,
        vramDeltaMb: round(expResult.estVramMb - stockResult.estVramMb, 2),
        meanDecodeSpeedupPct:
          stockMeanDecode > 0 ? round(((expMeanDecode - stockMeanDecode) / stockMeanDecode) * 100, 2) : 0,
        meanTtftDeltaMs: round(expMeanTtft - stockMeanTtft, 1),
      },
      stock: { ...stockResult, meanDecodeTokPerSec: stockMeanDecode, meanTtftMs: stockMeanTtft },
      exp003c: { ...expResult, meanDecodeTokPerSec: expMeanDecode, meanTtftMs: expMeanTtft },
    };

    const jsonPath = path.join(BENCH_EVAL_DIR, 'exp003c-vs-stock-qwen35.json');
    const mdPath = path.join(BENCH_EVAL_DIR, 'exp003c-vs-stock-qwen35.md');
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));

    const md = [
      '# Live Chrome WebGPU Benchmark: `EXP-003c` (`Pragma-1-4B` v81920) vs Stock `Qwen3.5-4B` (v248320)',
      '',
      `Generated: ${report.timestampIso}`,
      '',
      '## Architecture & VRAM Footprint',
      '',
      '| Metric | Stock `Qwen3.5-4B` | `EXP-003c` (`Pragma-1-4B`) | Delta |',
      '|---|---:|---:|---:|',
      `| Vocab Size | 248,320 | ${(expManifest.architecture?.vocab_size ?? 81920).toLocaleString()} | -67.0% |`,
      `| \`slab2\` (\`lm_head\`) Size | ${slab2MiBStock.toFixed(2)} MiB | ${slab2MiBExp.toFixed(2)} MiB | ${report.bundleComparison.slab2ReductionMiB} MiB (${slab2Ratio}x smaller) |`,
      `| Est. Runtime VRAM | ${stockResult.estVramMb} MiB | ${expResult.estVramMb} MiB | ${report.bundleComparison.vramDeltaMb} MiB |`,
      `| Cold Load Time | ${stockResult.coldLoadMs} ms | ${expResult.coldLoadMs} ms | ${round(expResult.coldLoadMs - stockResult.coldLoadMs, 1)} ms |`,
      '',
      '## Live WebGPU Performance & Accuracy',
      '',
      '| Workload | Stock TTFT / Decode | `EXP-003c` TTFT / Decode | Stock FPS (Jank) | `EXP-003c` FPS (Jank) | Stock / EXP Pass |',
      '|---|---:|---:|---:|---:|:---:|',
      `| Direct Chat (\`enableThinking=false\`) | ${stockResult.direct.ttftMs}ms / ${stockResult.direct.decodeTokPerSec} tok/s | ${expResult.direct.ttftMs}ms / ${expResult.direct.decodeTokPerSec} tok/s | ${stockResult.direct.avgFps} (${stockResult.direct.jankPct}%) | ${expResult.direct.avgFps} (${expResult.direct.jankPct}%) | ${stockResult.direct.passed} / ${expResult.direct.passed} |`,
      `| Adaptive Thinking (\`enableThinking=true\`) | ${stockResult.thinking.ttftMs}ms / ${stockResult.thinking.decodeTokPerSec} tok/s | ${expResult.thinking.ttftMs}ms / ${expResult.thinking.decodeTokPerSec} tok/s | ${stockResult.thinking.avgFps} (${stockResult.thinking.jankPct}%) | ${expResult.thinking.avgFps} (${expResult.thinking.jankPct}%) | ${stockResult.thinking.passed} / ${expResult.thinking.passed} |`,
      `| Multi-Tab Tool Chain (\`list_tabs\`+\`read_tab\`+\`run_sandbox\`) | ${stockResult.tools.ttftMs}ms / ${stockResult.tools.decodeTokPerSec} tok/s | ${expResult.tools.ttftMs}ms / ${expResult.tools.decodeTokPerSec} tok/s | ${stockResult.tools.avgFps} (${stockResult.tools.jankPct}%) | ${expResult.tools.avgFps} (${expResult.tools.jankPct}%) | ${stockResult.tools.passed} / ${expResult.tools.passed} |`,
      `| **3-Workload Mean** | **${stockMeanTtft}ms / ${stockMeanDecode} tok/s** | **${expMeanTtft}ms / ${expMeanDecode} tok/s** | — | — | **+${report.bundleComparison.meanDecodeSpeedupPct}% decode** |`,
      '',
      `Runtime errors during run: ${runtimeErrors.length}`,
      '',
    ].join('\n');
    fs.writeFileSync(mdPath, md);
    console.log(`\nSaved benchmark report: ${jsonPath}\nSaved markdown summary: ${mdPath}`);
    if (
      !expResult.direct.passed ||
      !expResult.thinking.passed ||
      !expResult.tools.passed ||
      runtimeErrors.length > 0
    ) {
      process.exitCode = 1;
    }
  } finally {
    if (context) await context.close().catch(() => {});
    await new Promise((r) => server.close(() => r()));
    await safeRemoveDir(PROFILE_DIR);
    await safeRemoveDir(STAGED_DIST_DIR);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
