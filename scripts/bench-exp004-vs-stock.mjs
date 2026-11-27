import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const BENCH_EVAL_DIR = path.resolve('benchmarks', 'evals');
const PROFILE_DIR = path.resolve('.bench-exp004-profile');
const STAGED_DIST_DIR = path.resolve('.bench-exp004-dist');
const EXP003C_DIR = 'D:/yours-pragma-lab/05_wgsl_bundles/EXP-003c-tool-qad';
const EXP004_DIR = 'D:/yours-pragma-lab/05_wgsl_bundles/EXP-004-gspo-rl';

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
    const bundleMounts = [
      ['/exp003c/', EXP003C_DIR],
      ['/exp004/', EXP004_DIR],
    ];
    for (const [prefix, dir] of bundleMounts) {
      if (pathname.startsWith(prefix)) {
        const rel = path.basename(pathname.slice(prefix.length));
        const full = path.join(dir, rel);
        if (!fs.existsSync(full) || !fs.statSync(full).isFile()) {
          res.writeHead(404);
          res.end('Missing asset');
          return;
        }
        const stat = fs.statSync(full);
        const ext = path.extname(rel).toLowerCase();
        res.writeHead(200, {
          'Content-Type':
            ext === '.json' ? 'application/json; charset=utf-8' : 'application/octet-stream',
          'Content-Length': stat.size,
          'Access-Control-Allow-Origin': '*',
        });
        fs.createReadStream(full).pipe(res);
        return;
      }
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
    if (
      m.isGenerating ||
      m.turnState === 'streaming' ||
      m.turnState === 'prefilling' ||
      m.turnState === 'tool_executing'
    ) {
      sawGen = true;
    }
    if (
      sawGen &&
      !m.isGenerating &&
      (m.turnState === 'done' || m.turnState === 'idle' || m.turnState === 'aborted')
    ) {
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

async function benchmarkSingleModel({ label, sidePanel, homePage }) {
  await homePage.bringToFront();
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setPageEnabled(true);
    window.__TINYGPT_UI__.setEnableThinking(false);
    window.__TINYGPT_UI__.refreshActiveTab();
    await window.__TINYGPT_UI__.clearChat();
  });
  await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.activePage?.url?.includes('/home')), 12000);

  const clientSnap = await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());

  await runBenchTurn(sidePanel, 'Reply with OK.');
  await sidePanel.evaluate(() => window.__TINYGPT_UI__.clearChat());

  const direct = await runBenchTurn(
    sidePanel,
    'Name the capital of Japan and write three concise bullet points explaining why coastal cities have milder temperatures than inland cities.',
    { fpsPage: homePage },
  );

  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setEnableThinking(true);
    await window.__TINYGPT_UI__.clearChat();
  });
  const thinking = await runBenchTurn(
    sidePanel,
    'Think step by step and compute (27 * 19) + 14. End with ONLY the exact number on the final line.',
    { fpsPage: homePage },
  );

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
  const thinkingOk =
    /\b527\b/.test(thinking.answerText) &&
    thinking.reasoningText.length > 0 &&
    thinking.leaks.length === 0;
  const toolsOk =
    tools.toolMessages.length >= 3 &&
    tools.toolMessages.some((t) => t.toolName === 'run_sandbox' || t.toolName === 'js_calc') &&
    /1,?050/.test(tools.answerText) &&
    tools.leaks.length === 0;

  const meanDecodeTokPerSec = round(
    ((direct.metrics.tokensPerSec ?? 0) +
      (thinking.metrics.tokensPerSec ?? 0) +
      (tools.metrics.tokensPerSec ?? 0)) /
      3,
    2,
  );
  const meanTtftMs = round(
    ((direct.metrics.ttftMs ?? 0) + (thinking.metrics.ttftMs ?? 0) + (tools.metrics.ttftMs ?? 0)) / 3,
    1,
  );

  return {
    label,
    modelId: clientSnap.telemetry.modelId,
    computeTier: clientSnap.telemetry.computeTier,
    coldLoadMs: round(clientSnap.telemetry.coldLoadMs ?? 0, 1),
    estVramMb: round(clientSnap.telemetry.estVramMb ?? 0, 2),
    vramDeltaMb: clientSnap.telemetry.vramDeltaMb ?? null,
    allocatedKvCapacity: clientSnap.telemetry.allocatedKvCapacity ?? 0,
    meanDecodeTokPerSec,
    meanTtftMs,
    direct: {
      passed: directOk,
      ttftMs: round(direct.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(direct.metrics.tokensPerSec ?? 0, 2),
      tau: direct.metrics.speculativeStats?.tau ?? null,
      speedupMultiplier: direct.metrics.speculativeStats?.speedupMultiplier ?? null,
      avgFps: direct.smoothness?.avgFps ?? 0,
      jankPct: direct.smoothness?.jankPctOver33ms ?? 0,
      leaks: direct.leaks.length,
    },
    thinking: {
      passed: thinkingOk,
      ttftMs: round(thinking.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(thinking.metrics.tokensPerSec ?? 0, 2),
      reasoningChars: thinking.reasoningText.length,
      tau: thinking.metrics.speculativeStats?.tau ?? null,
      avgFps: thinking.smoothness?.avgFps ?? 0,
      jankPct: thinking.smoothness?.jankPctOver33ms ?? 0,
      leaks: thinking.leaks.length,
    },
    tools: {
      passed: toolsOk,
      toolSteps: tools.toolMessages.length,
      ttftMs: round(tools.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(tools.metrics.tokensPerSec ?? 0, 2),
      tau: tools.metrics.speculativeStats?.tau ?? null,
      avgFps: tools.smoothness?.avgFps ?? 0,
      jankPct: tools.smoothness?.jankPctOver33ms ?? 0,
      leaks: tools.leaks.length,
      answerSample: tools.answerText.slice(0, 120),
    },
  };
}

async function main() {
  for (const [name, dir] of [
    ['EXP-003c', EXP003C_DIR],
    ['EXP-004', EXP004_DIR],
  ]) {
    if (!fs.existsSync(path.join(dir, 'manifest.json'))) {
      throw new Error(`${name} manifest not found at ${dir}`);
    }
  }

  const stockManifest = JSON.parse(
    fs.readFileSync(path.resolve('dist/models/qwen35-4b/manifest.json'), 'utf8'),
  );
  const exp003cManifest = JSON.parse(
    fs.readFileSync(path.join(EXP003C_DIR, 'manifest.json'), 'utf8'),
  );
  const exp004Manifest = JSON.parse(
    fs.readFileSync(path.join(EXP004_DIR, 'manifest.json'), 'utf8'),
  );

  await stageEvalExtensionDir();
  await safeRemoveDir(PROFILE_DIR);
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
  fs.mkdirSync(BENCH_EVAL_DIR, { recursive: true });

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
    console.log('=== 1/3 Benchmarking Stock Qwen3.5-4B (v248320) ===');
    const stock = await benchmarkSingleModel({
      label: 'Stock Qwen3.5-4B (v248320)',
      sidePanel,
      homePage,
    });
    console.log(JSON.stringify(stock, null, 2));

    console.log('\n=== 2/3 Benchmarking EXP-003c Tool-QAD (v81920) ===');
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
    const exp003c = await benchmarkSingleModel({
      label: 'EXP-003c Tool-QAD (v81920)',
      sidePanel,
      homePage,
    });
    console.log(JSON.stringify(exp003c, null, 2));

    console.log('\n=== 3/3 Benchmarking EXP-004 GSPO-RL (v81920) ===');
    await sidePanel.evaluate(async (url) => {
      await window.__TINYGPT_UI__.loadModelConfig({
        modelId: 'hf://jtaylorexe/pragma-exp004-gspo-rl',
        bundleBaseUrl: url,
      });
    }, `${baseUrl}/exp004`);
    await waitForUi(
      sidePanel,
      (m) =>
        m.status === 'ready' &&
        m.modelId === 'jtaylorexe/pragma-exp004-gspo-rl' &&
        Boolean(m.basePromptCached),
      180000,
    );
    const exp004 = await benchmarkSingleModel({
      label: 'EXP-004 GSPO-RL (v81920)',
      sidePanel,
      homePage,
    });
    console.log(JSON.stringify(exp004, null, 2));

    const slab2MiBStock = stockManifest.slab2Bytes / (1024 * 1024);
    const slab2MiBExp003c = exp003cManifest.slab2Bytes / (1024 * 1024);
    const slab2MiBExp004 = exp004Manifest.slab2Bytes / (1024 * 1024);

    const report = {
      timestampIso: new Date().toISOString(),
      runtimeErrors,
      bundleComparison: {
        stockVocab: 248320,
        exp003cVocab: exp003cManifest.architecture?.vocab_size ?? 81920,
        exp004Vocab: exp004Manifest.architecture?.vocab_size ?? 81920,
        stockSlab2Bytes: stockManifest.slab2Bytes,
        exp003cSlab2Bytes: exp003cManifest.slab2Bytes,
        exp004Slab2Bytes: exp004Manifest.slab2Bytes,
        exp004VsStockSlab2MiB: round(slab2MiBExp004 - slab2MiBStock, 2),
        exp004VsStockDecodePct:
          stock.meanDecodeTokPerSec > 0
            ? round(((exp004.meanDecodeTokPerSec - stock.meanDecodeTokPerSec) / stock.meanDecodeTokPerSec) * 100, 2)
            : 0,
        exp004VsExp003cDecodePct:
          exp003c.meanDecodeTokPerSec > 0
            ? round(((exp004.meanDecodeTokPerSec - exp003c.meanDecodeTokPerSec) / exp003c.meanDecodeTokPerSec) * 100, 2)
            : 0,
      },
      stock,
      exp003c,
      exp004,
    };

    const jsonPath = path.join(BENCH_EVAL_DIR, 'exp004-vs-exp003c-vs-stock.json');
    const mdPath = path.join(BENCH_EVAL_DIR, 'exp004-vs-exp003c-vs-stock.md');
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));

    const md = [
      '# Head-to-Head Live WebGPU Benchmark: `EXP-004` vs `EXP-003c` vs Stock `Qwen3.5-4B`',
      '',
      `Generated: ${report.timestampIso}`,
      '',
      '## Footprint & Model Configuration',
      '',
      '| Metric | Stock `Qwen3.5-4B` | `EXP-003c` Tool-QAD | `EXP-004` GSPO-RL | `EXP-004` vs Stock |',
      '|---|---:|---:|---:|---:|',
      `| Vocab Size | 248,320 | ${(report.bundleComparison.exp003cVocab).toLocaleString()} | ${(report.bundleComparison.exp004Vocab).toLocaleString()} | -67.0% |`,
      `| \`slab2\` (\`lm_head\`) | ${slab2MiBStock.toFixed(1)} MiB | ${slab2MiBExp003c.toFixed(1)} MiB | ${slab2MiBExp004.toFixed(1)} MiB | ${report.bundleComparison.exp004VsStockSlab2MiB} MiB |`,
      `| Est. Runtime VRAM | ${stock.estVramMb} MiB | ${exp003c.estVramMb} MiB | ${exp004.estVramMb} MiB | ${round(exp004.estVramMb - stock.estVramMb, 2)} MiB |`,
      `| Cold Load Time | ${stock.coldLoadMs} ms | ${exp003c.coldLoadMs} ms | ${exp004.coldLoadMs} ms | ${round(exp004.coldLoadMs - stock.coldLoadMs, 1)} ms |`,
      '',
      '## Live WebGPU Workloads (Direct / Thinking / Multi-Tab Tools)',
      '',
      '| Workload | Stock TTFT / tok/s | `EXP-003c` TTFT / tok/s | `EXP-004` TTFT / tok/s | Pass (Stock / 003c / 004) |',
      '|---|---:|---:|---:|:---:|',
      `| Direct Chat (\`enableThinking=false\`) | ${stock.direct.ttftMs}ms / ${stock.direct.decodeTokPerSec} | ${exp003c.direct.ttftMs}ms / ${exp003c.direct.decodeTokPerSec} | ${exp004.direct.ttftMs}ms / ${exp004.direct.decodeTokPerSec} | ${stock.direct.passed} / ${exp003c.direct.passed} / ${exp004.direct.passed} |`,
      `| Adaptive Thinking (\`enableThinking=true\`) | ${stock.thinking.ttftMs}ms / ${stock.thinking.decodeTokPerSec} | ${exp003c.thinking.ttftMs}ms / ${exp003c.thinking.decodeTokPerSec} | ${exp004.thinking.ttftMs}ms / ${exp004.thinking.decodeTokPerSec} | ${stock.thinking.passed} / ${exp003c.thinking.passed} / ${exp004.thinking.passed} |`,
      `| Multi-Tab Tool Chain (\`list_tabs\`+\`read_tab\`+\`run_sandbox\`) | ${stock.tools.ttftMs}ms / ${stock.tools.decodeTokPerSec} | ${exp003c.tools.ttftMs}ms / ${exp003c.tools.decodeTokPerSec} | ${exp004.tools.ttftMs}ms / ${exp004.tools.decodeTokPerSec} | ${stock.tools.passed} / ${exp003c.tools.passed} / ${exp004.tools.passed} |`,
      `| **3-Workload Mean** | **${stock.meanTtftMs}ms / ${stock.meanDecodeTokPerSec} tok/s** | **${exp003c.meanTtftMs}ms / ${exp003c.meanDecodeTokPerSec} tok/s** | **${exp004.meanTtftMs}ms / ${exp004.meanDecodeTokPerSec} tok/s** | **${report.bundleComparison.exp004VsStockDecodePct}% vs Stock** |`,
      '',
      `Runtime errors during run: ${runtimeErrors.length}`,
      '',
    ].join('\n');
    fs.writeFileSync(mdPath, md);
    console.log(`\nSaved benchmark report: ${jsonPath}\nSaved markdown summary: ${mdPath}`);

    if (
      !exp004.direct.passed ||
      !exp004.thinking.passed ||
      !exp004.tools.passed ||
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
