import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const BENCH_EVAL_DIR = path.resolve('benchmarks', 'evals');
const TRACES_DIR = path.resolve('benchmarks', 'traces');
const PROFILE_DIR = path.resolve('.bench-spec-e0-a1-profile');
const STAGED_DIST_DIR = path.resolve('.bench-spec-e0-a1-dist');
const EXP003C_DIR = 'D:/yours-pragma-lab/05_wgsl_bundles/EXP-003c-tool-qad';

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
    '/home': `${htmlHead('Home Dashboard')}<body><main><h1>TinyGPT Benchmark Home</h1><p>Local GPU benchmark host page.</p></main></body></html>`,
    '/spec-alpha': `${htmlHead('Alpha Cluster Spec')}<body><article><h1>Alpha Cluster Telemetry</h1><p>Cluster Identifier: <strong>ALPHA-TOKEN-718</strong>. Region: <strong>us-east-webgpu-1</strong>. Measured sustained request throughput: <strong>415</strong> req/s. Service Level Objective: 99.95% uptime with deterministic fail-closed sandboxing.</p></article></body></html>`,
    '/spec-beta': `${htmlHead('Beta Cluster Spec')}<body><article><h1>Beta Cluster Telemetry</h1><p>Cluster Identifier: <strong>BETA-TOKEN-924</strong>. Region: <strong>us-west-webgpu-2</strong>. Measured sustained request throughput: <strong>635</strong> req/s. Replication mode: synchronous DeltaNet checkpoint streaming.</p></article></body></html>`,
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

async function runSpecMicrobench(sidePanel) {
  await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.runSpecMicrobench());
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    const res = await sidePanel.evaluate(
      () => window.__TINYGPT_CLIENT__.getSnapshot().lastSpecMicrobench,
    );
    if (res) return res;
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error('Timed out waiting for SPEC_MICROBENCH_RESULT');
}

async function runBenchTurn(sidePanel, category, prompt, timeoutMs = 120000) {
  await sidePanel.evaluate(() => {
    window.__benchCapturedTraces = [];
    window.__benchLastSeenTraceRef = window.__TINYGPT_CLIENT__.getSnapshot().lastTrace ?? null;
  });
  const composer = sidePanel.locator('.yui-composer-input');
  await composer.fill(prompt);
  const t0 = Date.now();
  await sidePanel.locator('.yui-send-btn').click();

  let sawGen = false;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const { m } = await sidePanel.evaluate(() => {
      const snap = window.__TINYGPT_CLIENT__.getSnapshot();
      if (snap.lastTrace && snap.lastTrace !== window.__benchLastSeenTraceRef) {
        window.__benchLastSeenTraceRef = snap.lastTrace;
        window.__benchCapturedTraces.push(snap.lastTrace);
      }
      return { m: window.__TINYGPT_UI__.getMetrics() };
    });
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
  await new Promise((r) => setTimeout(r, 200));
  const elapsedMs = Date.now() - t0;
  const { metrics, traces, traj } = await sidePanel.evaluate(() => {
    const snap = window.__TINYGPT_CLIENT__.getSnapshot();
    if (snap.lastTrace && snap.lastTrace !== window.__benchLastSeenTraceRef) {
      window.__benchLastSeenTraceRef = snap.lastTrace;
      window.__benchCapturedTraces.push(snap.lastTrace);
    }
    return {
      metrics: window.__TINYGPT_UI__.getMetrics(),
      traces: window.__benchCapturedTraces ?? [],
      traj: window.__TINYGPT_UI__.getTrajectory(),
    };
  });
  const trace = traces[traces.length - 1] ?? null;
  const assistants = traj.messages.filter((m) => m.role === 'assistant');
  const lastAssistant = assistants[assistants.length - 1] ?? null;
  const toolMessages = traj.messages.filter((m) => m.role === 'tool');
  return {
    category,
    prompt,
    elapsedMs,
    metrics,
    trace,
    traces,
    answerText: lastAssistant?.content ?? '',
    reasoningText: lastAssistant?.reasoning ?? '',
    toolSteps: toolMessages.length,
  };
}

async function benchModel({ label, modelKey, sidePanel, alphaPage, homePage, tracesOut }) {
  const microbench = await runSpecMicrobench(sidePanel);
  console.log(`  [${label}] Microbench E0:`, JSON.stringify(microbench));

  await homePage.bringToFront();
  await sidePanel.evaluate(async () => {
    await window.__TINYGPT_CLIENT__.updateConfig({ enableSpeculative: true, gpuPacingMode: 'max-speed' });
    window.__TINYGPT_UI__.setPageEnabled(true);
    window.__TINYGPT_UI__.setEnableThinking(false);
    window.__TINYGPT_UI__.refreshActiveTab();
    await window.__TINYGPT_UI__.clearChat();
  });
  await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.activePage?.url?.includes('/home')), 15000);

  // Warmup turn
  await runBenchTurn(sidePanel, 'warmup', 'Reply with OK.');
  await sidePanel.evaluate(() => window.__TINYGPT_UI__.clearChat());

  const workloads = [];

  // 1. Casual / Open-ended generation (Adaptive A1 skips unproductive speculation)
  const casual = await runBenchTurn(
    sidePanel,
    'casual',
    'Name the capital of Japan and write three concise bullet points explaining why coastal cities have milder temperatures than inland cities.',
  );
  workloads.push(casual);

  // 2. System 2 Thinking / Math (<think> grammar scaffolding)
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setEnableThinking(true);
    await window.__TINYGPT_UI__.clearChat();
  });
  const thinking = await runBenchTurn(
    sidePanel,
    'thinking',
    'Think step by step and compute (27 * 19) + 14. End with ONLY the exact number on the final line.',
  );
  workloads.push(thinking);

  // 3. Copy / Grounded Extractive Summary (SAM + Anchor copy acceleration)
  await alphaPage.bringToFront();
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setEnableThinking(false);
    window.__TINYGPT_UI__.refreshActiveTab();
    await window.__TINYGPT_UI__.clearChat();
  });
  await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.activePage?.url?.includes('/spec-alpha')), 15000);
  const copy = await runBenchTurn(
    sidePanel,
    'copy',
    'List verbatim from this active page: the exact Cluster Identifier, Region, sustained request throughput, and Service Level Objective wording.',
  );
  workloads.push(copy);

  // 4. Multi-Tab Tool Chain (grammar + corpus tool-call scaffolding)
  await homePage.bringToFront();
  await sidePanel.evaluate(async () => {
    window.__TINYGPT_UI__.setEnableThinking(false);
    window.__TINYGPT_UI__.refreshActiveTab();
    await window.__TINYGPT_UI__.clearChat();
  });
  await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.activePage?.url?.includes('/home')), 15000);
  const tools = await runBenchTurn(
    sidePanel,
    'tools',
    'Use list_tabs and read_tab to inspect "Alpha Cluster Spec" and "Beta Cluster Spec", then use run_sandbox to compute Alpha req/s + Beta req/s and state ONLY the exact sum.',
  );
  workloads.push(tools);

  const summaryRows = {};
  for (const w of workloads) {
    const subTraces = w.traces && w.traces.length > 0 ? w.traces : w.trace ? [w.trace] : [];
    const subSp = subTraces.map((t) => t.speculativeStats).filter(Boolean);
    const sp =
      subSp.length > 1
        ? {
            drafted: subSp.reduce((s, x) => s + (x.drafted ?? 0), 0),
            accepted: subSp.reduce((s, x) => s + (x.accepted ?? 0), 0),
            steps: subSp.reduce((s, x) => s + (x.steps ?? 0), 0),
            skippedSteps: subSp.reduce((s, x) => s + (x.skippedSteps ?? 0), 0),
          }
        : w.metrics.speculativeStats;
    const tau = sp && sp.steps > 0 ? 1 + sp.accepted / sp.steps : 1;
    const alpha = sp && sp.drafted > 0 ? sp.accepted / sp.drafted : 0;
    const generatedTokens = subTraces.reduce((s, t) => s + (t.generatedIds?.length ?? 0), 0);
    summaryRows[w.category] = {
      ttftMs: round(w.metrics.ttftMs ?? 0, 1),
      decodeTokPerSec: round(w.metrics.tokensPerSec ?? 0, 2),
      tau: round(tau, 3),
      alpha: round(alpha, 3),
      drafted: sp?.drafted ?? 0,
      accepted: sp?.accepted ?? 0,
      steps: sp?.steps ?? 0,
      skippedSteps: sp?.skippedSteps ?? 0,
      subTurnCount: subTraces.length,
      toolSteps: w.toolSteps,
      generatedTokens,
      answerPreview: w.answerText.slice(0, 96),
    };
    subTraces.forEach((tr, subTurnIndex) => {
      tracesOut.push({
        modelKey,
        modelId: tr.modelId,
        category: w.category,
        subTurnIndex,
        promptText: tr.promptText,
        promptIds: tr.promptIds,
        generatedIds: tr.generatedIds,
        generatedText: tr.generatedText,
        speculativeStats: tr.speculativeStats,
      });
    });
  }

  return {
    label,
    modelKey,
    microbench,
    categories: summaryRows,
  };
}

async function main() {
  await stageEvalExtensionDir();
  await safeRemoveDir(PROFILE_DIR);
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
  fs.mkdirSync(BENCH_EVAL_DIR, { recursive: true });
  fs.mkdirSync(TRACES_DIR, { recursive: true });

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

    const tracesOut = [];

    await waitForUi(sidePanel, (m) => m.status === 'ready' && Boolean(m.basePromptCached), 180000);
    console.log('\n=== Benchmarking E0 + A1: Stock Qwen3.5-4B (v248320) ===');
    const stockReport = await benchModel({
      label: 'Stock Qwen3.5-4B (v248320)',
      modelKey: 'stock-qwen35-4b',
      sidePanel,
      alphaPage,
      homePage,
      tracesOut,
    });

    let expReport = null;
    if (fs.existsSync(path.join(EXP003C_DIR, 'manifest.json'))) {
      console.log('\n=== Loading & Benchmarking E0 + A1: EXP-003c (v81920) ===');
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
      expReport = await benchModel({
        label: 'EXP-003c Pragma-1-4B (v81920)',
        modelKey: 'exp003c-pragma-v81920',
        sidePanel,
        alphaPage,
        homePage,
        tracesOut,
      });
    }

    const report = {
      timestampIso: new Date().toISOString(),
      runtimeErrors,
      stock: stockReport,
      exp003c: expReport,
    };

    const jsonPath = path.join(BENCH_EVAL_DIR, 'speculative-e0-a1-report.json');
    const tracesPath = path.join(TRACES_DIR, 'real-extension-traces.json');
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    fs.writeFileSync(
      tracesPath,
      JSON.stringify(
        {
          schemaVersion: 1,
          generatedAt: report.timestampIso,
          traceCount: tracesOut.length,
          traces: tracesOut,
        },
        null,
        2,
      ),
    );
    console.log(`\nSaved E0+A1 report: ${jsonPath}\nSaved real extension traces (${tracesOut.length}): ${tracesPath}`);
    if (!stockReport.microbench.gatePass || (expReport && !expReport.microbench.gatePass)) {
      console.warn('WARNING: E0 T(5)/T(1) <= 2.7 gate NOT met on at least one model!');
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
