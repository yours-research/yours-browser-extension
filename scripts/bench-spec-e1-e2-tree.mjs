import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const BENCH_EVAL_DIR = path.resolve('benchmarks', 'evals');
const PROFILE_DIR = path.resolve('.bench-spec-tree-profile');
const STAGED_DIST_DIR = path.resolve('.bench-spec-tree-dist');
const EXP004_DIR = 'D:/yours-pragma-lab/05_wgsl_bundles/EXP-004-gspo-rl';

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
  const prodManifest = JSON.parse(fs.readFileSync(path.join(DIST_DIR, 'manifest.json'), 'utf8'));
  await safeRemoveDir(STAGED_DIST_DIR);
  fs.mkdirSync(STAGED_DIST_DIR, { recursive: true });
  for (const entry of fs.readdirSync(DIST_DIR, { withFileTypes: true })) {
    const srcPath = path.join(DIST_DIR, entry.name);
    const dstPath = path.join(STAGED_DIST_DIR, entry.name);
    if (entry.name === 'manifest.json') {
      fs.writeFileSync(
        dstPath,
        JSON.stringify(
          { ...prodManifest, host_permissions: ['http://127.0.0.1/*', 'https://*/*'] },
          null,
          2,
        ),
      );
    } else if (entry.isDirectory()) {
      fs.symlinkSync(srcPath, dstPath, 'junction');
    } else {
      fs.copyFileSync(srcPath, dstPath);
    }
  }
}

function buildServer() {
  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url || '/', 'http://127.0.0.1').pathname;
    if (pathname === '/home') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end('<!doctype html><html><head><title>Tree Spec Bench</title></head><body><h1>Tree Bench</h1></body></html>');
      return;
    }
    if (pathname.startsWith('/exp004/') && fs.existsSync(EXP004_DIR)) {
      const rel = path.basename(pathname.slice('/exp004/'.length));
      const full = path.join(EXP004_DIR, rel);
      if (fs.existsSync(full) && fs.statSync(full).isFile()) {
        const stat = fs.statSync(full);
        res.writeHead(200, {
          'Content-Type': path.extname(rel) === '.json' ? 'application/json' : 'application/octet-stream',
          'Content-Length': stat.size,
          'Access-Control-Allow-Origin': '*',
        });
        fs.createReadStream(full).pipe(res);
        return;
      }
    }
    res.writeHead(404);
    res.end('Not found');
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
  throw new Error(`Timed out waiting for UI state after ${timeoutMs}ms`);
}

async function runTurn(sidePanel, prompt, timeoutMs = 120000) {
  await sidePanel.locator('.yui-composer-input').fill(prompt);
  const t0 = Date.now();
  await sidePanel.locator('.yui-send-btn').click();
  let sawGen = false;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const m = await sidePanel.evaluate(() => window.__TINYGPT_UI__.getMetrics());
    if (m.isGenerating || m.turnState === 'streaming' || m.turnState === 'prefilling') sawGen = true;
    if (sawGen && !m.isGenerating && (m.turnState === 'done' || m.turnState === 'idle')) break;
    await new Promise((r) => setTimeout(r, 80));
  }
  await new Promise((r) => setTimeout(r, 200));
  const elapsedMs = Date.now() - t0;
  const metrics = await sidePanel.evaluate(() => window.__TINYGPT_UI__.getMetrics());
  const traj = await sidePanel.evaluate(() => window.__TINYGPT_UI__.getTrajectory());
  const assistants = traj.messages.filter((m) => m.role === 'assistant');
  return {
    elapsedMs,
    metrics,
    answerText: assistants[assistants.length - 1]?.content ?? '',
  };
}

async function main() {
  await stageEvalExtensionDir();
  await safeRemoveDir(PROFILE_DIR);
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
  fs.mkdirSync(BENCH_EVAL_DIR, { recursive: true });

  const { server, baseUrl } = await buildServer();
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

    // 1. Run real WebGPU E0 chain vs E1+E2 tree microbench on Stock Qwen3.5-4B
    await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.runSpecMicrobench());
    const stockMicro = await sidePanel.evaluate(async () => {
      const deadline = Date.now() + 90000;
      while (Date.now() < deadline) {
        const snap = window.__TINYGPT_CLIENT__.getSnapshot();
        if (snap.lastSpecMicrobench) return snap.lastSpecMicrobench;
        await new Promise((r) => setTimeout(r, 150));
      }
      throw new Error('Timed out waiting for lastSpecMicrobench');
    });
    console.log('Stock Qwen3.5-4B E0 vs E1+E2 microbench:', JSON.stringify(stockMicro, null, 2));

    // 2. Measure E1+E2 tree end-to-end speculative copy turn (after warmup)
    await sidePanel.evaluate(async () => {
      await window.__TINYGPT_CLIENT__.updateConfig({ enableSpeculative: true });
      window.__TINYGPT_UI__.setEnableThinking(false);
      await window.__TINYGPT_UI__.clearChat();
    });
    const echoPrompt =
      'Repeat the following exact specification word for word: "WebGPU speculative tree verification executes branch-aware DeltaNet recurrence, tree-masked grouped-query attention, and single-pass deferred KV compaction in one GPU command buffer without state backup copies."';
    const treeTurn = await runTurn(sidePanel, echoPrompt);

    // 3. Also load & microbench + echo on EXP-004 (v81920) if present
    let exp004Result = null;
    if (fs.existsSync(path.join(EXP004_DIR, 'manifest.json'))) {
      await sidePanel.evaluate(async (url) => {
        await window.__TINYGPT_UI__.loadModelConfig({
          modelId: 'jtaylorexe/pragma-exp004-gspo-rl',
          bundleBaseUrl: url,
          enableSpeculative: true,
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
      await sidePanel.waitForTimeout(250);
      await sidePanel.evaluate(() => window.__TINYGPT_CLIENT__.runSpecMicrobench());
      const expMicro = await sidePanel.evaluate(async () => {
        const deadline = Date.now() + 90000;
        while (Date.now() < deadline) {
          const snap = window.__TINYGPT_CLIENT__.getSnapshot();
          if (snap.lastSpecMicrobench) return snap.lastSpecMicrobench;
          await new Promise((r) => setTimeout(r, 150));
        }
        throw new Error('Timed out waiting for EXP-004 lastSpecMicrobench');
      });
      await sidePanel.evaluate(() => window.__TINYGPT_UI__.clearChat());
      const expTreeTurn = await runTurn(sidePanel, echoPrompt);
      exp004Result = {
        microbench: expMicro,
        copyTurn: {
          decodeTokPerSec: round(expTreeTurn.metrics.tokensPerSec ?? 0, 2),
          ttftMs: round(expTreeTurn.metrics.ttftMs ?? 0, 1),
          vramDeltaMb: expTreeTurn.metrics.vramDeltaMb ?? null,
          speculativeStats: expTreeTurn.metrics.speculativeStats,
          answerSample: expTreeTurn.answerText.slice(0, 160),
        },
      };
      console.log('EXP-004 E0 vs E1+E2 tree report:', JSON.stringify(exp004Result, null, 2));
    }

    const report = {
      timestampIso: new Date().toISOString(),
      runtimeErrors,
      stockQwen35: {
        microbench: stockMicro,
        copyTurn: {
          decodeTokPerSec: round(treeTurn.metrics.tokensPerSec ?? 0, 2),
          ttftMs: round(treeTurn.metrics.ttftMs ?? 0, 1),
          speculativeStats: treeTurn.metrics.speculativeStats,
          answerSample: treeTurn.answerText.slice(0, 160),
        },
      },
      exp004: exp004Result,
    };

    const outPath = path.join(BENCH_EVAL_DIR, 'speculative-e1-e2-tree-report.json');
    fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`Saved E1+E2 tree report: ${outPath}`);

    if (
      runtimeErrors.length > 0 ||
      !stockMicro.tree ||
      !stockMicro.gatePass ||
      (exp004Result && (!exp004Result.microbench?.tree || !exp004Result.microbench?.gatePass || exp004Result.microbench?.error))
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
