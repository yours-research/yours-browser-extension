import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { build as esbuild } from 'esbuild';
import { chromium } from 'playwright';

const BUNDLE_DIR = path.resolve('model-bundle', 'gemma4-e4b');
const WASM_PATH = path.resolve(
  'crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
);
const RUNS_DIR = path.resolve('benchmarks', 'runs');

const PLE_Q_BYTES = 2688;
const PLE_S_BYTES = 84;
const PLE_ROW_BYTES = PLE_Q_BYTES + PLE_S_BYTES; // 2772
const PLE_SCALES_OFFSET = 704643072; // 262144 * 2688

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function parseArgs(argv) {
  const opts = {
    phase: 'phase-bench',
    out: path.join(RUNS_DIR, 'g-phase-latest.json'),
    contexts: [1024, 8192, 32768],
    decodeSteps: 32,
    kvMode: 'int8',
    maxContext: 131072,
    runNiah: false,
    runBalanced: false,
    balancedOnly: false,
    runDraftSkipSweep: false,
    flags: {},
  };
  for (const arg of argv) {
    if (arg.startsWith('--phase=')) opts.phase = arg.slice(8);
    else if (arg.startsWith('--out=')) opts.out = path.resolve(arg.slice(6));
    else if (arg.startsWith('--contexts=')) {
      opts.contexts = arg
        .slice(11)
        .split(',')
        .map((n) => Number(n.trim()))
        .filter((n) => Number.isFinite(n) && n > 0);
    } else if (arg.startsWith('--decodeSteps=')) {
      opts.decodeSteps = Number(arg.slice(14)) || 32;
    } else if (arg.startsWith('--kvMode=')) {
      opts.kvMode = arg.slice(9);
    } else if (arg.startsWith('--maxContext=')) {
      opts.maxContext = Number(arg.slice(13)) || 131072;
    } else if (arg === '--niah') {
      opts.runNiah = true;
    } else if (arg === '--balanced') {
      opts.runBalanced = true;
    } else if (arg === '--balancedOnly') {
      opts.runBalanced = true;
      opts.balancedOnly = true;
    } else if (arg === '--draftSkipSweep') {
      opts.runDraftSkipSweep = true;
    } else if (arg.startsWith('--flags=')) {
      opts.flags = JSON.parse(arg.slice(8));
    }
  }
  return opts;
}

function cleanProfileDir(dir) {
  for (let i = 0; i < 8; i++) {
    try {
      if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
      }
      fs.mkdirSync(dir, { recursive: true });
      return;
    } catch {
      // ignore transient lock on Windows
    }
  }
}

function installFpsProbe() {
  let active = false;
  let rafId = 0;
  let lastTs = 0;
  const frameDeltas = [];
  const inputLatencies = [];
  let jankFrames = 0;
  let freezeFrames = 0;

  function tick(ts) {
    if (!active) return;
    if (lastTs > 0) {
      const dt = ts - lastTs;
      frameDeltas.push(dt);
      if (dt > 33.4) jankFrames += 1;
      if (dt > 100.0) freezeFrames += 1;
    }
    lastTs = ts;
    rafId = requestAnimationFrame(tick);
  }

  window.__FPS_PROBE__ = {
    start() {
      frameDeltas.length = 0;
      inputLatencies.length = 0;
      jankFrames = 0;
      freezeFrames = 0;
      lastTs = 0;
      active = true;
      rafId = requestAnimationFrame(tick);
    },
    async measureScrollAndInput() {
      const scroller = document.getElementById('scroller');
      if (scroller) {
        scroller.scrollTop = (scroller.scrollTop + 47) % Math.max(1, scroller.scrollHeight - 200);
      }
      const btn = document.getElementById('probe-btn');
      if (btn) btn.click();
      const t0 = performance.now();
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
      inputLatencies.push(performance.now() - t0);
    },
    stop() {
      active = false;
      if (rafId) cancelAnimationFrame(rafId);
      const sorted = [...frameDeltas].sort((a, b) => a - b);
      const pct = (p) => {
        if (!sorted.length) return 0;
        const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
        return Number(sorted[idx].toFixed(2));
      };
      const avgDt = sorted.length
        ? sorted.reduce((a, b) => a + b, 0) / sorted.length
        : 16.67;
      const avgInput = inputLatencies.length
        ? inputLatencies.reduce((a, b) => a + b, 0) / inputLatencies.length
        : 0;
      return {
        avgFps: Number((1000 / Math.max(0.1, avgDt)).toFixed(2)),
        p50FrameMs: pct(50),
        p95FrameMs: pct(95),
        p99FrameMs: pct(99),
        maxFrameMs: sorted.length ? Number(sorted[sorted.length - 1].toFixed(2)) : 0,
        jankPctOver33ms: sorted.length
          ? Number(((jankFrames / sorted.length) * 100).toFixed(2))
          : 0,
        freezePctOver100ms: sorted.length
          ? Number(((freezeFrames / sorted.length) * 100).toFixed(2))
          : 0,
        avgInputLatencyMs: Number(avgInput.toFixed(2)),
        frameCount: sorted.length,
      };
    },
  };
}

function buildProbeHtml() {
  const paragraphs = Array.from(
    { length: 40 },
    (_, i) =>
      `<p>Line ${i + 1}: Gemma 4 E4B QAT Dedicated Worker incremental optimization probe.</p>`,
  ).join('\n');

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Gemma 4 Phase Benchmark Harness</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 16px; background: #0b0e14; color: #e2e8f0; }
      .stage { display: flex; gap: 16px; align-items: center; margin-bottom: 12px; }
      canvas { border: 1px solid #2b3446; border-radius: 8px; background: #131a26; }
      #scroller { height: 220px; overflow-y: auto; border: 1px solid #2b3446; padding: 10px; }
      #stream-out { margin: 12px 0; padding: 10px; background: #111827; min-height: 48px; white-space: pre-wrap; font-family: monospace; font-size: 12px; }
    </style>
  </head>
  <body>
    <div class="stage">
      <canvas id="orbital-canvas" width="420" height="140"></canvas>
      <button id="probe-btn" type="button">Counter: 0</button>
    </div>
    <div id="stream-out"></div>
    <div id="scroller">${paragraphs}</div>
    <script>
      (() => {
        let clicks = 0;
        const btn = document.getElementById('probe-btn');
        btn.addEventListener('click', () => { clicks += 1; btn.textContent = 'Counter: ' + clicks; });
        const canvas = document.getElementById('orbital-canvas');
        const ctx = canvas.getContext('2d');
        function renderLoop(now) {
          const angle = (now * 0.003) % (Math.PI * 2);
          ctx.fillStyle = '#131a26';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(canvas.width / 2 + Math.cos(angle) * 48, canvas.height / 2 + Math.sin(angle) * 48, 10, 0, Math.PI * 2);
          ctx.fill();
          requestAnimationFrame(renderLoop);
        }
        requestAnimationFrame(renderLoop);
      })();
      (${installFpsProbe.toString()})();
      window.__GEMMA4_HOST__ = {
        worker: null,
        pending: new Map(),
        nextId: 1,
        streamText: '',
        ensureWorker() {
          if (this.worker) return this.worker;
          const w = new Worker('/worker.js', { type: 'module' });
          w.onmessage = (ev) => {
            const msg = ev.data;
            if (msg.type === 'token') {
              this.streamText = msg.textSoFar ?? '';
              const el = document.getElementById('stream-out');
              if (el) el.textContent = this.streamText;
              return;
            }
            if (msg.type === 'response') {
              const p = this.pending.get(msg.id);
              if (p) {
                this.pending.delete(msg.id);
                if (msg.error) p.reject(new Error(msg.error));
                else p.resolve(msg.result);
              }
            }
          };
          w.onerror = (err) => console.error('Worker error:', err.message);
          this.worker = w;
          return w;
        },
        callWorker(action, payload = {}) {
          const w = this.ensureWorker();
          const id = this.nextId++;
          return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            w.postMessage({ id, action, payload });
          });
        },
        abortWorkerNow() {
          if (this.worker) this.worker.postMessage({ action: 'ABORT_ACTIVE' });
        },
      };
    </script>
  </body>
</html>`;
}

async function buildWorkerBundle() {
  const workerSource = `
    import { env, AutoTokenizer } from '@huggingface/transformers';
    import { Gemma4WgslEngine } from './src/offscreen/gemma4WgslEngine.ts';

    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = '/models/';
    env.useBrowserCache = false;

    const eosIds = new Set([1, 106]);
    let tokenizer = null;
    let engine = null;
    let activeAbort = null;
    let conversationTokenIds = [];

    async function pleRowFetcher(tokenIds) {
      const res = await fetch('/ple-batch?ids=' + tokenIds.join(','));
      if (!res.ok) throw new Error('PLE batch fetch failed: ' + res.status);
      return new Uint8Array(await res.arrayBuffer());
    }

    function encodeChatPrompt(promptText) {
      const formatted = tokenizer.apply_chat_template(
        [{ role: 'user', content: promptText }],
        { tokenize: false, add_generation_prompt: true, enable_thinking: false },
      );
      const enc = tokenizer(formatted, { add_special_tokens: false });
      return Array.from(enc.input_ids.data, Number);
    }

    function computeBandwidth(decodeTokPerSec, seqPos, kvMode) {
      const activeWeightsBytes = 1.98e9;
      const slideToks = Math.min(seqPos, 512);
      const slideBytes = 35 * 2 * 2 * slideToks * 256 * 2;
      const globalBytesPerTok = kvMode === 'int4' ? 1152 : kvMode === 'int8' ? 2048 : 4096;
      const globalBytes = 7 * seqPos * globalBytesPerTok;
      const totalBytes = activeWeightsBytes + slideBytes + globalBytes;
      const achievedBandwidthGBs = Number(((totalBytes * decodeTokPerSec) / 1e9).toFixed(2));
      const bandwidthUtilizationPct = Number(((achievedBandwidthGBs / 936.0) * 100).toFixed(2));
      return { achievedBandwidthGBs, bandwidthUtilizationPct };
    }

    async function handleAction(action, payload) {
      if (action === 'LOAD') {
        const t0 = performance.now();
        if (!tokenizer) {
          tokenizer = await AutoTokenizer.from_pretrained('gemma4-e4b');
          const jinjaRes = await fetch('/models/gemma4-e4b/chat_template.jinja');
          if (jinjaRes.ok) tokenizer.chat_template = await jinjaRes.text();
        }
        if (engine) {
          engine.dispose();
          engine = null;
        }
        engine = new Gemma4WgslEngine();
        const stats = await engine.load({
          baseUrl: '/models/gemma4-e4b/',
          wasmUrl: '/wasm/tinygpt_wgsl_engine.wasm',
          kvMode: payload.kvMode ?? 'int8',
          maxContext: payload.maxContext ?? 131072,
          pacingMode: payload.pacingMode ?? 'max-speed',
          displayHz: payload.displayHz ?? 144,
          pleRowFetcher,
          ...(payload.flags || {}),
        });
        conversationTokenIds = [];
        return {
          ...stats,
          coldLoadMs: Math.round(performance.now() - t0),
          executionHost: 'worker',
          loadSource: 'bundled',
        };
      }

      if (action === 'RESET') {
        engine.resetContext();
        conversationTokenIds = [];
        return { ok: true };
      }

      if (action === 'RUN_GOLDEN') {
        engine.resetContext();
        const idsA = encodeChatPrompt('What is 2 + 2? Answer with just the number.');
        const resA = await engine.generate(idsA, { maxNewTokens: 16, eosIds });
        const textA = tokenizer.decode(resA.generatedIds, { skip_special_tokens: true }).trim();
        engine.resetContext();
        const idsB = encodeChatPrompt('Explain WebGPU in one short sentence.');
        const resB = await engine.generate(idsB, { maxNewTokens: 32, eosIds });
        const textB = tokenizer.decode(resB.generatedIds, { skip_special_tokens: true }).trim();
        return {
          goldenA: {
            text: textA,
            generatedIds: resA.generatedIds,
            top0: resA.step0Top5Logits?.[0],
            prefillTokPerSec: resA.prefillTokPerSec,
            decodeTokPerSec: resA.decodeTokPerSec,
          },
          goldenB: {
            text: textB,
            generatedIds: resB.generatedIds,
            top0: resB.step0Top5Logits?.[0],
            prefillTokPerSec: resB.prefillTokPerSec,
            decodeTokPerSec: resB.decodeTokPerSec,
          },
        };
      }

      if (action === 'SWEEP') {
        const points = [];
        let oomCeiling = null;
        for (const ctxLen of payload.contexts) {
          try {
            engine.resetContext();
            const ids = new Array(ctxLen);
            for (let i = 0; i < ctxLen; i++) ids[i] = 1000 + (i % 500);
            const pref = await engine.prefillTokens(ids, { chunkSize: 64, returnLogits: false });
            const decSteps = payload.decodeSteps ?? 32;
            const decRes = await engine.decodeSequence(pref.nextTokenId, decSteps);
            const decodeTokPerSec = decRes.decodeTokPerSec;
            const bw = computeBandwidth(decodeTokPerSec, ctxLen + decRes.tokens.length, payload.kvMode ?? 'int8');
            points.push({
              targetTokens: ctxLen,
              actualPromptTokens: ctxLen,
              decodeTokens: decSteps,
              chunkSize: 64,
              prefillMs: pref.prefillMs,
              ttftMs: pref.prefillMs,
              prefillTokPerSec: pref.prefillTokPerSec,
              decodeTokPerSec,
              ...bw,
              status: 'ok',
            });
          } catch (err) {
            const msg = err?.message || String(err);
            oomCeiling = { atContext: ctxLen, error: msg };
            points.push({ targetTokens: ctxLen, status: 'failed', error: msg });
            break;
          }
        }
        return { points, oomCeiling };
      }

      if (action === 'NIAH_32K') {
        const passkey = 'VELVET-ORBIT-7492';
        const targetTokens = 32768;
        const fillerUnit = tokenizer(
          'The brown fox jumps swiftly across calm mountain streams while stars glimmer overhead in crisp autumn air. ',
          { add_special_tokens: false },
        );
        const fillerIds = Array.from(fillerUnit.input_ids.data, Number);
        const needleEnc = tokenizer(
          ' IMPORTANT RECORD: The secret passkey for this session is ' + passkey + '. Keep this exact passkey safe. ',
          { add_special_tokens: false },
        );
        const needleIds = Array.from(needleEnc.input_ids.data, Number);
        const headEnc = tokenizer('<bos><|turn>user\\nRead the archive carefully and remember any passkey.\\n\\n', {
          add_special_tokens: false,
        });
        const headIds = Array.from(headEnc.input_ids.data, Number);
        const tailEnc = tokenizer(
          '\\n\\nQuestion: What is the exact secret passkey mentioned in the archive above? Reply with ONLY the passkey.<turn|>\\n<|turn>model\\n',
          { add_special_tokens: false },
        );
        const tailIds = Array.from(tailEnc.input_ids.data, Number);

        const bodyBudget = targetTokens - headIds.length - tailIds.length - needleIds.length;
        const halfBefore = Math.floor(bodyBudget * 0.5);
        const halfAfter = bodyBudget - halfBefore;

        const promptIds = [...headIds];
        while (promptIds.length < headIds.length + halfBefore) {
          const need = Math.min(fillerIds.length, headIds.length + halfBefore - promptIds.length);
          for (let i = 0; i < need; i++) promptIds.push(fillerIds[i]);
        }
        for (const id of needleIds) promptIds.push(id);
        while (promptIds.length < headIds.length + halfBefore + needleIds.length + halfAfter) {
          const need = Math.min(
            fillerIds.length,
            headIds.length + halfBefore + needleIds.length + halfAfter - promptIds.length,
          );
          for (let i = 0; i < need; i++) promptIds.push(fillerIds[i]);
        }
        for (const id of tailIds) promptIds.push(id);

        engine.resetContext();
        const res = await engine.generate(promptIds, { maxNewTokens: 24, eosIds, chunkSize: 64 });
        const responseText = tokenizer.decode(res.generatedIds, { skip_special_tokens: true }).trim();
        return {
          targetTokens,
          actualTokens: promptIds.length,
          needleDepthPct: 0.5,
          passkey,
          responseText,
          passed: responseText.includes(passkey),
          ttftMs: res.ttftMs,
          prefillTokPerSec: res.prefillTokPerSec,
          decodeTokPerSec: res.decodeTokPerSec,
        };
      }

      if (action === 'CHAT_TURN') {
        const isFollowup = Boolean(payload.appendFollowup && conversationTokenIds.length > 0);
        let promptIds;
        if (isFollowup) {
          const turnStr = '<turn|>\\n<|turn>user\\n' + payload.prompt + '<turn|>\\n<|turn>model\\n';
          const enc = tokenizer(turnStr, { add_special_tokens: false });
          promptIds = [...conversationTokenIds, ...Array.from(enc.input_ids.data, Number)];
        } else {
          engine.resetContext();
          promptIds = encodeChatPrompt(payload.prompt);
        }

        activeAbort = new AbortController();
        const streamIds = [];
        const res = await engine.generate(promptIds, {
          maxNewTokens: payload.maxNewTokens ?? 128,
          eosIds,
          enableSpeculative: Boolean(payload.enableSpeculative),
          draftSkipLayers: Number(payload.draftSkipLayers ?? 0),
          abortSignal: activeAbort.signal,
          onToken: (tokId) => {
            streamIds.push(tokId);
            if (payload.streamTokens) {
              const txt = tokenizer.decode(streamIds, { skip_special_tokens: true });
              self.postMessage({ type: 'token', textSoFar: txt });
            }
          },
        });
        activeAbort = null;
        conversationTokenIds = [...promptIds, ...res.generatedIds];
        const responseText = tokenizer.decode(res.generatedIds, { skip_special_tokens: true }).trim();
        return {
          prompt: payload.prompt,
          responseText,
          responseLength: responseText.length,
          ttftMs: res.ttftMs,
          prefillTokPerSec: res.prefillTokPerSec,
          decodeTokPerSec: res.decodeTokPerSec,
          activeContextTokens: engine.getSeqPos(),
          prefixReused: res.prefixReusedTokens > 0,
          prefixReusedTokens: res.prefixReusedTokens,
          aborted: res.aborted,
          speculativeStats: res.speculativeStats,
        };
      }

      if (action === 'DISPOSE') {
        if (engine) { engine.dispose(); engine = null; }
        return { ok: true };
      }

      throw new Error('Unknown action: ' + action);
    }

    self.onmessage = async (ev) => {
      const { id, action, payload } = ev.data;
      if (action === 'ABORT_ACTIVE') {
        if (activeAbort) activeAbort.abort();
        return;
      }
      try {
        const result = await handleAction(action, payload || {});
        self.postMessage({ type: 'response', id, result });
      } catch (err) {
        self.postMessage({ type: 'response', id, error: err?.message || String(err) });
      }
    };
  `;

  const result = await esbuild({
    stdin: { contents: workerSource, resolveDir: process.cwd(), loader: 'js' },
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
  });
  return result.outputFiles[0].text;
}

function getMimeType(filePath) {
  if (filePath.endsWith('.html')) return 'text/html; charset=utf-8';
  if (filePath.endsWith('.js') || filePath.endsWith('.mjs')) return 'text/javascript; charset=utf-8';
  if (filePath.endsWith('.json')) return 'application/json; charset=utf-8';
  if (filePath.endsWith('.jinja')) return 'text/plain; charset=utf-8';
  if (filePath.endsWith('.wasm')) return 'application/wasm';
  return 'application/octet-stream';
}

function serveStaticFile(req, res, filePath) {
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }
  const stat = fs.statSync(filePath);
  const mime = getMimeType(filePath);
  res.writeHead(200, { 'Content-Type': mime, 'Content-Length': stat.size });
  if (req.method === 'HEAD') { res.end(); return; }
  fs.createReadStream(filePath).pipe(res);
}

async function measureSmoothnessDuring(page, fn) {
  await page.evaluate(() => window.__FPS_PROBE__.start());
  let running = true;
  const pumper = (async () => {
    while (running) {
      await new Promise((r) => setTimeout(r, 120));
      if (!running) break;
      await page.evaluate(() => window.__FPS_PROBE__.measureScrollAndInput()).catch(() => {});
    }
  })();
  let result;
  try { result = await fn(); } finally {
    running = false;
    await pumper.catch(() => {});
  }
  const smoothness = await page.evaluate(() => window.__FPS_PROBE__.stop());
  return { result, smoothness };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  assert(fs.existsSync(path.join(BUNDLE_DIR, 'manifest.json')), 'Missing model-bundle/gemma4-e4b/manifest.json');
  assert(fs.existsSync(WASM_PATH), 'Missing WASM build. Run npm run build:wasm');

  const profileDir = path.resolve(`.bench-profile-${opts.phase.replace(/[^a-z0-9_-]/gi, '_')}`);
  cleanProfileDir(profileDir);

  const workerJs = await buildWorkerBundle();
  const probeHtml = buildProbeHtml();
  const pleFd = fs.openSync(path.join(BUNDLE_DIR, 'ple_tokens_q2.bin'), 'r');

  const server = http.createServer((req, res) => {
    const urlObj = new URL(req.url, 'http://127.0.0.1');
    const pathname = decodeURIComponent(urlObj.pathname);
    if (pathname === '/' || pathname === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(probeHtml);
      return;
    }
    if (pathname === '/favicon.ico') {
      res.writeHead(204);
      res.end();
      return;
    }
    if (pathname === '/worker.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8' });
      res.end(workerJs);
      return;
    }
    if (pathname === '/wasm/tinygpt_wgsl_engine.wasm') {
      serveStaticFile(req, res, WASM_PATH);
      return;
    }
    if (pathname === '/ple-batch') {
      const raw = urlObj.searchParams.get('ids') ?? '';
      const ids = raw ? raw.split(',').map((s) => Number(s) | 0) : [];
      const outBuf = Buffer.allocUnsafe(ids.length * PLE_ROW_BYTES);
      for (let i = 0; i < ids.length; i++) {
        const tid = ids[i] >= 0 && ids[i] < 262144 ? ids[i] : 0;
        const dstBase = i * PLE_ROW_BYTES;
        fs.readSync(pleFd, outBuf, dstBase, PLE_Q_BYTES, tid * PLE_Q_BYTES);
        fs.readSync(pleFd, outBuf, dstBase + PLE_Q_BYTES, PLE_S_BYTES, PLE_SCALES_OFFSET + tid * PLE_S_BYTES);
      }
      res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': outBuf.byteLength });
      res.end(outBuf);
      return;
    }
    if (pathname.startsWith('/models/gemma4-e4b/')) {
      serveStaticFile(req, res, path.join(BUNDLE_DIR, pathname.slice('/models/gemma4-e4b/'.length)));
      return;
    }
    res.writeHead(404);
    res.end('Not found');
  });

  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;

  const context = await chromium.launchPersistentContext(profileDir, {
    headless: false,
    viewport: { width: 1080, height: 920 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-first-run', '--no-default-browser-check'],
  });

  const consoleErrors = [];
  try {
    const page = context.pages()[0] ?? (await context.newPage());
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    page.on('pageerror', (e) => consoleErrors.push(e.message));

    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => Boolean(window.__GEMMA4_HOST__ && window.__FPS_PROBE__), { timeout: 20000 });

    console.log(`\n=== [${opts.phase}] kv=${opts.kvMode}, maxCtx=${opts.maxContext}, flags=${JSON.stringify(opts.flags)} ===`);

    let load = null;
    let golden = null;
    let sweep = null;
    let chatTurn1 = null;
    let speculativeCheck = null;
    let draftSkipSweep = null;
    let niah = null;

    if (!opts.balancedOnly) {
      load = await page.evaluate(
        (cfg) => window.__GEMMA4_HOST__.callWorker('LOAD', { ...cfg, pacingMode: 'max-speed', displayHz: 144 }),
        { kvMode: opts.kvMode, maxContext: opts.maxContext, flags: opts.flags },
      );
      console.log(`  Load: ${load.coldLoadMs}ms (tier=${load.computeTier}, dp4a=${load.hasDp4a}, vram=${load.vramPlan.totalVramMb} MiB)`);

      golden = await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('RUN_GOLDEN'));
      console.log(`  Golden A=${JSON.stringify(golden.goldenA.text)} (id=${golden.goldenA.top0?.id}, val=${golden.goldenA.top0?.val}) | Golden B id=${golden.goldenB.top0?.id} val=${golden.goldenB.top0?.val} (${golden.goldenB.decodeTokPerSec} tok/s)`);
      assert(golden.goldenA.text === '4' && golden.goldenA.generatedIds?.[0] === 236812, `Golden A failed: ${JSON.stringify(golden.goldenA)}`);
      assert(golden.goldenB.top0?.id === 6721, `Golden B top0 failed: ${JSON.stringify(golden.goldenB.top0)}`);

      sweep = await page.evaluate(
        (sCfg) => window.__GEMMA4_HOST__.callWorker('SWEEP', sCfg),
        { contexts: opts.contexts, decodeSteps: opts.decodeSteps, kvMode: opts.kvMode },
      );
      for (const pt of sweep.points) {
        console.log(
          pt.status === 'ok'
            ? `  ctx=${pt.targetTokens}: prefill=${pt.prefillTokPerSec} tok/s (${pt.prefillMs}ms), decode=${pt.decodeTokPerSec} tok/s (${pt.achievedBandwidthGBs} GB/s)`
            : `  ctx=${pt.targetTokens}: FAILED (${pt.error})`,
        );
      }

      chatTurn1 = await page.evaluate(() =>
        window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
          prompt: 'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.',
          maxNewTokens: 128,
          enableSpeculative: false,
        }),
      );
      console.log(`  Chat Turn 1: ${chatTurn1.decodeTokPerSec} tok/s, TTFT=${chatTurn1.ttftMs}ms`);

      const specPrompt =
        'Rewrite and clean up the following config bullet list verbatim in Markdown, keeping every field name and exact numeric limit unchanged:\n' +
        '- maxBufferSize: 2147483648\n' +
        '- maxStorageBufferBindingSize: 2147483644\n' +
        '- workers: DedicatedWorker (es module)\n' +
        '- kvCacheMode: int8 (131072 context)\n' +
        '- weightShards: 21 parts (<= 128 MiB each)\n' +
        '- decodePacing: balanced (144Hz smooth compositor)';

      const specBase = await page.evaluate(
        (p) => window.__GEMMA4_HOST__.callWorker('CHAT_TURN', { prompt: p, maxNewTokens: 256, enableSpeculative: false }),
        specPrompt,
      );
      const specOpt = await page.evaluate(
        (p) => window.__GEMMA4_HOST__.callWorker('CHAT_TURN', { prompt: p, maxNewTokens: 256, enableSpeculative: true }),
        specPrompt,
      );
      const parityPassed = specBase.responseText === specOpt.responseText;
      speculativeCheck = {
        parityPassed,
        baselineDecodeTokPerSec: specBase.decodeTokPerSec,
        speculativeDecodeTokPerSec: specOpt.decodeTokPerSec,
        speedupRatio:
          specBase.decodeTokPerSec && specOpt.decodeTokPerSec
            ? Number((specOpt.decodeTokPerSec / specBase.decodeTokPerSec).toFixed(3))
            : null,
        speculativeStats: specOpt.speculativeStats,
        baselineLength: specBase.responseText.length,
        speculativeLength: specOpt.responseText.length,
      };
      console.log(`  Speculative: ${specBase.decodeTokPerSec} -> ${specOpt.decodeTokPerSec} tok/s (${speculativeCheck.speedupRatio}x, acc=${specOpt.speculativeStats?.acceptanceRatePct}%, parity=${parityPassed})`);
      assert(parityPassed, `Speculative parity failed!\nBase: ${specBase.responseText}\nSpec: ${specOpt.responseText}`);
      assert((specOpt.speculativeStats?.accepted ?? 0) > 0, 'Expected accepted > 0');

      if (opts.runDraftSkipSweep) {
        draftSkipSweep = [];
        for (const skip of [6, 12, 18]) {
          const dRes = await page.evaluate(
            (s) =>
              window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
                prompt: 'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.',
                maxNewTokens: 128,
                enableSpeculative: false,
                draftSkipLayers: s,
              }),
            skip,
          );
          const parityOk = dRes.responseText === chatTurn1.responseText;
          const entry = {
            skipLayers: skip,
            activeDraftLayers: 42 - skip,
            decodeTokPerSec: dRes.decodeTokPerSec,
            speedupVsBaseline: Number((dRes.decodeTokPerSec / chatTurn1.decodeTokPerSec).toFixed(3)),
            parityPassed: parityOk,
            speculativeStats: dRes.speculativeStats,
          };
          draftSkipSweep.push(entry);
          console.log(
            `  DraftSkip(${skip}, ${42 - skip}L): ${dRes.decodeTokPerSec} tok/s (${entry.speedupVsBaseline}x, acc=${dRes.speculativeStats?.acceptanceRatePct}%, parity=${parityOk})`,
          );
        }
      }

      if (opts.runNiah) {
        niah = await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('NIAH_32K'));
        console.log(`  32K NIAH: passed=${niah.passed}, ttft=${niah.ttftMs}ms (${niah.prefillTokPerSec} pref / ${niah.decodeTokPerSec} dec), ans=${JSON.stringify(niah.responseText)}`);
        assert(niah.passed, `32K NIAH failed: ${niah.responseText}`);
      }

      await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('DISPOSE'));
    }

    let balanced = null;
    if (opts.runBalanced) {
      const { smoothness: idleSmoothness } = await measureSmoothnessDuring(page, () => new Promise((r) => setTimeout(r, 2500)));
      const detectedHz = idleSmoothness.p50FrameMs > 0 ? Math.round(1000 / idleSmoothness.p50FrameMs) : 60;
      const { result: bLoad, smoothness: loadSmoothness } = await measureSmoothnessDuring(page, () =>
        page.evaluate(
          (cfg) => window.__GEMMA4_HOST__.callWorker('LOAD', cfg),
          { kvMode: opts.kvMode, maxContext: opts.maxContext, pacingMode: 'balanced', displayHz: detectedHz, flags: opts.flags },
        ),
      );
      if (opts.balancedOnly) {
        load = bLoad;
        golden = await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('RUN_GOLDEN'));
        console.log(`  Balanced Golden A=${JSON.stringify(golden.goldenA.text)} (id=${golden.goldenA.top0?.id}, val=${golden.goldenA.top0?.val}) | Golden B id=${golden.goldenB.top0?.id} val=${golden.goldenB.top0?.val} (${golden.goldenB.decodeTokPerSec} tok/s)`);
        assert(golden.goldenA.text === '4' && golden.goldenA.generatedIds?.[0] === 236812, `Golden A failed: ${JSON.stringify(golden.goldenA)}`);
        assert(golden.goldenB.top0?.id === 6721, `Golden B top0 failed: ${JSON.stringify(golden.goldenB.top0)}`);
      }
      const { result: bSweep, smoothness: sweepSmoothness } = await measureSmoothnessDuring(page, () =>
        page.evaluate((kvMode) => window.__GEMMA4_HOST__.callWorker('SWEEP', { contexts: [8192, 32768], decodeSteps: 32, kvMode }), opts.kvMode),
      );
      const { result: bChat, smoothness: chatSmoothness } = await measureSmoothnessDuring(page, () =>
        page.evaluate(() =>
          window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
            prompt: 'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.',
            maxNewTokens: 128,
            enableSpeculative: true,
            streamTokens: true,
          }),
        ),
      );
      const bTurn2 = await page.evaluate(() =>
        window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
          prompt: 'Summarize bullet 1 in one short sentence.',
          maxNewTokens: 64,
          enableSpeculative: true,
          appendFollowup: true,
          streamTokens: true,
        }),
      );
      assert(bTurn2.prefixReused, 'Expected balanced turn2 prefixReused=true');

      await page.evaluate(() => {
        window.__GEMMA4_HOST__.streamText = '';
        window.__ABORT_P__ = window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
          prompt: 'Write a very detailed 1000-word technical essay on GPU compute shaders.',
          maxNewTokens: 512,
          enableSpeculative: false,
          streamTokens: true,
        });
      });
      await page.waitForFunction(() => (window.__GEMMA4_HOST__.streamText?.length ?? 0) >= 24, { timeout: 60000 });
      const abortMetrics = await page.evaluate(async () => {
        const t = performance.now();
        window.__GEMMA4_HOST__.abortWorkerNow();
        const r = await window.__ABORT_P__;
        return { stoppedCleanly: Boolean(r.aborted), abortLatencyMs: Number((performance.now() - t).toFixed(2)), preservedPartialChars: r.responseText.length };
      });
      assert(abortMetrics.stoppedCleanly && abortMetrics.preservedPartialChars >= 20, 'Clean abort failed');
      await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('DISPOSE'));

      balanced = { detectedHz, idleSmoothness, load: bLoad, loadSmoothness, sweep: bSweep, sweepSmoothness, chatTurn1: bChat, chatSmoothness, turn2: bTurn2, abortMetrics };
      console.log(`  Balanced: chat=${bChat.decodeTokPerSec} tok/s @ ${chatSmoothness.avgFps}fps (jank=${chatSmoothness.jankPctOver33ms}%), sweep=${sweepSmoothness.avgFps}fps (jank=${sweepSmoothness.jankPctOver33ms}%), abort=${abortMetrics.abortLatencyMs}ms`);
    }

    assert(consoleErrors.length === 0, `Console errors: ${consoleErrors.join(' | ')}`);

    fs.mkdirSync(path.dirname(opts.out), { recursive: true });
    fs.writeFileSync(
      opts.out,
      JSON.stringify(
        {
          phase: opts.phase,
          recordedAt: new Date().toISOString(),
          config: { kvMode: opts.kvMode, maxContext: opts.maxContext, flags: opts.flags },
          load,
          golden,
          sweep,
          chatTurn1,
          speculativeCheck,
          draftSkipSweep,
          niah,
          balanced,
          consoleErrorCount: consoleErrors.length,
          status: sweep?.oomCeiling ? 'partial' : 'ok',
        },
        null,
        2,
      ),
    );
    console.log(`Saved ${opts.out}`);
  } finally {
    await context.close().catch(() => {});
    server.close();
    try { fs.closeSync(pleFd); } catch {}
    fs.rmSync(profileDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('bench-gemma4-phase failed:', err);
  process.exit(1);
});
