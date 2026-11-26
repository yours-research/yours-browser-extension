import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { build as esbuild } from 'esbuild';
import { chromium } from 'playwright';

const BUNDLE_DIR = path.resolve('model-bundle', 'gemma4-e4b');
const WASM_PATH = path.resolve(
  'crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
);
const PROFILE_DIR = path.resolve('.bench-profile-gemma4-worker');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

const PLE_Q_BYTES = 2688;
const PLE_S_BYTES = 84;
const PLE_ROW_BYTES = PLE_Q_BYTES + PLE_S_BYTES; // 2772
const PLE_SCALES_OFFSET = 704643072;

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
      // retry on Windows lock
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
      const scroller = document.getElementById('scroller');
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

function buildProbeHtml() {
  const paragraphs = Array.from(
    { length: 60 },
    (_, i) =>
      `<p>Scroll stress paragraph ${i + 1}: verifying continuous 144Hz browser compositing, canvas rendering, and responsive main-thread scrolling during Gemma 4 E4B Dedicated Worker WebGPU inference.</p>`,
  ).join('\n');

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Gemma 4 E4B Dedicated Worker Smoothness Probe</title>
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
      .stage { display: flex; align-items: center; gap: 16px; }
      #orbital-canvas { border: 1px solid #263248; border-radius: 8px; background: #131a26; }
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
      #scroller p { margin: 0 0 10px; color: #cbd5e1; }
      #stream-out {
        padding: 12px;
        border-radius: 8px;
        background: #111827;
        border: 1px solid #1f2937;
        min-height: 60px;
        white-space: pre-wrap;
        font-family: ui-monospace, monospace;
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <div class="wave-banner">
      <span>Gemma 4 Dedicated Worker + Compositor Probe</span>
      <span id="frame-badge">Probe Active</span>
    </div>
    <div class="stage">
      <canvas id="orbital-canvas" width="420" height="160"></canvas>
      <button id="probe-btn" type="button">Counter: 0</button>
    </div>
    <div id="stream-out"></div>
    <div id="scroller">${paragraphs}</div>
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
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, 52, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(cx + Math.cos(angle) * 52, cy + Math.sin(angle) * 52, 10, 0, Math.PI * 2);
          ctx.fill();
          const barWidth = ((Math.sin(angle * 2) + 1) / 2) * (canvas.width - 40);
          ctx.fillStyle = '#22c55e';
          ctx.fillRect(20, canvas.height - 16, barWidth, 6);
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
              this.streamText = msg.textSoFar ?? (this.streamText + (msg.delta ?? ''));
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
          w.onerror = (err) => {
            console.error('Worker error:', err.message);
          };
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
          if (this.worker) {
            this.worker.postMessage({ action: 'ABORT_ACTIVE' });
          }
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
      const globalBytesPerTok = kvMode === 'int8' ? 2048 : 4096;
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
          const tTok0 = performance.now();
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
          goldenA: { text: textA, generatedIds: resA.generatedIds, top0: resA.step0Top5Logits?.[0] },
          goldenB: { text: textB, top0: resB.step0Top5Logits?.[0], decodeTokPerSec: resB.decodeTokPerSec },
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
            let nextTok = pref.nextTokenId;
            const decSteps = payload.decodeSteps ?? 32;
            const tDec0 = performance.now();
            for (let s = 0; s < decSteps; s++) {
              const d = await engine.decodeStep(nextTok);
              nextTok = d.nextTokenId;
            }
            const decodeMs = Math.max(0.1, performance.now() - tDec0);
            const decodeTokPerSec = Number(((decSteps * 1000) / decodeMs).toFixed(2));
            const bw = computeBandwidth(decodeTokPerSec, ctxLen + decSteps, payload.kvMode ?? 'int8');
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

      if (action === 'CHAT_TURN') {
        const isFollowup = Boolean(payload.appendFollowup && conversationTokenIds.length > 0);
        let promptIds;
        if (isFollowup) {
          const turnStr = '<turn|>\\n<|turn>user\\n' + payload.prompt + '<turn|>\\n<|turn>model\\n';
          const enc = tokenizer(turnStr, { add_special_tokens: false });
          const suffix = Array.from(enc.input_ids.data, Number);
          promptIds = [...conversationTokenIds, ...suffix];
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
        if (engine) {
          engine.dispose();
          engine = null;
        }
        return { ok: true };
      }

      throw new Error('Unknown worker action: ' + action);
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
    stdin: {
      contents: workerSource,
      resolveDir: process.cwd(),
      loader: 'js',
    },
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
  const range = req.headers.range;
  const mime = getMimeType(filePath);
  if (range) {
    const m = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (m) {
      const start = Number(m[1]);
      const end = m[2] ? Math.min(stat.size - 1, Number(m[2])) : stat.size - 1;
      res.writeHead(206, {
        'Content-Type': mime,
        'Content-Length': end - start + 1,
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
      });
      fs.createReadStream(filePath, { start, end }).pipe(res);
      return;
    }
  }
  res.writeHead(200, {
    'Content-Type': mime,
    'Content-Length': stat.size,
    'Accept-Ranges': 'bytes',
  });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  fs.createReadStream(filePath).pipe(res);
}

async function measureSmoothnessDuring(page, actionPromiseFactory) {
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
  try {
    result = await actionPromiseFactory();
  } finally {
    running = false;
    await pumper.catch(() => {});
  }
  const smoothness = await page.evaluate(() => window.__FPS_PROBE__.stop());
  return { result, smoothness };
}

async function main() {
  assert(fs.existsSync(path.join(BUNDLE_DIR, 'manifest.json')), 'Missing model-bundle/gemma4-e4b/manifest.json');
  assert(fs.existsSync(WASM_PATH), 'Missing WASM build. Run npm run build:wasm first.');

  cleanProfileDir();
  console.log('Bundling Gemma 4 E4B Dedicated Worker harness (Tracks G4 & G5)...');
  const workerJs = await buildWorkerBundle();
  const probeHtml = buildProbeHtml();
  const pleFd = fs.openSync(path.join(BUNDLE_DIR, 'ple_tokens_q2.bin'), 'r');

  const server = http.createServer((req, res) => {
    const urlObj = new URL(req.url, 'http://127.0.0.1');
    const pathname = decodeURIComponent(urlObj.pathname);
    if (pathname === '/' || pathname === '/probe.html' || pathname === '/index.html') {
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
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': outBuf.byteLength,
      });
      res.end(outBuf);
      return;
    }
    if (pathname.startsWith('/models/gemma4-e4b/')) {
      const rel = pathname.slice('/models/gemma4-e4b/'.length);
      serveStaticFile(req, res, path.join(BUNDLE_DIR, rel));
      return;
    }
    res.writeHead(404);
    res.end('Not found');
  });

  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;

  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    viewport: { width: 1080, height: 920 },
    args: [
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });

  const consoleErrors = [];
  try {
    const page = context.pages()[0] ?? (await context.newPage());
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });
    page.on('pageerror', (e) => consoleErrors.push(e.message));

    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => Boolean(window.__GEMMA4_HOST__ && window.__FPS_PROBE__), { timeout: 20000 });

    // Measure idle display Hz cleanly first
    const { smoothness: idleSmoothness } = await measureSmoothnessDuring(
      page,
      () => new Promise((r) => setTimeout(r, 3000)),
    );
    const detectedHz = idleSmoothness.p50FrameMs > 0 ? Math.round(1000 / idleSmoothness.p50FrameMs) : 60;
    console.log(`\n  Idle Smoothness (3s): ${JSON.stringify(idleSmoothness)} (detected ~${detectedHz}Hz)`);

    // =========================================================================
    // Track G4: Dedicated Worker Max-Speed 128K + Speculative Decoding Parity
    // =========================================================================
    console.log('\n=== [g04-gemma4-wgsl-max-speed-128k] Dedicated Worker Max-Speed ===');
    const g4Load = await page.evaluate(() =>
      window.__GEMMA4_HOST__.callWorker('LOAD', {
        kvMode: 'int8',
        maxContext: 131072,
        pacingMode: 'max-speed',
        displayHz: 144,
      }),
    );
    console.log(`  Worker Load: ${g4Load.coldLoadMs}ms (tier=${g4Load.computeTier}, vram=${g4Load.vramPlan.totalVramMb} MiB)`);

    const g4Golden = await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('RUN_GOLDEN'));
    console.log(`  Golden A=${JSON.stringify(g4Golden.goldenA.text)}, Golden B=${g4Golden.goldenB.decodeTokPerSec} tok/s`);
    assert(g4Golden.goldenA.text === '4', `G4 Golden A failed: ${g4Golden.goldenA.text}`);
    assert(g4Golden.goldenB.top0?.id === 6721, `G4 Golden B top0 failed: ${JSON.stringify(g4Golden.goldenB.top0)}`);

    const g4Sweep = await page.evaluate(() =>
      window.__GEMMA4_HOST__.callWorker('SWEEP', {
        contexts: [1024, 8192, 32768, 65536, 98304, 131072],
        decodeSteps: 32,
        kvMode: 'int8',
      }),
    );
    for (const pt of g4Sweep.points) {
      console.log(
        pt.status === 'ok'
          ? `  ctx=${pt.targetTokens}: prefill=${pt.prefillTokPerSec} tok/s, decode=${pt.decodeTokPerSec} tok/s (${pt.achievedBandwidthGBs} GB/s)`
          : `  ctx=${pt.targetTokens}: FAILED (${pt.error})`,
      );
    }

    const chatTurn1 = await page.evaluate(() =>
      window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
        prompt: 'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.',
        maxNewTokens: 128,
        enableSpeculative: false,
      }),
    );
    console.log(`  Chat Turn 1 (${chatTurn1.decodeTokPerSec} tok/s, TTFT=${chatTurn1.ttftMs}ms): ${JSON.stringify(chatTurn1.responseText.slice(0, 120))}`);

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
    const speculativeCheck = {
      prompt: specPrompt,
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
      responsePreview: specOpt.responseText.slice(0, 200),
    };
    console.log('  Speculative Parity:', JSON.stringify(speculativeCheck));
    assert(parityPassed, `G4 Speculative parity failed!\nBase: ${specBase.responseText}\nSpec: ${specOpt.responseText}`);
    assert((specOpt.speculativeStats?.accepted ?? 0) > 0, 'Expected speculativeStats.accepted > 0');

    await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('DISPOSE'));

    const g04Path = path.join(RUNS_DIR, 'g04-gemma4-wgsl-max-speed-128k.json');
    fs.writeFileSync(
      g04Path,
      JSON.stringify(
        {
          runId: 'g04-gemma4-wgsl-max-speed-128k',
          track: 'gemma4-track4-max-speed',
          modelId: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
          recordedAt: new Date().toISOString(),
          config: {
            kvMode: 'int8',
            maxContext: 131072,
            gpuPacingMode: 'max-speed',
            executionHost: 'worker',
          },
          coldLoad: g4Load,
          golden: g4Golden,
          sweep: { points: g4Sweep.points, oomCeiling: g4Sweep.oomCeiling },
          chatTurn1,
          speculativeCheck,
          consoleErrorCount: consoleErrors.length,
          consoleErrors,
          status: g4Sweep.oomCeiling ? 'partial' : 'ok',
        },
        null,
        2,
      ),
    );
    console.log(`Saved ${g04Path}`);

    // =========================================================================
    // Track G5: Dedicated Worker Balanced (144Hz Compositor Smoothness + Prefix + Abort)
    // =========================================================================
    console.log('\n=== [g05-gemma4-wgsl-worker-balanced-smoothness] Dedicated Worker Balanced ===');
    const { result: g5Load, smoothness: loadSmoothness } = await measureSmoothnessDuring(page, () =>
      page.evaluate(
        (hz) =>
          window.__GEMMA4_HOST__.callWorker('LOAD', {
            kvMode: 'int8',
            maxContext: 131072,
            pacingMode: 'balanced',
            displayHz: hz,
          }),
        detectedHz,
      ),
    );
    console.log('  Load Smoothness:', JSON.stringify(loadSmoothness));

    const { result: g5Sweep, smoothness: sweepSmoothness } = await measureSmoothnessDuring(page, () =>
      page.evaluate(() =>
        window.__GEMMA4_HOST__.callWorker('SWEEP', {
          contexts: [8192, 32768],
          decodeSteps: 32,
          kvMode: 'int8',
        }),
      ),
    );
    console.log('  Sweep Smoothness:', JSON.stringify(sweepSmoothness));

    const { result: g5ChatTurn1, smoothness: chatSmoothness } = await measureSmoothnessDuring(page, () =>
      page.evaluate(() =>
        window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
          prompt: 'List 4 benefits of running local LLMs in a Dedicated Web Worker in 4 short bullets.',
          maxNewTokens: 128,
          enableSpeculative: true,
          streamTokens: true,
        }),
      ),
    );
    console.log(`  G5 Chat Turn 1: ${g5ChatTurn1.decodeTokPerSec} tok/s, smoothness=${JSON.stringify(chatSmoothness)}`);

    const g5Turn2 = await page.evaluate(() =>
      window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
        prompt: 'Summarize bullet 1 in one short sentence.',
        maxNewTokens: 64,
        enableSpeculative: true,
        appendFollowup: true,
        streamTokens: true,
      }),
    );
    console.log(`  G5 Turn 2: prefixReused=${g5Turn2.prefixReused} (${g5Turn2.prefixReusedTokens} toks), ${g5Turn2.decodeTokPerSec} tok/s`);
    assert(g5Turn2.prefixReused === true, 'Expected prefixReused=true on Turn 2');

    // Abort latency test
    await page.evaluate(() => {
      window.__GEMMA4_HOST__.streamText = '';
      window.__GEMMA4_ABORT_PROMISE__ = window.__GEMMA4_HOST__.callWorker('CHAT_TURN', {
        prompt: 'Write a very detailed 1000-word technical essay on GPU compute shaders.',
        maxNewTokens: 512,
        enableSpeculative: false,
        streamTokens: true,
      });
    });
    await page.waitForFunction(() => (window.__GEMMA4_HOST__.streamText?.length ?? 0) >= 24, { timeout: 60000 });
    const abortMetrics = await page.evaluate(async () => {
      const tClick = performance.now();
      window.__GEMMA4_HOST__.abortWorkerNow();
      const res = await window.__GEMMA4_ABORT_PROMISE__;
      const abortLatencyMs = Number((performance.now() - tClick).toFixed(2));
      return {
        stoppedCleanly: Boolean(res.aborted),
        abortLatencyMs,
        preservedPartialChars: res.responseText.length,
        partialPreview: res.responseText.slice(0, 140),
      };
    });
    console.log('  G5 Abort Metrics:', JSON.stringify(abortMetrics));
    assert(abortMetrics.stoppedCleanly && abortMetrics.preservedPartialChars >= 20, 'Clean abort failed');

    await page.evaluate(() => window.__GEMMA4_HOST__.callWorker('DISPOSE'));

    const g05Path = path.join(RUNS_DIR, 'g05-gemma4-wgsl-worker-balanced-smoothness.json');
    fs.writeFileSync(
      g05Path,
      JSON.stringify(
        {
          runId: 'g05-gemma4-wgsl-worker-balanced-smoothness',
          track: 'gemma4-track5-balanced',
          modelId: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
          recordedAt: new Date().toISOString(),
          config: {
            kvMode: 'int8',
            maxContext: 131072,
            gpuPacingMode: 'balanced',
            executionHost: 'worker',
            displayHz: detectedHz,
          },
          idleSmoothness,
          coldLoad: g5Load,
          loadSmoothness,
          sweep: { points: g5Sweep.points, oomCeiling: g5Sweep.oomCeiling },
          sweepSmoothness,
          chatTurn1: g5ChatTurn1,
          chatSmoothness,
          turn2: g5Turn2,
          abortMetrics,
          consoleErrorCount: consoleErrors.length,
          consoleErrors,
          status: consoleErrors.length === 0 && !g5Sweep.oomCeiling ? 'ok' : 'partial',
        },
        null,
        2,
      ),
    );
    console.log(`Saved ${g05Path}`);
    assert(consoleErrors.length === 0, `Console errors: ${consoleErrors.join(' | ')}`);
  } finally {
    await context.close().catch(() => {});
    server.close();
    try { fs.closeSync(pleFd); } catch {}
    fs.rmSync(PROFILE_DIR, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('Gemma 4 Track G4/G5 Worker failed:', err);
  process.exit(1);
});
