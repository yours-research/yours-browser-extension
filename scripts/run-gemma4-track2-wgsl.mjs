import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { build as esbuild } from 'esbuild';
import { chromium } from 'playwright';

const BUNDLE_DIR = path.resolve('model-bundle', 'gemma4-e4b');
const WASM_PATH = path.resolve(
  'crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
);
const PROFILE_DIR = path.resolve('.bench-profile-gemma4-wgsl');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

const PLE_Q_BYTES = 2688;
const PLE_S_BYTES = 84;
const PLE_ROW_BYTES = PLE_Q_BYTES + PLE_S_BYTES; // 2772
const PLE_SCALES_OFFSET = 704643072; // 262144 * 2688

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

async function buildHarnessBundle() {
  const entrySource = `
    import { env, AutoTokenizer } from '@huggingface/transformers';
    import { Gemma4WgslEngine, GEMMA4_PLE_ROW_BYTES } from './src/offscreen/gemma4WgslEngine.ts';

    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = '/models/';
    env.useBrowserCache = false;

    async function pleRowFetcher(tokenIds) {
      const res = await fetch('/ple-batch?ids=' + tokenIds.join(','));
      if (!res.ok) throw new Error('PLE batch fetch failed: ' + res.status);
      return new Uint8Array(await res.arrayBuffer());
    }

    function computeBandwidth(decodeTokPerSec, seqPos, kvMode) {
      // Gemma 4 active weights per decode token:
      // 42 * Q/O/MLP/PLE + 24 * KV_proj + lm_head(Q2) ~ 2.07 GiB - shared KV skipped + active KV read
      const activeWeightsBytes = 1.98e9;
      // 20 sliding donors (max 512 tokens * D=256 * 2(KV) * 2H * 2B = 40.96 MB max, active up to 512)
      const slideToks = Math.min(seqPos, 512);
      const slideBytes = 35 * 2 * 2 * slideToks * 256 * 2;
      // 7 full layers (or 4 donor KV reads? Actually 7 attention passes read donor KV: FP16 4B*512*2=4096B/tok, INT8 2B*512*2=2048B/tok)
      const globalBytesPerTok = kvMode === 'int8' ? 2048 : 4096;
      const globalBytes = 7 * seqPos * globalBytesPerTok;
      const totalBytes = activeWeightsBytes + slideBytes + globalBytes;
      const achievedBandwidthGBs = Number(((totalBytes * decodeTokPerSec) / 1e9).toFixed(2));
      const bandwidthUtilizationPct = Number(((achievedBandwidthGBs / 936.0) * 100).toFixed(2));
      return { achievedBandwidthGBs, bandwidthUtilizationPct };
    }

    window.__GEMMA4_WGSL__ = {
      async runTracksG2AndG3() {
        const tTok0 = performance.now();
        const tokenizer = await AutoTokenizer.from_pretrained('gemma4-e4b');
        const jinjaRes = await fetch('/models/gemma4-e4b/chat_template.jinja');
        if (jinjaRes.ok) {
          tokenizer.chat_template = await jinjaRes.text();
        }
        const tokenizerMs = Math.round(performance.now() - tTok0);
        const eosIds = new Set([1, 106]);

        const encodeChatPrompt = (promptText) => {
          const formatted = tokenizer.apply_chat_template(
            [{ role: 'user', content: promptText }],
            { tokenize: false, add_generation_prompt: true, enable_thinking: false },
          );
          const enc = tokenizer(formatted, { add_special_tokens: false });
          return {
            formatted,
            ids: Array.from(enc.input_ids.data, (x) => Number(x)),
          };
        };

        const runGoldenSuite = async (engine, ortBaseline, strictLogitParity) => {
          engine.resetContext();
          const pA = encodeChatPrompt('What is 2 + 2? Answer with just the number.');
          const resA = await engine.generate(pA.ids, { maxNewTokens: 16, eosIds });
          const textA = tokenizer.decode(resA.generatedIds, { skip_special_tokens: true }).trim();
          console.log('[G4-WGSL] Golden A: text=' + JSON.stringify(textA) + ' ids=' + JSON.stringify(resA.generatedIds) + ' top0=' + JSON.stringify(resA.step0Top5Logits?.[0]));

          if (textA !== '4') {
            throw new Error('Golden A mismatch: expected "4", got ' + JSON.stringify(textA) + ', top5=' + JSON.stringify(resA.step0Top5Logits));
          }
          if (resA.generatedIds[0] !== 236812) {
            throw new Error('Golden A token0 mismatch: expected 236812, got ' + resA.generatedIds[0]);
          }
          if (strictLogitParity) {
            const top0Val = resA.step0Top5Logits?.[0]?.val ?? NaN;
            if (!Number.isFinite(top0Val) || Math.abs(top0Val - ortBaseline.goldenA.step0Top5Logits[0].val) > 1.0) {
              throw new Error('Golden A logit delta too large: wgsl=' + top0Val + ' ort=' + ortBaseline.goldenA.step0Top5Logits[0].val);
            }
          }

          engine.resetContext();
          const pB = encodeChatPrompt('Explain WebGPU in one short sentence.');
          const resB = await engine.generate(pB.ids, { maxNewTokens: 32, eosIds });
          const textB = tokenizer.decode(resB.generatedIds, { skip_special_tokens: true }).trim();
          console.log('[G4-WGSL] Golden B: text=' + JSON.stringify(textB) + ' top0=' + JSON.stringify(resB.step0Top5Logits?.[0]) + ' dec=' + resB.decodeTokPerSec);

          if (resB.step0Top5Logits?.[0]?.id !== 6721) {
            throw new Error('Golden B top0 id mismatch: expected 6721, got ' + JSON.stringify(resB.step0Top5Logits?.[0]));
          }
          if (strictLogitParity) {
            const top0Val = resB.step0Top5Logits?.[0]?.val ?? NaN;
            if (!Number.isFinite(top0Val) || Math.abs(top0Val - ortBaseline.goldenB.step0Top5Logits[0].val) > 1.0) {
              throw new Error('Golden B logit delta too large: wgsl=' + top0Val + ' ort=' + ortBaseline.goldenB.step0Top5Logits[0].val);
            }
          }

          return {
            goldenA: { ...resA, promptText: 'What is 2 + 2? Answer with just the number.', decodedText: textA },
            goldenB: { ...resB, promptText: 'Explain WebGPU in one short sentence.', decodedText: textB },
          };
        };

        const runSweep = async (engine, contexts, kvMode) => {
          const points = [];
          let oomCeiling = null;
          for (const ctxLen of contexts) {
            try {
              engine.resetContext();
              const ids = new Array(ctxLen);
              for (let i = 0; i < ctxLen; i++) ids[i] = 1000 + (i % 500);
              const pref = await engine.prefillTokens(ids, { chunkSize: 64, returnLogits: false });
              let nextTok = pref.nextTokenId;
              const decSteps = 32;
              const tDec0 = performance.now();
              for (let s = 0; s < decSteps; s++) {
                const d = await engine.decodeStep(nextTok);
                nextTok = d.nextTokenId;
              }
              const decodeMs = Math.max(0.1, performance.now() - tDec0);
              const decodeTokPerSec = Number(((decSteps * 1000) / decodeMs).toFixed(2));
              const bw = computeBandwidth(decodeTokPerSec, ctxLen + decSteps, kvMode);
              console.log('[G4-WGSL] Sweep ' + kvMode + ' ctx=' + ctxLen + ' prefill=' + pref.prefillTokPerSec + ' tok/s (' + pref.prefillMs + 'ms) dec=' + decodeTokPerSec + ' tok/s (' + bw.achievedBandwidthGBs + ' GB/s)');
              points.push({
                targetTokens: ctxLen,
                actualPromptTokens: ctxLen,
                decodeTokens: decSteps,
                chunkSize: 64,
                prefillMs: pref.prefillMs,
                ttftMs: pref.prefillMs,
                prefillTokPerSec: pref.prefillTokPerSec,
                pleFetchMs: pref.pleFetchMs,
                pleDequantMs: pref.pleDequantMs,
                decodeTokPerSec,
                ...bw,
                status: 'ok',
              });
            } catch (err) {
              const msg = err?.message || String(err);
              console.log('[G4-WGSL] Sweep FAILED ctx=' + ctxLen + ': ' + msg);
              oomCeiling = { atContext: ctxLen, error: msg };
              points.push({ targetTokens: ctxLen, status: 'failed', error: msg });
              break;
            }
          }
          return { points, oomCeiling };
        };

        const run32kNiah = async (engine) => {
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
          const text = tokenizer.decode(res.generatedIds, { skip_special_tokens: true }).trim();
          const passed = text.includes(passkey);
          console.log('[G4-WGSL] 32K NIAH passed=' + passed + ' ans=' + JSON.stringify(text) + ' ttft=' + res.ttftMs + 'ms dec=' + res.decodeTokPerSec + ' tok/s');
          return {
            targetTokens,
            actualTokens: promptIds.length,
            needleDepthPct: 0.5,
            passkey,
            responseText: text,
            passed,
            ttftMs: res.ttftMs,
            prefillTokPerSec: res.prefillTokPerSec,
            decodeTokPerSec: res.decodeTokPerSec,
          };
        };

        const ortRes = await fetch('/g01-baseline.json');
        const ortBaseline = await ortRes.json();

        // === Track G2: Native WGSL FP16 KV (1K..32K) + Strict Logit Parity Gate ===
        const engFp16 = new Gemma4WgslEngine();
        const g2Load = await engFp16.load({
          baseUrl: '/models/gemma4-e4b/',
          wasmUrl: '/wasm/tinygpt_wgsl_engine.wasm',
          kvMode: 'fp16',
          maxContext: 32768,
          pacingMode: 'max-speed',
          pleRowFetcher,
        });
        const g2Golden = await runGoldenSuite(engFp16, ortBaseline, true);
        const g2Sweep = await runSweep(engFp16, [1024, 4096, 8192, 16384, 32768], 'fp16');
        engFp16.dispose();

        // === Track G3: Native WGSL INT8 Global KV + 1024 Ring Sliding (1K..128K + 32K NIAH) ===
        const engInt8 = new Gemma4WgslEngine();
        const g3Load = await engInt8.load({
          baseUrl: '/models/gemma4-e4b/',
          wasmUrl: '/wasm/tinygpt_wgsl_engine.wasm',
          kvMode: 'int8',
          maxContext: 131072,
          pacingMode: 'max-speed',
          pleRowFetcher,
        });
        const g3Golden = await runGoldenSuite(engInt8, ortBaseline, false);
        const g3Sweep = await runSweep(engInt8, [1024, 8192, 32768, 65536, 98304, 131072], 'int8');
        const g3Niah = await run32kNiah(engInt8);
        engInt8.dispose();

        return {
          tokenizerMs,
          g2: {
            load: g2Load,
            ...g2Golden,
            sweep: g2Sweep,
          },
          g3: {
            load: g3Load,
            ...g3Golden,
            sweep: g3Sweep,
            niah: g3Niah,
          },
        };
      },
    };
  `;

  const result = await esbuild({
    stdin: {
      contents: entrySource,
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

async function main() {
  assert(fs.existsSync(path.join(BUNDLE_DIR, 'manifest.json')), 'Missing model-bundle/gemma4-e4b/manifest.json');
  assert(fs.existsSync(WASM_PATH), 'Missing WASM build. Run npm run build:wasm first.');
  const ortBaselinePath = path.join(RUNS_DIR, 'g01-gemma4-ort-baseline.json');
  assert(fs.existsSync(ortBaselinePath), 'Missing g01-gemma4-ort-baseline.json');

  cleanProfileDir();
  console.log('Bundling Gemma 4 E4B Native WGSL Track G2 & G3 harness...');
  const harnessJs = await buildHarnessBundle();
  const pleFd = fs.openSync(path.join(BUNDLE_DIR, 'ple_tokens_q2.bin'), 'r');

  const html = `<!doctype html><html><head><meta charset="utf-8" /><title>Gemma 4 E4B WGSL G2/G3</title></head>
  <body><div id="status">Running Gemma 4 WGSL...</div><script type="module" src="/harness.js"></script></body></html>`;

  const server = http.createServer((req, res) => {
    const urlObj = new URL(req.url, 'http://127.0.0.1');
    const pathname = decodeURIComponent(urlObj.pathname);
    if (pathname === '/' || pathname === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
      return;
    }
    if (pathname === '/favicon.ico') {
      res.writeHead(204);
      res.end();
      return;
    }
    if (pathname === '/harness.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8' });
      res.end(harnessJs);
      return;
    }
    if (pathname === '/wasm/tinygpt_wgsl_engine.wasm') {
      serveStaticFile(req, res, WASM_PATH);
      return;
    }
    if (pathname === '/g01-baseline.json') {
      serveStaticFile(req, res, ortBaselinePath);
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
    viewport: { width: 1080, height: 800 },
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
      if (m.text().includes('[G4-WGSL]')) console.log(m.text());
    });
    page.on('pageerror', (e) => consoleErrors.push(e.message));

    console.log('\n=== Running Track G2 (FP16 KV + Logit Parity) & Track G3 (INT8 Global KV 128K + 32K NIAH) ===');
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => Boolean(window.__GEMMA4_WGSL__), { timeout: 20000 });

    const suite = await page.evaluate(() => window.__GEMMA4_WGSL__.runTracksG2AndG3());

    fs.mkdirSync(RUNS_DIR, { recursive: true });
    const g02Payload = {
      runId: 'g02-gemma4-wgsl-fp16kv',
      track: 'gemma4-track2-wgsl-fp16kv',
      modelId: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
      recordedAt: new Date().toISOString(),
      tokenizerMs: suite.tokenizerMs,
      ...suite.g2,
      consoleErrorCount: consoleErrors.length,
      consoleErrors,
      status: suite.g2.sweep.oomCeiling ? 'partial' : 'ok',
    };
    const g02Path = path.join(RUNS_DIR, 'g02-gemma4-wgsl-fp16kv.json');
    fs.writeFileSync(g02Path, JSON.stringify(g02Payload, null, 2));
    console.log(`Saved ${g02Path}`);

    const g03Payload = {
      runId: 'g03-gemma4-wgsl-int8kv-128k',
      track: 'gemma4-track3-wgsl-int8kv-128k',
      modelId: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
      recordedAt: new Date().toISOString(),
      tokenizerMs: suite.tokenizerMs,
      ...suite.g3,
      consoleErrorCount: consoleErrors.length,
      consoleErrors,
      status: suite.g3.sweep.oomCeiling || !suite.g3.niah.passed ? 'partial' : 'ok',
    };
    const g03Path = path.join(RUNS_DIR, 'g03-gemma4-wgsl-int8kv-128k.json');
    fs.writeFileSync(g03Path, JSON.stringify(g03Payload, null, 2));
    console.log(`Saved ${g03Path}`);

    assert(consoleErrors.length === 0, `Console errors encountered: ${consoleErrors.join(' | ')}`);
    assert(suite.g3.niah.passed, 'G3 32K NIAH did not pass');
  } finally {
    await context.close().catch(() => {});
    server.close();
    try { fs.closeSync(pleFd); } catch {}
    fs.rmSync(PROFILE_DIR, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('Gemma 4 Track G2/G3 failed:', err);
  process.exit(1);
});
