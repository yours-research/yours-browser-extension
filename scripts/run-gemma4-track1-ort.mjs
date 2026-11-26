import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { build as esbuild } from 'esbuild';
import { chromium } from 'playwright';

const HF_MODEL_DIR = path.resolve('model-bundle', '.hf-tmp-gemma4-qat');
const ORT_DIST_DIR = path.resolve('node_modules', 'onnxruntime-web', 'dist');
const PROFILE_DIR = path.resolve('.bench-profile-gemma4');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

function cleanProfileDir() {
  for (let i = 0; i < 6; i += 1) {
    try {
      if (fs.existsSync(PROFILE_DIR)) {
        fs.rmSync(PROFILE_DIR, { recursive: true, force: true, maxRetries: 4, retryDelay: 200 });
      }
      fs.mkdirSync(PROFILE_DIR, { recursive: true });
      return;
    } catch {
      // retry briefly on Windows file locks
    }
  }
}

async function buildBrowserHarnessJs() {
  const entrySource = `
    import { env, AutoTokenizer, AutoModelForCausalLM, Tensor } from '@huggingface/transformers';

    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = '/models/';
    env.useBrowserCache = false;
    env.backends.onnx.wasm.wasmPaths = '/ort/';
    env.backends.onnx.wasm.proxy = false;

    function extractLastStepLogits(logitsTensor) {
      const dims = logitsTensor.dims;
      const vocabSize = dims[dims.length - 1];
      const seqLen = dims.length === 3 ? dims[1] : 1;
      const offset = (seqLen - 1) * vocabSize;
      const data = logitsTensor.data;
      const isF16 = data instanceof Uint16Array;
      const toNum = isF16 ? (idx) => {
        const h = data[idx];
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7c00) >> 10;
        const f = h & 0x03ff;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        if (e === 0x1f) return f ? NaN : s ? -Infinity : Infinity;
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
      } : (idx) => data[idx];

      let bestIdx = 0;
      let bestVal = -Infinity;
      const top = [];
      for (let i = 0; i < vocabSize; i++) {
        const v = toNum(offset + i);
        if (v > bestVal) {
          bestVal = v;
          bestIdx = i;
        }
        if (top.length < 5 || v > top[top.length - 1].val) {
          top.push({ id: i, val: Number(v.toFixed(4)) });
          top.sort((a, b) => b.val - a.val);
          if (top.length > 5) top.length = 5;
        }
      }
      return { bestIdx, top5: top };
    }

    function disposePast(pkv) {
      if (!pkv) return;
      if (typeof pkv.dispose === 'function') {
        try { pkv.dispose(); } catch {}
      } else if (typeof pkv === 'object') {
        for (const v of Object.values(pkv)) {
          try { v?.dispose?.(); } catch {}
        }
      }
    }

    function updatePastKeyValues(outputs, prevPast) {
      const next = Object.create(null);
      for (const [k, v] of Object.entries(outputs)) {
        if (k.startsWith('present.')) {
          next[k.replace(/^present\\./, 'past_key_values.')] = v;
        }
      }
      if (prevPast) {
        for (const [k, oldTensor] of Object.entries(prevPast)) {
          const newTensor = next[k];
          if (oldTensor && oldTensor !== newTensor) {
            try { oldTensor.dispose?.(); } catch {}
          }
          prevPast[k] = newTensor;
        }
        return prevPast;
      }
      return next;
    }

    window.__GEMMA4_ORT__ = {
      async runSuite() {
        const logs = [];
        const tLoadStart = performance.now();
        const tTok0 = performance.now();
        const tokenizer = await AutoTokenizer.from_pretrained('gemma4');
        const jinjaRes = await fetch('/models/gemma4/chat_template.jinja');
        if (jinjaRes.ok) {
          tokenizer.chat_template = await jinjaRes.text();
        }
        const tokenizerMs = Math.round(performance.now() - tTok0);

        let deviceConfigUsed = { embed_tokens: 'webgpu', decoder_model_merged: 'webgpu' };
        let webgpuEmbedError = null;
        let model = null;
        const tModel0 = performance.now();

        try {
          model = await AutoModelForCausalLM.from_pretrained('gemma4', {
            device: 'webgpu',
            dtype: { embed_tokens: 'q2f16', decoder_model_merged: 'q2f16' },
            local_files_only: true,
          });
        } catch (err) {
          webgpuEmbedError = err?.message || String(err);
          logs.push('Pure WebGPU fallback -> wasm embed_tokens: ' + webgpuEmbedError);
          deviceConfigUsed = { embed_tokens: 'wasm', decoder_model_merged: 'webgpu' };
          model = await AutoModelForCausalLM.from_pretrained('gemma4', {
            device: deviceConfigUsed,
            dtype: { embed_tokens: 'q2f16', decoder_model_merged: 'q2f16' },
            local_files_only: true,
          });
        }

        const modelLoadMs = Math.round(performance.now() - tModel0);
        const totalColdLoadMs = Math.round(performance.now() - tLoadStart);

        const eosIds = new Set(
          Array.isArray(model.config?.eos_token_id)
            ? model.config.eos_token_id
            : [model.config?.eos_token_id ?? 1, 106],
        );

        const keepLastLogit = new Tensor('int64', BigInt64Array.from([1n]), []);

        const runPromptGreedy = async (promptText, maxTokens) => {
          const formatted = tokenizer.apply_chat_template(
            [{ role: 'user', content: promptText }],
            { tokenize: false, add_generation_prompt: true, enable_thinking: false },
          );
          const encoded = tokenizer(formatted, { add_special_tokens: false });
          const rawIds = Array.from(encoded.input_ids.data, (x) => Number(x));
          let curIds = rawIds.map((x) => BigInt(x));

          let past_key_values = null;
          let fullMaskLen = curIds.length;
          let posStart = 0;

          const t0 = performance.now();
          let stepInput = new Tensor('int64', BigInt64Array.from(curIds), [1, curIds.length]);
          let attnMask = new Tensor('int64', new BigInt64Array(fullMaskLen).fill(1n), [1, fullMaskLen]);
          let posIds = new Tensor('int64', BigInt64Array.from({ length: curIds.length }, (_, i) => BigInt(posStart + i)), [1, curIds.length]);

          let out = await model.forward({
            input_ids: stepInput,
            attention_mask: attnMask,
            position_ids: posIds,
            past_key_values,
            num_logits_to_keep: keepLastLogit,
          });
          const prefillMs = Number((performance.now() - t0).toFixed(1));
          const ttftMs = prefillMs;
          const step0 = extractLastStepLogits(out.logits);
          try { out.logits?.dispose?.(); } catch {}
          past_key_values = updatePastKeyValues(out, past_key_values);
          console.log('[ORT-TRACE] prompt prefill OK len=' + rawIds.length + ' pkv22=' + JSON.stringify(past_key_values?.['past_key_values.22.key']?.dims) + ' pkv23=' + JSON.stringify(past_key_values?.['past_key_values.23.key']?.dims));

          const generatedIds = [];
          let nextTok = step0.bestIdx;
          const tDec0 = performance.now();

          for (let step = 0; step < maxTokens; step++) {
            if (eosIds.has(nextTok)) break;
            generatedIds.push(nextTok);
            if (generatedIds.length >= maxTokens) break;

            fullMaskLen += 1;
            posStart += (step === 0 ? curIds.length : 1);
            const tokT = new Tensor('int64', BigInt64Array.from([BigInt(nextTok)]), [1, 1]);
            const maskT = new Tensor('int64', new BigInt64Array(fullMaskLen).fill(1n), [1, fullMaskLen]);
            const posT = new Tensor('int64', BigInt64Array.from([BigInt(posStart)]), [1, 1]);

            const nextOut = await model.forward({
              input_ids: tokT,
              attention_mask: maskT,
              position_ids: posT,
              past_key_values,
              num_logits_to_keep: keepLastLogit,
            });
            const info = extractLastStepLogits(nextOut.logits);
            try { nextOut.logits?.dispose?.(); } catch {}
            past_key_values = updatePastKeyValues(nextOut, past_key_values);
            nextTok = info.bestIdx;
          }

          const decodeMs = Number(Math.max(1, performance.now() - tDec0).toFixed(1));
          const decodeTokPerSec = generatedIds.length > 0 ? Number(((generatedIds.length * 1000) / decodeMs).toFixed(2)) : 0;
          console.log('[ORT-TRACE] prompt decode complete tokens=' + generatedIds.length + ' dec=' + decodeTokPerSec + ' tok/s');
          disposePast(past_key_values);

          const decodedText = tokenizer.decode(generatedIds, { skip_special_tokens: true }).trim();
          return {
            promptText,
            formattedPrompt: formatted,
            inputIds: rawIds,
            promptTokens: rawIds.length,
            prefillMs,
            step0Top5Logits: step0.top5,
            generatedIds,
            decodeTokens: generatedIds.length,
            decodeMs,
            decodedText,
            ttftMs,
            decodeTokPerSec,
          };
        };

        const goldenA = await runPromptGreedy('What is 2 + 2? Answer with just the number.', 16);
        const goldenB = await runPromptGreedy('Explain WebGPU in one short sentence.', 32);

        const contextSweep = [];
        let oomCeiling = null;
        for (const ctxLen of [1024, 4096, 8192, 16384, 32768]) {
          let past_key_values = null;
          let logitsOut = null;
          try {
            console.log('[ORT-TRACE] contextSweep start ctxLen=' + ctxLen);
            const synthIds = new BigInt64Array(ctxLen);
            for (let i = 0; i < ctxLen; i++) synthIds[i] = BigInt(1000 + (i % 500));
            const CHUNK = 512;
            const tPref0 = performance.now();

            for (let off = 0; off < ctxLen; off += CHUNK) {
              const len = Math.min(CHUNK, ctxLen - off);
              const idsT = new Tensor('int64', synthIds.subarray(off, off + len), [1, len]);
              const maskT = new Tensor('int64', new BigInt64Array(off + len).fill(1n), [1, off + len]);
              const posT = new Tensor('int64', BigInt64Array.from({ length: len }, (_, k) => BigInt(off + k)), [1, len]);
              const res = await model.forward({
                input_ids: idsT,
                attention_mask: maskT,
                position_ids: posT,
                past_key_values,
                num_logits_to_keep: keepLastLogit,
              });
              if (off + len < ctxLen) {
                try { res.logits?.dispose?.(); } catch {}
              } else {
                logitsOut = res.logits;
              }
              past_key_values = updatePastKeyValues(res, past_key_values);
            }
            const prefillMs = Number((performance.now() - tPref0).toFixed(1));
            const prefillTokPerSec = Number(((ctxLen * 1000) / Math.max(1, prefillMs)).toFixed(2));
            let nextTok = extractLastStepLogits(logitsOut).bestIdx;
            try { logitsOut?.dispose?.(); } catch {}
            logitsOut = null;

            const decodeSteps = 32;
            const tDec0 = performance.now();
            for (let s = 0; s < decodeSteps; s++) {
              const totalLen = ctxLen + s + 1;
              const tokT = new Tensor('int64', BigInt64Array.from([BigInt(nextTok)]), [1, 1]);
              const maskT = new Tensor('int64', new BigInt64Array(totalLen).fill(1n), [1, totalLen]);
              const posT = new Tensor('int64', BigInt64Array.from([BigInt(ctxLen + s)]), [1, 1]);
              const res = await model.forward({
                input_ids: tokT,
                attention_mask: maskT,
                position_ids: posT,
                past_key_values,
                num_logits_to_keep: keepLastLogit,
              });
              nextTok = extractLastStepLogits(res.logits).bestIdx;
              try { res.logits?.dispose?.(); } catch {}
              past_key_values = updatePastKeyValues(res, past_key_values);
            }
            const decodeMs = Math.max(1, performance.now() - tDec0);
            const decodeTokPerSec = Number(((decodeSteps * 1000) / decodeMs).toFixed(2));
            disposePast(past_key_values);
            console.log('[ORT-TRACE] contextSweep finish ctxLen=' + ctxLen + ' prefill=' + prefillTokPerSec + ' decode=' + decodeTokPerSec);

            contextSweep.push({
              targetTokens: ctxLen,
              prefillMs,
              prefillTokPerSec,
              decodeTokPerSec,
              status: 'ok',
            });
          } catch (err) {
            try { logitsOut?.dispose?.(); } catch {}
            disposePast(past_key_values);
            const msg = err?.message || String(err);
            console.log('[ORT-TRACE] contextSweep FAILED ctxLen=' + ctxLen + ' err=' + msg);
            oomCeiling = { atContext: ctxLen, error: msg };
            contextSweep.push({ targetTokens: ctxLen, status: 'failed', error: msg });
            break;
          }
        }

        try { keepLastLogit.dispose?.(); } catch {}

        await model.dispose().catch(() => {});

        return {
          deviceConfigUsed,
          webgpuEmbedError,
          tokenizerMs,
          modelLoadMs,
          totalColdLoadMs,
          goldenA,
          goldenB,
          contextSweep,
          oomCeiling,
          logs,
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
  assert(
    fs.existsSync(path.join(HF_MODEL_DIR, 'onnx', 'decoder_model_merged_q2f16.onnx_data')),
    'Gemma 4 QAT files missing. Run npm run bundle:gemma4 first.',
  );
  cleanProfileDir();

  console.log('Bundling standalone ORT WebGPU harness for Gemma 4 E4B...');
  const harnessJs = await buildBrowserHarnessJs();

  const html = `<!doctype html><html><head><meta charset="utf-8" /><title>Gemma 4 E4B Track G1 ORT Baseline</title></head>
  <body><div id="status">Loading Gemma 4 E4B ORT WebGPU...</div><script type="module" src="/harness.js"></script></body></html>`;

  const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
    if (pathname === '/' || pathname === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
      return;
    }
    if (pathname === '/harness.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8' });
      res.end(harnessJs);
      return;
    }
    if (pathname.startsWith('/models/gemma4/')) {
      const rel = pathname.slice('/models/gemma4/'.length);
      serveStaticFile(req, res, path.join(HF_MODEL_DIR, rel));
      return;
    }
    if (pathname.startsWith('/ort/')) {
      const rel = pathname.slice('/ort/'.length);
      serveStaticFile(req, res, path.join(ORT_DIST_DIR, rel));
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
      if (m.text().includes('[ORT-TRACE]')) console.log(m.text());
    });
    page.on('pageerror', (e) => consoleErrors.push(e.message));

    console.log('\n=== [g01-gemma4-ort-baseline] Running Gemma 4 E4B ORT WebGPU Baseline ===');
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => Boolean(window.__GEMMA4_ORT__), { timeout: 20000 });

    const suite = await page.evaluate(() => window.__GEMMA4_ORT__.runSuite());
    console.log(
      `  Loaded in ${suite.totalColdLoadMs}ms (tok=${suite.tokenizerMs}ms, ort=${suite.modelLoadMs}ms) | device=${JSON.stringify(suite.deviceConfigUsed)}`,
    );
    if (suite.webgpuEmbedError) {
      console.log(`  Note: WebGPU embed_tokens fallback reason: ${suite.webgpuEmbedError}`);
    }
    console.log(
      `  Golden A ("2+2"): ttft=${suite.goldenA.ttftMs}ms, dec=${suite.goldenA.decodeTokPerSec} tok/s, ans=${JSON.stringify(suite.goldenA.decodedText)}, ids=[${suite.goldenA.generatedIds.join(',')}]`,
    );
    console.log(
      `  Golden B ("WebGPU"): ttft=${suite.goldenB.ttftMs}ms, dec=${suite.goldenB.decodeTokPerSec} tok/s, ans=${JSON.stringify(suite.goldenB.decodedText.slice(0, 120))}`,
    );
    for (const pt of suite.contextSweep) {
      console.log(
        pt.status === 'ok'
          ? `  Context ${pt.targetTokens}: prefill=${pt.prefillTokPerSec} tok/s (${pt.prefillMs}ms), decode=${pt.decodeTokPerSec} tok/s`
          : `  Context ${pt.targetTokens}: FAILED (${pt.error})`,
      );
    }

    const payload = {
      runId: 'g01-gemma4-ort-baseline',
      track: 'gemma4-track1-ort',
      modelId: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
      recordedAt: new Date().toISOString(),
      ...suite,
      consoleErrorCount: consoleErrors.length,
      consoleErrors,
      status: suite.contextSweep.some((c) => c.status === 'ok') ? 'ok' : 'partial',
    };
    fs.mkdirSync(RUNS_DIR, { recursive: true });
    const outPath = path.join(RUNS_DIR, 'g01-gemma4-ort-baseline.json');
    fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
    console.log(`Saved ${outPath}`);
  } finally {
    await context.close().catch(() => {});
    server.close();
    fs.rmSync(PROFILE_DIR, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('Gemma 4 Track G1 ORT failed:', err);
  process.exit(1);
});
