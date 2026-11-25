import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

async function launchExtensionPage() {
  if (!fs.existsSync(path.join(DIST_DIR, 'manifest.json'))) {
    throw new Error('dist/manifest.json missing. Run npm run build first.');
  }
  fs.mkdirSync(PROFILE_DIR, { recursive: true });

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

  let sw = context.serviceWorkers()[0];
  if (!sw) {
    sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
  }
  const extensionId = new URL(sw.url()).hostname;

  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('pageerror', (err) => {
    consoleErrors.push(err.message);
  });

  await page.goto(`chrome-extension://${extensionId}/sidepanel.html`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => Boolean(window.__TINYGPT_CLIENT__ && window.__TINYGPT_CLIENT__.getSnapshot().connected),
    { timeout: 25000 },
  );

  return { context, page, consoleErrors };
}

async function getSnapshot(page) {
  return page.evaluate(() => window.__TINYGPT_CLIENT__.getSnapshot());
}

async function waitForSnapshot(page, predicate, { timeoutMs = 600000, pollMs = 300, onTick } = {}) {
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
  throw new Error(`Timed out after ${timeoutMs}ms. Last snapshot: ${JSON.stringify(finalSnap.telemetry)}`);
}

function saveRun(runId, payload) {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  const filePath = path.join(RUNS_DIR, `${runId}.json`);
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  console.log(`Saved ${filePath}`);
}

async function runOpfsLoadTest() {
  console.log('\n=== [10-wgsl-correctness-and-opfs-load] Prepare 2.50 GiB OPFS Slabs & Cold Streaming Load ===');
  let session = await launchExtensionPage();
  let prepareResult;
  try {
    const prevPrepare = JSON.stringify((await getSnapshot(session.page)).lastWgslOpfs ?? null);
    await session.page.evaluate(() =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_WGSL_OPFS_BENCH',
        action: 'prepare',
        totalMb: 2560,
      }),
    );
    const snap = await waitForSnapshot(
      session.page,
      (s) => s.lastWgslOpfs && s.lastWgslOpfs.action === 'prepare' && JSON.stringify(s.lastWgslOpfs) !== prevPrepare,
      { timeoutMs: 10 * 60 * 1000 },
    );
    prepareResult = snap.lastWgslOpfs;
    console.log('  Prepared slabs:', prepareResult.details);
    if (!prepareResult.ok) throw new Error(prepareResult.details);
  } finally {
    await session.context.close();
  }

  console.log('  Closed Chromium completely. Relaunching to measure cold OPFS -> GPUQueue.writeBuffer...');
  session = await launchExtensionPage();
  let loadResult;
  try {
    const prevLoad = JSON.stringify((await getSnapshot(session.page)).lastWgslOpfs ?? null);
    await session.page.evaluate(() =>
      window.__TINYGPT_CLIENT__.send({
        type: 'RUN_WGSL_OPFS_BENCH',
        action: 'load',
        totalMb: 2560,
        chunkMibList: [16, 32, 64],
      }),
    );
    const snap = await waitForSnapshot(
      session.page,
      (s) => s.lastWgslOpfs && s.lastWgslOpfs.action === 'load' && JSON.stringify(s.lastWgslOpfs) !== prevLoad,
      { timeoutMs: 10 * 60 * 1000 },
    );
    loadResult = snap.lastWgslOpfs;
    console.log('  OPFS -> WebGPU Load Timings:', JSON.stringify(loadResult.timings, null, 2));

    // Clean up the temporary 2.50 GiB slab test files
    await session.page.evaluate(() =>
      window.__TINYGPT_CLIENT__.send({ type: 'RUN_WGSL_OPFS_BENCH', action: 'clear' }),
    );
    await waitForSnapshot(
      session.page,
      (s) => s.lastWgslOpfs && s.lastWgslOpfs.action === 'clear',
      { timeoutMs: 30000 },
    );
  } finally {
    await session.context.close();
  }

  return { prepareResult, loadResult };
}

async function runWgslConfig({ runId, label, config, extraOpfsResult = null }) {
  console.log(`\n=== [${runId}] ${label} ===`);
  const session = await launchExtensionPage();
  try {
    const before = await getSnapshot(session.page);
    const prevAt = before.lastWgslBench?.finishedAt ?? null;
    await session.page.evaluate(
      (cfg) => window.__TINYGPT_CLIENT__.send({ type: 'RUN_WGSL_BENCH', config: cfg }),
      config,
    );
    let lastMsg = '';
    const snap = await waitForSnapshot(
      session.page,
      (s) => s.lastWgslBench && s.lastWgslBench.finishedAt !== prevAt,
      {
        timeoutMs: 25 * 60 * 1000,
        pollMs: 400,
        onTick: (s) => {
          if (s.wgslProgress && s.wgslProgress !== lastMsg) {
            console.log(`  [WGSL] ${s.wgslProgress}`);
            lastMsg = s.wgslProgress;
          }
        },
      },
    );
    const res = snap.lastWgslBench;
    console.log(
      `  Compile: ${res.pipelineCompileMs.toFixed(1)}ms | Correctness: ${res.correctness.map((c) => `${c.kernel}:${c.passed ? 'PASS' : 'FAIL'}`).join(', ')}`,
    );
    for (const pt of res.points) {
      console.log(
        `  ctx=${pt.contextLength}: ${pt.decodeTokPerSec.toFixed(2)} tok/s (${pt.bandwidthUtilizationPct.toFixed(1)}% BW, VRAM=${pt.estVramMb.toFixed(0)} MiB) [${pt.status}]`,
      );
    }
    if (res.error) {
      console.error(`  Error: ${res.error}`);
    }

    const payload = {
      runId,
      track: 'track2-wgsl',
      recordedAt: new Date().toISOString(),
      label,
      config,
      wgslResult: res,
      ...(extraOpfsResult ? { opfsLoadBench: extraOpfsResult } : {}),
      consoleErrorCount: session.consoleErrors.length,
      offscreenErrors: snap.logs.filter((l) => l.level === 'error'),
      status: res.error ? 'error' : 'ok',
    };
    saveRun(runId, payload);
    return payload;
  } finally {
    await session.context.close();
  }
}

async function main() {
  const opfsBench = await runOpfsLoadTest();

  // 10/11: q4-scalar + FP16 KV (includes correctness + OPFS cold load timings)
  const r11 = await runWgslConfig({
    runId: '10-wgsl-q4-scalar-fp16kv',
    label: 'Rust/WASM + Custom WGSL Qwen3.5-4B (q4-scalar, static FP16 KV, single-encoder)',
    config: {
      gemvVariant: 'q4-scalar',
      kvMode: 'fp16',
      contextLengths: [1024, 8192, 32768, 65536],
      decodeTokens: 32,
      warmupTokens: 2,
      gpuBandwidthGBs: 936,
    },
    extraOpfsResult: opfsBench,
  });

  // 12: q4-vec4-f16 + FP16 KV
  const r12 = await runWgslConfig({
    runId: '11-wgsl-q4-vec4-f16-fp16kv',
    label: 'Rust/WASM + Custom WGSL Qwen3.5-4B (q4-vec4-f16, static FP16 KV, single-encoder)',
    config: {
      gemvVariant: 'q4-vec4-f16',
      kvMode: 'fp16',
      contextLengths: [1024, 8192, 32768, 65536],
      decodeTokens: 32,
      warmupTokens: 2,
      gpuBandwidthGBs: 936,
    },
  });

  // 13: q4-subgroups-f16 + FP16 KV
  const r13 = await runWgslConfig({
    runId: '12-wgsl-q4-subgroups-f16-fp16kv',
    label: 'Rust/WASM + Custom WGSL Qwen3.5-4B (q4-subgroups-f16 + subgroupAdd, static FP16 KV)',
    config: {
      gemvVariant: 'q4-subgroups-f16',
      kvMode: 'fp16',
      contextLengths: [1024, 8192, 32768, 65536],
      decodeTokens: 32,
      warmupTokens: 2,
      gpuBandwidthGBs: 936,
    },
  });

  // 14: q4-subgroups-f16 + INT8 KV up to 128K
  const r14 = await runWgslConfig({
    runId: '13-wgsl-q4-subgroups-f16-int8kv',
    label: 'Rust/WASM + Custom WGSL Qwen3.5-4B (q4-subgroups-f16 + INT8 Quantized KV, up to 128K)',
    config: {
      gemvVariant: 'q4-subgroups-f16',
      kvMode: 'int8',
      contextLengths: [1024, 8192, 32768, 65536, 131072],
      decodeTokens: 32,
      warmupTokens: 2,
      gpuBandwidthGBs: 936,
    },
  });

  console.log(
    '\n[Track 2 Complete] Statuses:',
    [r11, r12, r13, r14].map((r) => `${r.runId}:${r.status}`).join(', '),
  );
}

main().catch((err) => {
  console.error('[Track 2 Fatal]', err);
  process.exit(1);
});
