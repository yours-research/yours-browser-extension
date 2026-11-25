import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http2 from 'node:http2';
import https from 'node:https';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const TMP_DIR = path.join(ROOT_DIR, '.tmp-dl-bench');
const OUT_JSON_PATH = path.join(ROOT_DIR, 'benchmarks', 'download-bench.json');
const REPACKED_MANIFEST_PATH = path.join(ROOT_DIR, 'model-bundle', 'qwen35-4b', 'manifest.json');
const REPACKED_SLAB0_PATH = path.join(ROOT_DIR, 'model-bundle', 'qwen35-4b', 'slab0.part00.bin');

const MODEL_ID = 'onnx-community/Qwen3.5-4B-ONNX-OPT';
const HF_BASE_URL = `https://huggingface.co/${MODEL_ID}/resolve/main`;
const HF_API_BASE = `https://huggingface.co/api/models/${MODEL_ID}/tree/main`;

const PRE_REPACKED_TOTAL_BYTES = 2_651_081_621; // 2.469 GiB from model-bundle/qwen35-4b
const RAW_ONNX_270_GIB_BYTES = Math.round(2.7 * 1024 * 1024 * 1024); // 2,899,102,925 B
const MIB = 1024 * 1024;
const EXTENSION_TEST_ORIGIN = 'chrome-extension://tinygptbenchoriginplaceholder000000';

const USER_AGENT = 'tinygpt-download-bench/1.0 (Node.js ESM)';

function round(val, digits = 3) {
  const factor = 10 ** digits;
  return Math.round(val * factor) / factor;
}

function calcMetrics(bytes, elapsedMs) {
  const safeMs = Math.max(0.001, elapsedMs);
  const sec = safeMs / 1000;
  const mib = bytes / MIB;
  const mibPerSec = mib / sec;
  const decimalMBPerSec = bytes / 1_000_000 / sec;
  const mbps = (bytes * 8) / 1_000_000 / sec;
  const etaSec_2_47GiB = (PRE_REPACKED_TOTAL_BYTES / bytes) * sec;
  const etaSec_2_70GiB = (RAW_ONNX_270_GIB_BYTES / bytes) * sec;
  return {
    bytes,
    mib: round(mib, 4),
    elapsedMs: round(safeMs, 2),
    elapsedSec: round(sec, 3),
    mibPerSec: round(mibPerSec, 3),
    decimalMBPerSec: round(decimalMBPerSec, 3),
    mbps: round(mbps, 3),
    etaSec_2_47GiB: round(etaSec_2_47GiB, 2),
    etaSec_2_70GiB: round(etaSec_2_70GiB, 2),
  };
}

function percentile(sortedNums, p) {
  if (sortedNums.length === 0) return 0;
  const idx = Math.min(sortedNums.length - 1, Math.max(0, Math.floor((sortedNums.length - 1) * p)));
  return sortedNums[idx];
}

function summarizeDurations(durationsMs) {
  const sorted = [...durationsMs].sort((a, b) => a - b);
  const minMs = sorted[0] ?? 0;
  const medianMs = percentile(sorted, 0.5);
  const p95Ms = percentile(sorted, 0.95);
  const maxMs = sorted[sorted.length - 1] ?? 0;
  const meanMs = durationsMs.reduce((a, b) => a + b, 0) / Math.max(1, durationsMs.length);
  return {
    count: durationsMs.length,
    minMs: round(minMs, 2),
    medianMs: round(medianMs, 2),
    meanMs: round(meanMs, 2),
    p95Ms: round(p95Ms, 2),
    maxMs: round(maxMs, 2),
    stragglerRatioMaxOverMedian: round(medianMs > 0 ? maxMs / medianMs : 1, 3),
  };
}

async function hashFilePrefixSha256(filePath, byteLimit) {
  const hash = crypto.createHash('sha256');
  const fh = await fsp.open(filePath, 'r');
  const buf = Buffer.allocUnsafe(4 * MIB);
  let remaining = byteLimit;
  let offset = 0;
  try {
    while (remaining > 0) {
      const toRead = Math.min(buf.byteLength, remaining);
      const { bytesRead } = await fh.read(buf, 0, toRead, offset);
      if (bytesRead <= 0) {
        throw new Error(`Unexpected EOF hashing ${filePath} at offset ${offset} (remaining ${remaining})`);
      }
      hash.update(buf.subarray(0, bytesRead));
      offset += bytesRead;
      remaining -= bytesRead;
    }
  } finally {
    await fh.close();
  }
  return hash.digest('hex');
}

function headersToPlainObject(headers) {
  const out = {};
  for (const [k, v] of headers.entries()) {
    out[k.toLowerCase()] = v;
  }
  return out;
}

async function resolveCdnUrl(relPath) {
  const originalUrl = `${HF_BASE_URL}/${relPath}`;
  const t0 = performance.now();
  const res = await fetch(originalUrl, {
    method: 'HEAD',
    redirect: 'manual',
    headers: {
      'user-agent': USER_AGENT,
      origin: EXTENSION_TEST_ORIGIN,
    },
  });
  const resolveMs = performance.now() - t0;
  const loc = res.headers.get('location');
  const finalUrl = loc ? new URL(loc, originalUrl).href : originalUrl;
  return {
    originalUrl,
    finalUrl,
    status: res.status,
    resolveMs: round(resolveMs, 2),
    hubHeaders: headersToPlainObject(res.headers),
  };
}

async function probeEndpoints() {
  console.log('\n=== Probing Hugging Face Hub & Xet CAS CDN Headers ===');
  const targetFiles = [
    'config.json',
    'tokenizer.json',
    'onnx/embed_tokens_q4f16.onnx_data',
    'onnx/decoder_model_merged_q4f16.onnx',
    'onnx/decoder_model_merged_q4f16.onnx_data',
    'onnx/decoder_model_merged_q4f16.onnx_data_1',
  ];

  const [mainTreeRes, onnxTreeRes] = await Promise.all([
    fetch(HF_API_BASE),
    fetch(`${HF_API_BASE}/onnx`),
  ]);
  const mainTree = await mainTreeRes.json();
  const onnxTree = await onnxTreeRes.json();

  const fileSizeMap = new Map();
  for (const item of [...mainTree, ...onnxTree]) {
    if (item && typeof item.path === 'string' && typeof item.size === 'number') {
      fileSizeMap.set(item.path, item.size);
    }
  }

  const probes = [];
  const resolvedUrls = new Map();

  for (const relPath of targetFiles) {
    const hop1 = await resolveCdnUrl(relPath);
    const cdnRes = await fetch(hop1.finalUrl, {
      method: 'HEAD',
      redirect: 'manual',
      headers: {
        'user-agent': USER_AGENT,
        origin: EXTENSION_TEST_ORIGIN,
      },
    });

    const rangeRes = await fetch(hop1.finalUrl, {
      method: 'GET',
      headers: {
        'user-agent': USER_AGENT,
        origin: EXTENSION_TEST_ORIGIN,
        range: 'bytes=0-1023',
      },
    });
    const rangeSampleBytes = (await rangeRes.arrayBuffer()).byteLength;

    const cdnHeaders = headersToPlainObject(cdnRes.headers);
    const rangeHeaders = headersToPlainObject(rangeRes.headers);

    resolvedUrls.set(relPath, hop1.finalUrl);
    probes.push({
      relPath,
      apiSizeBytes: fileSizeMap.get(relPath) ?? Number(cdnHeaders['content-length'] ?? 0),
      hubStatus: hop1.status,
      resolveMs: hop1.resolveMs,
      cdnHost: new URL(hop1.finalUrl).host,
      cdnStatus: cdnRes.status,
      rangeStatus: rangeRes.status,
      rangeSampleBytes,
      acceptRanges: cdnHeaders['accept-ranges'] ?? rangeHeaders['accept-ranges'] ?? 'none',
      contentLength: Number(cdnHeaders['content-length'] ?? 0),
      contentEncoding: cdnHeaders['content-encoding'] ?? 'identity',
      etag: cdnHeaders['etag'] ?? '',
      xLinkedSize: Number(hop1.hubHeaders['x-linked-size'] ?? cdnHeaders['content-length'] ?? 0),
      xLinkedEtag: hop1.hubHeaders['x-linked-etag'] ?? '',
      xXetHash: hop1.hubHeaders['x-xet-hash'] ?? '',
      xHfCdnPop: cdnHeaders['x-hf-cdn-pop'] ?? '',
      via: hop1.hubHeaders['via'] ?? '',
      cacheControl: cdnHeaders['cache-control'] ?? hop1.hubHeaders['cache-control'] ?? '',
      rateLimit: hop1.hubHeaders['ratelimit'] ?? '',
      rateLimitPolicy: hop1.hubHeaders['ratelimit-policy'] ?? '',
      corsHubAllowOrigin: hop1.hubHeaders['access-control-allow-origin'] ?? 'none',
      corsHubExposeHeaders: hop1.hubHeaders['access-control-expose-headers'] ?? 'none',
      corsCdnAllowOrigin: rangeHeaders['access-control-allow-origin'] ?? cdnHeaders['access-control-allow-origin'] ?? 'none',
      corsCdnAllowHeaders: rangeHeaders['access-control-allow-headers'] ?? 'none',
      corsCdnExposeHeaders: rangeHeaders['access-control-expose-headers'] ?? 'none',
    });

    console.log(
      `  ${relPath}: hub=${hop1.status} (${hop1.resolveMs}ms) -> ${new URL(hop1.finalUrl).host} ` +
        `len=${cdnHeaders['content-length'] ?? 'chunked'} range=${rangeRes.status} acao=${rangeHeaders['access-control-allow-origin'] ?? 'none'} ` +
        `xet=${(hop1.hubHeaders['x-xet-hash'] ?? 'none').slice(0, 12)}`,
    );
  }

  const hfJsonBytes =
    (fileSizeMap.get('config.json') ?? 0) +
    (fileSizeMap.get('generation_config.json') ?? 0) +
    (fileSizeMap.get('tokenizer_config.json') ?? 0) +
    (fileSizeMap.get('tokenizer.json') ?? 0);
  const hfDecoderQ4f16Bytes =
    (fileSizeMap.get('onnx/decoder_model_merged_q4f16.onnx') ?? 0) +
    (fileSizeMap.get('onnx/decoder_model_merged_q4f16.onnx_data') ?? 0) +
    (fileSizeMap.get('onnx/decoder_model_merged_q4f16.onnx_data_1') ?? 0);
  const hfEmbedQ4f16Bytes =
    (fileSizeMap.get('onnx/embed_tokens_q4f16.onnx') ?? 0) +
    (fileSizeMap.get('onnx/embed_tokens_q4f16.onnx_data') ?? 0);
  const hfVisionQ4f16Bytes =
    (fileSizeMap.get('onnx/vision_encoder_q4f16.onnx') ?? 0) +
    (fileSizeMap.get('onnx/vision_encoder_q4f16.onnx_data') ?? 0);

  return {
    probes,
    resolvedUrls,
    repoByteTotals: {
      jsonAndTokenizerBytes: hfJsonBytes,
      decoderQ4f16Bytes: hfDecoderQ4f16Bytes,
      embedQ4f16Bytes: hfEmbedQ4f16Bytes,
      visionQ4f16Bytes: hfVisionQ4f16Bytes,
      decoderPlusJsonBytes: hfDecoderQ4f16Bytes + hfJsonBytes,
      decoderPlusEmbedPlusJsonBytes: hfDecoderQ4f16Bytes + hfEmbedQ4f16Bytes + hfJsonBytes,
      fullMultimodalQ4f16Bytes: hfDecoderQ4f16Bytes + hfEmbedQ4f16Bytes + hfVisionQ4f16Bytes + hfJsonBytes,
      preRepackedWgslBundleBytes: PRE_REPACKED_TOTAL_BYTES,
    },
  };
}

async function runTrial0Baseline(resolvedUrls) {
  console.log('\n=== Trial 0: Baseline Single-Stream Sequential Fetch ===');

  // 0A: Full onnx/embed_tokens_q4f16.onnx_data (~350.5 MiB) via standard fetch + redirect: follow
  const embedUrl = `${HF_BASE_URL}/onnx/embed_tokens_q4f16.onnx_data`;
  const embedPath = path.join(TMP_DIR, 'trial0-embed.bin');
  const embedHash = crypto.createHash('sha256');
  const t0Embed = performance.now();
  const embedRes = await fetch(embedUrl, { redirect: 'follow', headers: { 'user-agent': USER_AGENT } });
  if (!embedRes.ok || !embedRes.body) {
    throw new Error(`Trial 0 embed fetch failed: HTTP ${embedRes.status}`);
  }
  const embedFh = await fsp.open(embedPath, 'w');
  let embedBytes = 0;
  try {
    const reader = embedRes.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value && value.byteLength > 0) {
        embedHash.update(value);
        await embedFh.write(value, 0, value.byteLength, embedBytes);
        embedBytes += value.byteLength;
      }
    }
  } finally {
    await embedFh.close();
  }
  const embedElapsedMs = performance.now() - t0Embed;
  const embedSha256 = embedHash.digest('hex');
  const embedMetrics = {
    ...calcMetrics(embedBytes, embedElapsedMs),
    relPath: 'onnx/embed_tokens_q4f16.onnx_data',
    sha256: embedSha256,
  };
  console.log(
    `  0A embed_tokens full: ${embedMetrics.mib} MiB in ${embedMetrics.elapsedSec}s = ${embedMetrics.mibPerSec} MiB/s (${embedMetrics.mbps} Mbps), ETA 2.47GiB=${embedMetrics.etaSec_2_47GiB}s`,
  );

  // 0B: 256 MiB sequential prefix of onnx/decoder_model_merged_q4f16.onnx_data
  const targetPrefixBytes = 256 * MIB;
  const decoderUrl = `${HF_BASE_URL}/onnx/decoder_model_merged_q4f16.onnx_data`;
  const decoderPrefixPath = path.join(TMP_DIR, 'trial0-decoder-256m.bin');
  const decoderHash = crypto.createHash('sha256');
  const ac = new AbortController();
  const t0Dec = performance.now();
  const decRes = await fetch(decoderUrl, {
    redirect: 'follow',
    signal: ac.signal,
    headers: { 'user-agent': USER_AGENT },
  });
  if (!decRes.ok || !decRes.body) {
    throw new Error(`Trial 0 decoder fetch failed: HTTP ${decRes.status}`);
  }
  const decFh = await fsp.open(decoderPrefixPath, 'w');
  let decBytes = 0;
  try {
    const reader = decRes.body.getReader();
    while (decBytes < targetPrefixBytes) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value && value.byteLength > 0) {
        const take = Math.min(value.byteLength, targetPrefixBytes - decBytes);
        const slice = take === value.byteLength ? value : value.subarray(0, take);
        decoderHash.update(slice);
        await decFh.write(slice, 0, slice.byteLength, decBytes);
        decBytes += slice.byteLength;
        if (decBytes >= targetPrefixBytes) {
          ac.abort();
          try {
            await reader.cancel();
          } catch {
            // Expected abort on clean prefix termination.
          }
          break;
        }
      }
    }
  } finally {
    await decFh.close();
  }
  const decElapsedMs = performance.now() - t0Dec;
  if (decBytes !== targetPrefixBytes) {
    throw new Error(`Trial 0 decoder prefix truncated: ${decBytes} != ${targetPrefixBytes}`);
  }
  const decSha256 = decoderHash.digest('hex');
  const decoderMetrics = {
    ...calcMetrics(decBytes, decElapsedMs),
    relPath: 'onnx/decoder_model_merged_q4f16.onnx_data',
    windowBytes: targetPrefixBytes,
    sha256: decSha256,
  };
  console.log(
    `  0B decoder 256MiB prefix: ${decoderMetrics.mib} MiB in ${decoderMetrics.elapsedSec}s = ${decoderMetrics.mibPerSec} MiB/s (${decoderMetrics.mbps} Mbps), ETA 2.47GiB=${decoderMetrics.etaSec_2_47GiB}s`,
  );

  // Precompute ground-truth 16 MiB shard hashes from both verified Trial 0 files for Trial 2 & Trial 4.
  const shardHashes = {
    decoder16MiB: [],
    embed16MiB: [],
    decoder128MiBSha256: await hashFilePrefixSha256(decoderPrefixPath, 128 * MIB),
  };
  for (let i = 0; i < 16; i++) {
    const start = i * 16 * MIB;
    const h = crypto.createHash('sha256');
    const fh = await fsp.open(decoderPrefixPath, 'r');
    const buf = Buffer.allocUnsafe(16 * MIB);
    try {
      let off = 0;
      while (off < buf.byteLength) {
        const { bytesRead } = await fh.read(buf, off, buf.byteLength - off, start + off);
        if (bytesRead <= 0) break;
        off += bytesRead;
      }
      h.update(buf.subarray(0, off));
    } finally {
      await fh.close();
    }
    shardHashes.decoder16MiB.push(h.digest('hex'));
  }
  for (let i = 0; i < 4; i++) {
    const start = i * 16 * MIB;
    const h = crypto.createHash('sha256');
    const fh = await fsp.open(embedPath, 'r');
    const buf = Buffer.allocUnsafe(16 * MIB);
    try {
      let off = 0;
      while (off < buf.byteLength) {
        const { bytesRead } = await fh.read(buf, off, buf.byteLength - off, start + off);
        if (bytesRead <= 0) break;
        off += bytesRead;
      }
      h.update(buf.subarray(0, off));
    } finally {
      await fh.close();
    }
    shardHashes.embed16MiB.push(h.digest('hex'));
  }

  // Delete embed temp file now that its hashes are extracted; keep decoder 256m until Trial 1/4 finish if needed.
  await fsp.rm(embedPath, { force: true });
  await fsp.rm(decoderPrefixPath, { force: true });

  const totalBytes = embedBytes + decBytes;
  const totalElapsedMs = embedElapsedMs + decElapsedMs;
  const combined = {
    ...calcMetrics(totalBytes, totalElapsedMs),
    description: 'Combined single-stream sequential baseline (embed_tokens full + decoder 256 MiB prefix)',
  };

  return {
    embedTokensFull: embedMetrics,
    decoderPrefix256MiB: decoderMetrics,
    combined,
    referenceHashes: shardHashes,
    baselineMibPerSec: combined.mibPerSec,
  };
}

async function fetchRangeToFileHandle({
  url,
  rangeStart,
  rangeEnd,
  fh,
  writeOffset,
  expectedSha256 = '',
  computeSha256 = false,
  maxRetries = 3,
}) {
  let attempt = 0;
  const expectedLen = rangeEnd - rangeStart + 1;
  while (true) {
    attempt += 1;
    try {
      const res = await fetch(url, {
        headers: {
          'user-agent': USER_AGENT,
          range: `bytes=${rangeStart}-${rangeEnd}`,
        },
      });
      if (res.status !== 206 && res.status !== 200) {
        throw new Error(`HTTP ${res.status} for range ${rangeStart}-${rangeEnd}`);
      }
      if (!res.body) {
        throw new Error(`Empty response body for range ${rangeStart}-${rangeEnd}`);
      }
      const hash = computeSha256 || expectedSha256 ? crypto.createHash('sha256') : null;
      const reader = res.body.getReader();
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          const take = Math.min(value.byteLength, expectedLen - received);
          if (take <= 0) break;
          const chunk = take === value.byteLength ? value : value.subarray(0, take);
          if (hash) hash.update(chunk);
          await fh.write(chunk, 0, chunk.byteLength, writeOffset + received);
          received += chunk.byteLength;
        }
      }
      if (received !== expectedLen) {
        throw new Error(`Truncated range ${rangeStart}-${rangeEnd}: got ${received}/${expectedLen}`);
      }
      const actualSha256 = hash ? hash.digest('hex') : '';
      if (expectedSha256 && actualSha256 !== expectedSha256) {
        throw new Error(`SHA-256 mismatch on range ${rangeStart}-${rangeEnd}: ${actualSha256} != ${expectedSha256}`);
      }
      return { bytes: received, sha256: actualSha256, attempts: attempt };
    } catch (err) {
      if (attempt >= maxRetries) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 150 * attempt));
    }
  }
}

async function runWorkStealingDownload({
  tasks,
  concurrency,
  outFilePath,
  totalFileBytes,
  computeSha256 = false,
  verifyExpectedSha256 = false,
}) {
  await fsp.rm(outFilePath, { force: true });
  const fh = await fsp.open(outFilePath, 'w+');
  await fh.truncate(totalFileBytes);

  let nextIdx = 0;
  const chunkDurationsMs = new Array(tasks.length);
  const workerChunkCounts = new Array(Math.min(concurrency, tasks.length)).fill(0);
  let totalRetries = 0;
  const chunkResults = new Array(tasks.length);

  const tStart = performance.now();
  try {
    const workerCount = Math.min(concurrency, tasks.length);
    const workers = Array.from({ length: workerCount }, async (_, workerId) => {
      while (true) {
        const idx = nextIdx;
        nextIdx += 1;
        if (idx >= tasks.length) break;
        const task = tasks[idx];
        const c0 = performance.now();
        const res = await fetchRangeToFileHandle({
          url: task.url,
          rangeStart: task.rangeStart,
          rangeEnd: task.rangeEnd,
          fh,
          writeOffset: task.writeOffset,
          expectedSha256: verifyExpectedSha256 ? task.expectedSha256 : '',
          computeSha256,
        });
        chunkDurationsMs[idx] = performance.now() - c0;
        workerChunkCounts[workerId] += 1;
        totalRetries += res.attempts - 1;
        chunkResults[idx] = res;
      }
    });
    await Promise.all(workers);
    await fh.sync();
  } finally {
    await fh.close();
  }
  const elapsedMs = performance.now() - tStart;
  return {
    elapsedMs,
    chunkDurationsMs,
    workerChunkCounts,
    totalRetries,
    chunkResults,
  };
}

async function runTrial1ParallelRange(resolvedUrls, baselineMibPerSec, expectedPrefix256Sha256) {
  console.log('\n=== Trial 1: Parallel HTTP Range Chunked Downloader (Work-Stealing Queue) ===');
  const decoderCdnUrl = resolvedUrls.get('onnx/decoder_model_merged_q4f16.onnx_data');
  const chunkSizes = [8 * MIB, 16 * MIB];
  const concurrencies = [4, 8, 16, 32];
  const runs = [];

  for (const chunkSizeBytes of chunkSizes) {
    for (const concurrency of concurrencies) {
      // Ensure c=32 with 16 MiB chunks actually exercises 32 concurrent slots via 512 MiB window,
      // while keeping 256 MiB prefix SHA-256 verification against Trial 0!
      const windowBytes = chunkSizeBytes === 16 * MIB && concurrency === 32 ? 512 * MIB : 256 * MIB;
      const numChunks = windowBytes / chunkSizeBytes;
      const tasks = Array.from({ length: numChunks }, (_, i) => ({
        url: decoderCdnUrl,
        rangeStart: i * chunkSizeBytes,
        rangeEnd: (i + 1) * chunkSizeBytes - 1,
        writeOffset: i * chunkSizeBytes,
        expectedSha256: '',
      }));

      const outPath = path.join(TMP_DIR, `trial1-c${concurrency}-cs${chunkSizeBytes / MIB}m.bin`);
      const res = await runWorkStealingDownload({
        tasks,
        concurrency,
        outFilePath: outPath,
        totalFileBytes: windowBytes,
        computeSha256: false,
        verifyExpectedSha256: false,
      });

      const actualPrefix256Sha256 = await hashFilePrefixSha256(outPath, 256 * MIB);
      const sha256Match = actualPrefix256Sha256 === expectedPrefix256Sha256;
      if (!sha256Match) {
        throw new Error(
          `Trial 1 SHA-256 mismatch at c=${concurrency}, chunk=${chunkSizeBytes / MIB}MiB: ${actualPrefix256Sha256} != ${expectedPrefix256Sha256}`,
        );
      }
      await fsp.rm(outPath, { force: true });

      const m = calcMetrics(windowBytes, res.elapsedMs);
      const entry = {
        concurrency,
        effectiveWorkers: Math.min(concurrency, numChunks),
        chunkSizeMiB: chunkSizeBytes / MIB,
        chunkSizeBytes,
        chunkCount: numChunks,
        windowMiB: windowBytes / MIB,
        writeMode: 'positional-filehandle-write',
        queueMode: 'dynamic-work-stealing',
        ...m,
        speedupVsBaseline: round(m.mibPerSec / baselineMibPerSec, 3),
        prefix256Sha256Verified: sha256Match,
        prefix256Sha256: actualPrefix256Sha256,
        retries: res.totalRetries,
        chunkLatency: summarizeDurations(res.chunkDurationsMs),
        workDistributionMinMax: {
          minChunksPerWorker: Math.min(...res.workerChunkCounts),
          maxChunksPerWorker: Math.max(...res.workerChunkCounts),
        },
      };
      runs.push(entry);
      console.log(
        `  c=${String(concurrency).padStart(2)} chunk=${String(entry.chunkSizeMiB).padStart(2)}MiB win=${entry.windowMiB}MiB -> ` +
          `${entry.mibPerSec} MiB/s (${entry.mbps} Mbps, ${entry.speedupVsBaseline}x), ` +
          `ETA 2.47GiB=${entry.etaSec_2_47GiB}s, sha256OK=${sha256Match}`,
      );
    }
  }

  const best = [...runs].sort((a, b) => b.mibPerSec - a.mibPerSec)[0];
  return {
    targetFile: 'onnx/decoder_model_merged_q4f16.onnx_data',
    runs,
    bestConfig: {
      concurrency: best.concurrency,
      chunkSizeMiB: best.chunkSizeMiB,
      mibPerSec: best.mibPerSec,
      mbps: best.mbps,
      speedupVsBaseline: best.speedupVsBaseline,
      etaSec_2_47GiB: best.etaSec_2_47GiB,
    },
  };
}

async function runTrial2ShardedSecurityAndResume(resolvedUrls, baselineMibPerSec, refHashes) {
  console.log('\n=== Trial 2: Pre-Sharded Multi-File + Incremental SHA-256 + Corrupted Retry + Resume ===');
  const decUrl = resolvedUrls.get('onnx/decoder_model_merged_q4f16.onnx_data');
  const embUrl = resolvedUrls.get('onnx/embed_tokens_q4f16.onnx_data');

  // 16 x 16 MiB = 256 MiB across two real HF Xet files (12 decoder shards + 4 embed shards)
  const shardManifest = [];
  for (let i = 0; i < 12; i++) {
    shardManifest.push({
      shardId: `decoder.part${String(i).padStart(2, '0')}`,
      url: decUrl,
      rangeStart: i * 16 * MIB,
      rangeEnd: (i + 1) * 16 * MIB - 1,
      length: 16 * MIB,
      writeOffset: shardManifest.length * 16 * MIB,
      expectedSha256: refHashes.decoder16MiB[i],
    });
  }
  for (let i = 0; i < 4; i++) {
    shardManifest.push({
      shardId: `embed.part${String(i).padStart(2, '0')}`,
      url: embUrl,
      rangeStart: i * 16 * MIB,
      rangeEnd: (i + 1) * 16 * MIB - 1,
      length: 16 * MIB,
      writeOffset: shardManifest.length * 16 * MIB,
      expectedSha256: refHashes.embed16MiB[i],
    });
  }
  const totalBytes = shardManifest.reduce((s, x) => s + x.length, 0);

  // Measure pure CPU SHA-256 throughput on a 64 MiB buffer (repeated 4x = 256 MiB)
  const benchBuf = crypto.randomBytes(64 * MIB);
  const tCpu0 = performance.now();
  for (let k = 0; k < 4; k++) {
    crypto.createHash('sha256').update(benchBuf).digest('hex');
  }
  const cpuHashMs = performance.now() - tCpu0;
  const pureSha256CpuMiBPerSec = round((256 / (cpuHashMs / 1000)), 2);
  console.log(`  Pure Node crypto SHA-256 CPU throughput (256 MiB): ${pureSha256CpuMiBPerSec} MiB/s (${round(cpuHashMs, 2)} ms)`);

  const matrix = [];
  for (const concurrency of [8, 16]) {
    for (const withSha256 of [false, true]) {
      const outPath = path.join(TMP_DIR, `trial2-c${concurrency}-sha${withSha256 ? '1' : '0'}.bin`);
      const res = await runWorkStealingDownload({
        tasks: shardManifest,
        concurrency,
        outFilePath: outPath,
        totalFileBytes: totalBytes,
        computeSha256: withSha256,
        verifyExpectedSha256: withSha256,
      });
      await fsp.rm(outPath, { force: true });
      const m = calcMetrics(totalBytes, res.elapsedMs);
      const entry = {
        concurrency,
        withIncrementalSha256: withSha256,
        shardCount: shardManifest.length,
        shardSizeMiB: 16,
        ...m,
        speedupVsBaseline: round(m.mibPerSec / baselineMibPerSec, 3),
        allShardsVerified: withSha256,
      };
      matrix.push(entry);
      console.log(
        `  c=${concurrency} sha256=${String(withSha256).padEnd(5)} -> ${entry.mibPerSec} MiB/s (${entry.mbps} Mbps, ${entry.speedupVsBaseline}x)`,
      );
    }
  }

  // Corrupted-Chunk Detection + Single-Chunk Range Retry Test
  const corruptIdx = 5;
  const corruptTarget = shardManifest[corruptIdx];
  const corruptFilePath = path.join(TMP_DIR, 'trial2-corrupt-test.bin');
  const corruptFh = await fsp.open(corruptFilePath, 'w+');
  let detectedCorruption = false;
  let corruptedActualSha256 = '';
  let retryElapsedMs = 0;
  let verifiedAfterRetry = false;
  try {
    await corruptFh.truncate(16 * MIB);
    // Download shard 5 into buffer, flip 1 bit, compute SHA-256 -> detect mismatch
    const r = await fetch(corruptTarget.url, {
      headers: { 'user-agent': USER_AGENT, range: `bytes=${corruptTarget.rangeStart}-${corruptTarget.rangeEnd}` },
    });
    const rawBuf = Buffer.from(await r.arrayBuffer());
    rawBuf[4096] ^= 0xff; // inject 1-byte corruption
    corruptedActualSha256 = crypto.createHash('sha256').update(rawBuf).digest('hex');
    detectedCorruption = corruptedActualSha256 !== corruptTarget.expectedSha256;
    if (!detectedCorruption) {
      throw new Error('Injected corruption was unexpectedly not detected');
    }

    // Retry ONLY that single 16 MiB shard via Range and write to disk
    const tRetry0 = performance.now();
    const retryRes = await fetchRangeToFileHandle({
      url: corruptTarget.url,
      rangeStart: corruptTarget.rangeStart,
      rangeEnd: corruptTarget.rangeEnd,
      fh: corruptFh,
      writeOffset: 0,
      expectedSha256: corruptTarget.expectedSha256,
      computeSha256: true,
    });
    retryElapsedMs = performance.now() - tRetry0;
    const diskSha = await hashFilePrefixSha256(corruptFilePath, 16 * MIB);
    verifiedAfterRetry = retryRes.sha256 === corruptTarget.expectedSha256 && diskSha === corruptTarget.expectedSha256;
  } finally {
    await corruptFh.close();
    await fsp.rm(corruptFilePath, { force: true });
  }
  console.log(
    `  Corrupted chunk test (shard #${corruptIdx}): detected=${detectedCorruption}, single-shard retry=${round(retryElapsedMs, 2)}ms, verified=${verifiedAfterRetry}`,
  );

  // Mid-Download Resume from Partial Offset Map
  const resumeBinPath = path.join(TMP_DIR, 'trial2-resume.bin');
  const resumeMapPath = path.join(TMP_DIR, 'resume-state.json');
  const phase1Shards = shardManifest.slice(0, 8);
  const phase1Res = await runWorkStealingDownload({
    tasks: phase1Shards,
    concurrency: 8,
    outFilePath: resumeBinPath,
    totalFileBytes: totalBytes,
    computeSha256: true,
    verifyExpectedSha256: true,
  });

  // Persist partial offset map (simulating browser crash / tab close after 128 MiB)
  const resumeState = {
    totalBytes,
    committedShards: phase1Shards.map((s, idx) => ({
      shardId: s.shardId,
      writeOffset: s.writeOffset,
      length: s.length,
      sha256: phase1Res.chunkResults[idx].sha256,
    })),
  };
  await fsp.writeFile(resumeMapPath, JSON.stringify(resumeState, null, 2));

  // Simulate restart: load map, verify persisted 8 shards on disk, fetch ONLY remaining 8 shards
  const loadedState = JSON.parse(await fsp.readFile(resumeMapPath, 'utf8'));
  const tVerify0 = performance.now();
  const committedSet = new Set();
  const checkFh = await fsp.open(resumeBinPath, 'r');
  try {
    const checkBuf = Buffer.allocUnsafe(16 * MIB);
    for (const item of loadedState.committedShards) {
      let off = 0;
      while (off < item.length) {
        const { bytesRead } = await checkFh.read(checkBuf, off, item.length - off, item.writeOffset + off);
        if (bytesRead <= 0) break;
        off += bytesRead;
      }
      const actual = crypto.createHash('sha256').update(checkBuf.subarray(0, off)).digest('hex');
      const manifestEntry = shardManifest.find((s) => s.shardId === item.shardId);
      if (manifestEntry && actual === manifestEntry.expectedSha256) {
        committedSet.add(item.shardId);
      }
    }
  } finally {
    await checkFh.close();
  }
  const resumeVerifyMs = performance.now() - tVerify0;

  const remainingShards = shardManifest.filter((s) => !committedSet.has(s.shardId));
  const tPhase2 = performance.now();
  const resumeFh = await fsp.open(resumeBinPath, 'r+');
  let phase2Next = 0;
  try {
    const workers = Array.from({ length: 8 }, async () => {
      while (true) {
        const i = phase2Next++;
        if (i >= remainingShards.length) break;
        const s = remainingShards[i];
        await fetchRangeToFileHandle({
          url: s.url,
          rangeStart: s.rangeStart,
          rangeEnd: s.rangeEnd,
          fh: resumeFh,
          writeOffset: s.writeOffset,
          expectedSha256: s.expectedSha256,
          computeSha256: true,
        });
      }
    });
    await Promise.all(workers);
    await resumeFh.sync();
  } finally {
    await resumeFh.close();
  }
  const phase2ElapsedMs = performance.now() - tPhase2;

  // Final full-file verification across all 16 shards
  let allFinalVerified = true;
  const finalFh = await fsp.open(resumeBinPath, 'r');
  try {
    const buf = Buffer.allocUnsafe(16 * MIB);
    for (const s of shardManifest) {
      let off = 0;
      while (off < s.length) {
        const { bytesRead } = await finalFh.read(buf, off, s.length - off, s.writeOffset + off);
        if (bytesRead <= 0) break;
        off += bytesRead;
      }
      const actual = crypto.createHash('sha256').update(buf.subarray(0, off)).digest('hex');
      if (actual !== s.expectedSha256) {
        allFinalVerified = false;
      }
    }
  } finally {
    await finalFh.close();
    await fsp.rm(resumeBinPath, { force: true });
    await fsp.rm(resumeMapPath, { force: true });
  }

  console.log(
    `  Resume test: phase1=128MiB (${round(phase1Res.elapsedMs, 1)}ms), verify=${round(resumeVerifyMs, 1)}ms, ` +
      `phase2=128MiB (${round(phase2ElapsedMs, 1)}ms), saved=128MiB, fullVerified=${allFinalVerified}`,
  );

  return {
    pureSha256CpuMiBPerSec,
    matrix,
    corruptedChunkRetryTest: {
      corruptedShardId: corruptTarget.shardId,
      shardBytes: corruptTarget.length,
      expectedSha256: corruptTarget.expectedSha256,
      corruptedActualSha256,
      detectedCorruption,
      retryBytes: corruptTarget.length,
      retryElapsedMs: round(retryElapsedMs, 2),
      verifiedAfterRetry,
    },
    resumableOffsetMapTest: {
      totalTargetBytes: totalBytes,
      phase1DownloadedBytes: 8 * 16 * MIB,
      phase1ElapsedMs: round(phase1Res.elapsedMs, 2),
      resumeVerifyMs: round(resumeVerifyMs, 2),
      skippedAlreadyVerifiedBytes: committedSet.size * 16 * MIB,
      phase2DownloadedBytes: remainingShards.length * 16 * MIB,
      phase2ElapsedMs: round(phase2ElapsedMs, 2),
      phase2MibPerSec: round((remainingShards.length * 16) / (phase2ElapsedMs / 1000), 3),
      fullFinalVerificationPassed: allFinalVerified,
    },
  };
}

function shannonEntropyBitsPerByte(buf) {
  const counts = new Uint32Array(256);
  for (let i = 0; i < buf.byteLength; i++) {
    counts[buf[i]] += 1;
  }
  let entropy = 0;
  const n = buf.byteLength;
  for (let i = 0; i < 256; i++) {
    if (counts[i] > 0) {
      const p = counts[i] / n;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

async function runTrial3CompressionAndFormatROI(repoByteTotals, bestDownloadMibPerSec) {
  console.log('\n=== Trial 3: Compression & Format ROI Benchmark (16 MiB slab0.part00.bin) ===');
  const sliceLen = 16 * MIB;
  const fh = await fsp.open(REPACKED_SLAB0_PATH, 'r');
  const rawSlice = Buffer.allocUnsafe(sliceLen);
  try {
    let off = 0;
    while (off < sliceLen) {
      const { bytesRead } = await fh.read(rawSlice, off, sliceLen - off, off);
      if (bytesRead <= 0) break;
      off += bytesRead;
    }
    if (off !== sliceLen) {
      throw new Error(`Short read on ${REPACKED_SLAB0_PATH}: ${off} != ${sliceLen}`);
    }
  } finally {
    await fh.close();
  }

  const rawSha256 = crypto.createHash('sha256').update(rawSlice).digest('hex');
  const entropyBitsPerByte = round(shannonEntropyBitsPerByte(rawSlice), 4);
  console.log(`  16 MiB slab0.part00.bin Shannon entropy: ${entropyBitsPerByte} / 8.0000 bits/byte`);

  const codecs = [
    {
      name: 'gzip-1',
      supported: true,
      compress: (b) => zlib.gzipSync(b, { level: 1 }),
      decompress: (b) => zlib.gunzipSync(b),
    },
    {
      name: 'gzip-6',
      supported: true,
      compress: (b) => zlib.gzipSync(b, { level: 6 }),
      decompress: (b) => zlib.gunzipSync(b),
    },
    {
      name: 'brotli-4',
      supported: true,
      compress: (b) => zlib.brotliCompressSync(b, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 4 } }),
      decompress: (b) => zlib.brotliDecompressSync(b),
    },
    {
      name: 'brotli-6',
      supported: true,
      compress: (b) => zlib.brotliCompressSync(b, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 6 } }),
      decompress: (b) => zlib.brotliDecompressSync(b),
    },
  ];

  const hasZstd = typeof zlib.zstdCompressSync === 'function' && typeof zlib.zstdDecompressSync === 'function';
  if (hasZstd) {
    codecs.push({
      name: 'zstd-default',
      supported: true,
      compress: (b) => zlib.zstdCompressSync(b),
      decompress: (b) => zlib.zstdDecompressSync(b),
    });
  }

  const results = [];
  for (const codec of codecs) {
    const tc0 = performance.now();
    const compressed = codec.compress(rawSlice);
    const compressMs = performance.now() - tc0;

    // Average 3 decompression iterations for stable CPU throughput measurement
    const decompIterations = 3;
    let decompTotalMs = 0;
    let roundtripVerified = true;
    for (let i = 0; i < decompIterations; i++) {
      const td0 = performance.now();
      const out = codec.decompress(compressed);
      decompTotalMs += performance.now() - td0;
      if (i === 0) {
        const outSha = crypto.createHash('sha256').update(out).digest('hex');
        roundtripVerified = outSha === rawSha256;
      }
    }
    const decompressMs = decompTotalMs / decompIterations;

    const ratioPct = (compressed.byteLength / sliceLen) * 100;
    const savingsPct = 100 - ratioPct;
    const compressMiBPerSec = sliceLen / MIB / (compressMs / 1000);
    const decompressMiBPerSec = sliceLen / MIB / (decompressMs / 1000);

    // At peak download speed (bestDownloadMibPerSec), estimate net end-to-end time on 2.47 GiB if serialized vs pipelined
    const rawWireSec = PRE_REPACKED_TOTAL_BYTES / (bestDownloadMibPerSec * MIB);
    const compressedWireSec = (PRE_REPACKED_TOTAL_BYTES * (ratioPct / 100)) / (bestDownloadMibPerSec * MIB);
    const decompressFullSec = PRE_REPACKED_TOTAL_BYTES / (decompressMiBPerSec * MIB);
    const netDeltaSecSerialized = compressedWireSec + decompressFullSec - rawWireSec;
    // Breakeven link bandwidth R where wire time saved equals CPU decompression time:
    // savingsFraction / R = 1 / D  =>  R = D * savingsFraction
    const savingsFrac = Math.max(0, 1 - compressed.byteLength / sliceLen);
    const breakevenMiBPerSec = savingsFrac > 0 ? decompressMiBPerSec * savingsFrac : 0;
    const breakevenMbps = breakevenMiBPerSec * 8 * (MIB / 1_000_000);

    const entry = {
      algorithm: codec.name,
      supported: true,
      rawBytes: sliceLen,
      compressedBytes: compressed.byteLength,
      compressedRatioPct: round(ratioPct, 3),
      savingsPct: round(savingsPct, 3),
      compressMs: round(compressMs, 2),
      compressMiBPerSec: round(compressMiBPerSec, 2),
      decompressMs: round(decompressMs, 2),
      decompressMiBPerSec: round(decompressMiBPerSec, 2),
      roundtripVerified,
      breakevenLinkMiBPerSec: round(breakevenMiBPerSec, 2),
      breakevenLinkMbps: round(breakevenMbps, 2),
      projected2_47GiB_RawWireSec: round(rawWireSec, 2),
      projected2_47GiB_CompressedWireSec: round(compressedWireSec, 2),
      projected2_47GiB_DecompressCpuSec: round(decompressFullSec, 2),
      projected2_47GiB_SerializedDeltaSec: round(netDeltaSecSerialized, 2),
      helpsAtPeakSpeed: netDeltaSecSerialized < 0,
    };
    results.push(entry);
    console.log(
      `  ${codec.name.padEnd(10)}: ratio=${entry.compressedRatioPct}% (save ${entry.savingsPct}%), ` +
        `decomp=${entry.decompressMiBPerSec} MiB/s, breakeven=${entry.breakevenLinkMbps} Mbps, serializedDelta=${entry.projected2_47GiB_SerializedDeltaSec}s`,
    );
  }

  if (!hasZstd) {
    results.push({
      algorithm: 'zstd',
      supported: false,
      nodeVersion: process.version,
      reason: 'Built-in node:zlib zstd requires Node >= 22.15 / 23.8 (not present in Node v20.18.2)',
    });
    console.log(`  zstd      : unsupported in ${process.version} (documented in JSON)`);
  }

  const manifest = JSON.parse(await fsp.readFile(REPACKED_MANIFEST_PATH, 'utf8'));
  const repackedShardCount =
    manifest.shards.slab0.length +
    manifest.shards.slab1.length +
    manifest.shards.slab2.length +
    manifest.shards.aux.length +
    5; // + manifest.json + 4 JSON assets

  return {
    sampleFile: 'model-bundle/qwen35-4b/slab0.part00.bin',
    sampleBytes: sliceLen,
    sampleSha256: rawSha256,
    shannonEntropyBitsPerByte: entropyBitsPerByte,
    codecs: results,
    formatComparison: {
      preRepackedWgslBytes: repoByteTotals.preRepackedWgslBundleBytes,
      preRepackedWgslGiB: round(repoByteTotals.preRepackedWgslBundleBytes / (1024 ** 3), 3),
      preRepackedFileCount: repackedShardCount,
      clientWasmRepackSecPreRepacked: 0,
      rawOnnxDecoderPlusJsonBytes: repoByteTotals.decoderPlusJsonBytes,
      rawOnnxDecoderPlusEmbedPlusJsonBytes: repoByteTotals.decoderPlusEmbedPlusJsonBytes,
      rawOnnxFullMultimodalQ4f16Bytes: repoByteTotals.fullMultimodalQ4f16Bytes,
      rawOnnx270GiBReferenceBytes: RAW_ONNX_270_GIB_BYTES,
      estimatedClientWasmRepackSecRawOnnx: 30,
      bytesSavedVsDecoderPlusEmbed:
        repoByteTotals.decoderPlusEmbedPlusJsonBytes - repoByteTotals.preRepackedWgslBundleBytes,
      bytesSavedVs270GiB: RAW_ONNX_270_GIB_BYTES - repoByteTotals.preRepackedWgslBundleBytes,
      historicalFirstOnlineDownloadAndCompileSec: 773.12,
    },
  };
}

async function fetchRangeHttpsAgent(urlStr, rangeStart, rangeEnd, agent) {
  const url = new URL(urlStr);
  const expectedLen = rangeEnd - rangeStart + 1;
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || 443,
        path: url.pathname + url.search,
        method: 'GET',
        agent,
        headers: {
          'user-agent': USER_AGENT,
          range: `bytes=${rangeStart}-${rangeEnd}`,
        },
      },
      (res) => {
        if (res.statusCode !== 206 && res.statusCode !== 200) {
          res.resume();
          reject(new Error(`HTTPS agent status ${res.statusCode}`));
          return;
        }
        const chunks = [];
        let total = 0;
        res.on('data', (c) => {
          chunks.push(c);
          total += c.byteLength;
        });
        res.on('end', () => {
          if (total !== expectedLen) {
            reject(new Error(`HTTPS truncated ${total} != ${expectedLen}`));
            return;
          }
          const alpn = res.socket?.alpnProtocol || 'http/1.1';
          resolve({ buf: Buffer.concat(chunks, total), alpn });
        });
      },
    );
    req.on('error', reject);
    req.end();
  });
}

async function fetchRangeHttp2(client, pathAndQuery, rangeStart, rangeEnd) {
  const expectedLen = rangeEnd - rangeStart + 1;
  return new Promise((resolve, reject) => {
    const req = client.request({
      ':method': 'GET',
      ':path': pathAndQuery,
      'user-agent': USER_AGENT,
      range: `bytes=${rangeStart}-${rangeEnd}`,
    });
    let status = 0;
    req.on('response', (headers) => {
      status = Number(headers[':status'] ?? 0);
    });
    const chunks = [];
    let total = 0;
    req.on('data', (c) => {
      chunks.push(c);
      total += c.byteLength;
    });
    req.on('end', () => {
      if (status !== 206 && status !== 200) {
        reject(new Error(`HTTP/2 status ${status}`));
        return;
      }
      if (total !== expectedLen) {
        reject(new Error(`HTTP/2 truncated ${total} != ${expectedLen}`));
        return;
      }
      resolve({ buf: Buffer.concat(chunks, total) });
    });
    req.on('error', reject);
  });
}

async function runTrial4TransportComparison(resolvedUrls, baselineMibPerSec, expected128MiBSha256) {
  console.log('\n=== Trial 4: Transport / Protocol Comparison (128 MiB window) ===');
  const cdnUrl = resolvedUrls.get('onnx/decoder_model_merged_q4f16.onnx_data');
  const parsedCdn = new URL(cdnUrl);
  const windowBytes = 128 * MIB;
  const results = [];

  const configs = [
    { concurrency: 8, chunkSizeBytes: 16 * MIB },
    { concurrency: 16, chunkSizeBytes: 8 * MIB },
  ];

  for (const { concurrency, chunkSizeBytes } of configs) {
    const numChunks = windowBytes / chunkSizeBytes;
    const tasks = Array.from({ length: numChunks }, (_, i) => ({
      url: cdnUrl,
      rangeStart: i * chunkSizeBytes,
      rangeEnd: (i + 1) * chunkSizeBytes - 1,
      writeOffset: i * chunkSizeBytes,
      expectedSha256: '',
    }));

    // 1. undici fetch() pool
    {
      const outPath = path.join(TMP_DIR, `trial4-fetch-c${concurrency}.bin`);
      const res = await runWorkStealingDownload({
        tasks,
        concurrency,
        outFilePath: outPath,
        totalFileBytes: windowBytes,
      });
      const sha = await hashFilePrefixSha256(outPath, windowBytes);
      if (sha !== expected128MiBSha256) throw new Error(`Trial 4 fetch SHA mismatch: ${sha}`);
      await fsp.rm(outPath, { force: true });
      const m = calcMetrics(windowBytes, res.elapsedMs);
      results.push({
        transport: 'undici_fetch',
        tcpConnections: 'pooled_automatic',
        alpnProtocol: 'http/1.1_pool',
        concurrency,
        chunkSizeMiB: chunkSizeBytes / MIB,
        ...m,
        speedupVsBaseline: round(m.mibPerSec / baselineMibPerSec, 3),
        sha256Verified: true,
      });
      console.log(`  undici_fetch          c=${concurrency} (${chunkSizeBytes / MIB}MiB): ${m.mibPerSec} MiB/s (${m.mbps} Mbps)`);
    }

    // 2. node:https multi-socket KeepAlive Agent
    {
      const outPath = path.join(TMP_DIR, `trial4-https-c${concurrency}.bin`);
      const agent = new https.Agent({ keepAlive: true, maxSockets: concurrency, maxTotalSockets: concurrency });
      const fh = await fsp.open(outPath, 'w+');
      await fh.truncate(windowBytes);
      let nextIdx = 0;
      let observedAlpn = 'http/1.1';
      const t0 = performance.now();
      try {
        await Promise.all(
          Array.from({ length: concurrency }, async () => {
            while (true) {
              const idx = nextIdx++;
              if (idx >= tasks.length) break;
              const t = tasks[idx];
              const { buf, alpn } = await fetchRangeHttpsAgent(t.url, t.rangeStart, t.rangeEnd, agent);
              observedAlpn = alpn;
              await fh.write(buf, 0, buf.byteLength, t.writeOffset);
            }
          }),
        );
        await fh.sync();
      } finally {
        await fh.close();
        agent.destroy();
      }
      const elapsedMs = performance.now() - t0;
      const sha = await hashFilePrefixSha256(outPath, windowBytes);
      if (sha !== expected128MiBSha256) throw new Error(`Trial 4 https agent SHA mismatch: ${sha}`);
      await fsp.rm(outPath, { force: true });
      const m = calcMetrics(windowBytes, elapsedMs);
      results.push({
        transport: 'node_https_multisocket',
        tcpConnections: concurrency,
        alpnProtocol: observedAlpn,
        concurrency,
        chunkSizeMiB: chunkSizeBytes / MIB,
        ...m,
        speedupVsBaseline: round(m.mibPerSec / baselineMibPerSec, 3),
        sha256Verified: true,
      });
      console.log(`  node_https_multisock  c=${concurrency} (${chunkSizeBytes / MIB}MiB): ${m.mibPerSec} MiB/s (${m.mbps} Mbps)`);
    }

    // 3. node:http2 single-connection multiplexed streams
    {
      const outPath = path.join(TMP_DIR, `trial4-h2-c${concurrency}.bin`);
      const client = http2.connect(parsedCdn.origin);
      client.setMaxListeners(concurrency + 8);
      const fh = await fsp.open(outPath, 'w+');
      await fh.truncate(windowBytes);
      let nextIdx = 0;
      const pathAndQuery = parsedCdn.pathname + parsedCdn.search;
      const t0 = performance.now();
      try {
        await Promise.all(
          Array.from({ length: concurrency }, async () => {
            while (true) {
              const idx = nextIdx++;
              if (idx >= tasks.length) break;
              const t = tasks[idx];
              const { buf } = await fetchRangeHttp2(client, pathAndQuery, t.rangeStart, t.rangeEnd);
              await fh.write(buf, 0, buf.byteLength, t.writeOffset);
            }
          }),
        );
        await fh.sync();
      } finally {
        await fh.close();
        client.close();
      }
      const elapsedMs = performance.now() - t0;
      const sha = await hashFilePrefixSha256(outPath, windowBytes);
      if (sha !== expected128MiBSha256) throw new Error(`Trial 4 http2 SHA mismatch: ${sha}`);
      await fsp.rm(outPath, { force: true });
      const m = calcMetrics(windowBytes, elapsedMs);
      results.push({
        transport: 'node_http2_single_conn',
        tcpConnections: 1,
        alpnProtocol: 'h2',
        concurrency,
        chunkSizeMiB: chunkSizeBytes / MIB,
        ...m,
        speedupVsBaseline: round(m.mibPerSec / baselineMibPerSec, 3),
        sha256Verified: true,
      });
      console.log(`  node_http2_single     c=${concurrency} (${chunkSizeBytes / MIB}MiB): ${m.mibPerSec} MiB/s (${m.mbps} Mbps)`);
    }
  }

  return {
    windowMiB: windowBytes / MIB,
    cdnHost: parsedCdn.host,
    results,
  };
}

function assertNoNullOrUndefined(obj, currentPath = '$') {
  if (obj === null || obj === undefined) {
    throw new Error(`Found null/undefined value at ${currentPath}`);
  }
  if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) {
      assertNoNullOrUndefined(obj[i], `${currentPath}[${i}]`);
    }
    return;
  }
  if (typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj)) {
      assertNoNullOrUndefined(v, `${currentPath}.${k}`);
    }
  }
}

async function main() {
  await fsp.rm(TMP_DIR, { recursive: true, force: true });
  await fsp.mkdir(TMP_DIR, { recursive: true });
  await fsp.mkdir(path.dirname(OUT_JSON_PATH), { recursive: true });

  const startedAt = new Date().toISOString();
  const suiteT0 = performance.now();

  try {
    const endpointProbe = await probeEndpoints();
    const trial0 = await runTrial0Baseline(endpointProbe.resolvedUrls);
    const trial1 = await runTrial1ParallelRange(
      endpointProbe.resolvedUrls,
      trial0.baselineMibPerSec,
      trial0.decoderPrefix256MiB.sha256,
    );
    const trial2 = await runTrial2ShardedSecurityAndResume(
      endpointProbe.resolvedUrls,
      trial0.baselineMibPerSec,
      trial0.referenceHashes,
    );
    const trial3 = await runTrial3CompressionAndFormatROI(
      endpointProbe.repoByteTotals,
      trial1.bestConfig.mibPerSec,
    );
    const trial4 = await runTrial4TransportComparison(
      endpointProbe.resolvedUrls,
      trial0.baselineMibPerSec,
      trial0.referenceHashes.decoder128MiBSha256,
    );

    const report = {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      startedAt,
      totalBenchmarkElapsedSec: round((performance.now() - suiteT0) / 1000, 2),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        modelId: MODEL_ID,
        preRepackedBundleBytes: PRE_REPACKED_TOTAL_BYTES,
        rawOnnx270GiBBytes: RAW_ONNX_270_GIB_BYTES,
      },
      endpointProbe: {
        probes: endpointProbe.probes,
        repoByteTotals: endpointProbe.repoByteTotals,
      },
      trial0BaselineSequential: {
        embedTokensFull: trial0.embedTokensFull,
        decoderPrefix256MiB: trial0.decoderPrefix256MiB,
        combined: trial0.combined,
        baselineMibPerSec: trial0.baselineMibPerSec,
      },
      trial1ParallelRange: trial1,
      trial2ShardedSecurityAndResume: trial2,
      trial3CompressionAndFormatROI: trial3,
      trial4TransportComparison: trial4,
    };

    assertNoNullOrUndefined(report);
    await fsp.writeFile(OUT_JSON_PATH, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
    console.log(`\nSaved benchmark report to ${OUT_JSON_PATH} (zero null/undefined fields verified).`);
  } finally {
    await fsp.rm(TMP_DIR, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
