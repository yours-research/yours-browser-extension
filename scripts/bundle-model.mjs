import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const MODEL_ID = 'onnx-community/Qwen3.5-4B-ONNX-OPT';
const HF_BASE_URL = `https://huggingface.co/${MODEL_ID}/resolve/main`;

const BUNDLE_DIR = path.join(ROOT_DIR, 'model-bundle', 'qwen35-4b');
const HF_TMP_DIR = path.join(ROOT_DIR, 'model-bundle', '.hf-tmp');
const DIST_DIR = path.join(ROOT_DIR, 'dist');
const DIST_MODEL_DIR = path.join(DIST_DIR, 'models', 'qwen35-4b');

const BENCH_OPFS_ROOT = path.join(ROOT_DIR, '.bench-profile', 'Default', 'File System', '000', 't');
const BENCH_OPFS_DATA_DIR = path.join(BENCH_OPFS_ROOT, '00');
const BENCH_OPFS_LOG_PATH = path.join(BENCH_OPFS_ROOT, 'Paths', '000003.log');
const WASM_PATH = path.join(
  ROOT_DIR,
  'crates',
  'tinygpt-wgsl-engine',
  'target',
  'wasm32-unknown-unknown',
  'release',
  'tinygpt_wgsl_engine.wasm',
);

const JSON_ASSET_FILES = [
  'config.json',
  'tokenizer_config.json',
  'generation_config.json',
  'tokenizer.json',
];

const SHARD_MAX_BYTES = 128 * 1024 * 1024;
const COPY_CHUNK_BYTES = 16 * 1024 * 1024;

const HF_DOWNLOAD_FILES = [
  ...JSON_ASSET_FILES,
  'onnx/decoder_model_merged_q4f16.onnx',
  'onnx/decoder_model_merged_q4f16.onnx_data',
  'onnx/decoder_model_merged_q4f16.onnx_data_1',
];

const QWEN35_HIDDEN = 2560;
const QWEN35_INTERMEDIATE = 9216;
const QWEN35_NUM_LAYERS = 32;
const QWEN35_VOCAB_SIZE = 248320;
const QWEN35_FULL_ATTN_LAYERS = new Set([3, 7, 11, 15, 19, 23, 27, 31]);

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

function sha256Hex(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}

/**
 * Finds the latest OPFS backing file ID (8 zero-padded digits) committed in Chrome's
 * LevelDB Paths log for a given OPFS entry name.
 */
function findCommittedOpfsId(logBuffer, entryName) {
  const nameBuf = Buffer.from(entryName, 'utf8');
  const needle = Buffer.allocUnsafe(5 + nameBuf.length);
  needle[0] = 0x00;
  needle.writeUInt32LE(nameBuf.length, 1);
  nameBuf.copy(needle, 5);

  let latestId = null;
  let pos = 0;
  while (true) {
    const idx = logBuffer.indexOf(needle, pos);
    if (idx === -1) {
      break;
    }
    if (idx >= 11) {
      const prefix = logBuffer.subarray(idx - 11, idx).toString('latin1');
      if (/^00[\\/]\d{8}$/.test(prefix)) {
        latestId = prefix.slice(3);
      }
    }
    pos = idx + 1;
  }
  return latestId;
}

function copyFileIfChanged(srcPath, destPath) {
  const srcStat = fs.statSync(srcPath);
  if (fs.existsSync(destPath)) {
    const destStat = fs.statSync(destPath);
    if (destStat.isFile() && destStat.size === srcStat.size) {
      if (srcStat.size >= 65536 || fs.readFileSync(srcPath).equals(fs.readFileSync(destPath))) {
        return 'unchanged';
      }
    }
    fs.rmSync(destPath, { recursive: true, force: true });
  }
  fs.copyFileSync(srcPath, destPath);
  return 'copied';
}

function syncBundleToDist(bundleDir, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  const srcFiles = fs
    .readdirSync(bundleDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && !entry.name.endsWith('.tmp'))
    .map((entry) => entry.name);
  const srcSet = new Set(srcFiles);

  for (const existing of fs.readdirSync(destDir)) {
    if (!srcSet.has(existing)) {
      fs.rmSync(path.join(destDir, existing), { recursive: true, force: true });
    }
  }

  const results = [];
  for (const fileName of srcFiles) {
    const srcPath = path.join(bundleDir, fileName);
    const destPath = path.join(destDir, fileName);
    const srcSize = fs.statSync(srcPath).size;

    let needsSync = !fs.existsSync(destPath);
    if (!needsSync) {
      const destStat = fs.statSync(destPath);
      if (!destStat.isFile() || destStat.size !== srcSize) {
        needsSync = true;
      } else if (srcSize < 65536) {
        needsSync = !fs.readFileSync(srcPath).equals(fs.readFileSync(destPath));
      }
    }

    if (!needsSync) {
      results.push({ fileName, size: srcSize, action: 'unchanged' });
      continue;
    }

    if (fs.existsSync(destPath)) {
      fs.unlinkSync(destPath);
    }
    try {
      fs.linkSync(srcPath, destPath);
      results.push({ fileName, size: srcSize, action: 'hardlinked' });
    } catch {
      fs.copyFileSync(srcPath, destPath);
      results.push({ fileName, size: srcSize, action: 'copied' });
    }
  }
  return results;
}

function cleanUnexpectedBundleFiles(bundleDir, allowedNames) {
  const allowed = new Set(allowedNames);
  for (const existing of fs.readdirSync(bundleDir)) {
    if (!allowed.has(existing)) {
      fs.rmSync(path.join(bundleDir, existing), { recursive: true, force: true });
    }
  }
}

function splitFileIntoShards(
  srcPath,
  destDir,
  prefix,
  expectedTotalBytes,
  exportActions,
  options = {},
) {
  const maxShardBytes = options.maxShardBytes ?? SHARD_MAX_BYTES;
  const forceRewrite = Boolean(options.forceRewrite);
  const actualTotal = fs.statSync(srcPath).size;
  assert(
    actualTotal === expectedTotalBytes,
    `Source size mismatch for ${srcPath}: got ${actualTotal}, expected ${expectedTotalBytes}`,
  );
  assert(
    expectedTotalBytes % 4 === 0 && maxShardBytes % 4 === 0,
    `Sharded weights must be 4-byte aligned (${prefix}: total=${expectedTotalBytes}, max=${maxShardBytes})`,
  );

  const shards = [];
  const srcFd = fs.openSync(srcPath, 'r');
  const ioBuf = Buffer.allocUnsafe(COPY_CHUNK_BYTES);

  try {
    let srcOffset = 0;
    let partIdx = 0;

    while (srcOffset < expectedTotalBytes) {
      const partBytes = Math.min(maxShardBytes, expectedTotalBytes - srcOffset);
      const fileName = `${prefix}.part${String(partIdx).padStart(2, '0')}.bin`;
      const destPath = path.join(destDir, fileName);

      let skipWrite = false;
      if (!forceRewrite && fs.existsSync(destPath)) {
        const destStat = fs.statSync(destPath);
        if (destStat.isFile() && destStat.size === partBytes) {
          skipWrite = true;
        }
      }

      if (skipWrite) {
        exportActions.set(fileName, 'unchanged');
      } else {
        const tmpPath = `${destPath}.tmp`;
        fs.rmSync(tmpPath, { force: true });
        const dstFd = fs.openSync(tmpPath, 'w');
        try {
          let remaining = partBytes;
          let curPos = srcOffset;
          while (remaining > 0) {
            const toRead = Math.min(ioBuf.byteLength, remaining);
            const bytesRead = fs.readSync(srcFd, ioBuf, 0, toRead, curPos);
            assert(bytesRead > 0, `Unexpected EOF reading ${srcPath} at ${curPos}`);
            writeAllSync(dstFd, ioBuf.subarray(0, bytesRead));
            curPos += bytesRead;
            remaining -= bytesRead;
          }
        } finally {
          fs.closeSync(dstFd);
        }
        fs.rmSync(destPath, { force: true });
        fs.renameSync(tmpPath, destPath);
        exportActions.set(fileName, forceRewrite ? 'repacked-shard' : 'sharded');
      }

      shards.push({ file: fileName, bytes: partBytes });
      srcOffset += partBytes;
      partIdx += 1;
    }
  } finally {
    fs.closeSync(srcFd);
  }

  const shardSum = shards.reduce((sum, s) => sum + s.bytes, 0);
  assert(
    shardSum === expectedTotalBytes,
    `Shard sum mismatch for ${prefix}: got ${shardSum}, expected ${expectedTotalBytes}`,
  );
  return shards;
}

function exportFromBenchProfile() {
  assert(
    fs.existsSync(BENCH_OPFS_LOG_PATH) && fs.existsSync(BENCH_OPFS_DATA_DIR),
    `Local OPFS profile not found at ${BENCH_OPFS_ROOT}. Re-run with --from-hf to download from HuggingFace.`,
  );

  const logBuf = fs.readFileSync(BENCH_OPFS_LOG_PATH);

  // 1. Dynamically resolve and verify manifest.json and slab binaries via Paths/000003.log.
  const manifestId = findCommittedOpfsId(logBuf, 'manifest.json');
  assert(manifestId !== null, `Could not find manifest.json mapping in ${BENCH_OPFS_LOG_PATH}`);
  const manifestSrcPath = path.join(BENCH_OPFS_DATA_DIR, manifestId);
  assert(fs.existsSync(manifestSrcPath), `Missing OPFS manifest backing file ${manifestSrcPath}`);

  const manifest = JSON.parse(fs.readFileSync(manifestSrcPath, 'utf8'));
  assert(manifest.version === 1 && manifest.complete === true, `Invalid WGSL manifest in ${manifestSrcPath}`);
  for (const field of ['auxBytes', 'slab0Bytes', 'slab1Bytes', 'slab2Bytes']) {
    assert(
      Number.isInteger(manifest[field]) && manifest[field] > 0,
      `Invalid ${field} in ${manifestSrcPath}: ${manifest[field]}`,
    );
  }
  console.log(`Verified OPFS manifest.json -> 00/${manifestId} (modelId=${manifest.modelId})`);

  const slabSpecs = [
    { key: 'aux', name: 'aux.bin', manifestKey: 'auxBytes' },
    { key: 'slab0', name: 'slab0.bin', manifestKey: 'slab0Bytes' },
    { key: 'slab1', name: 'slab1.bin', manifestKey: 'slab1Bytes' },
    { key: 'slab2', name: 'slab2.bin', manifestKey: 'slab2Bytes' },
  ];

  const resolvedSlabs = new Map();
  for (const { key, name, manifestKey } of slabSpecs) {
    const fileId = findCommittedOpfsId(logBuf, name);
    assert(fileId !== null, `Paths/000003.log does not contain committed entry for ${name}`);
    const srcPath = path.join(BENCH_OPFS_DATA_DIR, fileId);
    assert(fs.existsSync(srcPath), `Missing OPFS backing file ${srcPath} for ${name}`);
    const actualSize = fs.statSync(srcPath).size;
    const expectedSize = manifest[manifestKey];
    assert(
      actualSize === expectedSize,
      `Size mismatch for ${name} (00/${fileId}): got ${actualSize} bytes, manifest expects ${expectedSize}`,
    );
    console.log(`Verified OPFS ${name} -> 00/${fileId} (${actualSize} bytes)`);
    resolvedSlabs.set(key, { srcPath, size: actualSize });
  }

  // 2. Dynamically scan small files (< 4096 bytes) for OPFS cache metadata ending in target JSON filenames.
  const entries = fs.readdirSync(BENCH_OPFS_DATA_DIR).sort();
  const matchedMeta = new Map();

  for (const entry of entries) {
    if (!/^\d{8}$/.test(entry)) {
      continue;
    }
    const metaPath = path.join(BENCH_OPFS_DATA_DIR, entry);
    const stat = fs.statSync(metaPath);
    if (!stat.isFile() || stat.size >= 4096) {
      continue;
    }
    let parsed;
    try {
      parsed = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    } catch {
      continue;
    }
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      typeof parsed.key !== 'string' ||
      typeof parsed.contentLength !== 'number' ||
      parsed.complete !== true
    ) {
      continue;
    }
    for (const targetName of JSON_ASSET_FILES) {
      if (!parsed.key.endsWith(`/${targetName}`)) {
        continue;
      }
      if (manifest.modelId && !parsed.key.includes(manifest.modelId)) {
        continue;
      }
      matchedMeta.set(targetName, { metaId: entry, meta: parsed });
    }
  }

  const resolvedJsonFiles = new Map();
  for (const targetName of JSON_ASSET_FILES) {
    const info = matchedMeta.get(targetName);
    assert(info, `Missing complete OPFS metadata entry for /${targetName} in ${BENCH_OPFS_DATA_DIR}`);
    const metaNum = Number(info.metaId);
    assert(Number.isInteger(metaNum) && metaNum >= 2, `Unexpected OPFS metadata file id ${info.metaId}`);
    const companionId = String(metaNum - 2).padStart(8, '0');
    const loggedBinId = findCommittedOpfsId(logBuf, `${sha256Hex(info.meta.key)}.bin`);
    if (loggedBinId !== null) {
      assert(
        loggedBinId === companionId,
        `OPFS log mismatch for ${targetName}: metadata ${info.metaId} implies ${companionId}, log says ${loggedBinId}`,
      );
    }

    const companionPath = path.join(BENCH_OPFS_DATA_DIR, companionId);
    assert(fs.existsSync(companionPath), `Missing companion backing file ${companionPath} for ${targetName}`);
    const companionSize = fs.statSync(companionPath).size;
    assert(
      companionSize === info.meta.contentLength,
      `Size mismatch for ${targetName} (00/${companionId}): got ${companionSize}, metadata contentLength=${info.meta.contentLength}`,
    );
    // Validate JSON payload before copying.
    JSON.parse(fs.readFileSync(companionPath, 'utf8'));
    console.log(
      `Verified OPFS ${targetName} -> meta 00/${info.metaId}, data 00/${companionId} (${companionSize} bytes)`,
    );
    resolvedJsonFiles.set(targetName, { srcPath: companionPath, size: companionSize });
  }

  // 3. Copy JSONs + aux_params.bin (avoiding Windows reserved device name AUX), shard slab0/1/2 <= 128 MiB, write manifest.json.
  fs.mkdirSync(BUNDLE_DIR, { recursive: true });
  const exportActions = new Map();

  for (const fileName of JSON_ASSET_FILES) {
    const { srcPath } = resolvedJsonFiles.get(fileName);
    const destPath = path.join(BUNDLE_DIR, fileName);
    exportActions.set(fileName, copyFileIfChanged(srcPath, destPath));
  }

  const auxInfo = resolvedSlabs.get('aux');
  exportActions.set(
    'aux_params.bin',
    copyFileIfChanged(auxInfo.srcPath, path.join(BUNDLE_DIR, 'aux_params.bin')),
  );

  const slab0Info = resolvedSlabs.get('slab0');
  const slab1Info = resolvedSlabs.get('slab1');
  const slab2Info = resolvedSlabs.get('slab2');

  const slab0Shards = splitFileIntoShards(slab0Info.srcPath, BUNDLE_DIR, 'slab0', slab0Info.size, exportActions);
  const slab1Shards = splitFileIntoShards(slab1Info.srcPath, BUNDLE_DIR, 'slab1', slab1Info.size, exportActions);
  const slab2Shards = splitFileIntoShards(slab2Info.srcPath, BUNDLE_DIR, 'slab2', slab2Info.size, exportActions);
  const auxShards = [{ file: 'aux_params.bin', bytes: auxInfo.size }];

  const bundledManifest = {
    version: manifest.version,
    modelId: manifest.modelId,
    complete: true,
    slab0Bytes: manifest.slab0Bytes,
    slab1Bytes: manifest.slab1Bytes,
    slab2Bytes: manifest.slab2Bytes,
    auxBytes: manifest.auxBytes,
    shardMaxBytes: SHARD_MAX_BYTES,
    shards: {
      slab0: slab0Shards,
      slab1: slab1Shards,
      slab2: slab2Shards,
      aux: auxShards,
    },
    createdAt: manifest.createdAt,
  };
  const manifestDestPath = path.join(BUNDLE_DIR, 'manifest.json');
  const manifestJson = JSON.stringify(bundledManifest, null, 2);
  const manifestSame =
    fs.existsSync(manifestDestPath) && fs.readFileSync(manifestDestPath, 'utf8') === manifestJson;
  if (!manifestSame) {
    fs.writeFileSync(manifestDestPath, manifestJson);
    exportActions.set('manifest.json', 'written');
  } else {
    exportActions.set('manifest.json', 'unchanged');
  }

  const expectedFiles = [
    ...JSON_ASSET_FILES,
    'aux_params.bin',
    ...slab0Shards.map((s) => s.file),
    ...slab1Shards.map((s) => s.file),
    ...slab2Shards.map((s) => s.file),
    'manifest.json',
  ];
  cleanUnexpectedBundleFiles(BUNDLE_DIR, expectedFiles);

  return { expectedFiles, exportActions };
}

// --- HuggingFace Fallback (--from-hf) & WASM Repack ---

function computeAuxLayout() {
  let off = 0;
  const invFreqOff = off;
  off += 32;
  const finalNormOff = off;
  off += QWEN35_HIDDEN;

  const layers = [];
  for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
    const inputNormOff = off;
    off += QWEN35_HIDDEN;
    const postNormOff = off;
    off += QWEN35_HIDDEN;
    const isFullAttn = QWEN35_FULL_ATTN_LAYERS.has(l);
    if (isFullAttn) {
      const qNormOff = off;
      off += 256;
      const kNormOff = off;
      off += 256;
      layers.push({ inputNormOff, postNormOff, isFullAttn: true, dnAuxOff: 0, convWOff: 0, qNormOff, kNormOff });
    } else {
      const dnAuxOff = off;
      off += 32 + 32 + 128;
      const convWOff = off;
      off += 8192 * 4;
      layers.push({ inputNormOff, postNormOff, isFullAttn: false, dnAuxOff, convWOff, qNormOff: 0, kNormOff: 0 });
    }
  }

  return { invFreqOff, finalNormOff, totalF32s: off, layers };
}

function readVarint(buf, pos) {
  let lo = 0;
  let hi = 0;
  let shift = 0;
  while (pos < buf.length) {
    const b = buf[pos++];
    if (shift < 28) {
      lo |= (b & 0x7f) << shift;
    } else if (shift === 28) {
      lo |= (b & 0x0f) << 28;
      hi |= (b & 0x70) >>> 4;
    } else {
      hi |= (b & 0x7f) << (shift - 32);
    }
    if ((b & 0x80) === 0) {
      break;
    }
    shift += 7;
  }
  return { val: hi * 4294967296 + (lo >>> 0), pos };
}

function parseProtoFields(buf, start, end, onField) {
  let pos = start;
  while (pos < end) {
    const tag = readVarint(buf, pos);
    pos = tag.pos;
    const field = Math.floor(tag.val / 8);
    const wire = tag.val & 7;
    if (wire === 0) {
      const v = readVarint(buf, pos);
      pos = v.pos;
      onField(field, wire, v.val);
    } else if (wire === 1) {
      onField(field, wire, pos);
      pos += 8;
    } else if (wire === 2) {
      const len = readVarint(buf, pos);
      pos = len.pos;
      const sliceEnd = pos + len.val;
      onField(field, wire, pos, sliceEnd);
      pos = sliceEnd;
    } else if (wire === 5) {
      onField(field, wire, pos);
      pos += 4;
    } else {
      throw new Error(`Unsupported protobuf wire type ${wire} at offset ${pos}`);
    }
  }
}

const utf8Decoder = new TextDecoder();

function parseTensorProto(buf, start, end) {
  const tensor = {
    name: '',
    dims: [],
    dataType: 0,
    rawData: null,
    externalLocation: null,
    externalOffset: 0,
    externalLength: 0,
  };

  parseProtoFields(buf, start, end, (field, wire, a, b) => {
    if (field === 1 && wire === 0) {
      tensor.dims.push(a);
    } else if (field === 1 && wire === 2 && b !== undefined) {
      let p = a;
      while (p < b) {
        const r = readVarint(buf, p);
        tensor.dims.push(r.val);
        p = r.pos;
      }
    } else if (field === 2 && wire === 0) {
      tensor.dataType = a;
    } else if (field === 8 && wire === 2 && b !== undefined) {
      tensor.name = utf8Decoder.decode(buf.subarray(a, b));
    } else if (field === 9 && wire === 2 && b !== undefined) {
      tensor.rawData = buf.subarray(a, b);
    } else if (field === 13 && wire === 2 && b !== undefined) {
      let key = '';
      let val = '';
      parseProtoFields(buf, a, b, (kf, kw, ka, kb) => {
        if (kw === 2 && kb !== undefined) {
          if (kf === 1) key = utf8Decoder.decode(buf.subarray(ka, kb));
          else if (kf === 2) val = utf8Decoder.decode(buf.subarray(ka, kb));
        }
      });
      if (key === 'location') tensor.externalLocation = val;
      else if (key === 'offset') tensor.externalOffset = Number(val);
      else if (key === 'length') tensor.externalLength = Number(val);
    }
  });

  return tensor;
}

function parseOnnxTensorManifest(onnxBytes) {
  const map = new Map();
  parseProtoFields(onnxBytes, 0, onnxBytes.length, (field, wire, start, end) => {
    if (field === 7 && wire === 2 && end !== undefined) {
      parseProtoFields(onnxBytes, start, end, (gf, gw, gStart, gEnd) => {
        if (gf === 5 && gw === 2 && gEnd !== undefined) {
          const tensor = parseTensorProto(onnxBytes, gStart, gEnd);
          if (tensor.name) {
            map.set(tensor.name, tensor);
          }
        }
      });
    }
  });
  return map;
}

function readExactFromFd(fd, position, length) {
  const out = Buffer.allocUnsafe(length);
  let offset = 0;
  while (offset < length) {
    const read = fs.readSync(fd, out, offset, length - offset, position + offset);
    if (read <= 0) {
      throw new Error(`Unexpected EOF reading external ONNX tensor at offset ${position + offset}`);
    }
    offset += read;
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

function writeAllSync(fd, buf) {
  let offset = 0;
  while (offset < buf.byteLength) {
    offset += fs.writeSync(fd, buf, offset, buf.byteLength - offset);
  }
}

function readTensorBytes(tensor, extFds, subOffset = 0, subLength) {
  if (tensor.rawData) {
    const start = subOffset;
    const end = subLength !== undefined ? start + subLength : tensor.rawData.length;
    return tensor.rawData.slice(start, end);
  }
  if (!tensor.externalLocation) {
    throw new Error(`Tensor ${tensor.name} has neither rawData nor externalLocation.`);
  }
  const fd = extFds[tensor.externalLocation];
  if (fd === undefined) {
    throw new Error(`Missing external ONNX file ${tensor.externalLocation} for ${tensor.name}.`);
  }
  const start = tensor.externalOffset + subOffset;
  const len = subLength !== undefined ? subLength : tensor.externalLength - subOffset;
  return readExactFromFd(fd, start, len);
}

function repackQ4MatrixToFd(fd, manifestMap, prefix, expectedRows, expectedCols, extFds, wasm) {
  const tQuant = manifestMap.get(`${prefix}_weight_quant`);
  const tScales = manifestMap.get(`${prefix}_weight_scales`);
  const tZp = manifestMap.get(`${prefix}_weight_zp`);
  if (!tQuant || !tScales || !tZp) {
    throw new Error(`Missing Q4 ONNX initializers for prefix ${prefix}`);
  }
  const blocksPerRow = expectedCols / 32;
  if (tQuant.dims[0] !== expectedRows || tQuant.dims[1] !== blocksPerRow) {
    throw new Error(
      `Shape mismatch on ${prefix}: got [${tQuant.dims.join(',')}], expected [${expectedRows},${blocksPerRow},16]`,
    );
  }

  const zpStride = Math.ceil(blocksPerRow / 2);
  const quantRowBytes = blocksPerRow * 16;
  const scalesRowBytes = blocksPerRow * 2;
  const outRowBytes = blocksPerRow * 20;
  const bytesPerRow = quantRowBytes + scalesRowBytes + zpStride + outRowBytes + 32;
  const scratchBase = wasm.wasm_scratch_ptr();
  const scratchLen = wasm.wasm_scratch_len();
  const maxBatchRows = Math.max(1, Math.min(256, Math.floor((scratchLen - 64) / bytesPerRow)));

  let totalWritten = 0;
  for (let rowStart = 0; rowStart < expectedRows; rowStart += maxBatchRows) {
    const rows = Math.min(maxBatchRows, expectedRows - rowStart);
    const qBytes = readTensorBytes(tQuant, extFds, rowStart * quantRowBytes, rows * quantRowBytes);
    const sBytes = readTensorBytes(tScales, extFds, rowStart * scalesRowBytes, rows * scalesRowBytes);
    const zBytes = readTensorBytes(tZp, extFds, rowStart * zpStride, rows * zpStride);

    const qPtr = scratchBase;
    const sPtr = (qPtr + qBytes.byteLength + 3) & ~3;
    const zPtr = (sPtr + sBytes.byteLength + 3) & ~3;
    const outPtr = (zPtr + zBytes.byteLength + 3) & ~3;

    const memU8 = new Uint8Array(wasm.memory.buffer);
    memU8.set(qBytes, qPtr);
    memU8.set(sBytes, sPtr);
    memU8.set(zBytes, zPtr);

    wasm.repack_q4_rows(rows, blocksPerRow, qPtr, sPtr, zPtr, outPtr);

    const outByteLen = rows * outRowBytes;
    const outView = new Uint8Array(wasm.memory.buffer, outPtr, outByteLen);
    writeAllSync(fd, outView);
    totalWritten += outByteLen;
  }

  return totalWritten;
}

async function downloadToHfTmp(relPath) {
  const url = `${HF_BASE_URL}/${relPath}`;
  const destPath = path.join(HF_TMP_DIR, relPath);
  const metaPath = `${destPath}.meta.json`;
  const partPath = `${destPath}.part`;

  fs.mkdirSync(path.dirname(destPath), { recursive: true });

  if (fs.existsSync(destPath) && fs.existsSync(metaPath)) {
    try {
      const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
      const actualSize = fs.statSync(destPath).size;
      if (meta.complete === true && meta.contentLength === actualSize) {
        console.log(`HF cache hit: ${relPath} (${actualSize} bytes)`);
        return destPath;
      }
    } catch {
      // Re-download on invalid metadata.
    }
  }

  console.log(`Downloading ${url} ...`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok || !res.body) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  }
  const contentLengthHeader = res.headers.get('content-length');
  const expectedLength = contentLengthHeader ? Number(contentLengthHeader) : null;

  fs.rmSync(partPath, { force: true });
  await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(partPath));

  const actualSize = fs.statSync(partPath).size;
  if (expectedLength !== null && Number.isFinite(expectedLength) && actualSize !== expectedLength) {
    throw new Error(`Truncated download for ${relPath}: got ${actualSize} bytes, expected ${expectedLength}`);
  }

  fs.renameSync(partPath, destPath);
  fs.writeFileSync(
    metaPath,
    JSON.stringify({ url, contentLength: actualSize, complete: true, downloadedAt: new Date().toISOString() }, null, 2),
  );
  console.log(`Downloaded ${relPath} (${actualSize} bytes)`);
  return destPath;
}

async function exportFromHuggingFace() {
  assert(
    fs.existsSync(WASM_PATH),
    `Missing Rust WASM engine at ${WASM_PATH}. Run npm run build:wasm first.`,
  );

  for (const relPath of HF_DOWNLOAD_FILES) {
    await downloadToHfTmp(relPath);
  }

  const wasmBytes = fs.readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const wasm = instance.exports;

  const onnxPath = path.join(HF_TMP_DIR, 'onnx', 'decoder_model_merged_q4f16.onnx');
  const data0Path = path.join(HF_TMP_DIR, 'onnx', 'decoder_model_merged_q4f16.onnx_data');
  const data1Path = path.join(HF_TMP_DIR, 'onnx', 'decoder_model_merged_q4f16.onnx_data_1');

  console.log('Parsing ONNX tensor manifest...');
  const onnxBytes = new Uint8Array(fs.readFileSync(onnxPath));
  const manifestMap = parseOnnxTensorManifest(onnxBytes);

  fs.mkdirSync(BUNDLE_DIR, { recursive: true });

  const extFds = {
    'decoder_model_merged_q4f16.onnx_data': fs.openSync(data0Path, 'r'),
    'decoder_model_merged_q4f16.onnx_data_1': fs.openSync(data1Path, 'r'),
  };

  try {
    // 1. Pack aux_params.bin (avoiding Windows reserved device name AUX)
    console.log('Packing FP32 norms & DeltaNet parameters into aux_params.bin...');
    const auxLayout = computeAuxLayout();
    const auxF32 = new Float32Array(auxLayout.totalF32s);
    const writeF32Tensor = (tensorName, dstOffset, expectedCount) => {
      const t = manifestMap.get(tensorName);
      if (!t) throw new Error(`Missing ONNX initializer ${tensorName}`);
      const raw = readTensorBytes(t, extFds);
      const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      if (f32.length !== expectedCount) {
        throw new Error(`Aux tensor ${tensorName} length ${f32.length} != ${expectedCount}`);
      }
      auxF32.set(f32, dstOffset);
    };

    writeF32Tensor('model.inv_freq', auxLayout.invFreqOff, 32);
    writeF32Tensor('model.layers.32.final_norm_layernorm.weight', auxLayout.finalNormOff, QWEN35_HIDDEN);

    for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
      const la = auxLayout.layers[l];
      writeF32Tensor(`model.layers.${l}.input_layernorm.weight`, la.inputNormOff, QWEN35_HIDDEN);
      writeF32Tensor(`model.layers.${l}.post_attention_layernorm.weight`, la.postNormOff, QWEN35_HIDDEN);
      if (la.isFullAttn) {
        writeF32Tensor(`model.layers.${l}.attn.q_norm.layernorm.weight`, la.qNormOff, 256);
        writeF32Tensor(`model.layers.${l}.attn.k_norm.layernorm.weight`, la.kNormOff, 256);
      } else {
        writeF32Tensor(`model.layers.${l}.gdn.A_neg_exp`, la.dnAuxOff, 32);
        writeF32Tensor(`model.layers.${l}.gdn.dt_bias`, la.dnAuxOff + 32, 32);
        writeF32Tensor(`model.layers.${l}.gdn.norm.weight`, la.dnAuxOff + 64, 128);

        const tConv = manifestMap.get(`model.layers.${l}.gdn.conv1d.weight_3d`);
        if (!tConv) throw new Error(`Missing conv1d weights for layer ${l}`);
        const convF16Bytes = readTensorBytes(tConv, extFds);
        const count = 8192 * 4;
        const srcPtr = wasm.wasm_scratch_ptr();
        const dstPtr = (srcPtr + convF16Bytes.byteLength + 3) & ~3;
        new Uint8Array(wasm.memory.buffer).set(convF16Bytes, srcPtr);
        wasm.f16_slice_to_f32(count, srcPtr, dstPtr);
        auxF32.set(new Float32Array(wasm.memory.buffer, dstPtr, count), la.convWOff);
      }
    }

    const auxTmpPath = path.join(BUNDLE_DIR, 'aux_params.bin.tmp');
    fs.writeFileSync(auxTmpPath, new Uint8Array(auxF32.buffer));
    fs.renameSync(auxTmpPath, path.join(BUNDLE_DIR, 'aux_params.bin'));

    // 2. Pack temporary slab0/1 and shard into <= 128 MiB files
    const slab0TmpPath = path.join(BUNDLE_DIR, 'slab0.bin.tmp');
    const slab1TmpPath = path.join(BUNDLE_DIR, 'slab1.bin.tmp');
    const slab0Fd = fs.openSync(slab0TmpPath, 'w');
    const slab1Fd = fs.openSync(slab1TmpPath, 'w');
    let slab0Bytes = 0;
    let slab1Bytes = 0;

    try {
      for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
        console.log(`Repacking Q4 layer ${l + 1}/${QWEN35_NUM_LAYERS} via Rust WASM...`);
        const fd = l < 16 ? slab0Fd : slab1Fd;
        let layerWritten = 0;

        if (!QWEN35_FULL_ATTN_LAYERS.has(l)) {
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_gdn_in_proj_qkv_MatMul`, 8192, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_gdn_in_proj_z_MatMul`, 4096, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_gdn_in_proj_b_MatMul`, 32, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_gdn_in_proj_a_MatMul`, 32, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_gdn_out_proj_MatMul`, QWEN35_HIDDEN, 4096, extFds, wasm);
        } else {
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_attn_q_proj_MatMul`, 8192, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_attn_k_proj_MatMul`, 1024, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_attn_v_proj_MatMul`, 1024, QWEN35_HIDDEN, extFds, wasm);
          layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_attn_o_proj_MatMul`, QWEN35_HIDDEN, 4096, extFds, wasm);
        }

        layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_mlp_gate_proj_MatMul`, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, extFds, wasm);
        layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_mlp_up_proj_MatMul`, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, extFds, wasm);
        layerWritten += repackQ4MatrixToFd(fd, manifestMap, `model_layers_${l}_mlp_down_proj_MatMul`, QWEN35_HIDDEN, QWEN35_INTERMEDIATE, extFds, wasm);

        if (l < 16) slab0Bytes += layerWritten;
        else slab1Bytes += layerWritten;
      }
    } finally {
      fs.closeSync(slab0Fd);
      fs.closeSync(slab1Fd);
    }

    // 3. Pack temporary slab2
    console.log('Repacking LM Head (248,320 x 2,560) into slab2 shards via Rust WASM...');
    const slab2TmpPath = path.join(BUNDLE_DIR, 'slab2.bin.tmp');
    const slab2Fd = fs.openSync(slab2TmpPath, 'w');
    let slab2Bytes = 0;
    try {
      slab2Bytes = repackQ4MatrixToFd(
        slab2Fd,
        manifestMap,
        'lm_head_MatMul',
        QWEN35_VOCAB_SIZE,
        QWEN35_HIDDEN,
        extFds,
        wasm,
      );
    } finally {
      fs.closeSync(slab2Fd);
    }

    const exportActions = new Map([['aux_params.bin', 'repacked']]);
    const slab0Shards = splitFileIntoShards(slab0TmpPath, BUNDLE_DIR, 'slab0', slab0Bytes, exportActions, { forceRewrite: true });
    fs.rmSync(slab0TmpPath, { force: true });
    const slab1Shards = splitFileIntoShards(slab1TmpPath, BUNDLE_DIR, 'slab1', slab1Bytes, exportActions, { forceRewrite: true });
    fs.rmSync(slab1TmpPath, { force: true });
    const slab2Shards = splitFileIntoShards(slab2TmpPath, BUNDLE_DIR, 'slab2', slab2Bytes, exportActions, { forceRewrite: true });
    fs.rmSync(slab2TmpPath, { force: true });

    // 4. Write manifest.json and copy config/tokenizer JSON files
    const manifest = {
      version: 1,
      modelId: MODEL_ID,
      complete: true,
      slab0Bytes,
      slab1Bytes,
      slab2Bytes,
      auxBytes: auxF32.byteLength,
      shardMaxBytes: SHARD_MAX_BYTES,
      shards: {
        slab0: slab0Shards,
        slab1: slab1Shards,
        slab2: slab2Shards,
        aux: [{ file: 'aux_params.bin', bytes: auxF32.byteLength }],
      },
      createdAt: new Date().toISOString(),
    };
    fs.writeFileSync(path.join(BUNDLE_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));
    exportActions.set('manifest.json', 'written');

    for (const fileName of JSON_ASSET_FILES) {
      const srcPath = path.join(HF_TMP_DIR, fileName);
      JSON.parse(fs.readFileSync(srcPath, 'utf8'));
      exportActions.set(fileName, copyFileIfChanged(srcPath, path.join(BUNDLE_DIR, fileName)));
    }

    const expectedFiles = [
      ...JSON_ASSET_FILES,
      'aux_params.bin',
      ...slab0Shards.map((s) => s.file),
      ...slab1Shards.map((s) => s.file),
      ...slab2Shards.map((s) => s.file),
      'manifest.json',
    ];
    cleanUnexpectedBundleFiles(BUNDLE_DIR, expectedFiles);
    return { expectedFiles, exportActions };
  } finally {
    for (const fd of Object.values(extFds)) {
      fs.closeSync(fd);
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  let fromHf = false;
  for (const arg of args) {
    if (arg === '--from-hf') {
      fromHf = true;
    } else if (arg === '--help' || arg === '-h') {
      console.log('Usage: node scripts/bundle-model.mjs [--from-hf]');
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  const { expectedFiles, exportActions } = fromHf
    ? await exportFromHuggingFace()
    : exportFromBenchProfile();

  console.log(`\nModel bundle ready at ${BUNDLE_DIR}:`);
  let totalBytes = 0;
  for (const fileName of expectedFiles) {
    const filePath = path.join(BUNDLE_DIR, fileName);
    const size = fs.statSync(filePath).size;
    totalBytes += size;
    const action = exportActions.get(fileName) ?? 'present';
    console.log(`  - ${fileName.padEnd(24)} ${String(size).padStart(12)} bytes  [${action}]`);
  }
  console.log(`  Total (${expectedFiles.length} files): ${totalBytes} bytes`);

  if (fs.existsSync(DIST_DIR)) {
    const distResults = syncBundleToDist(BUNDLE_DIR, DIST_MODEL_DIR);
    console.log(`\nSynced dist model bundle at ${DIST_MODEL_DIR}:`);
    for (const { fileName, size, action } of distResults) {
      console.log(`  - ${fileName.padEnd(24)} ${String(size).padStart(12)} bytes  [${action}]`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
