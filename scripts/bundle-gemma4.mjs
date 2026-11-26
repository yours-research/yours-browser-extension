import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const MODEL_ID = 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX';
const HF_BASE_URL = `https://huggingface.co/${MODEL_ID}/resolve/main`;
const HF_TMP_DIR = path.join(ROOT_DIR, 'model-bundle', '.hf-tmp-gemma4-qat');
const BUNDLE_DIR = path.join(ROOT_DIR, 'model-bundle', 'gemma4-e4b');
const RUNS_DIR = path.join(ROOT_DIR, 'benchmarks', 'runs');

const SHARD_MAX_BYTES = 128 * 1024 * 1024;
const COPY_CHUNK_BYTES = 16 * 1024 * 1024;

const JSON_ASSET_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'generation_config.json',
  'chat_template.jinja',
];

const HF_DOWNLOAD_FILES = [
  ...JSON_ASSET_FILES,
  'onnx/embed_tokens_q2f16.onnx',
  'onnx/embed_tokens_q2f16.onnx_data',
  'onnx/decoder_model_merged_q2f16.onnx',
  'onnx/decoder_model_merged_q2f16.onnx_data',
  'onnx/decoder_model_merged_q2f16.onnx_data_1',
];

const NUM_LAYERS = 42;
const FULL_ATTN_LAYERS = new Set([5, 11, 17, 23, 29, 35, 41]);
const KV_SHARED_START_LAYER = 24;

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 0x1f) {
    return f ? NaN : s ? -Infinity : Infinity;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

const _f32Scratch = new ArrayBuffer(4);
const _f32View = new Float32Array(_f32Scratch);
const _u32View = new Uint32Array(_f32Scratch);

function f32ToF16Bits(v) {
  _f32View[0] = v;
  const bits = _u32View[0];
  const sign = (bits >>> 16) & 0x8000;
  const exp = (bits >>> 23) & 0xff;
  const frac = bits & 0x007fffff;
  if (exp === 0xff) {
    return sign | 0x7c00 | (frac ? 1 : 0);
  }
  const expF16 = exp - 127 + 15;
  if (expF16 >= 31) return sign | 0x7c00;
  if (expF16 <= 0) {
    if (expF16 < -10) return sign;
    const m = frac | 0x00800000;
    const shift = 14 - expF16;
    const roundBit = 1 << (shift - 1);
    const mant = (m + roundBit - 1 + ((m >>> shift) & 1)) >>> shift;
    return sign | mant;
  }
  const lsb = (frac >>> 13) & 1;
  const rounded = frac + 0x0fff + lsb;
  if (rounded & 0x00800000) {
    const newExp = expF16 + 1;
    if (newExp >= 31) return sign | 0x7c00;
    return sign | (newExp << 10);
  }
  return sign | (expF16 << 10) | (rounded >>> 13);
}

function align16(off) {
  return Math.ceil(off / 16) * 16;
}

function bytesToF16Array(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Array(Math.floor(bytes.byteLength / 2));
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number(f16ToF32(dv.getUint16(i * 2, true)).toFixed(6));
  }
  return out;
}

const utf8Decoder = new TextDecoder();

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
      onField(field, wire, pos, pos + 8);
      pos += 8;
    } else if (wire === 2) {
      const len = readVarint(buf, pos);
      pos = len.pos;
      const sliceEnd = pos + len.val;
      onField(field, wire, pos, sliceEnd);
      pos = sliceEnd;
    } else if (wire === 5) {
      onField(field, wire, pos, pos + 4);
      pos += 4;
    } else {
      throw new Error(`Unsupported protobuf wire type ${wire} at offset ${pos}`);
    }
  }
}

function parseTensorProto(buf, start, end) {
  const tensor = {
    name: '',
    dims: [],
    dataType: 0,
    rawData: null,
    externalLocation: null,
    externalOffset: 0,
    externalLength: 0,
    floatData: [],
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
    } else if (field === 4 && wire === 2 && b !== undefined) {
      const dv = new DataView(buf.buffer, buf.byteOffset + a, b - a);
      for (let i = 0; i + 4 <= b - a; i += 4) tensor.floatData.push(dv.getFloat32(i, true));
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

function parseAttributeProto(buf, start, end) {
  let name = '';
  let value = null;
  parseProtoFields(buf, start, end, (field, wire, a, b) => {
    if (field === 1 && wire === 2 && b !== undefined) name = utf8Decoder.decode(buf.subarray(a, b));
    else if (field === 2 && wire === 5) value = new DataView(buf.buffer, buf.byteOffset + a, 4).getFloat32(0, true);
    else if (field === 3 && wire === 0) value = a;
    else if (field === 4 && wire === 2 && b !== undefined) value = utf8Decoder.decode(buf.subarray(a, b));
  });
  return { name, value };
}

function parseOnnxGraph(onnxBytes) {
  const tensors = new Map();
  const nodes = [];
  parseProtoFields(onnxBytes, 0, onnxBytes.length, (field, wire, start, end) => {
    if (field === 7 && wire === 2 && end !== undefined) {
      parseProtoFields(onnxBytes, start, end, (gf, gw, gStart, gEnd) => {
        if (gf === 5 && gw === 2 && gEnd !== undefined) {
          const tensor = parseTensorProto(onnxBytes, gStart, gEnd);
          if (tensor.name) tensors.set(tensor.name, tensor);
        } else if (gf === 1 && gw === 2 && gEnd !== undefined) {
          const node = { name: '', opType: '', inputs: [], outputs: [], attrs: {} };
          parseProtoFields(onnxBytes, gStart, gEnd, (nf, nw, na, nb) => {
            if (nf === 1 && nw === 2 && nb !== undefined) node.inputs.push(utf8Decoder.decode(onnxBytes.subarray(na, nb)));
            else if (nf === 2 && nw === 2 && nb !== undefined) node.outputs.push(utf8Decoder.decode(onnxBytes.subarray(na, nb)));
            else if (nf === 3 && nw === 2 && nb !== undefined) node.name = utf8Decoder.decode(onnxBytes.subarray(na, nb));
            else if (nf === 4 && nw === 2 && nb !== undefined) node.opType = utf8Decoder.decode(onnxBytes.subarray(na, nb));
            else if (nf === 5 && nw === 2 && nb !== undefined) {
              const attr = parseAttributeProto(onnxBytes, na, nb);
              if (attr.name) node.attrs[attr.name] = attr.value;
            }
          });
          nodes.push(node);
        }
      });
    }
  });
  return { tensors, nodes };
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

function readTensorBytes(tensor, extFds, subOffset = 0, subLength) {
  if (tensor.rawData) {
    const start = subOffset;
    const end = subLength !== undefined ? start + subLength : tensor.rawData.length;
    return tensor.rawData.slice(start, end);
  }
  if (!tensor.externalLocation) {
    throw new Error(`Tensor ${tensor.name} has neither rawData nor externalLocation`);
  }
  const fd = extFds[tensor.externalLocation];
  if (fd === undefined) {
    throw new Error(`Missing external ONNX file ${tensor.externalLocation} for ${tensor.name}`);
  }
  const start = tensor.externalOffset + subOffset;
  const len = subLength !== undefined ? subLength : tensor.externalLength - subOffset;
  return readExactFromFd(fd, start, len);
}

function getTensorByteLength(tensor) {
  if (tensor.rawData) return tensor.rawData.byteLength;
  if (tensor.externalLength > 0) return tensor.externalLength;
  const bpe = tensor.dataType === 10 ? 2 : tensor.dataType === 2 ? 1 : tensor.dataType === 1 ? 4 : tensor.dataType === 7 ? 8 : 1;
  return tensor.dims.reduce((a, b) => a * b, 1) * bpe;
}

function writeAllSync(fd, buf) {
  let offset = 0;
  while (offset < buf.byteLength) {
    offset += fs.writeSync(fd, buf, offset, buf.byteLength - offset);
  }
}

async function downloadFile(relPath) {
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
        console.log(`  HF cache hit: ${relPath} (${(actualSize / 1e6).toFixed(1)} MB)`);
        return destPath;
      }
    } catch {
      // Re-download
    }
  }

  console.log(`  Downloading ${relPath} from ${MODEL_ID} ...`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok || !res.body) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  }
  const cl = Number(res.headers.get('content-length') || 0);
  fs.rmSync(partPath, { force: true });

  let downloaded = 0;
  let nextLogAt = 250 * 1024 * 1024;
  const bodyStream = Readable.fromWeb(res.body);
  bodyStream.on('data', (chunk) => {
    downloaded += chunk.length;
    if (downloaded >= nextLogAt) {
      const pct = cl > 0 ? ` (${((downloaded / cl) * 100).toFixed(1)}%)` : '';
      console.log(`    ${relPath}: ${(downloaded / 1e6).toFixed(1)} MB${pct}`);
      nextLogAt += 250 * 1024 * 1024;
    }
  });

  await pipeline(bodyStream, fs.createWriteStream(partPath));
  const actualSize = fs.statSync(partPath).size;
  if (cl > 0 && actualSize !== cl) {
    throw new Error(`Truncated download for ${relPath}: got ${actualSize}, expected ${cl}`);
  }

  fs.rmSync(destPath, { force: true });
  fs.renameSync(partPath, destPath);
  fs.writeFileSync(
    metaPath,
    JSON.stringify({ url, contentLength: actualSize, complete: true, downloadedAt: new Date().toISOString() }, null, 2),
  );
  console.log(`  Downloaded ${relPath} (${(actualSize / 1e6).toFixed(1)} MB)`);
  return destPath;
}

function openDownloadsContext() {
  const embOnnxPath = path.join(HF_TMP_DIR, 'onnx', 'embed_tokens_q2f16.onnx');
  const decOnnxPath = path.join(HF_TMP_DIR, 'onnx', 'decoder_model_merged_q2f16.onnx');
  const embGraph = parseOnnxGraph(new Uint8Array(fs.readFileSync(embOnnxPath)));
  const decGraph = parseOnnxGraph(new Uint8Array(fs.readFileSync(decOnnxPath)));

  const embFds = {
    'embed_tokens_q2f16.onnx_data': fs.openSync(path.join(HF_TMP_DIR, 'onnx', 'embed_tokens_q2f16.onnx_data'), 'r'),
  };
  const decFds = {
    'decoder_model_merged_q2f16.onnx_data': fs.openSync(path.join(HF_TMP_DIR, 'onnx', 'decoder_model_merged_q2f16.onnx_data'), 'r'),
    'decoder_model_merged_q2f16.onnx_data_1': fs.openSync(path.join(HF_TMP_DIR, 'onnx', 'decoder_model_merged_q2f16.onnx_data_1'), 'r'),
  };
  return {
    embGraph,
    decGraph,
    embFds,
    decFds,
    close() {
      Object.values(embFds).forEach((fd) => fs.closeSync(fd));
      Object.values(decFds).forEach((fd) => fs.closeSync(fd));
    },
  };
}

function inspectMath() {
  console.log('\n=== [g00-gemma4-math-audit] Exact Weight & Math Audit ===');
  const ctx = openDownloadsContext();
  try {
    const { embGraph, decGraph, embFds, decFds } = ctx;

    const l0InputNorm = decGraph.tensors.get('model.layers.0.input_layernorm.weight');
    const l0InputNormVals = bytesToF16Array(readTensorBytes(l0InputNorm, decFds, 0, 16));
    const l0PostAttnNorm = decGraph.tensors.get('model.layers.0.post_attention_layernorm.weight');
    const l0PostAttnVals = bytesToF16Array(readTensorBytes(l0PostAttnNorm, decFds, 0, 16));
    const l0QNorm = decGraph.tensors.get('model.layers.0.attn.q_norm.layernorm.weight');
    const l0QNormVals = bytesToF16Array(readTensorBytes(l0QNorm, decFds, 0, 16));

    const scalars = {};
    for (const l of [0, 5, 24, 41]) {
      const t = decGraph.tensors.get(`model.layers.${l}.layer_scalar`);
      scalars[`layer_${l}`] = bytesToF16Array(readTensorBytes(t, decFds))[0];
    }

    const cosLocal = decGraph.tensors.get('cos_cache_local');
    const sinLocal = decGraph.tensors.get('sin_cache_local');
    const cosGlobal = decGraph.tensors.get('cos_cache_global');
    const sinGlobal = decGraph.tensors.get('sin_cache_global');

    const sampleCacheRows = (t, strideElems) => {
      const out = {};
      for (const pos of [0, 1, 100]) {
        const row = bytesToF16Array(readTensorBytes(t, decFds, pos * strideElems * 2, strideElems * 2));
        out[`pos_${pos}`] = {
          first8: row.slice(0, 8),
          last8: row.slice(-8),
        };
      }
      return out;
    };

    const cosLocalSample = sampleCacheRows(cosLocal, 128);
    const cosGlobalSample = sampleCacheRows(cosGlobal, 256);

    // Check GatherBlockQuantized & MatMulNBits attributes
    const gatherNodes = embGraph.nodes.filter((n) => n.opType === 'GatherBlockQuantized');
    const matmulNBitsSample = {};
    for (const key of [
      '/model/layers.0/attn/q_proj/MatMul',
      '/model/layers.0/mlp/gate_up_proj/MatMul',
      '/model/layers.0/per_layer/per_layer_input_gate/MatMul',
      '/model/layers.0/per_layer/per_layer_projection/MatMul',
      '/lm_head/MatMul',
    ]) {
      const node = decGraph.nodes.find((n) => n.name === key);
      if (node) matmulNBitsSample[key] = { inputs: node.inputs, attrs: node.attrs };
    }

    const tEmbQ = embGraph.tensors.get('model.embed_tokens.weight_q');
    const tEmbS = embGraph.tensors.get('model.embed_tokens.scales');
    const tPleQ = embGraph.tensors.get('model.embed_tokens_per_layer.weight_q');
    const tPleS = embGraph.tensors.get('model.embed_tokens_per_layer.scales');
    const tLmQ = decGraph.tensors.get('lm_head.MatMul.qweight');
    const tLmS = decGraph.tensors.get('lm_head.MatMul.scales_row');
    const tQ0 = decGraph.tensors.get('model.layers.0.attn.q_proj.MatMul.qweight');
    const tQ0S = decGraph.tensors.get('model.layers.0.attn.q_proj.MatMul.scales_row');

    const sampleLmBytes = Array.from(readTensorBytes(tLmQ, decFds, 0, 16));
    const sampleLmScales = bytesToF16Array(readTensorBytes(tLmS, decFds, 0, 16));
    const sampleQ0Bytes = Array.from(readTensorBytes(tQ0, decFds, 0, 16));
    const sampleQ0Scales = bytesToF16Array(readTensorBytes(tQ0S, decFds, 0, 16));
    const sampleEmbBytes = Array.from(readTensorBytes(tEmbQ, embFds, 0, 16));
    const sampleEmbScales = bytesToF16Array(readTensorBytes(tEmbS, embFds, 0, 20));

    // Exact byte breakdown across every category
    const breakdown = {
      attentionQ4Bytes: 0,
      mlpQ4Bytes: 0,
      pleLinearQ8Bytes: 0,
      pleProjFp16Bytes: 0,
      lmHeadQ2Bytes: 0,
      embedTokensQ2Bytes: getTensorByteLength(tEmbQ) + getTensorByteLength(tEmbS),
      pleTokensQ2Bytes: getTensorByteLength(tPleQ) + getTensorByteLength(tPleS),
      normsAndScalarsBytes: 0,
      ropeCachesBytes:
        getTensorByteLength(cosLocal) +
        getTensorByteLength(sinLocal) +
        getTensorByteLength(cosGlobal) +
        getTensorByteLength(sinGlobal),
      otherConstantsBytes: 0,
    };

    for (const t of decGraph.tensors.values()) {
      const sz = getTensorByteLength(t);
      if (t.name.startsWith('lm_head.')) {
        breakdown.lmHeadQ2Bytes += sz;
      } else if (t.name.startsWith('model.per_layer_projection.MatMul')) {
        breakdown.pleProjFp16Bytes += sz;
      } else if (t.name.includes('.per_layer.')) {
        breakdown.pleLinearQ8Bytes += sz;
      } else if (t.name.includes('.attn.') && (t.name.endsWith('.qweight') || t.name.endsWith('.scales_row'))) {
        breakdown.attentionQ4Bytes += sz;
      } else if (t.name.includes('.mlp.') && (t.name.endsWith('.qweight') || t.name.endsWith('.scales_row'))) {
        breakdown.mlpQ4Bytes += sz;
      } else if (t.name.includes('layernorm') || t.name.includes('layer_scalar') || t.name.includes('per_layer_projection_norm')) {
        breakdown.normsAndScalarsBytes += sz;
      } else if (!t.name.startsWith('cos_cache_') && !t.name.startsWith('sin_cache_')) {
        breakdown.otherConstantsBytes += sz;
      }
    }

    const gpuWeightsTotalBytes =
      breakdown.attentionQ4Bytes +
      breakdown.mlpQ4Bytes +
      breakdown.pleLinearQ8Bytes +
      breakdown.pleProjFp16Bytes +
      breakdown.lmHeadQ2Bytes +
      breakdown.normsAndScalarsBytes;
    const cpuOpfsEmbedBytes = breakdown.embedTokensQ2Bytes + breakdown.pleTokensQ2Bytes;

    const audit = {
      runId: 'g00-gemma4-math-audit',
      modelId: MODEL_ID,
      recordedAt: new Date().toISOString(),
      normsSample: {
        l0_input_layernorm_f16: l0InputNormVals,
        l0_post_attn_layernorm_f16: l0PostAttnVals,
        l0_q_norm_f16: l0QNormVals,
        note: 'Inspect whether weights hover near ~1.0 (pre-shifted 1+w) or ~0.0 (needs +1).',
      },
      layerScalarsF16: scalars,
      ropeCaches: {
        cos_cache_local_dims: cosLocal.dims,
        cos_cache_global_dims: cosGlobal.dims,
        localSample: cosLocalSample,
        globalSample: cosGlobalSample,
      },
      quantizationNodes: {
        gatherBlockQuantized: gatherNodes.map((n) => ({ name: n.name, inputs: n.inputs, attrs: n.attrs })),
        matMulNBitsSample: matmulNBitsSample,
      },
      tensorByteSamples: {
        embedTokensQ2: { qweightDims: tEmbQ.dims, scalesDims: tEmbS.dims, qBytes0: sampleEmbBytes, scales0: sampleEmbScales },
        pleTokensQ2: { qweightDims: tPleQ.dims, scalesDims: tPleS.dims },
        lmHeadQ2: { qweightDims: tLmQ.dims, scalesDims: tLmS.dims, qBytes0: sampleLmBytes, scales0: sampleLmScales },
        layer0QProjQ4: { qweightDims: tQ0.dims, scalesDims: tQ0S.dims, qBytes0: sampleQ0Bytes, scales0: sampleQ0Scales },
      },
      footprintBreakdownBytes: {
        ...breakdown,
        gpuWeightsWithoutRoPECachesBytes: gpuWeightsTotalBytes,
        gpuWeightsGiB: Number((gpuWeightsTotalBytes / 1024 ** 3).toFixed(3)),
        cpuOpfsEmbeddingsBytes: cpuOpfsEmbedBytes,
        cpuOpfsEmbeddingsGiB: Number((cpuOpfsEmbedBytes / 1024 ** 3).toFixed(3)),
        combinedTextTotalGiB: Number(((gpuWeightsTotalBytes + cpuOpfsEmbedBytes) / 1024 ** 3).toFixed(3)),
      },
    };

    fs.mkdirSync(RUNS_DIR, { recursive: true });
    const outPath = path.join(RUNS_DIR, 'g00-gemma4-math-audit.json');
    fs.writeFileSync(outPath, JSON.stringify(audit, null, 2));

    console.log('  l0 input_layernorm[0..7]:', l0InputNormVals);
    console.log('  layer_scalars:', scalars);
    console.log('  cos_global pos=1 last8:', cosGlobalSample.pos_1.last8);
    console.log('  GatherBlockQuantized:', JSON.stringify(audit.quantizationNodes.gatherBlockQuantized));
    console.log('  lm_head attrs:', JSON.stringify(matmulNBitsSample['/lm_head/MatMul']));
    console.log(
      `  GPU weights (slabs0/1/2 + aux): ${audit.footprintBreakdownBytes.gpuWeightsGiB} GiB | CPU/OPFS Q2 embeddings: ${audit.footprintBreakdownBytes.cpuOpfsEmbeddingsGiB} GiB | Total: ${audit.footprintBreakdownBytes.combinedTextTotalGiB} GiB`,
    );
    console.log(`Saved ${outPath}`);
    return audit;
  } finally {
    ctx.close();
  }
}

function splitBufferOrFileIntoShards(srcPath, destDir, prefix, expectedBytes) {
  const actual = fs.statSync(srcPath).size;
  assert(actual === expectedBytes, `Size mismatch for ${srcPath}: ${actual} != ${expectedBytes}`);
  const shards = [];
  const srcFd = fs.openSync(srcPath, 'r');
  const ioBuf = Buffer.allocUnsafe(COPY_CHUNK_BYTES);
  try {
    let off = 0;
    let idx = 0;
    while (off < expectedBytes) {
      const partBytes = Math.min(SHARD_MAX_BYTES, expectedBytes - off);
      const fileName = `${prefix}.part${String(idx).padStart(2, '0')}.bin`;
      const destPath = path.join(destDir, fileName);
      const tmpPath = `${destPath}.tmp`;
      fs.rmSync(tmpPath, { force: true });
      const dstFd = fs.openSync(tmpPath, 'w');
      try {
        let rem = partBytes;
        let cur = off;
        while (rem > 0) {
          const n = fs.readSync(srcFd, ioBuf, 0, Math.min(ioBuf.byteLength, rem), cur);
          assert(n > 0, `Unexpected EOF reading ${srcPath}`);
          writeAllSync(dstFd, ioBuf.subarray(0, n));
          cur += n;
          rem -= n;
        }
      } finally {
        fs.closeSync(dstFd);
      }
      fs.rmSync(destPath, { force: true });
      fs.renameSync(tmpPath, destPath);
      shards.push({ file: fileName, bytes: partBytes });
      off += partBytes;
      idx += 1;
    }
  } finally {
    fs.closeSync(srcFd);
  }
  return shards;
}

function streamCopyTensorToFd(fd, tensor, extFds) {
  const totalBytes = getTensorByteLength(tensor);
  assert(totalBytes > 0, `Empty tensor ${tensor.name}`);
  if (tensor.rawData) {
    writeAllSync(fd, tensor.rawData);
    return totalBytes;
  }
  const srcFd = extFds[tensor.externalLocation];
  assert(srcFd !== undefined, `Missing ext fd ${tensor.externalLocation}`);
  const buf = Buffer.allocUnsafe(Math.min(COPY_CHUNK_BYTES, totalBytes));
  let rem = totalBytes;
  let pos = tensor.externalOffset;
  while (rem > 0) {
    const n = fs.readSync(srcFd, buf, 0, Math.min(buf.byteLength, rem), pos);
    assert(n > 0, `Unexpected EOF on ${tensor.name}`);
    writeAllSync(fd, buf.subarray(0, n));
    pos += n;
    rem -= n;
  }
  return totalBytes;
}

function writeAlignedPairToFd(fd, currentByteOffset, qTensorOrList, sTensorOrList, extFds, label) {
  const qList = Array.isArray(qTensorOrList) ? qTensorOrList : [qTensorOrList];
  const sList = Array.isArray(sTensorOrList) ? sTensorOrList : [sTensorOrList];

  const qOffset = align16(currentByteOffset);
  if (qOffset > currentByteOffset) {
    writeAllSync(fd, Buffer.alloc(qOffset - currentByteOffset));
  }
  const qSubOffsets = [];
  let qBytes = 0;
  for (const qt of qList) {
    const startOff = qOffset + qBytes;
    const b = streamCopyTensorToFd(fd, qt, extFds);
    qSubOffsets.push({ qOffset: startOff, qBytes: b });
    qBytes += b;
  }

  const sRawEnd = qOffset + qBytes;
  const sOffset = align16(sRawEnd);
  if (sOffset > sRawEnd) {
    writeAllSync(fd, Buffer.alloc(sOffset - sRawEnd));
  }
  const sSubOffsets = [];
  let sBytes = 0;
  for (const st of sList) {
    const startOff = sOffset + sBytes;
    const b = streamCopyTensorToFd(fd, st, extFds);
    sSubOffsets.push({ sOffset: startOff, sBytes: b });
    sBytes += b;
  }

  const totalEnd = sOffset + sBytes;
  const nextOffset = align16(totalEnd);
  if (nextOffset > totalEnd) {
    writeAllSync(fd, Buffer.alloc(nextOffset - totalEnd));
  }
  return {
    name: label,
    qOffset,
    qBytes,
    sOffset,
    sBytes,
    qSubOffsets,
    sSubOffsets,
    nextOffset,
  };
}

function quantizePleProjFp16ToQ8(rawFp16Bytes, K = 2560, N = 10752) {
  const srcDv = new DataView(rawFp16Bytes.buffer, rawFp16Bytes.byteOffset, rawFp16Bytes.byteLength);
  const f32Mat = new Float32Array(K * N);
  const colMax = new Float32Array(N);

  for (let k = 0; k < K; k += 1) {
    const rowBase = k * N;
    for (let n = 0; n < N; n += 1) {
      const val = f16ToF32(srcDv.getUint16((rowBase + n) * 2, true));
      f32Mat[rowBase + n] = val;
      const a = Math.abs(val);
      if (a > colMax[n]) colMax[n] = a;
    }
  }

  const qBuf = Buffer.allocUnsafe(N * K);
  const sBuf = Buffer.allocUnsafe(N * 2);
  const sDv = new DataView(sBuf.buffer, sBuf.byteOffset, sBuf.byteLength);
  const scales = new Float32Array(N);
  for (let n = 0; n < N; n += 1) {
    const sc = Math.max(colMax[n] / 127.0, 1e-7);
    const scF16 = f32ToF16Bits(sc);
    sDv.setUint16(n * 2, scF16, true);
    scales[n] = f16ToF32(scF16);
  }

  for (let n = 0; n < N; n += 1) {
    const invScale = 1.0 / Math.max(scales[n], 1e-8);
    const dstBase = n * K;
    for (let k = 0; k < K; k += 1) {
      const q = Math.max(-127, Math.min(127, Math.round(f32Mat[k * N + n] * invScale)));
      qBuf[dstBase + k] = (q + 128) & 0xff;
    }
  }
  return { qBuf, sBuf };
}

function repackGemma4Bundle() {
  console.log('\n=== Repacking Gemma 4 E4B QAT v4 -> model-bundle/gemma4-e4b/ ===');
  const ctx = openDownloadsContext();
  fs.mkdirSync(BUNDLE_DIR, { recursive: true });

  try {
    const { embGraph, decGraph, embFds, decFds } = ctx;

    // 1. Copy config & tokenizer metadata
    for (const fileName of JSON_ASSET_FILES) {
      const src = path.join(HF_TMP_DIR, fileName);
      const dst = path.join(BUNDLE_DIR, fileName);
      if (fs.existsSync(src)) fs.copyFileSync(src, dst);
    }

    // 2. Pack aux_params.bin (16-byte aligned + companion Q8 per_layer_projection)
    const auxTmpPath = path.join(BUNDLE_DIR, 'aux_params.bin.tmp');
    const auxFd = fs.openSync(auxTmpPath, 'w');
    let auxOffset = 0;
    const writeAuxTensor = (tensorName) => {
      const t = decGraph.tensors.get(tensorName);
      assert(t, `Missing decoder tensor ${tensorName}`);
      const off = align16(auxOffset);
      if (off > auxOffset) {
        writeAllSync(auxFd, Buffer.alloc(off - auxOffset));
      }
      const bytes = streamCopyTensorToFd(auxFd, t, decFds);
      const next = align16(off + bytes);
      if (next > off + bytes) {
        writeAllSync(auxFd, Buffer.alloc(next - (off + bytes)));
      }
      auxOffset = next;
      return { offset: off, bytes, dims: t.dims, dataType: t.dataType };
    };

    const auxLayout = {
      per_layer_projection_norm: writeAuxTensor('model.per_layer_projection_norm.weight'),
      per_layer_projection_weight: writeAuxTensor('model.per_layer_projection.MatMul.weight'),
      final_norm: writeAuxTensor('model.layers.42.final_norm_layernorm.weight'),
      layers: [],
    };

    // Quantize [2560, 10752] FP16 per_layer_projection to [10752, 2560] symmetric Q8 ( Experiment 3A )
    const pleProjTensor = decGraph.tensors.get('model.per_layer_projection.MatMul.weight');
    const pleProjQ8 = quantizePleProjFp16ToQ8(readTensorBytes(pleProjTensor, decFds), 2560, 10752);
    const pleProjQ8QOff = align16(auxOffset);
    if (pleProjQ8QOff > auxOffset) writeAllSync(auxFd, Buffer.alloc(pleProjQ8QOff - auxOffset));
    writeAllSync(auxFd, pleProjQ8.qBuf);
    const pleProjQ8SOff = align16(pleProjQ8QOff + pleProjQ8.qBuf.byteLength);
    if (pleProjQ8SOff > pleProjQ8QOff + pleProjQ8.qBuf.byteLength) {
      writeAllSync(auxFd, Buffer.alloc(pleProjQ8SOff - (pleProjQ8QOff + pleProjQ8.qBuf.byteLength)));
    }
    writeAllSync(auxFd, pleProjQ8.sBuf);
    const pleProjQ8End = align16(pleProjQ8SOff + pleProjQ8.sBuf.byteLength);
    if (pleProjQ8End > pleProjQ8SOff + pleProjQ8.sBuf.byteLength) {
      writeAllSync(auxFd, Buffer.alloc(pleProjQ8End - (pleProjQ8SOff + pleProjQ8.sBuf.byteLength)));
    }
    auxOffset = pleProjQ8End;
    auxLayout.per_layer_projection_q8 = {
      qOffset: pleProjQ8QOff,
      qBytes: pleProjQ8.qBuf.byteLength,
      sOffset: pleProjQ8SOff,
      sBytes: pleProjQ8.sBuf.byteLength,
      rows: 10752,
      cols: 2560,
    };

    for (let l = 0; l < NUM_LAYERS; l += 1) {
      auxLayout.layers.push({
        layer: l,
        isFullAttn: FULL_ATTN_LAYERS.has(l),
        isKvShared: l >= KV_SHARED_START_LAYER,
        input_layernorm: writeAuxTensor(`model.layers.${l}.input_layernorm.weight`),
        q_norm: writeAuxTensor(`model.layers.${l}.attn.q_norm.layernorm.weight`),
        ...(l < KV_SHARED_START_LAYER
          ? { k_norm: writeAuxTensor(`model.layers.${l}.attn.k_norm.layernorm.weight`) }
          : {}),
        post_attention_layernorm: writeAuxTensor(`model.layers.${l}.post_attention_layernorm.weight`),
        pre_feedforward_layernorm: writeAuxTensor(`model.layers.${l}.pre_feedforward_layernorm.weight`),
        post_feedforward_layernorm: writeAuxTensor(`model.layers.${l}.post_feedforward_layernorm.weight`),
        post_per_layer_input_norm: writeAuxTensor(`model.layers.${l}.post_per_layer_input_norm.weight`),
        layer_scalar: writeAuxTensor(`model.layers.${l}.layer_scalar`),
      });
    }
    fs.closeSync(auxFd);
    fs.rmSync(path.join(BUNDLE_DIR, 'aux_params.bin'), { force: true });
    fs.renameSync(auxTmpPath, path.join(BUNDLE_DIR, 'aux_params.bin'));

    // 3. Pack slab0 (layers 0..20) & slab1 (layers 21..41) with 16-byte alignment & fused QKV for donors
    const slab0Tmp = path.join(BUNDLE_DIR, 'slab0.bin.tmp');
    const slab1Tmp = path.join(BUNDLE_DIR, 'slab1.bin.tmp');
    const slab0Fd = fs.openSync(slab0Tmp, 'w');
    const slab1Fd = fs.openSync(slab1Tmp, 'w');
    let slab0Off = 0;
    let slab1Off = 0;
    const layerOffsets = [];

    for (let l = 0; l < NUM_LAYERS; l += 1) {
      const slabIdx = l < 21 ? 0 : 1;
      const fd = slabIdx === 0 ? slab0Fd : slab1Fd;
      let cur = slabIdx === 0 ? slab0Off : slab1Off;

      const addPair = (tensorBase, label) => {
        const q = decGraph.tensors.get(`${tensorBase}.qweight`);
        const s = decGraph.tensors.get(`${tensorBase}.scales_row`);
        assert(q && s, `Missing Qweight/scales_row for ${tensorBase}`);
        const entry = writeAlignedPairToFd(fd, cur, q, s, decFds, label);
        cur = entry.nextOffset;
        return { qOffset: entry.qOffset, qBytes: entry.qBytes, sOffset: entry.sOffset, sBytes: entry.sBytes, qDims: q.dims };
      };

      const isKvShared = l >= KV_SHARED_START_LAYER;
      let attnProjEntries;
      if (!isKvShared) {
        const qp = `model.layers.${l}.attn.q_proj.MatMul`;
        const kp = `model.layers.${l}.attn.k_proj.MatMul`;
        const vp = `model.layers.${l}.attn.v_proj.MatMul`;
        const qT = decGraph.tensors.get(`${qp}.qweight`);
        const kT = decGraph.tensors.get(`${kp}.qweight`);
        const vT = decGraph.tensors.get(`${vp}.qweight`);
        const qS = decGraph.tensors.get(`${qp}.scales_row`);
        const kS = decGraph.tensors.get(`${kp}.scales_row`);
        const vS = decGraph.tensors.get(`${vp}.scales_row`);
        assert(qT && kT && vT && qS && kS && vS, `Missing Q/K/V for layer ${l}`);
        const fused = writeAlignedPairToFd(fd, cur, [qT, kT, vT], [qS, kS, vS], decFds, 'qkv_proj');
        cur = fused.nextOffset;
        attnProjEntries = {
          qkv_proj: {
            qOffset: fused.qOffset,
            qBytes: fused.qBytes,
            sOffset: fused.sOffset,
            sBytes: fused.sBytes,
            qDims: [qT.dims[0] + kT.dims[0] + vT.dims[0], qT.dims[1], qT.dims[2]],
          },
          q_proj: {
            qOffset: fused.qSubOffsets[0].qOffset,
            qBytes: fused.qSubOffsets[0].qBytes,
            sOffset: fused.sSubOffsets[0].sOffset,
            sBytes: fused.sSubOffsets[0].sBytes,
            qDims: qT.dims,
          },
          k_proj: {
            qOffset: fused.qSubOffsets[1].qOffset,
            qBytes: fused.qSubOffsets[1].qBytes,
            sOffset: fused.sSubOffsets[1].sOffset,
            sBytes: fused.sSubOffsets[1].sBytes,
            qDims: kT.dims,
          },
          v_proj: {
            qOffset: fused.qSubOffsets[2].qOffset,
            qBytes: fused.qSubOffsets[2].qBytes,
            sOffset: fused.sSubOffsets[2].sOffset,
            sBytes: fused.sSubOffsets[2].sBytes,
            qDims: vT.dims,
          },
        };
      } else {
        attnProjEntries = {
          q_proj: addPair(`model.layers.${l}.attn.q_proj.MatMul`, 'q_proj'),
        };
      }

      const info = {
        layer: l,
        slab: slabIdx,
        isFullAttn: FULL_ATTN_LAYERS.has(l),
        isKvShared,
        ...attnProjEntries,
        o_proj: addPair(`model.layers.${l}.attn.o_proj.MatMul`, 'o_proj'),
        gate_up_proj: addPair(`model.layers.${l}.mlp.gate_up_proj.MatMul`, 'gate_up_proj'),
        down_proj: addPair(`model.layers.${l}.mlp.down_proj.MatMul`, 'down_proj'),
        ple_gate: addPair(`model.layers.${l}.per_layer.per_layer_input_gate.MatMul`, 'ple_gate'),
        ple_proj: addPair(`model.layers.${l}.per_layer.per_layer_projection.MatMul`, 'ple_proj'),
      };

      layerOffsets.push(info);
      if (slabIdx === 0) slab0Off = cur;
      else slab1Off = cur;
    }
    fs.closeSync(slab0Fd);
    fs.closeSync(slab1Fd);

    const slab0Shards = splitBufferOrFileIntoShards(slab0Tmp, BUNDLE_DIR, 'slab0', slab0Off);
    const slab1Shards = splitBufferOrFileIntoShards(slab1Tmp, BUNDLE_DIR, 'slab1', slab1Off);
    fs.rmSync(slab0Tmp, { force: true });
    fs.rmSync(slab1Tmp, { force: true });

    // 4. Pack slab2 (lm_head Q2-QAT + scales_row, 16-byte aligned)
    const slab2Tmp = path.join(BUNDLE_DIR, 'slab2.bin.tmp');
    const slab2Fd = fs.openSync(slab2Tmp, 'w');
    const lmEntry = writeAlignedPairToFd(
      slab2Fd,
      0,
      decGraph.tensors.get('lm_head.MatMul.qweight'),
      decGraph.tensors.get('lm_head.MatMul.scales_row'),
      decFds,
      'lm_head',
    );
    fs.closeSync(slab2Fd);
    const slab2Off = lmEntry.nextOffset;
    const slab2Shards = splitBufferOrFileIntoShards(slab2Tmp, BUNDLE_DIR, 'slab2', slab2Off);
    fs.rmSync(slab2Tmp, { force: true });

    // 5. Pack CPU/OPFS embedding binaries + 256 interleaved PLE shards + GPU 32K hot PLE cache
    const packEmbedBin = (outName, qTensorName, sTensorName) => {
      const tmp = path.join(BUNDLE_DIR, `${outName}.tmp`);
      const fd = fs.openSync(tmp, 'w');
      const info = writeAlignedPairToFd(
        fd,
        0,
        embGraph.tensors.get(qTensorName),
        embGraph.tensors.get(sTensorName),
        embFds,
        outName,
      );
      fs.closeSync(fd);
      fs.rmSync(path.join(BUNDLE_DIR, outName), { force: true });
      fs.renameSync(tmp, path.join(BUNDLE_DIR, outName));
      return { file: outName, bytes: info.nextOffset, qOffset: info.qOffset, qBytes: info.qBytes, sOffset: info.sOffset, sBytes: info.sBytes };
    };

    const embedTokensInfo = packEmbedBin(
      'embed_tokens_q2.bin',
      'model.embed_tokens.weight_q',
      'model.embed_tokens.scales',
    );
    const pleTokensInfo = packEmbedBin(
      'ple_tokens_q2.bin',
      'model.embed_tokens_per_layer.weight_q',
      'model.embed_tokens_per_layer.scales',
    );

    // Emit 256 extension-friendly interleaved PLE shards (`ple_shards/ple_000.bin`..`ple_255.bin`) + 32K GPU Hot Cache
    const PLE_Q_BYTES = 2688;
    const PLE_S_BYTES = 84;
    const PLE_ROW_BYTES = PLE_Q_BYTES + PLE_S_BYTES; // 2772 bytes
    const TOKENS_PER_SHARD = 1024;
    const SHARD_COUNT = 256; // 256 * 1024 = 262,144
    const pleShardsDir = path.join(BUNDLE_DIR, 'ple_shards');
    fs.mkdirSync(pleShardsDir, { recursive: true });

    const pleFd = fs.openSync(path.join(BUNDLE_DIR, 'ple_tokens_q2.bin'), 'r');
    const hotMap = new Uint32Array(262144).fill(0xffffffff);
    const HOT_COUNT = 32768;
    const pleHotData = Buffer.allocUnsafe(HOT_COUNT * PLE_ROW_BYTES);
    let hotSlot = 0;

    // Assign hot slots: [0..30719] (special tokens + top SentencePiece tokens) and [236544..238591] (ASCII/digits/punctuation)
    for (let id = 0; id < 30720; id += 1) {
      hotMap[id] = hotSlot++;
    }
    for (let id = 236544; id < 238592; id += 1) {
      hotMap[id] = hotSlot++;
    }
    assert(hotSlot === HOT_COUNT, `Expected ${HOT_COUNT} hot slots, got ${hotSlot}`);

    const shardQBuf = Buffer.allocUnsafe(TOKENS_PER_SHARD * PLE_Q_BYTES);
    const shardSBuf = Buffer.allocUnsafe(TOKENS_PER_SHARD * PLE_S_BYTES);
    const shardOut = Buffer.allocUnsafe(TOKENS_PER_SHARD * PLE_ROW_BYTES);
    try {
      for (let s = 0; s < SHARD_COUNT; s += 1) {
        const baseTok = s * TOKENS_PER_SHARD;
        fs.readSync(pleFd, shardQBuf, 0, shardQBuf.byteLength, pleTokensInfo.qOffset + baseTok * PLE_Q_BYTES);
        fs.readSync(pleFd, shardSBuf, 0, shardSBuf.byteLength, pleTokensInfo.sOffset + baseTok * PLE_S_BYTES);
        for (let i = 0; i < TOKENS_PER_SHARD; i += 1) {
          const dstOff = i * PLE_ROW_BYTES;
          shardQBuf.copy(shardOut, dstOff, i * PLE_Q_BYTES, (i + 1) * PLE_Q_BYTES);
          shardSBuf.copy(shardOut, dstOff + PLE_Q_BYTES, i * PLE_S_BYTES, (i + 1) * PLE_S_BYTES);
          const hs = hotMap[baseTok + i];
          if (hs !== 0xffffffff) {
            shardOut.copy(pleHotData, hs * PLE_ROW_BYTES, dstOff, dstOff + PLE_ROW_BYTES);
          }
        }
        const shardFile = path.join(pleShardsDir, `ple_${String(s).padStart(3, '0')}.bin`);
        fs.writeFileSync(shardFile, shardOut);
      }
    } finally {
      fs.closeSync(pleFd);
    }
    fs.writeFileSync(path.join(BUNDLE_DIR, 'ple_hot_32k.bin'), pleHotData);
    fs.writeFileSync(path.join(BUNDLE_DIR, 'ple_hot_map.bin'), Buffer.from(hotMap.buffer));

    const cfgJson = JSON.parse(fs.readFileSync(path.join(HF_TMP_DIR, 'config.json'), 'utf8'));
    const txtCfg = cfgJson.text_config ?? cfgJson;

    const manifest = {
      version: 4,
      arch: 'gemma4-e4b-qat',
      modelId: MODEL_ID,
      complete: true,
      packedAt: new Date().toISOString(),
      textConfig: {
        hiddenSize: txtCfg.hidden_size,
        intermediateSize: txtCfg.intermediate_size,
        numLayers: txtCfg.num_hidden_layers,
        vocabSize: txtCfg.vocab_size,
        numAttentionHeads: txtCfg.num_attention_heads,
        numKeyValueHeads: txtCfg.num_key_value_heads,
        headDim: txtCfg.head_dim,
        globalHeadDim: txtCfg.global_head_dim,
        slidingWindow: txtCfg.sliding_window,
        numKvSharedLayers: txtCfg.num_kv_shared_layers,
        hiddenSizePerLayerInput: txtCfg.hidden_size_per_layer_input,
        finalLogitSoftcapping: txtCfg.final_logit_softcapping,
        rmsNormEps: txtCfg.rms_norm_eps,
        layerTypes: txtCfg.layer_types,
        ropeParameters: txtCfg.rope_parameters,
      },
      slab0Bytes: slab0Off,
      slab1Bytes: slab1Off,
      slab2Bytes: slab2Off,
      auxBytes: auxOffset,
      shardMaxBytes: SHARD_MAX_BYTES,
      shards: {
        slab0: slab0Shards,
        slab1: slab1Shards,
        slab2: slab2Shards,
        aux: [{ file: 'aux_params.bin', bytes: auxOffset }],
      },
      lmHeadLayout: {
        qOffset: lmEntry.qOffset,
        qBytes: lmEntry.qBytes,
        sOffset: lmEntry.sOffset,
        sBytes: lmEntry.sBytes,
      },
      auxLayout,
      layerOffsets,
      cpuEmbeddings: {
        embedTokens: embedTokensInfo,
        pleTokens: pleTokensInfo,
      },
      pleShards: {
        dir: 'ple_shards',
        shardCount: SHARD_COUNT,
        tokensPerShard: TOKENS_PER_SHARD,
        rowBytes: PLE_ROW_BYTES,
      },
      pleHot: {
        dataFile: 'ple_hot_32k.bin',
        dataBytes: pleHotData.byteLength,
        mapFile: 'ple_hot_map.bin',
        mapBytes: hotMap.byteLength,
        entryCount: HOT_COUNT,
      },
    };

    const manifestPath = path.join(BUNDLE_DIR, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    console.log(
      `Packed Gemma 4 E4B WGSL v4 bundle: slab0=${(slab0Off / 1e6).toFixed(1)} MB (${slab0Shards.length} parts), slab1=${(slab1Off / 1e6).toFixed(1)} MB (${slab1Shards.length} parts), slab2=${(slab2Off / 1e6).toFixed(1)} MB (${slab2Shards.length} parts), aux=${(auxOffset / 1e6).toFixed(1)} MB, pleShards=${SHARD_COUNT}, pleHot=${HOT_COUNT} rows`,
    );
    console.log(`Manifest written to ${manifestPath}`);
  } finally {
    ctx.close();
  }
}

async function main() {
  const args = new Set(process.argv.slice(2));
  const downloadOnly = args.has('--download-only');
  const inspectOnly = args.has('--inspect-math');
  const repackOnly = args.has('--repack');
  const runAll = !downloadOnly && !inspectOnly && !repackOnly;

  if (downloadOnly || runAll) {
    console.log(`\n=== Downloading Gemma 4 E4B QAT Mobile (${MODEL_ID}) ===`);
    for (const rel of HF_DOWNLOAD_FILES) {
      await downloadFile(rel);
    }
  }
  if (inspectOnly || runAll) {
    inspectMath();
  }
  if (repackOnly || runAll) {
    repackGemma4Bundle();
  }
}

main().catch((err) => {
  console.error('bundle-gemma4 failed:', err);
  process.exit(1);
});
