import type { OpfsModelCache } from './opfsCache.ts';

export const WGSL_REAL_CACHE_DIR = 'tinygpt-wgsl-qwen35-4b-v1';

export const QWEN35_HIDDEN = 2560;
export const QWEN35_INTERMEDIATE = 9216;
export const QWEN35_NUM_LAYERS = 32;
export const QWEN35_VOCAB_SIZE = 248320;
export const QWEN35_FULL_ATTN_LAYERS = new Set([3, 7, 11, 15, 19, 23, 27, 31]);

export interface OnnxTensorMeta {
  name: string;
  dims: number[];
  dataType: number;
  rawData: Uint8Array | null;
  externalLocation: string | null;
  externalOffset: number;
  externalLength: number;
}

export interface RepackWasmExports {
  memory: WebAssembly.Memory;
  wasm_scratch_ptr(): number;
  wasm_scratch_len(): number;
  q4_row_u32s(k: number): number;
  repack_q4_rows(
    rows: number,
    blocksPerRow: number,
    quantPtr: number,
    scalesPtr: number,
    zpPtr: number,
    outU32Ptr: number,
  ): void;
  f16_slice_to_f32(count: number, srcU16Ptr: number, dstF32Ptr: number): void;
}

export interface LayerAuxLayout {
  inputNormOff: number;
  postNormOff: number;
  isFullAttn: boolean;
  dnAuxOff: number; // [A_neg(32), dt_bias(32), gdn_norm(128)]
  convWOff: number; // [8192 * 4]
  qNormOff: number; // [256]
  kNormOff: number; // [256]
}

export interface AuxLayout {
  invFreqOff: number;
  finalNormOff: number;
  totalF32s: number;
  layers: LayerAuxLayout[];
}

export interface WgslSlabShard {
  file: string;
  bytes: number;
}

export interface WgslSlabManifest {
  version: number;
  modelId: string;
  complete: boolean;
  slab0Bytes: number;
  slab1Bytes: number;
  slab2Bytes: number;
  auxBytes: number;
  shardMaxBytes?: number;
  shards?: {
    slab0: WgslSlabShard[];
    slab1: WgslSlabShard[];
    slab2: WgslSlabShard[];
    aux: WgslSlabShard[];
  };
  createdAt: string;
}

export interface WgslWeightSlabFiles {
  loadedFromCache: boolean;
  repackMs: number;
  slab0: File;
  slab1: File;
  slab2: File;
  aux: File;
  manifest: WgslSlabManifest;
}

export function computeAuxLayout(): AuxLayout {
  let off = 0;
  const invFreqOff = off;
  off += 32;
  const finalNormOff = off;
  off += QWEN35_HIDDEN;

  const layers: LayerAuxLayout[] = [];
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
      layers.push({
        inputNormOff,
        postNormOff,
        isFullAttn: true,
        dnAuxOff: 0,
        convWOff: 0,
        qNormOff,
        kNormOff,
      });
    } else {
      const dnAuxOff = off;
      off += 32 + 32 + 128; // 192 f32s
      const convWOff = off;
      off += 8192 * 4;
      layers.push({
        inputNormOff,
        postNormOff,
        isFullAttn: false,
        dnAuxOff,
        convWOff,
        qNormOff: 0,
        kNormOff: 0,
      });
    }
  }

  return { invFreqOff, finalNormOff, totalF32s: off, layers };
}

function readVarint(buf: Uint8Array, pos: number): { val: number; pos: number } {
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

function parseProtoFields(
  buf: Uint8Array,
  start: number,
  end: number,
  onField: (field: number, wire: number, valOrStart: number, sliceEnd?: number) => void,
): void {
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

function parseTensorProto(buf: Uint8Array, start: number, end: number): OnnxTensorMeta {
  const tensor: OnnxTensorMeta = {
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

/**
 * Extracts all initializer tensor descriptors from an ONNX ModelProto buffer.
 */
export function parseOnnxTensorManifest(onnxBytes: Uint8Array): Map<string, OnnxTensorMeta> {
  const map = new Map<string, OnnxTensorMeta>();
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

async function readTensorBytes(
  tensor: OnnxTensorMeta,
  extFiles: Record<string, File>,
  subOffset = 0,
  subLength?: number,
): Promise<Uint8Array> {
  if (tensor.rawData) {
    const start = subOffset;
    const end = subLength !== undefined ? start + subLength : tensor.rawData.length;
    return tensor.rawData.slice(start, end);
  }
  if (!tensor.externalLocation) {
    throw new Error(`Tensor ${tensor.name} has neither rawData nor externalLocation.`);
  }
  const file = extFiles[tensor.externalLocation];
  if (!file) {
    throw new Error(`Missing external ONNX file ${tensor.externalLocation} for ${tensor.name}.`);
  }
  const start = tensor.externalOffset + subOffset;
  const len = subLength !== undefined ? subLength : tensor.externalLength - subOffset;
  return new Uint8Array(await file.slice(start, start + len).arrayBuffer());
}

async function repackQ4MatrixToWritable(
  writable: FileSystemWritableFileStream,
  manifest: Map<string, OnnxTensorMeta>,
  prefix: string,
  expectedRows: number,
  expectedCols: number,
  extFiles: Record<string, File>,
  wasm: RepackWasmExports,
): Promise<number> {
  const tQuant = manifest.get(`${prefix}_weight_quant`);
  const tScales = manifest.get(`${prefix}_weight_scales`);
  const tZp = manifest.get(`${prefix}_weight_zp`);
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
    const qBytes = await readTensorBytes(tQuant, extFiles, rowStart * quantRowBytes, rows * quantRowBytes);
    const sBytes = await readTensorBytes(tScales, extFiles, rowStart * scalesRowBytes, rows * scalesRowBytes);
    const zBytes = await readTensorBytes(tZp, extFiles, rowStart * zpStride, rows * zpStride);

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
    const outView = new Uint8Array(wasm.memory.buffer, outPtr, outByteLen).slice();
    await writable.write(outView as unknown as FileSystemWriteChunkType);
    totalWritten += outByteLen;
  }

  return totalWritten;
}

export async function clearWgslWeightCache(opfsCache: OpfsModelCache): Promise<void> {
  try {
    const root = await opfsCache.getRootDirectory();
    await root.removeEntry(WGSL_REAL_CACHE_DIR, { recursive: true });
  } catch {
    // Ignore when absent
  }
}

export async function ensureWgslWeightSlabs(
  opfsCache: OpfsModelCache,
  wasm: RepackWasmExports,
  options: {
    modelId: string;
    onProgress?: (label: string, pct: number) => void;
  },
): Promise<WgslWeightSlabFiles> {
  const root = await opfsCache.getRootDirectory();
  try {
    const dir = await root.getDirectoryHandle(WGSL_REAL_CACHE_DIR, { create: false });
    const manifestFile = await (await dir.getFileHandle('manifest.json')).getFile();
    const manifest = JSON.parse(await manifestFile.text()) as WgslSlabManifest;
    if (manifest.version === 1 && manifest.complete && manifest.modelId === options.modelId) {
      const [slab0, slab1, slab2, aux] = await Promise.all([
        (await dir.getFileHandle('slab0.bin')).getFile(),
        (await dir.getFileHandle('slab1.bin')).getFile(),
        (await dir.getFileHandle('slab2.bin')).getFile(),
        (await dir.getFileHandle('aux.bin')).getFile(),
      ]);
      if (
        slab0.size === manifest.slab0Bytes &&
        slab1.size === manifest.slab1Bytes &&
        slab2.size === manifest.slab2Bytes &&
        aux.size === manifest.auxBytes
      ) {
        return { loadedFromCache: true, repackMs: 0, slab0, slab1, slab2, aux, manifest };
      }
    }
  } catch {
    // Cache miss or incomplete — rebuild cleanly below
  }

  const repackStart = performance.now();
  const baseUrl = `https://huggingface.co/${options.modelId}/resolve/main/onnx`;
  options.onProgress?.('Resolving Qwen3.5-4B ONNX weight files from OPFS...', 5);

  const [onnxFile, data0File, data1File] = await Promise.all([
    opfsCache.ensureCachedFile(`${baseUrl}/decoder_model_merged_q4f16.onnx`),
    opfsCache.ensureCachedFile(`${baseUrl}/decoder_model_merged_q4f16.onnx_data`),
    opfsCache.ensureCachedFile(`${baseUrl}/decoder_model_merged_q4f16.onnx_data_1`),
  ]);
  const extFiles: Record<string, File> = {
    'decoder_model_merged_q4f16.onnx_data': data0File,
    'decoder_model_merged_q4f16.onnx_data_1': data1File,
  };

  options.onProgress?.('Parsing ONNX tensor graph manifest...', 12);
  const onnxBytes = new Uint8Array(await onnxFile.arrayBuffer());
  const manifestMap = parseOnnxTensorManifest(onnxBytes);

  try {
    await root.removeEntry(WGSL_REAL_CACHE_DIR, { recursive: true });
  } catch {
    // Ignore
  }
  const dir = await root.getDirectoryHandle(WGSL_REAL_CACHE_DIR, { create: true });

  // 1. Pack aux.bin (all FP32 norms, RoPE inv_freq, DeltaNet A_neg/dt_bias/norm/conv1d)
  options.onProgress?.('Packing FP32 norms & DeltaNet parameters into aux.bin...', 16);
  const auxLayout = computeAuxLayout();
  const auxF32 = new Float32Array(auxLayout.totalF32s);
  const writeF32Tensor = async (tensorName: string, dstOffset: number, expectedCount: number) => {
    const t = manifestMap.get(tensorName);
    if (!t) throw new Error(`Missing ONNX initializer ${tensorName}`);
    const raw = await readTensorBytes(t, extFiles);
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    if (f32.length !== expectedCount) {
      throw new Error(`Aux tensor ${tensorName} length ${f32.length} != ${expectedCount}`);
    }
    auxF32.set(f32, dstOffset);
  };

  await writeF32Tensor('model.inv_freq', auxLayout.invFreqOff, 32);
  await writeF32Tensor('model.layers.32.final_norm_layernorm.weight', auxLayout.finalNormOff, QWEN35_HIDDEN);

  for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
    const la = auxLayout.layers[l];
    await writeF32Tensor(`model.layers.${l}.input_layernorm.weight`, la.inputNormOff, QWEN35_HIDDEN);
    await writeF32Tensor(`model.layers.${l}.post_attention_layernorm.weight`, la.postNormOff, QWEN35_HIDDEN);
    if (la.isFullAttn) {
      await writeF32Tensor(`model.layers.${l}.attn.q_norm.layernorm.weight`, la.qNormOff, 256);
      await writeF32Tensor(`model.layers.${l}.attn.k_norm.layernorm.weight`, la.kNormOff, 256);
    } else {
      await writeF32Tensor(`model.layers.${l}.gdn.A_neg_exp`, la.dnAuxOff, 32);
      await writeF32Tensor(`model.layers.${l}.gdn.dt_bias`, la.dnAuxOff + 32, 32);
      await writeF32Tensor(`model.layers.${l}.gdn.norm.weight`, la.dnAuxOff + 64, 128);

      const tConv = manifestMap.get(`model.layers.${l}.gdn.conv1d.weight_3d`);
      if (!tConv) throw new Error(`Missing conv1d weights for layer ${l}`);
      const convF16Bytes = await readTensorBytes(tConv, extFiles);
      const count = 8192 * 4;
      const srcPtr = wasm.wasm_scratch_ptr();
      const dstPtr = (srcPtr + convF16Bytes.byteLength + 3) & ~3;
      new Uint8Array(wasm.memory.buffer).set(convF16Bytes, srcPtr);
      wasm.f16_slice_to_f32(count, srcPtr, dstPtr);
      auxF32.set(new Float32Array(wasm.memory.buffer, dstPtr, count), la.convWOff);
    }
  }

  const auxHandle = await dir.getFileHandle('aux.bin', { create: true });
  const auxWritable = await auxHandle.createWritable();
  await auxWritable.write(new Uint8Array(auxF32.buffer) as unknown as FileSystemWriteChunkType);
  await auxWritable.close();

  // 2. Repack Q4 layers 0..15 -> slab0.bin, layers 16..31 -> slab1.bin, lm_head -> slab2.bin
  const slab0Handle = await dir.getFileHandle('slab0.bin', { create: true });
  const slab0Writable = await slab0Handle.createWritable();
  const slab1Handle = await dir.getFileHandle('slab1.bin', { create: true });
  const slab1Writable = await slab1Handle.createWritable();

  let slab0Bytes = 0;
  let slab1Bytes = 0;

  try {
    for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
      const pct = 20 + (l / QWEN35_NUM_LAYERS) * 60;
      options.onProgress?.(`Repacking Qwen3.5-4B Q4 layer ${l + 1}/${QWEN35_NUM_LAYERS} via Rust WASM...`, pct);
      const w = l < 16 ? slab0Writable : slab1Writable;
      let layerWritten = 0;

      if (!QWEN35_FULL_ATTN_LAYERS.has(l)) {
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_gdn_in_proj_qkv_MatMul`, 8192, QWEN35_HIDDEN, extFiles, wasm);
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_gdn_in_proj_z_MatMul`, 4096, QWEN35_HIDDEN, extFiles, wasm);
        // Pack in_proj_b (rows 0..31) + in_proj_a (rows 32..63) contiguously as one [64, 2560] matrix
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_gdn_in_proj_b_MatMul`, 32, QWEN35_HIDDEN, extFiles, wasm);
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_gdn_in_proj_a_MatMul`, 32, QWEN35_HIDDEN, extFiles, wasm);
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_gdn_out_proj_MatMul`, QWEN35_HIDDEN, 4096, extFiles, wasm);
      } else {
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_attn_q_proj_MatMul`, 8192, QWEN35_HIDDEN, extFiles, wasm);
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_attn_k_proj_MatMul`, 1024, QWEN35_HIDDEN, extFiles, wasm);
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_attn_v_proj_MatMul`, 1024, QWEN35_HIDDEN, extFiles, wasm);
        layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_attn_o_proj_MatMul`, QWEN35_HIDDEN, 4096, extFiles, wasm);
      }

      layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_mlp_gate_proj_MatMul`, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, extFiles, wasm);
      layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_mlp_up_proj_MatMul`, QWEN35_INTERMEDIATE, QWEN35_HIDDEN, extFiles, wasm);
      layerWritten += await repackQ4MatrixToWritable(w, manifestMap, `model_layers_${l}_mlp_down_proj_MatMul`, QWEN35_HIDDEN, QWEN35_INTERMEDIATE, extFiles, wasm);

      if (l < 16) slab0Bytes += layerWritten;
      else slab1Bytes += layerWritten;
    }
    await slab0Writable.close();
    await slab1Writable.close();
  } catch (err) {
    await slab0Writable.abort().catch(() => {});
    await slab1Writable.abort().catch(() => {});
    throw err;
  }

  options.onProgress?.('Repacking LM Head / Tied Embeddings (248,320 x 2,560) via Rust WASM...', 84);
  const slab2Handle = await dir.getFileHandle('slab2.bin', { create: true });
  const slab2Writable = await slab2Handle.createWritable();
  let slab2Bytes = 0;
  try {
    slab2Bytes = await repackQ4MatrixToWritable(
      slab2Writable,
      manifestMap,
      'lm_head_MatMul',
      QWEN35_VOCAB_SIZE,
      QWEN35_HIDDEN,
      extFiles,
      wasm,
    );
    await slab2Writable.close();
  } catch (err) {
    await slab2Writable.abort().catch(() => {});
    throw err;
  }

  const manifest: WgslSlabManifest = {
    version: 1,
    modelId: options.modelId,
    complete: true,
    slab0Bytes,
    slab1Bytes,
    slab2Bytes,
    auxBytes: auxF32.byteLength,
    createdAt: new Date().toISOString(),
  };
  const mHandle = await dir.getFileHandle('manifest.json', { create: true });
  const mWritable = await mHandle.createWritable();
  await mWritable.write(JSON.stringify(manifest, null, 2));
  await mWritable.close();

  const [slab0, slab1, slab2, aux] = await Promise.all([
    slab0Handle.getFile(),
    slab1Handle.getFile(),
    slab2Handle.getFile(),
    auxHandle.getFile(),
  ]);

  return {
    loadedFromCache: false,
    repackMs: performance.now() - repackStart,
    slab0,
    slab1,
    slab2,
    aux,
    manifest,
  };
}
