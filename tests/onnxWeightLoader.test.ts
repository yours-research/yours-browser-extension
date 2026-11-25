import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  computeAuxLayout,
  parseOnnxTensorManifest,
  QWEN35_FULL_ATTN_LAYERS,
  QWEN35_HIDDEN,
  QWEN35_NUM_LAYERS,
} from '../src/offscreen/onnxWeightLoader.ts';

interface RepackWasmTestExports {
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
  cpu_ref_q4_gemv(rows: number, cols: number, wPtr: number, xPtr: number, outPtr: number): void;
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WASM_PATH = path.resolve(
  __dirname,
  '../crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
);

function encodeVarint(val: number): number[] {
  const out: number[] = [];
  let v = val >>> 0;
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  out.push(v);
  return out;
}

function encodeTag(field: number, wire: number): number[] {
  return encodeVarint((field << 3) | wire);
}

function encodeBytes(field: number, bytes: Uint8Array): Uint8Array {
  const tag = encodeTag(field, 2);
  const len = encodeVarint(bytes.length);
  const out = new Uint8Array(tag.length + len.length + bytes.length);
  out.set(tag, 0);
  out.set(len, tag.length);
  out.set(bytes, tag.length + len.length);
  return out;
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

function encodeStringPair(key: string, value: string): Uint8Array {
  const enc = new TextEncoder();
  return concatBytes([
    encodeBytes(1, enc.encode(key)),
    encodeBytes(2, enc.encode(value)),
  ]);
}

function encodeTensorProto(opts: {
  name: string;
  dims: number[];
  dataType: number;
  rawData?: Uint8Array;
  external?: { location: string; offset: number; length: number };
}): Uint8Array {
  const enc = new TextEncoder();
  const packedDims: number[] = [];
  for (const d of opts.dims) {
    packedDims.push(...encodeVarint(d));
  }
  const chunks: Uint8Array[] = [
    encodeBytes(1, new Uint8Array(packedDims)),
    new Uint8Array([...encodeTag(2, 0), ...encodeVarint(opts.dataType)]),
    encodeBytes(8, enc.encode(opts.name)),
  ];
  if (opts.rawData) {
    chunks.push(encodeBytes(9, opts.rawData));
  }
  if (opts.external) {
    chunks.push(encodeBytes(13, encodeStringPair('location', opts.external.location)));
    chunks.push(encodeBytes(13, encodeStringPair('offset', String(opts.external.offset))));
    chunks.push(encodeBytes(13, encodeStringPair('length', String(opts.external.length))));
  }
  return concatBytes(chunks);
}

describe('onnxWeightLoader', () => {
  it('parses ONNX ModelProto raw_data and external_data initializers accurately', () => {
    const f32Vals = new Float32Array([1.25, -2.5, 3.75, 0.5]);
    const rawTensor = encodeTensorProto({
      name: 'model.inv_freq',
      dims: [4],
      dataType: 1,
      rawData: new Uint8Array(f32Vals.buffer).slice(),
    });
    const extTensor = encodeTensorProto({
      name: 'model.layers.0.mlp.gate_proj.weight_Q4',
      dims: [9216, 80, 16],
      dataType: 2,
      external: {
        location: 'decoder_model_merged_q4f16.onnx_data',
        offset: 123456,
        length: 11796480,
      },
    });
    const graphProto = concatBytes([
      encodeBytes(5, rawTensor),
      encodeBytes(5, extTensor),
    ]);
    // Include an odd-length ir_version + producer_name before graph to ensure non-aligned offsets
    const modelProto = concatBytes([
      new Uint8Array([...encodeTag(1, 0), ...encodeVarint(9)]),
      encodeBytes(2, new TextEncoder().encode('onnx-test')),
      encodeBytes(7, graphProto),
    ]);

    const manifest = parseOnnxTensorManifest(modelProto);
    expect(manifest.size).toBe(2);

    const invFreq = manifest.get('model.inv_freq');
    expect(invFreq).toBeDefined();
    expect(invFreq!.dims).toEqual([4]);
    expect(invFreq!.dataType).toBe(1);
    expect(invFreq!.externalLocation).toBeNull();
    expect(invFreq!.rawData).not.toBeNull();
    // Copying slice gives clean 4-byte aligned Float32Array regardless of rawData.byteOffset
    const alignedCopy = invFreq!.rawData!.slice();
    const decodedF32 = new Float32Array(alignedCopy.buffer);
    expect(Array.from(decodedF32)).toEqual([1.25, -2.5, 3.75, 0.5]);

    const gateQ4 = manifest.get('model.layers.0.mlp.gate_proj.weight_Q4');
    expect(gateQ4).toBeDefined();
    expect(gateQ4!.dims).toEqual([9216, 80, 16]);
    expect(gateQ4!.dataType).toBe(2);
    expect(gateQ4!.rawData).toBeNull();
    expect(gateQ4!.externalLocation).toBe('decoder_model_merged_q4f16.onnx_data');
    expect(gateQ4!.externalOffset).toBe(123456);
    expect(gateQ4!.externalLength).toBe(11796480);
  });

  it('computes Qwen3.5-4B aux layout matching 24 DeltaNet and 8 FullAttn layers', () => {
    const layout = computeAuxLayout();
    expect(layout.invFreqOff).toBe(0);
    expect(layout.finalNormOff).toBe(32);
    expect(layout.layers).toHaveLength(QWEN35_NUM_LAYERS);

    let fullCount = 0;
    let dnCount = 0;
    for (let l = 0; l < QWEN35_NUM_LAYERS; l++) {
      const layer = layout.layers[l];
      if (QWEN35_FULL_ATTN_LAYERS.has(l)) {
        fullCount++;
        expect(layer.isFullAttn).toBe(true);
        expect(layer.kNormOff - layer.qNormOff).toBe(256);
      } else {
        dnCount++;
        expect(layer.isFullAttn).toBe(false);
        expect(layer.convWOff - layer.dnAuxOff).toBe(192);
      }
      expect(layer.postNormOff - layer.inputNormOff).toBe(QWEN35_HIDDEN);
    }
    expect(fullCount).toBe(8);
    expect(dnCount).toBe(24);
    const expectedTotal =
      32 +
      QWEN35_HIDDEN +
      QWEN35_NUM_LAYERS * (2 * QWEN35_HIDDEN) +
      8 * 512 +
      24 * (192 + 8192 * 4);
    expect(layout.totalF32s).toBe(expectedTotal);
  });

  it('repacks ONNX MatMulNBits Q4+scale+zp via Rust WASM matching CPU reference GEMV', async () => {
    const wasmBytes = fs.readFileSync(WASM_PATH);
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    const wasm = instance.exports as unknown as RepackWasmTestExports;

    const base = wasm.wasm_scratch_ptr();

    // 1. Check f16_slice_to_f32 with exact IEEE-754 binary16 values:
    // 0x3c00 = 1.0, 0xbc00 = -1.0, 0x3800 = 0.5, 0x3400 = 0.25, 0x4200 = 3.0, 0xc000 = -2.0
    const srcU16 = new Uint16Array([0x3c00, 0xbc00, 0x3800, 0x3400, 0x4200, 0xc000]);
    const dstPtr = base + 64;
    new Uint16Array(wasm.memory.buffer, base, srcU16.length).set(srcU16);
    wasm.f16_slice_to_f32(srcU16.length, base, dstPtr);
    const decoded = Array.from(new Float32Array(wasm.memory.buffer, dstPtr, srcU16.length));
    expect(decoded).toEqual([1.0, -1.0, 0.5, 0.25, 3.0, -2.0]);

    // 2. Check repack_q4_rows against exact (q - zp) * scale reference GEMV
    const rows = 2;
    const cols = 64;
    const blocksPerRow = cols / 32; // 2
    const quantSize = rows * blocksPerRow * 16; // 64 bytes
    const scalesCount = rows * blocksPerRow; // 4 u16s = 8 bytes
    const zpBytes = rows * Math.ceil(blocksPerRow / 2); // 2 bytes

    const qData = new Uint8Array(quantSize);
    for (let i = 0; i < quantSize; i++) {
      const qEven = (i * 3 + 1) & 0x0f;
      const qOdd = (i * 5 + 7) & 0x0f;
      qData[i] = qEven | (qOdd << 4);
    }
    // Exact binary16 scales: [0.5, 0.25, -0.5, 1.0]
    const sU16 = new Uint16Array([0x3800, 0x3400, 0xb800, 0x3c00]);
    const sFloat = [0.5, 0.25, -0.5, 1.0];
    // Packed 4-bit zero points: row 0 -> [8, 5] => 0x58; row 1 -> [7, 9] => 0x97
    const zpData = new Uint8Array([0x58, 0x97]);
    const zpFloat = [
      [8, 5],
      [7, 9],
    ];

    const xVec = new Float32Array(cols);
    for (let k = 0; k < cols; k++) {
      xVec[k] = ((k % 7) - 3) * 0.25;
    }

    const qPtr = base;
    const sPtr = qPtr + 128;
    const zpPtr = sPtr + 64;
    const xPtr = zpPtr + 64;
    const outWPtr = xPtr + cols * 4;
    const gemvOutPtr = outWPtr + rows * wasm.q4_row_u32s(cols) * 4;

    new Uint8Array(wasm.memory.buffer, qPtr, quantSize).set(qData);
    new Uint16Array(wasm.memory.buffer, sPtr, scalesCount).set(sU16);
    new Uint8Array(wasm.memory.buffer, zpPtr, zpBytes).set(zpData);
    new Float32Array(wasm.memory.buffer, xPtr, cols).set(xVec);

    wasm.repack_q4_rows(rows, blocksPerRow, qPtr, sPtr, zpPtr, outWPtr);
    wasm.cpu_ref_q4_gemv(rows, cols, outWPtr, xPtr, gemvOutPtr);

    const wasmOut = Array.from(new Float32Array(wasm.memory.buffer, gemvOutPtr, rows));

    // Exact TS dequantization reference: y[r] = sum_{b,j} ((q - zp) * scale) * x[k]
    const refOut: number[] = [];
    for (let r = 0; r < rows; r++) {
      let acc = 0;
      for (let b = 0; b < blocksPerRow; b++) {
        const scale = sFloat[r * blocksPerRow + b];
        const zp = zpFloat[r][b];
        for (let byteIdx = 0; byteIdx < 16; byteIdx++) {
          const packed = qData[(r * blocksPerRow + b) * 16 + byteIdx];
          const k0 = b * 32 + byteIdx * 2;
          const k1 = k0 + 1;
          acc += ((packed & 0x0f) - zp) * scale * xVec[k0];
          acc += (((packed >> 4) & 0x0f) - zp) * scale * xVec[k1];
        }
      }
      refOut.push(acc);
    }

    for (let r = 0; r < rows; r++) {
      expect(Math.abs(wasmOut[r] - refOut[r])).toBeLessThan(1e-3);
    }
  });
});
