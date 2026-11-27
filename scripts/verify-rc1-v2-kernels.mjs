import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const WASM_PATH = path.resolve(
  'crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
);

const SHADER_IDS = {
  DN_TREE_V2: 48,
  GQA_TREE_V2: 49,
  Q4_GEMV_SMALLM_V2: 54,
  Q4_GATE_UP_SWIGLU_SMALLM_V2: 55,
  Q4_GEMV_SMALLM_V2_F16: 56,
  Q4_GATE_UP_SWIGLU_SMALLM_V2_F16: 57,
  GQA_TREE_V2_SG: 58,
};

function createRng(seed) {
  let state = seed >>> 0;
  return {
    nextU32() {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return (t ^ (t >>> 14)) >>> 0;
    },
    next() {
      return this.nextU32() / 4294967296;
    },
    int(n) {
      return this.nextU32() % n;
    },
  };
}

const f32Arr = new Float32Array(1);
const u32View = new Uint32Array(f32Arr.buffer);
function f32ToBits(v) {
  f32Arr[0] = v;
  return u32View[0] >>> 0;
}
function bitsToF32(b) {
  u32View[0] = b >>> 0;
  return f32Arr[0];
}
function stageF32(v) {
  const b = f32ToBits(v);
  if ((b & 0x7f800000) === 0) return bitsToF32(b & 0x80000000);
  return bitsToF32(b);
}
function addF(a, b) {
  return Math.fround(a + b);
}
function subF(a, b) {
  return Math.fround(a - b);
}
function mulF(a, b) {
  return Math.fround(a * b);
}
function divF(a, b) {
  return Math.fround(a / b);
}
function rsqrtF(x) {
  return Math.fround(1 / Math.sqrt(Math.fround(x)));
}
function f16ToF32(h) {
  const s = (h >>> 15) & 1;
  const e = (h >>> 10) & 31;
  const f = h & 1023;
  if (e === 0) {
    return f === 0 ? (s ? -0 : 0) : Math.fround((s ? -1 : 1) * (f / 1024) * 2 ** -14);
  }
  if (e === 31) return f === 0 ? (s ? -Infinity : Infinity) : NaN;
  return Math.fround((s ? -1 : 1) * (1 + f / 1024) * 2 ** (e - 15));
}
function f32ToF16(v) {
  const b = f32ToBits(v);
  const s = (b >>> 16) & 0x8000;
  const exp = ((b >>> 23) & 0xff) - 127 + 15;
  const frac = b & 0x7fffff;
  if (exp <= 0) return s;
  if (exp >= 31) return s | 0x7c00;
  return s | (exp << 10) | (frac >>> 13);
}
function expNeg(x) {
  x = Math.fround(x);
  if (x <= Math.fround(-87.33654)) return 0;
  const n = Math.trunc(Math.round(mulF(x, Math.fround(1.4426950408889634))));
  const fn = Math.fround(n);
  const r = subF(subF(x, mulF(fn, Math.fround(0.693359375))), mulF(fn, Math.fround(-0.00021219444005469058)));
  let y = addF(stageF32(mulF(r, Math.fround(0.0013888889))), Math.fround(0.0083333338));
  y = addF(stageF32(mulF(r, y)), Math.fround(0.0416666679));
  y = addF(stageF32(mulF(r, y)), Math.fround(0.1666666716));
  y = addF(stageF32(mulF(r, y)), 0.5);
  y = addF(stageF32(mulF(r, y)), 1.0);
  y = addF(stageF32(mulF(r, y)), 1.0);
  const pow2n = bitsToF32(((n + 127) << 23) >>> 0);
  return mulF(pow2n, y);
}

function q4SmallMV2Ref(params, weights, input, initial, fused = false) {
  const [rows, cols, weight, x, y, residual, tokens, strideOrUp] = params;
  const stride = fused || !strideOrUp ? rows : strideOrUp;
  const blocks = Math.ceil(cols / 32);
  const output = Float32Array.from(initial);
  function dot(offset, token, row) {
    let acc = 0;
    for (let c = 0; c < cols; c++) {
      const base = offset + 5 * (row * blocks + (c >>> 5));
      const affine = weights[base + 4];
      const q = (weights[base + ((c & 31) >>> 3)] >>> (4 * (c & 7))) & 15;
      const w = addF(mulF(q, f16ToF32(affine & 65535)), f16ToF32(affine >>> 16));
      acc = addF(acc, stageF32(mulF(w, input[x + token * cols + c])));
    }
    return acc;
  }
  for (let token = 0; token < tokens; token++) {
    for (let row = 0; row < rows; row++) {
      let value = dot(weight, token, row);
      if (fused) {
        const e = expNeg(-Math.abs(value));
        const sigmoid = stageF32(mulF(value < 0 ? e : 1, stageF32(divF(1, addF(1, e)))));
        value = stageF32(mulF(stageF32(mulF(value, sigmoid)), dot(strideOrUp, token, row)));
      }
      const dst = y + token * stride + row;
      output[dst] = residual ? addF(output[dst], value) : value;
    }
  }
  return Array.from(output, f32ToBits);
}

function masksOf(parents) {
  const masks = [];
  for (let i = 0; i < parents.length; i++) {
    const p = parents[i];
    masks.push(p < 0 ? (1 << i) >>> 0 : (masks[p] | (1 << i)) >>> 0);
  }
  return masks;
}

function dnTreeV2Ref(params, parents, rec, initial) {
  const [slots, heads, dk, dv] = params;
  const stride = 2 * dk + dv + 2;
  const states = [Float32Array.from(initial)];
  const out = new Float32Array(slots * heads * dv);
  for (let slot = 0; slot < slots; slot++) {
    const pState = slot === 0 ? states[0] : states[parents[slot]];
    const nState = Float32Array.from(pState);
    for (let h = 0; h < heads; h++) {
      const base = (slot * heads + h) * stride;
      const key = base + dk;
      const val = key + dk;
      const decay = rec[val + dv];
      const beta = rec[val + dv + 1];
      for (let j = 0; j < dv; j++) {
        let pred = 0;
        for (let i = 0; i < dk; i++) {
          const idx = (h * dk + i) * dv + j;
          nState[idx] = mulF(nState[idx], decay);
          pred = addF(pred, mulF(nState[idx], rec[key + i]));
        }
        const err = mulF(subF(rec[val + j], pred), beta);
        let y = 0;
        for (let i = 0; i < dk; i++) {
          const idx = (h * dk + i) * dv + j;
          nState[idx] = addF(nState[idx], mulF(rec[key + i], err));
          y = addF(y, mulF(nState[idx], rec[base + i]));
        }
        out[(slot * heads + h) * dv + j] = y;
      }
    }
    states[slot] = nState;
  }
  return { output: Array.from(out, f32ToBits), states };
}

function partitionCount(len) {
  if (len <= 64) return 1;
  if (len <= 512) return 8;
  if (len <= 4096) return 16;
  return 32;
}

function ropeTableRef(positions, rotary, theta = 10000000) {
  const half = rotary / 2;
  const table = new Array(positions * half * 2);
  for (let pos = 0; pos < positions; pos++) {
    for (let i = 0; i < half; i++) {
      const angle = pos / theta ** ((2 * i) / rotary);
      table[(pos * half + i) * 2] = Math.fround(Math.cos(angle));
      table[(pos * half + i) * 2 + 1] = Math.fround(Math.sin(angle));
    }
  }
  return table;
}

function popcount(x) {
  let c = 0;
  for (let v = x >>> 0; v; v >>>= 1) c += v & 1;
  return c;
}

function gqaTreeV2Ref(params, masks, query, rope, cache) {
  const [prefix, capacity, qHeads, kvHeads, dim, int8, base, stride, rotary, maxParts] = params;
  const words = int8 ? dim / 4 + 1 : dim / 2;
  const half = rotary / 2;
  const invRoot = rsqrtF(dim);
  const EMPTY_PEAK = Math.fround(-3.40282346e38);
  const slots = masks.length;
  const partials = new Array(slots * qHeads * maxParts * (dim + 2)).fill(0);
  const out = new Array(slots * qHeads * dim).fill(0);
  const q = new Float32Array(dim);
  const k = new Float32Array(dim);
  const v = new Float32Array(dim);
  const acc = new Float32Array(dim);
  const lanes = new Float32Array(64);
  const load = (dst, off, scale) => {
    for (let d = 0; d < dim; d++) {
      dst[d] = !int8
        ? f16ToF32((cache[off + (d >>> 1)] >>> ((d % 2) * 16)) & 65535)
        : mulF((cache[off + (d >>> 2)] << (24 - (d % 4) * 8)) >> 24, scale);
    }
  };
  for (let slot = 0; slot < slots; slot++) {
    const mask = masks[slot];
    const len = prefix + popcount(mask);
    const draft = [];
    for (let s = 0; s <= slot; s++) if (mask & (1 << s)) draft.push(prefix + s);
    const count = partitionCount(len);
    const chunk = Math.ceil(len / count);
    for (let head = 0; head < qHeads; head++) {
      const row = slot * qHeads + head;
      const kvHead = Math.floor(head / (qHeads / kvHeads));
      const qBase = row * stride;
      for (let d = 0; d < dim; d++) {
        let val = query[qBase + d];
        if (d < half) {
          const ang = ((len - 1) * half + d) * 2;
          val = subF(mulF(val, rope[ang]), mulF(query[qBase + d + half], rope[ang + 1]));
        } else if (d < rotary) {
          const i = d - half;
          const ang = ((len - 1) * half + i) * 2;
          val = addF(mulF(val, rope[ang]), mulF(query[qBase + i], rope[ang + 1]));
        }
        q[d] = val;
      }
      for (let part = 0; part < count; part++) {
        const lo = part * chunk;
        const hi = Math.min(len, lo + chunk);
        acc.fill(0);
        let peak = EMPTY_PEAK;
        let mass = 0;
        for (let logical = lo; logical < hi; logical++) {
          const t = logical < prefix ? logical : draft[logical - prefix];
          const kb = base + (kvHead * capacity + t) * words;
          const vb = kb + kvHeads * capacity * words;
          const scales = int8 ? cache[kb + words - 1] : 0;
          load(k, kb, int8 ? f16ToF32(scales & 65535) : 1);
          lanes.fill(0);
          for (let d = 0; d < dim; d++) lanes[d % 64] = addF(lanes[d % 64], mulF(q[d], k[d]));
          for (let span = 32; span; span >>= 1)
            for (let i = 0; i < span; i++) lanes[i] = addF(lanes[i], lanes[i + span]);
          const score = mulF(lanes[0], invRoot);
          const high = Math.max(peak, score);
          const a = expNeg(subF(peak, high));
          const b = expNeg(subF(score, high));
          mass = addF(mulF(mass, a), b);
          load(v, vb, int8 ? f16ToF32(scales >>> 16) : 1);
          for (let d = 0; d < dim; d++) acc[d] = addF(mulF(acc[d], a), mulF(v[d], b));
          peak = high;
        }
        const recIdx = (row * maxParts + part) * (dim + 2);
        partials[recIdx] = peak;
        partials[recIdx + 1] = mass;
        for (let d = 0; d < dim; d++) partials[recIdx + 2 + d] = acc[d];
      }
    }
  }
  for (let slot = 0; slot < slots; slot++) {
    const count = partitionCount(prefix + popcount(masks[slot]));
    for (let head = 0; head < qHeads; head++) {
      const row = slot * qHeads + head;
      acc.fill(0);
      let peak = EMPTY_PEAK;
      let mass = 0;
      for (let part = 0; part < count; part++) {
        const recIdx = (row * maxParts + part) * (dim + 2);
        const pPeak = partials[recIdx];
        const pMass = partials[recIdx + 1];
        const high = Math.max(peak, pPeak);
        const a = expNeg(subF(peak, high));
        const b = expNeg(subF(pPeak, high));
        mass = addF(mulF(mass, a), mulF(pMass, b));
        for (let d = 0; d < dim; d++) acc[d] = addF(mulF(acc[d], a), mulF(partials[recIdx + 2 + d], b));
        peak = high;
      }
      const invMass = divF(1, mass);
      for (let d = 0; d < dim; d++) out[row * dim + d] = mulF(acc[d], invMass);
    }
  }
  return out.map(f32ToBits);
}

function buildTestVectors() {
  const vectors = [];
  const rng = createRng(20261127);

  // 1. Q4_GEMV_SMALLM_V2 (linear + residual + vocab shard) & Q4_GATE_UP_SWIGLU_SMALLM_V2 (fused)
  {
    const rows = 67;
    const cols = 160;
    const blocks = Math.ceil(cols / 32);
    const wSize = rows * blocks * 5;
    const weights = Array.from({ length: wSize * 2 + 128 }, (_, idx) => {
      if (idx % 5 === 4) {
        const s = f32ToF16((rng.next() - 0.5) * 0.15 + 0.05);
        const z = f32ToF16((rng.next() - 0.5) * 0.4);
        return (s | (z << 16)) >>> 0;
      }
      return rng.nextU32();
    });
    const upOff = wSize + 35;
    for (const tokens of [1, 5, 16]) {
      const input = Array.from({ length: tokens * cols }, () => Math.fround((rng.next() - 0.5) * 1.4));
      const initial = Array.from({ length: tokens * rows }, () => Math.fround((rng.next() - 0.5) * 0.3));
      const linParams = [rows, cols, 0, 0, 0, tokens % 2, tokens, 0];
      vectors.push({
        name: `q4-v2-linear-m${tokens}`,
        shader: 'Q4_GEMV_SMALLM_V2',
        entryPoint: 'main',
        dispatch: [Math.ceil(rows / 64), 1, 1],
        bindings: [
          { type: 'uniform', words: linParams },
          { type: 'read-only-storage', words: weights },
          { type: 'read-only-storage', words: input.map(f32ToBits) },
          {
            type: 'storage',
            words: initial.map(f32ToBits),
            expected: q4SmallMV2Ref(linParams, weights, input, initial, false),
          },
        ],
      });
      const fusedParams = [rows, cols, 0, 0, 0, 0, tokens, upOff];
      vectors.push({
        name: `q4-v2-swiglu-m${tokens}`,
        shader: 'Q4_GATE_UP_SWIGLU_SMALLM_V2',
        entryPoint: 'main',
        dispatch: [Math.ceil(rows / 64), 1, 1],
        q4SwigluSelfCheck: { rows, cols, upOff, tokens },
        bindings: [
          { type: 'uniform', words: fusedParams },
          { type: 'read-only-storage', words: weights },
          { type: 'read-only-storage', words: input.map(f32ToBits) },
          {
            type: 'storage',
            words: initial.map(f32ToBits),
            expectedLen: tokens * rows,
          },
        ],
      });
    }
    const stride = 89;
    const yOff = 11;
    const shardInput = Array.from({ length: 13 * cols }, () => Math.fround((rng.next() - 0.5) * 1.2));
    const shardInputF16Bits = shardInput.map(f32ToBits);
    vectors.push({
      name: 'q4-v2-f16-linear-m13',
      shader: 'Q4_GEMV_SMALLM_V2_F16',
      entryPoint: 'main',
      dispatch: [Math.ceil(rows / 16), 1, 1],
      q4LinearSelfCheck: { rows, cols, tokens: 13, residual: 0, stride: 0 },
      bindings: [
        { type: 'uniform', words: [rows, cols, 0, 0, 0, 0, 13, 0] },
        { type: 'read-only-storage', words: weights },
        { type: 'read-only-storage', words: shardInputF16Bits },
        { type: 'storage', words: Array(13 * rows).fill(0), expectedLen: 13 * rows },
      ],
    });
    vectors.push({
      name: 'q4-v2-f16-swiglu-m13',
      shader: 'Q4_GATE_UP_SWIGLU_SMALLM_V2_F16',
      entryPoint: 'main',
      dispatch: [Math.ceil(rows / 16), 1, 1],
      q4SwigluSelfCheck: { rows, cols, upOff, tokens: 13, rowsPerWg: 16 },
      bindings: [
        { type: 'uniform', words: [rows, cols, 0, 0, 0, 0, 13, upOff] },
        { type: 'read-only-storage', words: weights },
        { type: 'read-only-storage', words: shardInputF16Bits },
        { type: 'storage', words: Array(13 * rows).fill(0), expectedLen: 13 * rows },
      ],
    });
    const shardInit = Array.from({ length: 13 * stride }, () => -13.5);
    const shardParams = [rows, cols, 0, 0, yOff, 0, 13, stride];
    vectors.push({
      name: 'q4-v2-vocab-shard-m13',
      shader: 'Q4_GEMV_SMALLM_V2',
      entryPoint: 'main',
      dispatch: [Math.ceil(rows / 64), 1, 1],
      bindings: [
        { type: 'uniform', words: shardParams },
        { type: 'read-only-storage', words: weights },
        { type: 'read-only-storage', words: shardInput.map(f32ToBits) },
        {
          type: 'storage',
          words: shardInit.map(f32ToBits),
          expected: q4SmallMV2Ref(shardParams, weights, shardInput, shardInit, false),
        },
      ],
    });
  }

  // 2. DN_TREE_V2 (verify M=16 + commit)
  {
    const parents = [-1, 0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 11];
    const masks = masksOf(parents);
    const heads = 2;
    const dk = 32;
    const dv = 64;
    const slots = parents.length;
    const stride = 2 * dk + dv + 2;
    const rec = Array.from({ length: slots * heads * stride }, (_, i) => {
      const o = i % stride;
      if (o === stride - 2) return Math.fround(0.6 + 0.35 * rng.next());
      if (o === stride - 1) return Math.fround(0.2 + 0.7 * rng.next());
      return Math.fround((rng.next() - 0.5) * 0.5);
    });
    const initial = Array.from({ length: heads * dk * dv }, () => Math.fround((rng.next() - 0.5) * 0.05));
    const ref = dnTreeV2Ref([slots, heads, dk, dv], parents, rec, initial);
    const chosenSlot = 11;
    const chosen = [3, 99, 0, 0, chosenSlot, 11];
    vectors.push({
      name: 'dn-tree-v2-verify-m16',
      shader: 'DN_TREE_V2',
      entryPoint: 'verify',
      dispatch: [Math.ceil(dv / 32), heads, 1],
      bindings: [
        { type: 'uniform', words: [slots, heads, dk, dv] },
        { type: 'read-only-storage', words: masks },
        { type: 'read-only-storage', words: rec.map(f32ToBits) },
        { type: 'storage', words: initial.map(f32ToBits), expected: initial.map(f32ToBits) },
        { type: 'storage', words: Array(slots * heads * dv).fill(0), expected: ref.output },
        { type: 'read-only-storage', words: chosen },
      ],
    });
    vectors.push({
      name: 'dn-tree-v2-commit',
      shader: 'DN_TREE_V2',
      entryPoint: 'commit',
      dispatch: [Math.ceil(dv / 32), heads, 1],
      bindings: [
        { type: 'uniform', words: [slots, heads, dk, dv] },
        { type: 'read-only-storage', words: masks },
        { type: 'read-only-storage', words: rec.map(f32ToBits) },
        {
          type: 'storage',
          words: initial.map(f32ToBits),
          expected: Array.from(ref.states[chosenSlot], f32ToBits),
        },
        { type: 'storage', words: Array(slots * heads * dv).fill(0) },
        { type: 'read-only-storage', words: [1, 99, 0, 11, 0] },
      ],
    });
  }

  // 3. GQA_TREE_V2 (both fp16 and int8 across P=1 (L=512), P=4 (L=513), P=16 (L=4097): verify M=N tree slot == M=1 scalar on GPU!)
  const treeCases = [
    { name: 'gqa-v2-fp16-512', shader: 'GQA_TREE_V2', prefix: 500, parents: [-1, 0, 0, 1, 2, 2, 4, 6, 6, 8, 9, 10], int8: 0 },
    { name: 'gqa-v2-fp16-513', shader: 'GQA_TREE_V2', prefix: 511, parents: [-1, 0, 1, 0, 3, 2, 4, 6], int8: 0 },
    { name: 'gqa-v2-int8-512', shader: 'GQA_TREE_V2', prefix: 500, parents: [-1, 0, 0, 1, 2, 2, 4, 6, 6, 8, 9, 10], int8: 1 },
    { name: 'gqa-v2-int8-513', shader: 'GQA_TREE_V2', prefix: 511, parents: [-1, 0, 1, 0, 3, 2, 4, 6], int8: 1 },
    { name: 'gqa-v2-int8-4097', shader: 'GQA_TREE_V2', prefix: 4094, parents: [-1, 0, 1, 0, 3, 2, 4, 6], int8: 1 },
    { name: 'gqa-v2-sg-int8-512', shader: 'GQA_TREE_V2_SG', prefix: 500, parents: [-1, 0, 0, 1, 2, 2, 4, 6, 6, 8, 9, 10], int8: 1 },
    { name: 'gqa-v2-sg-int8-4097', shader: 'GQA_TREE_V2_SG', prefix: 4094, parents: [-1, 0, 1, 0, 3, 2, 4, 6], int8: 1 },
  ];
  for (const tc of treeCases) {
    const dim = 256;
    const qHeads = 4;
    const kvHeads = 1;
    const rotary = 64;
    const slots = tc.parents.length;
    const capacity = tc.prefix + slots;
    const masks = masksOf(tc.parents);
    const maxLen = Math.max(...masks.map((m) => tc.prefix + popcount(m)));
    const maxParts = partitionCount(maxLen);
    const words = tc.int8 ? dim / 4 + 1 : dim / 2;
    const cache = Array.from({ length: 2 * kvHeads * capacity * words }, (_, i) => {
      if (!tc.int8) return (f32ToF16((rng.next() - 0.5) * 1.5) | (f32ToF16((rng.next() - 0.5) * 1.5) << 16)) >>> 0;
      return i % words === words - 1
        ? (f32ToF16(1 / 256 + rng.next() / 64) | (f32ToF16(1 / 256 + rng.next() / 64) << 16)) >>> 0
        : rng.nextU32();
    });
    const stride = 512;
    const query = Array.from({ length: slots * qHeads * stride }, () => Math.fround((rng.next() - 0.5) * 1.2));
    const rope = ropeTableRef(tc.prefix + slots, rotary);
    const params = [tc.prefix, capacity, qHeads, kvHeads, dim, tc.int8, 0, stride, rotary, maxParts, 0, 0];
    const queryBits = query.map(f32ToBits);
    const ropeBits = rope.map(f32ToBits);
    vectors.push({
      name: tc.name,
      shader: tc.shader,
      twoPass: { maxParts, qHeads, slots },
      gqaSelfCheck: {
        prefix: tc.prefix,
        capacity,
        qHeads,
        kvHeads,
        dim,
        int8: tc.int8,
        stride,
        rotary,
        masks,
        words,
      },
      bindings: [
        { type: 'uniform', words: params },
        { type: 'read-only-storage', words: masks },
        { type: 'read-only-storage', words: queryBits },
        { type: 'read-only-storage', words: ropeBits },
        { type: 'read-only-storage', words: cache },
        { type: 'storage', words: Array(slots * qHeads * maxParts * (dim + 2)).fill(0) },
        { type: 'storage', words: Array(slots * qHeads * dim).fill(0), expectedLen: slots * qHeads * dim },
      ],
    });
  }

  return vectors;
}

async function main() {
  const vectors = buildTestVectors();
  const wasmBase64 = fs.readFileSync(WASM_PATH).toString('base64');
  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end('<!doctype html><html><body></body></html>');
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  const browser = await chromium.launch({
    headless: false,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded' });
    const results = await page.evaluate(
      async ({ vecs, wasmB64, shaderMap }) => {
        const binStr = atob(wasmB64);
        const bytes = new Uint8Array(binStr.length);
        for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
        const { instance } = await WebAssembly.instantiate(bytes, {});
        const exp = instance.exports;
        const mem = new Uint8Array(exp.memory.buffer);
        const dec = new TextDecoder();
        const sources = {};
        for (const [name, id] of Object.entries(shaderMap)) {
          const len = exp.wgsl_build_shader(id);
          const ptr = exp.wgsl_get_shader_ptr();
          sources[name] = dec.decode(mem.subarray(ptr, ptr + len));
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        const requiredFeatures = [];
        if (adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16');
        if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
        const device = await adapter.requestDevice({ requiredFeatures });
        const mods = new Map();
        const report = [];
        for (const vec of vecs) {
          let mod = mods.get(vec.shader);
          if (!mod) {
            mod = device.createShaderModule({ code: sources[vec.shader] });
            const info = await mod.getCompilationInfo();
            const errs = info.messages.filter((m) => m.type === 'error');
            if (errs.length > 0) throw new Error(`WGSL compile (${vec.shader}): ${JSON.stringify(errs)}`);
            mods.set(vec.shader, mod);
          }
          const bufs = [];
          const bglEntries = [];
          const bgEntries = [];
          for (let i = 0; i < vec.bindings.length; i++) {
            const b = vec.bindings[i];
            const byteSize = Math.max(16, b.words.length * 4);
            let usage = GPUBufferUsage.COPY_DST;
            let bufferType = 'storage';
            if (b.type === 'uniform') {
              usage |= GPUBufferUsage.UNIFORM;
              bufferType = 'uniform';
            } else if (b.type === 'read-only-storage') {
              usage |= GPUBufferUsage.STORAGE;
              bufferType = 'read-only-storage';
            } else {
              usage |= GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
              bufferType = 'storage';
            }
            const buf = device.createBuffer({ size: byteSize, usage });
            device.queue.writeBuffer(buf, 0, new Uint32Array(b.words));
            bufs.push(buf);
            bglEntries.push({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: bufferType } });
            bgEntries.push({ binding: i, resource: { buffer: buf } });
          }
          const bgl = device.createBindGroupLayout({ entries: bglEntries });
          const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const bg = device.createBindGroup({ layout: bgl, entries: bgEntries });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          if (vec.twoPass) {
            const p1 = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: 'partial' } });
            const p2 = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: 'combine' } });
            pass.setPipeline(p1);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(vec.twoPass.qHeads, vec.twoPass.slots, vec.twoPass.maxParts);
            pass.setPipeline(p2);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(vec.twoPass.qHeads, vec.twoPass.slots, 1);
          } else {
            const pipe = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: vec.entryPoint } });
            pass.setPipeline(pipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(vec.dispatch[0], vec.dispatch[1], vec.dispatch[2]);
          }
          pass.end();
          const rbs = [];
          for (let i = 0; i < vec.bindings.length; i++) {
            const b = vec.bindings[i];
            if (b.expected || b.expectedLen) {
              const wordsLen = b.expected ? b.expected.length : b.expectedLen;
              const st = device.createBuffer({ size: wordsLen * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
              enc.copyBufferToBuffer(bufs[i], 0, st, 0, wordsLen * 4);
              rbs.push({ i, st, expected: b.expected ?? null });
            }
          }
          device.queue.submit([enc.finish()]);
          let exactMismatches = 0;
          let maxAbsErr = 0;
          let firstMismatch = null;
          let gpuOutWords = null;
          for (const rb of rbs) {
            await rb.st.mapAsync(GPUMapMode.READ);
            const gotU = new Uint32Array(rb.st.getMappedRange().slice(0));
            const gotF = new Float32Array(gotU.buffer);
            rb.st.unmap();
            rb.st.destroy();
            if (!rb.expected) {
              gpuOutWords = gotU;
              continue;
            }
            const expU = new Uint32Array(rb.expected);
            const expF = new Float32Array(expU.buffer);
            for (let w = 0; w < expU.length; w++) {
              if (gotU[w] !== expU[w]) {
                exactMismatches++;
                const err = Number.isFinite(gotF[w]) && Number.isFinite(expF[w]) ? Math.abs(gotF[w] - expF[w]) : Infinity;
                if (err > maxAbsErr) maxAbsErr = err;
                if (!firstMismatch) {
                  firstMismatch = { word: w, gotU: gotU[w], expU: expU[w], gotF: gotF[w], expF: expF[w] };
                }
              }
            }
          }
          for (const buf of bufs) buf.destroy();
          if (vec.q4LinearSelfCheck && gpuOutWords) {
            const sc = vec.q4LinearSelfCheck;
            const gotF = new Float32Array(gpuOutWords.buffer);
            const wWords = new Uint32Array(vec.bindings[1].words);
            const inWords = new Uint32Array(vec.bindings[2].words);
            const pipe = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: 'main' } });
            for (let tok = 0; tok < sc.tokens; tok++) {
              const sParams = new Uint32Array([sc.rows, sc.cols, 0, tok * sc.cols, 0, 0, 1, 0]);
              const sBufDefs = [
                { data: sParams, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
                { data: wWords, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: inWords, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array(sc.rows), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
              ];
              const sBufs = sBufDefs.map((d) => {
                const b = device.createBuffer({ size: Math.max(16, d.data.byteLength), usage: d.usage });
                device.queue.writeBuffer(b, 0, d.data);
                return b;
              });
              const sbg = device.createBindGroup({
                layout: bgl,
                entries: sBufs.map((b, idx) => ({ binding: idx, resource: { buffer: b } })),
              });
              const senc = device.createCommandEncoder();
              const spass = senc.beginComputePass();
              spass.setPipeline(pipe);
              spass.setBindGroup(0, sbg);
              spass.dispatchWorkgroups(Math.ceil(sc.rows / 16), 1, 1);
              spass.end();
              const st = device.createBuffer({ size: sc.rows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
              senc.copyBufferToBuffer(sBufs[3], 0, st, 0, sc.rows * 4);
              device.queue.submit([senc.finish()]);
              await st.mapAsync(GPUMapMode.READ);
              const expU = new Uint32Array(st.getMappedRange().slice(0));
              const expF = new Float32Array(expU.buffer);
              st.unmap();
              st.destroy();
              for (const sb of sBufs) sb.destroy();
              for (let r = 0; r < sc.rows; r++) {
                const wIdx = tok * sc.rows + r;
                if (gpuOutWords[wIdx] !== expU[r]) {
                  exactMismatches++;
                  const err = Math.abs(gotF[wIdx] - expF[r]);
                  if (err > maxAbsErr) maxAbsErr = err;
                  if (!firstMismatch) {
                    firstMismatch = { token: tok, row: r, gotU: gpuOutWords[wIdx], expU: expU[r], gotF: gotF[wIdx], expF: expF[r] };
                  }
                }
              }
            }
          }
          if (vec.q4SwigluSelfCheck && gpuOutWords) {
            const sc = vec.q4SwigluSelfCheck;
            const gotF = new Float32Array(gpuOutWords.buffer);
            const wWords = new Uint32Array(vec.bindings[1].words);
            const inWords = new Uint32Array(vec.bindings[2].words);
            const pipe = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: 'main' } });
            for (let tok = 0; tok < sc.tokens; tok++) {
              const sParams = new Uint32Array([sc.rows, sc.cols, 0, tok * sc.cols, 0, 0, 1, sc.upOff]);
              const sBufDefs = [
                { data: sParams, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
                { data: wWords, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: inWords, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array(sc.rows), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
              ];
              const sBufs = sBufDefs.map((d) => {
                const b = device.createBuffer({ size: Math.max(16, d.data.byteLength), usage: d.usage });
                device.queue.writeBuffer(b, 0, d.data);
                return b;
              });
              const sbg = device.createBindGroup({
                layout: bgl,
                entries: sBufs.map((b, idx) => ({ binding: idx, resource: { buffer: b } })),
              });
              const senc = device.createCommandEncoder();
              const spass = senc.beginComputePass();
              spass.setPipeline(pipe);
              spass.setBindGroup(0, sbg);
              spass.dispatchWorkgroups(Math.ceil(sc.rows / (sc.rowsPerWg ?? 64)), 1, 1);
              spass.end();
              const st = device.createBuffer({ size: sc.rows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
              senc.copyBufferToBuffer(sBufs[3], 0, st, 0, sc.rows * 4);
              device.queue.submit([senc.finish()]);
              await st.mapAsync(GPUMapMode.READ);
              const expU = new Uint32Array(st.getMappedRange().slice(0));
              const expF = new Float32Array(expU.buffer);
              st.unmap();
              st.destroy();
              for (const sb of sBufs) sb.destroy();
              for (let r = 0; r < sc.rows; r++) {
                const wIdx = tok * sc.rows + r;
                if (gpuOutWords[wIdx] !== expU[r]) {
                  exactMismatches++;
                  const err = Math.abs(gotF[wIdx] - expF[r]);
                  if (err > maxAbsErr) maxAbsErr = err;
                  if (!firstMismatch) {
                    firstMismatch = { token: tok, row: r, gotU: gpuOutWords[wIdx], expU: expU[r], gotF: gotF[wIdx], expF: expF[r] };
                  }
                }
              }
            }
          }
          if (vec.gqaSelfCheck && gpuOutWords) {
            const scMeta = vec.gqaSelfCheck;
            const gotF = new Float32Array(gpuOutWords.buffer);
            const queryWords = vec.bindings[2].words;
            const ropeWords = vec.bindings[3].words;
            const cacheWords = vec.bindings[4].words;
            const partCountFn = (l) => (l <= 64 ? 1 : l <= 512 ? 8 : l <= 4096 ? 16 : 32);
            const popcnt = (x) => {
              let c = 0;
              for (let v = x >>> 0; v; v >>>= 1) c += v & 1;
              return c;
            };
            const p1 = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: 'partial' } });
            const p2 = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: 'combine' } });
            for (let slot = 0; slot < scMeta.masks.length; slot++) {
              const mask = scMeta.masks[slot];
              const len = scMeta.prefix + popcnt(mask);
              const sCap = len;
              const sCache = new Uint32Array(2 * scMeta.kvHeads * sCap * scMeta.words);
              const draftPhys = [];
              for (let s = 0; s <= slot; s++) if (mask & (1 << s)) draftPhys.push(scMeta.prefix + s);
              for (let h = 0; h < scMeta.kvHeads; h++) {
                for (let logIdx = 0; logIdx < len; logIdx++) {
                  const t = logIdx < scMeta.prefix ? logIdx : draftPhys[logIdx - scMeta.prefix];
                  const srcK = (h * scMeta.capacity + t) * scMeta.words;
                  const srcV = srcK + scMeta.kvHeads * scMeta.capacity * scMeta.words;
                  const dstK = (h * sCap + logIdx) * scMeta.words;
                  const dstV = dstK + scMeta.kvHeads * sCap * scMeta.words;
                  for (let w = 0; w < scMeta.words; w++) {
                    sCache[dstK + w] = cacheWords[srcK + w];
                    sCache[dstV + w] = cacheWords[srcV + w];
                  }
                }
              }
              const sParts = partCountFn(len);
              const sQuery = queryWords.slice(slot * scMeta.qHeads * scMeta.stride, (slot + 1) * scMeta.qHeads * scMeta.stride);
              const sParams = new Uint32Array([len - 1, sCap, scMeta.qHeads, scMeta.kvHeads, scMeta.dim, scMeta.int8, 0, scMeta.stride, scMeta.rotary, sParts, 0, 0]);
              const sBufDefs = [
                { data: sParams, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array([1]), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array(sQuery), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array(ropeWords), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: sCache, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array(scMeta.qHeads * sParts * (scMeta.dim + 2)), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
                { data: new Uint32Array(scMeta.qHeads * scMeta.dim), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
              ];
              const sBufs = sBufDefs.map((d) => {
                const b = device.createBuffer({ size: Math.max(16, d.data.byteLength), usage: d.usage });
                device.queue.writeBuffer(b, 0, d.data);
                return b;
              });
              const sbg = device.createBindGroup({
                layout: bgl,
                entries: sBufs.map((b, idx) => ({ binding: idx, resource: { buffer: b } })),
              });
              const senc = device.createCommandEncoder();
              const spass = senc.beginComputePass();
              spass.setPipeline(p1);
              spass.setBindGroup(0, sbg);
              spass.dispatchWorkgroups(scMeta.qHeads, 1, sParts);
              spass.setPipeline(p2);
              spass.setBindGroup(0, sbg);
              spass.dispatchWorkgroups(scMeta.qHeads, 1, 1);
              spass.end();
              const outLen = scMeta.qHeads * scMeta.dim;
              const st = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
              senc.copyBufferToBuffer(sBufs[6], 0, st, 0, outLen * 4);
              device.queue.submit([senc.finish()]);
              await st.mapAsync(GPUMapMode.READ);
              const expU = new Uint32Array(st.getMappedRange().slice(0));
              const expF = new Float32Array(expU.buffer);
              st.unmap();
              st.destroy();
              for (const sb of sBufs) sb.destroy();
              const baseWord = slot * outLen;
              for (let w = 0; w < outLen; w++) {
                const gw = gpuOutWords[baseWord + w];
                const ew = expU[w];
                if (gw !== ew) {
                  exactMismatches++;
                  const err = Math.abs(gotF[baseWord + w] - expF[w]);
                  if (err > maxAbsErr) maxAbsErr = err;
                  if (!firstMismatch) {
                    firstMismatch = { slot, word: w, gotU: gw, expU: ew, gotF: gotF[baseWord + w], expF: expF[w] };
                  }
                }
              }
            }
          }
          report.push({ name: vec.name, shader: vec.shader, exactMismatches, maxAbsErr, firstMismatch });
        }
        return report;
      },
      { vecs: vectors, wasmB64: wasmBase64, shaderMap: SHADER_IDS },
    );
    console.table(results);
    const failed = results.filter((r) => r.exactMismatches > 0);
    if (failed.length > 0) {
      console.error('RC1 V2 kernel parity failures:', JSON.stringify(failed, null, 2));
      process.exitCode = 1;
    } else {
      console.log(`ALL ${results.length}/${results.length} RC1 V2 kernel vectors PASSED (0 mismatches, bit-exact u32).`);
    }
  } finally {
    await browser.close();
    await new Promise((r) => server.close(r));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
