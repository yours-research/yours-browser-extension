import {
  add,
  bitsToF32,
  div,
  expNeg,
  f16ToF32,
  f32ToBits,
  invSqrt,
  logOnePlus,
  mul,
  rsqrt,
  stage,
  sub,
} from './f32.js';

export function q4SmallM(params: number[], weights: number[], input: number[], initial: number[]) {
  const [rows, cols, wo, xo, yo, residual, tokens] = params;
  const out = initial.slice();
  for (let t = 0; t < tokens; t++)
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) {
        const b = wo + 5 * ((r * cols) / 32 + (c >>> 5));
        const affine = weights[b + 4];
        const nibble = (weights[b + ((c & 31) >>> 3)] >>> ((c & 7) * 4)) & 15;
        const w = add(mul(nibble, f16ToF32(affine & 65535)), f16ToF32(affine >>> 16));
        sum = add(sum, mul(w, input[xo + t * cols + c]));
      }
      const dst = yo + t * rows + r;
      out[dst] = residual ? add(sum, initial[dst]) : sum;
    }
  return out;
}

/** Independent sequential oracle. Materialize each parent's outgoing state,
 * unlike WGSL which rescans a path and keeps a single column in registers. */
export function deltaTree(params: number[], parents: number[], rec: number[], initial: number[]) {
  const [slots, heads, dk, dv] = params;
  const states: number[][] = [];
  const output: number[] = [];
  for (let slot = 0; slot < slots; slot++) {
    const state = (slot === 0 ? initial : states[parents[slot]]).slice();
    for (let h = 0; h < heads; h++) {
      const q = (slot * heads + h) * (2 * dk + dv + 2);
      const k = q + dk,
        v = k + dk;
      for (let j = 0; j < dv; j++) {
        let prediction = 0,
          readout = 0;
        for (let i = 0; i < dk; i++) {
          const a = (h * dk + i) * dv + j;
          state[a] = mul(state[a], rec[v + dv]);
          prediction = add(prediction, mul(rec[k + i], state[a]));
        }
        const error = mul(sub(rec[v + j], prediction), rec[v + dv + 1]);
        for (let i = 0; i < dk; i++) {
          const a = (h * dk + i) * dv + j;
          state[a] = add(state[a], mul(rec[k + i], error));
          readout = add(readout, mul(rec[q + i], state[a]));
        }
        output.push(readout);
      }
    }
    states.push(state);
  }
  return { states, output };
}

export function convTree(parents: number[], raw: number[], weights: number[], initial: number[]) {
  const channels = initial.length / 3;
  const histories: number[][] = [],
    output: number[] = [];
  for (let slot = 0; slot < parents.length; slot++) {
    const incoming = slot === 0 ? initial : histories[parents[slot]];
    const history: number[] = [];
    for (let c = 0; c < channels; c++) {
      const taps = [...incoming.slice(3 * c, 3 * c + 3), raw[slot * channels + c]];
      let u = 0;
      for (let i = 0; i < 4; i++) u = add(u, mul(taps[i], weights[4 * c + i]));
      const e = expNeg(-Math.abs(u));
      output.push(mul(u < 0 ? mul(u, e) : u, div(1, add(1, e))));
      history.push(...taps.slice(1));
    }
    histories.push(history);
  }
  return { histories, output };
}

export function ancestorMasks(parents: number[]) {
  if (!parents.length || parents.length > 16 || parents[0] !== -1)
    throw new Error('expected root and at most 16 slots');
  const masks: number[] = [];
  parents.forEach((p, i) => {
    if (!Number.isInteger(p) || (i > 0 && (p < 0 || p >= i))) throw new Error('bad parent');
    masks.push((1 << i) | (i ? masks[p] : 0));
  });
  return masks;
}

export function gqaTree(p: number[], parents: number[], query: number[], cache: number[]) {
  const [prefix, capacity, heads, kvHeads, dim, int8, base, stride] = p;
  const words = int8 ? dim / 4 + 1 : dim / 2;
  const output: number[] = [];
  for (let slot = 0; slot < parents.length; slot++) {
    const path: number[] = [];
    for (let s = slot; s >= 0; s = parents[s]) path.unshift(prefix + s);
    const visible = [...Array.from({ length: prefix }, (_, i) => i), ...path];
    for (let h = 0; h < heads; h++) {
      const acc = new Array<number>(dim).fill(0);
      let peak = -Infinity,
        mass = 0;
      for (const pos of visible) {
        const kb = base + (Math.floor((h * kvHeads) / heads) * capacity + pos) * words;
        const vb = kb + kvHeads * capacity * words;
        function load(b: number, d: number) {
          if (!int8) return f16ToF32((cache[b + (d >>> 1)] >>> ((d % 2) * 16)) & 65535);
          const byte = (cache[b + (d >>> 2)] >>> ((d % 4) * 8)) & 255;
          const scales = cache[kb + words - 1];
          return mul(byte >= 128 ? byte - 256 : byte, f16ToF32(b === kb ? scales & 65535 : scales >>> 16));
        }
        const partial = new Array<number>(64).fill(0);
        for (let d = 0; d < dim; d++) {
          partial[d % 64] = add(partial[d % 64], mul(query[(slot * heads + h) * stride + d], load(kb, d)));
        }
        for (let span = 32; span; span >>= 1)
          for (let i = 0; i < span; i++) partial[i] = add(partial[i], partial[i + span]);
        const score = mul(partial[0], rsqrt(dim));
        const next = Math.max(peak, score);
        const a = expNeg(sub(peak, next)),
          b = expNeg(sub(score, next));
        mass = add(mul(mass, a), b);
        for (let d = 0; d < dim; d++) acc[d] = add(mul(acc[d], a), mul(load(vb, d), b));
        peak = next;
      }
      output.push(...acc.map((v) => mul(v, div(1, mass))));
    }
  }
  return output;
}

export function compactKV(p: number[], planes: number, slots: number[], cache: number[]) {
  const [prefix, capacity, words, base] = p;
  const out = cache.slice();
  for (let h = 0; h < planes; h++)
    slots.forEach((src, dst) => {
      const start = base + (h * capacity + prefix) * words;
      for (let j = 0; j < words; j++) out[start + dst * words + j] = cache[start + src * words + j];
    });
  return out;
}

export function rankBatch(p: number[], bits: number[], argmax = false) {
  const [vocab, stride, k, offset] = p;
  const rows = Math.floor((bits.length - offset) / stride);
  const out: number[] = [];
  for (let r = 0; r < rows; r++) {
    const start = offset + r * stride;
    const order = Array.from({ length: vocab }, (_, i) => i);
    order.sort((a, b) => bitsToF32(bits[start + b]) - bitsToF32(bits[start + a]) || a - b);
    if (argmax) out.push(order[0]);
    else for (const id of order.slice(0, k)) out.push(id, bits[start + id]);
  }
  return out;
}

export function greedyChoice(parents: number[], tokens: number[], best: number[]) {
  const path = [0];
  while (true) {
    const leaf = path[path.length - 1];
    const child = parents.findIndex((p, i) => p === leaf && tokens[i] === best[leaf]);
    if (child < 0) return [path.length - 1, best[leaf], ...path];
    path.push(child);
  }
}

export const toBits = (values: number[]) => values.map(f32ToBits);

export function deltaIO(p: number[], qkv: number[], ab: number[], aux: number[], y: number[], z: number[]) {
  const [heads, keyHeads, dim, slots] = p;
  const packed: number[] = [],
    output: number[] = [];
  const sigmoid = (v: number) => {
    const e = expNeg(-Math.abs(v));
    return mul(v < 0 ? e : 1, div(1, add(1, e)));
  };
  function squares(values: number[]) {
    const partial = Array<number>(128).fill(0);
    values.forEach((v, i) => {
      partial[i] = mul(v, v);
    });
    for (let span = 64; span; span >>= 1)
      for (let j = 0; j < span; j++) partial[j] = add(partial[j], partial[j + span]);
    return partial[0];
  }
  for (let s = 0; s < slots; s++)
    for (let h = 0; h < heads; h++) {
      const base = s * (2 * keyHeads + heads) * dim;
      const key = base + Math.floor((h * keyHeads) / heads) * dim;
      const q = qkv.slice(key, key + dim),
        k = qkv.slice(key + keyHeads * dim, key + (keyHeads + 1) * dim);
      const epsilon = Math.fround(1e-6);
      const iq = mul(invSqrt(add(squares(q), epsilon)), invSqrt(dim)),
        ik = invSqrt(add(squares(k), epsilon));
      const value = base + (2 * keyHeads + h) * dim;
      const b = s * 2 * heads + h;
      const u = add(ab[b + heads], aux[heads + h]);
      const softplus = add(Math.max(u, 0), logOnePlus(expNeg(-Math.abs(u))));
      packed.push(
        ...q.map((v) => mul(v, iq)),
        ...k.map((v) => mul(v, ik)),
        ...qkv.slice(value, value + dim),
        expNeg(mul(aux[h], softplus)),
        sigmoid(ab[b]),
      );
      const pos = (s * heads + h) * dim;
      const inv = invSqrt(add(div(squares(y.slice(pos, pos + dim)), dim), epsilon));
      for (let j = 0; j < dim; j++) {
        const normalized = stage(mul(y[pos + j], inv));
        const silu = stage(mul(z[pos + j], stage(sigmoid(z[pos + j]))));
        output.push(mul(stage(mul(normalized, silu)), aux[2 * heads + j]));
      }
    }
  return { packed, output };
}
