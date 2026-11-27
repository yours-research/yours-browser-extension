/**
 * f32-exact CPU oracle for wgsl/gqa_tree_v2.wgsl, plus golden-vector builders.
 * Operation order mirrors the shader: pass 1 writes one [peak, mass, acc] record
 * per (slot, query head, partition); pass 2 merges records in ascending order.
 */
import { add, div, expNeg, f16ToF32, f32ToBits, mul, rsqrt, sub } from './f32.js';
import type { KernelVector } from './golden.js';
import { ancestorMasks } from './reference.js';

/** Peak of a record that has seen no position (WGSL -3.40282346e38). */
export const GQA_V2_EMPTY_PEAK = Math.fround(-3.40282346e38);

/** Number of partitions for a logical history of `length` positions. */
export function partitionCount(length: number): number {
  if (length <= 512) return 1;
  if (length <= 4096) return 4;
  return 16;
}

/**
 * Rotary table indexed by logical position: cos at (pos*(rotary/2)+i)*2 and sin
 * at the next word. The kernel treats it as input data; only f32 values matter.
 * rotary == 0 yields the one-element dummy binding.
 */
export function ropeTable(positions: number, rotary: number, theta = 10000): number[] {
  if (rotary === 0) return [0];
  const half = rotary / 2;
  const table = new Array<number>(positions * half * 2);
  for (let pos = 0; pos < positions; pos++)
    for (let i = 0; i < half; i++) {
      const angle = pos / theta ** ((2 * i) / rotary);
      table[(pos * half + i) * 2] = Math.fround(Math.cos(angle));
      table[(pos * half + i) * 2 + 1] = Math.fround(Math.sin(angle));
    }
  return table;
}

export interface GqaTreeV2Case {
  name?: string;
  /** [prefix, capacity, q_heads, kv_heads, dim, int8, base, q_stride, rotary, max_parts] */
  params: number[];
  /** Inclusive ancestor masks, one per slot; or a parents array starting with -1. */
  masks: number[];
  query: ArrayLike<number>;
  rope: ArrayLike<number>;
  cache: ArrayLike<number>;
}

export interface GqaTreeV2Buffers {
  /** f32 values, length slots*q_heads*max_parts*(dim+2). */
  partials: number[];
  /** f32 values, length slots*q_heads*dim. */
  out: number[];
}

function resolveMasks(masksOrParents: number[]): number[] {
  const masks = masksOrParents[0] === -1 ? ancestorMasks(masksOrParents) : masksOrParents;
  if (!masks.length || masks.length > 16) throw new Error('expected 1 to 16 slots');
  masks.forEach((mask, slot) => {
    if (!Number.isInteger(mask) || mask < 0 || mask >= 2 ** (slot + 1))
      throw new Error('mask of slot ' + slot + ' names a later slot');
  });
  return masks;
}

function popcount(mask: number): number {
  let count = 0;
  for (let m = mask; m; m >>>= 1) count += m & 1;
  return count;
}

function shape(params: number[], masks: number[], rope: ArrayLike<number>) {
  const [prefix, capacity, qHeads, kvHeads, dim, int8, , , rotary, maxParts] = params;
  const slots = masks.length;
  if (dim < 1 || dim > 256 || dim % (int8 ? 4 : 2)) throw new Error('unsupported dim');
  if (qHeads % kvHeads) throw new Error('q_heads must be a multiple of kv_heads');
  if (capacity < prefix + slots) throw new Error('cache capacity too small');
  if (rotary % 2 || rotary > dim) throw new Error('rotary must be even and at most dim');
  for (const mask of masks) {
    const length = prefix + popcount(mask);
    if (maxParts < partitionCount(length)) throw new Error('max_parts too small for L=' + length);
    if (rotary && (length < 1 || rope.length < length * rotary)) throw new Error('rope table too small');
  }
  return {
    slots,
    partialsLength: slots * qHeads * maxParts * (dim + 2),
    outLength: slots * qHeads * dim,
  };
}

/** Pass 1. Writes the records of the active partitions into `partials`. */
function partialPass(
  params: number[],
  masks: number[],
  query: ArrayLike<number>,
  rope: ArrayLike<number>,
  cache: ArrayLike<number>,
  partials: number[],
) {
  const [prefix, capacity, qHeads, kvHeads, dim, int8, base, stride, rotary, maxParts] = params;
  const words = int8 ? dim / 4 + 1 : dim / 2;
  const half = rotary / 2;
  const inverseRoot = rsqrt(dim);
  const q = new Float32Array(dim);
  const k = new Float32Array(dim);
  const v = new Float32Array(dim);
  const acc = new Float32Array(dim);
  const lanes = new Float32Array(64);

  function load(target: Float32Array, offset: number, scale: number) {
    for (let d = 0; d < dim; d++) {
      if (!int8) target[d] = f16ToF32((cache[offset + (d >>> 1)] >>> ((d % 2) * 16)) & 65535);
      else target[d] = mul((cache[offset + (d >>> 2)] << (24 - (d % 4) * 8)) >> 24, scale);
    }
  }

  for (let slot = 0; slot < masks.length; slot++) {
    const mask = masks[slot];
    const length = prefix + popcount(mask);
    // Physical positions of this slot's draft history, chronological.
    const draft: number[] = [];
    for (let s = 0; s <= slot; s++) if (mask & (1 << s)) draft.push(prefix + s);
    const count = partitionCount(length);
    const chunk = Math.ceil(length / count);
    for (let head = 0; head < qHeads; head++) {
      const row = slot * qHeads + head;
      const kvHead = Math.floor(head / (qHeads / kvHeads));
      const queryBase = row * stride;
      for (let d = 0; d < dim; d++) {
        let value = query[queryBase + d];
        if (d < half) {
          const angle = ((length - 1) * half + d) * 2;
          value = sub(mul(value, rope[angle]), mul(query[queryBase + d + half], rope[angle + 1]));
        } else if (d < rotary) {
          const i = d - half;
          const angle = ((length - 1) * half + i) * 2;
          value = add(mul(value, rope[angle]), mul(query[queryBase + i], rope[angle + 1]));
        }
        q[d] = value;
      }
      for (let part = 0; part < count; part++) {
        const lo = part * chunk;
        const hi = Math.min(length, lo + chunk);
        acc.fill(0);
        let peak = GQA_V2_EMPTY_PEAK,
          mass = 0;
        for (let logical = lo; logical < hi; logical++) {
          const t = logical < prefix ? logical : draft[logical - prefix];
          const kb = base + (kvHead * capacity + t) * words;
          const vb = kb + kvHeads * capacity * words;
          const scales = int8 ? cache[kb + words - 1] : 0;
          load(k, kb, int8 ? f16ToF32(scales & 65535) : 1);
          lanes.fill(0);
          for (let d = 0; d < dim; d++) lanes[d % 64] = add(lanes[d % 64], mul(q[d], k[d]));
          for (let span = 32; span; span >>= 1)
            for (let i = 0; i < span; i++) lanes[i] = add(lanes[i], lanes[i + span]);
          const score = mul(lanes[0], inverseRoot);
          const high = Math.max(peak, score);
          const a = expNeg(sub(peak, high)),
            b = expNeg(sub(score, high));
          mass = add(mul(mass, a), b);
          load(v, vb, int8 ? f16ToF32(scales >>> 16) : 1);
          for (let d = 0; d < dim; d++) acc[d] = add(mul(acc[d], a), mul(v[d], b));
          peak = high;
        }
        const record = (row * maxParts + part) * (dim + 2);
        partials[record] = peak;
        partials[record + 1] = mass;
        for (let d = 0; d < dim; d++) partials[record + 2 + d] = acc[d];
      }
    }
  }
}

/** Pass 2. Merges each row's active records and writes the normalized output. */
function combinePass(params: number[], masks: number[], partials: number[], out: number[]) {
  const [prefix, , qHeads, , dim, , , , , maxParts] = params;
  const acc = new Float32Array(dim);
  for (let slot = 0; slot < masks.length; slot++) {
    const count = partitionCount(prefix + popcount(masks[slot]));
    for (let head = 0; head < qHeads; head++) {
      const row = slot * qHeads + head;
      acc.fill(0);
      let peak = GQA_V2_EMPTY_PEAK,
        mass = 0;
      for (let part = 0; part < count; part++) {
        const record = (row * maxParts + part) * (dim + 2);
        const partPeak = partials[record],
          partMass = partials[record + 1];
        const high = Math.max(peak, partPeak);
        const a = expNeg(sub(peak, high)),
          b = expNeg(sub(partPeak, high));
        mass = add(mul(mass, a), mul(partMass, b));
        for (let d = 0; d < dim; d++) acc[d] = add(mul(acc[d], a), mul(partials[record + 2 + d], b));
        peak = high;
      }
      for (let d = 0; d < dim; d++) out[row * dim + d] = mul(acc[d], div(1, mass));
    }
  }
}

/**
 * Runs both passes. `initial` supplies the prior contents of the writable
 * buffers (default zeros); words the shader leaves untouched keep those values.
 * Values are f32; compare through f32ToBits.
 */
export function gqaTreeV2(
  params: number[],
  masksOrParents: number[],
  query: ArrayLike<number>,
  rope: ArrayLike<number>,
  cache: ArrayLike<number>,
  initial: Partial<GqaTreeV2Buffers> = {},
): GqaTreeV2Buffers {
  const masks = resolveMasks(masksOrParents);
  const { partialsLength, outLength } = shape(params, masks, rope);
  const partials = initial.partials?.slice() ?? new Array<number>(partialsLength).fill(0);
  const out = initial.out?.slice() ?? new Array<number>(outLength).fill(0);
  if (partials.length !== partialsLength || out.length !== outLength)
    throw new Error('initial buffer length mismatch');
  // The shader sees these inputs as binary32.
  partialPass(params, masks, Float32Array.from(query), Float32Array.from(rope), cache, partials);
  combinePass(params, masks, partials, out);
  return { partials, out };
}

export interface GqaTreeV2VectorOptions {
  /** Attach oracle output words as `expected` (default true). */
  expected?: boolean;
  /** f32 value pre-filled into partials and out, exposing stray writes (default 0). */
  fill?: number;
}

/**
 * The two dispatches of one case, sharing one bind-group layout. With
 * `expected`, the combine vector's input partials are the oracle's pass-1
 * result, so each vector replays on its own. Without it, copy the partial
 * dispatch's binding 5 readback into `combine.bindings[5].words` before running.
 */
export function gqaTreeV2Vectors(
  testCase: GqaTreeV2Case,
  options: GqaTreeV2VectorOptions = {},
): { partial: KernelVector; combine: KernelVector; oracle?: GqaTreeV2Buffers } {
  const { params, query, rope, cache, name = 'gqa-v2' } = testCase;
  const masks = resolveMasks(testCase.masks);
  const { partialsLength, outLength } = shape(params, masks, rope);
  const [, , qHeads, , , , , , , maxParts] = params;
  const fill = options.fill ?? 0;
  const initial = {
    partials: new Array<number>(partialsLength).fill(fill),
    out: new Array<number>(outLength).fill(fill),
  };
  const oracle =
    (options.expected ?? true) ? gqaTreeV2(params, masks, query, rope, cache, initial) : undefined;
  const bits = (values: ArrayLike<number>) => Array.from(values, f32ToBits);
  const initialPartials = bits(initial.partials),
    initialOut = bits(initial.out);
  const expectedPartials = oracle && bits(oracle.partials);
  const uniform = [...params.slice(0, 10), 0, 0];
  const inputs = [bits(query), bits(rope), Array.from(cache)];
  function vector(
    entryPoint: 'partial' | 'combine',
    dispatch: KernelVector['dispatch'],
    partials: number[],
    expected?: [number[], number[]],
  ): KernelVector {
    return {
      name: name + '-' + entryPoint,
      shader: 'gqa_tree_v2',
      entryPoint,
      dispatch,
      bindings: [
        { type: 'uniform', words: uniform },
        { type: 'read-only-storage', words: masks },
        ...inputs.map((words) => ({ type: 'read-only-storage' as const, words })),
        { type: 'storage', words: partials, expected: expected?.[0] },
        { type: 'storage', words: initialOut, expected: expected?.[1] },
      ],
    };
  }
  return {
    partial: vector(
      'partial',
      [qHeads, masks.length, maxParts],
      initialPartials,
      expectedPartials && [expectedPartials, initialOut],
    ),
    combine: vector(
      'combine',
      [qHeads, masks.length, 1],
      expectedPartials ?? initialPartials,
      expectedPartials && [expectedPartials, bits(oracle!.out)],
    ),
    oracle,
  };
}
