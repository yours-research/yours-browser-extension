import { add, div, f16ToF32, invSqrt, mul, stage } from '../kernels/f32.js';
import { convTree, deltaIO, deltaTree } from '../kernels/reference.js';
import { q4SmallMV2 } from '../kernels/q4SmallMV2.js';
import type { GdnBundle } from './gdnEagle3Loader.js';

export interface GdnState {
  matrix: number[];
  history: number[];
}
export interface GdnStepResult extends GdnState {
  feature: number[];
  logits: number[];
}

export function gdnRms(x: ArrayLike<number>, weights: ArrayLike<number>, offset = 0): number[] {
  if (!x.length || weights.length < offset + x.length) throw new Error('bad RMSNorm shape');
  const partial = new Array<number>(64).fill(0);
  for (let i = 0; i < x.length; i++) {
    const lane = i % 64;
    partial[lane] = add(partial[lane], stage(mul(x[i], x[i])));
  }
  for (let span = 32; span > 0; span >>= 1) {
    for (let i = 0; i < span; i++) partial[i] = add(partial[i], partial[i + span]);
  }
  const inv = stage(invSqrt(add(stage(div(partial[0], x.length)), Math.fround(1e-6))));
  return Array.from(x, (value, i) => mul(stage(mul(value, inv)), weights[offset + i]));
}

function tableValue(
  table: ArrayLike<number>,
  hidden: number,
  token: number,
  d: number,
  offset: number,
): number {
  const at = offset + (token * Math.ceil(hidden / 32) + (d >>> 5)) * 5;
  const q = (table[at + ((d & 31) >>> 3)] >>> (4 * (d & 7))) & 15;
  return add(mul(q, f16ToF32(table[at + 4] & 65535)), f16ToF32(table[at + 4] >>> 16));
}

export function gdnEmbed(table: ArrayLike<number>, feature: ArrayLike<number>, token: number, offset = 0) {
  return Array.from(feature, (x, d) => add(x, stage(tableValue(table, feature.length, token, d, offset))));
}

export function gdnSubsetLogits(
  table: ArrayLike<number>,
  x: ArrayLike<number>,
  subset: ArrayLike<number>,
  offset = 0,
) {
  return Array.from(subset, (token) => {
    let sum = 0;
    for (let d = 0; d < x.length; d++) {
      sum = add(sum, stage(mul(tableValue(table, x.length, token, d, offset), x[d])));
    }
    return sum;
  });
}

function linear(bundle: GdnBundle, name: string, x: ArrayLike<number>, residual?: ArrayLike<number>) {
  const [rows, cols] = bundle.tensor(name).shape;
  if (x.length !== cols || (residual && residual.length !== rows)) throw new Error('bad input for ' + name);
  const params = [rows, cols, 0, 0, 0, residual ? 1 : 0, 1, 0];
  return Array.from(q4SmallMV2(params, bundle.words(name), x, residual ?? new Float32Array(rows)));
}

/** Target features from layers 3,15,31 concatenated in exporter order. */
export function gdnFuse(bundle: GdnBundle, features: ArrayLike<number>) {
  return linear(bundle, 'fusion.weight', features);
}

/** One immutable recurrent step. A sibling can reuse the same prior state.
 * Only Q4 projection weights are quantized; activations/state use binary32. */
export function gdnStep(
  bundle: GdnBundle,
  tableWords: ArrayLike<number>,
  feature: ArrayLike<number>,
  token: number,
  prior?: GdnState,
): GdnStepResult {
  const { hidden, heads, dim, intermediate, vocab } = bundle.config;
  const channels = 3 * heads * dim;
  if (
    !Number.isInteger(token) ||
    token < 0 ||
    token >= vocab ||
    feature.length !== hidden ||
    tableWords.length < vocab * Math.ceil(hidden / 32) * 5
  )
    throw new Error('invalid draft token, feature or table');
  const state = prior ?? {
    matrix: Array<number>(heads * dim * dim).fill(0),
    history: Array<number>(channels * 3).fill(0),
  };
  if (state.matrix.length !== heads * dim * dim || state.history.length !== channels * 3) {
    throw new Error('invalid incoming GDN state');
  }
  const mixed = gdnEmbed(tableWords, feature, token);
  const x = gdnRms(mixed, bundle.floats('input_norm.weight'));
  const raw = linear(bundle, 'qkv.weight', x);
  const z = linear(bundle, 'z.weight', x);
  const ab = linear(bundle, 'ab.weight', x);
  const aux = [
    ...Array.from(bundle.floats('log_rate'), (u) => Math.fround(-Math.exp(u))),
    ...bundle.floats('dt_bias'),
    ...bundle.floats('head_norm.weight'),
  ];
  const conv = convTree([-1], raw, Array.from(bundle.floats('conv')), state.history);
  const io = [heads, heads, dim, 1];
  const { packed } = deltaIO(io, conv.output, ab, aux, Array<number>(z.length).fill(0), z);
  const dn = deltaTree([1, heads, dim, dim], [-1], packed, state.matrix);
  const { output } = deltaIO(io, conv.output, ab, aux, dn.output, z);
  const postDn = linear(bundle, 'out.weight', output, mixed);
  const ff = gdnRms(postDn, bundle.floats('ff_norm.weight'));

  // Exported tensors are views on one validated blob. The fused shader reads
  // both matrices at their own word offsets without repacking either matrix.
  const gateWord = bundle.tensor('gate.weight').offset / 4;
  const upWord = bundle.tensor('up.weight').offset / 4;
  const fused = q4SmallMV2(
    [intermediate, hidden, gateWord, 0, 0, 0, 1, upWord],
    new Uint32Array(bundle.binary),
    ff,
    new Float32Array(intermediate),
    true,
  );
  const next = linear(bundle, 'down.weight', fused, postDn);
  const logits = gdnSubsetLogits(
    tableWords,
    gdnRms(next, bundle.floats('final_norm.weight')),
    bundle.subsetIds,
  );
  return { feature: next, logits, matrix: dn.states[0], history: conv.histories[0] };
}
