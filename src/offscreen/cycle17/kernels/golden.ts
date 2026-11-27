import { f32ToF16 } from './f32.js';
import { Rng } from '../util/rng.js';
import {
  ancestorMasks,
  compactKV,
  convTree,
  deltaIO,
  deltaTree,
  gqaTree,
  greedyChoice,
  q4SmallM,
  rankBatch,
  toBits,
} from './reference.js';

export interface VectorBinding {
  type: 'uniform' | 'read-only-storage' | 'storage';
  words: number[];
  expected?: number[]; // entire writable binding including untouched offsets
}
export interface KernelVector {
  name: string;
  shader: string;
  entryPoint: string;
  dispatch: [number, number, number];
  bindings: VectorBinding[];
}
const uniform = (words: number[]): VectorBinding => ({ type: 'uniform', words });
const read = (words: number[]): VectorBinding => ({ type: 'read-only-storage', words });
const write = (words: number[], expected: number[]): VectorBinding => ({ type: 'storage', words, expected });
const seq = (n: number, fn: (i: number) => number) => Array.from({ length: n }, (_, i) => fn(i));
const zeros = (n: number) => Array<number>(n).fill(0);
const halves = (low: number, high: number) => (f32ToF16(low) | (f32ToF16(high) << 16)) >>> 0;

export function goldenVectors(): KernelVector[] {
  const rng = new Rng(7002);
  const frac = () => (rng.int(15) - 7) / 8;
  const vectors: KernelVector[] = [];
  function job(
    name: string,
    shader: string,
    entryPoint: string,
    dispatch: KernelVector['dispatch'],
    bindings: VectorBinding[],
  ) {
    vectors.push({ name, shader, entryPoint, dispatch, bindings });
  }
  for (const m of [1, 5, 16]) {
    const p = [5, 64, 2, 1, 3, 1, m, 0];
    const weights = [71, 81, ...seq(50, (i) => (i % 5 === 4 ? halves(0.125, -0.5) : rng.nextU32()))];
    const x = seq(64 * m + 1, frac),
      y = seq(m * 5 + 4, frac);
    job(
      'q4-m' + m,
      'Q4_GEMV_SMALLM',
      'main',
      [2, 1, 1],
      [uniform(p), read(weights), read(toBits(x)), write(toBits(y), toBits(q4SmallM(p, weights, x, y)))],
    );
  }
  // Siblings include a rejected subtree preceding the accepted slots [0,2,4].
  const parents = [-1, 0, 0, 1, 2, 2];
  const n = parents.length,
    masks = ancestorMasks(parents);
  const tokens = [99, 3, 7, 4, 2, 8],
    best = [7, 4, 2, 5, 9, 0];
  const choice = greedyChoice(parents, tokens, best);
  const accepted = choice.slice(2),
    leaf = accepted.at(-1)!;
  function accept(label: string, ids: number[]) {
    const result = greedyChoice(parents, tokens, ids);
    job(
      label,
      'ACCEPT_TREE',
      'main',
      [1, 1, 1],
      [
        uniform([n]),
        read(parents.map((x) => x >>> 0)),
        read(tokens),
        read(ids),
        write(zeros(n + 2), result.concat(zeros(n + 2 - result.length))),
      ],
    );
  }
  accept('greedy-branch', best);
  accept('root-reject', [1, ...best.slice(1)]);
  const p = [n, 2, 4, 3];
  const rec = seq(n * 2 * (8 + 3 + 2), (i) => (i % 13 === 11 ? 0.75 : i % 13 === 12 ? 0.5 : frac()));
  const s = seq(24, frac),
    dn = deltaTree(p, parents, rec, s);
  const history = seq(9, frac),
    raw = seq(n * 3, frac);
  const weights = seq(12, (i) => [0.5, 0.25, 0, -0.25][i % 4]);
  const conv = convTree(parents, raw, weights, history);
  for (const mode of ['verify', 'commit'] as const) {
    const verify = mode === 'verify';
    job(
      'dn-' + mode,
      'DN_TREE',
      mode,
      [1, 2, verify ? n : 1],
      [
        uniform(p),
        read(masks),
        read(toBits(rec)),
        write(toBits(s), toBits(verify ? s : dn.states[leaf])),
        write(zeros(dn.output.length), verify ? toBits(dn.output) : zeros(dn.output.length)),
        read(choice),
      ],
    );
    job(
      'conv-' + mode,
      'DN_CONV_TREE',
      mode,
      [1, verify ? n : 1, 1],
      [
        uniform([3, n, 0, 0]),
        read(masks),
        read(toBits(raw)),
        read(toBits(weights)),
        write(toBits(history), toBits(verify ? history : conv.histories[leaf])),
        write(zeros(conv.output.length), verify ? toBits(conv.output) : zeros(conv.output.length)),
        read(choice),
      ],
    );
  }
  for (const int8 of [0, 1]) {
    const prefix = 2,
      capacity = prefix + n + 1,
      dim = 4,
      heads = 4,
      kvHeads = 2,
      base = 1;
    const words = int8 ? dim / 4 + 1 : dim / 2;
    const cache = [
      123,
      ...seq(2 * kvHeads * capacity * words, (i) =>
        int8 ? (i % words === words - 1 ? halves(0.125, 0.25) : rng.nextU32()) : halves(frac(), frac()),
      ),
    ];
    const query = seq(n * heads * (dim + 1), frac);
    const p = [prefix, capacity, heads, kvHeads, dim, int8, base, dim + 1];
    const out = gqaTree(p, parents, query, cache);
    job(
      'gqa-' + (int8 ? 'int8' : 'fp16'),
      'GQA_TREE',
      'main',
      [heads, n, 1],
      [uniform(p), read(masks), read(toBits(query)), read(cache), write(zeros(out.length), toBits(out))],
    );
    const cp = [prefix, capacity, words, base];
    job(
      'compact-' + (int8 ? 'int8' : 'fp16'),
      'KV_COMPACT',
      'main',
      [1, 2 * kvHeads, 1],
      [uniform(cp), read(choice), write(cache, compactKV(cp, 2 * kvHeads, accepted, cache))],
    );
  }
  for (const k of [1, 7, 64]) {
    const p = [135, 137, k, 2],
      rows = 3;
    const logits = toBits(seq(2 + 137 * rows, (i) => (i % 7 === 0 ? -Infinity : frac())));
    for (const argmax of k === 1 ? [false, true] : [false]) {
      const out = rankBatch(p, logits, argmax);
      job(
        argmax ? 'argmax' : 'top-' + k,
        'TOPK_BATCH',
        argmax ? 'argmax_batch' : 'topk_batch',
        [rows, 1, 1],
        [uniform(p), read(logits), write(zeros(out.length), out)],
      );
    }
  }
  const io = [4, 2, 4, 2],
    qkv = seq(64, frac),
    ab = seq(16, frac);
  const aux = [...seq(4, (i) => -(i + 1) / 2), ...seq(4, frac), ...seq(4, frac)];
  const y = seq(32, frac),
    z = seq(32, frac);
  const { packed, output } = deltaIO(io, qkv, ab, aux, y, z);
  for (const entry of ['pack', 'gate']) {
    job(
      'dn-' + entry,
      'DN_IO',
      entry,
      [4, 2, 1],
      [
        uniform(io),
        read(toBits(qkv)),
        read(toBits(ab)),
        read(toBits(aux)),
        write(zeros(packed.length), entry === 'pack' ? toBits(packed) : zeros(packed.length)),
        read(toBits(y)),
        read(toBits(z)),
        write(zeros(output.length), entry === 'gate' ? toBits(output) : zeros(output.length)),
      ],
    );
  }
  return vectors;
}
