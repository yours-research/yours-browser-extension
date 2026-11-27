import { bitsToF32 } from '../kernels/f32.js';
import { Rng } from '../util/rng.js';
import { SamplingParams } from './distribution.js';
import { DraftTree, verifyTree, VerifyResult } from './rejection.js';

/** Decode TOPK_BATCH's interleaved (token, raw f32 bits) buffer. The root row
 * precedes one row per draft node. q remains the original proposal probability,
 * including when its distribution puts mass outside the target's support. */
export function verifyTopKReadback(
  tree: DraftTree,
  words: Uint32Array,
  vocab: number,
  params: SamplingParams,
  rng: Rng,
): VerifyResult {
  const k = params.topK ?? 0;
  if (
    !Number.isInteger(vocab) ||
    vocab < 1 ||
    !Number.isInteger(k) ||
    k < 1 ||
    k > 64 ||
    !(params.temperature > 0) ||
    !Number.isFinite(params.temperature)
  ) {
    throw new Error('Path A requires positive finite temperature and explicit topK in 1..64');
  }
  const width = Math.min(vocab, k);
  if (words.length !== 2 * width * (tree.tokens.length + 1)) throw new Error('bad readback size');
  const ids = new Set(tree.tokens);
  for (let i = 0; i < words.length; i += 2) ids.add(words[i]);
  const support = [...ids].sort((a, b) => a - b); // preserve dense inverse-CDF order
  if (support.some((id) => !Number.isInteger(id) || id < 0 || id >= vocab)) throw new Error('bad token id');
  const index = new Map(support.map((id, i) => [id, i]));
  const rows: Float32Array[] = [];
  for (let offset = 0; offset < words.length; offset += 2 * width) {
    const row = new Float32Array(support.length).fill(-Infinity);
    const seen = new Set<number>();
    let hasMass = false;
    for (let j = 0; j < width; j++) {
      const id = words[offset + 2 * j];
      const logit = bitsToF32(words[offset + 2 * j + 1]);
      if (seen.has(id) || Number.isNaN(logit) || logit === Infinity) throw new Error('invalid top-k row');
      seen.add(id);
      hasMass ||= Number.isFinite(logit);
      row[index.get(id)!] = logit;
    }
    if (!hasMass) throw new Error('empty target support');
    rows.push(row);
  }
  const remapped: DraftTree = {
    parents: tree.parents,
    tokens: tree.tokens.map((t) => index.get(t)!),
    q: tree.q?.map((q) => {
      if (q && q.length !== vocab) throw new Error('bad proposal size');
      return q && Float64Array.from(support, (id) => q[id]);
    }),
  };
  const result = verifyTree(remapped, rows, params, rng);
  return { ...result, emitted: result.emitted.map((i) => support[i]) };
}
