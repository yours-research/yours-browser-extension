/**
 * Lossless speculative verification (Leviathan et al. 2023 / Chen et al. 2023),
 * generalised to token trees via recursive rejection sampling (SpecInfer,
 * Miao et al. 2024) and specialised for deterministic (zero-weight) drafters.
 *
 * Correctness contract: for any drafter, the sequence of emitted tokens has
 * exactly the same distribution as autoregressive sampling from
 * `processLogits(targetLogits, params)` at every position. In greedy mode the
 * emitted tokens are *identical* to greedy autoregressive decode.
 *
 * Drafter proposal model
 * ----------------------
 * Each drafted token carries an optional proposal distribution `q`:
 *   - `q === undefined`  -> deterministic drafter (suffix automaton, n-gram,
 *                           grammar). The proposal is a point mass at `token`.
 *                           Accept w.p. p(token); on reject sample from p with
 *                           `token` removed (renormalised).
 *   - `q` given          -> token was drawn x ~ q (EAGLE / MTP / layer-skip
 *                           drafters sampling at temperature > 0).
 *                           Accept w.p. min(1, p(x)/q(x)); on reject sample
 *                           from norm(max(0, p - q)).
 * Siblings in a tree must be independent draws from their own q (or
 * deterministic). Under that condition sequential recursive rejection is
 * lossless (each rejection hands the residual to the next sibling, which is a
 * valid single-draft speculative sample of that residual).
 */
import { Rng } from '../util/rng.js';
import { SamplingParams, argmax, isGreedy, processLogits, residual, sampleWithUniform } from './distribution.js';

export interface DraftToken {
  token: number;
  /** Proposal distribution the token was sampled from; undefined = deterministic. */
  q?: Float64Array;
}

export interface VerifyResult {
  /** Number of draft tokens accepted (0..K). */
  accepted: number;
  /** Tokens to append: accepted drafts followed by exactly one correction/bonus token. */
  emitted: number[];
  /** For trees: node indices of the accepted path (root excluded). */
  path: number[];
}

/**
 * Lazily processed target distributions. Only positions the verifier actually
 * visits are softmaxed / top-k'd, which matters at |V| = 81 920.
 */
export class TargetView {
  private cache: (Float64Array | undefined)[] = [];
  constructor(
    private readonly logits: readonly ArrayLike<number>[],
    private readonly params: SamplingParams,
  ) {}
  get length(): number {
    return this.logits.length;
  }
  argmax(i: number): number {
    return argmax(this.logits[i]);
  }
  dist(i: number): Float64Array {
    let d = this.cache[i];
    if (!d) {
      d = processLogits(this.logits[i], this.params);
      this.cache[i] = d;
    }
    return d;
  }
}

function acceptProb(p: ArrayLike<number>, d: DraftToken): number {
  const px = p[d.token];
  if (!d.q) return px; // point mass: min(1, p/1)
  const qx = d.q[d.token];
  if (!(qx > 0)) throw new Error(`draft token ${d.token} has zero proposal probability`);
  return Math.min(1, px / qx);
}

/**
 * Linear-chain verification. `targetLogits[i]` is the target's logits for the
 * position *after* consuming draft[0..i-1] (so targetLogits[0] follows the last
 * committed token and targetLogits[K] is the bonus position). Requires
 * targetLogits.length === draft.length + 1.
 */
export function verifyChain(
  draft: readonly DraftToken[],
  targetLogits: readonly ArrayLike<number>[],
  params: SamplingParams,
  rng: Rng,
): VerifyResult {
  if (targetLogits.length !== draft.length + 1) {
    throw new Error(`verifyChain: need ${draft.length + 1} target positions, got ${targetLogits.length}`);
  }
  const view = new TargetView(targetLogits, params);
  const emitted: number[] = [];
  const path: number[] = [];

  if (isGreedy(params)) {
    for (let i = 0; i < draft.length; i++) {
      const a = view.argmax(i);
      if (draft[i].token !== a) {
        emitted.push(a);
        return { accepted: i, emitted, path };
      }
      emitted.push(a);
      path.push(i);
    }
    emitted.push(view.argmax(draft.length));
    return { accepted: draft.length, emitted, path };
  }

  for (let i = 0; i < draft.length; i++) {
    const p = view.dist(i);
    const d = draft[i];
    if (rng.next() < acceptProb(p, d)) {
      emitted.push(d.token);
      path.push(i);
      continue;
    }
    const r = residual(p, d.q ?? d.token);
    // r === null only when p == q exactly, where acceptance prob is 1; unreachable.
    emitted.push(sampleWithUniform(r ?? p, rng.next()));
    return { accepted: i, emitted, path };
  }
  emitted.push(sampleWithUniform(view.dist(draft.length), rng.next()));
  return { accepted: draft.length, emitted, path };
}

/**
 * Token tree. Node i holds `tokens[i]`, its parent index `parents[i]` (-1 =
 * child of the root, i.e. of the last committed token) and optional proposal
 * `q[i]`. Nodes must be topologically ordered (parent index < child index).
 * Sibling order is the order of appearance and is the order in which
 * recursive rejection tries them (put the drafter's best guess first).
 */
export interface DraftTree {
  tokens: number[];
  parents: number[];
  q?: (Float64Array | undefined)[];
}

export function treeDepths(tree: DraftTree): number[] {
  const d: number[] = [];
  for (let i = 0; i < tree.tokens.length; i++) {
    const p = tree.parents[i];
    if (p >= i) throw new Error('DraftTree nodes must be topologically ordered');
    d.push(p < 0 ? 1 : d[p] + 1);
  }
  return d;
}

export function childrenOf(tree: DraftTree): number[][] {
  const kids: number[][] = Array.from({ length: tree.tokens.length + 1 }, () => []);
  // slot 0 = root, slot i+1 = node i
  for (let i = 0; i < tree.tokens.length; i++) kids[tree.parents[i] + 1].push(i);
  return kids;
}

/**
 * Tree verification. `targetLogits[0]` = after root; `targetLogits[i+1]` =
 * after node i (i.e. conditioned on the root->i path). Length must be N+1.
 */
export function verifyTree(
  tree: DraftTree,
  targetLogits: readonly ArrayLike<number>[],
  params: SamplingParams,
  rng: Rng,
): VerifyResult {
  const n = tree.tokens.length;
  if (targetLogits.length !== n + 1) throw new Error(`verifyTree: need ${n + 1} target positions`);
  treeDepths(tree); // validates ordering
  const kids = childrenOf(tree);
  const view = new TargetView(targetLogits, params);
  const emitted: number[] = [];
  const path: number[] = [];
  let slot = 0; // 0 = root, i+1 = node i

  if (isGreedy(params)) {
    for (;;) {
      const a = view.argmax(slot);
      emitted.push(a);
      const next = kids[slot].find((c) => tree.tokens[c] === a);
      if (next === undefined) return { accepted: path.length, emitted, path };
      path.push(next);
      slot = next + 1;
    }
  }

  for (;;) {
    let p: Float64Array = view.dist(slot);
    let acceptedChild = -1;
    for (const c of kids[slot]) {
      const d: DraftToken = { token: tree.tokens[c], q: tree.q?.[c] };
      if (rng.next() < acceptProb(p, d)) {
        acceptedChild = c;
        break;
      }
      const r = residual(p, d.q ?? d.token);
      if (!r) throw new Error('verifyTree: empty residual after rejection (p == q)');
      p = r;
    }
    if (acceptedChild < 0) {
      emitted.push(sampleWithUniform(p, rng.next()));
      return { accepted: path.length, emitted, path };
    }
    emitted.push(tree.tokens[acceptedChild]);
    path.push(acceptedChild);
    slot = acceptedChild + 1;
  }
}

/** Convert a linear draft into the equivalent single-branch tree. */
export function chainToTree(draft: readonly DraftToken[]): DraftTree {
  return {
    tokens: draft.map((d) => d.token),
    parents: draft.map((_, i) => i - 1),
    q: draft.map((d) => d.q),
  };
}
