/**
 * Target-distribution processing shared by autoregressive decode and the
 * speculative verifier. Losslessness is defined relative to THIS processed
 * distribution: whatever temperature / top-k / top-p transform the
 * autoregressive sampler applies, the verifier must apply the identical
 * transform to the target logits at every draft position.
 *
 * All math is float64 and fully deterministic (stable sort with index
 * tie-break) so that the processed distribution is a pure function of logits.
 */

export interface SamplingParams {
  /** 0 (or <= 0) selects greedy decoding. */
  temperature: number;
  /** Keep the k most likely tokens (0 / undefined = disabled). */
  topK?: number;
  /** Nucleus mass threshold in (0, 1] (1 / undefined = disabled). */
  topP?: number;
}

export const GREEDY: SamplingParams = { temperature: 0 };

export function isGreedy(p: SamplingParams): boolean {
  return !(p.temperature > 0);
}

/** argmax with lowest-index tie-break (matches the WGSL argmax reduction). */
export function argmax(logits: ArrayLike<number>): number {
  let best = 0;
  let bestV = logits[0];
  for (let i = 1; i < logits.length; i++) {
    const v = logits[i];
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}

/**
 * Convert logits to the processed sampling distribution.
 * Order of operations (same as HF `generate` and tinygpt `sampling.ts`
 * conventions): temperature -> top-k -> softmax -> top-p -> renormalise.
 * Greedy params return a one-hot distribution at argmax.
 */
export function processLogits(logits: ArrayLike<number>, params: SamplingParams): Float64Array {
  const n = logits.length;
  const out = new Float64Array(n);
  if (isGreedy(params)) {
    out[argmax(logits)] = 1;
    return out;
  }
  const t = params.temperature;
  const topK = params.topK && params.topK > 0 ? Math.min(params.topK, n) : n;
  const topP = params.topP !== undefined && params.topP > 0 && params.topP < 1 ? params.topP : 1;

  // Order indices by descending logit, index ascending on ties (deterministic).
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => logits[b] - logits[a] || a - b);

  const keep = order.slice(0, topK);
  const maxL = logits[keep[0]] / t;
  let z = 0;
  const w = new Float64Array(keep.length);
  for (let i = 0; i < keep.length; i++) {
    // Extremely small positive temperatures can overflow even at f32 logits.
    w[i] = Math.exp(Number.isFinite(maxL) ? logits[keep[i]] / t - maxL : (logits[keep[i]] - logits[keep[0]]) / t);
    z += w[i];
  }
  // Nucleus: smallest prefix (in sorted order) whose mass >= topP. Always keeps >= 1 token.
  let cut = keep.length;
  if (topP < 1) {
    let cum = 0;
    for (let i = 0; i < keep.length; i++) {
      cum += w[i] / z;
      if (cum >= topP) {
        cut = i + 1;
        break;
      }
    }
  }
  let z2 = 0;
  for (let i = 0; i < cut; i++) z2 += w[i];
  for (let i = 0; i < cut; i++) out[keep[i]] = w[i] / z2;
  return out;
}

/** Inverse-CDF sample with a supplied uniform u in [0,1). Skips zero-mass tokens. */
export function sampleWithUniform(probs: ArrayLike<number>, u: number): number {
  let cum = 0;
  let last = -1;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p <= 0) continue;
    last = i;
    cum += p;
    if (u < cum) return i;
  }
  if (last < 0) throw new Error('sampleWithUniform: distribution has no mass');
  return last; // float round-off guard
}

/** Residual distribution norm(max(0, p - q)); returns null when it has no mass. */
export function residual(p: ArrayLike<number>, q: ArrayLike<number> | number): Float64Array | null {
  const n = p.length;
  const out = new Float64Array(n);
  let z = 0;
  for (let i = 0; i < n; i++) {
    const qi = typeof q === 'number' ? (i === q ? 1 : 0) : q[i];
    const d = p[i] - qi;
    if (d > 0) {
      out[i] = d;
      z += d;
    }
  }
  if (!(z > 0)) return null;
  for (let i = 0; i < n; i++) out[i] /= z;
  return out;
}
