/**
 * Engine B (EXP-007) hardware microbench for Cycle 17. Browser-safe: no node
 * imports. Bundled to dist/cycle17-dropin/bench/bench-engine-b-head.mjs.
 *
 * Measures, in about 15 seconds on the target GPU:
 *  - headMs(k): wall time of one single-submit learned chain of k = 1..5 steps,
 *    from encode to the 8k-byte readback, with persistent plans warmed up;
 *  - P: the chain reject surcharge, median(reject) - median(accept) at M = 5,
 *    through the extension's own speculative microbench hook, so it can be
 *    re-measured before and after P9a-P9c.
 * `toAdapterOptions` turns the result into the OffscreenSpecAdapter cost
 * profile (stepMs, rejectPenaltyMs) and cutoff (headMs).
 */
import type { DraftAnchor } from '../draft/gdnEagle3Runtime.js';

export interface ChainRuntime {
  proposeChain(anchor: DraftAnchor, k: number): Promise<{ tokens: number[] }>;
}

/** Wraps the extension's speculative microbench: wall ms of one M-token chain verify step. */
export interface VerifyProbe {
  measure(M: number, reject: boolean): Promise<number>;
}

export interface BenchOptions {
  ks?: readonly number[];
  warmup?: number;
  reps?: number;
  /** Stop adding repetitions after this many ms (keeps the whole run near 15 s). */
  budgetMs?: number;
  now?: () => number;
}

export interface HeadLatency {
  k: number;
  medianMs: number;
  p10Ms: number;
  p90Ms: number;
  samples: number;
}

export function quantile(sorted: readonly number[], q: number): number {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

const clock = () => (globalThis as { performance?: { now(): number } }).performance?.now() ?? Date.now();

/** headMs per chain length. Interleaves k values so drift affects all of them equally. */
export async function benchHeadLatency(runtime: ChainRuntime, anchor: DraftAnchor, o: BenchOptions = {}): Promise<HeadLatency[]> {
  const ks = o.ks ?? [1, 2, 3, 4, 5];
  const now = o.now ?? clock;
  const reps = o.reps ?? 40, warmup = o.warmup ?? 3, budget = o.budgetMs ?? 10_000;
  for (let w = 0; w < warmup; w++) for (const k of ks) await runtime.proposeChain(anchor, k);
  const samples = new Map(ks.map((k) => [k, [] as number[]]));
  const start = now();
  for (let r = 0; r < reps; r++) {
    for (const k of ks) {
      const t0 = now();
      const out = await runtime.proposeChain(anchor, k);
      const dt = now() - t0;
      if (out.tokens.length !== k) throw new Error(`Expected ${k} draft tokens, got ${out.tokens.length}`);
      samples.get(k)!.push(dt);
    }
    if (now() - start > budget && r >= 4) break;
  }
  return ks.map((k) => {
    const s = [...samples.get(k)!].sort((a, b) => a - b);
    return { k, medianMs: quantile(s, 0.5), p10Ms: quantile(s, 0.1), p90Ms: quantile(s, 0.9), samples: s.length };
  });
}

/** Reject surcharge P at chain size M from paired accept/reject verify steps. */
export async function benchRejectSurcharge(probe: VerifyProbe, o: { M?: number; reps?: number; warmup?: number } = {}) {
  const M = o.M ?? 5, reps = o.reps ?? 15;
  for (let w = 0; w < (o.warmup ?? 2); w++) {
    await probe.measure(M, false);
    await probe.measure(M, true);
  }
  const accept: number[] = [], reject: number[] = [];
  for (let r = 0; r < reps; r++) {
    // Alternate the order so neither arm always runs on a warmer device.
    const first = r % 2 === 0;
    const a = first ? await probe.measure(M, false) : NaN;
    const b = await probe.measure(M, true);
    const c = first ? a : await probe.measure(M, false);
    accept.push(c);
    reject.push(b);
  }
  const sa = [...accept].sort((x, y) => x - y), sr = [...reject].sort((x, y) => x - y);
  const acceptMs = quantile(sa, 0.5), rejectMs = quantile(sr, 0.5);
  return { M, acceptMs, rejectMs, rejectPenaltyMs: Math.max(0, rejectMs - acceptMs), samples: reps };
}

export interface EngineBBenchReport {
  format: 'engine-b-head-bench';
  version: 1;
  device?: string;
  head: HeadLatency[];
  /** headMs to use for a k=5 learned chain (median). */
  headMs: number;
  reject?: Awaited<ReturnType<typeof benchRejectSurcharge>>;
  elapsedMs: number;
}

export async function runEngineBBench(runtime: ChainRuntime, anchor: DraftAnchor, probe?: VerifyProbe, o: BenchOptions & { device?: string } = {}): Promise<EngineBBenchReport> {
  const now = o.now ?? clock;
  const t0 = now();
  const head = await benchHeadLatency(runtime, anchor, o);
  const reject = probe ? await benchRejectSurcharge(probe) : undefined;
  const k5 = head.find((h) => h.k === 5) ?? head[head.length - 1];
  return { format: 'engine-b-head-bench', version: 1, device: o.device, head, headMs: k5.medianMs, reject, elapsedMs: now() - t0 };
}

/**
 * OffscreenSpecAdapter options from a measured report: the chain step curve
 * (stepMs[M - 1], from the live microbench) with the measured reject
 * surcharge, and the measured head cost for the cutoff's EV gate.
 */
export function toAdapterOptions(report: EngineBBenchReport, stepMs: readonly number[], fallbackRejectPenaltyMs: number) {
  if (!stepMs.length || stepMs.some((v) => !(v > 0))) throw new Error('stepMs must be positive wall times for M = 1..');
  return {
    profile: { stepMs: [...stepMs], rejectPenaltyMs: report.reject?.rejectPenaltyMs ?? fallbackRejectPenaltyMs },
    cutoff: { headMs: report.headMs },
  };
}
