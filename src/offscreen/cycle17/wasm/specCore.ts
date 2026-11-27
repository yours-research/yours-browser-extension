/**
 * Host wrapper for crates/yours-spec-core compiled to wasm32-unknown-unknown
 * (no imports; ~100 KB). Drop-in for the tinygpt-browser offscreen document:
 *
 *   const core = await SpecCoreWasm.instantiate(bytes);
 *   const a = core.engineA({ corpus, profile, sessionCalibrator: true });
 *   a.begin(); a.observe(promptIds, false);
 *   a.setGrammar(alignedGrammarSeqs);        // host tokenizer aligns grammar text
 *   const draft = a.propose(7, true);         // { tokens, parents, probs }
 *   ... verify on GPU ...
 *   a.feedback(acceptedPath); a.observe(emitted, true);
 *
 * Per-call data is staged in one preallocated scratch region; the wasm side
 * never allocates after construction.
 */
import { DraftTree, VerifyResult } from '../verifier/rejection.js';

/** Minimal WebAssembly surface (the repo compiles without the DOM lib). */
interface WasmApi {
  instantiate(bytes: ArrayBuffer | ArrayBufferView, imports: object): Promise<{ instance: { exports: unknown } }>;
}
const WA = (globalThis as unknown as { WebAssembly: WasmApi }).WebAssembly;

interface Exports {
  memory: { buffer: ArrayBuffer };
  yse_alloc(bytes: number): number;
  yse_free(ptr: number, bytes: number): void;
  yse_engine_new(maxCtx: number, historyCap: number, maxNodes: number, flags: number, draftMs: number, corpus: number, corpusLen: number, stepMs: number, maxM: number, rejectPenaltyMs: number): number;
  yse_engine_free(e: number): void;
  yse_engine_begin(e: number): void;
  yse_engine_observe(e: number, toks: number, n: number, generated: number): number;
  yse_engine_grammar_clear(e: number): void;
  yse_engine_grammar_push(e: number, prior: number, toks: number, n: number): number;
  yse_engine_propose(e: number, maxNodes: number, tree: number): number;
  yse_engine_out_tokens(e: number): number;
  yse_engine_out_parents(e: number): number;
  yse_engine_out_probs(e: number): number;
  yse_engine_feedback(e: number, path: number, n: number): number;
  yse_engine_end(e: number): void;
  yse_pool_new(width: number, capacity: number): number;
  yse_pool_free(p: number): void;
  yse_pool_plan(p: number, hist: number, histLen: number, baseTokens: number, baseParents: number, baseLen: number, m: number, usePool: number): number;
  yse_pool_out_tokens(p: number): number;
  yse_pool_out_parents(p: number): number;
  yse_pool_feedback(p: number, hist: number, histLen: number, argmax: number, n: number, emitted: number): number;
  yse_patha_new(): number;
  yse_patha_free(p: number): void;
  yse_patha_verify(p: number, tokens: number, parents: number, n: number, words: number, wordsLen: number, vocab: number, temperature: number, topK: number, topP: number, rngState: number): number;
  yse_patha_out_emitted(p: number): number;
  yse_patha_out_emitted_len(p: number): number;
  yse_patha_out_path(p: number, out: number): number;
  yse_rng_seed(seed: number, out: number): void;
  yse_cutoff_new(stepMs: number, maxM: number, rejectPenaltyMs: number, draftMs: number, headMs: number, threshold: number, decay: number, priorStrength: number, chainPrior: number, nPrior: number, flags: number, uncertaintyZ: number, valueDecay: number, exploreEvery: number): number;
  yse_cutoff_free(c: number): void;
  yse_cutoff_begin(c: number): void;
  yse_cutoff_reset(c: number): void;
  yse_cutoff_invoke(c: number, tool: number, aProbs: number, aParents: number, nA: number, maxNodes: number): number;
  yse_cutoff_decide(c: number, tool: number, aProbs: number, nA: number, bTokens: number, bProbs: number, nB: number, maxNodes: number): number;
  yse_cutoff_gated(c: number): number;
  yse_cutoff_out_probs(c: number): number;
  yse_cutoff_commit(c: number, toks: number, n: number): number;
  yse_cutoff_conditional(c: number, tool: number, depth: number, cond: number): number;
}

/** Adaptive learned-head cutoff (crates/yours-spec-core/src/adaptive.rs). */
export interface LearnedCutoffOptions {
  /** Verify cost curve stepMs[M - 1] and reject surcharge of the chain path. */
  profile: { stepMs: readonly number[]; rejectPenaltyMs: number };
  /** Host drafting estimate inside T(M) (default 0.05 ms). */
  draftMs?: number;
  /** Wall ms of one complete learned chain, used only by the EV gate (default 0.4). */
  headMs?: number;
  /** Engine A root confidence above which the head is skipped (default 0.7). */
  threshold?: number;
  /** Per-key EWMA retention in (0, 1]; 1 = cumulative Beta counts (default 0.95). */
  decay?: number;
  /** Beta prior pseudo-count weight (default 4). */
  priorStrength?: number;
  /** Lab chain conditionals q1, q2|1, ... (up to 5); omitted depths use the raw head probability. */
  chainPrior?: readonly number[];
  /** Skip the head when its optimistic calibrated chain cannot beat Engine A by headMs. */
  evGate?: boolean;
  /** Decide on the Beta posterior mean minus z standard deviations (default 0.5). */
  uncertaintyZ?: number;
  /** Separate calibration for tool-call vs prose steps (default true; false = Milestone 8/9 shared table). */
  regimes?: boolean;
  /** Per-regime value gate: skip the head while its recent expected gain is below headMs (default false; it cost throughput on the development cohorts). */
  valueGate?: boolean;
  /** EWMA retention of the per-regime gain (default 0.8). */
  valueDecay?: number;
  /** While gated off, still call the head every this many eligible steps (default 4). */
  exploreEvery?: number;
}

export interface EngineAWasmOptions {
  /** Static corpus documents (token ids). */
  corpus?: readonly (readonly number[])[];
  /** Adaptive K(t): stepMs[M - 1] for M = 1..maxM, plus rejection penalty. */
  profile?: { stepMs: readonly number[]; rejectPenaltyMs: number };
  sessionCalibrator?: boolean;
  draftMs?: number;
  maxCtx?: number;
  historyCapacity?: number;
  maxNodes?: number;
  disable?: Partial<Record<'ctx' | 'anchor' | 'corpus' | 'history' | 'grammar', boolean>>;
}

const SCRATCH_BYTES = 1 << 20;

export class SpecCoreWasm {
  private scratch: number;
  private top = 0;
  private constructor(readonly x: Exports) {
    this.scratch = x.yse_alloc(SCRATCH_BYTES);
  }

  static async instantiate(bytes: ArrayBuffer | ArrayBufferView): Promise<SpecCoreWasm> {
    const { instance } = await WA.instantiate(bytes, {});
    return new SpecCoreWasm(instance.exports as unknown as Exports);
  }

  // --- scratch staging (views are re-created per call: memory may grow) ---------
  reset(): void {
    this.top = 0;
  }
  stage(arr: ArrayLike<number>, kind: 'i32' | 'u32' | 'f64'): number {
    const bytes = kind === 'f64' ? 8 : 4;
    this.top = (this.top + 7) & ~7;
    const ptr = this.scratch + this.top;
    this.top += Math.max(1, arr.length) * bytes;
    if (this.top > SCRATCH_BYTES) throw new Error('spec-core scratch overflow');
    const buf = this.x.memory.buffer;
    if (kind === 'f64') new Float64Array(buf, ptr, arr.length).set(arr);
    else if (kind === 'i32') new Int32Array(buf, ptr, arr.length).set(arr);
    else new Uint32Array(buf, ptr, arr.length).set(arr);
    return ptr;
  }
  i32(ptr: number, n: number): number[] {
    return Array.from(new Int32Array(this.x.memory.buffer, ptr, n));
  }
  u32(ptr: number, n: number): number[] {
    return Array.from(new Uint32Array(this.x.memory.buffer, ptr, n));
  }
  f64(ptr: number, n: number): number[] {
    return Array.from(new Float64Array(this.x.memory.buffer, ptr, n));
  }

  engineA(o: EngineAWasmOptions = {}): EngineAWasm {
    const flat: number[] = [];
    for (const d of o.corpus ?? []) flat.push(...d, -2147483648);
    const names = ['ctx', 'anchor', 'corpus', 'history', 'grammar'] as const;
    let flags = o.sessionCalibrator ? 1 : 0;
    names.forEach((n, i) => {
      if (o.disable?.[n]) flags |= 1 << (i + 1);
    });
    this.reset();
    const corpus = this.stage(flat, 'i32');
    const steps = o.profile?.stepMs ?? [];
    const stepPtr = this.stage(steps, 'f64');
    const h = this.x.yse_engine_new(o.maxCtx ?? 32768, o.historyCapacity ?? 65536, o.maxNodes ?? 31, flags, o.draftMs ?? 0.05, corpus, flat.length, stepPtr, steps.length, o.profile?.rejectPenaltyMs ?? 0);
    return new EngineAWasm(this, h);
  }

  pool(width = 4, capacity = 512): JacobiPoolWasm {
    return new JacobiPoolWasm(this, this.x.yse_pool_new(width, capacity));
  }

  cutoff(o: LearnedCutoffOptions): LearnedCutoffWasm {
    this.reset();
    const steps = this.stage(o.profile.stepMs, 'f64');
    const prior = o.chainPrior ?? [];
    const priorPtr = this.stage(prior, 'f64');
    const h = this.x.yse_cutoff_new(steps, o.profile.stepMs.length, o.profile.rejectPenaltyMs, o.draftMs ?? 0.05, o.headMs ?? 0.4,
      o.threshold ?? 0.7, o.decay ?? 0.95, o.priorStrength ?? 4, priorPtr, prior.length, (o.evGate ? 1 : 0) | (o.regimes === false ? 2 : 0) | (o.valueGate === true ? 4 : 0), o.uncertaintyZ ?? 0.5, o.valueDecay ?? 0.8, o.exploreEvery ?? 4);
    if (!h) throw new Error('Invalid learned cutoff configuration');
    return new LearnedCutoffWasm(this, h);
  }

  pathA(): PathAWasm {
    return new PathAWasm(this, this.x.yse_patha_new());
  }

  /** xoshiro128** state for `new Rng(seed)`. */
  rngState(seed: number): Uint32Array {
    this.reset();
    const p = this.stage([0, 0, 0, 0], 'u32');
    this.x.yse_rng_seed(seed >>> 0, p);
    return Uint32Array.from(this.u32(p, 4));
  }
}

export class EngineAWasm {
  constructor(
    private readonly c: SpecCoreWasm,
    readonly handle: number,
  ) {}
  begin(): void {
    this.c.x.yse_engine_begin(this.handle);
  }
  observe(tokens: readonly number[], generated: boolean): void {
    this.c.reset();
    if (this.c.x.yse_engine_observe(this.handle, this.c.stage(tokens, 'i32'), tokens.length, generated ? 1 : 0) < 0) throw new Error('context full');
  }
  setGrammar(seqs: readonly { prior: number; toks: readonly number[] }[]): void {
    this.c.x.yse_engine_grammar_clear(this.handle);
    for (const s of seqs) {
      this.c.reset();
      if (this.c.x.yse_engine_grammar_push(this.handle, s.prior, this.c.stage(s.toks, 'i32'), s.toks.length) < 0) throw new Error('grammar capacity');
    }
  }
  propose(maxNodes: number, tree: boolean): { tokens: number[]; parents: number[]; probs: number[] } {
    const x = this.c.x;
    const n = x.yse_engine_propose(this.handle, maxNodes, tree ? 1 : 0);
    return { tokens: this.c.i32(x.yse_engine_out_tokens(this.handle), n), parents: this.c.i32(x.yse_engine_out_parents(this.handle), n), probs: this.c.f64(x.yse_engine_out_probs(this.handle), n) };
  }
  feedback(path: readonly number[]): void {
    this.c.reset();
    if (this.c.x.yse_engine_feedback(this.handle, this.c.stage(path, 'u32'), path.length) < 0) throw new Error('bad path');
  }
  end(): void {
    this.c.x.yse_engine_end(this.handle);
  }
  free(): void {
    this.c.x.yse_engine_free(this.handle);
  }
}

export class LearnedCutoffWasm {
  constructor(
    private readonly c: SpecCoreWasm,
    readonly handle: number,
  ) {}
  beginTurn(): void {
    this.c.x.yse_cutoff_begin(this.handle);
  }
  reset(): void {
    this.c.x.yse_cutoff_reset(this.handle);
  }
  /** Should the learned head run, given Engine A's (already K(t)-cut) proposal? */
  shouldInvoke(a: { probs?: readonly number[]; parents: readonly number[] }, maxNodes: number, tool = false): boolean {
    const c = this.c;
    c.reset();
    const probs = a.probs ?? a.parents.map(() => 0);
    return c.x.yse_cutoff_invoke(this.handle, tool ? 1 : 0, c.stage(probs, 'f64'), c.stage(a.parents, 'i32'), a.parents.length, maxNodes) === 1;
  }
  /** Head calls the per-regime value gate declined. */
  get gated(): number {
    return this.c.x.yse_cutoff_gated(this.handle);
  }
  /** k* >= 0 when the cut learned chain wins, -1 when Engine A wins; plus calibrated path probabilities. */
  decide(aProbs: readonly number[], bTokens: readonly number[], bProbs: readonly number[], maxNodes: number, tool = false): { k: number; calibrated: number[] } {
    const c = this.c;
    c.reset();
    const n = Math.min(bTokens.length, bProbs.length, 16);
    const k = c.x.yse_cutoff_decide(this.handle, tool ? 1 : 0, c.stage(aProbs, 'f64'), aProbs.length, c.stage(bTokens, 'i32'), c.stage(bProbs, 'f64'), n, maxNodes);
    return { k, calibrated: c.f64(c.x.yse_cutoff_out_probs(this.handle), n) };
  }
  /** Tokens the target emitted this step (accepted drafts + bonus). */
  commit(emitted: readonly number[]): number {
    this.c.reset();
    return this.c.x.yse_cutoff_commit(this.handle, this.c.stage(emitted, 'i32'), emitted.length);
  }
  conditional(depth: number, cond: number, tool = false): number {
    return this.c.x.yse_cutoff_conditional(this.handle, tool ? 1 : 0, depth, cond);
  }
  free(): void {
    this.c.x.yse_cutoff_free(this.handle);
  }
}

export class JacobiPoolWasm {
  constructor(
    private readonly c: SpecCoreWasm,
    readonly handle: number,
  ) {}
  plan(history: readonly number[], base: DraftTree, m: number, usePool = true): DraftTree {
    const c = this.c;
    c.reset();
    const h = c.stage(history, 'i32');
    const t = c.stage(base.tokens, 'i32');
    const p = c.stage(base.parents, 'i32');
    const n = c.x.yse_pool_plan(this.handle, h, history.length, t, p, base.tokens.length, m, usePool ? 1 : 0);
    if (n < 0) throw new Error('invalid context or verify budget');
    return { tokens: c.i32(c.x.yse_pool_out_tokens(this.handle), n), parents: c.i32(c.x.yse_pool_out_parents(this.handle), n) };
  }
  feedback(history: readonly number[], argmax: readonly (number | undefined)[], emitted: number): void {
    const c = this.c;
    c.reset();
    const h = c.stage(history, 'i32');
    const a = c.stage(argmax.map((v) => (v === undefined ? -1 : v)), 'i32');
    if (c.x.yse_pool_feedback(this.handle, h, history.length, a, argmax.length, emitted) < 0) throw new Error('one prediction per evaluated slot');
  }
  free(): void {
    this.c.x.yse_pool_free(this.handle);
  }
}

export class PathAWasm {
  constructor(
    private readonly c: SpecCoreWasm,
    readonly handle: number,
  ) {}
  /** `rngState` (4 x u32) is advanced in place, like the TS Rng. */
  verify(tree: DraftTree, words: Uint32Array, vocab: number, params: { temperature: number; topK: number; topP?: number }, rngState: Uint32Array): VerifyResult {
    const c = this.c;
    c.reset();
    const t = c.stage(tree.tokens, 'u32');
    const p = c.stage(tree.parents, 'i32');
    const w = c.stage(words, 'u32');
    const r = c.stage(rngState, 'u32');
    const acc = c.x.yse_patha_verify(this.handle, t, p, tree.tokens.length, w, words.length, vocab, params.temperature, params.topK, params.topP ?? 1, r);
    rngState.set(c.u32(r, 4));
    if (acc < 0) throw new Error('Path A rejected the readback');
    const ne = c.x.yse_patha_out_emitted_len(this.handle);
    const emitted = c.u32(c.x.yse_patha_out_emitted(this.handle), ne);
    const pathPtr = c.stage(new Array(16).fill(0), 'u32');
    const np = c.x.yse_patha_out_path(this.handle, pathPtr);
    return { accepted: acc, emitted, path: c.u32(pathPtr, np) };
  }
  free(): void {
    this.c.x.yse_patha_free(this.handle);
  }
}
