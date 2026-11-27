/**
 * Offscreen-document adapter for tinygpt-browser (RC1 patches P8 + P9).
 *
 * One object per conversation that owns:
 *  - Engine A in yours_spec_core.wasm (bit-identical to the TS EngineA,
 *    zero steady-state allocations), plus the host-side tool grammar. The
 *    grammar needs the runtime tokenizer to align text continuations with the
 *    live token boundary; that logic mirrors EngineA's text tail and
 *    alignContinuation exactly.
 *  - The production HybridDraftRouter. It loads Engine C (EXP-007
 *    GDN-EAGLE-3) lazily from a manifest URL, and falls back to zero-VRAM
 *    Engine A when the bundle is absent (404/410), when no anchor is supplied,
 *    or when Engine A is confident.
 *  - The E3 Path-A sampler (wasm) for lossless T > 0 verification from the
 *    TOPK_BATCH readback.
 *
 * Browser-safe: no node imports. Typechecked standalone inside the RC1 pack.
 */
import type { BenchDrafter, DraftBudget, ScoredDraftTree } from '../engines/types.js';
import { buildToolCorpus } from '../engines/toolCorpus.js';
import {
  type CallDialect,
  EXTENSION_TOOLS,
  type JsonStyle,
  LAB_TOOLS,
  SPACED_JSON,
  type ToolSchema,
  detectDialect,
  detectJsonStyle,
  grammarContinuations,
  parseToolsFromPrompt,
} from '../engines/toolGrammar.js';
import { HybridDraftRouter, thresholdPolicy, type RoutePolicy, type RoutedProposal } from '../draft/hybridRouter.js';
import { loadGdnBundle } from '../draft/gdnEagle3Loader.js';
import { GdnDraftRuntime, type DraftAnchor, type DraftRuntimeOptions, type ReadShader, type TiedTable } from '../draft/gdnEagle3Runtime.js';
import type { DraftTree, VerifyResult } from '../verifier/rejection.js';
import { SpecCoreWasm, type EngineAWasm, type LearnedCutoffOptions, type LearnedCutoffWasm, type PathAWasm } from '../wasm/specCore.js';


/** The runtime tokenizer surface the grammar needs. */
export interface AdapterTokenizer {
  encode(text: string): number[];
  decode(ids: readonly number[]): string;
}

export interface LearnedDrafter {
  propose(anchor: DraftAnchor, budget: DraftBudget): Promise<ScoredDraftTree>;
  dispose(): void;
}

export type FusedDraftPlan =
  | { kind: 'proposal'; proposal: RoutedProposal }
  | { kind: 'fused'; runtime: GdnDraftRuntime; k: number };

export interface OffscreenSpecAdapterOptions {
  /** yours_spec_core.wasm bytes (from the RC1 pack). */
  wasm: ArrayBuffer | ArrayBufferView;
  tokenizer: AdapterTokenizer;
  /** Measured step curve of the ACTIVE verify path: stepMs[M - 1], plus the rejection penalty. */
  profile: { stepMs: readonly number[]; rejectPenaltyMs: number };
  /** Static drafting corpus; defaults to schema-rendered tool exemplars. */
  corpusDocs?: string[];
  /** Router threshold on Engine A's best root-child probability (default 0.7). */
  threshold?: number;
  /** Constant drafting-latency estimate inside K(t) (default 0.05 ms). */
  draftMs?: number;
  maxCtx?: number;
  /** Engine C (EXP-007). Omit, or point at a missing manifest, for Engine A only. */
  engineC?: {
    manifestUrl: URL; device: GPUDevice; table: TiedTable; readShader: ReadShader; expectedSha256?: string;
    /** Kernel tier: 'auto' (default, fastest supported) or 'reference' (bit-exact scalar kernels). */
    mode?: DraftRuntimeOptions['mode'];
  };
  /**
   * Adaptive per-step learned cutoff (A+B+C) in spec-core WASM. Default on.
   * `false` restores the RC1 router (the learned chain fills the budget).
   */
  cutoff?: false | Omit<LearnedCutoffOptions, 'profile' | 'threshold' | 'draftMs'>;
  /** Test hook or shared host cache: supply the learned drafter directly (overrides `engineC`). */
  loadLearned?: () => Promise<LearnedDrafter | null>;
  /** When false, `dispose()` releases WASM session state without disposing the shared learned head. */
  ownsLearnedHead?: boolean;
}

/** wasm Engine A + host grammar, as a BenchDrafter (drop-in for the TS EngineA). */
export class WasmEngineA implements BenchDrafter {
  readonly name = 'engineA-wasm';
  private ctxTail: number[] = [];
  private textTail = '';
  private tools: ToolSchema[] = [...LAB_TOOLS, ...EXTENSION_TOOLS];
  private dialect: CallDialect = 'xml';
  private style: JsonStyle = SPACED_JSON;
  private thinking = true;
  /** The last proposal was drafted inside a tool call (open <tool_call> or a live grammar continuation). */
  toolStep = false;

  constructor(
    readonly engine: EngineAWasm,
    private readonly tok: AdapterTokenizer,
  ) {}

  beginSample(promptText: string, enableThinking?: boolean): void {
    this.engine.begin();
    this.ctxTail = [];
    this.textTail = '';
    const parsed = parseToolsFromPrompt(promptText);
    this.tools = parsed.length ? parsed : [...LAB_TOOLS, ...EXTENSION_TOOLS];
    this.dialect = detectDialect(promptText);
    this.style = detectJsonStyle(promptText);
    this.thinking = enableThinking ?? !/enable_thinking\s*=\s*false/.test(promptText);
  }

  observe(tokens: readonly number[], generated: boolean): void {
    this.engine.observe(tokens, generated);
    for (const t of tokens) this.ctxTail.push(t);
    if (this.ctxTail.length > 6) this.ctxTail = this.ctxTail.slice(-6);
    this.textTail += this.tok.decode(tokens);
    if (this.textTail.length > 4096) this.textTail = this.textTail.slice(-2048);
  }

  /** Same rule as EngineA.alignContinuation. */
  private align(text: string, closed: boolean): number[] | null {
    for (const r of [2, 6]) {
      if (this.ctxTail.length < r) continue;
      const tail = this.ctxTail.slice(-r);
      const enc = this.tok.encode(this.tok.decode(tail) + text);
      let ok = enc.length > r;
      for (let i = 0; ok && i < r; i++) ok = enc[i] === tail[i];
      if (!ok) continue;
      const rest = enc.slice(r);
      if (!closed) rest.pop();
      return rest;
    }
    return null;
  }

  grammarSeqs(): { prior: number; toks: number[] }[] {
    const out: { prior: number; toks: number[] }[] = [];
    for (const c of grammarContinuations(this.textTail, this.tools, this.dialect, this.thinking, this.style)) {
      const toks = this.align(c.text, c.closed);
      if (toks && toks.length) out.push({ prior: c.prob, toks });
    }
    return out;
  }

  propose(budget: DraftBudget): ScoredDraftTree {
    const grammar = budget.maxNodes > 0 ? this.grammarSeqs() : [];
    const open = this.textTail.lastIndexOf('<tool_call>');
    this.toolStep = grammar.length > 0 || (open >= 0 && this.textTail.lastIndexOf('</tool_call>') < open);
    this.engine.setGrammar(grammar);
    return this.engine.propose(budget.maxNodes, budget.tree);
  }

  feedback(_tree: ScoredDraftTree, acceptedPath: readonly number[]): void {
    this.engine.feedback(acceptedPath);
  }

  endSample(): void {
    this.engine.end();
  }
}

export class OffscreenSpecAdapter {
  private closed = false;
  private open() {
    if (this.closed) throw new Error('Offscreen speculative adapter is closed');
  }
  private constructor(
    readonly core: SpecCoreWasm,
    readonly engineA: WasmEngineA,
    readonly router: HybridDraftRouter,
    private readonly pathA: PathAWasm,
    readonly cutoff?: LearnedCutoffWasm,
    private readonly ownsLearnedHead = true,
  ) {}

  static async create(o: OffscreenSpecAdapterOptions): Promise<OffscreenSpecAdapter> {
    const core = await SpecCoreWasm.instantiate(o.wasm);
    const corpus = (o.corpusDocs ?? buildToolCorpus()).map((d) => o.tokenizer.encode(d));
    const engine = core.engineA({ corpus, profile: o.profile, sessionCalibrator: true, draftMs: o.draftMs ?? 0.05, maxCtx: o.maxCtx ?? 32768 });
    const engineA = new WasmEngineA(engine, o.tokenizer);
    let bundle: ReturnType<typeof loadGdnBundle> | undefined;
    const load =
      o.loadLearned ??
      (async (): Promise<LearnedDrafter | null> => {
        if (!o.engineC) return null;
        const { manifestUrl, device, table, readShader, mode } = o.engineC;
        bundle ??= loadGdnBundle(manifestUrl).catch((error) => {
          bundle = undefined;
          throw error;
        });
        const valid = await bundle;
        if (valid && o.engineC.expectedSha256 && valid.sha256 !== o.engineC.expectedSha256)
          throw new Error('Draft head does not match the deployment SHA-256');
        return valid && GdnDraftRuntime.create(device, valid, table, readShader, { mode });
      });
    const threshold = o.threshold ?? 0.7;
    const cutoff = o.cutoff === false ? undefined
      : core.cutoff({ ...o.cutoff, profile: o.profile, threshold, draftMs: o.draftMs ?? 0.05 });
    const policy = cutoff ? adaptivePolicy(cutoff, threshold, () => engineA.toolStep) : thresholdPolicy(threshold);
    return new OffscreenSpecAdapter(
      core,
      engineA,
      new HybridDraftRouter(engineA, load, threshold, policy),
      core.pathA(),
      cutoff,
      o.ownsLearnedHead !== false,
    );
  }

  /** Pre-load Engine B GPU weights and pipelines before turn 1 decode starts. */
  async prewarm(): Promise<void> {
    this.open();
    await this.router.getOrLoadHead();
  }

  beginTurn(promptIds: readonly number[], promptText: string, enableThinking?: boolean): void {
    this.open();
    this.engineA.beginSample(promptText, enableThinking);
    this.engineA.observe(promptIds, false);
    this.cutoff?.beginTurn();
  }

  /**
   * Draft for the next verify step. Pass the anchor (target taps + pending
   * token) to allow the learned head. With the adaptive cutoff, pass
   * `tree = false`: the head drafts one chain that WASM trims to k*.
   */
  propose(maxNodes: number, anchor?: DraftAnchor, tree = true): Promise<RoutedProposal> {
    return this.router.propose({ maxNodes, tree }, anchor);
  }

  /**
   * Single-decision routing helper for hosts that support fused single-submit
   * Draft + Verify (`GdnDraftRuntime.encodeChain` encoded ahead of target verify
   * in the same command buffer, eliminating one GPU submit + mapAsync round-trip).
   *
   * Returns `{ kind: 'fused', runtime, k }` when Engine A has no proposal, the
   * policy approves invoking Engine B, and the loaded head is a `GdnDraftRuntime`.
   * Otherwise resolves the normal `RoutedProposal` without double-calling `shouldInvoke`.
   */
  async planOrPropose(maxNodes: number, anchor?: DraftAnchor, allowFused = true): Promise<FusedDraftPlan> {
    this.open();
    if (!Number.isInteger(maxNodes) || maxNodes < 0 || maxNodes > 15) {
      throw new Error('Invalid draft budget');
    }
    const budget: DraftBudget = { maxNodes, tree: false };
    const tree = this.engineA.propose(budget);
    if (!maxNodes || !anchor || !this.router.shouldInvoke(tree, budget)) {
      return { kind: 'proposal', proposal: { source: 'A', tree } };
    }
    const headCap = this.router.maxHeadNodes(tree, budget);
    if (!headCap) {
      return { kind: 'proposal', proposal: { source: 'A', tree } };
    }
    const head = await this.router.getOrLoadHead();
    if (!head) {
      return { kind: 'proposal', proposal: { source: 'A', tree } };
    }
    if (allowFused && tree.tokens.length === 0 && head instanceof GdnDraftRuntime) {
      return { kind: 'fused', runtime: head, k: headCap };
    }
    const headBudget = headCap === maxNodes ? budget : { maxNodes: headCap, tree: false };
    const learned = await head.propose(anchor, headBudget);
    return {
      kind: 'proposal',
      proposal: { ...this.router.selectProposal(tree, learned, budget), headInvoked: true },
    };
  }

  /** Record the decoded learned chain after a fused Draft + Verify step completes on GPU. */
  commitFused(
    learned: { tokens: readonly number[]; probs?: readonly number[] },
    emitted: readonly number[],
  ): void {
    this.open();
    if (learned.probs && learned.probs.length === learned.tokens.length) {
      this.cutoff?.recordFused(learned.tokens, learned.probs, this.engineA.toolStep);
    }
    if (emitted.length > 0) {
      this.engineA.observe(emitted, true);
      this.cutoff?.commit(emitted);
    }
  }

  /** For the extension's shared scalar/speculative token-emission hook. */
  observeEmitted(ids: readonly number[]): void {
    this.open();
    this.engineA.observe(ids, true);
    this.cutoff?.commit(ids);
  }

  /** After verify: feed the accepted node path (Engine A learns only from its own proposals), then the emitted tokens. */
  commit(proposal: RoutedProposal, acceptedPath: readonly number[], emitted: readonly number[]): void {
    this.open();
    this.router.feedback(proposal, acceptedPath);
    this.engineA.observe(emitted, true);
    this.cutoff?.commit(emitted);
  }

  /** Lossless T > 0 verification from the TOPK_BATCH readback (rngState advanced in place). */
  verifyStochastic(tree: DraftTree, topkWords: Uint32Array, vocab: number, params: { temperature: number; topK: number; topP?: number }, rngState: Uint32Array): VerifyResult {
    this.open();
    return this.pathA.verify(tree, topkWords, vocab, params, rngState);
  }

  rngState(seed: number): Uint32Array {
    return this.core.rngState(seed);
  }

  endTurn(): void {
    this.open();
    this.engineA.endSample();
  }

  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    if (this.ownsLearnedHead) {
      this.router.dispose();
    }
    this.pathA.free();
    this.cutoff?.free();
    this.engineA.engine.free();
  }
}

/**
 * WASM-backed A+B+C policy. Engine A (SAM + grammar) is proposed first. The
 * head runs only when Engine A is not root-confident (and, with the EV gate,
 * only when it could win). A learned chain with per-step probabilities is
 * calibrated and trimmed to k* by spec-core; Engine A's proposal is kept when
 * its expected tokens/ms is higher. Trees or chains without probabilities
 * fall back to the RC1 behaviour.
 */
export function adaptivePolicy(cutoff: LearnedCutoffWasm, threshold: number, toolStep: () => boolean = () => false): RoutePolicy {
  const legacy = thresholdPolicy(threshold);
  return {
    invoke: (a, budget) => cutoff.shouldInvoke(a, budget.maxNodes, toolStep()),
    maxHeadNodes: (_a, budget) => (budget.tree ? budget.maxNodes : cutoff.recommendedK(budget.maxNodes, toolStep())),
    select(a, learned, budget) {
      const chain = learned.parents.every((p, i) => p === i - 1);
      if (!learned.probs || learned.probs.length !== learned.tokens.length || !chain) return legacy.select(a, learned, budget);
      const { k, calibrated } = cutoff.decide(a.probs ?? [], learned.tokens, learned.probs, budget.maxNodes, toolStep());
      if (k < 0) return { source: 'A', tree: a };
      return { source: 'C', tree: { tokens: learned.tokens.slice(0, k), parents: learned.parents.slice(0, k), probs: calibrated.slice(0, k) } };
    },
  };
}
