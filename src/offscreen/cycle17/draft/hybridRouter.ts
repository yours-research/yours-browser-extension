import type { BenchDrafter, DraftBudget, ScoredDraftTree } from '../engines/types.js';
import { loadGdnBundle } from './gdnEagle3Loader.js';
import { GdnDraftRuntime, type DraftAnchor, type ReadShader, type TiedTable } from './gdnEagle3Runtime.js';

export interface RoutedProposal {
  source: 'A' | 'C';
  tree: ScoredDraftTree;
  /** The learned head ran for this step (its GPU time is spent whichever tree won). */
  headInvoked?: boolean;
}

/**
 * Routing decision points. The default reproduces the RC1 router: call the
 * learned head whenever Engine A is not confident at the root and always use
 * its tree. The adaptive WASM policy (OffscreenSpecAdapter) also trims the
 * learned chain to k* and keeps Engine A when that has the higher expected
 * tokens per ms.
 */
export interface RoutePolicy {
  invoke(engineA: ScoredDraftTree, budget: DraftBudget): boolean;
  /** Optional head-chain budget cap before invoking the learned head (0 skips the head). */
  maxHeadNodes?(engineA: ScoredDraftTree, budget: DraftBudget): number;
  select(engineA: ScoredDraftTree, learned: ScoredDraftTree, budget: DraftBudget): Omit<RoutedProposal, 'headInvoked'>;
}

export function thresholdPolicy(threshold: number): RoutePolicy {
  return {
    invoke: (tree) => !tree.parents.some((parent, i) => parent === -1 && (tree.probs?.[i] ?? 0) >= threshold),
    select: (_a, learned) => ({ source: 'C', tree: learned }),
  };
}
interface LearnedDrafter {
  propose(anchor: DraftAnchor, budget: DraftBudget): Promise<ScoredDraftTree>;
  dispose(): void;
}

/** One router per conversation. Feed committed tokens to Engine A on both
 * routes. Only its own proposals should update its acceptance calibrator.
 *
 * By default (`evictOnIdle = false`), once the learned head is loaded it stays
 * resident in VRAM for the lifetime of the router (until `dispose()` or an
 * explicit `releaseIdle()` call). Evicting on every skipped/Engine-A step
 * destroys ~35 MB of GPU buffers + 16 compiled pipelines and triggers ~45 ms
 * cold reloads on every value-gated probe or Engine A ↔ Engine B transition.
 */
export class HybridDraftRouter {
  private pending?: Promise<LearnedDrafter | null>;
  private users = 0;
  private evict = false;
  private closed = false;
  constructor(
    readonly engineA: BenchDrafter,
    private readonly load: () => Promise<LearnedDrafter | null>,
    threshold = 0.7,
    private readonly policy: RoutePolicy = thresholdPolicy(threshold),
    private readonly evictOnIdle = false,
  ) {
    if (!(threshold >= 0 && threshold <= 1)) throw new Error('Confidence threshold must be in [0,1]');
  }
  /** Release the resident learned-head GPU resources when idle (e.g. on memory pressure or long idle timeout). */
  releaseIdle() {
    this.evict = true;
    if (this.users || !this.pending) return;
    const old = this.pending;
    this.pending = undefined;
    void old.then(
      (head) => head?.dispose(),
      () => {},
    );
  }
  shouldInvoke(engineA: ScoredDraftTree, budget: DraftBudget): boolean {
    return this.policy.invoke(engineA, budget);
  }
  maxHeadNodes(engineA: ScoredDraftTree, budget: DraftBudget): number {
    return this.policy.maxHeadNodes
      ? Math.min(budget.maxNodes, Math.max(0, this.policy.maxHeadNodes(engineA, budget)))
      : budget.maxNodes;
  }
  selectProposal(engineA: ScoredDraftTree, learned: ScoredDraftTree, budget: DraftBudget): Omit<RoutedProposal, 'headInvoked'> {
    return this.policy.select(engineA, learned, budget);
  }
  /** Load (once) and return the resident learned head without invoking `propose()`. */
  async getOrLoadHead(): Promise<LearnedDrafter | null> {
    if (this.closed) throw new Error('Draft router is closed');
    this.users++;
    this.evict = false;
    const pending = (this.pending ??= Promise.resolve().then(this.load));
    try {
      const head = await pending;
      if (this.closed) return null;
      return head;
    } catch (error) {
      if (this.pending === pending) this.releaseIdle();
      throw error;
    } finally {
      this.users--;
      if (this.evict || this.closed) this.releaseIdle();
    }
  }
  async propose(budget: DraftBudget, anchor?: DraftAnchor): Promise<RoutedProposal> {
    if (this.closed) throw new Error('Draft router is closed');
    if (!Number.isInteger(budget.maxNodes) || budget.maxNodes < 0 || budget.maxNodes > 15)
      throw new Error('Invalid draft budget');
    const tree = this.engineA.propose(budget);
    if (!budget.maxNodes || !anchor || !this.policy.invoke(tree, budget)) {
      if (this.evictOnIdle) this.releaseIdle();
      return { source: 'A', tree };
    }
    const headCap = this.policy.maxHeadNodes ? Math.min(budget.maxNodes, Math.max(0, this.policy.maxHeadNodes(tree, budget))) : budget.maxNodes;
    if (!headCap) {
      if (this.evictOnIdle) this.releaseIdle();
      return { source: 'A', tree };
    }
    const headBudget = headCap === budget.maxNodes ? budget : { ...budget, maxNodes: headCap };
    this.users++;
    this.evict = false;
    const pending = (this.pending ??= Promise.resolve().then(this.load));
    try {
      const head = await pending;
      if (!head) return { source: 'A', tree };
      const learned = await head.propose(anchor, headBudget);
      return { ...this.policy.select(tree, learned, budget), headInvoked: true };
    } catch (error) {
      if (this.pending === pending) this.releaseIdle();
      throw error;
    } finally {
      this.users--;
      if (this.evict || this.closed) this.releaseIdle();
    }
  }
  feedback(proposal: RoutedProposal, path: readonly number[]) {
    if (proposal.source === 'A') this.engineA.feedback?.(proposal.tree, path);
  }
  dispose() {
    this.closed = true;
    this.releaseIdle();
  }
}

/** Fetch once on first demand. High-confidence Engine A resumes with zero
 * draft-head VRAM. A later cold span can re-upload the validated CPU payload. */
export function createGdnRouter(
  engineA: BenchDrafter,
  options: {
    manifestUrl?: URL;
    device: GPUDevice;
    table: TiedTable;
    readShader: ReadShader;
    threshold?: number;
  },
) {
  let bundle: ReturnType<typeof loadGdnBundle> | undefined;
  return new HybridDraftRouter(
    engineA,
    async () => {
      if (!options.manifestUrl) return null;
      bundle ??= loadGdnBundle(options.manifestUrl).catch((error) => {
        bundle = undefined;
        throw error;
      });
      const valid = await bundle;
      return valid && GdnDraftRuntime.create(options.device, valid, options.table, options.readShader);
    },
    options.threshold,
  );
}
