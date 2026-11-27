/** Common interface for benchmarked drafters (Engine A, baselines, ablations). */
import { DraftTree } from '../verifier/rejection.js';

export interface ScoredDraftTree extends DraftTree {
  /** Path probability estimate of each node (best-first order is not required). */
  probs?: number[];
  /** Calibration feature key per node (for feedback). */
  keys?: string[];
}

export interface DraftBudget {
  /** Hard cap on drafted nodes (verify M = nodes + 1). */
  maxNodes: number;
  /** Allow branching (token tree) or force a single chain. */
  tree: boolean;
}

export interface BenchDrafter {
  readonly name: string;
  /** Start a new conversation sample. `promptText` lets grammar-aware drafters read tool schemas. */
  beginSample(promptText: string): void;
  /** Append tokens to the live context (prefill segments or accepted+bonus tokens). */
  observe(tokens: readonly number[], generated: boolean): void;
  /** Propose a draft for the next position. */
  propose(budget: DraftBudget): ScoredDraftTree;
  /** Verify outcome: accepted node indices along the path (root excluded). */
  feedback?(tree: ScoredDraftTree, acceptedPath: readonly number[]): void;
  /** Sample finished (persist cross-session memory etc.). */
  endSample?(): void;
}
