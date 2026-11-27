/**
 * P9b reference: defer the DeltaNet rollback of a rejected speculative step
 * into the next queue submission instead of waiting for it.
 *
 * At reject time the extension encodes the rollback (state restore + replay)
 * into its own command buffer and stores it here. Nothing waits on the GPU.
 *  - The hot path (`decodeSpeculativeStep`, chain verify) submits
 *    `[pendingRollback, verify]` in ONE `queue.submit`. Its `writeBuffer`
 *    calls touch only `tokenIdsBuf` and the per-plan verify uniforms, which
 *    the rollback does not read, so queue order is preserved. It must flush
 *    before `ensureKvCapacity` grows the cache.
 *  - Every other entry point flushes FIRST, before any `writeBuffer`, buffer
 *    reallocation or readback: scalar `decodeStep` (it submits once per
 *    LM-head slice), prefill, background/context prefill, slot
 *    save/restore, `resetState`, `snapshotState`/`restoreState`, KV growth in
 *    `ensureKvCapacity`, the tree step, end of turn and `dispose`.
 * The rollback still writes its uniforms through `prefillUniformBuf` at
 * reject time; flushing before prefill keeps that write ahead of any later
 * prefill write in queue order. Generic over the command-buffer type so the
 * CPU tests drive it with a simulated queue.
 */
export type RollbackFlushPoint =
  | 'decode-step'
  | 'prefill'
  | 'prefill-background'
  | 'slot-save'
  | 'slot-restore'
  | 'reset'
  | 'snapshot'
  | 'restore-snapshot'
  | 'kv-grow'
  | 'tree-step'
  | 'end-of-turn'
  | 'dispose';

/** Entry points that may merge the pending rollback into their own submit. */
export const ROLLBACK_MERGE_POINTS = ['spec-step'] as const;
export const ROLLBACK_FLUSH_POINTS: readonly RollbackFlushPoint[] = [
  'decode-step', 'prefill', 'prefill-background', 'slot-save', 'slot-restore', 'reset', 'snapshot',
  'restore-snapshot', 'kv-grow', 'tree-step', 'end-of-turn', 'dispose',
];

export class DeferredRollback<CommandBuffer> {
  private pending: CommandBuffer | null = null;
  /** Counters for telemetry (P10 `rollbackWaitMs` becomes 0 for deferred rollbacks). */
  readonly stats = { deferred: 0, merged: 0, flushed: 0, flushedBy: {} as Partial<Record<RollbackFlushPoint, number>> };

  constructor(private readonly submit: (buffers: CommandBuffer[]) => void) {}

  get isPending(): boolean {
    return this.pending !== null;
  }

  /** Store a finished rollback command buffer. At most one can be pending. */
  defer(rollback: CommandBuffer): void {
    if (this.pending !== null) throw new Error('A DeltaNet rollback is already pending; flush before the next verify step');
    this.pending = rollback;
    this.stats.deferred++;
  }

  /** Hot-path submit: the pending rollback (if any) runs first, in the same queue.submit. */
  submitWith(buffers: CommandBuffer[]): void {
    const p = this.pending;
    this.pending = null;
    if (p !== null) this.stats.merged++;
    this.submit(p === null ? buffers : [p, ...buffers]);
  }

  /** Submit the pending rollback alone. Call before any non-hot-path GPU work. */
  flush(point: RollbackFlushPoint): void {
    const p = this.pending;
    if (p === null) return;
    this.pending = null;
    this.stats.flushed++;
    this.stats.flushedBy[point] = (this.stats.flushedBy[point] ?? 0) + 1;
    this.submit([p]);
  }

  /** Device loss or teardown: drop without submitting (the state is being discarded). */
  discard(): void {
    this.pending = null;
  }
}
