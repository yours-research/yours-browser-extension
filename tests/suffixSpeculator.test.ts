import { describe, expect, it } from 'vitest';
import { proposeSuffixDraft, SuffixSpeculator } from '../src/offscreen/suffixSpeculator.ts';

describe('SuffixSpeculator', () => {
  it('proposes up to maxDraft continuation tokens from an earlier suffix match', () => {
    const history = [10, 20, 30, 40, 50, 60, 70, 80, 99, 10, 20, 30, 40];
    const spec = new SuffixSpeculator(history);
    expect(spec.propose()).toEqual([50, 60, 70, 80]);
    expect(spec.propose({ maxDraft: 2 })).toEqual([50, 60]);
    expect(proposeSuffixDraft(history)).toEqual([50, 60, 70, 80]);
  });

  it('prefers longer suffix matches over shorter ones and breaks ties by most recent match', () => {
    // Earlier match of length 4 ([1,2,3,4] -> [100,101]) vs later match of length 3 ([2,3,4] -> [200,201])
    const history = [1, 2, 3, 4, 100, 101, 102, 103, 9, 2, 3, 4, 200, 201, 8, 1, 2, 3, 4];
    const spec = new SuffixSpeculator(history);
    expect(spec.propose({ minMatch: 3, maxMatch: 8, maxDraft: 3 })).toEqual([100, 101, 102]);

    // Tie in length: should pick the more recent occurrence
    const tieHistory = [1, 2, 3, 11, 12, 99, 1, 2, 3, 21, 22, 88, 1, 2, 3];
    const tieSpec = new SuffixSpeculator(tieHistory);
    expect(tieSpec.propose({ minMatch: 3, maxMatch: 8, maxDraft: 2 })).toEqual([21, 22]);
  });

  it('returns empty array when no match exists or history is too short', () => {
    const spec = new SuffixSpeculator([1, 2, 3]);
    expect(spec.propose()).toEqual([]);

    spec.pushTokens([4, 5, 6, 7]);
    expect(spec.propose()).toEqual([]);
    expect(proposeSuffixDraft([1, 2])).toEqual([]);
  });

  it('supports incremental updates, external history sync, reset, and minMatch < 3', () => {
    const spec = new SuffixSpeculator();
    spec.pushTokens([5, 6, 7, 8, 9, 10]);
    spec.pushTokens([5, 6, 7]);
    expect(spec.propose()).toEqual([8, 9, 10, 5]);

    // External history extension
    expect(spec.propose([5, 6, 7, 8, 9, 10, 5, 6, 7, 8])).toEqual([9, 10, 5, 6]);

    // minMatch = 2 fallback
    expect(spec.propose([10, 20, 30, 40, 20, 30], { minMatch: 2, maxDraft: 2 })).toEqual([40, 20]);

    spec.reset([1, 2, 3, 4, 1, 2, 3]);
    expect(spec.propose({ maxDraft: 1 })).toEqual([4]);
  });

  it('maintains sub-millisecond amortized lookup across 128K-token context', () => {
    const history = new Array<number>(131072);
    for (let i = 0; i < history.length; i++) {
      history[i] = (i * 2654435761) % 240000;
    }
    // Inject a known 6-token phrase near 20K and repeat its suffix at the end
    history[20000] = 42;
    history[20001] = 43;
    history[20002] = 44;
    history[20003] = 45;
    history[20004] = 46;
    history[20005] = 47;
    history[20006] = 777;
    history[20007] = 888;
    history[20008] = 999;

    const spec = new SuffixSpeculator(history);
    spec.pushTokens([42, 43, 44, 45, 46, 47]);

    const t0 = performance.now();
    let last: number[] = [];
    for (let i = 0; i < 1000; i++) {
      last = spec.propose({ maxDraft: 3 });
    }
    const elapsedMs = performance.now() - t0;

    expect(last).toEqual([777, 888, 999]);
    expect(elapsedMs).toBeLessThan(50);
  });
});
