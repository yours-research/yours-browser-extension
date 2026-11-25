export interface SuffixSpeculateOptions {
  minMatch?: number;
  maxMatch?: number;
  maxDraft?: number;
}

const DEFAULT_MIN_MATCH = 3;
const DEFAULT_MAX_MATCH = 8;
const DEFAULT_MAX_DRAFT = 4;
const MAX_POSITIONS_PER_TRIGRAM = 16;

function hashTrigram(a: number, b: number, c: number): number {
  return (Math.imul(a, 131071) ^ Math.imul(b, 65521) ^ (c | 0)) >>> 0;
}

function scanLongestSuffixMatch(
  history: readonly number[],
  minMatch: number,
  maxMatch: number,
  maxDraft: number,
): number[] {
  const n = history.length;
  if (maxDraft <= 0 || n < minMatch + 1) {
    return [];
  }
  const lastIdx = n - 1;
  const maxCheck = Math.min(maxMatch, n - 1);
  let bestLen = minMatch - 1;
  let bestEndPos = -1;

  for (let endPos = n - 2; endPos >= minMatch - 1; endPos--) {
    if (history[endPos] !== history[lastIdx]) {
      continue;
    }
    let len = 1;
    while (
      len < maxCheck &&
      endPos - len >= 0 &&
      history[endPos - len] === history[lastIdx - len]
    ) {
      len++;
    }
    if (len > bestLen) {
      bestLen = len;
      bestEndPos = endPos;
      if (bestLen >= maxMatch) {
        break;
      }
    }
  }

  if (bestEndPos < 0) {
    return [];
  }
  const outEnd = Math.min(n, bestEndPos + 1 + maxDraft);
  return history.slice(bestEndPos + 1, outEnd);
}

/**
 * Stateless helper for one-off suffix n-gram lookup over a token history.
 */
export function proposeSuffixDraft(
  history: readonly number[],
  options?: SuffixSpeculateOptions,
): number[] {
  const minMatch = Math.max(1, options?.minMatch ?? DEFAULT_MIN_MATCH);
  const maxMatch = Math.max(minMatch, options?.maxMatch ?? DEFAULT_MAX_MATCH);
  const maxDraft = Math.max(0, options?.maxDraft ?? DEFAULT_MAX_DRAFT);
  return scanLongestSuffixMatch(history, minMatch, maxMatch, maxDraft);
}

/**
 * Zero-weight prompt-lookup / suffix speculative decoder using an incremental
 * trigram position index for O(1) amortized draft proposals up to 128K context.
 */
export class SuffixSpeculator {
  private history: number[] = [];
  private trigramIndex = new Map<number, number[]>();

  constructor(initialHistory?: readonly number[]) {
    if (initialHistory && initialHistory.length > 0) {
      this.pushTokens(initialHistory);
    }
  }

  reset(initialHistory?: readonly number[]): void {
    this.history = [];
    this.trigramIndex.clear();
    if (initialHistory && initialHistory.length > 0) {
      this.pushTokens(initialHistory);
    }
  }

  getHistory(): readonly number[] {
    return this.history;
  }

  pushToken(tokenId: number): void {
    const idx = this.history.length;
    this.history.push(tokenId);
    if (idx >= 2) {
      const h = hashTrigram(this.history[idx - 2], this.history[idx - 1], tokenId);
      let list = this.trigramIndex.get(h);
      if (!list) {
        list = [idx];
        this.trigramIndex.set(h, list);
      } else {
        list.push(idx);
        if (list.length > MAX_POSITIONS_PER_TRIGRAM) {
          list.shift();
        }
      }
    }
  }

  pushTokens(tokenIds: readonly number[]): void {
    for (let i = 0; i < tokenIds.length; i++) {
      this.pushToken(tokenIds[i]);
    }
  }

  private syncExternalHistory(external: readonly number[]): void {
    const curLen = this.history.length;
    if (external.length >= curLen) {
      let matchesPrefix = true;
      // Fast check tail & head before full verification if needed
      if (
        curLen > 0 &&
        (external[0] !== this.history[0] || external[curLen - 1] !== this.history[curLen - 1])
      ) {
        matchesPrefix = false;
      } else {
        for (let i = 0; i < curLen; i++) {
          if (external[i] !== this.history[i]) {
            matchesPrefix = false;
            break;
          }
        }
      }
      if (matchesPrefix) {
        for (let i = curLen; i < external.length; i++) {
          this.pushToken(external[i]);
        }
        return;
      }
    }
    this.reset(external);
  }

  propose(options?: SuffixSpeculateOptions): number[];
  propose(history: readonly number[], options?: SuffixSpeculateOptions): number[];
  propose(
    historyOrOptions?: readonly number[] | SuffixSpeculateOptions,
    maybeOptions?: SuffixSpeculateOptions,
  ): number[] {
    let options: SuffixSpeculateOptions | undefined;
    if (Array.isArray(historyOrOptions)) {
      this.syncExternalHistory(historyOrOptions);
      options = maybeOptions;
    } else {
      options = historyOrOptions as SuffixSpeculateOptions | undefined;
    }

    const minMatch = Math.max(1, options?.minMatch ?? DEFAULT_MIN_MATCH);
    const maxMatch = Math.max(minMatch, options?.maxMatch ?? DEFAULT_MAX_MATCH);
    const maxDraft = Math.max(0, options?.maxDraft ?? DEFAULT_MAX_DRAFT);

    const n = this.history.length;
    if (maxDraft <= 0 || n < minMatch + 1) {
      return [];
    }

    if (minMatch < 3) {
      return scanLongestSuffixMatch(this.history, minMatch, maxMatch, maxDraft);
    }

    const last0 = this.history[n - 3];
    const last1 = this.history[n - 2];
    const last2 = this.history[n - 1];
    const h = hashTrigram(last0, last1, last2);
    const candidates = this.trigramIndex.get(h);
    if (!candidates || candidates.length === 0) {
      return [];
    }

    const maxCheck = Math.min(maxMatch, n - 1);
    let bestLen = minMatch - 1;
    let bestEndPos = -1;

    for (let i = candidates.length - 1; i >= 0; i--) {
      const endPos = candidates[i];
      if (endPos >= n - 1 || endPos < 2) {
        continue;
      }
      if (
        this.history[endPos] !== last2 ||
        this.history[endPos - 1] !== last1 ||
        this.history[endPos - 2] !== last0
      ) {
        continue;
      }
      let len = 3;
      while (
        len < maxCheck &&
        endPos - len >= 0 &&
        this.history[endPos - len] === this.history[n - 1 - len]
      ) {
        len++;
      }
      if (len > bestLen) {
        bestLen = len;
        bestEndPos = endPos;
        if (bestLen >= maxMatch) {
          break;
        }
      }
    }

    if (bestEndPos < 0) {
      return [];
    }
    const outEnd = Math.min(n, bestEndPos + 1 + maxDraft);
    return this.history.slice(bestEndPos + 1, outEnd);
  }
}
