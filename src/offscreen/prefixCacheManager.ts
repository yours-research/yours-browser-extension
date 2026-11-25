import type { ChatMessage } from '../shared/protocol.ts';

export const BASE_SYSTEM_SLOT_ID = 'base-system';
export const TURN_CHECKPOINT_SLOT_ID = 'turn-checkpoint';

export interface GpuPrefixSlotBackend {
  readonly activeSeqPos: number;
  saveGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; byteSize: number }>;
  restoreGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; restoreMs: number }>;
  deleteGpuPrefixSlot(slotId: string): boolean;
  resetState(): Promise<void>;
}

export type PrefixSlotKind = 'base' | 'tab' | 'draft' | 'checkpoint';

export interface PrefixSlotMeta {
  url?: string;
  contentHash?: string;
}

export interface PrefixSlotEntry {
  slotId: string;
  kind: PrefixSlotKind;
  tokenIds: number[];
  messages: ChatMessage[];
  seqPos: number;
  byteSize: number;
  lastUsedAt: number;
  meta?: PrefixSlotMeta;
}

export interface PrefixMatchResult {
  source: 'live' | 'slot' | 'none';
  slotId: string | null;
  matchedTokens: number;
  slot?: PrefixSlotEntry;
}

export interface PrefixCacheStats {
  hasBasePrompt: boolean;
  hasTurnCheckpoint: boolean;
  tabSlotCount: number;
  maxTabSlots: number;
  totalSlots: number;
  totalBytes: number;
  slots: Array<{
    slotId: string;
    kind: PrefixSlotKind;
    tokens: number;
    seqPos: number;
    byteSize: number;
    lastUsedAt: number;
    meta?: PrefixSlotMeta;
  }>;
}

export interface PrefixCacheManagerOptions {
  maxTabSlots?: number;
  now?: () => number;
}

/**
 * DeltaNet is a recurrent state machine: a cached prefix is valid ONLY when
 * prefix is non-empty, no longer than full, and matches every token index identically.
 */
export function isExactTokenPrefix(prefix: readonly number[], full: readonly number[]): boolean {
  if (prefix.length === 0 || prefix.length > full.length) {
    return false;
  }
  for (let i = 0; i < prefix.length; i += 1) {
    if (prefix[i] !== full[i]) {
      return false;
    }
  }
  return true;
}

function cloneMessages(messages: readonly ChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}

interface InternalSlotRecord {
  entry: PrefixSlotEntry;
  touchSeq: number;
}

export class PrefixCacheManager {
  private readonly backend: GpuPrefixSlotBackend;
  private readonly maxTabSlots: number;
  private readonly now: () => number;
  private readonly slots = new Map<string, InternalSlotRecord>();
  private touchCounter = 0;

  constructor(
    backend: GpuPrefixSlotBackend,
    optionsOrMaxTabSlots: number | PrefixCacheManagerOptions = {},
  ) {
    this.backend = backend;
    if (typeof optionsOrMaxTabSlots === 'number') {
      this.maxTabSlots = Math.max(0, Math.floor(optionsOrMaxTabSlots));
      this.now = () => Date.now();
    } else {
      this.maxTabSlots = Math.max(0, Math.floor(optionsOrMaxTabSlots.maxTabSlots ?? 3));
      this.now = optionsOrMaxTabSlots.now ?? (() => Date.now());
    }
  }

  private nextTouchSeq(): number {
    this.touchCounter += 1;
    return this.touchCounter;
  }

  findBestMatch(targetTokenIds: readonly number[], liveTokenIds: readonly number[]): PrefixMatchResult {
    const liveMatched = isExactTokenPrefix(liveTokenIds, targetTokenIds) ? liveTokenIds.length : 0;

    let bestSlot: PrefixSlotEntry | undefined;
    let bestSlotMatched = 0;

    for (const record of this.slots.values()) {
      const { entry } = record;
      if (
        entry.tokenIds.length > bestSlotMatched &&
        isExactTokenPrefix(entry.tokenIds, targetTokenIds)
      ) {
        bestSlot = entry;
        bestSlotMatched = entry.tokenIds.length;
      }
    }

    // Break ties in favor of 'live' (0 ms GPU state copy).
    if (liveMatched > 0 && liveMatched >= bestSlotMatched) {
      return {
        source: 'live',
        slotId: null,
        matchedTokens: liveMatched,
      };
    }

    if (bestSlot && bestSlotMatched > 0) {
      return {
        source: 'slot',
        slotId: bestSlot.slotId,
        matchedTokens: bestSlotMatched,
        slot: bestSlot,
      };
    }

    return {
      source: 'none',
      slotId: null,
      matchedTokens: 0,
    };
  }

  async saveBasePromptSlot(
    tokenIds: readonly number[],
    messages: readonly ChatMessage[],
  ): Promise<PrefixSlotEntry> {
    const saved = await this.backend.saveGpuPrefixSlot(BASE_SYSTEM_SLOT_ID);
    const entry: PrefixSlotEntry = {
      slotId: BASE_SYSTEM_SLOT_ID,
      kind: 'base',
      tokenIds: [...tokenIds],
      messages: cloneMessages(messages),
      seqPos: saved.seqPos,
      byteSize: saved.byteSize,
      lastUsedAt: this.now(),
    };
    this.slots.set(BASE_SYSTEM_SLOT_ID, {
      entry,
      touchSeq: this.nextTouchSeq(),
    });
    return entry;
  }

  async saveTabSlot(
    slotKey: string,
    tokenIds: readonly number[],
    messages: readonly ChatMessage[],
    meta?: PrefixSlotMeta,
  ): Promise<PrefixSlotEntry> {
    if (slotKey === BASE_SYSTEM_SLOT_ID || slotKey === TURN_CHECKPOINT_SLOT_ID) {
      throw new Error(`Reserved prefix slot key cannot be used for tab/draft cache: ${slotKey}`);
    }

    const kind: 'tab' | 'draft' = slotKey.startsWith('draft') ? 'draft' : 'tab';
    const saved = await this.backend.saveGpuPrefixSlot(slotKey);
    const entry: PrefixSlotEntry = {
      slotId: slotKey,
      kind,
      tokenIds: [...tokenIds],
      messages: cloneMessages(messages),
      seqPos: saved.seqPos,
      byteSize: saved.byteSize,
      lastUsedAt: this.now(),
      meta: meta ? { ...meta } : undefined,
    };

    this.slots.set(slotKey, {
      entry,
      touchSeq: this.nextTouchSeq(),
    });

    this.evictLruTabSlotsIfNeeded(slotKey);
    return entry;
  }

  async saveTurnCheckpoint(
    tokenIds: readonly number[],
    messages: readonly ChatMessage[],
  ): Promise<PrefixSlotEntry> {
    const saved = await this.backend.saveGpuPrefixSlot(TURN_CHECKPOINT_SLOT_ID);
    const entry: PrefixSlotEntry = {
      slotId: TURN_CHECKPOINT_SLOT_ID,
      kind: 'checkpoint',
      tokenIds: [...tokenIds],
      messages: cloneMessages(messages),
      seqPos: saved.seqPos,
      byteSize: saved.byteSize,
      lastUsedAt: this.now(),
    };
    this.slots.set(TURN_CHECKPOINT_SLOT_ID, {
      entry,
      touchSeq: this.nextTouchSeq(),
    });
    return entry;
  }

  async restoreSlot(
    slotId: string,
  ): Promise<{ seqPos: number; restoreMs: number; entry: PrefixSlotEntry }> {
    const record = this.slots.get(slotId);
    if (!record) {
      throw new Error(`Prefix slot not found: ${slotId}`);
    }
    const restored = await this.backend.restoreGpuPrefixSlot(slotId);
    record.entry.lastUsedAt = this.now();
    record.touchSeq = this.nextTouchSeq();
    return {
      seqPos: restored.seqPos,
      restoreMs: restored.restoreMs,
      entry: record.entry,
    };
  }

  async resetToBasePrompt(): Promise<{
    seqPos: number;
    restoreMs: number;
    entry: PrefixSlotEntry;
  } | null> {
    if (!this.hasBasePrompt()) {
      await this.backend.resetState();
      return null;
    }
    return this.restoreSlot(BASE_SYSTEM_SLOT_ID);
  }

  hasBasePrompt(): boolean {
    return this.slots.has(BASE_SYSTEM_SLOT_ID);
  }

  getBasePromptInfo(): { cached: boolean; tokens: number } {
    const record = this.slots.get(BASE_SYSTEM_SLOT_ID);
    if (!record) {
      return { cached: false, tokens: 0 };
    }
    return {
      cached: true,
      tokens: record.entry.tokenIds.length,
    };
  }

  clearVolatileSlots(): void {
    for (const [slotId, record] of this.slots.entries()) {
      if (record.entry.kind === 'base') {
        continue;
      }
      this.backend.deleteGpuPrefixSlot(slotId);
      this.slots.delete(slotId);
    }
  }

  clearAll(): void {
    for (const slotId of this.slots.keys()) {
      this.backend.deleteGpuPrefixSlot(slotId);
    }
    this.slots.clear();
  }

  getStats(): PrefixCacheStats {
    let tabSlotCount = 0;
    let totalBytes = 0;
    const slots: PrefixCacheStats['slots'] = [];

    for (const { entry } of this.slots.values()) {
      if (entry.kind === 'tab' || entry.kind === 'draft') {
        tabSlotCount += 1;
      }
      totalBytes += entry.byteSize;
      slots.push({
        slotId: entry.slotId,
        kind: entry.kind,
        tokens: entry.tokenIds.length,
        seqPos: entry.seqPos,
        byteSize: entry.byteSize,
        lastUsedAt: entry.lastUsedAt,
        meta: entry.meta ? { ...entry.meta } : undefined,
      });
    }

    return {
      hasBasePrompt: this.slots.has(BASE_SYSTEM_SLOT_ID),
      hasTurnCheckpoint: this.slots.has(TURN_CHECKPOINT_SLOT_ID),
      tabSlotCount,
      maxTabSlots: this.maxTabSlots,
      totalSlots: this.slots.size,
      totalBytes,
      slots,
    };
  }

  private evictLruTabSlotsIfNeeded(protectedSlotKey: string): void {
    const volatileEntries: InternalSlotRecord[] = [];
    for (const record of this.slots.values()) {
      if (record.entry.kind === 'tab' || record.entry.kind === 'draft') {
        volatileEntries.push(record);
      }
    }

    if (volatileEntries.length <= this.maxTabSlots) {
      return;
    }

    volatileEntries.sort((a, b) => {
      if (a.entry.lastUsedAt !== b.entry.lastUsedAt) {
        return a.entry.lastUsedAt - b.entry.lastUsedAt;
      }
      return a.touchSeq - b.touchSeq;
    });

    let remaining = volatileEntries.length;
    for (const candidate of volatileEntries) {
      if (remaining <= this.maxTabSlots) {
        break;
      }
      if (this.maxTabSlots > 0 && candidate.entry.slotId === protectedSlotKey) {
        continue;
      }
      this.backend.deleteGpuPrefixSlot(candidate.entry.slotId);
      this.slots.delete(candidate.entry.slotId);
      remaining -= 1;
    }
  }
}
