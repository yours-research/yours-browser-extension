import type { MessageNode } from '@yours/harness';

export interface SessionCheckpointRecord {
  id: string;
  threadId: string;
  leafNodeId: string;
  title: string;
  summary: string;
  savedAt: number;
  timestampLabel: string;
  turnCount: number;
  pinnedTabCount: number;
  contextSnapshot: string;
}

const CHECKPOINTS_STORAGE_KEY = 'yours-session-checkpoints';
const MAX_CHECKPOINTS_PER_THREAD = 6;

export function formatCheckpointTimestamp(timestampMs: number): string {
  const diffSec = Math.max(0, Math.round((Date.now() - timestampMs) / 1000));
  if (diffSec < 15) return 'Just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  try {
    return new Date(timestampMs).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return 'Saved';
  }
}

export function countConversationTurns(nodes: readonly MessageNode[]): number {
  let turns = 0;
  for (const node of nodes) {
    if (node.message.role === 'user') {
      turns += 1;
    }
  }
  return turns;
}

export function createCheckpointSnapshot(params: {
  threadId: string;
  nodes: readonly MessageNode[];
  pinnedTabCount: number;
  contextSnapshot: string;
  label?: string;
  nowMs?: number;
}): SessionCheckpointRecord | null {
  const visible = params.nodes.filter((n) => n.message.role !== 'system');
  const leaf = visible[visible.length - 1];
  if (!params.threadId || !leaf) {
    return null;
  }
  const nowMs = params.nowMs ?? Date.now();
  const turnCount = countConversationTurns(visible);
  const lastAssistant = [...visible].reverse().find((n) => n.message.role === 'assistant' && n.message.content.trim());
  const previewSource = lastAssistant?.message.content ?? leaf.message.content ?? '';
  const summaryClean = previewSource.replace(/\s+/g, ' ').trim();
  const summary =
    summaryClean.length > 72 ? `${summaryClean.slice(0, 71).trimEnd()}…` : summaryClean || 'Saved thread state';

  return {
    id: `ckpt-${nowMs}-${Math.random().toString(36).slice(2, 7)}`,
    threadId: params.threadId,
    leafNodeId: leaf.id,
    title: params.label?.trim() || `Checkpoint · Turn ${Math.max(1, turnCount)}`,
    summary,
    savedAt: nowMs,
    timestampLabel: formatCheckpointTimestamp(nowMs),
    turnCount: Math.max(1, turnCount),
    pinnedTabCount: Math.max(0, params.pinnedTabCount),
    contextSnapshot: params.contextSnapshot,
  };
}

export function upsertCheckpointRecord(
  existing: readonly SessionCheckpointRecord[],
  record: SessionCheckpointRecord,
): SessionCheckpointRecord[] {
  const withoutDup = existing.filter(
    (item) => !(item.threadId === record.threadId && item.leafNodeId === record.leafNodeId),
  );
  const next = [record, ...withoutDup];
  const counts = new Map<string, number>();
  return next.filter((item) => {
    const c = (counts.get(item.threadId) ?? 0) + 1;
    counts.set(item.threadId, c);
    return c <= MAX_CHECKPOINTS_PER_THREAD;
  });
}

export function loadSavedCheckpoints(): SessionCheckpointRecord[] {
  if (typeof window === 'undefined' || !window.localStorage) return [];
  try {
    const raw = window.localStorage.getItem(CHECKPOINTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is SessionCheckpointRecord =>
        Boolean(item) &&
        typeof item === 'object' &&
        typeof (item as SessionCheckpointRecord).id === 'string' &&
        typeof (item as SessionCheckpointRecord).threadId === 'string' &&
        typeof (item as SessionCheckpointRecord).leafNodeId === 'string',
    );
  } catch {
    return [];
  }
}

export function persistSavedCheckpoints(records: readonly SessionCheckpointRecord[]): void {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(CHECKPOINTS_STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Ignore storage quota errors
  }
}
