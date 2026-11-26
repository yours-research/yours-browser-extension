import { describe, expect, it } from 'vitest';
import type { MessageNode } from '@yours/harness';
import {
  countConversationTurns,
  createCheckpointSnapshot,
  formatCheckpointTimestamp,
  upsertCheckpointRecord,
} from '../src/sidepanel/sessionCheckpoints.ts';

function makeNode(
  id: string,
  role: 'user' | 'assistant' | 'system',
  content: string,
): MessageNode {
  return {
    id,
    threadId: 'thread-ckpt',
    parentId: null,
    childrenIds: [],
    createdAt: 1000,
    message: {
      id,
      role,
      content,
      createdAt: 1000,
    },
  };
}

describe('sessionCheckpoints', () => {
  it('captures turn count, pinned tab count, context snapshot, and trimmed summary', () => {
    const nodes = [
      makeNode('sys', 'system', 'System prompt'),
      makeNode('u1', 'user', 'Check cluster health'),
      makeNode('a1', 'assistant', 'Alpha Cluster Spec is healthy at 415 req/s.'),
    ];
    expect(countConversationTurns(nodes)).toBe(1);

    const cp = createCheckpointSnapshot({
      threadId: 'thread-ckpt',
      nodes,
      pinnedTabCount: 1,
      contextSnapshot: '1.2K / 32.5K tok',
      nowMs: 1_700_000_000_000,
    });

    expect(cp).not.toBeNull();
    expect(cp?.leafNodeId).toBe('a1');
    expect(cp?.turnCount).toBe(1);
    expect(cp?.pinnedTabCount).toBe(1);
    expect(cp?.contextSnapshot).toBe('1.2K / 32.5K tok');
    expect(cp?.summary).toContain('Alpha Cluster Spec');
    expect(formatCheckpointTimestamp(Date.now())).toBe('Just now');
  });

  it('returns null on empty thread and deduplicates/caps checkpoints per thread', () => {
    expect(
      createCheckpointSnapshot({
        threadId: 'thread-ckpt',
        nodes: [makeNode('sys', 'system', 'System prompt')],
        pinnedTabCount: 0,
        contextSnapshot: '0 / 32K tok',
      }),
    ).toBeNull();

    let list: ReturnType<typeof upsertCheckpointRecord> = [];
    for (let i = 1; i <= 8; i++) {
      const cp = createCheckpointSnapshot({
        threadId: 'thread-ckpt',
        nodes: [makeNode(`u${i}`, 'user', `Q${i}`), makeNode(`a${i}`, 'assistant', `A${i}`)],
        pinnedTabCount: 0,
        contextSnapshot: `${i}00 / 32K tok`,
        nowMs: 1_700_000_000_000 + i * 1000,
      })!;
      list = upsertCheckpointRecord(list, cp);
    }

    expect(list).toHaveLength(6);
    expect(list[0]?.leafNodeId).toBe('a8');
  });
});
