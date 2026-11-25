import type { ThreadMessage } from '@assistant-ui/react';
import { describe, expect, it, vi } from 'vitest';
import type { ClientToEngineMessage, EngineToClientMessage } from '../src/shared/protocol.ts';
import { createTinyGptChatAdapter } from '../src/sidepanel/chatAdapter.ts';
import { OffscreenEngineClient, type EnginePortLike } from '../src/sidepanel/engineClient.ts';

function createMockTransport() {
  const messageListeners: Array<(msg: EngineToClientMessage) => void> = [];
  const posted: ClientToEngineMessage[] = [];

  const port: EnginePortLike = {
    postMessage(msg) {
      posted.push(msg);
    },
    disconnect: vi.fn(),
    onMessage: {
      addListener(cb) {
        messageListeners.push(cb);
      },
    },
    onDisconnect: {
      addListener: vi.fn(),
    },
  };

  const client = new OffscreenEngineClient({
    ensureOffscreen: async () => {},
    connect: () => port,
  });

  return {
    client,
    posted,
    emit(msg: EngineToClientMessage) {
      for (const listener of messageListeners) {
        listener(msg);
      }
    },
  };
}

function makeUserMessage(text: string): ThreadMessage {
  return {
    id: 'msg-1',
    createdAt: new Date(),
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    metadata: { custom: {} },
  };
}

describe('createTinyGptChatAdapter', () => {
  it('yields cumulative text (not raw deltas) as required by assistant-ui', async () => {
    const { client, posted, emit } = createMockTransport();
    const adapter = createTinyGptChatAdapter(client);
    const abortController = new AbortController();

    const yielded: string[] = [];
    const runPromise = (async () => {
      const iter = adapter.run({
        messages: [makeUserMessage('Say hello')],
        abortSignal: abortController.signal,
        runConfig: {},
        context: {},
        unstable_getMessage: () => makeUserMessage('Say hello'),
      });
      if (Symbol.asyncIterator in iter) {
        for await (const chunk of iter) {
          const textPart = chunk.content?.find((p) => p.type === 'text');
          if (textPart && 'text' in textPart) {
            yielded.push(textPart.text);
          }
        }
      }
    })();

    // Wait until GENERATE is posted over the mock port.
    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'GENERATE')).toBe(true);
    });
    const genMsg = posted.find((m) => m.type === 'GENERATE') as Extract<
      ClientToEngineMessage,
      { type: 'GENERATE' }
    >;
    expect(genMsg.messages).toEqual([{ role: 'user', content: 'Say hello' }]);

    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: 'Hello', tokenCount: 1 });
    await Promise.resolve();
    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: ', world', tokenCount: 2 });
    await Promise.resolve();
    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: '!', tokenCount: 3 });
    emit({ type: 'GENERATION_COMPLETE', requestId: genMsg.requestId, finishReason: 'stop' });

    await runPromise;
    expect(yielded.length).toBeGreaterThanOrEqual(1);
    expect(yielded[yielded.length - 1]).toBe('Hello, world!');
    for (let i = 1; i < yielded.length; i++) {
      expect(yielded[i].startsWith(yielded[i - 1])).toBe(true);
    }
  });

  it('forwards AbortSignal to ABORT message on the Offscreen port', async () => {
    const { client, posted, emit } = createMockTransport();
    const adapter = createTinyGptChatAdapter(client);
    const abortController = new AbortController();

    const yielded: string[] = [];
    const runPromise = (async () => {
      const iter = adapter.run({
        messages: [makeUserMessage('Stream forever')],
        abortSignal: abortController.signal,
        runConfig: {},
        context: {},
        unstable_getMessage: () => makeUserMessage('Stream forever'),
      });
      if (Symbol.asyncIterator in iter) {
        for await (const chunk of iter) {
          const textPart = chunk.content?.find((p) => p.type === 'text');
          if (textPart && 'text' in textPart) {
            yielded.push(textPart.text);
          }
        }
      }
    })();

    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'GENERATE')).toBe(true);
    });
    const genMsg = posted.find((m) => m.type === 'GENERATE') as Extract<
      ClientToEngineMessage,
      { type: 'GENERATE' }
    >;

    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: 'Partial', tokenCount: 1 });
    await vi.waitFor(() => {
      expect(yielded).toContain('Partial');
    });

    abortController.abort();
    await runPromise;

    const abortMsg = posted.find((m) => m.type === 'ABORT');
    expect(abortMsg).toEqual({ type: 'ABORT', requestId: genMsg.requestId });
  });
});
