import type { ChatModelAdapter, ThreadMessage } from '@assistant-ui/react';
import type { ChatMessage } from '../shared/protocol.ts';
import type { OffscreenEngineClient } from './engineClient.ts';

function extractTextContent(message: ThreadMessage): string {
  if (!Array.isArray(message.content)) {
    return '';
  }
  const textParts: string[] = [];
  for (const part of message.content) {
    if (part.type === 'text' && typeof part.text === 'string') {
      textParts.push(part.text);
    }
  }
  return textParts.join('\n');
}

export function toChatMessages(messages: readonly ThreadMessage[]): ChatMessage[] {
  const result: ChatMessage[] = [];
  for (const msg of messages) {
    if (msg.role !== 'system' && msg.role !== 'user' && msg.role !== 'assistant') {
      continue;
    }
    const text = extractTextContent(msg);
    result.push({ role: msg.role, content: text });
  }
  return result;
}

function makeRequestId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  return `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/**
 * Creates a ChatModelAdapter for @assistant-ui/react's useLocalRuntime.
 * Yields cumulative fullText on every token delta (as required by assistant-ui)
 * and forwards AbortSignal cancellations over the MV3 Offscreen Document port.
 */
export function createTinyGptChatAdapter(client: OffscreenEngineClient): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const requestId = makeRequestId();
      const chatMessages = toChatMessages(messages);

      let fullText = '';
      let done = false;
      let streamError: string | null = null;
      const pendingDeltas: string[] = [];
      let notify: (() => void) | null = null;

      const wake = () => {
        const fn = notify;
        notify = null;
        fn?.();
      };

      const onAbort = () => {
        client.abortGeneration(requestId);
        done = true;
        wake();
      };

      if (abortSignal.aborted) {
        onAbort();
        return;
      }
      abortSignal.addEventListener('abort', onAbort, { once: true });

      void client.startGeneration(requestId, chatMessages, {
        onDelta(delta) {
          pendingDeltas.push(delta);
          wake();
        },
        onComplete() {
          done = true;
          wake();
        },
        onError(error) {
          streamError = error;
          done = true;
          wake();
        },
      });

      try {
        while (!done || pendingDeltas.length > 0) {
          if (pendingDeltas.length === 0) {
            await new Promise<void>((resolve) => {
              notify = resolve;
            });
          }

          if (pendingDeltas.length > 0) {
            while (pendingDeltas.length > 0) {
              fullText += pendingDeltas.shift()!;
            }
            yield {
              content: [{ type: 'text', text: fullText }],
            };
          }
        }

        if (streamError) {
          throw new Error(streamError);
        }
      } finally {
        abortSignal.removeEventListener('abort', onAbort);
      }
    },
  };
}
