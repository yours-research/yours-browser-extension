import type { ChatModelAdapter, ThreadMessage } from '@assistant-ui/react';
import type { ChatMessage } from '../shared/protocol.ts';
import { buildBaseSystemMessage } from '../shared/systemPrompt.ts';
import {
  detectPendingToolCall,
  formatToolResultMessage,
  parseToolCalls,
} from '../tools/toolProtocol.ts';
import {
  runAgentTurn,
  type AgentEngineBridge,
  type AgentToolBridge,
  type ToolParserAdapter,
  type ToolStatusUpdate,
} from './agentController.ts';
import type { OffscreenEngineClient } from './engineClient.ts';

export interface TinyGptChatAdapterOptions {
  enableAgentTools?: boolean;
  getActivePageContext?: () => { contextMsg: ChatMessage; groundingText: string } | null;
  onToolStatus?: (status: ToolStatusUpdate | null) => void;
  onPermissionRequired?: (req: { toolName: string; label: string }) => void;
  onBeforeGenerate?: () => void;
}

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

function stripToolCallPreview(raw: string): string {
  const withoutComplete = raw.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '');
  const openIdx = withoutComplete.toLowerCase().lastIndexOf('<tool_call>');
  const visible = openIdx >= 0 ? withoutComplete.slice(0, openIdx) : withoutComplete;
  return visible.trimStart();
}

function createSinglePassEngineBridge(
  client: OffscreenEngineClient,
  baseRequestId: string,
): AgentEngineBridge {
  let activeReqId = baseRequestId;
  return {
    async generate(requestId, chatMessages, options) {
      activeReqId = `${baseRequestId}-${requestId}`;
      return new Promise((resolve, reject) => {
        let acc = '';
        const onAbort = () => client.abortGeneration(activeReqId);
        options?.signal?.addEventListener('abort', onAbort, { once: true });

        void client.startGeneration(
          activeReqId,
          chatMessages,
          {
            onDelta(delta, tokenCount) {
              acc += delta;
              options?.onDelta?.(delta, tokenCount);
            },
            onComplete(finishReason) {
              options?.signal?.removeEventListener('abort', onAbort);
              resolve({ text: acc, finishReason });
            },
            onError(error) {
              options?.signal?.removeEventListener('abort', onAbort);
              reject(new Error(error));
            },
          },
          undefined,
          options?.groundingText,
        );
      });
    },
    abort() {
      client.abortGeneration(activeReqId);
    },
  };
}

/**
 * Creates a ChatModelAdapter for @assistant-ui/react's useLocalRuntime.
 * Yields cumulative fullText on every token delta (as required by assistant-ui)
 * and forwards AbortSignal cancellations over the MV3 Offscreen Document port.
 */
export function createTinyGptChatAdapter(
  client: OffscreenEngineClient,
  options?: TinyGptChatAdapterOptions,
): ChatModelAdapter {
  if (!options?.enableAgentTools) {
    return {
      async *run({ messages, abortSignal }) {
        options?.onBeforeGenerate?.();
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

  return {
    async *run({ messages, abortSignal }) {
      options.onBeforeGenerate?.();
      const baseRequestId = makeRequestId();
      const chatMessages = toChatMessages(messages);
      const activePage = options.getActivePageContext?.() ?? null;
      const engineBridge = createSinglePassEngineBridge(client, baseRequestId);

      const parserAdapter: ToolParserAdapter = {
        detectPendingToolCall(text: string) {
          const pending = detectPendingToolCall(text);
          if (!pending) return null;
          return { name: pending.toolName ?? 'tool' };
        },
        extractToolCall(text: string) {
          const parsed = parseToolCalls(text);
          const first = parsed.toolCalls[0];
          if (!first) {
            return { call: null, cleanText: parsed.cleanText || text };
          }
          return {
            call: { name: first.name, arguments: first.arguments },
            cleanText: parsed.cleanText,
          };
        },
      };

      const toolBridge: AgentToolBridge = {
        async executeTool(call, ctx) {
          const res = await client.execTool(call);
          const text = formatToolResultMessage(call.name, res);
          if (res.ok && res.formatted && call.name === 'fetch_and_read_url') {
            ctx?.onSourceReady?.(res.formatted);
          }
          return {
            ok: res.ok,
            text,
            needsPermission: res.needsPermission,
            permissionLabel: res.needsPermission
              ? 'Grant Web Research Permission (https://*/*)'
              : undefined,
          };
        },
      };

      const updates: string[] = [];
      let done = false;
      let runError: string | null = null;
      let notify: (() => void) | null = null;

      const wake = () => {
        const fn = notify;
        notify = null;
        fn?.();
      };

      let committedPrefix = '';

      void runAgentTurn({
        messages: chatMessages,
        baseSystemMessage: buildBaseSystemMessage(),
        activePageContext: activePage?.contextMsg ?? null,
        initialGroundingText: activePage?.groundingText ?? '',
        engine: engineBridge,
        tools: toolBridge,
        parser: parserAdapter,
        signal: abortSignal,
        onDelta(_delta, accStepText) {
          const cleanStep = stripToolCallPreview(accStepText);
          const combined = committedPrefix
            ? `${committedPrefix}${cleanStep ? `\n\n${cleanStep}` : ''}`
            : cleanStep;
          if (combined) {
            updates.push(combined);
            wake();
          }
        },
        onToolStatus(status) {
          options.onToolStatus?.(status);
        },
        onToolResult(info) {
          if (info.result.ok && committedPrefix) {
            return;
          }
        },
        onAssistantMessage(msg) {
          if (msg.content.trim()) {
            committedPrefix = committedPrefix
              ? `${committedPrefix}\n\n${msg.content}`
              : msg.content;
          }
        },
      })
        .then((res) => {
          if (res.stoppedForPermission && res.permissionRequest) {
            options.onPermissionRequired?.(res.permissionRequest);
            const fallbackMsg = res.finalText
              ? `${res.finalText}\n\nWeb Research permission is off. Enable Web Research above to run live web searches.`
              : 'Web Research permission is off. Click "Enable Web Research" above to allow HTTPS search and page fetching.';
            updates.push(fallbackMsg);
          } else {
            options.onToolStatus?.(null);
            if (res.finalText && res.finalText !== updates[updates.length - 1]) {
              const combined =
                committedPrefix && res.finalText !== committedPrefix
                  ? committedPrefix
                  : res.finalText;
              updates.push(combined);
            }
          }
          done = true;
          wake();
        })
        .catch((err) => {
          options.onToolStatus?.(null);
          runError = err instanceof Error ? err.message : String(err);
          done = true;
          wake();
        });

      let lastYielded = '';
      while (!done || updates.length > 0) {
        if (updates.length === 0) {
          await new Promise<void>((resolve) => {
            notify = resolve;
          });
        }
        if (updates.length > 0) {
          const latest = updates.splice(0, updates.length).pop()!;
          if (latest !== lastYielded) {
            lastYielded = latest;
            yield {
              content: [{ type: 'text', text: latest }],
            };
          }
        }
      }

      if (runError) {
        throw new Error(runError);
      }
    },
  };
}
