import { ChatSession } from '@yours/harness/controller';
import type { ToolDefinition, ToolExecutor } from '@yours/harness';
import { InMemoryStore } from '@yours/harness/storage';
import { describe, expect, it, vi } from 'vitest';
import { isMessagePrefix } from '../src/offscreen/sampling.ts';
import type { ClientToEngineMessage, EngineToClientMessage } from '../src/shared/protocol.ts';
import { OffscreenEngineClient, type EnginePortLike } from '../src/sidepanel/engineClient.ts';
import { createWgslRuntimeAdapter } from '../src/sidepanel/runtimeFactory.ts';

function createMockTransport(autoReplyPrefill = false) {
  const messageListeners: Array<(msg: EngineToClientMessage) => void> = [];
  const posted: ClientToEngineMessage[] = [];

  const port: EnginePortLike = {
    postMessage(msg) {
      posted.push(msg);
      if (autoReplyPrefill && msg.type === 'PREFILL_BACKGROUND' && msg.requestId) {
        const reqId = msg.requestId;
        const slotKey = msg.slotKey;
        queueMicrotask(() => {
          emit({
            type: 'PREFILL_COMPLETE',
            requestId: reqId,
            slotKey,
            tokens: msg.messages.length * 4,
            elapsedMs: 1.5,
          });
        });
      } else if (autoReplyPrefill && msg.type === 'EVICT_SLOT') {
        const reqId = msg.requestId;
        const slotKey = msg.slotKey;
        queueMicrotask(() => {
          emit({
            type: 'EVICT_COMPLETE',
            requestId: reqId,
            slotKey,
          });
        });
      }
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

  function emit(msg: EngineToClientMessage) {
    for (const listener of [...messageListeners]) {
      listener(msg);
    }
  }

  const client = new OffscreenEngineClient({
    ensureOffscreen: async () => {},
    connect: () => port,
  });

  return {
    client,
    posted,
    emit,
  };
}

describe('OffscreenEngineClient + WgslWorkerAdapter + ChatSession', () => {
  it('streams incremental tokens from WorkerPortTransport through WgslWorkerAdapter', async () => {
    const { client, posted, emit } = createMockTransport(true);
    await client.connect();

    const adapter = createWgslRuntimeAdapter(client, {
      modelId: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
      maxContextTokens: 32768,
    });
    const session = new ChatSession({
      adapter,
      store: new InMemoryStore(),
      systemPrompt: 'You are a helpful local assistant.',
    });
    await session.init();

    const sendPromise = session.sendMessage('Say hello');

    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'GENERATE')).toBe(true);
    });
    const genMsg = posted.find((m) => m.type === 'GENERATE') as Extract<
      ClientToEngineMessage,
      { type: 'GENERATE' }
    >;
    expect(genMsg.messages.some((m) => m.role === 'user' && m.content === 'Say hello')).toBe(true);

    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: 'Hello', tokenCount: 1 });
    await Promise.resolve();
    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: ', world!', tokenCount: 2 });
    emit({ type: 'GENERATION_COMPLETE', requestId: genMsg.requestId, finishReason: 'stop' });

    await sendPromise;

    const assistantMessages = session.getSnapshot().messages.filter((m) => m.role === 'assistant');
    expect(assistantMessages.length).toBe(1);
    expect(assistantMessages[0].content).toBe('Hello, world!');
  });

  it('handles PREFILL_BACKGROUND and EVICT_SLOT over WorkerPortTransport', async () => {
    const { client, posted, emit } = createMockTransport();
    await client.connect();
    const adapter = createWgslRuntimeAdapter(client);

    const prefillPromise = adapter.prefillSlot('tab:hash1', [
      { id: '1', role: 'system', content: 'Base', createdAt: 1 },
    ]);

    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'PREFILL_BACKGROUND')).toBe(true);
    });
    const prefillMsg = posted.find((m) => m.type === 'PREFILL_BACKGROUND') as Extract<
      ClientToEngineMessage,
      { type: 'PREFILL_BACKGROUND' }
    >;
    expect(prefillMsg.slotKey).toBe('tab:hash1');
    expect(typeof prefillMsg.requestId).toBe('string');

    emit({
      type: 'PREFILL_COMPLETE',
      requestId: prefillMsg.requestId!,
      slotKey: 'tab:hash1',
      tokens: 42,
      elapsedMs: 12.5,
    });
    await expect(prefillPromise).resolves.toEqual({ tokens: 42, elapsedMs: 12.5 });

    const evictPromise = adapter.evictSlot('tab:hash1');
    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'EVICT_SLOT')).toBe(true);
    });
    const evictMsg = posted.find((m) => m.type === 'EVICT_SLOT') as Extract<
      ClientToEngineMessage,
      { type: 'EVICT_SLOT' }
    >;
    emit({
      type: 'EVICT_COMPLETE',
      requestId: evictMsg.requestId,
      slotKey: 'tab:hash1',
    });
    await expect(evictPromise).resolves.toBeUndefined();
  });

  it('sends ABORT when aborted via ChatSession', async () => {
    const { client, posted, emit } = createMockTransport(true);
    await client.connect();

    const adapter = createWgslRuntimeAdapter(client);
    const session = new ChatSession({
      adapter,
      store: new InMemoryStore(),
    });
    await session.init();

    const sendPromise = session.sendMessage('Stream forever');
    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'GENERATE')).toBe(true);
    });
    const genMsg = posted.find((m) => m.type === 'GENERATE') as Extract<
      ClientToEngineMessage,
      { type: 'GENERATE' }
    >;

    emit({ type: 'TOKEN_DELTA', requestId: genMsg.requestId, delta: 'Start...', tokenCount: 1 });
    await Promise.resolve();

    session.abort();
    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'ABORT' && m.requestId === genMsg.requestId)).toBe(true);
    });
    emit({
      type: 'GENERATION_COMPLETE',
      requestId: genMsg.requestId,
      finishReason: 'abort',
    });

    await sendPromise;
    expect(session.getSnapshot().isGenerating).toBe(false);
    expect(posted.some((m) => m.type === 'ABORT' && m.requestId === genMsg.requestId)).toBe(true);
  });

  it('skips blocking Slot-0 prefill, forwards stopOnToolCall/groundingText/rawContent, and trims isMessagePrefix', async () => {
    expect(
      isMessagePrefix(
        [
          { role: 'system', content: 'sys' },
          { role: 'assistant', content: 'Done.\n  ' },
        ],
        [
          { role: 'system', content: 'sys' },
          { role: 'assistant', content: 'Done.' },
          { role: 'user', content: 'Next?' },
        ],
      ),
    ).toBe(true);

    const { client, posted, emit } = createMockTransport(false);
    await client.connect();

    const calcTool: ToolDefinition = {
      name: 'run_sandbox',
      description: 'Run math',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string' } },
        required: ['expression'],
      },
      safety: {
        readOnlyHint: true,
        consequentialHint: false,
        networkHint: false,
        untrustedContentHint: false,
      },
    };
    const toolExecutor: ToolExecutor = {
      listTools: () => [calcTool],
      async execute(name, args) {
        return {
          ok: true,
          toolName: name,
          formatted: `Result: ${String(args.expression ?? '')} = 42`,
          tainted: false,
        };
      },
    };

    const adapter = createWgslRuntimeAdapter(client, {
      modelId: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
      maxContextTokens: 32768,
    });
    const session = new ChatSession({
      adapter,
      store: new InMemoryStore(),
      systemPrompt: 'Compact assistant.',
      tools: [calcTool],
      toolExecutor,
    });
    await session.init();

    const sendPromise = session.sendMessage('Compute 6 * 7', {
      groundingText: 'Initial page grounding',
    });

    await vi.waitFor(() => {
      expect(posted.some((m) => m.type === 'GENERATE')).toBe(true);
    });

    // Turn 1 must start GENERATE immediately without blocking on PREFILL_BACKGROUND
    expect(posted.some((m) => m.type === 'PREFILL_BACKGROUND')).toBe(false);

    const firstGen = posted.find((m) => m.type === 'GENERATE') as Extract<
      ClientToEngineMessage,
      { type: 'GENERATE' }
    > & { groundingText?: string; stopOnToolCall?: boolean };
    expect(firstGen.stopOnToolCall).toBe(true);
    expect(firstGen.groundingText).toBe('Initial page grounding');

    const rawToolXml =
      '<tool_call><function=run_sandbox><parameter=expression>6 * 7</parameter></function></tool_call>';
    emit({
      type: 'TOKEN_DELTA',
      requestId: firstGen.requestId,
      delta: rawToolXml,
      tokenCount: 4,
    });
    emit({ type: 'GENERATION_COMPLETE', requestId: firstGen.requestId, finishReason: 'stop' });

    await vi.waitFor(() => {
      expect(posted.filter((m) => m.type === 'GENERATE')).toHaveLength(2);
    });

    const secondGen = posted.filter((m) => m.type === 'GENERATE')[1] as Extract<
      ClientToEngineMessage,
      { type: 'GENERATE' }
    > & { groundingText?: string };
    expect(secondGen.groundingText).toContain('Initial page grounding');
    expect(secondGen.groundingText).toContain('42');
    expect(secondGen.messages.some((m) => m.role === 'assistant' && m.content === rawToolXml)).toBe(
      true,
    );

    emit({
      type: 'TOKEN_DELTA',
      requestId: secondGen.requestId,
      delta: 'The answer is 42.',
      tokenCount: 5,
    });
    emit({ type: 'GENERATION_COMPLETE', requestId: secondGen.requestId, finishReason: 'stop' });

    await sendPromise;

    const saved = session.getSnapshot().messages;
    const intermediateToolStep = saved.find((m) => m.role === 'assistant' && m.toolName);
    expect(intermediateToolStep?.content).toBe('');
    expect(intermediateToolStep?.rawContent).toBe(rawToolXml);
    expect(saved[saved.length - 1]?.content).toBe('The answer is 42.');
  });
});
