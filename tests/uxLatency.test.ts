import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChatMessage } from '../src/shared/protocol.ts';
import {
  BASE_SYSTEM_PROMPT,
  buildActivePageContextMessage,
  buildBaseSystemMessage,
  buildModeSystemMessage,
  buildToolResultFollowup,
  hashStringFnv1a,
} from '../src/shared/systemPrompt.ts';
import {
  BASE_SYSTEM_SLOT_ID,
  PrefixCacheManager,
  TURN_CHECKPOINT_SLOT_ID,
  isExactTokenPrefix,
  type GpuPrefixSlotBackend,
} from '../src/offscreen/prefixCacheManager.ts';
import {
  STANDARD_REFRESH_RATES,
  measureDisplayRefreshRate,
  snapRefreshRate,
} from '../src/sidepanel/displayHz.ts';
import {
  ActiveTabSpeculativePrefillController,
  TypingSpeculativePrefillController,
  extractStableDraftPrefix,
  runAgentTurn,
  shouldAutoPreloadModel,
  type AgentEngineBridge,
  type AgentToolBridge,
  type ToolStatusUpdate,
} from '../src/sidepanel/agentController.ts';

function createMockGpuBackend(): GpuPrefixSlotBackend & {
  savedSlots: Map<string, { seqPos: number; byteSize: number }>;
  deletedSlots: string[];
  restoredSlots: string[];
  resetCount: number;
  setActiveSeqPos(pos: number): void;
} {
  let seqPos = 0;
  const savedSlots = new Map<string, { seqPos: number; byteSize: number }>();
  const deletedSlots: string[] = [];
  const restoredSlots: string[] = [];
  let resetCount = 0;

  return {
    get activeSeqPos() {
      return seqPos;
    },
    savedSlots,
    deletedSlots,
    restoredSlots,
    get resetCount() {
      return resetCount;
    },
    setActiveSeqPos(pos: number) {
      seqPos = pos;
    },
    async saveGpuPrefixSlot(slotId: string) {
      const record = { seqPos, byteSize: Math.max(1024, seqPos * 256) };
      savedSlots.set(slotId, record);
      return record;
    },
    async restoreGpuPrefixSlot(slotId: string) {
      const record = savedSlots.get(slotId);
      if (!record) {
        throw new Error(`Missing backend slot: ${slotId}`);
      }
      seqPos = record.seqPos;
      restoredSlots.push(slotId);
      return { seqPos: record.seqPos, restoreMs: 0.42 };
    },
    deleteGpuPrefixSlot(slotId: string) {
      deletedSlots.push(slotId);
      return savedSlots.delete(slotId);
    },
    async resetState() {
      seqPos = 0;
      resetCount += 1;
    },
  };
}

describe('systemPrompt', () => {
  it('defines a compact deterministic BASE_SYSTEM_PROMPT with all 6 Tier-1 tools', () => {
    expect(BASE_SYSTEM_PROMPT.length).toBeGreaterThan(400);
    expect(BASE_SYSTEM_PROMPT.length).toBeLessThan(1200);
    // Should not embed dynamic date/time tokens
    expect(BASE_SYSTEM_PROMPT).not.toMatch(/\b20\d{2}-\d{2}-\d{2}\b/);
    expect(BASE_SYSTEM_PROMPT).toContain('<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>');

    for (const toolName of [
      'read_active_page',
      'web_search',
      'fetch_and_read_url',
      'list_open_tabs',
      'read_tab',
      'js_calc',
    ]) {
      expect(BASE_SYSTEM_PROMPT).toContain(toolName);
    }

    expect(buildBaseSystemMessage()).toEqual({
      role: 'system',
      content: BASE_SYSTEM_PROMPT,
    });
  });

  it('formats active page context, tool followups, and FNV-1a hashes deterministically', () => {
    const pageMsg = buildActivePageContextMessage({
      title: 'WebGPU Spec',
      url: 'https://www.w3.org/TR/webgpu/',
      content: 'Main article text',
      mode: 'article',
    });
    expect(pageMsg).toEqual({
      role: 'system',
      content: '[Active Tab: WebGPU Spec (https://www.w3.org/TR/webgpu/) | mode=article]\nMain article text',
    });

    const followup = buildToolResultFollowup('web_search', '1. Result A');
    expect(followup).toContain('<tool_result name="web_search">');
    expect(followup).toContain('1. Result A');

    const h1 = hashStringFnv1a('https://example.com\nhello');
    const h2 = hashStringFnv1a('https://example.com\nhello');
    const h3 = hashStringFnv1a('https://example.com\nworld');
    expect(h1).toMatch(/^[0-9a-f]{8}$/);
    expect(h1).toBe(h2);
    expect(h1).not.toBe(h3);
  });

  it('builds strict deterministic mode-specific system prompts across all 4 Page x Research modes', () => {
    const mOffOff = buildModeSystemMessage({ pageEnabled: false, pageAttached: false, researchEnabled: false });
    expect(mOffOff.content).not.toContain('- web_search');
    expect(mOffOff.content).not.toContain('- fetch_and_read_url');
    expect(mOffOff.content).not.toContain('- read_active_page');
    expect(mOffOff.content).toContain('- js_calc');

    const mOnAttachedOff = buildModeSystemMessage({ pageEnabled: true, pageAttached: true, researchEnabled: false });
    expect(mOnAttachedOff.content).not.toContain('- read_active_page');
    expect(mOnAttachedOff.content).not.toContain('- web_search');
    expect(mOnAttachedOff.content).toContain('already attached below');

    const mOffOn = buildModeSystemMessage({ pageEnabled: false, pageAttached: false, researchEnabled: true });
    expect(mOffOn.content).toContain('- web_search');
    expect(mOffOn.content).toContain('- fetch_and_read_url');
    expect(mOffOn.content).not.toContain('- read_active_page');

    const mOnOn = buildModeSystemMessage({ pageEnabled: true, pageAttached: false, researchEnabled: true });
    expect(mOnOn.content).toContain('- read_active_page');
    expect(mOnOn.content).toContain('- web_search');
  });
});

describe('PrefixCacheManager', () => {
  it('enforces exact token prefix matching including DeltaNet divergence checks', () => {
    expect(isExactTokenPrefix([], [1, 2, 3])).toBe(false);
    expect(isExactTokenPrefix([1, 2, 3, 4], [1, 2, 3])).toBe(false);
    expect(isExactTokenPrefix([1, 2], [1, 2, 3])).toBe(true);
    expect(isExactTokenPrefix([1, 2, 3], [1, 2, 3])).toBe(true);
    // Middle-token divergence invalidates DeltaNet prefix
    expect(isExactTokenPrefix([1, 9, 3], [1, 2, 3, 4])).toBe(false);
  });

  it('pins base-system and turn-checkpoint while evicting LRU tab/draft slots past maxTabSlots', async () => {
    let clock = 1000;
    const backend = createMockGpuBackend();
    const manager = new PrefixCacheManager(backend, {
      maxTabSlots: 3,
      now: () => (clock += 10),
    });

    backend.setActiveSeqPos(200);
    await manager.saveBasePromptSlot([1, 2, 3, 4], [buildBaseSystemMessage()]);
    expect(manager.hasBasePrompt()).toBe(true);
    expect(manager.getBasePromptInfo()).toEqual({ cached: true, tokens: 4 });

    backend.setActiveSeqPos(450);
    await manager.saveTurnCheckpoint([1, 2, 3, 4, 5, 6], [buildBaseSystemMessage()]);

    backend.setActiveSeqPos(300);
    await manager.saveTabSlot('tab:a', [1, 2, 3, 4, 10], [], { url: 'https://a.test', contentHash: 'ha' });
    await manager.saveTabSlot('tab:b', [1, 2, 3, 4, 20], [], { url: 'https://b.test', contentHash: 'hb' });
    await manager.saveTabSlot('draft:1', [1, 2, 3, 4, 30], []);

    // Touch tab:a so tab:b becomes the oldest LRU volatile slot
    await manager.restoreSlot('tab:a');

    // Adding a 4th volatile slot should evict tab:b (NOT base-system or turn-checkpoint)
    await manager.saveTabSlot('tab:c', [1, 2, 3, 4, 40], [], { url: 'https://c.test', contentHash: 'hc' });

    expect(backend.deletedSlots).toEqual(['tab:b']);
    const stats = manager.getStats();
    expect(stats.hasBasePrompt).toBe(true);
    expect(stats.hasTurnCheckpoint).toBe(true);
    expect(stats.tabSlotCount).toBe(3);
    expect(stats.slots.map((s) => s.slotId)).not.toContain('tab:b');
    expect(stats.slots.map((s) => s.slotId)).toContain(BASE_SYSTEM_SLOT_ID);
    expect(stats.slots.map((s) => s.slotId)).toContain(TURN_CHECKPOINT_SLOT_ID);
  });

  it('selects longest exact prefix and breaks ties in favor of live', async () => {
    const backend = createMockGpuBackend();
    const manager = new PrefixCacheManager(backend);

    backend.setActiveSeqPos(4);
    await manager.saveBasePromptSlot([1, 2, 3, 4], []);
    backend.setActiveSeqPos(7);
    await manager.saveTabSlot('tab:docs', [1, 2, 3, 4, 5, 6, 7], []);

    // Live matches 7 tokens (tie with tab:docs) -> live wins
    expect(manager.findBestMatch([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7])).toEqual({
      source: 'live',
      slotId: null,
      matchedTokens: 7,
    });

    // Live diverged at index 5 -> tab:docs wins with 7 tokens over base-system's 4
    const match = manager.findBestMatch([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 99, 7]);
    expect(match.source).toBe('slot');
    expect(match.slotId).toBe('tab:docs');
    expect(match.matchedTokens).toBe(7);

    // No match at all when token 0 differs
    expect(manager.findBestMatch([99, 2, 3], [1, 2, 3])).toEqual({
      source: 'none',
      slotId: null,
      matchedTokens: 0,
    });
  });

  it('restores base-system in resetToBasePrompt and clears volatile slots cleanly', async () => {
    const backend = createMockGpuBackend();
    const manager = new PrefixCacheManager(backend);

    // Without base prompt, resetToBasePrompt calls backend.resetState and returns null
    expect(await manager.resetToBasePrompt()).toBeNull();
    expect(backend.resetCount).toBe(1);

    backend.setActiveSeqPos(210);
    await manager.saveBasePromptSlot([10, 20, 30], []);
    backend.setActiveSeqPos(500);
    await manager.saveTabSlot('tab:1', [10, 20, 30, 40], []);
    await manager.saveTurnCheckpoint([10, 20, 30, 40, 50], []);

    const restored = await manager.resetToBasePrompt();
    expect(restored).not.toBeNull();
    expect(restored?.seqPos).toBe(210);
    expect(backend.restoredSlots).toContain(BASE_SYSTEM_SLOT_ID);

    manager.clearVolatileSlots();
    expect(manager.hasBasePrompt()).toBe(true);
    expect(manager.getStats().tabSlotCount).toBe(0);
    expect(manager.getStats().hasTurnCheckpoint).toBe(false);

    manager.clearAll();
    expect(manager.hasBasePrompt()).toBe(false);
    expect(manager.getStats().totalSlots).toBe(0);
  });
});

describe('displayHz', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('snaps standard refresh rates cleanly and clamps invalid/backgrounded readings to 60', () => {
    expect(STANDARD_REFRESH_RATES).toEqual([60, 75, 90, 120, 144, 165, 240]);
    for (const hz of [58, 60, 64]) {
      expect(snapRefreshRate(hz)).toBe(60);
    }
    for (const hz of [115, 120, 124]) {
      expect(snapRefreshRate(hz)).toBe(120);
    }
    for (const hz of [138, 144, 150]) {
      expect(snapRefreshRate(hz)).toBe(144);
    }
    expect(snapRefreshRate(90.4)).toBe(90);
    expect(snapRefreshRate(164.8)).toBe(165);
    expect(snapRefreshRate(239.2)).toBe(240);

    expect(snapRefreshRate(15)).toBe(60);
    expect(snapRefreshRate(0)).toBe(60);
    expect(snapRefreshRate(Number.NaN)).toBe(60);
    expect(snapRefreshRate(Number.POSITIVE_INFINITY)).toBe(60);
  });

  it('measures display refresh rate from rAF median delta while filtering >100ms pauses', async () => {
    // Missing rAF falls back to 60
    vi.stubGlobal('requestAnimationFrame', undefined);
    expect(await measureDisplayRefreshRate(10)).toBe(60);

    // 120 Hz (~8.33 ms per frame) with two >100 ms background pause outliers
    const timestamps = [
      0,
      8.3,
      16.6,
      250.0, // outlier pause
      258.4,
      266.7,
      275.0,
      450.0, // outlier pause
      458.3,
      466.6,
      475.0,
    ];
    let idx = 0;
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      const ts = timestamps[idx] ?? timestamps[timestamps.length - 1] + 8.3;
      idx += 1;
      queueMicrotask(() => cb(ts));
      return idx;
    });

    const measured = await measureDisplayRefreshRate(10);
    expect(measured).toBe(120);
  });
});

describe('UX speculative prefill controllers & helpers', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('checks shouldAutoPreloadModel and extractStableDraftPrefix', () => {
    expect(shouldAutoPreloadModel('unloaded')).toBe(true);
    expect(shouldAutoPreloadModel('downloading')).toBe(false);
    expect(shouldAutoPreloadModel('ready')).toBe(false);
    expect(shouldAutoPreloadModel('generating')).toBe(false);

    expect(extractStableDraftPrefix('short')).toBe('');
    expect(extractStableDraftPrefix('supercalifragilistic')).toBe('');
    expect(extractStableDraftPrefix('Explain WebGPU shaders in de')).toBe(
      'Explain WebGPU shaders in',
    );
    expect(extractStableDraftPrefix('Explain WebGPU shaders in ')).toBe(
      'Explain WebGPU shaders in',
    );
  });

  it('debounces typing vs paste bursts and aborts on backspace divergence or submit', async () => {
    const onPrefill = vi.fn(async (_slotKey: string, _messages: ChatMessage[]) => {});
    const onAbort = vi.fn();

    const ctrl = new TypingSpeculativePrefillController({
      buildMessages: (stable) => [{ role: 'user', content: stable }],
      isEngineIdle: () => true,
      onPrefill,
      onAbort,
    });

    // Normal typing -> 350 ms debounce
    ctrl.onTextChanged('Explain WebGPU ');
    await vi.advanceTimersByTimeAsync(200);
    expect(onPrefill).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(160);
    expect(onPrefill).toHaveBeenCalledTimes(1);
    expect(onPrefill).toHaveBeenLastCalledWith('draft:composer', [
      { role: 'user', content: 'Explain WebGPU' },
    ]);

    // Backspace earlier than prefilled prefix -> immediate abort
    ctrl.onTextChanged('Explain Web');
    expect(onAbort).toHaveBeenCalledTimes(1);

    // Paste burst (>= 120 chars delta) -> 50 ms debounce
    const longPaste =
      'Please summarize the following technical specification about WebGPU compute shaders, subgroup matrix operations, and prefix caching in detail for me ';
    ctrl.onTextChanged(longPaste);
    await vi.advanceTimersByTimeAsync(55);
    expect(onPrefill).toHaveBeenCalledTimes(2);

    // cancelForSubmit aborts and cancels pending timers
    ctrl.cancelForSubmit();
    expect(onAbort).toHaveBeenCalledTimes(2);
    ctrl.dispose();
  });

  it('deduplicates active tab extraction and aborts immediately on tab change or submit', async () => {
    let tabData = {
      title: 'Docs',
      url: 'https://docs.example.com/page1',
      content: 'This is a sufficiently long article body from the active browser tab for prefill.',
      mode: 'article',
    };
    const onPrefill = vi.fn(
      async (
        _slotKey: string,
        _messages: ChatMessage[],
        _meta: { url: string; contentHash: string },
      ) => {},
    );
    const onAbort = vi.fn();

    const tabCtrl = new ActiveTabSpeculativePrefillController({
      extractActiveTab: async () => tabData,
      buildMessages: (t) => [{ role: 'system', content: `${t.title}:${t.contentHash}` }],
      isEngineIdle: () => true,
      onPrefill,
      onAbort,
    });

    tabCtrl.scheduleExtraction();
    await vi.advanceTimersByTimeAsync(260);
    expect(onPrefill).toHaveBeenCalledTimes(1);
    const [slotKey, , meta] = onPrefill.mock.calls[0]!;
    expect(slotKey.startsWith('tab:')).toBe(true);
    expect(meta.url).toBe('https://docs.example.com/page1');

    // Re-scheduling on unchanged tab skips prefill
    tabCtrl.scheduleExtraction();
    await vi.advanceTimersByTimeAsync(260);
    expect(onPrefill).toHaveBeenCalledTimes(1);

    // Switching tab aborts immediately and extracts new tab after 250ms
    tabData = {
      title: 'Page 2',
      url: 'https://docs.example.com/page2',
      content: 'Completely different page content that should trigger a new tab cache prefill.',
      mode: 'article',
    };
    tabCtrl.notifyTabChanged();
    expect(onAbort).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(260);
    expect(onPrefill).toHaveBeenCalledTimes(2);

    tabCtrl.cancelForSubmit();
    expect(onAbort).toHaveBeenCalledTimes(2);
    tabCtrl.dispose();
  });
});

describe('runAgentTurn', () => {
  it('runs multi-step tool loop, streams pending chips, and pipelines onSourceReady grounding', async () => {
    const generateCalls: Array<{
      requestId: string;
      messages: ChatMessage[];
      groundingText?: string;
    }> = [];
    const statuses: ToolStatusUpdate[] = [];
    const incrementalChunks: string[] = [];

    const engine: AgentEngineBridge = {
      async generate(requestId, messages, options) {
        generateCalls.push({
          requestId,
          messages: messages.map((m) => ({ ...m })),
          groundingText: options?.groundingText,
        });

        if (generateCalls.length === 1) {
          options?.onDelta?.('<tool_call>{"name":"web_search"', 1);
          options?.onDelta?.(',"arguments":{"query":"webgpu"}}</tool_call>', 2);
          return {
            text: '<tool_call>{"name":"web_search","arguments":{"query":"webgpu"}}</tool_call>',
            finishReason: 'stop',
          };
        }

        if (generateCalls.length === 2) {
          const call2 = '<tool_call>{"name":"fetch_and_read_url","arguments":{"urls":["https://example.com"]}}</tool_call>';
          options?.onDelta?.(call2, 1);
          return { text: call2, finishReason: 'stop' };
        }

        options?.onDelta?.('Final grounded answer.', 1);
        return { text: 'Final grounded answer.', finishReason: 'stop' };
      },
    };

    const tools: AgentToolBridge = {
      async executeTool(call, context) {
        if (call.name === 'web_search') {
          return {
            ok: true,
            text: 'Found https://example.com',
          };
        }
        context?.onSourceReady?.('Source 1: WebGPU enables fast local GPU compute.');
        context?.onSourceReady?.('Source 2: DeltaNet supports O(1) recurrent state.');
        return {
          ok: true,
          text: 'Fetched 2 sources.',
        };
      },
    };

    const result = await runAgentTurn({
      messages: [{ role: 'user', content: 'Explain WebGPU and DeltaNet.' }],
      baseSystemMessage: buildBaseSystemMessage(),
      engine,
      tools,
      onToolStatus: (s) => statuses.push(s),
      onIncrementalContextReady: (txt) => incrementalChunks.push(txt),
    });

    expect(result.finishReason).toBe('stop');
    expect(result.finalText).toBe('Final grounded answer.');
    expect(result.stepsExecuted).toBe(3);
    expect(result.toolCalls).toHaveLength(2);
    expect(incrementalChunks).toEqual([
      'Source 1: WebGPU enables fast local GPU compute.',
      'Source 2: DeltaNet supports O(1) recurrent state.',
    ]);

    // Step 3 generation received accumulated groundingText from pipelined onSourceReady
    expect(generateCalls[2]?.groundingText).toContain(
      'Source 1: WebGPU enables fast local GPU compute.',
    );
    expect(generateCalls[2]?.groundingText).toContain(
      'Source 2: DeltaNet supports O(1) recurrent state.',
    );

    // Pending chip was emitted immediately during streaming
    expect(statuses.some((s) => s.phase === 'pending_call' && s.label === 'Searching...')).toBe(
      true,
    );
    expect(statuses.some((s) => s.phase === 'completed' && s.toolName === 'fetch_and_read_url')).toBe(
      true,
    );
  });

  it('halts cleanly on needsPermission so UI can show 1-click grant prompt', async () => {
    const statuses: ToolStatusUpdate[] = [];
    const engine: AgentEngineBridge = {
      async generate() {
        return {
          text: '<tool_call>{"name":"web_search","arguments":{"query":"latest news"}}</tool_call>',
          finishReason: 'stop',
        };
      },
    };

    const tools: AgentToolBridge = {
      async executeTool() {
        return {
          ok: false,
          text: 'Host permission required.',
          needsPermission: true,
          permissionLabel: 'Grant Web Research Permission',
        };
      },
    };

    const result = await runAgentTurn({
      messages: [{ role: 'user', content: 'What is the latest news?' }],
      engine,
      tools,
      onToolStatus: (s) => statuses.push(s),
    });

    expect(result.stoppedForPermission).toBe(true);
    expect(result.finishReason).toBe('permission_required');
    expect(result.permissionRequest).toEqual({
      toolName: 'web_search',
      label: 'Grant Web Research Permission',
    });
    expect(statuses[statuses.length - 1]).toMatchObject({
      phase: 'permission_required',
      toolName: 'web_search',
      label: 'Grant Web Research Permission',
    });
  });
});
