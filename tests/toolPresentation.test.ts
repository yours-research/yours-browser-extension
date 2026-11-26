import { describe, expect, it } from 'vitest';
import type { MessageNode, ToolTraceRecord } from '@yours/harness';
import {
  buildGroupStepsFromToolNodes,
  describeToolStep,
  formatDurationMs,
  formatToolActivityLabel,
  formatToolChainSummary,
  mergeActiveTurnToolSteps,
  type ToolExecutionTelemetry,
} from '../src/sidepanel/toolPresentation.ts';

function makeNode(
  id: string,
  role: 'user' | 'assistant' | 'tool',
  content: string,
  extra: Partial<MessageNode['message']> = {},
  createdAt = 1000,
): MessageNode {
  return {
    id,
    threadId: 'thread-1',
    parentId: null,
    childrenIds: [],
    createdAt,
    message: {
      id,
      role,
      content,
      createdAt,
      ...extra,
    },
  };
}

describe('toolPresentation', () => {
  it('maps all canonical, legacy alias, and custom tools to clean Title Case verbs and details without raw snake_case', () => {
    const cases: Array<{
      name: string;
      args?: Record<string, unknown>;
      output?: string;
      status: 'running' | 'done' | 'error' | 'blocked';
      expectedRunningOrDoneVerb: string;
      expectedStatusKind: string;
      expectedDetailSubstring?: string;
    }> = [
      {
        name: 'read_active_page',
        status: 'running',
        expectedRunningOrDoneVerb: 'Reading active page',
        expectedStatusKind: 'reading',
        expectedDetailSubstring: 'TinyGPT Browser',
      },
      {
        name: 'read_active_tab',
        status: 'done',
        expectedRunningOrDoneVerb: 'Read active page',
        expectedStatusKind: 'reading',
        expectedDetailSubstring: 'TinyGPT Browser',
      },
      {
        name: 'list_open_tabs',
        output: '1. [11] Alpha\n2. [12] Beta\n3. [13] Gamma',
        status: 'done',
        expectedRunningOrDoneVerb: 'Checked open tabs',
        expectedStatusKind: 'tabs',
        expectedDetailSubstring: '3 tabs',
      },
      {
        name: 'list_tabs',
        args: { window_scope: 'all' },
        status: 'running',
        expectedRunningOrDoneVerb: 'Checking open tabs',
        expectedStatusKind: 'tabs',
        expectedDetailSubstring: 'All windows',
      },
      {
        name: 'read_tab',
        args: { tab_id: 11, title: 'Alpha Cluster Spec' },
        status: 'done',
        expectedRunningOrDoneVerb: 'Read tab',
        expectedStatusKind: 'reading',
        expectedDetailSubstring: 'Alpha Cluster Spec',
      },
      {
        name: 'web_search',
        args: { query: 'WebGPU subgroups WGSL' },
        status: 'running',
        expectedRunningOrDoneVerb: 'Searching the web',
        expectedStatusKind: 'searching',
        expectedDetailSubstring: 'WebGPU subgroups WGSL',
      },
      {
        name: 'search_web_tab',
        args: { query: 'MV3 CSP compliance' },
        status: 'done',
        expectedRunningOrDoneVerb: 'Searched the web',
        expectedStatusKind: 'searching',
        expectedDetailSubstring: 'MV3 CSP compliance',
      },
      {
        name: 'fetch_and_read_url',
        args: { url: 'https://www.github.com/yours-research/yours-ui' },
        status: 'running',
        expectedRunningOrDoneVerb: 'Reading web source',
        expectedStatusKind: 'reading',
        expectedDetailSubstring: 'github.com',
      },
      {
        name: 'fetch_background_tab',
        args: { url: 'https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice' },
        status: 'done',
        expectedRunningOrDoneVerb: 'Read web source',
        expectedStatusKind: 'reading',
        expectedDetailSubstring: 'developer.mozilla.org',
      },
      {
        name: 'js_calc',
        args: { expression: '512 * 515' },
        status: 'done',
        expectedRunningOrDoneVerb: 'Calculated',
        expectedStatusKind: 'calculating',
        expectedDetailSubstring: '512 * 515',
      },
      {
        name: 'run_sandbox',
        args: { expression: 'Math.sqrt(144)' },
        status: 'running',
        expectedRunningOrDoneVerb: 'Calculating',
        expectedStatusKind: 'calculating',
        expectedDetailSubstring: 'Math.sqrt(144)',
      },
      {
        name: 'memory_recall',
        args: { query: 'preferred model' },
        status: 'done',
        expectedRunningOrDoneVerb: 'Recalled memory',
        expectedStatusKind: 'memory',
        expectedDetailSubstring: 'preferred model',
      },
      {
        name: 'memory_save',
        args: { key: 'tone', content: 'concise answers' },
        status: 'running',
        expectedRunningOrDoneVerb: 'Saving memory',
        expectedStatusKind: 'memory',
        expectedDetailSubstring: 'tone',
      },
      {
        name: 'interact_page',
        args: { action: 'click', selector: '#submit-btn' },
        status: 'done',
        expectedRunningOrDoneVerb: 'Interacted with page',
        expectedStatusKind: 'analyzing',
        expectedDetailSubstring: 'click',
      },
      {
        name: 'custom_graph_lookup',
        status: 'running',
        expectedRunningOrDoneVerb: 'Running Custom Graph Lookup',
        expectedStatusKind: 'analyzing',
      },
    ];

    for (const tc of cases) {
      const step = describeToolStep({
        id: `step-${tc.name}`,
        toolName: tc.name,
        status: tc.status,
        args: tc.args,
        output: tc.output,
        durationMs: 42,
        activePage: { title: 'TinyGPT Browser', url: 'https://example.com/home' },
      });

      expect(step.tool).toBe(tc.expectedRunningOrDoneVerb);
      expect(step.actionLabel).toBe(tc.expectedRunningOrDoneVerb);
      expect(step.tool).not.toMatch(/Executing local MV3 tool/i);
      expect(step.tool).not.toMatch(/_/);
      expect(step.state).toBe(tc.expectedStatusKind);
      expect(step.statusKind).toBe(tc.expectedStatusKind);
      expect(step.latencyLabel).toBe('42ms');
      if (tc.expectedDetailSubstring) {
        expect(step.label).toContain(tc.expectedDetailSubstring);
      }

      const activity = formatToolActivityLabel(tc.name, tc.args, {
        title: 'TinyGPT Browser',
        url: 'https://example.com/home',
      });
      expect(activity).not.toMatch(/_/);
      expect(activity).not.toMatch(/Executing local MV3 tool/i);
    }
  });

  it('formats sub-second and multi-second durations and chain summaries cleanly', () => {
    expect(formatDurationMs(0)).toBe('1ms');
    expect(formatDurationMs(318)).toBe('318ms');
    expect(formatDurationMs(2410)).toBe('2.4s');
    expect(formatDurationMs(12400)).toBe('12s');
    expect(formatDurationMs(undefined)).toBeUndefined();

    const settled = [
      describeToolStep({ id: '1', toolName: 'list_tabs', status: 'done', durationMs: 120 }),
      describeToolStep({ id: '2', toolName: 'read_tab', status: 'done', durationMs: 1280 }),
      describeToolStep({ id: '3', toolName: 'run_sandbox', status: 'done', durationMs: 1000 }),
    ];
    expect(formatToolChainSummary(settled, false)).toEqual({
      title: 'Used 3 tools · 2.4s',
      summary: 'Used 3 tools',
    });
    expect(formatToolChainSummary([settled[0]!], false)).toEqual({
      title: 'Used 1 tool · 120ms',
      summary: undefined,
    });

    const active = [
      settled[0]!,
      describeToolStep({
        id: '2',
        toolName: 'search_web_tab',
        status: 'running',
        args: { query: 'local webgpu inference' },
      }),
    ];
    expect(formatToolChainSummary(active, true)).toEqual({
      title: 'Searching the web · “local webgpu inference”',
      summary: 'Used 2 tools',
    });
  });

  it('merges completed active-turn steps with pendingToolCalls and liveToolExecutions into ONE chain', () => {
    const completedTurnSteps = [
      describeToolStep({
        id: 'node-1',
        toolName: 'list_tabs',
        status: 'done',
        output: '1. [1] Alpha\n2. [2] Beta',
        durationMs: 18,
      }),
    ];
    const pendingToolCalls = [
      { id: 'pending_1', name: 'read_tab' },
      { id: 'pending_2', name: 'run_sandbox' },
    ];
    const liveToolExecutions: ToolExecutionTelemetry[] = [
      {
        sequence: 0,
        name: 'list_tabs',
        status: 'done',
        startedAtMs: 100,
        durationMs: 18,
        output: '1. [1] Alpha\n2. [2] Beta',
      },
      {
        sequence: 1,
        name: 'read_tab',
        args: { tab_id: 1, title: 'Alpha Cluster Spec' },
        status: 'done',
        startedAtMs: 200,
        durationMs: 65,
        output: 'Alpha spec body',
      },
      {
        sequence: 2,
        name: 'run_sandbox',
        args: { expression: '512 * 515' },
        status: 'running',
        startedAtMs: 300,
      },
    ];

    const merged = mergeActiveTurnToolSteps({
      completedTurnSteps,
      pendingToolCalls,
      liveToolExecutions,
    });

    expect(merged).toHaveLength(3);
    expect(merged.map((s) => s.tool)).toEqual(['Checked open tabs', 'Read tab', 'Calculating']);
    expect(merged.map((s) => s.status)).toEqual(['done', 'done', 'running']);
    expect(merged[0]?.label).toBe('2 tabs');
    expect(merged[1]?.label).toBe('Alpha Cluster Spec');
    expect(merged[1]?.latencyLabel).toBe('65ms');
    expect(merged[2]?.label).toBe('512 * 515');
  });

  it('hydrates historical tool nodes with persisted ToolTraceRecord durations and args', () => {
    const toolNode = makeNode(
      'tool-1',
      'tool',
      'Result text',
      { toolName: 'search_web_tab', createdAt: 5000 },
      5000,
    );
    const callNode = makeNode(
      'call-1',
      'assistant',
      '',
      { toolName: 'search_web_tab', toolArgs: { query: 'TinyGPT' }, createdAt: 4200 },
      4200,
    );
    const traces: ToolTraceRecord[] = [
      {
        id: 'trace-1',
        threadId: 'thread-1',
        turnId: 'turn-1',
        step: 0,
        toolName: 'search_web_tab',
        arguments: { query: 'TinyGPT' },
        resultFormatted: 'Result text',
        ok: true,
        tainted: true,
        guardVerdict: 'allow',
        durationMs: 347,
        createdAt: 5050,
      },
    ];

    const steps = buildGroupStepsFromToolNodes({
      toolNodes: [{ node: toolNode, prevMessage: callNode.message }],
      persistedTraces: traces,
    });

    expect(steps).toHaveLength(1);
    expect(steps[0]?.tool).toBe('Searched the web');
    expect(steps[0]?.label).toBe('“TinyGPT”');
    expect(steps[0]?.latencyLabel).toBe('347ms');
  });
});
