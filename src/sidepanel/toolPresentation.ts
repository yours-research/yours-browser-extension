import type { MessageNode, ToolTraceRecord } from '@yours/harness';
import type { ToolTraceStep, ToolTraceStatus } from '@yours/ui';

export type ToolStatusKind = NonNullable<ToolTraceStep['state']>;

export type ToolIconKind = ToolTraceStep['icon'];

export interface ToolExecutionTelemetry {
  sequence: number;
  name: string;
  args?: Record<string, unknown>;
  status: ToolTraceStatus;
  startedAtMs: number;
  durationMs?: number;
  output?: string;
  data?: unknown;
  sourceOrigin?: string;
  error?: string;
}

export interface PendingToolCallInfo {
  id: string;
  name: string;
}

export interface DescribedToolStep extends ToolTraceStep {
  rawToolName: string;
  statusKind: ToolStatusKind;
  detail?: string;
  durationMs?: number;
}

export interface ActivePageHint {
  title?: string | null;
  url?: string | null;
}

interface ToolVerbSpec {
  statusKind: ToolStatusKind;
  icon: ToolIconKind;
  running: string;
  done: string;
  blocked: string;
  error: string;
}

const TOOL_SPECS: Record<string, ToolVerbSpec> = {
  read_active_tab: {
    statusKind: 'reading',
    icon: 'page-scan',
    running: 'Reading active page',
    done: 'Read active page',
    blocked: 'Blocked active page read',
    error: 'Failed reading active page',
  },
  read_active_page: {
    statusKind: 'reading',
    icon: 'page-scan',
    running: 'Reading active page',
    done: 'Read active page',
    blocked: 'Blocked active page read',
    error: 'Failed reading active page',
  },
  list_tabs: {
    statusKind: 'tabs',
    icon: 'page-scan',
    running: 'Checking open tabs',
    done: 'Checked open tabs',
    blocked: 'Blocked open tabs check',
    error: 'Failed checking open tabs',
  },
  list_open_tabs: {
    statusKind: 'tabs',
    icon: 'page-scan',
    running: 'Checking open tabs',
    done: 'Checked open tabs',
    blocked: 'Blocked open tabs check',
    error: 'Failed checking open tabs',
  },
  read_tab: {
    statusKind: 'reading',
    icon: 'page-scan',
    running: 'Reading tab',
    done: 'Read tab',
    blocked: 'Blocked tab read',
    error: 'Failed reading tab',
  },
  search_web_tab: {
    statusKind: 'searching',
    icon: 'globe',
    running: 'Searching the web',
    done: 'Searched the web',
    blocked: 'Blocked web search',
    error: 'Web search failed',
  },
  web_search: {
    statusKind: 'searching',
    icon: 'globe',
    running: 'Searching the web',
    done: 'Searched the web',
    blocked: 'Blocked web search',
    error: 'Web search failed',
  },
  fetch_background_tab: {
    statusKind: 'reading',
    icon: 'globe',
    running: 'Reading web source',
    done: 'Read web source',
    blocked: 'Blocked web source',
    error: 'Failed reading source',
  },
  fetch_and_read_url: {
    statusKind: 'reading',
    icon: 'globe',
    running: 'Reading web source',
    done: 'Read web source',
    blocked: 'Blocked web source',
    error: 'Failed reading source',
  },
  run_sandbox: {
    statusKind: 'calculating',
    icon: 'compare',
    running: 'Calculating',
    done: 'Calculated',
    blocked: 'Blocked calculation',
    error: 'Calculation failed',
  },
  js_calc: {
    statusKind: 'calculating',
    icon: 'compare',
    running: 'Calculating',
    done: 'Calculated',
    blocked: 'Blocked calculation',
    error: 'Calculation failed',
  },
  memory_recall: {
    statusKind: 'memory',
    icon: 'bookmark',
    running: 'Recalling memory',
    done: 'Recalled memory',
    blocked: 'Blocked memory recall',
    error: 'Memory recall failed',
  },
  memory_save: {
    statusKind: 'memory',
    icon: 'bookmark',
    running: 'Saving memory',
    done: 'Saved memory',
    blocked: 'Blocked memory save',
    error: 'Memory save failed',
  },
  interact_page: {
    statusKind: 'analyzing',
    icon: 'sliders',
    running: 'Interacting with page',
    done: 'Interacted with page',
    blocked: 'Blocked page action',
    error: 'Page action failed',
  },
  read_document: {
    statusKind: 'reading',
    icon: 'page-scan',
    running: 'Reading document',
    done: 'Read document',
    blocked: 'Blocked document read',
    error: 'Document read failed',
  },
  tool: {
    statusKind: 'analyzing',
    icon: 'sliders',
    running: 'Preparing tool',
    done: 'Used tool',
    blocked: 'Blocked tool step',
    error: 'Tool step failed',
  },
};

export function humanizeToolSlug(toolName: string): string {
  const cleaned = toolName.trim().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ');
  if (!cleaned) return 'Tool Step';
  return cleaned.replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function getToolVerbSpec(toolName?: string): ToolVerbSpec {
  const key = (toolName ?? '').trim().toLowerCase();
  if (key && TOOL_SPECS[key]) {
    return TOOL_SPECS[key]!;
  }
  const titled = humanizeToolSlug(key || 'tool');
  return {
    statusKind: 'analyzing',
    icon: 'sliders',
    running: `Running ${titled}`,
    done: `Ran ${titled}`,
    blocked: `Blocked ${titled}`,
    error: `${titled} failed`,
  };
}

export function normalizeToolKey(toolName?: string): string {
  const raw = (toolName ?? '').trim().toLowerCase();
  switch (raw) {
    case 'read_active_page':
      return 'read_active_tab';
    case 'list_open_tabs':
      return 'list_tabs';
    case 'web_search':
      return 'search_web_tab';
    case 'fetch_and_read_url':
      return 'fetch_background_tab';
    case 'js_calc':
      return 'run_sandbox';
    default:
      return raw || 'tool';
  }
}

export function formatDurationMs(durationMs?: number): string | undefined {
  if (typeof durationMs !== 'number' || !Number.isFinite(durationMs) || durationMs < 0) {
    return undefined;
  }
  if (durationMs < 1000) {
    return `${Math.max(1, Math.round(durationMs))}ms`;
  }
  const seconds = durationMs / 1000;
  return `${seconds >= 10 ? Math.round(seconds) : seconds.toFixed(1)}s`;
}

function truncateText(text: string, maxLen = 44): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (compact.length <= maxLen) return compact;
  return `${compact.slice(0, maxLen - 1).trimEnd()}…`;
}

function extractHostname(rawUrl?: string | null): string | undefined {
  if (!rawUrl || typeof rawUrl !== 'string') return undefined;
  try {
    const parsed = new URL(rawUrl.trim());
    const host = parsed.hostname.replace(/^www\./i, '');
    return host || undefined;
  } catch {
    return undefined;
  }
}

function extractTabCount(output?: string, data?: unknown): number | null {
  if (Array.isArray(data)) {
    return data.length;
  }
  if (data && typeof data === 'object' && Array.isArray((data as { tabs?: unknown[] }).tabs)) {
    return (data as { tabs: unknown[] }).tabs.length;
  }
  if (!output) return null;
  const tabHeaderMatches = output.match(/^\s*(?:\[\d+\]|\d+\.)\s+/gm);
  if (tabHeaderMatches && tabHeaderMatches.length > 0) {
    return tabHeaderMatches.length;
  }
  const openTabsMatch = /\b(\d+)\s+open\s+tabs?\b/i.exec(output);
  if (openTabsMatch) {
    const parsed = Number.parseInt(openTabsMatch[1]!, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function extractTabTitleFromOutput(output?: string): string | undefined {
  if (!output) return undefined;
  const titleLine = /^(?:Title|Tab):\s*(.+)$/im.exec(output);
  if (titleLine?.[1]?.trim()) {
    return truncateText(titleLine[1].trim(), 42);
  }
  return undefined;
}

export function extractToolDetail(params: {
  toolName?: string;
  args?: Record<string, unknown>;
  output?: string;
  data?: unknown;
  sourceOrigin?: string;
  activePage?: ActivePageHint;
}): string | undefined {
  const canonical = normalizeToolKey(params.toolName);
  const args = params.args ?? {};

  switch (canonical) {
    case 'search_web_tab': {
      const q = typeof args.query === 'string' ? args.query.trim() : '';
      return q ? `“${truncateText(q, 38)}”` : undefined;
    }
    case 'fetch_background_tab': {
      const host =
        extractHostname(typeof args.url === 'string' ? args.url : undefined) ??
        extractHostname(params.sourceOrigin);
      return host ?? (typeof args.url === 'string' ? truncateText(args.url, 40) : undefined);
    }
    case 'read_active_tab': {
      const argTitle = typeof args.title === 'string' ? args.title.trim() : '';
      const activeTitle = params.activePage?.title?.trim();
      const host =
        extractHostname(params.sourceOrigin) ?? extractHostname(params.activePage?.url);
      if (activeTitle) return truncateText(activeTitle, 40);
      if (argTitle) return truncateText(argTitle, 40);
      return host;
    }
    case 'read_tab': {
      const argTitle = typeof args.title === 'string' ? args.title.trim() : '';
      if (argTitle) return truncateText(argTitle, 40);
      const outputTitle = extractTabTitleFromOutput(params.output);
      if (outputTitle) return outputTitle;
      const tabId = args.tab_id ?? args.tabId ?? args.id;
      if (typeof tabId === 'number' || (typeof tabId === 'string' && tabId.trim())) {
        return `Tab #${String(tabId).trim()}`;
      }
      return extractHostname(params.sourceOrigin);
    }
    case 'list_tabs': {
      const count = extractTabCount(params.output, params.data);
      if (count !== null) {
        return `${count} ${count === 1 ? 'tab' : 'tabs'}`;
      }
      if (args.window_scope === 'all' || args.windowScope === 'all') {
        return 'All windows';
      }
      return undefined;
    }
    case 'run_sandbox': {
      const expr =
        typeof args.expression === 'string'
          ? args.expression
          : typeof args.code === 'string'
            ? args.code
            : typeof args.data === 'string'
              ? args.data
              : typeof args.mode === 'string'
                ? args.mode
                : '';
      return expr.trim() ? truncateText(expr.trim(), 40) : undefined;
    }
    case 'memory_recall': {
      const query =
        typeof args.query === 'string'
          ? args.query
          : typeof args.key === 'string'
            ? args.key
            : '';
      return query.trim() ? `“${truncateText(query.trim(), 38)}”` : undefined;
    }
    case 'memory_save': {
      const value =
        typeof args.key === 'string'
          ? args.key
          : typeof args.content === 'string'
            ? args.content
            : typeof args.fact === 'string'
              ? args.fact
              : '';
      return value.trim() ? truncateText(value.trim(), 40) : undefined;
    }
    case 'interact_page': {
      const action = typeof args.action === 'string' ? args.action.trim() : '';
      const target =
        typeof args.selector === 'string'
          ? args.selector.trim()
          : extractHostname(params.sourceOrigin) ??
            extractHostname(params.activePage?.url) ??
            '';
      if (action && target) return truncateText(`${action} · ${target}`, 42);
      if (action) return action;
      return target ? truncateText(target, 42) : undefined;
    }
    case 'read_document': {
      const target =
        extractHostname(typeof args.url === 'string' ? args.url : undefined) ??
        (typeof args.path === 'string' ? args.path.trim() : '') ??
        (typeof args.format === 'string' ? args.format.trim() : '');
      return target ? truncateText(target, 40) : undefined;
    }
    default: {
      const candidate =
        typeof args.query === 'string'
          ? args.query
          : typeof args.url === 'string'
            ? extractHostname(args.url) ?? args.url
            : typeof args.title === 'string'
              ? args.title
              : '';
      return candidate.trim() ? truncateText(candidate.trim(), 40) : undefined;
    }
  }
}

export function formatToolActivityLabel(
  toolName: string,
  args?: Record<string, unknown>,
  activePage?: ActivePageHint,
): string {
  const spec = getToolVerbSpec(toolName);
  const detail = extractToolDetail({ toolName, args, activePage });
  return detail ? `${spec.running} · ${detail}` : `${spec.running}…`;
}

export function describeToolStep(params: {
  id: string;
  toolName?: string;
  status: ToolTraceStatus;
  args?: Record<string, unknown>;
  input?: string;
  output?: string;
  data?: unknown;
  sourceOrigin?: string;
  durationMs?: number;
  activePage?: ActivePageHint;
}): DescribedToolStep {
  const rawName = (params.toolName ?? 'tool').trim() || 'tool';
  const spec = getToolVerbSpec(rawName);
  const verb =
    params.status === 'running'
      ? spec.running
      : params.status === 'blocked'
        ? spec.blocked
        : params.status === 'error'
          ? spec.error
          : spec.done;

  let parsedArgs = params.args;
  if (!parsedArgs && params.input) {
    try {
      const maybe = JSON.parse(params.input) as unknown;
      if (maybe && typeof maybe === 'object' && !Array.isArray(maybe)) {
        parsedArgs = maybe as Record<string, unknown>;
      }
    } catch {
      // Ignore malformed serialized args
    }
  }

  const detail = extractToolDetail({
    toolName: rawName,
    args: parsedArgs,
    output: params.output,
    data: params.data,
    sourceOrigin: params.sourceOrigin,
    activePage: params.activePage,
  });

  const resolvedInput =
    params.input ??
    (parsedArgs && Object.keys(parsedArgs).length > 0 ? JSON.stringify(parsedArgs) : undefined);

  const state: ToolStatusKind =
    params.status === 'blocked'
      ? 'blocked'
      : params.status === 'error'
        ? 'error'
        : spec.statusKind;

  return {
    id: params.id,
    tool: verb,
    actionLabel: verb,
    rawToolName: rawName,
    label: detail,
    detail,
    icon: spec.icon,
    status: params.status,
    state,
    statusKind: state,
    durationMs: params.durationMs,
    latencyLabel: formatDurationMs(params.durationMs),
    input: resolvedInput,
    output: params.output,
  };
}

export function formatToolChainSummary(
  steps: readonly DescribedToolStep[],
  isActiveTurn: boolean,
): { title: string; summary?: string } {
  if (steps.length === 0) {
    return { title: 'Tool steps', summary: 'Used 0 tools' };
  }
  const runningStep = steps.find((s) => s.status === 'running');
  const totalDurationMs = steps.reduce((sum, step) => {
    return typeof step.durationMs === 'number' && Number.isFinite(step.durationMs)
      ? sum + step.durationMs
      : sum;
  }, 0);
  const hasAnyDuration = steps.some(
    (s) => typeof s.durationMs === 'number' && Number.isFinite(s.durationMs),
  );
  const durationText = hasAnyDuration ? formatDurationMs(totalDurationMs) : undefined;

  if (isActiveTurn || runningStep) {
    const activeLead = runningStep ?? steps[steps.length - 1]!;
    return {
      title: activeLead.label ? `${activeLead.tool} · ${activeLead.label}` : activeLead.tool,
      summary: steps.length > 1 ? `Used ${steps.length} tools` : undefined,
    };
  }

  const countLabel = `${steps.length} ${steps.length === 1 ? 'tool' : 'tools'}`;
  const headline = durationText ? `Used ${countLabel} · ${durationText}` : `Used ${countLabel}`;
  return {
    title: headline,
    summary: steps.length > 1 ? `Used ${countLabel}` : undefined,
  };
}

export function inferToolNodeStatus(content: string): ToolTraceStatus {
  if (/^blocked|denied/i.test(content)) return 'blocked';
  if (/^error\b|failed/i.test(content)) return 'error';
  return 'done';
}

function findMatchingPersistedTrace(
  toolNode: MessageNode,
  persistedTraces: readonly ToolTraceRecord[],
  usedTraceIndices: Set<number>,
): ToolTraceRecord | undefined {
  const msg = toolNode.message;
  const targetName = normalizeToolKey(msg.toolName);
  const trimmedOutput = msg.content.trim().slice(0, 120);

  let bestIdx = -1;
  let bestTimeDiff = Number.POSITIVE_INFINITY;

  for (let i = 0; i < persistedTraces.length; i += 1) {
    if (usedTraceIndices.has(i)) continue;
    const trace = persistedTraces[i]!;
    if (normalizeToolKey(trace.toolName) !== targetName) continue;
    const diff = Math.abs(trace.createdAt - msg.createdAt);
    const matchesOutput =
      !trimmedOutput || trace.resultFormatted.trim().slice(0, 120) === trimmedOutput;
    if (matchesOutput && diff < 10_000 && diff < bestTimeDiff) {
      bestTimeDiff = diff;
      bestIdx = i;
    }
  }

  if (bestIdx === -1) {
    for (let i = 0; i < persistedTraces.length; i += 1) {
      if (usedTraceIndices.has(i)) continue;
      const trace = persistedTraces[i]!;
      if (normalizeToolKey(trace.toolName) !== targetName) continue;
      const diff = Math.abs(trace.createdAt - msg.createdAt);
      if (diff < 30_000 && diff < bestTimeDiff) {
        bestTimeDiff = diff;
        bestIdx = i;
      }
    }
  }

  if (bestIdx >= 0) {
    usedTraceIndices.add(bestIdx);
    return persistedTraces[bestIdx];
  }
  return undefined;
}

export function buildGroupStepsFromToolNodes(params: {
  toolNodes: readonly { node: MessageNode; prevMessage?: MessageNode['message'] }[];
  persistedTraces?: readonly ToolTraceRecord[];
  activePage?: ActivePageHint;
}): DescribedToolStep[] {
  const usedTraceIndices = new Set<number>();
  return params.toolNodes.map(({ node, prevMessage }) => {
    const tm = node.message;
    const matchedTrace = params.persistedTraces
      ? findMatchingPersistedTrace(node, params.persistedTraces, usedTraceIndices)
      : undefined;
    const prevArgs =
      prevMessage?.role === 'assistant' && prevMessage.toolArgs
        ? prevMessage.toolArgs
        : matchedTrace?.arguments;
    const rawDelta =
      prevMessage?.role === 'assistant' &&
      typeof tm.createdAt === 'number' &&
      typeof prevMessage.createdAt === 'number' &&
      tm.createdAt >= prevMessage.createdAt
        ? tm.createdAt - prevMessage.createdAt
        : undefined;
    const durationMs = matchedTrace?.durationMs ?? (rawDelta && rawDelta > 0 ? rawDelta : undefined);

    return describeToolStep({
      id: node.id,
      toolName: tm.toolName ?? matchedTrace?.toolName ?? 'tool',
      status: inferToolNodeStatus(tm.content),
      args: prevArgs,
      output: tm.content.slice(0, 320),
      durationMs,
      activePage: params.activePage,
    });
  });
}

/**
 * Merges completed tool nodes belonging to the active in-flight turn (if any) with live
 * `pendingToolCalls` + `liveToolExecutions` into ONE unified ordered list of tool steps.
 */
export function mergeActiveTurnToolSteps(params: {
  completedTurnSteps?: readonly DescribedToolStep[];
  pendingToolCalls: readonly PendingToolCallInfo[];
  liveToolExecutions: readonly ToolExecutionTelemetry[];
  activePage?: ActivePageHint;
}): DescribedToolStep[] {
  const steps: DescribedToolStep[] = [...(params.completedTurnSteps ?? [])];
  const matchedExecIndices = new Set<number>();

  // Skip live executions already represented by persisted completedTurnSteps
  for (const existing of steps) {
    const targetName = normalizeToolKey(existing.rawToolName);
    const idx = params.liveToolExecutions.findIndex(
      (exec, i) => !matchedExecIndices.has(i) && normalizeToolKey(exec.name) === targetName,
    );
    if (idx >= 0) {
      matchedExecIndices.add(idx);
    }
  }

  for (const tc of params.pendingToolCalls) {
    const targetName = normalizeToolKey(tc.name);
    const execIdx = params.liveToolExecutions.findIndex(
      (exec, i) => !matchedExecIndices.has(i) && normalizeToolKey(exec.name) === targetName,
    );
    const exec = execIdx >= 0 ? params.liveToolExecutions[execIdx] : undefined;
    if (execIdx >= 0) {
      matchedExecIndices.add(execIdx);
    }

    steps.push(
      describeToolStep({
        id: tc.id,
        toolName: exec?.name ?? tc.name,
        status: exec?.status ?? 'running',
        args: exec?.args,
        output: exec?.output?.slice(0, 320),
        data: exec?.data,
        sourceOrigin: exec?.sourceOrigin,
        durationMs: exec?.durationMs,
        activePage: params.activePage,
      }),
    );
  }

  // Append any remaining live executions not yet reflected in pendingToolCalls
  for (let i = 0; i < params.liveToolExecutions.length; i += 1) {
    if (matchedExecIndices.has(i)) continue;
    const exec = params.liveToolExecutions[i]!;
    steps.push(
      describeToolStep({
        id: `live-exec-${exec.sequence}`,
        toolName: exec.name,
        status: exec.status,
        args: exec.args,
        output: exec.output?.slice(0, 320),
        data: exec.data,
        sourceOrigin: exec.sourceOrigin,
        durationMs: exec.durationMs,
        activePage: params.activePage,
      }),
    );
  }

  return steps;
}
