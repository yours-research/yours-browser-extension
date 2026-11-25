import type { ChatMessage } from './protocol.ts';

/**
 * Compact, 100% deterministic Base System Prompt (~200 tokens) pinned in GPU Slot 0.
 * Never include dynamic timestamps, dates, or per-session IDs here so the prefix
 * cache remains valid across conversations and restarts.
 */
export const BASE_SYSTEM_PROMPT = [
  'You are TinyGPT, a fast, accurate local browser assistant.',
  'Be direct and concise. Never fabricate tool output or cite unseen pages.',
  'Inspect the active page before answering questions about what the user is viewing, search before guessing recent facts, and cite sources by title and URL.',
  '',
  'To invoke a tool, emit one block and stop:',
  '<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>',
  '',
  'Available tools:',
  '- read_active_page(mode?: "auto" | "article" | "selection" | "interactive"): Read the current tab.',
  '- web_search(query: string, limit?: number): Search the web for current facts and sources.',
  '- fetch_and_read_url(urls: string[], query?: string): Fetch and extract readable content from URLs.',
  '- list_open_tabs(): List open browser tabs with tabId, title, and URL.',
  '- read_tab(tabId: number, mode?: string): Read content from a specific open tab.',
  '- js_calc(expression: string): Evaluate deterministic math, stats, units, dates, or JSON.',
].join('\n');

export interface ActivePageContextInput {
  title: string;
  url: string;
  content: string;
  mode: string;
}

export interface AssistantModeOptions {
  pageEnabled: boolean;
  pageAttached: boolean;
  researchEnabled: boolean;
}

/**
 * Deterministic, mode-specific system prompt that advertises ONLY the tools enabled
 * by the active Page / Research toggles so the model never wastes a turn calling
 * disabled web search or re-reading an already-attached active tab.
 */
export function buildModeSystemMessage(options: AssistantModeOptions): ChatMessage {
  const { pageEnabled, pageAttached, researchEnabled } = options;

  const lines: string[] = [
    'You are TinyGPT, a fast, accurate local browser assistant.',
    'Answer general knowledge and coding questions directly and concisely without calling tools.',
    'Never fabricate tool output or cite unseen URLs.',
  ];

  if (!researchEnabled) {
    lines.push('Web search is OFF. Do NOT call web_search or fetch_and_read_url.');
  }
  if (pageEnabled && pageAttached) {
    lines.push(
      'The active browser tab content is already attached below. Answer questions about it directly without calling read_active_page.',
    );
  } else if (!pageEnabled) {
    lines.push('Active page reading is OFF unless the user explicitly mentions another open tab.');
  }

  const tools: string[] = [];
  if (pageEnabled && !pageAttached) {
    tools.push('- read_active_page(mode?: "auto" | "article" | "selection" | "interactive"): Read the current tab.');
  }
  if (researchEnabled) {
    tools.push('- web_search(query: string, limit?: number): Search the web for current events, facts, or URLs.');
    tools.push('- fetch_and_read_url(urls: string[], query?: string): Fetch and extract readable content from URLs.');
  }
  if (pageEnabled) {
    tools.push('- list_open_tabs(): List open browser tabs with tabId, title, and URL.');
    tools.push('- read_tab(tabId: number, mode?: string): Read content from a specific open tab.');
  }
  tools.push('- js_calc(expression: string): Evaluate exact math, stats, units, or date arithmetic.');

  lines.push('');
  lines.push('Only invoke a listed tool when genuinely required by emitting one block and stopping:');
  lines.push('<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>');
  lines.push('');
  lines.push('Available tools:');
  lines.push(...tools);

  return {
    role: 'system',
    content: lines.join('\n'),
  };
}

export function buildBaseSystemMessage(): ChatMessage {
  return {
    role: 'system',
    content: BASE_SYSTEM_PROMPT,
  };
}

export function buildActivePageContextMessage(page: ActivePageContextInput): ChatMessage {
  const title = page.title.trim() || 'Untitled';
  const url = page.url.trim() || 'about:blank';
  const mode = page.mode.trim() || 'auto';
  const content = page.content.trim();

  return {
    role: 'system',
    content: `[Active Tab: ${title} (${url}) | mode=${mode}]\n${content}`,
  };
}

export function buildToolResultFollowup(toolName: string, resultText: string): string {
  const safeName = toolName.trim() || 'tool';
  return `<tool_result name="${safeName}">\n${resultText}\n</tool_result>\nContinue answering using the tool output above.`;
}

/**
 * Deterministic 32-bit FNV-1a hex hash for tab URL + distilled content deduplication.
 */
export function hashStringFnv1a(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}
