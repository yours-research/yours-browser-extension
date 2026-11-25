import type { ChatMessage, ModelLoadStatus } from '../shared/protocol.ts';
import { buildToolResultFollowup, hashStringFnv1a } from '../shared/systemPrompt.ts';

export function shouldAutoPreloadModel(status: ModelLoadStatus): boolean {
  return status === 'unloaded';
}

/**
 * Trims a draft composer string back to the last whitespace boundary so we never
 * speculatively prefill a half-typed trailing word that BPE-tokenizes differently
 * once completed. Returns '' if the stable prefix is shorter than minChars (12).
 */
export function extractStableDraftPrefix(text: string, minChars = 12): string {
  if (text.length < minChars) {
    return '';
  }
  const endsWithWhitespace = /\s$/.test(text);
  const candidate = endsWithWhitespace
    ? text.replace(/\s+$/, '')
    : text.replace(/\S+$/, '').replace(/\s+$/, '');
  return candidate.length >= minChars ? candidate : '';
}

export interface TypingSpeculativePrefillOptions {
  minChars?: number;
  typeDebounceMs?: number;
  pasteDebounceMs?: number;
  pasteThresholdChars?: number;
  slotKey?: string;
  buildMessages: (stableText: string) => ChatMessage[];
  isEngineIdle: () => boolean;
  onPrefill: (slotKey: string, messages: ChatMessage[]) => Promise<void>;
  onAbort: () => void;
}

export class TypingSpeculativePrefillController {
  private readonly minChars: number;
  private readonly typeDebounceMs: number;
  private readonly pasteDebounceMs: number;
  private readonly pasteThresholdChars: number;
  private readonly slotKey: string;
  private readonly buildMessages: (stableText: string) => ChatMessage[];
  private readonly isEngineIdle: () => boolean;
  private readonly onPrefill: (slotKey: string, messages: ChatMessage[]) => Promise<void>;
  private readonly onAbort: () => void;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastText = '';
  private lastPrefilledPrefix = '';
  private prefillInFlight = false;
  private disposed = false;

  constructor(options: TypingSpeculativePrefillOptions) {
    this.minChars = options.minChars ?? 12;
    this.typeDebounceMs = options.typeDebounceMs ?? 350;
    this.pasteDebounceMs = options.pasteDebounceMs ?? 50;
    this.pasteThresholdChars = options.pasteThresholdChars ?? 120;
    this.slotKey = options.slotKey ?? 'draft:composer';
    this.buildMessages = options.buildMessages;
    this.isEngineIdle = options.isEngineIdle;
    this.onPrefill = options.onPrefill;
    this.onAbort = options.onAbort;
  }

  onTextChanged(nextText: string): void {
    if (this.disposed) {
      return;
    }

    const delta = nextText.length - this.lastText.length;
    this.lastText = nextText;
    this.clearTimer();

    const divergedFromPrefilled =
      this.lastPrefilledPrefix.length > 0 && !nextText.startsWith(this.lastPrefilledPrefix);

    if (nextText.length === 0 || divergedFromPrefilled) {
      this.lastPrefilledPrefix = '';
      this.prefillInFlight = false;
      this.onAbort();
      if (nextText.length === 0) {
        return;
      }
    }

    const stablePrefix = extractStableDraftPrefix(nextText, this.minChars);
    if (!stablePrefix || stablePrefix === this.lastPrefilledPrefix) {
      return;
    }

    const debounceMs = delta >= this.pasteThresholdChars ? this.pasteDebounceMs : this.typeDebounceMs;

    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.disposed || !this.isEngineIdle()) {
        return;
      }
      if (!this.lastText.startsWith(stablePrefix)) {
        return;
      }

      this.lastPrefilledPrefix = stablePrefix;
      this.prefillInFlight = true;
      const messages = this.buildMessages(stablePrefix);
      void this.onPrefill(this.slotKey, messages)
        .catch(() => {})
        .finally(() => {
          this.prefillInFlight = false;
        });
    }, debounceMs);
  }

  cancelForSubmit(): void {
    this.clearTimer();
    this.lastText = '';
    this.lastPrefilledPrefix = '';
    this.prefillInFlight = false;
    this.onAbort();
  }

  isPrefillInFlight(): boolean {
    return this.prefillInFlight;
  }

  dispose(): void {
    this.disposed = true;
    this.clearTimer();
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}

export interface ExtractedActiveTab {
  title: string;
  url: string;
  content: string;
  mode?: string;
  truncated?: boolean;
}

export interface ActiveTabSpeculativePrefillOptions {
  debounceMs?: number;
  minContentChars?: number;
  extractActiveTab: () => Promise<ExtractedActiveTab | null>;
  buildMessages: (tab: ExtractedActiveTab & { contentHash: string }) => ChatMessage[];
  isEngineIdle: () => boolean;
  onPrefill: (
    slotKey: string,
    messages: ChatMessage[],
    meta: { url: string; contentHash: string },
  ) => Promise<void>;
  onAbort: () => void;
}

export class ActiveTabSpeculativePrefillController {
  private readonly debounceMs: number;
  private readonly minContentChars: number;
  private readonly extractActiveTab: () => Promise<ExtractedActiveTab | null>;
  private readonly buildMessages: (
    tab: ExtractedActiveTab & { contentHash: string },
  ) => ChatMessage[];
  private readonly isEngineIdle: () => boolean;
  private readonly onPrefill: (
    slotKey: string,
    messages: ChatMessage[],
    meta: { url: string; contentHash: string },
  ) => Promise<void>;
  private readonly onAbort: () => void;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastUrl = '';
  private lastContentHash = '';
  private prefillInFlight = false;
  private generationId = 0;
  private disposed = false;

  constructor(options: ActiveTabSpeculativePrefillOptions) {
    this.debounceMs = options.debounceMs ?? 250;
    this.minContentChars = options.minContentChars ?? 40;
    this.extractActiveTab = options.extractActiveTab;
    this.buildMessages = options.buildMessages;
    this.isEngineIdle = options.isEngineIdle;
    this.onPrefill = options.onPrefill;
    this.onAbort = options.onAbort;
  }

  scheduleExtraction(): void {
    if (this.disposed) {
      return;
    }
    this.clearTimer();
    const gen = this.generationId;

    this.timer = setTimeout(() => {
      this.timer = null;
      void this.runExtraction(gen);
    }, this.debounceMs);
  }

  notifyTabChanged(reschedule = true): void {
    if (this.disposed) {
      return;
    }
    this.clearTimer();
    this.generationId += 1;
    if (this.prefillInFlight) {
      this.lastUrl = '';
      this.lastContentHash = '';
    }
    this.prefillInFlight = false;
    this.onAbort();
    if (reschedule) {
      this.scheduleExtraction();
    }
  }

  cancelForSubmit(): void {
    this.clearTimer();
    this.generationId += 1;
    if (this.prefillInFlight) {
      this.lastUrl = '';
      this.lastContentHash = '';
    }
    this.prefillInFlight = false;
    this.onAbort();
  }

  dispose(): void {
    this.disposed = true;
    this.generationId += 1;
    this.clearTimer();
  }

  private async runExtraction(gen: number): Promise<void> {
    if (this.disposed || gen !== this.generationId || !this.isEngineIdle()) {
      return;
    }

    const extracted = await this.extractActiveTab().catch(() => null);
    if (this.disposed || gen !== this.generationId || !extracted) {
      return;
    }

    const trimmedContent = extracted.content.trim();
    if (trimmedContent.length < this.minContentChars) {
      return;
    }

    const contentHash = hashStringFnv1a(`${extracted.url}\n${trimmedContent}`);
    if (extracted.url === this.lastUrl && contentHash === this.lastContentHash) {
      return;
    }

    if (!this.isEngineIdle()) {
      return;
    }

    const slotKey = `tab:${hashStringFnv1a(extracted.url)}`;
    const messages = this.buildMessages({
      ...extracted,
      content: trimmedContent,
      contentHash,
    });

    this.lastUrl = extracted.url;
    this.lastContentHash = contentHash;
    this.prefillInFlight = true;

    try {
      await this.onPrefill(slotKey, messages, {
        url: extracted.url,
        contentHash,
      });
    } catch {
      if (gen === this.generationId) {
        this.lastUrl = '';
        this.lastContentHash = '';
      }
    } finally {
      if (gen === this.generationId) {
        this.prefillInFlight = false;
      }
    }
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}

export interface ToolCallRequest {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolParserAdapter {
  detectPendingToolCall(text: string): { name: string } | null;
  extractToolCall(text: string): {
    call: ToolCallRequest | null;
    cleanText: string;
  };
}

export interface AgentToolExecutionResult {
  ok: boolean;
  text: string;
  needsPermission?: boolean;
  permissionLabel?: string;
}

export interface AgentToolExecuteContext {
  onSourceReady?: (sourceText: string) => void;
  signal?: AbortSignal;
}

export interface AgentToolBridge {
  executeTool(
    call: ToolCallRequest,
    context?: AgentToolExecuteContext,
  ): Promise<AgentToolExecutionResult>;
}

export interface AgentGenerateOptions {
  groundingText?: string;
  signal?: AbortSignal;
  onDelta?: (delta: string, tokenCount: number) => void;
}

export interface AgentGenerateResult {
  text: string;
  finishReason: 'stop' | 'length' | 'abort';
}

export interface AgentEngineBridge {
  generate(
    requestId: string,
    messages: ChatMessage[],
    options?: AgentGenerateOptions,
  ): Promise<AgentGenerateResult>;
  abort?(requestId: string): void;
}

export type ToolStatusPhase =
  | 'pending_call'
  | 'running'
  | 'completed'
  | 'error'
  | 'permission_required';

export interface ToolStatusUpdate {
  step: number;
  phase: ToolStatusPhase;
  toolName: string;
  label: string;
  arguments?: Record<string, unknown>;
  error?: string;
}

const TOOL_CALL_BLOCK_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/;
const PARTIAL_TOOL_NAME_RE = /"name"\s*:\s*"([^"]+)"/;

export function createDefaultToolParser(): ToolParserAdapter {
  return {
    detectPendingToolCall(text: string) {
      const openIdx = text.lastIndexOf('<tool_call>');
      if (openIdx === -1) {
        return null;
      }
      const closeIdx = text.indexOf('</tool_call>', openIdx);
      if (closeIdx !== -1) {
        return null;
      }
      const tail = text.slice(openIdx);
      const nameMatch = PARTIAL_TOOL_NAME_RE.exec(tail);
      return { name: nameMatch?.[1] ?? 'tool' };
    },

    extractToolCall(text: string) {
      const match = TOOL_CALL_BLOCK_RE.exec(text);
      if (!match) {
        return { call: null, cleanText: text };
      }

      const rawJson = match[1].trim();
      try {
        const parsed = JSON.parse(rawJson) as {
          name?: unknown;
          arguments?: unknown;
        };
        if (typeof parsed.name !== 'string' || parsed.name.trim() === '') {
          return { call: null, cleanText: text };
        }
        const args =
          parsed.arguments &&
          typeof parsed.arguments === 'object' &&
          !Array.isArray(parsed.arguments)
            ? (parsed.arguments as Record<string, unknown>)
            : {};

        const cleanText = text.replace(match[0], '').trim();
        return {
          call: { name: parsed.name.trim(), arguments: args },
          cleanText,
        };
      } catch {
        return { call: null, cleanText: text };
      }
    },
  };
}

export function formatToolActivityLabel(toolName: string): string {
  switch (toolName) {
    case 'read_active_page':
      return 'Reading active page...';
    case 'web_search':
      return 'Searching...';
    case 'fetch_and_read_url':
      return 'Reading web sources...';
    case 'list_open_tabs':
      return 'Listing open tabs...';
    case 'read_tab':
      return 'Reading tab...';
    case 'js_calc':
      return 'Calculating...';
    default:
      return `Running ${toolName}...`;
  }
}

export interface RunAgentTurnParams {
  messages: ChatMessage[];
  baseSystemMessage?: ChatMessage;
  activePageContext?: ChatMessage | null;
  initialGroundingText?: string;
  maxToolSteps?: number;
  engine: AgentEngineBridge;
  tools?: AgentToolBridge;
  parser?: ToolParserAdapter;
  signal?: AbortSignal;
  makeRequestId?: (step: number) => string;
  onDelta?: (delta: string, accumulatedStepText: string, step: number) => void;
  onToolStatus?: (status: ToolStatusUpdate) => void;
  onToolResult?: (info: {
    step: number;
    call: ToolCallRequest;
    result: AgentToolExecutionResult;
  }) => void;
  onIncrementalContextReady?: (sourceText: string) => void;
  onAssistantMessage?: (message: ChatMessage, step: number) => void;
}

export interface RunAgentTurnResult {
  finalText: string;
  messages: ChatMessage[];
  stepsExecuted: number;
  toolCalls: Array<{
    step: number;
    call: ToolCallRequest;
    result: AgentToolExecutionResult;
  }>;
  groundingText: string;
  stoppedForPermission: boolean;
  permissionRequest?: {
    toolName: string;
    label: string;
  };
  finishReason: 'stop' | 'length' | 'abort' | 'permission_required' | 'max_steps';
}

function buildInitialTurnMessages(
  messages: readonly ChatMessage[],
  baseSystemMessage?: ChatMessage,
  activePageContext?: ChatMessage | null,
): ChatMessage[] {
  const result: ChatMessage[] = [];
  if (baseSystemMessage) {
    const alreadyHasBase =
      messages[0]?.role === 'system' && messages[0]?.content === baseSystemMessage.content;
    if (!alreadyHasBase) {
      result.push({ ...baseSystemMessage });
    }
  }
  if (activePageContext) {
    const alreadyHasPage = messages.some(
      (m) => m.role === activePageContext.role && m.content === activePageContext.content,
    );
    if (!alreadyHasPage) {
      result.push({ ...activePageContext });
    }
  }
  for (const msg of messages) {
    result.push({ role: msg.role, content: msg.content });
  }
  return result;
}

export async function runAgentTurn(params: RunAgentTurnParams): Promise<RunAgentTurnResult> {
  const maxToolSteps = Math.max(0, params.maxToolSteps ?? 4);
  const parser = params.parser ?? createDefaultToolParser();
  const turnMessages = buildInitialTurnMessages(
    params.messages,
    params.baseSystemMessage,
    params.activePageContext,
  );
  const toolCalls: RunAgentTurnResult['toolCalls'] = [];
  let groundingText = params.initialGroundingText ?? '';

  for (let step = 0; step <= maxToolSteps; step += 1) {
    if (params.signal?.aborted) {
      return {
        finalText: '',
        messages: turnMessages,
        stepsExecuted: step,
        toolCalls,
        groundingText,
        stoppedForPermission: false,
        finishReason: 'abort',
      };
    }

    const requestId = params.makeRequestId
      ? params.makeRequestId(step)
      : `agent-step-${step}-${Date.now()}`;

    let accText = '';
    let lastPendingToolName: string | null = null;

    const genResult = await params.engine.generate(requestId, turnMessages, {
      groundingText: groundingText || undefined,
      signal: params.signal,
      onDelta: (delta) => {
        accText += delta;
        const pending = parser.detectPendingToolCall(accText);
        if (pending && pending.name !== lastPendingToolName) {
          lastPendingToolName = pending.name;
          params.onToolStatus?.({
            step,
            phase: 'pending_call',
            toolName: pending.name,
            label: formatToolActivityLabel(pending.name),
          });
        }
        params.onDelta?.(delta, accText, step);
      },
    });

    const rawText = genResult.text || accText;

    if (genResult.finishReason === 'abort' || params.signal?.aborted) {
      return {
        finalText: rawText,
        messages: turnMessages,
        stepsExecuted: step + 1,
        toolCalls,
        groundingText,
        stoppedForPermission: false,
        finishReason: 'abort',
      };
    }

    const { call, cleanText } = parser.extractToolCall(rawText);

    if (!call || !params.tools || toolCalls.length >= maxToolSteps) {
      const assistantMsg: ChatMessage = { role: 'assistant', content: cleanText };
      turnMessages.push(assistantMsg);
      params.onAssistantMessage?.(assistantMsg, step);

      return {
        finalText: cleanText,
        messages: turnMessages,
        stepsExecuted: step + 1,
        toolCalls,
        groundingText,
        stoppedForPermission: false,
        finishReason:
          call && toolCalls.length >= maxToolSteps ? 'max_steps' : genResult.finishReason,
      };
    }

    turnMessages.push({ role: 'assistant', content: rawText });

    params.onToolStatus?.({
      step,
      phase: 'running',
      toolName: call.name,
      label: formatToolActivityLabel(call.name),
      arguments: call.arguments,
    });

    const result = await params.tools.executeTool(call, {
      signal: params.signal,
      onSourceReady: (sourceText) => {
        if (!sourceText) {
          return;
        }
        groundingText = groundingText ? `${groundingText}\n\n${sourceText}` : sourceText;
        params.onIncrementalContextReady?.(sourceText);
      },
    });

    toolCalls.push({ step, call, result });
    params.onToolResult?.({ step, call, result });

    if (result.needsPermission) {
      const permissionLabel = result.permissionLabel ?? 'Grant Web Research Permission';
      params.onToolStatus?.({
        step,
        phase: 'permission_required',
        toolName: call.name,
        label: permissionLabel,
        arguments: call.arguments,
      });

      return {
        finalText: cleanText,
        messages: turnMessages,
        stepsExecuted: step + 1,
        toolCalls,
        groundingText,
        stoppedForPermission: true,
        permissionRequest: {
          toolName: call.name,
          label: permissionLabel,
        },
        finishReason: 'permission_required',
      };
    }

    params.onToolStatus?.({
      step,
      phase: result.ok ? 'completed' : 'error',
      toolName: call.name,
      label: result.ok ? `${call.name} complete` : `${call.name} failed`,
      arguments: call.arguments,
      error: result.ok ? undefined : result.text,
    });

    turnMessages.push({
      role: 'user',
      content: buildToolResultFollowup(call.name, result.text),
    });
  }

  return {
    finalText: '',
    messages: turnMessages,
    stepsExecuted: maxToolSteps + 1,
    toolCalls,
    groundingText,
    stoppedForPermission: false,
    finishReason: 'max_steps',
  };
}
