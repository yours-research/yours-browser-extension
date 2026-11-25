import type {
  ParsedToolCall,
  PendingToolDetection,
  Tier1ToolName,
  ToolDefinition,
  ToolExecutionResult,
} from './types.ts';

export const TIER1_TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    name: 'read_active_page',
    description:
      'Extract clean readable text, selected text, YouTube captions, PDF text, or interactive elements from the currently active browser tab.',
    parameters: {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          description: 'Extraction mode (default auto).',
          enum: ['auto', 'article', 'selection', 'interactive'],
        },
        maxChars: {
          type: 'integer',
          description: 'Maximum characters to return (default 8000).',
          minimum: 500,
          maximum: 24000,
        },
      },
    },
  },
  {
    name: 'web_search',
    description: 'Search the web using DuckDuckGo and return top results with URLs and snippets.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query string.',
        },
        limit: {
          type: 'integer',
          description: 'Number of results (1-8, default 5).',
          minimum: 1,
          maximum: 8,
        },
      },
      required: ['query'],
    },
    requiresHostPermission: true,
  },
  {
    name: 'fetch_and_read_url',
    description:
      'Fetch 1-5 URLs in parallel, distill HTML to clean markdown, and extract the most relevant passages.',
    parameters: {
      type: 'object',
      properties: {
        urls: {
          type: 'array',
          description: 'List of 1-5 http/https URLs to read.',
          items: { type: 'string' },
        },
        query: {
          type: 'string',
          description: 'Optional focus query for BM25 passage ranking.',
        },
        maxCharsPerSource: {
          type: 'integer',
          description: 'Max characters per source excerpt (default 4800).',
          minimum: 500,
          maximum: 16000,
        },
      },
      required: ['urls'],
    },
    requiresHostPermission: true,
  },
  {
    name: 'list_open_tabs',
    description: 'List currently open browser tabs (id, title, url, active, windowId).',
    parameters: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'read_tab',
    description: 'Extract readable content or interactive elements from a specific open tab by tabId.',
    parameters: {
      type: 'object',
      properties: {
        tabId: {
          type: 'integer',
          description: 'Numeric Chrome tab ID from list_open_tabs.',
        },
        mode: {
          type: 'string',
          description: 'Extraction mode (default auto).',
          enum: ['auto', 'article', 'selection', 'interactive'],
        },
        maxChars: {
          type: 'integer',
          description: 'Maximum characters to return (default 8000).',
          minimum: 500,
          maximum: 24000,
        },
      },
      required: ['tabId'],
    },
  },
  {
    name: 'js_calc',
    description:
      'Deterministic CSP-safe calculator for math expressions, unit conversions, date arithmetic, JSON path queries, and CSV stats.',
    parameters: {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          description: 'Calculation mode (default expr).',
          enum: ['expr', 'unit', 'date', 'json', 'csv'],
        },
        expression: {
          type: 'string',
          description: 'Math expression for expr mode.',
        },
        value: {
          type: 'string',
          description: 'Value for unit/date/json modes.',
        },
        from: {
          type: 'string',
          description: 'Source unit for unit mode.',
        },
        to: {
          type: 'string',
          description: 'Target unit for unit mode.',
        },
        dateA: {
          type: 'string',
          description: 'First ISO date.',
        },
        dateB: {
          type: 'string',
          description: 'Second ISO date.',
        },
        days: {
          type: 'number',
          description: 'Day offset for date mode.',
        },
        data: {
          type: 'string',
          description: 'Raw JSON or CSV text.',
        },
        path: {
          type: 'string',
          description: 'JSON path like a.b[0].c.',
        },
      },
    },
  },
];

const VALID_TOOL_NAMES = new Set<Tier1ToolName>([
  'read_active_page',
  'web_search',
  'fetch_and_read_url',
  'list_open_tabs',
  'read_tab',
  'js_calc',
]);

const TOOL_UI_LABELS: Record<Tier1ToolName, string> = {
  read_active_page: 'Reading active page...',
  web_search: 'Searching the web...',
  fetch_and_read_url: 'Reading sources...',
  list_open_tabs: 'Checking open tabs...',
  read_tab: 'Reading tab...',
  js_calc: 'Calculating...',
};

const VALID_PAGE_MODES = new Set(['auto', 'article', 'selection', 'interactive']);
const VALID_CALC_MODES = new Set(['expr', 'unit', 'date', 'json', 'csv']);

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.round(value)));
}

export function validateToolCall(call: {
  name: string;
  arguments: Record<string, unknown>;
}): { valid: boolean; normalized?: ParsedToolCall; error?: string } {
  if (!call || typeof call.name !== 'string' || !VALID_TOOL_NAMES.has(call.name as Tier1ToolName)) {
    return { valid: false, error: `Unknown tool name "${String(call?.name ?? '')}".` };
  }

  const name = call.name as Tier1ToolName;
  const args = call.arguments && typeof call.arguments === 'object' && !Array.isArray(call.arguments)
    ? call.arguments
    : {};

  switch (name) {
    case 'read_active_page': {
      const mode = typeof args.mode === 'string' && VALID_PAGE_MODES.has(args.mode) ? args.mode : 'auto';
      const maxChars = clampInt(args.maxChars, 500, 24000, 8000);
      return { valid: true, normalized: { name, arguments: { mode, maxChars } } };
    }

    case 'web_search': {
      if (typeof args.query !== 'string' || !args.query.trim()) {
        return { valid: false, error: 'web_search requires a non-empty string "query".' };
      }
      const limit = clampInt(args.limit, 1, 8, 5);
      return {
        valid: true,
        normalized: { name, arguments: { query: args.query.trim(), limit } },
      };
    }

    case 'fetch_and_read_url': {
      if (!Array.isArray(args.urls)) {
        return { valid: false, error: 'fetch_and_read_url requires an array "urls".' };
      }
      const cleanUrls = args.urls
        .filter((u): u is string => typeof u === 'string' && u.trim().length > 0)
        .map((u) => u.trim())
        .slice(0, 5);
      if (cleanUrls.length === 0) {
        return { valid: false, error: 'fetch_and_read_url requires at least 1 valid URL string.' };
      }
      const normalizedArgs: Record<string, unknown> = {
        urls: cleanUrls,
        maxCharsPerSource: clampInt(args.maxCharsPerSource, 500, 16000, 4800),
      };
      if (typeof args.query === 'string' && args.query.trim()) {
        normalizedArgs.query = args.query.trim();
      }
      return { valid: true, normalized: { name, arguments: normalizedArgs } };
    }

    case 'list_open_tabs':
      return { valid: true, normalized: { name, arguments: {} } };

    case 'read_tab': {
      if (typeof args.tabId !== 'number' || !Number.isInteger(args.tabId) || args.tabId < 0) {
        return { valid: false, error: 'read_tab requires a valid non-negative integer "tabId".' };
      }
      const mode = typeof args.mode === 'string' && VALID_PAGE_MODES.has(args.mode) ? args.mode : 'auto';
      const maxChars = clampInt(args.maxChars, 500, 24000, 8000);
      return {
        valid: true,
        normalized: { name, arguments: { tabId: args.tabId, mode, maxChars } },
      };
    }

    case 'js_calc': {
      const mode =
        typeof args.mode === 'string'
          ? args.mode
          : typeof args.expression === 'string'
            ? 'expr'
            : undefined;
      if (mode && !VALID_CALC_MODES.has(mode)) {
        return { valid: false, error: `Invalid js_calc mode "${mode}".` };
      }
      const resolvedMode = mode ?? 'expr';
      if (resolvedMode === 'expr' && (typeof args.expression !== 'string' || !args.expression.trim())) {
        return { valid: false, error: 'js_calc in expr mode requires a non-empty "expression".' };
      }
      return {
        valid: true,
        normalized: {
          name,
          arguments: { ...args, mode: resolvedMode },
        },
      };
    }
  }
}

function escapeUnescapedNewlinesInStrings(jsonText: string): string {
  let out = '';
  let inString = false;
  let escaped = false;

  for (let i = 0; i < jsonText.length; i++) {
    const ch = jsonText[i];
    if (inString) {
      if (escaped) {
        out += ch;
        escaped = false;
      } else if (ch === '\\') {
        out += ch;
        escaped = true;
      } else if (ch === '"') {
        out += ch;
        inString = false;
      } else if (ch === '\n') {
        out += '\\n';
      } else if (ch === '\r') {
        out += '\\r';
      } else if (ch === '\t') {
        out += '\\t';
      } else {
        out += ch;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
    }
    out += ch;
  }

  return out;
}

function stripTrailingCommas(jsonText: string): string {
  let out = '';
  let inString = false;
  let escaped = false;

  for (let i = 0; i < jsonText.length; i++) {
    const ch = jsonText[i];
    if (inString) {
      out += ch;
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      out += ch;
      continue;
    }

    if (ch === ',') {
      let j = i + 1;
      while (j < jsonText.length && /\s/.test(jsonText[j])) {
        j++;
      }
      if (jsonText[j] === '}' || jsonText[j] === ']') {
        continue;
      }
    }

    out += ch;
  }

  return out;
}

function parseXmlFunctionCall(rawBody: string): { name?: unknown; arguments?: Record<string, unknown> } | null {
  const fnMatch = /<function=([^>\s]+)>([\s\S]*?)<\/function>/i.exec(rawBody);
  if (!fnMatch) {
    return null;
  }
  const name = fnMatch[1].trim();
  const body = fnMatch[2];
  const args: Record<string, unknown> = {};
  const paramRegex = /<parameter=([^>\s]+)>\s*([\s\S]*?)\s*<\/parameter>/gi;
  let pm: RegExpExecArray | null;
  while ((pm = paramRegex.exec(body)) !== null) {
    const key = pm[1].trim();
    const rawVal = pm[2].trim();
    if (
      (rawVal.startsWith('[') && rawVal.endsWith(']')) ||
      (rawVal.startsWith('{') && rawVal.endsWith('}')) ||
      rawVal === 'true' ||
      rawVal === 'false' ||
      /^-?\d+(?:\.\d+)?$/.test(rawVal)
    ) {
      try {
        args[key] = JSON.parse(rawVal);
        continue;
      } catch {
        // Keep string fallback
      }
    }
    args[key] = rawVal;
  }
  return { name, arguments: args };
}

function repairAndParseJson(rawBlock: string): unknown {
  let candidate = rawBlock.trim();

  // Strip fenced ```json ... ``` or ``` ... ``` wrappers
  const fenceMatch = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(candidate);
  if (fenceMatch) {
    candidate = fenceMatch[1].trim();
  }

  const xmlCall = parseXmlFunctionCall(candidate);
  if (xmlCall) {
    return xmlCall;
  }

  try {
    return JSON.parse(candidate);
  } catch {
    const repaired = stripTrailingCommas(escapeUnescapedNewlinesInStrings(candidate));
    return JSON.parse(repaired);
  }
}

export function parseToolCalls(text: string): {
  cleanText: string;
  toolCalls: ParsedToolCall[];
  parseErrors: string[];
} {
  const toolCalls: ParsedToolCall[] = [];
  const parseErrors: string[] = [];

  if (!text) {
    return { cleanText: '', toolCalls, parseErrors };
  }

  const tagRegex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi;
  let cleanParts: string[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tagRegex.exec(text)) !== null) {
    cleanParts.push(text.slice(lastIndex, match.index));
    lastIndex = match.index + match[0].length;

    const rawBody = match[1];
    try {
      const parsed = repairAndParseJson(rawBody) as {
        name?: unknown;
        arguments?: unknown;
        args?: unknown;
      };
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        parseErrors.push('Tool call payload must be a JSON object.');
        continue;
      }
      const rawArgs = (parsed.arguments ?? parsed.args ?? {}) as Record<string, unknown>;
      const validation = validateToolCall({
        name: typeof parsed.name === 'string' ? parsed.name : '',
        arguments: rawArgs,
      });
      if (!validation.valid || !validation.normalized) {
        parseErrors.push(validation.error ?? 'Invalid tool call.');
        continue;
      }
      toolCalls.push({
        ...validation.normalized,
        raw: match[0],
      });
    } catch (err: unknown) {
      parseErrors.push(err instanceof Error ? err.message : 'Failed to parse tool_call JSON.');
    }
  }

  cleanParts.push(text.slice(lastIndex));
  const cleanText = cleanParts.join('').replace(/\n{3,}/g, '\n\n').trim();

  return { cleanText, toolCalls, parseErrors };
}

export function detectPendingToolCall(streamedText: string): PendingToolDetection | null {
  if (!streamedText) {
    return null;
  }
  const lower = streamedText.toLowerCase();
  const lastOpen = lower.lastIndexOf('<tool_call>');
  if (lastOpen === -1) {
    return null;
  }
  const lastClose = lower.lastIndexOf('</tool_call>');
  if (lastClose > lastOpen) {
    return null;
  }

  const partial = streamedText.slice(lastOpen + '<tool_call>'.length);
  const nameMatch =
    /"name"\s*:\s*"([a-zA-Z0-9_]+)"/.exec(partial) ?? /<function=([a-zA-Z0-9_]+)>/.exec(partial);
  const toolName = nameMatch ? nameMatch[1] : null;
  const label =
    toolName && VALID_TOOL_NAMES.has(toolName as Tier1ToolName)
      ? TOOL_UI_LABELS[toolName as Tier1ToolName]
      : 'Running tool...';

  return {
    isOpen: true,
    toolName,
    label,
  };
}

export function formatToolResultMessage(toolName: string, result: ToolExecutionResult): string {
  const payload = result.formatted
    ? result.formatted
    : JSON.stringify(
        result.ok
          ? result.data ?? {}
          : {
              error: result.error ?? 'Tool execution failed.',
              ...(result.needsPermission ? { needsPermission: true } : {}),
              ...(result.blockedByCaptcha ? { blockedByCaptcha: true, manualUrl: result.manualUrl } : {}),
            },
      );
  return `<tool_result name="${toolName}" ok="${result.ok ? 'true' : 'false'}">\n${payload}\n</tool_result>`;
}

type GrammarFrame = 'object_start_or_key' | 'object_colon' | 'object_value' | 'object_comma_or_end' | 'array_start_or_val' | 'array_comma_or_end';

function analyzeJsonPrefix(partial: string): { valid: boolean; hint: string } {
  const trimmedStart = partial.trimStart();
  if (!trimmedStart) {
    return { valid: true, hint: '{' };
  }
  if (trimmedStart[0] !== '{') {
    return { valid: false, hint: '{' };
  }

  const stack: GrammarFrame[] = [];
  let i = 0;
  let rootClosed = false;

  while (i < trimmedStart.length) {
    // Skip whitespace
    while (i < trimmedStart.length && /\s/.test(trimmedStart[i])) {
      i++;
    }
    if (i >= trimmedStart.length) {
      break;
    }

    const ch = trimmedStart[i];
    const top = stack[stack.length - 1];

    if (stack.length === 0) {
      if (!rootClosed && ch === '{') {
        stack.push('object_start_or_key');
        i++;
        continue;
      }
      return { valid: false, hint: 'EOF' };
    }

    if (top === 'object_start_or_key') {
      if (ch === '}') {
        stack.pop();
        i++;
        if (stack.length === 0) {
          rootClosed = true;
        } else {
          const parent = stack[stack.length - 1];
          if (parent === 'object_value') stack[stack.length - 1] = 'object_comma_or_end';
          else if (parent === 'array_start_or_val') stack[stack.length - 1] = 'array_comma_or_end';
        }
        continue;
      }
      if (ch === '"') {
        i++;
        let closed = false;
        while (i < trimmedStart.length) {
          if (trimmedStart[i] === '\\') {
            i += 2;
            continue;
          }
          if (trimmedStart[i] === '"') {
            closed = true;
            i++;
            break;
          }
          i++;
        }
        if (!closed) {
          return { valid: true, hint: 'key_string_char_or_"' };
        }
        stack[stack.length - 1] = 'object_colon';
        continue;
      }
      return { valid: false, hint: '"key" or }' };
    }

    if (top === 'object_colon') {
      if (ch === ':') {
        stack[stack.length - 1] = 'object_value';
        i++;
        continue;
      }
      return { valid: false, hint: ':' };
    }

    if (top === 'object_comma_or_end') {
      if (ch === ',') {
        stack[stack.length - 1] = 'object_start_or_key';
        i++;
        continue;
      }
      if (ch === '}') {
        stack.pop();
        i++;
        if (stack.length === 0) {
          rootClosed = true;
        } else {
          const parent = stack[stack.length - 1];
          if (parent === 'object_value') stack[stack.length - 1] = 'object_comma_or_end';
          else if (parent === 'array_start_or_val') stack[stack.length - 1] = 'array_comma_or_end';
        }
        continue;
      }
      return { valid: false, hint: ', or }' };
    }

    if (top === 'array_comma_or_end') {
      if (ch === ',') {
        stack[stack.length - 1] = 'array_start_or_val';
        i++;
        continue;
      }
      if (ch === ']') {
        stack.pop();
        i++;
        if (stack.length > 0) {
          const parent = stack[stack.length - 1];
          if (parent === 'object_value') stack[stack.length - 1] = 'object_comma_or_end';
          else if (parent === 'array_start_or_val') stack[stack.length - 1] = 'array_comma_or_end';
        }
        continue;
      }
      return { valid: false, hint: ', or ]' };
    }

    if (top === 'object_value' || top === 'array_start_or_val') {
      if (top === 'array_start_or_val' && ch === ']') {
        stack.pop();
        i++;
        if (stack.length > 0) {
          const parent = stack[stack.length - 1];
          if (parent === 'object_value') stack[stack.length - 1] = 'object_comma_or_end';
          else if (parent === 'array_start_or_val') stack[stack.length - 1] = 'array_comma_or_end';
        }
        continue;
      }

      const advanceValueDone = () => {
        if (top === 'object_value') stack[stack.length - 1] = 'object_comma_or_end';
        else stack[stack.length - 1] = 'array_comma_or_end';
      };

      if (ch === '{') {
        stack.push('object_start_or_key');
        i++;
        continue;
      }
      if (ch === '[') {
        stack.push('array_start_or_val');
        i++;
        continue;
      }
      if (ch === '"') {
        i++;
        let closed = false;
        while (i < trimmedStart.length) {
          if (trimmedStart[i] === '\\') {
            i += 2;
            continue;
          }
          if (trimmedStart[i] === '"') {
            closed = true;
            i++;
            break;
          }
          i++;
        }
        if (!closed) {
          return { valid: true, hint: 'string_char_or_"' };
        }
        advanceValueDone();
        continue;
      }
      if (ch === '-' || (ch >= '0' && ch <= '9')) {
        const numMatch = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(trimmedStart.slice(i));
        const partialNumMatch = /^-?(?:\d+\.?|\d*\.\d+(?:[eE][+-]?)?)?/.exec(trimmedStart.slice(i));
        if (i + (numMatch?.[0].length ?? 0) === trimmedStart.length && partialNumMatch) {
          return { valid: true, hint: 'number_digit_or_delimiter' };
        }
        if (!numMatch) {
          return { valid: false, hint: 'number' };
        }
        i += numMatch[0].length;
        advanceValueDone();
        continue;
      }
      if (ch === 't' || ch === 'f' || ch === 'n') {
        const kw = ch === 't' ? 'true' : ch === 'f' ? 'false' : 'null';
        const rem = trimmedStart.slice(i);
        if (kw.startsWith(rem)) {
          return { valid: true, hint: kw };
        }
        if (rem.startsWith(kw)) {
          i += kw.length;
          advanceValueDone();
          continue;
        }
        return { valid: false, hint: kw };
      }

      return { valid: false, hint: 'json_value' };
    }
  }

  if (stack.length === 0) {
    return { valid: true, hint: 'EOF' };
  }
  const top = stack[stack.length - 1];
  switch (top) {
    case 'object_start_or_key':
      return { valid: true, hint: '"key" or }' };
    case 'object_colon':
      return { valid: true, hint: ':' };
    case 'object_value':
      return { valid: true, hint: 'json_value' };
    case 'object_comma_or_end':
      return { valid: true, hint: ', or }' };
    case 'array_start_or_val':
      return { valid: true, hint: 'json_value or ]' };
    case 'array_comma_or_end':
      return { valid: true, hint: ', or ]' };
  }
}

export const ToolCallJsonGrammar = {
  isValidToolJsonPrefix(partial: string): boolean {
    return analyzeJsonPrefix(partial).valid;
  },
  allowedNextTokenHint(partial: string): string {
    return analyzeJsonPrefix(partial).hint;
  },
};
