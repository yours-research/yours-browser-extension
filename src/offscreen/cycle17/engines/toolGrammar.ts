/**
 * Grammar-aware drafting for browser-assistant structure:
 *   - Qwen3.5 XML tool calls  <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
 *   - Extension JSON tool calls <tool_call>{"name": "...", "arguments": {...}}</tool_call>
 *   - <think> scaffolding (`<think>\n`, `</think>\n\n`) and turn ends.
 *
 * Given the decoded text so far, returns a small set of *forced or nearly
 * forced* continuation strings with prior probabilities. Names, parameter
 * keys, enum values and booleans are enumerated from the tool schemas that
 * are actually in the prompt (or a built-in registry), so the drafter is
 * robust to the three schema dialects in the ecosystem (lab snake_case v1.0,
 * extension camelCase tier-1, yours-tools canonical v1.1).
 */

export type ParamType = 'string' | 'integer' | 'number' | 'boolean' | 'array' | 'object';
export interface ToolParam {
  name: string;
  type: ParamType;
  enum?: string[];
  required: boolean;
}
export interface ToolSchema {
  name: string;
  params: ToolParam[];
}
export interface GrammarCandidate {
  text: string;
  prob: number;
  /** True when the continuation ends on a hard token boundary (newline / tag). */
  closed: boolean;
}

const p = (name: string, type: ParamType, required = false, en?: string[]): ToolParam => ({ name, type, required, enum: en });

/** Pragma training schema (yours-research-lab lab/TOOLS_SCHEMA.json v1.0, snake_case). */
export const LAB_TOOLS: ToolSchema[] = [
  { name: 'read_active_tab', params: [p('include_links', 'boolean'), p('max_chars', 'integer')] },
  { name: 'list_tabs', params: [p('window_scope', 'string', false, ['current', 'all'])] },
  { name: 'read_tab', params: [p('tab_id', 'integer', true), p('max_chars', 'integer')] },
  { name: 'search_web_tab', params: [p('query', 'string', true), p('max_results', 'integer')] },
  { name: 'fetch_background_tab', params: [p('url', 'string', true), p('max_chars', 'integer')] },
];

/** Extension runtime tier-1 schema (tinygpt-browser src/tools/toolProtocol.ts, camelCase, legacy names). */
export const EXTENSION_TOOLS: ToolSchema[] = [
  { name: 'read_active_page', params: [p('query', 'string'), p('mode', 'string', false, ['auto', 'article', 'selection', 'interactive']), p('maxChars', 'integer')] },
  { name: 'web_search', params: [p('query', 'string', true), p('limit', 'integer'), p('readTopN', 'integer')] },
  { name: 'fetch_and_read_url', params: [p('urls', 'array', true), p('query', 'string'), p('maxCharsPerSource', 'integer')] },
  { name: 'list_open_tabs', params: [] },
  { name: 'read_tab', params: [p('tabId', 'integer', true), p('query', 'string'), p('mode', 'string', false, ['auto', 'article', 'selection', 'interactive']), p('maxChars', 'integer')] },
  {
    name: 'js_calc',
    params: [
      p('mode', 'string', false, ['expr', 'unit', 'date', 'json', 'csv']),
      p('expression', 'string'), p('value', 'string'), p('from', 'string'), p('to', 'string'),
      p('dateA', 'string'), p('dateB', 'string'), p('days', 'number'), p('data', 'string'), p('path', 'string'),
    ],
  },
];

/** Parse `<tools>{...}\n{...}</tools>` JSON-lines blocks (Qwen chat template) from a prompt. */
export function parseToolsFromPrompt(prompt: string): ToolSchema[] {
  const out: ToolSchema[] = [];
  const m = /<tools>([\s\S]*?)<\/tools>/.exec(prompt);
  if (!m) return parseToolList(prompt);
  for (const line of m[1].split('\n')) {
    const s = line.trim();
    if (!s.startsWith('{')) continue;
    try {
      const j = JSON.parse(s) as { function?: { name: string; parameters?: { properties?: Record<string, { type?: string; enum?: string[] }>; required?: string[] } }; name?: string };
      const fn = j.function ?? (j as unknown as { name: string; parameters?: { properties?: Record<string, { type?: string; enum?: string[] }>; required?: string[] } });
      const req = new Set(fn.parameters?.required ?? []);
      const params = Object.entries(fn.parameters?.properties ?? {}).map(([k, v]) => p(k, (v.type as ParamType) ?? 'string', req.has(k), v.enum));
      // required first, then declaration order (how models usually serialise)
      params.sort((a, b) => Number(b.required) - Number(a.required));
      out.push({ name: fn.name, params });
    } catch {
      /* ignore malformed lines */
    }
  }
  return out;
}

/**
 * tinygpt-browser system-prompt tool list (one tool per line):
 *   `- read_tab(include_links?: boolean, mode?: "auto" | "article", tab_id: integer): desc`
 * `?` marks optional parameters; a `"a" | "b"` type is a string enum; `unknown`
 * maps to `object` (free-form, no grammar). Required parameters are moved first
 * (stable), as for the <tools> block.
 */
export function parseToolList(prompt: string): ToolSchema[] {
  const out: ToolSchema[] = [];
  const known: ParamType[] = ['string', 'integer', 'number', 'boolean', 'array', 'object'];
  for (const line of prompt.split('\n')) {
    const m = /^- ([A-Za-z0-9_]+)\(([^)]*)\):/.exec(line);
    if (!m) continue;
    const params: ToolParam[] = [];
    for (const raw of m[2].split(', ')) {
      const a = /^([A-Za-z0-9_]+)(\??): (.+)$/.exec(raw.trim());
      if (!a) continue;
      const ty = a[3].trim();
      const required = a[2] !== '?';
      if (ty.startsWith('"')) {
        const en = ty.split('|').map((x) => x.trim().replace(/^"|"$/g, ''));
        params.push(p(a[1], 'string', required, en));
      } else {
        params.push(p(a[1], known.includes(ty as ParamType) ? (ty as ParamType) : 'object', required));
      }
    }
    params.sort((x, y) => Number(y.required) - Number(x.required));
    out.push({ name: m[1], params });
  }
  return out;
}

export type CallDialect = 'xml' | 'json';

/** JSON separator style: `{"a": 1, "b": 2}` (spaced) vs `{"a":1,"b":2}` (compact). */
export interface JsonStyle {
  colon: ':' | ': ';
  comma: ',' | ', ';
}
export const SPACED_JSON: JsonStyle = { colon: ': ', comma: ', ' };

/** Style instructed by the prompt's `<tool_call>{"name"...` exemplar (default spaced). */
export function detectJsonStyle(prompt: string): JsonStyle {
  const m = /<tool_call>\s*\{"name"(\s*):(\s*)"[^"]*"(\s*),(\s*)"arguments"/.exec(prompt);
  if (!m) return SPACED_JSON;
  return { colon: m[2] ? ': ' : ':', comma: m[4] ? ', ' : ',' };
}

/**
 * Local style of a call in progress: the most recent separator that is already
 * followed by a character wins (models mix styles, e.g. compact header and
 * spaced arguments); otherwise the prompt-level fallback.
 */
export function localJsonStyle(tail: string, fallback: JsonStyle): JsonStyle {
  let colon = fallback.colon;
  let comma = fallback.comma;
  for (let i = tail.length - 2; i >= 1; i--) {
    if (tail[i] === ':' && tail[i - 1] === '"') {
      colon = tail[i + 1] === ' ' ? ': ' : ':';
      break;
    }
  }
  for (let i = tail.length - 2; i >= 0; i--) {
    if (tail[i] === ',' && (tail[i + 1] === ' ' || tail[i + 1] === '"')) {
      comma = tail[i + 1] === ' ' ? ', ' : ',';
      break;
    }
  }
  return { colon, comma };
}

/** Infer which tool-call dialect the prompt instructs. */
export function detectDialect(prompt: string): CallDialect {
  return /<tool_call>\s*\{\s*"name"/.test(prompt) ? 'json' : 'xml';
}

const FORCED = 0.96;

function split(names: string[], prefix: string, suffix: (n: string) => string): GrammarCandidate[] {
  const hits = names.filter((n) => n.startsWith(prefix) && n.length > prefix.length);
  const exact = names.includes(prefix);
  if (!hits.length && !exact) return [];
  const all = exact ? [prefix, ...hits] : hits;
  return all.map((n) => ({ text: n.slice(prefix.length) + suffix(n), prob: FORCED / all.length, closed: true }));
}

/**
 * Forced / enumerable continuations of `text` (assistant output so far,
 * including the prompt tail). Returns [] when generation is free-form.
 */
export function grammarContinuations(
  text: string,
  tools: ToolSchema[],
  dialect: CallDialect,
  thinking = true,
  style: JsonStyle = SPACED_JSON,
): GrammarCandidate[] {
  // --- think / turn scaffolding -------------------------------------------------
  if (text.endsWith('<|im_start|>assistant\n')) return [{ text: thinking ? '<think>\n' : '<think>\n\n</think>\n\n', prob: FORCED, closed: true }];
  if (text.endsWith('</think>')) return [{ text: '\n\n', prob: FORCED, closed: true }];
  if (text.endsWith('</tool_call>')) return [{ text: '<|im_end|>', prob: 0.9, closed: true }];

  const open = text.lastIndexOf('<tool_call>');
  if (open < 0 || text.lastIndexOf('</tool_call>') > open) return [];
  const tail = text.slice(open + '<tool_call>'.length);
  const names = tools.map((t) => t.name);

  if (tail === '') {
    return dialect === 'json' ? [{ text: `{"name"${style.colon}"`, prob: FORCED, closed: false }] : [{ text: '\n<function=', prob: FORCED, closed: false }];
  }
  if (dialect === 'xml' || /^\s*<function=/.test(tail)) return xmlRules(tail, tools, names);
  return jsonRules(tail, tools, names, localJsonStyle(tail, style));
}

function xmlRules(tail: string, tools: ToolSchema[], names: string[]): GrammarCandidate[] {
  if (tail === '\n') return [{ text: '<function=', prob: FORCED, closed: false }];
  const fm = /^\s*<function=([A-Za-z0-9_]*)$/.exec(tail);
  if (fm) {
    return split(names, fm[1], (n) => {
      const t = tools.find((x) => x.name === n)!;
      return t.params.length ? '>\n<parameter=' : '>\n</function>\n</tool_call>';
    });
  }
  const fn = /^\s*<function=([A-Za-z0-9_]+)>/.exec(tail);
  if (!fn) return [];
  const tool = tools.find((t) => t.name === fn[1]);
  if (!tool) return [];
  const used = new Set([...tail.matchAll(/<parameter=([A-Za-z0-9_]+)>/g)].map((m) => m[1]));

  if (/<\/function>$/.test(tail)) return [{ text: '\n</tool_call>', prob: FORCED, closed: true }];
  const pm = /<parameter=([A-Za-z0-9_]*)$/.exec(tail);
  if (pm) {
    const free = tool.params.filter((x) => !used.has(x.name) || x.name === pm[1]).map((x) => x.name);
    return split(free, pm[1], () => '>\n');
  }
  if (/<\/parameter>\n$/.test(tail) || /<function=[A-Za-z0-9_]+>\n$/.test(tail)) {
    const remainingReq = tool.params.filter((x) => x.required && !used.has(x.name));
    if (remainingReq.length) return [{ text: `<parameter=${remainingReq[0].name}>\n`, prob: 0.9, closed: true }];
    const remaining = tool.params.filter((x) => !used.has(x.name));
    const outs: GrammarCandidate[] = [{ text: '</function>\n</tool_call>', prob: 0.6, closed: true }];
    if (remaining.length) outs.push({ text: '<parameter=', prob: 0.35, closed: false });
    return outs;
  }
  // Inside a parameter value: <parameter=K>\n VALUE
  const vm = /<parameter=([A-Za-z0-9_]+)>\n([^\n]*)$/.exec(tail);
  if (vm) {
    const param = tool.params.find((x) => x.name === vm[1]);
    const v = vm[2];
    if (param?.enum) return split(param.enum, v, () => '\n</parameter>\n');
    if (param?.type === 'boolean') return split(['true', 'false'], v, () => '\n</parameter>\n');
    return [];
  }
  // Value finished with newline -> close parameter.
  if (/<parameter=[A-Za-z0-9_]+>\n[^\n]+\n$/.test(tail)) return [{ text: '</parameter>\n', prob: 0.9, closed: true }];
  return [];
}

type JsonCtx =
  | { kind: 'name-value'; partial: string }
  | { kind: 'after-name' }
  | { kind: 'key'; partial: string; used: string[]; quote: boolean }
  | { kind: 'after-key'; key: string; space: boolean }
  | { kind: 'string-value'; key: string; partial: string }
  | { kind: 'after-value'; used: string[] }
  | { kind: 'unknown' };

/** Tiny incremental scanner for `{"name": "...", "arguments": {...}}` prefixes. */
function scanJson(tail: string): { ctx: JsonCtx; fn?: string } {
  const nm = /^\s*\{\s*"name"\s*:\s*"([A-Za-z0-9_]*)$/.exec(tail);
  if (nm) return { ctx: { kind: 'name-value', partial: nm[1] } };
  const full = /^\s*\{\s*"name"\s*:\s*"([A-Za-z0-9_]+)"/.exec(tail);
  if (!full) return { ctx: { kind: 'unknown' } };
  const fn = full[1];
  const rest = tail.slice(full[0].length);
  if (rest === '') return { ctx: { kind: 'after-name' }, fn };
  const am = /^\s*,\s*"arguments"\s*:\s*\{/.exec(rest);
  if (!am) return { ctx: { kind: 'unknown' }, fn };
  const body = rest.slice(am[0].length);
  // Walk the flat argument object (string / number / bool / simple arrays).
  const used: string[] = [];
  let i = 0;
  let state: 'expect-key' | 'in-key' | 'expect-colon' | 'expect-value' | 'in-string' | 'in-scalar' | 'in-array' | 'after-value' = 'expect-key';
  let key = '';
  let buf = '';
  while (i < body.length) {
    const c = body[i];
    switch (state) {
      case 'expect-key':
        if (c === '"') {
          state = 'in-key';
          buf = '';
        } else if (c === '}') return { ctx: { kind: 'unknown' }, fn };
        break;
      case 'in-key':
        if (c === '"') {
          key = buf;
          used.push(key);
          state = 'expect-colon';
        } else buf += c;
        break;
      case 'expect-colon':
        if (c === ':') state = 'expect-value';
        break;
      case 'expect-value':
        if (c === '"') {
          state = 'in-string';
          buf = '';
        } else if (c === '[') state = 'in-array';
        else if (!/\s/.test(c)) {
          state = 'in-scalar';
          buf = c;
        }
        break;
      case 'in-string':
        if (c === '\\') i++;
        else if (c === '"') state = 'after-value';
        else buf += c;
        break;
      case 'in-scalar':
        if (c === ',') state = 'expect-key';
        else if (c === '}') return { ctx: { kind: 'unknown' }, fn };
        else buf += c;
        break;
      case 'in-array':
        if (c === ']') state = 'after-value';
        break;
      case 'after-value':
        if (c === ',') state = 'expect-key';
        else if (c === '}') return { ctx: { kind: 'unknown' }, fn };
        break;
    }
    i++;
  }
  switch (state) {
    case 'expect-key':
      return { ctx: body.trimEnd().endsWith(',') || body.trim() === '' ? { kind: 'key', partial: '', used, quote: true } : { kind: 'unknown' }, fn };
    case 'in-key':
      return { ctx: { kind: 'key', partial: buf, used: used, quote: false }, fn };
    case 'expect-colon':
    case 'expect-value':
      return { ctx: state === 'expect-value' && /:\s?$/.test(body) ? { kind: 'after-key', key, space: body.endsWith(' ') } : { kind: 'unknown' }, fn };
    case 'in-string':
      return { ctx: { kind: 'string-value', key, partial: buf }, fn };
    case 'after-value':
      return { ctx: { kind: 'after-value', used }, fn };
    default:
      return { ctx: { kind: 'unknown' }, fn };
  }
}

function jsonRules(tail: string, tools: ToolSchema[], names: string[], st: JsonStyle): GrammarCandidate[] {
  const { ctx, fn } = scanJson(tail);
  const tool = fn ? tools.find((t) => t.name === fn) : undefined;
  switch (ctx.kind) {
    case 'name-value':
      return split(names, ctx.partial, (n) => {
        const t = tools.find((x) => x.name === n)!;
        return t.params.length ? `"${st.comma}"arguments"${st.colon}{"` : `"${st.comma}"arguments"${st.colon}{}}</tool_call>`;
      });
    case 'after-name':
      return [{ text: `${st.comma}"arguments"${st.colon}{"`, prob: FORCED, closed: false }];
    case 'key': {
      if (!tool) return [];
      const free = tool.params.filter((x) => !ctx.used.includes(x.name) || x.name === ctx.partial).map((x) => x.name);
      const close = `"${st.colon}`;
      if (ctx.quote) return split(free, '', () => close).map((c) => ({ ...c, text: '"' + c.text }));
      return split(free, ctx.partial, () => close);
    }
    case 'after-key': {
      const prm = tool?.params.find((x) => x.name === ctx.key);
      if (!prm) return [];
      // Colon already typed but the style wants a space before the value.
      const pre = !ctx.space && st.colon === ': ' ? ' ' : '';
      if (prm.enum) return prm.enum.map((e) => ({ text: `${pre}"${e}"`, prob: FORCED / prm.enum!.length, closed: false }));
      if (prm.type === 'boolean') return [{ text: pre + 'false', prob: 0.5, closed: false }, { text: pre + 'true', prob: 0.45, closed: false }];
      if (prm.type === 'string') return [{ text: pre + '"', prob: FORCED, closed: false }];
      if (prm.type === 'array') return [{ text: pre + '["', prob: FORCED, closed: false }];
      return [];
    }
    case 'string-value': {
      const prm = tool?.params.find((x) => x.name === ctx.key);
      if (prm?.enum) return split(prm.enum, ctx.partial, () => '"');
      return [];
    }
    case 'after-value': {
      if (!tool) return [];
      const remainingReq = tool.params.filter((x) => x.required && !ctx.used.includes(x.name));
      if (remainingReq.length) return [{ text: `${st.comma}"${remainingReq[0].name}"${st.colon}`, prob: 0.9, closed: false }];
      return [
        { text: '}}</tool_call>', prob: 0.6, closed: true },
        { text: `${st.comma}"`, prob: 0.3, closed: false },
      ];
    }
    default:
      return [];
  }
}
