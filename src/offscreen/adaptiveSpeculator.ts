/**
 * Phase A1: Cost-aware adaptive multi-source Engine A drafter for TinyGPT Browser.
 *
 * Combines:
 *  - Online Suffix Automaton (SAM) matchers over live context, tool/template corpus,
 *    and cross-session assistant history
 *  - Copy anchor tracking edited spans within grounding/context tokens
 *  - Grammar continuations for Qwen3.5 XML & JSON tool calls and <think> tags
 *  - Online Beta-Bernoulli calibrator + adaptive K(t) budget controller using
 *    measured WebGPU verify latency T(M) (returns K = 0 when speculation cannot pay).
 */

export interface EngineATokenizer {
  encode(text: string): number[];
  decode(tokens: readonly number[]): string;
}

export interface VerifyCostProfile {
  readonly name: string;
  readonly maxM: number;
  readonly rejectPenaltyMs: number;
  stepMs(M: number): number;
}

/**
 * Live EMA-calibrated WebGPU verify cost profile (initialized to Phase E0
 * small-batch curve and refined online from real decode / verify timings).
 */
export class AdaptiveVerifyCostModel implements VerifyCostProfile {
  readonly name = 'webgpu-adaptive-e0';
  readonly maxM: number;
  t1Ms: number;
  slopePerTokenMs: number;
  rejectPenaltyMs: number;

  constructor(options?: {
    maxM?: number;
    initialT1Ms?: number;
    initialSlopeMs?: number;
    initialRejectPenaltyMs?: number;
  }) {
    this.maxM = options?.maxM ?? 8;
    this.t1Ms = options?.initialT1Ms ?? 10.9;
    // Calibrated to E0 small-batch verify slope (~0.38 * T1 per extra token)
    this.slopePerTokenMs = options?.initialSlopeMs ?? this.t1Ms * 0.38;
    this.rejectPenaltyMs = options?.initialRejectPenaltyMs ?? 0.9;
  }

  stepMs(M: number): number {
    if (M <= 1) return this.t1Ms;
    return this.t1Ms + this.slopePerTokenMs * (M - 1);
  }

  recordStepTiming(M: number, wallMs: number, hadRejection: boolean): void {
    if (!Number.isFinite(wallMs) || wallMs <= 0) return;
    if (M <= 1) {
      this.t1Ms = 0.85 * this.t1Ms + 0.15 * wallMs;
      return;
    }
    const extra = Math.max(0.25, wallMs - this.t1Ms - (hadRejection ? this.rejectPenaltyMs * 0.5 : 0));
    const perTokSlope = Math.max(0.15 * this.t1Ms, Math.min(1.5 * this.t1Ms, extra / (M - 1)));
    this.slopePerTokenMs = 0.85 * this.slopePerTokenMs + 0.15 * perTokSlope;
  }
}

export class SuffixAutomaton {
  len: Int32Array;
  link: Int32Array;
  cnt: Int32Array;
  lastEnd: Int32Array;
  prevEnd: Int32Array;
  next: Map<number, number>[] = [];
  size = 1;
  last = 0;
  readonly text: number[] = [];

  constructor(
    initialCapacity = 1024,
    readonly statHops = 64,
  ) {
    const cap = Math.max(2, initialCapacity * 2);
    this.len = new Int32Array(cap);
    this.link = new Int32Array(cap);
    this.cnt = new Int32Array(cap);
    this.lastEnd = new Int32Array(cap);
    this.prevEnd = new Int32Array(cap);
    this.link[0] = -1;
    this.lastEnd[0] = -1;
    this.prevEnd[0] = -1;
    this.next.push(new Map());
  }

  get length(): number {
    return this.text.length;
  }

  private grow(): void {
    const cap = this.len.length * 2;
    const growArr = (a: Int32Array): Int32Array => {
      const b = new Int32Array(cap);
      b.set(a);
      return b;
    };
    this.len = growArr(this.len);
    this.link = growArr(this.link);
    this.cnt = growArr(this.cnt);
    this.lastEnd = growArr(this.lastEnd);
    this.prevEnd = growArr(this.prevEnd);
  }

  private newState(len: number): number {
    if (this.size >= this.len.length) this.grow();
    const s = this.size++;
    this.len[s] = len;
    this.link[s] = -1;
    this.cnt[s] = 0;
    this.lastEnd[s] = -1;
    this.prevEnd[s] = -1;
    this.next.push(new Map());
    return s;
  }

  extend(c: number): void {
    const i = this.text.length;
    this.text.push(c);
    const cur = this.newState(this.len[this.last] + 1);
    this.cnt[cur] = 1;
    this.lastEnd[cur] = i;
    let p = this.last;
    while (p !== -1 && !this.next[p]!.has(c)) {
      this.next[p]!.set(c, cur);
      p = this.link[p]!;
    }
    if (p === -1) {
      this.link[cur] = 0;
    } else {
      const q = this.next[p]!.get(c)!;
      if (this.len[p]! + 1 === this.len[q]!) {
        this.link[cur] = q;
      } else {
        const clone = this.newState(this.len[p]! + 1);
        this.next[clone] = new Map(this.next[q]!);
        this.link[clone] = this.link[q]!;
        this.cnt[clone] = this.cnt[q]!;
        this.lastEnd[clone] = this.lastEnd[q]!;
        this.prevEnd[clone] = this.prevEnd[q]!;
        while (p !== -1 && this.next[p]!.get(c) === q) {
          this.next[p]!.set(c, clone);
          p = this.link[p]!;
        }
        this.link[q] = clone;
        this.link[cur] = clone;
      }
    }
    this.last = cur;
    let s = this.link[cur]!;
    for (let h = 0; s > 0 && h < this.statHops; h++, s = this.link[s]!) {
      this.cnt[s]!++;
      if (this.lastEnd[s] !== i) {
        this.prevEnd[s] = this.lastEnd[s]!;
        this.lastEnd[s] = i;
      }
    }
  }

  extendAll(tokens: Iterable<number>): void {
    for (const t of tokens) this.extend(t);
  }

  longestRepeatedSuffix(minLen = 1): { state: number; len: number } | null {
    let s = this.link[this.last]!;
    while (s > 0) {
      if (this.len[s]! < minLen) return null;
      if (this.cnt[s]! >= 2 && this.prevEnd[s]! >= 0) return { state: s, len: this.len[s]! };
      s = this.link[s]!;
    }
    return null;
  }

  continuations(state: number, limit: number): [number, number, number][] {
    const out: [number, number, number][] = [];
    for (const [tok, t] of this.next[state]!) out.push([tok, t, this.cnt[t]!]);
    out.sort((a, b) => b[2] - a[2] || this.lastEnd[b[1]]! - this.lastEnd[a[1]]! || a[0] - b[0]);
    return out.length > limit ? out.slice(0, limit) : out;
  }

  totalContinuationCount(state: number): number {
    let z = 0;
    for (const t of this.next[state]!.values()) z += this.cnt[t]!;
    return z;
  }
}

export class SamMatcher {
  state = 0;
  len = 0;
  constructor(readonly sam: SuffixAutomaton) {}

  reset(): void {
    this.state = 0;
    this.len = 0;
  }

  push(c: number): void {
    const sam = this.sam;
    while (this.state !== 0 && !sam.next[this.state]!.has(c)) {
      this.state = sam.link[this.state]!;
      this.len = sam.len[this.state]!;
    }
    const t = sam.next[this.state]!.get(c);
    if (t !== undefined) {
      this.state = t;
      this.len++;
    } else {
      this.state = 0;
      this.len = 0;
    }
  }
}

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
  closed: boolean;
}

const p = (name: string, type: ParamType, required = false, en?: string[]): ToolParam => ({
  name,
  type,
  required,
  enum: en,
});

export const LAB_TOOLS: ToolSchema[] = [
  { name: 'read_active_tab', params: [p('include_links', 'boolean'), p('max_chars', 'integer')] },
  { name: 'list_tabs', params: [p('window_scope', 'string', false, ['current', 'all'])] },
  { name: 'read_tab', params: [p('tab_id', 'integer', true), p('max_chars', 'integer')] },
  { name: 'search_web_tab', params: [p('query', 'string', true), p('max_results', 'integer')] },
  { name: 'fetch_background_tab', params: [p('url', 'string', true), p('max_chars', 'integer')] },
];

export const EXTENSION_TOOLS: ToolSchema[] = [
  {
    name: 'read_active_page',
    params: [
      p('query', 'string'),
      p('mode', 'string', false, ['auto', 'article', 'selection', 'interactive']),
      p('maxChars', 'integer'),
    ],
  },
  { name: 'web_search', params: [p('query', 'string', true), p('limit', 'integer'), p('readTopN', 'integer')] },
  { name: 'fetch_and_read_url', params: [p('urls', 'array', true), p('query', 'string'), p('maxCharsPerSource', 'integer')] },
  { name: 'list_open_tabs', params: [] },
  {
    name: 'read_tab',
    params: [
      p('tabId', 'integer', true),
      p('query', 'string'),
      p('mode', 'string', false, ['auto', 'article', 'selection', 'interactive']),
      p('maxChars', 'integer'),
    ],
  },
  {
    name: 'js_calc',
    params: [
      p('mode', 'string', false, ['expr', 'unit', 'date', 'json', 'csv']),
      p('expression', 'string'),
      p('value', 'string'),
      p('from', 'string'),
      p('to', 'string'),
      p('dateA', 'string'),
      p('dateB', 'string'),
      p('days', 'number'),
      p('data', 'string'),
      p('path', 'string'),
    ],
  },
];

export function parseToolsFromPrompt(prompt: string): ToolSchema[] {
  const out: ToolSchema[] = [];
  const m = /<tools>([\s\S]*?)<\/tools>/.exec(prompt);
  if (!m) return out;
  for (const line of m[1]!.split('\n')) {
    const s = line.trim();
    if (!s.startsWith('{')) continue;
    try {
      const j = JSON.parse(s) as {
        function?: {
          name: string;
          parameters?: { properties?: Record<string, { type?: string; enum?: string[] }>; required?: string[] };
        };
        name?: string;
      };
      const fn =
        j.function ??
        (j as unknown as {
          name: string;
          parameters?: { properties?: Record<string, { type?: string; enum?: string[] }>; required?: string[] };
        });
      const req = new Set(fn.parameters?.required ?? []);
      const params = Object.entries(fn.parameters?.properties ?? {}).map(([k, v]) =>
        p(k, (v.type as ParamType) ?? 'string', req.has(k), v.enum),
      );
      params.sort((a, b) => Number(b.required) - Number(a.required));
      out.push({ name: fn.name, params });
    } catch {
      // ignore malformed JSON lines
    }
  }
  return out;
}

export type CallDialect = 'xml' | 'json';

export function detectDialect(prompt: string): CallDialect {
  return /<tool_call>\s*\{\s*"name"/.test(prompt) ? 'json' : 'xml';
}

const FORCED_PROB = 0.96;

function splitPrefix(names: string[], prefix: string, suffix: (n: string) => string): GrammarCandidate[] {
  const hits = names.filter((n) => n.startsWith(prefix) && n.length > prefix.length);
  const exact = names.includes(prefix);
  if (!hits.length && !exact) return [];
  const all = exact ? [prefix, ...hits] : hits;
  return all.map((n) => ({ text: n.slice(prefix.length) + suffix(n), prob: FORCED_PROB / all.length, closed: true }));
}

function xmlRules(tail: string, tools: ToolSchema[], names: string[]): GrammarCandidate[] {
  if (tail === '\n') return [{ text: '<function=', prob: FORCED_PROB, closed: false }];
  const fm = /^\s*<function=([A-Za-z0-9_]*)$/.exec(tail);
  if (fm) {
    return splitPrefix(names, fm[1]!, (n) => {
      const t = tools.find((x) => x.name === n)!;
      return t.params.length ? '>\n<parameter=' : '>\n</function>\n</tool_call>';
    });
  }
  const fn = /^\s*<function=([A-Za-z0-9_]+)>/.exec(tail);
  if (!fn) return [];
  const tool = tools.find((t) => t.name === fn[1]);
  if (!tool) return [];
  const used = new Set([...tail.matchAll(/<parameter=([A-Za-z0-9_]+)>/g)].map((m) => m[1]!));

  if (/<\/function>$/.test(tail)) return [{ text: '\n</tool_call>', prob: FORCED_PROB, closed: true }];
  const pm = /<parameter=([A-Za-z0-9_]*)$/.exec(tail);
  if (pm) {
    const free = tool.params.filter((x) => !used.has(x.name) || x.name === pm[1]).map((x) => x.name);
    return splitPrefix(free, pm[1]!, () => '>\n');
  }
  if (/<\/parameter>\n$/.test(tail) || /<function=[A-Za-z0-9_]+>\n$/.test(tail)) {
    const remainingReq = tool.params.filter((x) => x.required && !used.has(x.name));
    if (remainingReq.length) return [{ text: `<parameter=${remainingReq[0]!.name}>\n`, prob: 0.9, closed: true }];
    const remaining = tool.params.filter((x) => !used.has(x.name));
    const outs: GrammarCandidate[] = [{ text: '</function>\n</tool_call>', prob: 0.6, closed: true }];
    if (remaining.length) outs.push({ text: '<parameter=', prob: 0.35, closed: false });
    return outs;
  }
  const vm = /<parameter=([A-Za-z0-9_]+)>\n([^\n]*)$/.exec(tail);
  if (vm) {
    const param = tool.params.find((x) => x.name === vm[1]);
    const v = vm[2]!;
    if (param?.enum) return splitPrefix(param.enum, v, () => '\n</parameter>\n');
    if (param?.type === 'boolean') return splitPrefix(['true', 'false'], v, () => '\n</parameter>\n');
    return [];
  }
  if (/<parameter=[A-Za-z0-9_]+>\n[^\n]+\n$/.test(tail)) return [{ text: '</parameter>\n', prob: 0.9, closed: true }];
  return [];
}

type JsonCtx =
  | { kind: 'name-value'; partial: string }
  | { kind: 'after-name' }
  | { kind: 'key'; partial: string; used: string[] }
  | { kind: 'after-key'; key: string }
  | { kind: 'string-value'; key: string; partial: string }
  | { kind: 'after-value'; used: string[] }
  | { kind: 'unknown' };

function scanJson(tail: string): { ctx: JsonCtx; fn?: string } {
  const nm = /^\s*\{\s*"name"\s*:\s*"([A-Za-z0-9_]*)$/.exec(tail);
  if (nm) return { ctx: { kind: 'name-value', partial: nm[1]! } };
  const full = /^\s*\{\s*"name"\s*:\s*"([A-Za-z0-9_]+)"/.exec(tail);
  if (!full) return { ctx: { kind: 'unknown' } };
  const fn = full[1]!;
  const rest = tail.slice(full[0].length);
  if (rest === '') return { ctx: { kind: 'after-name' }, fn };
  const am = /^\s*,\s*"arguments"\s*:\s*\{/.exec(rest);
  if (!am) return { ctx: { kind: 'unknown' }, fn };
  const body = rest.slice(am[0].length);
  const used: string[] = [];
  let i = 0;
  let state:
    | 'expect-key'
    | 'in-key'
    | 'expect-colon'
    | 'expect-value'
    | 'in-string'
    | 'in-scalar'
    | 'in-array'
    | 'after-value' = 'expect-key';
  let key = '';
  let buf = '';
  while (i < body.length) {
    const c = body[i]!;
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
      return {
        ctx: body.trimEnd().endsWith(',') || body.trim() === '' ? { kind: 'key', partial: '', used } : { kind: 'unknown' },
        fn,
      };
    case 'in-key':
      return { ctx: { kind: 'key', partial: buf, used }, fn };
    case 'expect-colon':
    case 'expect-value':
      return { ctx: state === 'expect-value' && /:\s$/.test(body) ? { kind: 'after-key', key } : { kind: 'unknown' }, fn };
    case 'in-string':
      return { ctx: { kind: 'string-value', key, partial: buf }, fn };
    case 'after-value':
      return { ctx: { kind: 'after-value', used }, fn };
    default:
      return { ctx: { kind: 'unknown' }, fn };
  }
}

function jsonRules(tail: string, tools: ToolSchema[], names: string[]): GrammarCandidate[] {
  const { ctx, fn } = scanJson(tail);
  const tool = fn ? tools.find((t) => t.name === fn) : undefined;
  switch (ctx.kind) {
    case 'name-value':
      return splitPrefix(names, ctx.partial, (n) => {
        const t = tools.find((x) => x.name === n)!;
        return t.params.length ? '", "arguments": {"' : '", "arguments": {}}</tool_call>';
      });
    case 'after-name':
      return [{ text: ', "arguments": {"', prob: FORCED_PROB, closed: false }];
    case 'key': {
      if (!tool) return [];
      const free = tool.params.filter((x) => !ctx.used.includes(x.name) || x.name === ctx.partial).map((x) => x.name);
      if (ctx.partial === '' && tail.endsWith(', ')) {
        return splitPrefix(free, '', () => '": ').map((c) => ({ ...c, text: '"' + c.text }));
      }
      return splitPrefix(free, ctx.partial, () => '": ');
    }
    case 'after-key': {
      const prm = tool?.params.find((x) => x.name === ctx.key);
      if (!prm) return [];
      if (prm.enum) return prm.enum.map((e) => ({ text: `"${e}"`, prob: FORCED_PROB / prm.enum!.length, closed: false }));
      if (prm.type === 'boolean') return [{ text: 'false', prob: 0.5, closed: false }, { text: 'true', prob: 0.45, closed: false }];
      if (prm.type === 'string') return [{ text: '"', prob: FORCED_PROB, closed: false }];
      if (prm.type === 'array') return [{ text: '["', prob: FORCED_PROB, closed: false }];
      return [];
    }
    case 'string-value': {
      const prm = tool?.params.find((x) => x.name === ctx.key);
      if (prm?.enum) return splitPrefix(prm.enum, ctx.partial, () => '"');
      return [];
    }
    case 'after-value': {
      if (!tool) return [];
      const remainingReq = tool.params.filter((x) => x.required && !ctx.used.includes(x.name));
      if (remainingReq.length) return [{ text: `, "${remainingReq[0]!.name}": `, prob: 0.9, closed: false }];
      return [
        { text: '}}</tool_call>', prob: 0.6, closed: true },
        { text: ', "', prob: 0.3, closed: false },
      ];
    }
    default:
      return [];
  }
}

export function grammarContinuations(
  text: string,
  tools: ToolSchema[],
  dialect: CallDialect,
  thinking = true,
): GrammarCandidate[] {
  if (text.endsWith('<|im_start|>assistant\n')) {
    return [{ text: thinking ? '<think>\n' : '<think>\n\n</think>\n\n', prob: FORCED_PROB, closed: true }];
  }
  if (text.endsWith('</think>')) return [{ text: '\n\n', prob: FORCED_PROB, closed: true }];
  if (text.endsWith('</tool_call>')) return [{ text: '<|im_end|>', prob: 0.9, closed: true }];

  const open = text.lastIndexOf('<tool_call>');
  if (open < 0 || text.lastIndexOf('</tool_call>') > open) return [];
  const tail = text.slice(open + '<tool_call>'.length);
  const names = tools.map((t) => t.name);

  if (tail === '') {
    return dialect === 'json'
      ? [{ text: '{"name": "', prob: FORCED_PROB, closed: false }]
      : [{ text: '\n<function=', prob: FORCED_PROB, closed: false }];
  }
  if (dialect === 'xml' || /^\s*<function=/.test(tail)) return xmlRules(tail, tools, names);
  return jsonRules(tail, tools, names);
}

function exampleValue(param: ToolParam): string {
  if (param.enum && param.enum.length > 0) return param.enum[0]!;
  switch (param.type) {
    case 'boolean':
      return 'false';
    case 'integer':
      return /max_?chars|maxChars/i.test(param.name) ? '24000' : /limit|max_results/i.test(param.name) ? '5' : '1';
    case 'number':
      return '1';
    case 'array':
      return '["https://example.com"]';
    default:
      return param.name === 'url' ? 'https://example.com' : '';
  }
}

export function buildToolCorpus(tools: ToolSchema[] = [...LAB_TOOLS, ...EXTENSION_TOOLS]): string[] {
  const docs: string[] = [];
  for (const t of tools) {
    const req = t.params.filter((param) => param.required);
    const variants = [req, t.params.slice(0, Math.max(req.length, 1)), t.params];
    for (const v of variants) {
      const xmlBody = v.map((pm) => `<parameter=${pm.name}>\n${exampleValue(pm)}\n</parameter>\n`).join('');
      docs.push(`<tool_call>\n<function=${t.name}>\n${xmlBody}</function>\n</tool_call><|im_end|>`);
      const jsonArgs = v.map((pm) => {
        const val = exampleValue(pm);
        const lit = pm.type === 'string' ? JSON.stringify(val) : val;
        return `"${pm.name}": ${lit}`;
      });
      docs.push(`<tool_call>{"name": "${t.name}", "arguments": {${jsonArgs.join(', ')}}}</tool_call><|im_end|>`);
    }
  }
  docs.push('<|im_end|>\n<|im_start|>user\n<tool_response>\n');
  docs.push('\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n');
  docs.push('<|im_start|>assistant\n<think>\n\n</think>\n\n');
  docs.push('</think>\n\n');
  return docs;
}

export class AcceptanceCalibrator {
  private readonly a = new Map<string, number>();
  private readonly b = new Map<string, number>();
  constructor(private readonly priorStrength = 4) {}

  p(key: string, prior: number): number {
    const a = this.a.get(key) ?? 0;
    const b = this.b.get(key) ?? 0;
    return (a + prior * this.priorStrength) / (a + b + this.priorStrength);
  }

  observe(key: string, accepted: boolean): void {
    const m = accepted ? this.a : this.b;
    m.set(key, (m.get(key) ?? 0) + 1);
  }

  snapshot(): Record<string, { accepted: number; rejected: number }> {
    const out: Record<string, { accepted: number; rejected: number }> = {};
    for (const k of new Set([...this.a.keys(), ...this.b.keys()])) {
      out[k] = { accepted: this.a.get(k) ?? 0, rejected: this.b.get(k) ?? 0 };
    }
    return out;
  }
}

export function lenBucket(L: number): string {
  if (L <= 1) return 'L1';
  if (L <= 2) return 'L2';
  if (L <= 3) return 'L3';
  if (L <= 5) return 'L4-5';
  if (L <= 8) return 'L6-8';
  if (L <= 12) return 'L9-12';
  if (L <= 20) return 'L13-20';
  if (L <= 40) return 'L21-40';
  return 'L41+';
}

export function shareBucket(s: number): string {
  if (s >= 0.95) return 'S95';
  if (s >= 0.75) return 'S75';
  if (s >= 0.5) return 'S50';
  if (s >= 0.25) return 'S25';
  return 'S0';
}

export interface BudgetChoice {
  nodes: number;
  expectedTokens: number;
  expectedMs: number;
  tokPerMs: number;
}

export function chooseBudget(
  pathProbs: readonly number[],
  profile: VerifyCostProfile,
  maxNodes: number,
  draftMs: number,
): BudgetChoice {
  const ms0 = profile.stepMs(1) + draftMs;
  let best: BudgetChoice = { nodes: 0, expectedTokens: 1, expectedMs: ms0, tokPerMs: 1 / ms0 };
  let sum = 0;
  const cap = Math.min(maxNodes, pathProbs.length, profile.maxM - 1);
  for (let n = 1; n <= cap; n++) {
    sum += pathProbs[n - 1]!;
    const pReject = 1 - pathProbs[n - 1]!;
    const ms = profile.stepMs(n + 1) + draftMs + pReject * profile.rejectPenaltyMs;
    const tok = 1 + sum;
    if (tok / ms > best.tokPerMs) {
      best = { nodes: n, expectedTokens: tok, expectedMs: ms, tokPerMs: tok / ms };
    }
  }
  return best;
}

type SourceName = 'ctx' | 'anchor' | 'corpus' | 'history' | 'grammar';

type Cursor =
  | { kind: 'sam'; source: Exclude<SourceName, 'grammar'>; sam: SuffixAutomaton; state: number; len: number }
  | { kind: 'seq'; source: 'grammar' | 'anchor'; tag: string; toks: number[]; idx: number; prior: number };

interface Child {
  token: number;
  cond: number;
  key: string;
  cursors: Cursor[];
}

export interface ScoredDraftChain {
  tokens: number[];
  parents: number[];
  probs: number[];
  keys: string[];
}

export interface EngineAOptions {
  tokenizer: EngineATokenizer;
  extraTools?: ToolSchema[];
  corpus?: string[];
  sources?: Partial<Record<SourceName, boolean>>;
  adaptiveProfile?: VerifyCostProfile;
  minProb?: number;
}

const SOURCE_PRIOR: Record<SourceName, number> = { ctx: 1, anchor: 1, corpus: 0.9, history: 0.75, grammar: 1 };
const MIN_EXTERNAL_MATCH = 2;
const ANCHOR_RESYNC = 8;
const ANCHOR_MAX_DIVERGE = 12;

function lenPrior(L: number): number {
  return (0.94 * L) / (L + 1.2);
}

/**
 * Adaptive Engine A chain drafter for single-batch WebGPU verification (`K <= 7`).
 */
export class EngineASpeculator {
  private readonly tok: EngineATokenizer;
  private readonly enabled: Record<SourceName, boolean>;
  private readonly minProb: number;
  private readonly defaultTools: ToolSchema[];
  readonly calibrator = new AcceptanceCalibrator();
  private readonly corpusSam = new SuffixAutomaton(4096);
  private readonly historySam = new SuffixAutomaton(4096);
  private ctxSam = new SuffixAutomaton(4096);
  private corpusMatch = new SamMatcher(this.corpusSam);
  private historyMatch = new SamMatcher(this.historySam);
  private ctx: number[] = [];
  private anchor = -1;
  private anchorSync = false;
  private anchorDiverge = 0;
  private generated: number[][] = [];
  private textTail = '';
  private tools: ToolSchema[];
  private dialect: CallDialect = 'xml';
  private thinking = true;
  private sepId = -1;
  private lastObserveGenerated = false;
  private draftMsEma = 0.04;
  lastBudgetNodes = 0;

  constructor(private readonly opts: EngineAOptions) {
    this.tok = opts.tokenizer;
    this.enabled = { ctx: true, anchor: true, corpus: true, history: true, grammar: true, ...(opts.sources ?? {}) };
    this.minProb = opts.minProb ?? 0.02;
    this.defaultTools = [...(opts.extraTools ?? []), ...LAB_TOOLS, ...EXTENSION_TOOLS];
    this.tools = [...this.defaultTools];
    const corpusDocs = opts.corpus ?? buildToolCorpus(this.defaultTools);
    for (const doc of corpusDocs) {
      this.corpusSam.extendAll(this.tok.encode(doc));
      this.corpusSam.extend(this.sepId--);
    }
  }

  beginSample(promptText: string, thinkingOverride?: boolean): void {
    this.ctxSam = new SuffixAutomaton(4096);
    this.corpusMatch = new SamMatcher(this.corpusSam);
    this.historyMatch = new SamMatcher(this.historySam);
    this.ctx = [];
    this.anchor = -1;
    this.anchorSync = false;
    this.anchorDiverge = 0;
    this.generated = [];
    this.textTail = '';
    this.lastObserveGenerated = false;
    const parsed = parseToolsFromPrompt(promptText);
    this.tools = parsed.length ? parsed : [...this.defaultTools];
    this.dialect = detectDialect(promptText);
    this.thinking =
      thinkingOverride !== undefined ? thinkingOverride : !/enable_thinking\s*=\s*false/.test(promptText);
  }

  observe(tokens: readonly number[], generated: boolean): void {
    for (const t of tokens) {
      if (generated && this.enabled.anchor) this.advanceAnchor(t);
      this.ctx.push(t);
      this.ctxSam.extend(t);
      this.corpusMatch.push(t);
      this.historyMatch.push(t);
      if (generated && this.enabled.anchor && (this.anchor < 0 || !this.anchorSync)) {
        const m = this.ctxSam.longestRepeatedSuffix(3);
        if (m) {
          const e = this.ctxSam.prevEnd[m.state]!;
          if (e >= 0 && e < this.ctx.length - 1) {
            this.anchor = e;
            this.anchorSync = true;
            this.anchorDiverge = 0;
          }
        }
      }
    }
    if (generated && tokens.length > 0) {
      const last = this.generated[this.generated.length - 1];
      if (last && this.lastObserveGenerated) last.push(...tokens);
      else this.generated.push([...tokens]);
    }
    this.lastObserveGenerated = generated;
    if (this.enabled.grammar && tokens.length > 0) {
      // Decode only tail snippets to keep per-step overhead near zero
      const slice = tokens.length > 512 ? tokens.slice(-512) : tokens;
      this.textTail += this.tok.decode(slice);
      if (this.textTail.length > 4096) this.textTail = this.textTail.slice(-2048);
    }
  }

  private advanceAnchor(t: number): void {
    if (this.anchor < 0) return;
    const n = this.ctx.length;
    if (this.anchor + 1 < n && this.ctx[this.anchor + 1] === t) {
      this.anchor++;
      this.anchorSync = true;
      this.anchorDiverge = 0;
      return;
    }
    for (let j = this.anchor + 2; j < Math.min(n, this.anchor + 2 + ANCHOR_RESYNC); j++) {
      if (this.ctx[j] === t) {
        this.anchor = j;
        this.anchorSync = true;
        this.anchorDiverge = 0;
        return;
      }
    }
    this.anchorSync = false;
    if (++this.anchorDiverge > ANCHOR_MAX_DIVERGE) this.anchor = -1;
  }

  endSample(): void {
    if (!this.enabled.history) return;
    for (const seg of this.generated) {
      if (seg.length === 0) continue;
      this.historySam.extendAll(seg);
      this.historySam.extend(this.sepId--);
    }
    this.generated = [];
  }

  private rootCursors(): Cursor[] {
    const cur: Cursor[] = [];
    if (this.enabled.ctx) {
      const m = this.ctxSam.longestRepeatedSuffix(1);
      if (m) cur.push({ kind: 'sam', source: 'ctx', sam: this.ctxSam, state: m.state, len: m.len });
    }
    if (this.enabled.anchor && this.anchor >= 0 && this.anchor + 1 < this.ctx.length) {
      const toks = this.ctx.slice(this.anchor + 1, this.anchor + 17);
      cur.push({
        kind: 'seq',
        source: 'anchor',
        tag: this.anchorSync ? 'sync' : 'diverged',
        toks,
        idx: 0,
        prior: this.anchorSync ? 0.7 : 0.3,
      });
    }
    if (this.enabled.corpus && this.corpusMatch.len >= MIN_EXTERNAL_MATCH) {
      cur.push({
        kind: 'sam',
        source: 'corpus',
        sam: this.corpusSam,
        state: this.corpusMatch.state,
        len: this.corpusMatch.len,
      });
    }
    if (this.enabled.history && this.historyMatch.len >= MIN_EXTERNAL_MATCH) {
      cur.push({
        kind: 'sam',
        source: 'history',
        sam: this.historySam,
        state: this.historyMatch.state,
        len: this.historyMatch.len,
      });
    }
    if (this.enabled.grammar) cur.push(...this.grammarCursors());
    return cur;
  }

  private grammarCursors(): Cursor[] {
    const cands = grammarContinuations(this.textTail, this.tools, this.dialect, this.thinking);
    const out: Cursor[] = [];
    for (const c of cands) {
      const toks = this.alignContinuation(c.text, c.closed);
      if (toks && toks.length) {
        out.push({
          kind: 'seq',
          source: 'grammar',
          tag: c.prob >= 0.9 ? 'forced' : 'enum',
          toks,
          idx: 0,
          prior: c.prob,
        });
      }
    }
    return out;
  }

  private alignContinuation(text: string, closed: boolean): number[] | null {
    for (const r of [2, 6]) {
      if (this.ctx.length < r) continue;
      const tail = this.ctx.slice(-r);
      const enc = this.tok.encode(this.tok.decode(tail) + text);
      let ok = enc.length > r;
      for (let i = 0; ok && i < r; i++) ok = enc[i] === tail[i];
      if (!ok) continue;
      const rest = enc.slice(r);
      if (!closed) rest.pop();
      return rest;
    }
    return null;
  }

  private bestNextChild(cursors: readonly Cursor[]): Child | null {
    const byTok = new Map<number, Child>();
    const add = (token: number, cond: number, key: string, c: Cursor): void => {
      const e = byTok.get(token);
      if (!e) byTok.set(token, { token, cond, key, cursors: [c] });
      else {
        e.cursors.push(c);
        if (cond > e.cond) {
          e.cond = cond;
          e.key = key;
        }
      }
    };
    for (const c of cursors) {
      if (c.kind === 'seq') {
        if (c.idx >= c.toks.length) continue;
        const key = `${c.source}|${c.tag}|d${Math.min(c.idx, 4)}`;
        add(c.toks[c.idx]!, this.calibrator.p(key, c.prior), key, { ...c, idx: c.idx + 1 });
        continue;
      }
      const conts = c.sam.continuations(c.state, 3);
      if (!conts.length) continue;
      const z = c.sam.totalContinuationCount(c.state) + (c.len >= 2 ? 0.15 : 0.35);
      for (const [token, target, cnt] of conts) {
        if (token < 0) continue;
        const share = cnt / z;
        const key = `${c.source}|${lenBucket(c.len)}|${shareBucket(share)}`;
        const prior = Math.min(0.97, share * lenPrior(c.len) * SOURCE_PRIOR[c.source]);
        add(token, this.calibrator.p(key, prior), key, { ...c, state: target, len: c.len + 1 });
      }
    }
    let best: Child | null = null;
    for (const ch of byTok.values()) {
      if (!best || ch.cond > best.cond || (ch.cond === best.cond && ch.token < best.token)) {
        best = ch;
      }
    }
    return best;
  }

  /**
   * Proposes an adaptive greedy chain `[t_0, ..., t_{K-1}]` where `0 <= K <= maxNodes`.
   * Returns `K = 0` (`tokens: []`) when expected acceptance fails the cost-model threshold.
   */
  proposeChain(maxNodes: number): ScoredDraftChain {
    const t0 = performance.now();
    const tokens: number[] = [];
    const parents: number[] = [];
    const probs: number[] = [];
    const keys: string[] = [];

    let cur = maxNodes > 0 ? this.rootCursors() : [];
    let p = 1;
    for (let idx = 0; idx < maxNodes && cur.length > 0; idx++) {
      const ch = this.bestNextChild(cur);
      if (!ch) break;
      const nextP = p * ch.cond;
      if (nextP < this.minProb) break;
      tokens.push(ch.token);
      parents.push(idx - 1);
      probs.push(nextP);
      keys.push(ch.key);
      p = nextP;
      cur = ch.cursors;
    }

    let n = tokens.length;
    if (this.opts.adaptiveProfile && n > 0) {
      n = chooseBudget(probs, this.opts.adaptiveProfile, maxNodes, this.draftMsEma).nodes;
    }
    this.lastBudgetNodes = n;
    const dt = performance.now() - t0;
    this.draftMsEma = 0.9 * this.draftMsEma + 0.1 * dt;
    return {
      tokens: tokens.slice(0, n),
      parents: parents.slice(0, n),
      probs: probs.slice(0, n),
      keys: keys.slice(0, n),
    };
  }

  feedback(chain: ScoredDraftChain, acceptedCount: number): void {
    const clamped = Math.max(0, Math.min(acceptedCount, chain.tokens.length));
    for (let i = 0; i < clamped; i++) {
      this.calibrator.observe(chain.keys[i]!, true);
    }
    if (clamped < chain.tokens.length) {
      this.calibrator.observe(chain.keys[clamped]!, false);
    }
  }
}
