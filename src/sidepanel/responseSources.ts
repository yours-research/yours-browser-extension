import type { MessageNode } from '@yours/harness';
import type { FetchedSourceResult } from '@yours/tools';

export interface TurnResponseSource {
  id: string;
  url: string;
  domain: string;
  title: string;
  snippet: string;
}

const QUOTE_AWARE_TAG_RE = /<(?:[^>"']|"[^"]*"|'[^']*')*>/g;
const TRACKING_PARAMS = new Set([
  'utm_source',
  'utm_medium',
  'utm_campaign',
  'utm_term',
  'utm_content',
  'fbclid',
  'gclid',
]);

export function normalizeSourceUrl(rawUrl: string | undefined): string | undefined {
  if (!rawUrl || typeof rawUrl !== 'string' || rawUrl.length > 4096) {
    return undefined;
  }
  try {
    const parsed = new URL(rawUrl.trim());
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      return undefined;
    }
    if (parsed.username || parsed.password) {
      return undefined;
    }
    parsed.hash = '';
    for (const key of Array.from(parsed.searchParams.keys())) {
      if (TRACKING_PARAMS.has(key.toLowerCase())) {
        parsed.searchParams.delete(key);
      }
    }
    let href = parsed.toString();
    if (parsed.pathname.length > 1 && href.endsWith('/')) {
      href = href.slice(0, -1);
    }
    return href;
  } catch {
    return undefined;
  }
}

export function extractDomainLabel(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./i, '');
  } catch {
    return url;
  }
}

function isDecorativeSymbolLine(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) return false;
  const nonWs = Array.from(trimmed.replace(/\s+/g, ''));
  if (nonWs.length < 6) return false;
  let nonAlnum = 0;
  for (const ch of nonWs) {
    if (!/[\p{L}\p{N}]/u.test(ch)) nonAlnum++;
  }
  return nonAlnum / nonWs.length > 0.65;
}

function isCssOrAttrDebris(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) return false;
  return /^(?:[^<\w]*=\s*(?:\d+(?:\.\d+)?(?:rem|px|em|vw|vh|%)|["'][^"']*["'])|\s*(?:rel|media|data-target|data-[a-z0-9-]+|href|src|srcset|crossorigin|integrity|sizes)=["'][^"']*["'])/i.test(
    trimmed,
  );
}

export function sanitizeSourceSnippetText(
  candidates: Array<string | undefined>,
  maxChars = 200,
): string {
  for (const raw of candidates) {
    if (!raw || typeof raw !== 'string') continue;
    const stripped = raw
      .replace(QUOTE_AWARE_TAG_RE, ' ')
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/\[([^\]]+)\]\((?:[^)]+)\)/g, '$1')
      .replace(/\r\n?/g, '\n');

    const lines = stripped
      .split('\n')
      .map((l) => l.replace(/\s+/g, ' ').trim())
      .filter((l) => {
        if (!l || l.length < 12) return false;
        if (/^#{1,6}\s/.test(l)) return false;
        if (l.startsWith('|') && l.endsWith('|')) return false;
        if (/^[-*_]{3,}$/.test(l)) return false;
        if (isCssOrAttrDebris(l) || isDecorativeSymbolLine(l)) return false;
        return true;
      });

    for (const line of lines) {
      const cleaned = line.replace(/^[-*•>\s]+/, '').trim();
      if (!cleaned || isDecorativeSymbolLine(cleaned) || isCssOrAttrDebris(cleaned)) {
        continue;
      }
      if (cleaned.length <= maxChars) return cleaned;
      const sliced = cleaned.slice(0, maxChars - 1);
      const lastSpace = sliced.lastIndexOf(' ');
      const cut = lastSpace > maxChars * 0.6 ? lastSpace : maxChars - 1;
      return `${sliced.slice(0, cut).trimEnd()}…`;
    }
  }
  return '';
}

export function toTurnResponseSources(
  fetched: readonly FetchedSourceResult[],
): TurnResponseSource[] {
  const map = new Map<string, TurnResponseSource>();
  for (const item of fetched) {
    if (item.error) continue;
    const url = normalizeSourceUrl(item.url);
    if (!url) continue;
    const domain = extractDomainLabel(url);
    const title = sanitizeSourceSnippetText([item.title], 120) || item.title || domain;
    const snippet = sanitizeSourceSnippetText([item.snippet, item.excerpt], 200);
    const prev = map.get(url);
    if (!prev) {
      map.set(url, { id: url, url, domain, title, snippet });
    } else {
      map.set(url, {
        ...prev,
        title: prev.title === prev.domain && title !== domain ? title : prev.title,
        snippet: !prev.snippet || snippet.length > prev.snippet.length ? snippet : prev.snippet,
      });
    }
  }
  return Array.from(map.values());
}

function parseSourcesFromToolContent(raw: string): TurnResponseSource[] {
  const out: TurnResponseSource[] = [];
  const cleaned = raw.replace(/\[BEGIN UNTRUSTED CONTENT[^\]]*\]|\[END UNTRUSTED CONTENT\]/g, '');

  // Pattern 1: "### Title\nURL: https://...\n\nExcerpt..."
  const headingMatches = cleaned.matchAll(
    /###\s+([^\n]+)\nURL:\s*(https?:\/\/\S+)([\s\S]*?)(?=\n###\s|\n---|\n##\s|$)/g,
  );
  for (const m of headingMatches) {
    const url = normalizeSourceUrl(m[2]);
    if (!url) continue;
    const domain = extractDomainLabel(url);
    const title = sanitizeSourceSnippetText([m[1]], 120) || domain;
    const snippet = sanitizeSourceSnippetText([m[3]], 200);
    out.push({ id: url, url, domain, title, snippet });
  }

  // Pattern 2: "1. Title\n   URL: https://...\n   Snippet"
  const listMatches = cleaned.matchAll(
    /\n?\d+\.\s+([^\n]+)\n\s+URL:\s*(https?:\/\/\S+)(?:\n\s+([^\n]+))?/g,
  );
  for (const m of listMatches) {
    const url = normalizeSourceUrl(m[2]);
    if (!url) continue;
    const domain = extractDomainLabel(url);
    const title = sanitizeSourceSnippetText([m[1]], 120) || domain;
    const snippet = sanitizeSourceSnippetText([m[3]], 200);
    out.push({ id: url, url, domain, title, snippet });
  }

  return out;
}

export function findTurnKeyForAssistantNode(
  visibleNodes: readonly MessageNode[],
  assistantIndex: number,
): string {
  for (let k = assistantIndex - 1; k >= 0; k--) {
    if (visibleNodes[k]!.message.role === 'user') {
      return visibleNodes[k]!.id;
    }
  }
  return '__root__';
}

/**
 * Collects deduplicated sources scoped strictly to the turn containing `assistantIndex`
 * (from the nearest preceding `user` node to `assistantIndex`), returning clean
 * `TurnResponseSource` items for only the final assistant answer of that turn.
 */
export function collectTurnSourcesForAssistant(params: {
  visibleNodes: readonly MessageNode[];
  assistantIndex: number;
  sourcesByTurn?: Readonly<Record<string, TurnResponseSource[]>>;
}): TurnResponseSource[] {
  const { visibleNodes, assistantIndex, sourcesByTurn } = params;
  const node = visibleNodes[assistantIndex];
  if (!node || node.message.role !== 'assistant') {
    return [];
  }

  let startIdx = assistantIndex;
  while (startIdx > 0 && visibleNodes[startIdx - 1]!.message.role !== 'user') {
    startIdx -= 1;
  }

  const turnKey = findTurnKeyForAssistantNode(visibleNodes, assistantIndex);
  const dedup = new Map<string, TurnResponseSource>();
  const mergeList = (list: readonly TurnResponseSource[]) => {
    for (const s of list) {
      const prev = dedup.get(s.url);
      if (!prev) {
        dedup.set(s.url, s);
      } else {
        dedup.set(s.url, {
          ...prev,
          title: prev.title === prev.domain && s.title !== s.domain ? s.title : prev.title,
          snippet: s.snippet || prev.snippet,
        });
      }
    }
  };

  for (let i = startIdx; i <= assistantIndex; i++) {
    const m = visibleNodes[i]!.message;
    if (
      m.role === 'tool' &&
      (m.toolName === 'search_web_tab' ||
        m.toolName === 'web_search' ||
        m.toolName === 'fetch_background_tab' ||
        m.toolName === 'fetch_and_read_url' ||
        m.toolName === 'read_tab' ||
        m.toolName === 'read_active_tab')
    ) {
      mergeList(parseSourcesFromToolContent(m.content));
    }
  }

  if (sourcesByTurn?.[turnKey]) {
    mergeList(sourcesByTurn[turnKey]!);
  }

  return Array.from(dedup.values());
}

/**
 * Converts standalone numeric citations `[1]`, `[2]` inside assistant prose (outside fenced/inline code
 * and existing Markdown links) into clickable Markdown citation links matching turn `sources`.
 */
export function linkifyInlineCitations(
  markdown: string,
  sources: readonly TurnResponseSource[],
): string {
  if (!markdown || !sources || sources.length === 0) {
    return markdown;
  }
  const tokens = markdown.split(/(`[^`]*`|\[[^\]]+\]\([^)]+\))/g);
  return tokens
    .map((part, idx) => {
      if (idx % 2 === 1) {
        return part;
      }
      return part.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?!\()/g, (full, inner: string) => {
        const nums = inner
          .split(',')
          .map((s) => Number.parseInt(s.trim(), 10))
          .filter((n) => Number.isFinite(n) && n >= 1 && n <= sources.length);
        if (nums.length === 0) {
          return full;
        }
        return nums
          .map((n) => {
            const src = sources[n - 1]!;
            return `[[${n}]](${src.url})`;
          })
          .join(' ');
      });
    })
    .join('');
}
