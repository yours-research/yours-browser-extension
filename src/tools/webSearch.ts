import { decodeHtmlEntities } from './htmlDistiller.ts';
import type { HiddenTabHtmlResponse, SearchResultItem, WebSearchResult } from './types.ts';

const ANTI_BOT_PATTERNS = [
  /cf-turnstile/i,
  /challenges\.cloudflare\.com/i,
  /\/cdn-cgi\/challenge-platform/i,
  /\bg-recaptcha\b/i,
  /\bhcaptcha\b/i,
  /data-sitekey/i,
  /verify you are human/i,
  /are you a robot/i,
  /please complete the security check/i,
  /unusual traffic from your computer/i,
  /duckduckgo\.com\/anomaly\.js/i,
  /anomaly-modal/i,
  /captcha[-_]?form/i,
  /solve the captcha/i,
];

export function detectAntiBotChallenge(html: string, status?: number): boolean {
  if (status === 429) {
    return true;
  }
  const text = html || '';
  for (const pattern of ANTI_BOT_PATTERNS) {
    if (pattern.test(text)) {
      return true;
    }
  }
  if (status === 403) {
    if (
      text.length < 15000 &&
      /(cloudflare|just a moment|attention required|access denied|captcha|bot|security check|forbidden)/i.test(text)
    ) {
      return true;
    }
    return text.trim().length === 0;
  }
  return false;
}

function unwrapDuckDuckGoHref(rawHref: string): string | null {
  const decodedHref = decodeHtmlEntities(rawHref).trim();
  if (!decodedHref) {
    return null;
  }

  let normalizedHref = decodedHref;
  if (normalizedHref.startsWith('//')) {
    normalizedHref = `https:${normalizedHref}`;
  } else if (normalizedHref.startsWith('/')) {
    normalizedHref = `https://duckduckgo.com${normalizedHref}`;
  }

  try {
    const parsed = new URL(normalizedHref);
    // Unwrap DDG redirect: /l/?uddg=https%3A%2F%2F...
    if (parsed.hostname.endsWith('duckduckgo.com') && parsed.pathname.startsWith('/l/')) {
      const uddg = parsed.searchParams.get('uddg');
      if (uddg) {
        const target = new URL(uddg);
        if (target.protocol === 'http:' || target.protocol === 'https:') {
          return target.toString();
        }
      }
      return null;
    }
    // Skip DuckDuckGo ad/internal tracking links
    if (parsed.hostname.endsWith('duckduckgo.com') && parsed.pathname.startsWith('/y.js')) {
      return null;
    }
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.toString();
    }
    return null;
  } catch {
    return null;
  }
}

function stripTagsClean(fragment: string): string {
  return decodeHtmlEntities(fragment.replace(/<[^>]+>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

export function parseDuckDuckGoHtmlResults(html: string, limit = 5): SearchResultItem[] {
  if (!html) {
    return [];
  }

  const maxResults = Math.max(1, Math.min(8, Math.round(limit)));
  const items: SearchResultItem[] = [];
  const seenUrls = new Set<string>();

  const pushItem = (rawHref: string, rawTitle: string, rawSnippet: string) => {
    if (items.length >= maxResults) {
      return;
    }
    const url = unwrapDuckDuckGoHref(rawHref);
    if (!url || seenUrls.has(url)) {
      return;
    }
    const title = stripTagsClean(rawTitle);
    const snippet = stripTagsClean(rawSnippet);
    if (!title) {
      return;
    }
    try {
      const domain = new URL(url).hostname.replace(/^www\./, '');
      seenUrls.add(url);
      items.push({ title, url, snippet, domain });
    } catch {
      // ignore invalid URL
    }
  };

  // Format 1: html.duckduckgo.com/html/ (.result blocks)
  const linkAnchorRegex = /<a\b([^>]*class\s*=\s*["'][^"']*\bresult__a\b[^"']*["'][^>]*)>([\s\S]*?)<\/a>/gi;
  const anchorMatches: Array<{ index: number; href: string; titleHtml: string }> = [];
  let match: RegExpExecArray | null;

  while ((match = linkAnchorRegex.exec(html)) !== null) {
    const attrs = match[1];
    const hrefMatch = /href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(attrs);
    const href = hrefMatch ? hrefMatch[1] ?? hrefMatch[2] ?? hrefMatch[3] ?? '' : '';
    anchorMatches.push({
      index: match.index,
      href,
      titleHtml: match[2],
    });
  }

  for (let i = 0; i < anchorMatches.length && items.length < maxResults; i++) {
    const cur = anchorMatches[i];
    const nextStart = i + 1 < anchorMatches.length ? anchorMatches[i + 1].index : html.length;
    const blockSlice = html.slice(cur.index, Math.min(nextStart, cur.index + 4000));
    const snippetMatch =
      /<(?:a|div|span|td)\b[^>]*class\s*=\s*["'][^"']*\bresult__snippet\b[^"']*["'][^>]*>([\s\S]*?)<\/(?:a|div|span|td)>/i.exec(
        blockSlice,
      );
    pushItem(cur.href, cur.titleHtml, snippetMatch ? snippetMatch[1] : '');
  }

  if (items.length > 0) {
    return items;
  }

  // Format 2: lite.duckduckgo.com/lite/ (a.result-link + td.result-snippet)
  const liteRegex = /<a\b([^>]*class\s*=\s*["'][^"']*\bresult-link\b[^"']*["'][^>]*)>([\s\S]*?)<\/a>/gi;
  const liteMatches: Array<{ index: number; href: string; titleHtml: string }> = [];
  while ((match = liteRegex.exec(html)) !== null) {
    const attrs = match[1];
    const hrefMatch = /href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(attrs);
    const href = hrefMatch ? hrefMatch[1] ?? hrefMatch[2] ?? hrefMatch[3] ?? '' : '';
    liteMatches.push({ index: match.index, href, titleHtml: match[2] });
  }

  for (let i = 0; i < liteMatches.length && items.length < maxResults; i++) {
    const cur = liteMatches[i];
    const nextStart = i + 1 < liteMatches.length ? liteMatches[i + 1].index : html.length;
    const blockSlice = html.slice(cur.index, Math.min(nextStart, cur.index + 3000));
    const snippetMatch =
      /<td\b[^>]*class\s*=\s*["'][^"']*\bresult-snippet\b[^"']*["'][^>]*>([\s\S]*?)<\/td>/i.exec(blockSlice);
    pushItem(cur.href, cur.titleHtml, snippetMatch ? snippetMatch[1] : '');
  }

  return items;
}

export interface WebSearchDeps {
  hasPermission?: () => Promise<boolean>;
  fetchFn?: typeof fetch;
  openHiddenTabAndExtract?: (url: string) => Promise<HiddenTabHtmlResponse>;
  timeoutMs?: number;
}

async function defaultCheckPermission(): Promise<boolean> {
  if (typeof chrome === 'undefined' || !chrome.permissions?.contains) {
    return false;
  }
  try {
    return await chrome.permissions.contains({ origins: ['https://*/*'] });
  } catch {
    return false;
  }
}

export async function runWebSearch(
  query: string,
  limit = 5,
  deps?: WebSearchDeps,
): Promise<WebSearchResult> {
  const cleanQuery = query.trim();
  const clampedLimit = Math.max(1, Math.min(8, Math.round(limit)));

  if (!cleanQuery) {
    return { ok: false, query: cleanQuery, results: [], error: 'Search query cannot be empty.' };
  }

  // Option A fail-closed permission check BEFORE any fetch() or chrome.tabs.create()
  const hasPermission = await (deps?.hasPermission ? deps.hasPermission() : defaultCheckPermission());
  if (!hasPermission) {
    return {
      ok: false,
      query: cleanQuery,
      results: [],
      needsPermission: true,
      error: 'Optional host permission (https://*/*) is required before running web_search.',
    };
  }

  const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(cleanQuery)}`;
  const fetchFn = deps?.fetchFn ?? fetch;
  const timeoutMs = deps?.timeoutMs ?? 10000;

  let pathAStatus: number | undefined;
  let pathAHtml = '';

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetchFn(searchUrl, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          Accept: 'text/html,application/xhtml+xml',
        },
      });
      pathAStatus = response.status;
      pathAHtml = await response.text();
    } finally {
      clearTimeout(timer);
    }

    if (detectAntiBotChallenge(pathAHtml, pathAStatus)) {
      return {
        ok: false,
        query: cleanQuery,
        results: [],
        blockedByCaptcha: true,
        manualUrl: searchUrl,
        error: 'DuckDuckGo returned an anti-bot/CAPTCHA challenge. Please open manually.',
      };
    }

    if (pathAStatus >= 200 && pathAStatus < 400) {
      const parsed = parseDuckDuckGoHtmlResults(pathAHtml, clampedLimit);
      if (parsed.length > 0) {
        return {
          ok: true,
          query: cleanQuery,
          results: parsed,
          fromHiddenTab: false,
        };
      }
    }
  } catch {
    // Fall through to hidden-tab fallback if available
  }

  // Path B: Reusable hidden background tab fallback (ONLY when needed, never bypassing CAPTCHAs)
  if (deps?.openHiddenTabAndExtract) {
    try {
      const tabRes = await deps.openHiddenTabAndExtract(searchUrl);
      if (detectAntiBotChallenge(tabRes.html)) {
        return {
          ok: false,
          query: cleanQuery,
          results: [],
          fromHiddenTab: true,
          blockedByCaptcha: true,
          manualUrl: searchUrl,
          error: 'Search blocked by anti-bot/CAPTCHA challenge in background tab.',
        };
      }
      const parsed = parseDuckDuckGoHtmlResults(tabRes.html, clampedLimit);
      return {
        ok: parsed.length > 0,
        query: cleanQuery,
        results: parsed,
        fromHiddenTab: true,
        ...(parsed.length === 0 ? { error: 'No search results found.' } : {}),
      };
    } catch (err: unknown) {
      return {
        ok: false,
        query: cleanQuery,
        results: [],
        fromHiddenTab: true,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  return {
    ok: false,
    query: cleanQuery,
    results: [],
    error: pathAStatus
      ? `Search request failed with status ${pathAStatus} and returned 0 results.`
      : 'Search request failed or returned 0 results.',
  };
}
