import { resolveToolAlias } from '@yours/tools/schemas';
import { rankPassagesBm25 } from './bm25Ranker.ts';
import { sanitizeUntrustedText } from './guardBridge.ts';
import { runJsCalc } from './jsCalc.ts';
import { extractTabContent } from './pageExtractor.ts';
import { validateToolCall } from './toolProtocol.ts';
import { createExtensionToolRuntime } from './yoursToolRuntime.ts';
import type {
  ExtractedPageContent,
  FetchedSourceResult,
  HiddenTabHtmlResponse,
  JsCalcInput,
  OpenTabInfo,
  PageExtractMode,
  ToolExecutionResult,
} from './types.ts';
import { runFetchAndReadUrls, type UrlFetcherDeps } from './urlFetcher.ts';
import { runWebSearch, type WebSearchDeps } from './webSearch.ts';

export async function checkResearchPermission(): Promise<boolean> {
  if (typeof chrome === 'undefined' || !chrome.permissions?.contains) {
    return false;
  }
  try {
    return await chrome.permissions.contains({ origins: ['https://*/*'] });
  } catch {
    return false;
  }
}

export async function requestResearchPermission(): Promise<boolean> {
  if (typeof chrome === 'undefined' || !chrome.permissions?.request) {
    return false;
  }
  try {
    return await chrome.permissions.request({ origins: ['https://*/*'] });
  } catch {
    return false;
  }
}

export async function listOpenTabs(): Promise<OpenTabInfo[]> {
  if (typeof chrome === 'undefined' || !chrome.tabs?.query) {
    return [];
  }
  const tabs = await chrome.tabs.query({});
  return tabs
    .filter((t): t is chrome.tabs.Tab & { id: number } => typeof t.id === 'number')
    .map((t) => ({
      id: t.id,
      title: t.title ?? '',
      url: t.url ?? '',
      active: Boolean(t.active),
      windowId: t.windowId,
    }));
}

async function getActiveTabId(): Promise<number | null> {
  if (typeof chrome === 'undefined' || !chrome.tabs?.query) {
    return null;
  }
  const isNormalWebUrl = (url?: string) =>
    typeof url === 'string' && (url.startsWith('http://') || url.startsWith('https://'));

  const activeTabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  const focusedTab = activeTabs[0];
  if (focusedTab?.id !== undefined && isNormalWebUrl(focusedTab.url)) {
    return focusedTab.id;
  }

  // Side Panel window or standalone extension tab can be focused: pick active normal web tab first
  const allActiveTabs = await chrome.tabs.query({ active: true });
  const activeWebTab = allActiveTabs.find((t) => t.id !== undefined && isNormalWebUrl(t.url));
  if (activeWebTab?.id !== undefined) {
    return activeWebTab.id;
  }

  const allTabs = await chrome.tabs.query({});
  const webTabs = allTabs
    .filter((t): t is chrome.tabs.Tab & { id: number } => t.id !== undefined && isNormalWebUrl(t.url))
    .sort((a, b) => (b.lastAccessed ?? 0) - (a.lastAccessed ?? 0));
  if (webTabs[0]) {
    return webTabs[0].id;
  }

  return focusedTab?.id ?? allActiveTabs[0]?.id ?? null;
}

export async function extractActiveTabPreview(maxChars = 5000): Promise<ExtractedPageContent> {
  const tabId = await getActiveTabId();
  if (tabId === null) {
    return {
      mode: 'article',
      title: '',
      url: '',
      content: '',
      contentHash: '00000000',
      charCount: 0,
      truncated: false,
      error: 'No active browser tab found.',
    };
  }
  const page = await extractTabContent(tabId, { mode: 'auto', maxChars });
  if (page.content) {
    const sanitized = sanitizeUntrustedText(page.content, page.url);
    return {
      ...page,
      content: sanitized.cleanText,
      charCount: sanitized.cleanText.length,
    };
  }
  return page;
}

/**
 * Scrolls to and visually selects a matching text excerpt inside the active web tab
 * while respecting `prefers-reduced-motion: reduce`.
 */
export async function highlightActiveTabSelection(
  text: string,
): Promise<{ ok: boolean; found: boolean; url?: string; error?: string }> {
  const snippet = text.replace(/\s+/g, ' ').trim();
  if (!snippet) {
    return { ok: false, found: false, error: 'Selection excerpt is empty.' };
  }
  const tabId = await getActiveTabId();
  if (tabId === null) {
    return { ok: false, found: false, error: 'No active browser tab found.' };
  }
  if (typeof chrome === 'undefined' || !chrome.scripting?.executeScript) {
    return { ok: false, found: false, error: 'chrome.scripting API is unavailable.' };
  }

  const results = await chrome.scripting.executeScript({
    target: { tabId },
    args: [snippet],
    func: (rawNeedle: string) => {
      const root = document.body || document.documentElement;
      if (!root) return { found: false, url: location.href };

      const normalizedNeedle = rawNeedle.replace(/\s+/g, ' ').trim().toLowerCase();
      const shortNeedle =
        normalizedNeedle.length > 96 ? normalizedNeedle.slice(0, 96).trim() : normalizedNeedle;
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          const tag = parent.tagName;
          if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT') {
            return NodeFilter.FILTER_REJECT;
          }
          if (!node.textContent || !node.textContent.trim()) {
            return NodeFilter.FILTER_SKIP;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      });

      let targetRange: Range | null = null;
      let targetEl: HTMLElement | null = null;
      let node: Node | null;
      while ((node = walker.nextNode())) {
        const rawText = node.textContent ?? '';
        const lowerText = rawText.toLowerCase();
        let idx = lowerText.indexOf(normalizedNeedle);
        let matchLen = normalizedNeedle.length;
        if (idx < 0 && shortNeedle.length >= 8) {
          idx = lowerText.indexOf(shortNeedle);
          matchLen = shortNeedle.length;
        }
        if (idx >= 0) {
          const r = document.createRange();
          r.setStart(node, idx);
          r.setEnd(node, Math.min(rawText.length, idx + matchLen));
          targetRange = r;
          targetEl = node.parentElement;
          break;
        }
      }

      if (!targetRange && typeof (window as unknown as { find?: (q: string) => boolean }).find === 'function') {
        const sel = window.getSelection();
        sel?.removeAllRanges();
        const found = (window as unknown as { find: (q: string) => boolean }).find(
          rawNeedle.slice(0, 96),
        );
        if (found && sel && sel.rangeCount > 0) {
          targetRange = sel.getRangeAt(0);
          const container = targetRange.commonAncestorContainer;
          targetEl = container.nodeType === 1 ? (container as HTMLElement) : container.parentElement;
        }
      }

      if (!targetRange) {
        return { found: false, url: location.href };
      }

      const sel = window.getSelection();
      if (sel) {
        sel.removeAllRanges();
        sel.addRange(targetRange);
      }

      const reduced =
        typeof window.matchMedia === 'function' &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const scrollTarget = targetEl ?? targetRange.commonAncestorContainer.parentElement;
      if (scrollTarget && typeof scrollTarget.scrollIntoView === 'function') {
        scrollTarget.scrollIntoView({
          behavior: reduced ? 'auto' : 'smooth',
          block: 'center',
        });
      }

      const styleId = 'yours-ext-highlight-style';
      if (!document.getElementById(styleId) && document.head) {
        const st = document.createElement('style');
        st.id = styleId;
        st.textContent =
          '.yours-ext-highlight-pulse{outline:2px solid #2563eb;outline-offset:3px;border-radius:4px;}';
        document.head.appendChild(st);
      }
      if (scrollTarget) {
        scrollTarget.classList.add('yours-ext-highlight-pulse');
        window.setTimeout(() => {
          scrollTarget.classList.remove('yours-ext-highlight-pulse');
        }, 1800);
      }

      return { found: true, url: location.href };
    },
  });

  const first = results?.[0]?.result as { found?: boolean; url?: string } | undefined;
  return {
    ok: true,
    found: Boolean(first?.found),
    ...(first?.url ? { url: first.url } : {}),
  };
}

export interface ExecuteToolOptions {
  hasPermission?: () => Promise<boolean>;
  searchViaChromiumSerp?: WebSearchDeps['searchViaChromiumSerp'];
  fetchFn?: typeof fetch;
  openHiddenTabAndExtract?: (url: string) => Promise<HiddenTabHtmlResponse>;
  onSourceReady?: (src: FetchedSourceResult) => void;
}

const LEGACY_TIER1_NAMES = new Set([
  'read_active_page',
  'list_open_tabs',
  'read_tab',
  'js_calc',
  'web_search',
  'fetch_and_read_url',
]);

export async function executeToolCall(
  rawCall: { name: string; arguments?: Record<string, unknown> },
  options?: ExecuteToolOptions,
): Promise<ToolExecutionResult> {
  if (!LEGACY_TIER1_NAMES.has(rawCall.name) && resolveToolAlias(rawCall.name)) {
    const runtime = createExtensionToolRuntime({
      adapter: {
        hasPermission: options?.hasPermission,
        searchViaChromiumSerp: options?.searchViaChromiumSerp,
        fetchFn: options?.fetchFn,
        openHiddenTabAndExtract: options?.openHiddenTabAndExtract,
        onSourceReady: options?.onSourceReady,
      },
    });
    const res = await runtime.execute(rawCall);
    return res as unknown as ToolExecutionResult;
  }

  const validation = validateToolCall({
    name: rawCall.name,
    arguments: rawCall.arguments ?? {},
  });

  if (!validation.valid || !validation.normalized) {
    return {
      ok: false,
      toolName: (rawCall.name as ToolExecutionResult['toolName']) ?? 'read_active_page',
      error: validation.error ?? 'Invalid tool call.',
    };
  }

  const call = validation.normalized;
  const checkPerm = options?.hasPermission ?? checkResearchPermission;

  switch (call.name) {
    case 'read_active_page': {
      const tabId = await getActiveTabId();
      if (tabId === null) {
        return { ok: false, toolName: 'read_active_page', error: 'No active browser tab found.' };
      }
      const mode = (call.arguments.mode as PageExtractMode) ?? 'auto';
      const maxChars = (call.arguments.maxChars as number) ?? 8000;
      const query = typeof call.arguments.query === 'string' ? call.arguments.query.trim() : '';
      const page = await extractTabContent(tabId, { mode, maxChars });
      if (page.error && !page.content) {
        return { ok: false, toolName: 'read_active_page', data: page, error: page.error };
      }
      const searchable = page.fullContent || page.content;
      const rawFormatted = query
        ? `[${page.mode.toUpperCase()}: MATCHED FOR "${query}"] ${page.title}\nURL: ${page.url}\n\n${rankPassagesBm25(searchable, query, maxChars).excerpt}`
        : `[${page.mode.toUpperCase()}] ${page.title}\nURL: ${page.url}\n\n${page.content}`;
      const formatted = sanitizeUntrustedText(rawFormatted, page.url).cleanText;
      return { ok: true, toolName: 'read_active_page', data: page, formatted };
    }

    case 'list_open_tabs': {
      const tabs = await listOpenTabs();
      const formatted =
        tabs.length === 0
          ? 'No open tabs found.'
          : tabs
              .map((t) => `[${t.id}]${t.active ? ' *active*' : ''} ${t.title || '(untitled)'} — ${t.url}`)
              .join('\n');
      return { ok: true, toolName: 'list_open_tabs', data: tabs, formatted };
    }

    case 'read_tab': {
      const tabId = call.arguments.tabId as number;
      const mode = (call.arguments.mode as PageExtractMode) ?? 'auto';
      const maxChars = (call.arguments.maxChars as number) ?? 8000;
      const query = typeof call.arguments.query === 'string' ? call.arguments.query.trim() : '';
      const page = await extractTabContent(tabId, { mode, maxChars });
      if (page.error && !page.content) {
        return { ok: false, toolName: 'read_tab', data: page, error: page.error };
      }
      const searchable = page.fullContent || page.content;
      const rawFormatted = query
        ? `[${page.mode.toUpperCase()}: MATCHED FOR "${query}"] ${page.title}\nURL: ${page.url}\n\n${rankPassagesBm25(searchable, query, maxChars).excerpt}`
        : `[${page.mode.toUpperCase()}] ${page.title}\nURL: ${page.url}\n\n${page.content}`;
      const formatted = sanitizeUntrustedText(rawFormatted, page.url).cleanText;
      return { ok: true, toolName: 'read_tab', data: page, formatted };
    }

    case 'js_calc': {
      const calcResult = runJsCalc(call.arguments as unknown as JsCalcInput);
      return {
        ok: calcResult.ok,
        toolName: 'js_calc',
        data: calcResult,
        formatted: calcResult.formatted,
        error: calcResult.error,
      };
    }

    case 'web_search': {
      const granted = await checkPerm();
      if (!granted) {
        return {
          ok: false,
          toolName: 'web_search',
          needsPermission: true,
          error: 'Optional host permission (https://*/*) is required for web_search.',
        };
      }
      const searchDeps: WebSearchDeps = {
        hasPermission: async () => true,
        searchViaChromiumSerp: options?.searchViaChromiumSerp,
        fetchFn: options?.fetchFn,
        openHiddenTabAndExtract: options?.openHiddenTabAndExtract,
      };
      const queryStr = call.arguments.query as string;
      const limit = (call.arguments.limit as number) ?? 5;
      const readTopN =
        typeof call.arguments.readTopN === 'number'
          ? call.arguments.readTopN
          : options?.searchViaChromiumSerp || options?.openHiddenTabAndExtract
            ? 2
            : 0;
      const res = await runWebSearch(queryStr, limit, searchDeps);
      if (res.blockedByCaptcha) {
        return {
          ok: false,
          toolName: 'web_search',
          data: res,
          blockedByCaptcha: true,
          manualUrl: res.manualUrl,
          error: res.error,
        };
      }
      if (!res.ok) {
        return { ok: false, toolName: 'web_search', data: res, error: res.error };
      }

      const formattedResults = res.results
        .map((r, i) => `${i + 1}. ${r.title}\n   URL: ${r.url}\n   ${r.snippet}`)
        .join('\n\n');

      // Fused 1-step Firecrawl-style Search + Read: extract top unique-domain pages in parallel
      if (readTopN > 0 && res.results.length > 0 && (options?.searchViaChromiumSerp || options?.openHiddenTabAndExtract || call.arguments.readTopN !== undefined)) {
        const pickedUrls: string[] = [];
        const seenDomains = new Set<string>();
        for (const item of res.results) {
          if (pickedUrls.length >= readTopN) break;
          if (!seenDomains.has(item.domain)) {
            seenDomains.add(item.domain);
            pickedUrls.push(item.url);
          }
        }
        if (pickedUrls.length > 0) {
          const scraped = await runFetchAndReadUrls(pickedUrls, {
            query: queryStr,
            maxCharsPerSource: 2600,
            concurrency: 3,
            onSourceReady: options?.onSourceReady,
            deps: {
              hasPermission: async () => true,
              fetchFn: options?.fetchFn,
              openHiddenTabAndExtract: options?.openHiddenTabAndExtract,
              timeoutMs: 6500,
            },
          });
          const usable = scraped.filter((s) => !s.error && s.excerpt.trim().length > 0);
          res.scrapedSources = usable;
          if (usable.length > 0) {
            const excerptBlock = usable
              .map((s) => `### ${s.title || s.url}\nURL: ${s.url}\n\n${s.excerpt}`)
              .join('\n\n---\n\n');
            return {
              ok: true,
              toolName: 'web_search',
              data: res,
              formatted: sanitizeUntrustedText(`${formattedResults}\n\n## Top Source Excerpts\n\n${excerptBlock}`).cleanText,
            };
          }
        }
      }

      return {
        ok: true,
        toolName: 'web_search',
        data: res,
        formatted: sanitizeUntrustedText(formattedResults).cleanText,
      };
    }

    case 'fetch_and_read_url': {
      const granted = await checkPerm();
      if (!granted) {
        return {
          ok: false,
          toolName: 'fetch_and_read_url',
          needsPermission: true,
          error: 'Optional host permission (https://*/*) is required for fetch_and_read_url.',
        };
      }
      const fetcherDeps: UrlFetcherDeps = {
        hasPermission: async () => true,
        fetchFn: options?.fetchFn,
        openHiddenTabAndExtract: options?.openHiddenTabAndExtract,
      };
      const sources = await runFetchAndReadUrls(call.arguments.urls as string[], {
        query: call.arguments.query as string | undefined,
        maxCharsPerSource: (call.arguments.maxCharsPerSource as number) ?? 4800,
        onSourceReady: options?.onSourceReady,
        deps: fetcherDeps,
      });

      const anySuccess = sources.some((s) => !s.error && s.excerpt.length > 0);
      const captchaBlocked = sources.find((s) => s.blockedByCaptcha);
      const rawFormatted = sources
        .map((s) => {
          if (s.error) {
            return `### ${s.url}\nError: ${s.error}${s.manualUrl ? ` (Manual URL: ${s.manualUrl})` : ''}`;
          }
          return `### ${s.title || s.url}\nURL: ${s.url}\n\n${s.excerpt}`;
        })
        .join('\n\n---\n\n');
      const formatted = sanitizeUntrustedText(rawFormatted).cleanText;

      return {
        ok: anySuccess,
        toolName: 'fetch_and_read_url',
        data: sources,
        formatted,
        ...(captchaBlocked && !anySuccess
          ? {
              blockedByCaptcha: true,
              manualUrl: captchaBlocked.manualUrl,
              error: captchaBlocked.error,
            }
          : !anySuccess
            ? { error: sources[0]?.error ?? 'Failed to fetch sources.' }
            : {}),
      };
    }
  }
}
