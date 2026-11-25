import { rankPassagesBm25 } from './bm25Ranker.ts';
import { runJsCalc } from './jsCalc.ts';
import { extractTabContent } from './pageExtractor.ts';
import { validateToolCall } from './toolProtocol.ts';
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
  return extractTabContent(tabId, { mode: 'auto', maxChars });
}

export interface ExecuteToolOptions {
  hasPermission?: () => Promise<boolean>;
  searchViaChromiumSerp?: WebSearchDeps['searchViaChromiumSerp'];
  fetchFn?: typeof fetch;
  openHiddenTabAndExtract?: (url: string) => Promise<HiddenTabHtmlResponse>;
  onSourceReady?: (src: FetchedSourceResult) => void;
}

export async function executeToolCall(
  rawCall: { name: string; arguments?: Record<string, unknown> },
  options?: ExecuteToolOptions,
): Promise<ToolExecutionResult> {
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
      const formatted = query
        ? `[${page.mode.toUpperCase()}: MATCHED FOR "${query}"] ${page.title}\nURL: ${page.url}\n\n${rankPassagesBm25(searchable, query, maxChars).excerpt}`
        : `[${page.mode.toUpperCase()}] ${page.title}\nURL: ${page.url}\n\n${page.content}`;
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
      const formatted = query
        ? `[${page.mode.toUpperCase()}: MATCHED FOR "${query}"] ${page.title}\nURL: ${page.url}\n\n${rankPassagesBm25(searchable, query, maxChars).excerpt}`
        : `[${page.mode.toUpperCase()}] ${page.title}\nURL: ${page.url}\n\n${page.content}`;
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
              formatted: `${formattedResults}\n\n## Top Source Excerpts\n\n${excerptBlock}`,
            };
          }
        }
      }

      return { ok: true, toolName: 'web_search', data: res, formatted: formattedResults };
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
      const formatted = sources
        .map((s) => {
          if (s.error) {
            return `### ${s.url}\nError: ${s.error}${s.manualUrl ? ` (Manual URL: ${s.manualUrl})` : ''}`;
          }
          return `### ${s.title || s.url}\nURL: ${s.url}\n\n${s.excerpt}`;
        })
        .join('\n\n---\n\n');

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
