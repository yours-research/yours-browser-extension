import {
  checkResearchPermission,
  extractActiveTabPreview,
  requestResearchPermission,
} from '../tools/browserTools.ts';
import { inPageExtractScript } from '../tools/pageExtractor.ts';
import type { ChromiumSerpResponse, HiddenTabHtmlResponse, SearchResultItem } from '../tools/types.ts';
import { unwrapBingRedirectUrl } from '../tools/webSearch.ts';
import { createExtensionToolRuntime, type ExtensionToolRuntime } from '../tools/yoursToolRuntime.ts';

const OFFSCREEN_PATH = 'offscreen.html';
const HIDDEN_TAB_NAV_TIMEOUT_MS = 10_000;
const SERP_NAV_TIMEOUT_MS = 7_500;

let creatingOffscreen: Promise<void> | null = null;
let hiddenResearchWindowId: number | null = null;
let hiddenResearchTabId: number | null = null;
let hiddenTabIdleTimer: ReturnType<typeof setTimeout> | null = null;
let hiddenTabLock: Promise<unknown> = Promise.resolve();
let swToolRuntime: ExtensionToolRuntime | null = null;

function getSwToolRuntime(): ExtensionToolRuntime {
  if (!swToolRuntime) {
    swToolRuntime = createExtensionToolRuntime({
      adapter: {
        preferHeadlessFetch: true,
        defaultEngine: 'ddg',
        searchViaChromiumSerp: runChromiumSerpSearch,
        openHiddenTabAndExtract,
      },
    });
  }
  return swToolRuntime;
}

async function configureSidePanelBehavior(): Promise<void> {
  try {
    await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  } catch (err) {
    console.warn('[TinyGPT] Failed to set sidePanel behavior:', err);
  }
}

export async function ensureOffscreenDocument(): Promise<void> {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_PATH);
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    documentUrls: [offscreenUrl],
  });

  if (existingContexts.length > 0) {
    return;
  }

  if (creatingOffscreen) {
    await creatingOffscreen;
    return;
  }

  const requestedReasons = [
    chrome.offscreen.Reason.WORKERS,
    chrome.offscreen.Reason.USER_MEDIA,
    chrome.offscreen.Reason.AUDIO_PLAYBACK,
    chrome.offscreen.Reason.DOM_PARSER,
  ];

  creatingOffscreen = (async () => {
    try {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_PATH,
        reasons: requestedReasons,
        justification:
          'Host persistent WebGPU LLM inference engine, local audio/voice pipeline, elevated GPU device, and OPFS weight cache.',
      });
    } catch {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_PATH,
        reasons: [chrome.offscreen.Reason.WORKERS],
        justification:
          'Host persistent WebGPU LLM inference engine, elevated GPU device, and OPFS weight cache.',
      });
    }
  })().finally(() => {
    creatingOffscreen = null;
  });

  await creatingOffscreen;
}

function closeHiddenResearchTabNow(): void {
  if (hiddenTabIdleTimer !== null) {
    clearTimeout(hiddenTabIdleTimer);
    hiddenTabIdleTimer = null;
  }
  const tabId = hiddenResearchTabId;
  hiddenResearchWindowId = null;
  hiddenResearchTabId = null;
  if (tabId !== null) {
    void chrome.tabs.remove(tabId).catch(() => {});
  }
}

async function getOrCreateHiddenResearchTab(): Promise<number> {
  if (hiddenResearchTabId !== null) {
    try {
      await chrome.tabs.get(hiddenResearchTabId);
      return hiddenResearchTabId;
    } catch {
      hiddenResearchTabId = null;
      hiddenResearchWindowId = null;
    }
  }

  const created = await chrome.tabs.create({ url: 'about:blank', active: false });
  if (created.id === undefined) {
    throw new Error('Failed to create inactive background research tab.');
  }
  hiddenResearchWindowId = created.windowId ?? null;
  hiddenResearchTabId = created.id;
  return created.id;
}

async function navigateHiddenTabAndWait(tabId: number, url: string, timeoutMs: number): Promise<void> {
  let settled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let onUpdatedRef: ((updatedTabId: number, changeInfo: { status?: string }) => void) | null = null;
  let onRemovedRef: ((removedTabId: number) => void) | null = null;

  const cleanup = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (onUpdatedRef) {
      chrome.tabs.onUpdated.removeListener(onUpdatedRef);
      onUpdatedRef = null;
    }
    if (onRemovedRef) {
      chrome.tabs.onRemoved.removeListener(onRemovedRef);
      onRemovedRef = null;
    }
  };

  const completionPromise = new Promise<void>((resolve, reject) => {
    timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('Hidden tab navigation timed out.'));
    }, timeoutMs);

    onUpdatedRef = (updatedTabId: number, changeInfo: { status?: string }) => {
      if (updatedTabId === tabId && changeInfo.status === 'complete') {
        if (settled) return;
        settled = true;
        cleanup();
        resolve();
      }
    };

    onRemovedRef = (removedTabId: number) => {
      if (removedTabId === tabId) {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error('Hidden research tab was closed during navigation.'));
      }
    };

    chrome.tabs.onUpdated.addListener(onUpdatedRef);
    chrome.tabs.onRemoved.addListener(onRemovedRef);
  });

  try {
    await chrome.tabs.update(tabId, { url, active: false });
  } catch (err) {
    if (!settled) {
      settled = true;
      cleanup();
    }
    throw err;
  }

  await completionPromise;
}

function readHiddenTabDomScript(): Promise<HiddenTabHtmlResponse> {
  return new Promise((resolve) => {
    let settled = false;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    const finish = () => {
      if (settled) return;
      settled = true;
      observer.disconnect();
      if (idleTimer !== null) clearTimeout(idleTimer);
      clearTimeout(hardTimer);
      resolve({
        html: document.documentElement ? document.documentElement.outerHTML : '',
        title: document.title || '',
        url: location.href || '',
      });
    };
    const bump = () => {
      if (idleTimer !== null) clearTimeout(idleTimer);
      idleTimer = setTimeout(finish, 350);
    };
    const observer = new MutationObserver(() => bump());
    if (document.body) {
      observer.observe(document.body, { childList: true, subtree: true });
    }
    const hardTimer = setTimeout(finish, 900);
    bump();
  });
}

/**
 * Self-contained in-page SERP extractor injected into Chrome's minimized research window.
 * Never clicks, solves, or bypasses CAPTCHAs; reports anti-bot challenges cleanly.
 */
function inPageSerpExtractScript(
  engine: 'bing' | 'ddg' | 'startpage',
  limit: number,
): Promise<{
  blockedByCaptcha: boolean;
  rawResults: Array<{ title: string; href: string; snippet: string }>;
}> {
  const maxItems = Math.max(1, Math.min(8, Math.round(limit)));
  const extractNow = (): {
    blockedByCaptcha: boolean;
    rawResults: Array<{ title: string; href: string; snippet: string }>;
  } => {
    const items: Array<{ title: string; href: string; snippet: string }> = [];
    const seen = new Set<string>();

    const pushItem = (title: string, href: string, snippet: string) => {
      const cleanTitle = title.replace(/\s+/g, ' ').trim();
      const cleanHref = href.trim();
      const cleanSnippet = snippet.replace(/\s+/g, ' ').trim();
      if (!cleanTitle || !cleanHref || seen.has(cleanHref)) return;
      seen.add(cleanHref);
      items.push({ title: cleanTitle, href: cleanHref, snippet: cleanSnippet });
    };

    if (engine === 'bing') {
      const nodes = Array.from(document.querySelectorAll('li.b_algo'));
      for (const node of nodes) {
        if (items.length >= maxItems) break;
        if (node.closest('.b_ad') || node.classList.contains('b_ad')) continue;
        const link = node.querySelector('h2 a') as HTMLAnchorElement | null;
        if (!link) continue;
        const snippetEl =
          node.querySelector('.b_caption p') ??
          node.querySelector('.b_algoSlug') ??
          node.querySelector('p');
        pushItem(link.textContent ?? '', link.href || link.getAttribute('href') || '', snippetEl?.textContent ?? '');
      }
    } else if (engine === 'ddg') {
      const nodes = Array.from(
        document.querySelectorAll('article[data-testid="result"], [data-nrn="result"]'),
      );
      for (const node of nodes) {
        if (items.length >= maxItems) break;
        const link = (node.querySelector('a[data-testid="result-title-a"]') ??
          node.querySelector('h2 a')) as HTMLAnchorElement | null;
        if (!link) continue;
        const snippetEl = node.querySelector('[data-result="snippet"]');
        pushItem(link.textContent ?? '', link.href || link.getAttribute('href') || '', snippetEl?.textContent ?? '');
      }
    } else {
      const nodes = Array.from(document.querySelectorAll('.w-gl__result, .result'));
      for (const node of nodes) {
        if (items.length >= maxItems) break;
        const link = (node.querySelector('a.w-gl__result-title') ??
          node.querySelector('a.result-title') ??
          node.querySelector('h3 a, a[href^="http"]')) as HTMLAnchorElement | null;
        if (!link) continue;
        const snippetEl = node.querySelector('.w-gl__description, p');
        pushItem(link.textContent ?? '', link.href || link.getAttribute('href') || '', snippetEl?.textContent ?? '');
      }
    }

    const headSample = `${document.title} ${(document.body?.innerText ?? '').slice(0, 1200)} ${location.href}`;
    const hasChallenge =
      items.length === 0 &&
      /unusual traffic|verify you are human|are you a robot|anomaly-modal|anomaly\.js|complete the security check|sorry\/index|\/captcha/i.test(
        headSample,
      );

    return {
      blockedByCaptcha: hasChallenge,
      rawResults: items,
    };
  };

  return new Promise((resolve) => {
    const first = extractNow();
    if (first.blockedByCaptcha || first.rawResults.length >= Math.min(3, maxItems)) {
      resolve(first);
      return;
    }

    let settled = false;
    const finish = (snap = extractNow()) => {
      if (settled) return;
      settled = true;
      observer.disconnect();
      clearInterval(poller);
      clearTimeout(deadline);
      resolve(snap);
    };

    const observer = new MutationObserver(() => {
      const snap = extractNow();
      if (snap.blockedByCaptcha || snap.rawResults.length >= maxItems) {
        finish(snap);
      }
    });
    if (document.documentElement) {
      observer.observe(document.documentElement, { childList: true, subtree: true });
    }
    const poller = setInterval(() => {
      const snap = extractNow();
      if (snap.blockedByCaptcha || snap.rawResults.length >= Math.min(3, maxItems)) {
        finish(snap);
      }
    }, 120);
    const deadline = setTimeout(() => finish(), 4000);
  });
}

function normalizeSerpResults(
  engine: 'bing' | 'ddg' | 'startpage',
  rawResults: Array<{ title: string; href: string; snippet: string }>,
  limit: number,
): SearchResultItem[] {
  const out: SearchResultItem[] = [];
  const seen = new Set<string>();
  for (const item of rawResults) {
    if (out.length >= limit) break;
    const resolved =
      engine === 'bing' ? (unwrapBingRedirectUrl(item.href) ?? item.href) : item.href;
    try {
      const u = new URL(resolved);
      if (u.protocol !== 'http:' && u.protocol !== 'https:') continue;
      if (u.hostname.endsWith('bing.com') && u.pathname.startsWith('/ck')) continue;
      if (u.hostname.endsWith('duckduckgo.com') && u.pathname.startsWith('/y.js')) continue;
      if (u.hostname.endsWith('startpage.com') && u.pathname.startsWith('/sp/')) continue;
      const finalUrl = u.toString();
      if (seen.has(finalUrl)) continue;
      seen.add(finalUrl);
      out.push({
        title: item.title,
        url: finalUrl,
        snippet: item.snippet,
        domain: u.hostname.replace(/^www\./, ''),
      });
    } catch {
      // ignore invalid URL
    }
  }
  return out;
}

async function runChromiumSerpSearch(
  query: string,
  limit = 5,
): Promise<ChromiumSerpResponse> {
  const run = async (): Promise<ChromiumSerpResponse> => {
    const startMs = Date.now();
    const granted = await checkResearchPermission();
    if (!granted) {
      throw new Error('Optional host permission (https://*/*) is required for Chromium SERP search.');
    }

    const clampedLimit = Math.max(1, Math.min(8, Math.round(limit)));
    const q = encodeURIComponent(query);
    const engines: Array<{ engine: 'bing' | 'ddg' | 'startpage'; url: string }> = [
      { engine: 'ddg', url: `https://duckduckgo.com/?q=${q}` },
      { engine: 'bing', url: `https://www.bing.com/search?q=${q}` },
      { engine: 'startpage', url: `https://www.startpage.com/do/search?q=${q}` },
    ];

    if (hiddenTabIdleTimer !== null) {
      clearTimeout(hiddenTabIdleTimer);
      hiddenTabIdleTimer = null;
    }

    const tabId = await getOrCreateHiddenResearchTab();
    let anyBlocked = false;
    let lastError = 'No organic results found.';

    try {
      for (const candidate of engines) {
        try {
          await navigateHiddenTabAndWait(tabId, candidate.url, SERP_NAV_TIMEOUT_MS);
          const injections = await chrome.scripting.executeScript({
            target: { tabId },
            func: inPageSerpExtractScript,
            args: [candidate.engine, clampedLimit],
          });
          const res = injections?.[0]?.result as
            | {
                blockedByCaptcha: boolean;
                rawResults: Array<{ title: string; href: string; snippet: string }>;
              }
            | undefined;
          if (!res) {
            lastError = `${candidate.engine}: script returned no result.`;
            continue;
          }
          if (res.blockedByCaptcha) {
            anyBlocked = true;
            lastError = `${candidate.engine}: anti-bot/CAPTCHA gate encountered.`;
            continue;
          }
          const normalized = normalizeSerpResults(candidate.engine, res.rawResults, clampedLimit);
          if (normalized.length > 0) {
            return {
              ok: true,
              engine: candidate.engine,
              query,
              results: normalized,
              elapsedMs: Date.now() - startMs,
            };
          }
        } catch (err: unknown) {
          lastError = `${candidate.engine}: ${err instanceof Error ? err.message : String(err)}`;
        }
      }

      return {
        ok: false,
        engine: 'ddg',
        query,
        results: [],
        elapsedMs: Date.now() - startMs,
        ...(anyBlocked
          ? {
              blockedByCaptcha: true,
              manualUrl: `https://duckduckgo.com/?q=${q}`,
              error: 'Search engines requested human verification (CAPTCHA). Open manually.',
            }
          : { error: lastError }),
      };
    } finally {
      closeHiddenResearchTabNow();
    }
  };

  const next = hiddenTabLock.then(run, run);
  hiddenTabLock = next.catch(() => {});
  return next;
}

async function openHiddenTabAndExtract(url: string): Promise<HiddenTabHtmlResponse> {
  const run = async (): Promise<HiddenTabHtmlResponse> => {
    // Defense-in-depth: fail closed if optional host permission has not been granted
    const granted = await checkResearchPermission();
    if (!granted) {
      throw new Error('Optional host permission (https://*/*) is required for hidden tab access.');
    }

    const parsed = new URL(url);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Unsupported protocol for hidden research tab.');
    }

    if (hiddenTabIdleTimer !== null) {
      clearTimeout(hiddenTabIdleTimer);
      hiddenTabIdleTimer = null;
    }

    const tabId = await getOrCreateHiddenResearchTab();
    try {
      await navigateHiddenTabAndWait(tabId, parsed.toString(), HIDDEN_TAB_NAV_TIMEOUT_MS);

      const domRes = await chrome.scripting.executeScript({
        target: { tabId },
        func: readHiddenTabDomScript,
      });
      const payload = (domRes?.[0]?.result as HiddenTabHtmlResponse | undefined) ?? {
        html: '',
        title: '',
        url: parsed.toString(),
      };

      // Extract live visibility-aware Firecrawl-grade Markdown inside the hydrated background tab
      let markdown = '';
      let extractedTitle = payload.title ?? '';
      let extractedUrl = payload.url ?? parsed.toString();
      try {
        const mdRes = await chrome.scripting.executeScript({
          target: { tabId },
          func: inPageExtractScript,
          args: ['article', 24000],
        });
        const page = mdRes?.[0]?.result;
        if (page) {
          markdown = page.fullContent || page.content || '';
          extractedTitle = page.title || extractedTitle;
          extractedUrl = page.url || extractedUrl;
        }
      } catch {
        // Fall back to outerHTML distillation if in-page extractor fails
      }

      return {
        html: payload.html,
        title: extractedTitle,
        url: extractedUrl,
        ...(markdown ? { markdown } : {}),
      };
    } finally {
      closeHiddenResearchTabNow();
    }
  };

  // Serialize hidden-tab navigation so concurrent URL fallbacks reuse the single tab safely
  const next = hiddenTabLock.then(run, run);
  hiddenTabLock = next.catch(() => {});
  return next;
}

if (typeof chrome !== 'undefined' && chrome.tabs?.onRemoved) {
  chrome.tabs.onRemoved.addListener((removedTabId) => {
    if (removedTabId === hiddenResearchTabId) {
      hiddenResearchTabId = null;
      hiddenResearchWindowId = null;
    }
  });
}

if (typeof chrome !== 'undefined' && chrome.windows?.onRemoved) {
  chrome.windows.onRemoved.addListener((removedWindowId) => {
    if (removedWindowId === hiddenResearchWindowId) {
      hiddenResearchWindowId = null;
      hiddenResearchTabId = null;
    }
  });
}

chrome.runtime.onInstalled.addListener(() => {
  void configureSidePanelBehavior();
});

chrome.runtime.onStartup.addListener(() => {
  void configureSidePanelBehavior();
});

void configureSidePanelBehavior();

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (typeof message !== 'object' || message === null || !('type' in message)) {
    return false;
  }

  const msg = message as {
    type: string;
    call?: { name: string; arguments?: Record<string, unknown> };
    skipSpotlight?: boolean;
  };

  if (msg.type === 'ENSURE_OFFSCREEN') {
    ensureOffscreenDocument()
      .then(() => sendResponse({ ok: true }))
      .catch((err: unknown) => {
        const errorMessage = err instanceof Error ? err.message : String(err);
        sendResponse({ ok: false, error: errorMessage });
      });
    return true;
  }

  if (msg.type === 'EXEC_TOOL') {
    if (!msg.call || typeof msg.call.name !== 'string') {
      sendResponse({ ok: false, error: 'Missing tool call payload.' });
      return false;
    }
    getSwToolRuntime()
      .execute(msg.call, {
        environment: 'chrome-mv3',
        sessionId: 'sw-exec',
        skipSpotlight: Boolean(msg.skipSpotlight),
      })
      .then((result) => sendResponse(result))
      .catch((err: unknown) => {
        const errorMessage = err instanceof Error ? err.message : String(err);
        sendResponse({ ok: false, toolName: msg.call?.name, error: errorMessage });
      });
    return true;
  }

  if (msg.type === 'CHECK_RESEARCH_PERMISSION') {
    checkResearchPermission()
      .then((granted) => sendResponse({ ok: true, granted }))
      .catch((err: unknown) => {
        sendResponse({ ok: false, granted: false, error: err instanceof Error ? err.message : String(err) });
      });
    return true;
  }

  if (msg.type === 'REQUEST_RESEARCH_PERMISSION') {
    requestResearchPermission()
      .then((granted) => sendResponse({ ok: true, granted }))
      .catch((err: unknown) => {
        sendResponse({ ok: false, granted: false, error: err instanceof Error ? err.message : String(err) });
      });
    return true;
  }

  if (msg.type === 'EXTRACT_ACTIVE_TAB_PREVIEW') {
    extractActiveTabPreview()
      .then((page) => sendResponse({ ok: true, page }))
      .catch((err: unknown) => {
        sendResponse({ ok: false, error: err instanceof Error ? err.message : String(err) });
      });
    return true;
  }

  return false;
});
