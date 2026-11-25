import {
  checkResearchPermission,
  executeToolCall,
  extractActiveTabPreview,
  requestResearchPermission,
} from '../tools/browserTools.ts';
import type { HiddenTabHtmlResponse } from '../tools/types.ts';

const OFFSCREEN_PATH = 'offscreen.html';
const HIDDEN_TAB_IDLE_MS = 30_000;
const HIDDEN_TAB_NAV_TIMEOUT_MS = 12_000;

let creatingOffscreen: Promise<void> | null = null;
let hiddenResearchTabId: number | null = null;
let hiddenTabIdleTimer: ReturnType<typeof setTimeout> | null = null;
let hiddenTabLock: Promise<unknown> = Promise.resolve();

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

  creatingOffscreen = chrome.offscreen
    .createDocument({
      url: OFFSCREEN_PATH,
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification:
        'Host persistent WebGPU LLM inference engine, elevated GPU device, and OPFS weight cache.',
    })
    .finally(() => {
      creatingOffscreen = null;
    });

  await creatingOffscreen;
}

function scheduleHiddenTabCleanup(): void {
  if (hiddenTabIdleTimer !== null) {
    clearTimeout(hiddenTabIdleTimer);
  }
  hiddenTabIdleTimer = setTimeout(() => {
    hiddenTabIdleTimer = null;
    const tabId = hiddenResearchTabId;
    hiddenResearchTabId = null;
    if (tabId !== null) {
      void chrome.tabs.remove(tabId).catch(() => {});
    }
  }, HIDDEN_TAB_IDLE_MS);
}

async function getOrCreateHiddenResearchTab(): Promise<number> {
  if (hiddenResearchTabId !== null) {
    try {
      await chrome.tabs.get(hiddenResearchTabId);
      return hiddenResearchTabId;
    } catch {
      hiddenResearchTabId = null;
    }
  }
  const created = await chrome.tabs.create({ url: 'about:blank', active: false });
  if (created.id === undefined) {
    throw new Error('Failed to create hidden background research tab.');
  }
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

function readHiddenTabDomScript(): HiddenTabHtmlResponse {
  return {
    html: document.documentElement ? document.documentElement.outerHTML : '',
    title: document.title || '',
    url: location.href || '',
  };
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
    await navigateHiddenTabAndWait(tabId, parsed.toString(), HIDDEN_TAB_NAV_TIMEOUT_MS);

    const results = await chrome.scripting.executeScript({
      target: { tabId },
      func: readHiddenTabDomScript,
    });

    scheduleHiddenTabCleanup();

    const payload = results?.[0]?.result as HiddenTabHtmlResponse | undefined;
    return payload ?? { html: '', title: '', url: parsed.toString() };
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

  const msg = message as { type: string; call?: { name: string; arguments?: Record<string, unknown> } };

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
    executeToolCall(msg.call, { openHiddenTabAndExtract })
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
