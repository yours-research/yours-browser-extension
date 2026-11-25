const OFFSCREEN_PATH = 'offscreen.html';

let creatingOffscreen: Promise<void> | null = null;

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

chrome.runtime.onInstalled.addListener(() => {
  void configureSidePanelBehavior();
});

chrome.runtime.onStartup.addListener(() => {
  void configureSidePanelBehavior();
});

void configureSidePanelBehavior();

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (
    typeof message === 'object' &&
    message !== null &&
    'type' in message &&
    (message as { type: string }).type === 'ENSURE_OFFSCREEN'
  ) {
    ensureOffscreenDocument()
      .then(() => sendResponse({ ok: true }))
      .catch((err: unknown) => {
        const errorMessage = err instanceof Error ? err.message : String(err);
        sendResponse({ ok: false, error: errorMessage });
      });
    return true;
  }
  return false;
});
