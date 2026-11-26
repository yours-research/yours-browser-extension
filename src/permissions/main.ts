import { completeMv3MicOnboarding } from '@yours/voice/adapters';

const statusEl = document.getElementById('perm-status');
const grantBtn = document.getElementById('grant-mic-btn') as HTMLButtonElement | null;

async function runOnboarding(): Promise<void> {
  if (statusEl) {
    statusEl.textContent = 'Requesting microphone access...';
  }
  const result = await completeMv3MicOnboarding();
  if (statusEl) {
    statusEl.textContent = result.granted
      ? 'Microphone granted! You can return to the Yours side panel.'
      : `Microphone unavailable: ${result.error ?? 'permission denied'}`;
  }
  if (result.granted && window.top === window && typeof chrome !== 'undefined' && chrome.tabs?.getCurrent) {
    chrome.tabs.getCurrent((tab) => {
      if (tab?.id !== undefined) {
        setTimeout(() => {
          void chrome.tabs.remove(tab.id!).catch(() => {});
        }, 600);
      }
    });
  }
}

grantBtn?.addEventListener('click', () => {
  void runOnboarding();
});

// Trigger automatically on load inside extension iframe/tab onboarding flow.
void runOnboarding();
