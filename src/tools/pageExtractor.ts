import { parseYouTubeTranscriptXml } from './htmlDistiller.ts';
import type { ExtractedPageContent, PageExtractMode } from './types.ts';

/**
 * Self-contained page extraction function injected via `chrome.scripting.executeScript`.
 * IMPORTANT: Must NEVER reference outer-scope imports, helpers, or variables.
 */
export function inPageExtractScript(mode: PageExtractMode, maxChars: number): ExtractedPageContent {
  const fnv1aHash = (input: string): string => {
    let hash = 0x811c9dc5;
    for (let i = 0; i < input.length; i++) {
      hash ^= input.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
  };

  const truncateClean = (text: string, limit: number): { text: string; truncated: boolean } => {
    if (text.length <= limit) {
      return { text, truncated: false };
    }
    if (limit <= 0) {
      return { text: '', truncated: text.length > 0 };
    }
    const slice = text.slice(0, limit);
    const lastBreak = Math.max(slice.lastIndexOf('\n'), slice.lastIndexOf(' '));
    const cutoff = lastBreak > limit * 0.65 ? lastBreak : limit;
    return {
      text: `${slice.slice(0, cutoff).trimEnd()}\n… [truncated]`,
      truncated: true,
    };
  };

  const pageTitle = (document.title || '').replace(/\s+/g, ' ').trim();
  const pageUrl = location.href || '';
  const safeMaxChars = typeof maxChars === 'number' && maxChars > 0 ? maxChars : 8000;

  try {
    // 1. Selection check (auto or explicit selection)
    if (mode === 'auto' || mode === 'selection') {
      const selectionText = (window.getSelection?.()?.toString() ?? '').trim();
      if (selectionText.length >= 10 || mode === 'selection') {
        const { text, truncated } = truncateClean(selectionText, safeMaxChars);
        return {
          mode: 'selection',
          title: pageTitle,
          url: pageUrl,
          content: text,
          contentHash: fnv1aHash(selectionText),
          charCount: selectionText.length,
          truncated,
        };
      }
    }

    // 2. Interactive snapshot mode
    if (mode === 'interactive') {
      const selector = 'a, button, input, select, textarea, [role="button"], [role="link"]';
      const nodes = Array.from(document.querySelectorAll(selector));
      const lines: string[] = [];
      let idCounter = 1;

      for (const el of nodes) {
        const htmlEl = el as HTMLElement;
        const style = window.getComputedStyle(htmlEl);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
          continue;
        }
        const rect = htmlEl.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
          continue;
        }

        const tag = htmlEl.tagName.toLowerCase();
        const inputEl = htmlEl as HTMLInputElement;
        const explicitRole = htmlEl.getAttribute('role');
        const role =
          explicitRole ||
          (tag === 'a'
            ? 'link'
            : tag === 'input'
              ? `input:${inputEl.type || 'text'}`
              : tag);

        const rawName =
          htmlEl.getAttribute('aria-label') ||
          htmlEl.innerText ||
          inputEl.placeholder ||
          htmlEl.getAttribute('title') ||
          inputEl.name ||
          htmlEl.getAttribute('alt') ||
          '';
        const cleanName = rawName.replace(/[\r\n|]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80) || '-';

        const states: string[] = [];
        if ('disabled' in inputEl && inputEl.disabled) states.push('disabled');
        if (htmlEl.getAttribute('aria-disabled') === 'true') states.push('disabled');
        if ('checked' in inputEl && (inputEl.type === 'checkbox' || inputEl.type === 'radio')) {
          states.push(inputEl.checked ? 'checked' : 'unchecked');
        }
        if (tag === 'input' && inputEl.type !== 'password' && inputEl.type !== 'checkbox' && inputEl.type !== 'radio' && inputEl.value) {
          states.push(`val=${inputEl.value.slice(0, 30)}`);
        }
        if (tag === 'select') {
          const sel = htmlEl as HTMLSelectElement;
          const opt = sel.options[sel.selectedIndex];
          if (opt) states.push(`val=${opt.text.slice(0, 30)}`);
        }
        const stateStr = states.length > 0 ? states.join(',') : '-';

        const cx = Math.round(rect.left + rect.width / 2);
        const cy = Math.round(rect.top + rect.height / 2);
        lines.push(`[${idCounter} | ${role} | ${cleanName} | ${stateStr} | (${cx},${cy})]`);
        idCounter++;
      }

      const rawInteractive = lines.join('\n');
      const { text, truncated } = truncateClean(rawInteractive, safeMaxChars);
      return {
        mode: 'interactive',
        title: pageTitle,
        url: pageUrl,
        content: text,
        contentHash: fnv1aHash(rawInteractive),
        charCount: rawInteractive.length,
        truncated,
      };
    }

    // 3. YouTube detection (DOM transcript or ytInitialPlayerResponse captionTracks)
    const host = location.hostname.toLowerCase();
    if (host === 'youtube.com' || host.endsWith('.youtube.com') || host === 'youtu.be') {
      const transcriptSegments = Array.from(document.querySelectorAll('ytd-transcript-segment-renderer'));
      let inlineTranscript = '';
      if (transcriptSegments.length > 0) {
        const segLines: string[] = [];
        for (const seg of transcriptSegments) {
          const ts = (seg.querySelector('.segment-timestamp')?.textContent ?? '').trim();
          const txt = (seg.querySelector('.segment-text')?.textContent ?? '').replace(/\s+/g, ' ').trim();
          if (txt) {
            segLines.push(ts ? `[${ts}] ${txt}` : txt);
          }
        }
        inlineTranscript = segLines.join('\n');
      }

      let captionBaseUrl = '';
      let languageCode = '';

      type PlayerTrack = { baseUrl?: string; languageCode?: string; kind?: string };
      type PlayerResp = {
        captions?: { playerCaptionsTracklistRenderer?: { captionTracks?: PlayerTrack[] } };
      };

      const winResp = (window as unknown as { ytInitialPlayerResponse?: PlayerResp }).ytInitialPlayerResponse;
      let tracks: PlayerTrack[] =
        winResp?.captions?.playerCaptionsTracklistRenderer?.captionTracks ?? [];

      if (tracks.length === 0) {
        const scripts = Array.from(document.querySelectorAll('script'));
        for (const s of scripts) {
          const srcText = s.textContent ?? '';
          if (srcText.includes('captionTracks')) {
            const m = /"captionTracks"\s*:\s*(\[[\s\S]*?\])/.exec(srcText);
            if (m) {
              try {
                tracks = JSON.parse(m[1]) as PlayerTrack[];
                if (tracks.length > 0) break;
              } catch {
                // ignore malformed script chunk
              }
            }
          }
        }
      }

      const validTracks = tracks.filter(
        (t) => typeof t.baseUrl === 'string' && t.baseUrl.length > 0 && typeof t.languageCode === 'string',
      );
      if (validTracks.length > 0) {
        const isEn = (code: string) => /^en(?:[-_]|$)/i.test(code);
        const picked =
          validTracks.find((t) => isEn(t.languageCode!) && t.kind !== 'asr') ??
          validTracks.find((t) => isEn(t.languageCode!)) ??
          validTracks.find((t) => t.kind !== 'asr') ??
          validTracks[0];
        captionBaseUrl = (picked.baseUrl ?? '').replace(/\\u0026/g, '&');
        languageCode = picked.languageCode ?? '';
      }

      if (inlineTranscript || captionBaseUrl) {
        const { text, truncated } = truncateClean(inlineTranscript, safeMaxChars);
        return {
          mode: 'youtube',
          title: pageTitle.replace(/\s*-\s*YouTube$/i, '').trim(),
          url: pageUrl,
          content: text,
          contentHash: fnv1aHash(inlineTranscript || captionBaseUrl),
          charCount: inlineTranscript.length,
          truncated,
          youtubeCaptionUrl: captionBaseUrl || undefined,
          youtubeLanguageCode: languageCode || undefined,
        };
      }
    }

    // 4. PDF.js textLayer detection
    const textLayers = Array.from(document.querySelectorAll('.textLayer'));
    if (textLayers.length > 0) {
      const pdfText = textLayers
        .map((layer) => (layer.textContent ?? '').replace(/\s+/g, ' ').trim())
        .filter(Boolean)
        .join('\n\n');
      if (pdfText.length >= 30) {
        const { text, truncated } = truncateClean(pdfText, safeMaxChars);
        return {
          mode: 'pdf',
          title: pageTitle,
          url: pageUrl,
          content: text,
          contentHash: fnv1aHash(pdfText),
          charCount: pdfText.length,
          truncated,
        };
      }
    }

    // 5. Article / main / body readable extraction
    const articleEl = document.querySelector('article');
    const mainEl = document.querySelector('main');
    const bodyEl = document.body || document.documentElement;

    const getLen = (el: Element | null) =>
      el ? ((el as HTMLElement).innerText || el.textContent || '').trim().length : 0;

    let sourceEl: Element | null = bodyEl;
    if (getLen(articleEl) >= 150) {
      sourceEl = articleEl;
    } else if (getLen(mainEl) >= 150) {
      sourceEl = mainEl;
    }

    if (!sourceEl) {
      return {
        mode: 'article',
        title: pageTitle,
        url: pageUrl,
        content: '',
        contentHash: fnv1aHash(''),
        charCount: 0,
        truncated: false,
      };
    }

    const cloned = sourceEl.cloneNode(true) as Element;
    const junkNodes = cloned.querySelectorAll('script, style, nav, aside, footer, noscript, svg');
    for (const node of Array.from(junkNodes)) {
      node.remove();
    }

    const parts: string[] = [];
    const blockTags = new Set([
      'P',
      'DIV',
      'SECTION',
      'ARTICLE',
      'MAIN',
      'BLOCKQUOTE',
      'UL',
      'OL',
      'TABLE',
      'TR',
      'HR',
    ]);

    const walk = (node: Node): void => {
      if (node.nodeType === Node.TEXT_NODE) {
        const txt = (node.textContent ?? '').replace(/\s+/g, ' ');
        if (txt.trim()) {
          parts.push(txt);
        }
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) {
        return;
      }
      const el = node as Element;
      const tag = el.tagName.toUpperCase();

      if (tag === 'BR') {
        parts.push('\n');
        return;
      }
      if (tag === 'PRE') {
        const code = (el.textContent ?? '').trim();
        if (code) {
          parts.push(`\n\n\`\`\`\n${code}\n\`\`\`\n\n`);
        }
        return;
      }
      if (/^H[1-6]$/.test(tag)) {
        const level = Number.parseInt(tag[1], 10);
        const hText = (el.textContent ?? '').replace(/\s+/g, ' ').trim();
        if (hText) {
          parts.push(`\n\n${'#'.repeat(level)} ${hText}\n\n`);
        }
        return;
      }
      if (tag === 'LI') {
        const liText = (el.textContent ?? '').replace(/\s+/g, ' ').trim();
        if (liText) {
          parts.push(`\n- ${liText}`);
        }
        return;
      }

      const isBlock = blockTags.has(tag);
      if (isBlock) parts.push('\n');
      for (const child of Array.from(el.childNodes)) {
        walk(child);
      }
      if (isBlock) parts.push('\n');
    };

    walk(cloned);

    const normalized = parts
      .join('')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n[ \t]+/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    const { text, truncated } = truncateClean(normalized, safeMaxChars);
    return {
      mode: 'article',
      title: pageTitle,
      url: pageUrl,
      content: text,
      contentHash: fnv1aHash(normalized),
      charCount: normalized.length,
      truncated,
    };
  } catch (err: unknown) {
    return {
      mode: mode === 'interactive' ? 'interactive' : mode === 'selection' ? 'selection' : 'article',
      title: pageTitle,
      url: pageUrl,
      content: '',
      contentHash: '00000000',
      charCount: 0,
      truncated: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function hasOptionalHostPermission(): Promise<boolean> {
  if (typeof chrome === 'undefined' || !chrome.permissions?.contains) {
    return false;
  }
  try {
    return await chrome.permissions.contains({ origins: ['https://*/*'] });
  } catch {
    return false;
  }
}

export async function extractTabContent(
  tabId: number,
  options?: {
    mode?: PageExtractMode;
    maxChars?: number;
    fetchCaptionXml?: (url: string) => Promise<string>;
  },
): Promise<ExtractedPageContent> {
  const mode: PageExtractMode = options?.mode ?? 'auto';
  const maxChars = options?.maxChars ?? 8000;

  try {
    const injections = await chrome.scripting.executeScript({
      target: { tabId },
      func: inPageExtractScript,
      args: [mode, maxChars],
    });

    const extracted = injections?.[0]?.result as ExtractedPageContent | undefined;
    if (!extracted) {
      return {
        mode: mode === 'interactive' ? 'interactive' : mode === 'selection' ? 'selection' : 'article',
        title: '',
        url: '',
        content: '',
        contentHash: '00000000',
        charCount: 0,
        truncated: false,
        error: 'Page script returned no result (restricted chrome:// or extension page?).',
      };
    }

    // If YouTube page had no inline DOM transcript, fetch caption XML only if permitted
    if (extracted.mode === 'youtube' && extracted.youtubeCaptionUrl && !extracted.content) {
      const canFetch = options?.fetchCaptionXml ? true : await hasOptionalHostPermission();
      if (canFetch) {
        try {
          const xml = options?.fetchCaptionXml
            ? await options.fetchCaptionXml(extracted.youtubeCaptionUrl)
            : await (await fetch(extracted.youtubeCaptionUrl)).text();
          const transcript = parseYouTubeTranscriptXml(xml, maxChars);
          if (transcript) {
            return {
              ...extracted,
              content: transcript,
              charCount: transcript.length,
              truncated: transcript.endsWith('… [truncated]'),
            };
          }
        } catch {
          // Offline or caption fetch failed — keep extracted metadata cleanly
        }
      }
    }

    return extracted;
  } catch (err: unknown) {
    return {
      mode: mode === 'interactive' ? 'interactive' : mode === 'selection' ? 'selection' : 'article',
      title: '',
      url: '',
      content: '',
      contentHash: '00000000',
      charCount: 0,
      truncated: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
