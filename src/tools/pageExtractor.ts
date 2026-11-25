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

  const extractMetadata = (): Record<string, string> => {
    const meta: Record<string, string> = {};
    const pick = (key: string, val: unknown) => {
      if (meta[key] || typeof val !== 'string') return;
      const clean = val.replace(/\s+/g, ' ').trim();
      if (clean && clean.length <= 320) meta[key] = clean;
    };

    const metaTags = Array.from(document.querySelectorAll('meta[name], meta[property]'));
    for (const m of metaTags) {
      const k = (m.getAttribute('name') || m.getAttribute('property') || '').toLowerCase();
      const v = m.getAttribute('content') || '';
      if (!v) continue;
      if (k === 'author' || k === 'article:author') pick('author', v);
      else if (k === 'article:published_time' || k === 'date' || k === 'pubdate') pick('published', v);
      else if (k === 'og:site_name') pick('site', v);
      else if (k === 'description' || k === 'og:description') pick('description', v);
    }

    const ldScripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]')).slice(0, 6);
    const visitLd = (obj: unknown, depth = 0) => {
      if (!obj || typeof obj !== 'object' || depth > 4) return;
      if (Array.isArray(obj)) {
        for (const item of obj) visitLd(item, depth + 1);
        return;
      }
      const rec = obj as Record<string, unknown>;
      if (Array.isArray(rec['@graph'])) visitLd(rec['@graph'], depth + 1);
      if (typeof rec.author === 'string') pick('author', rec.author);
      else if (rec.author && typeof rec.author === 'object') {
        pick('author', (rec.author as Record<string, unknown>).name);
      }
      if (typeof rec.datePublished === 'string') pick('published', rec.datePublished);
      if (typeof rec.description === 'string') pick('description', rec.description);
      const offers = Array.isArray(rec.offers) ? rec.offers[0] : rec.offers;
      if (offers && typeof offers === 'object') {
        const off = offers as Record<string, unknown>;
        if (off.price !== undefined && off.price !== null) {
          const cur = typeof off.priceCurrency === 'string' ? `${off.priceCurrency} ` : '';
          pick('price', `${cur}${String(off.price)}`);
        }
      }
      if (rec.aggregateRating && typeof rec.aggregateRating === 'object') {
        const ar = rec.aggregateRating as Record<string, unknown>;
        if (ar.ratingValue !== undefined && ar.ratingValue !== null) {
          pick('rating', `${String(ar.ratingValue)}/5`);
        }
      }
    };

    for (const s of ldScripts) {
      const raw = (s.textContent ?? '').trim();
      if (!raw) continue;
      try {
        visitLd(JSON.parse(raw));
      } catch {
        // ignore invalid JSON-LD
      }
    }
    return meta;
  };

  try {
    const metadata = extractMetadata();
    const rawSelectionText = (window.getSelection?.()?.toString() ?? '').trim();

    // 1. Selection check (auto or explicit selection)
    if (mode === 'auto' || mode === 'selection') {
      if (rawSelectionText.length >= 10 || mode === 'selection') {
        const { text, truncated } = truncateClean(rawSelectionText, safeMaxChars);
        return {
          mode: 'selection',
          title: pageTitle,
          url: pageUrl,
          content: text,
          fullContent: rawSelectionText,
          selectionText: rawSelectionText,
          metadata,
          contentHash: fnv1aHash(rawSelectionText),
          charCount: rawSelectionText.length,
          truncated,
        };
      }
    }

    const isVisibleEl = (el: Element): boolean => {
      const htmlEl = el as HTMLElement;
      if (htmlEl.hidden || el.getAttribute('aria-hidden') === 'true') {
        return false;
      }
      const style = window.getComputedStyle(htmlEl);
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
        return false;
      }
      return true;
    };

    // 2. Interactive snapshot mode (piercing open Shadow DOM)
    if (mode === 'interactive') {
      const selector = 'a, button, input, select, textarea, [role="button"], [role="link"]';
      const collected: Element[] = [];
      const collectInteractive = (root: ParentNode) => {
        for (const el of Array.from(root.querySelectorAll(selector))) {
          collected.push(el);
        }
        for (const hostEl of Array.from(root.querySelectorAll('*'))) {
          const sr = (hostEl as HTMLElement).shadowRoot;
          if (sr) collectInteractive(sr);
        }
      };
      collectInteractive(document);

      const lines: string[] = [];
      let idCounter = 1;

      for (const el of collected) {
        const htmlEl = el as HTMLElement;
        if (!isVisibleEl(htmlEl)) {
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
        fullContent: rawInteractive,
        metadata,
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

    // 5. Article / feed / live visibility-aware Firecrawl-grade Markdown extractor
    const getVisibleLen = (el: Element | null): number => {
      if (!el || !isVisibleEl(el)) return 0;
      return ((el as HTMLElement).innerText || el.textContent || '').trim().length;
    };

    const collectVisibleArticles = (root: ParentNode, acc: Element[]) => {
      for (const a of Array.from(root.querySelectorAll('article'))) {
        if (getVisibleLen(a) >= 60) {
          acc.push(a);
        }
      }
      for (const hostEl of Array.from(root.querySelectorAll('*'))) {
        const sr = (hostEl as HTMLElement).shadowRoot;
        if (sr) collectVisibleArticles(sr, acc);
      }
    };

    const visibleArticles: Element[] = [];
    collectVisibleArticles(document, visibleArticles);

    // Filter out nested articles (e.g. quoted tweet inside outer tweet)
    const topArticles = visibleArticles.filter(
      (a) => !visibleArticles.some((other) => other !== a && other.contains(a)),
    );

    let roots: Element[] = [];
    if (topArticles.length === 1 && getVisibleLen(topArticles[0]) >= 180) {
      roots = [topArticles[0]];
    } else if (topArticles.length > 1) {
      const mainEl = document.querySelector('main, [role="main"]');
      const mainHasArticles =
        mainEl &&
        isVisibleEl(mainEl) &&
        topArticles.filter((a) => mainEl.contains(a)).length >= 2;
      roots = mainHasArticles ? [mainEl] : topArticles;
    } else {
      const candidates = [
        document.querySelector('main'),
        document.querySelector('[role="main"]'),
        document.querySelector('#main, #content, .content'),
        document.body || document.documentElement,
      ].filter((el): el is Element => Boolean(el) && isVisibleEl(el!));

      let bestEl: Element | null = null;
      let bestScore = -1;
      for (const cand of candidates) {
        const txtLen = getVisibleLen(cand);
        if (txtLen < 80 && cand !== document.body && cand !== document.documentElement) continue;
        let linkChars = 0;
        for (const a of Array.from(cand.querySelectorAll('a'))) {
          linkChars += (a.textContent ?? '').trim().length;
        }
        const score = txtLen - linkChars * 0.7;
        if (score > bestScore) {
          bestScore = score;
          bestEl = cand;
        }
      }
      if (bestEl) {
        roots = [bestEl];
      } else if (document.body || document.documentElement) {
        roots = [(document.body || document.documentElement) as Element];
      }
    }

    if (roots.length === 0) {
      return {
        mode: 'article',
        title: pageTitle,
        url: pageUrl,
        content: '',
        fullContent: '',
        metadata,
        contentHash: fnv1aHash(''),
        charCount: 0,
        truncated: false,
      };
    }

    const skipTags = new Set([
      'SCRIPT',
      'STYLE',
      'NOSCRIPT',
      'SVG',
      'TEMPLATE',
      'CANVAS',
      'AUDIO',
      'VIDEO',
      'MAP',
    ]);
    const blockTags = new Set([
      'P',
      'DIV',
      'SECTION',
      'ARTICLE',
      'MAIN',
      'HEADER',
      'HR',
      'UL',
      'OL',
      'DL',
      'DT',
      'DD',
      'FIGCAPTION',
      'FIGURE',
    ]);
    const noiseClassRe =
      /\b(?:cookie[-_]?(?:banner|consent|notice|popup|modal|wall)|gdpr[-_]?(?:banner|consent|modal)|onetrust|cookiebot|ad[-_]?(?:slot|banner|container|unit)|adsbygoogle|newsletter[-_]?(?:popup|banner|modal))\b/i;

    const shouldSkipElement = (el: Element): boolean => {
      const tag = el.tagName.toUpperCase();
      if (skipTags.has(tag)) return true;
      if (!isVisibleEl(el)) return true;
      if (tag === 'NAV' || tag === 'FOOTER' || tag === 'ASIDE') return true;
      const role = (el.getAttribute('role') || '').toLowerCase();
      if (role === 'navigation' || role === 'banner' || role === 'contentinfo' || role === 'complementary') {
        return true;
      }
      const idCls = `${el.id || ''} ${typeof el.className === 'string' ? el.className : ''}`;
      if (idCls.trim() && noiseClassRe.test(idCls)) {
        return true;
      }
      return false;
    };

    const resolveSafeHttpUrl = (rawHref: string): string | null => {
      const trimmed = rawHref.trim();
      if (!trimmed || trimmed.startsWith('#')) return null;
      const lower = trimmed.toLowerCase();
      if (lower.startsWith('javascript:') || lower.startsWith('vbscript:') || lower.startsWith('data:')) {
        return null;
      }
      try {
        const u = new URL(trimmed, pageUrl || undefined);
        if (u.protocol === 'http:' || u.protocol === 'https:') {
          return u.toString();
        }
        return null;
      } catch {
        return null;
      }
    };

    const collectInlineText = (node: Node): string => {
      if (node.nodeType === Node.TEXT_NODE) {
        return (node.textContent ?? '').replace(/\s+/g, ' ');
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return '';
      const el = node as Element;
      if (shouldSkipElement(el)) return '';
      const tag = el.tagName.toUpperCase();
      if (tag === 'BR') return ' ';
      if (tag === 'CODE') {
        const codeTxt = (el.textContent ?? '').replace(/\s+/g, ' ').trim();
        return codeTxt ? `\`${codeTxt.replace(/`/g, "'")}\`` : '';
      }
      if (tag === 'A') {
        const label = Array.from(el.childNodes).map(collectInlineText).join('').replace(/\s+/g, ' ').trim();
        if (!label) return '';
        const href = el.getAttribute('href') || '';
        const absUrl = resolveSafeHttpUrl(href);
        if (!absUrl || absUrl === label) return label;
        return `[${label}](${absUrl})`;
      }
      if (tag === 'STRONG' || tag === 'B') {
        const t = Array.from(el.childNodes).map(collectInlineText).join('').replace(/\s+/g, ' ').trim();
        return t ? `**${t}**` : '';
      }
      if (tag === 'EM' || tag === 'I') {
        const t = Array.from(el.childNodes).map(collectInlineText).join('').replace(/\s+/g, ' ').trim();
        return t ? `*${t}*` : '';
      }
      const sr = (el as HTMLElement).shadowRoot;
      const children = sr ? Array.from(sr.childNodes) : Array.from(el.childNodes);
      return children.map(collectInlineText).join('');
    };

    const serializeTable = (tableEl: Element): string => {
      const rows: string[][] = [];
      for (const tr of Array.from(tableEl.querySelectorAll('tr'))) {
        if (!isVisibleEl(tr)) continue;
        const cells: string[] = [];
        for (const cell of Array.from(tr.children)) {
          const cTag = cell.tagName.toUpperCase();
          if (cTag !== 'TH' && cTag !== 'TD') continue;
          const cellTxt = collectInlineText(cell).replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim();
          cells.push(cellTxt);
        }
        if (cells.length > 0 && cells.some((c) => c.length > 0)) {
          rows.push(cells);
        }
      }
      if (rows.length === 0) return '';
      const cols = rows.reduce((m, r) => Math.max(m, r.length), 0);
      const normRow = (r: string[]) => {
        const copy = [...r];
        while (copy.length < cols) copy.push('');
        return `| ${copy.join(' | ')} |`;
      };
      const out = [normRow(rows[0]), `| ${Array(cols).fill('---').join(' | ')} |`];
      for (let i = 1; i < rows.length; i += 1) {
        out.push(normRow(rows[i]));
      }
      return `\n\n${out.join('\n')}\n\n`;
    };

    const outline: string[] = [];
    const seenOutline = new Set<string>();
    const rootSections: string[] = [];

    const serializeRoot = (root: Element): string => {
      const parts: string[] = [];
      const walk = (node: Node): void => {
        if (node.nodeType === Node.TEXT_NODE) {
          const txt = (node.textContent ?? '').replace(/\s+/g, ' ');
          if (txt.trim()) parts.push(txt);
          return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) {
          return;
        }
        const el = node as Element;
        if (shouldSkipElement(el)) {
          return;
        }
        const tag = el.tagName.toUpperCase();

        if (tag === 'BR') {
          parts.push('\n');
          return;
        }
        if (tag === 'HR') {
          parts.push('\n\n---\n\n');
          return;
        }
        if (tag === 'PRE') {
          const innerCode = el.querySelector('code');
          const cls = `${el.className || ''} ${innerCode?.className || ''}`;
          const langM = /(?:language|lang)-([a-zA-Z0-9_-]+)/i.exec(cls);
          const lang = langM ? langM[1] : '';
          const code = (innerCode?.textContent ?? el.textContent ?? '').replace(/\r\n/g, '\n').trim();
          if (code) {
            parts.push(`\n\n\`\`\`${lang}\n${code}\n\`\`\`\n\n`);
          }
          return;
        }
        if (tag === 'CODE') {
          const codeTxt = (el.textContent ?? '').replace(/\s+/g, ' ').trim();
          if (codeTxt) parts.push(`\`${codeTxt.replace(/`/g, "'")}\``);
          return;
        }
        if (tag === 'TABLE') {
          const mdTable = serializeTable(el);
          if (mdTable) parts.push(mdTable);
          return;
        }
        if (/^H[1-6]$/.test(tag)) {
          const level = Number.parseInt(tag[1], 10);
          const hText = collectInlineText(el).replace(/\s+/g, ' ').trim();
          if (hText) {
            parts.push(`\n\n${'#'.repeat(level)} ${hText}\n\n`);
            if (level <= 3 && outline.length < 35) {
              const plain = hText.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').trim();
              if (plain && !seenOutline.has(plain)) {
                seenOutline.add(plain);
                outline.push(`${'#'.repeat(level)} ${plain}`);
              }
            }
          }
          return;
        }
        if (tag === 'A') {
          const linkMd = collectInlineText(el);
          if (linkMd) parts.push(linkMd);
          return;
        }
        if (tag === 'STRONG' || tag === 'B' || tag === 'EM' || tag === 'I') {
          const inlineMd = collectInlineText(el);
          if (inlineMd) parts.push(inlineMd);
          return;
        }
        if (tag === 'BLOCKQUOTE') {
          const bq = Array.from(el.childNodes).map(collectInlineText).join('').replace(/\s+/g, ' ').trim();
          if (bq) parts.push(`\n\n> ${bq}\n\n`);
          return;
        }
        if (tag === 'LI') {
          const liTxt = Array.from(el.childNodes).map(collectInlineText).join('').replace(/\s+/g, ' ').trim();
          if (liTxt) parts.push(`\n- ${liTxt}`);
          return;
        }
        if (tag === 'IMG') {
          const alt = (el.getAttribute('alt') || '').replace(/\s+/g, ' ').trim();
          if (alt.length >= 4) {
            parts.push(` [Image: ${alt}] `);
          }
          return;
        }
        if (tag === 'INPUT' || tag === 'TEXTAREA') {
          const inp = el as HTMLInputElement;
          if (inp.type !== 'password' && inp.type !== 'hidden' && inp.value && inp.value.trim()) {
            const label = inp.getAttribute('aria-label') || inp.placeholder || inp.name || 'Field';
            parts.push(`\n[${label}: ${inp.value.trim().slice(0, 300)}]\n`);
          }
          return;
        }
        if (tag === 'IFRAME') {
          try {
            const subBody = (el as HTMLIFrameElement).contentDocument?.body;
            if (subBody && isVisibleEl(subBody)) {
              parts.push('\n');
              walk(subBody);
              parts.push('\n');
            }
          } catch {
            // Cross-origin iframe ignored safely
          }
          return;
        }

        const isBlock = blockTags.has(tag);
        if (isBlock) parts.push('\n');

        const sr = (el as HTMLElement).shadowRoot;
        if (sr) {
          for (const srChild of Array.from(sr.childNodes)) {
            walk(srChild);
          }
        }
        for (const child of Array.from(el.childNodes)) {
          walk(child);
        }
        if (isBlock) parts.push('\n');
      };

      walk(root);
      return parts
        .join('')
        .replace(/[ \t]+\n/g, '\n')
        .replace(/\n[ \t]+/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
    };

    for (const r of roots) {
      const md = serializeRoot(r);
      if (md) rootSections.push(md);
    }

    const rawMarkdown = rootSections.join('\n\n---\n\n').trim();

    // Capture currently visible viewport excerpt so scrolled reading feels instant
    const vh = window.innerHeight || 800;
    const viewportBlocks: string[] = [];
    let vpChars = 0;
    for (const root of roots) {
      const vNodes = Array.from(root.querySelectorAll('h1, h2, h3, p, li, pre, blockquote, tr'));
      for (const vn of vNodes) {
        if (vpChars >= 1800) break;
        if (!isVisibleEl(vn)) continue;
        const rect = (vn as HTMLElement).getBoundingClientRect();
        if (rect.height <= 0 || rect.bottom <= 0 || rect.top >= vh) continue;
        const txt = ((vn as HTMLElement).innerText || vn.textContent || '').replace(/\s+/g, ' ').trim();
        if (txt.length >= 12 && !viewportBlocks.includes(txt)) {
          viewportBlocks.push(txt);
          vpChars += txt.length + 1;
        }
      }
      if (vpChars >= 1800) break;
    }
    const viewportExcerpt = viewportBlocks.join('\n').slice(0, 1800).trim();

    const metaPairs = Object.entries(metadata).map(([k, v]) => `- ${k}: ${v}`);
    const metaBlock = metaPairs.length > 0 ? `## Metadata\n${metaPairs.join('\n')}` : '';
    const fullRaw = [metaBlock, rawMarkdown].filter(Boolean).join('\n\n').trim();
    const { text: fullContent } = truncateClean(fullRaw, 120000);

    let previewText = fullRaw;
    let truncated = false;
    if (fullRaw.length > safeMaxChars) {
      truncated = true;
      const prefixBlocks: string[] = [];
      if (metaBlock) prefixBlocks.push(metaBlock);
      if (outline.length >= 2) {
        prefixBlocks.push(`## Page Outline\n${outline.slice(0, 20).map((h) => `- ${h}`).join('\n')}`);
      }
      if ((window.scrollY || 0) > 250 && viewportExcerpt.length >= 80) {
        prefixBlocks.push(`## Currently Visible On Screen\n${viewportExcerpt}`);
      }
      const headerStr = prefixBlocks.join('\n\n');
      const remaining = Math.max(400, safeMaxChars - headerStr.length - 120);
      const introSlice = truncateClean(rawMarkdown, remaining).text;
      previewText = [headerStr, introSlice].filter(Boolean).join('\n\n').trim();
      previewText += `\n[Showing ${previewText.length} of ${fullRaw.length} chars; full page indexed locally]`;
    }

    return {
      mode: 'article',
      title: pageTitle,
      url: pageUrl,
      content: previewText,
      fullContent,
      outline,
      viewportExcerpt,
      selectionText: rawSelectionText || undefined,
      metadata,
      contentHash: fnv1aHash(fullRaw),
      charCount: fullRaw.length,
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
