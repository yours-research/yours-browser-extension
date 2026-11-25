import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { rankPassagesBm25 } from '../src/tools/bm25Ranker.ts';
import { executeToolCall } from '../src/tools/browserTools.ts';
import {
  distillHtmlToMarkdown,
  extractYouTubeCaptionTrack,
  formatInteractiveElements,
  parseYouTubeTranscriptXml,
} from '../src/tools/htmlDistiller.ts';
import { runJsCalc } from '../src/tools/jsCalc.ts';
import {
  detectPendingToolCall,
  formatToolResultMessage,
  parseToolCalls,
  TIER1_TOOL_DEFINITIONS,
  ToolCallJsonGrammar,
  validateToolCall,
} from '../src/tools/toolProtocol.ts';
import type { FetchedSourceResult } from '../src/tools/types.ts';
import { runFetchAndReadUrls } from '../src/tools/urlFetcher.ts';
import {
  detectAntiBotChallenge,
  parseBingWebHtmlResults,
  parseDuckDuckGoHtmlResults,
  parseStartpageHtmlResults,
  runWebSearch,
  unwrapBingRedirectUrl,
} from '../src/tools/webSearch.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

describe('manifest Option A & MV3 CSP compliance', () => {
  it('enforces Option A permissions, no default host_permissions, and strict CSP', () => {
    const manifestPath = path.join(ROOT_DIR, 'public', 'manifest.json');
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as {
      permissions: string[];
      optional_host_permissions?: string[];
      host_permissions?: string[];
      content_security_policy?: { extension_pages?: string };
    };

    expect(manifest.permissions).toEqual(
      expect.arrayContaining([
        'sidePanel',
        'offscreen',
        'storage',
        'unlimitedStorage',
        'activeTab',
        'scripting',
        'tabs',
      ]),
    );
    expect(manifest.optional_host_permissions).toEqual(['https://*/*']);
    expect(manifest.host_permissions).toBeUndefined();
    expect(manifest.content_security_policy?.extension_pages).toBe(
      "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';",
    );
  });

  it('contains zero eval() or new Function() in src/tools or src/background', () => {
    const dirs = [path.join(ROOT_DIR, 'src', 'tools'), path.join(ROOT_DIR, 'src', 'background')];
    for (const dir of dirs) {
      for (const file of fs.readdirSync(dir)) {
        if (!file.endsWith('.ts')) continue;
        const src = fs.readFileSync(path.join(dir, file), 'utf8');
        expect(src).not.toMatch(/\beval\s*\(/);
        expect(src).not.toMatch(/\bnew\s+Function\s*\(/);
      }
    }
  });
});

describe('htmlDistiller', () => {
  it('distills HTML to clean markdown, strips noise, prefers <article>, and decodes entities', () => {
    const html = `
      <!doctype html>
      <html>
        <head>
          <title>WebGPU &amp; Rust Guide &#8212; Docs</title>
          <style>.hidden { display: none; }</style>
          <script>window.__secret = "drop me";</script>
        </head>
        <body>
          <header><nav><a href="/home">Home</a><a href="/pricing">Pricing</a></nav></header>
          <div class="cookie-consent">Accept all cookies now!</div>
          <aside>Related ads here</aside>
          <article>
            <h1>Getting Started</h1>
            <p>WebGPU runs fast compute shaders with <code>wgsl</code> &amp; Rust &#x26A1;.</p>
            <ul>
              <li>Step one: <a href="/docs/setup">Setup Guide</a></li>
              <li>Step two: <a href="javascript:alert(1)">Unsafe Link</a></li>
            </ul>
            <pre><code class="language-rust">fn main() { println!("hi"); }</code></pre>
            <table>
              <tr><th>Backend</th><th>Speed</th></tr>
              <tr><td>WGSL</td><td>Fast</td></tr>
            </table>
          </article>
          <footer>Copyright 2026</footer>
        </body>
      </html>
    `;

    const res = distillHtmlToMarkdown(html, {
      baseUrl: 'https://example.com/blog/post',
      maxChars: 8000,
    });

    expect(res.title).toBe('WebGPU & Rust Guide — Docs');
    expect(res.markdown).toContain('# Getting Started');
    expect(res.markdown).toContain('`wgsl` & Rust ⚡.');
    expect(res.markdown).toContain('- Step one: [Setup Guide](https://example.com/docs/setup)');
    expect(res.markdown).toContain('- Step two: Unsafe Link');
    expect(res.markdown).not.toContain('javascript:');
    expect(res.markdown).toContain('```rust\nfn main() { println!("hi"); }\n```');
    expect(res.markdown).toContain('| Backend | Speed |\n| --- | --- |\n| WGSL | Fast |');
    expect(res.markdown).not.toContain('Accept all cookies');
    expect(res.markdown).not.toContain('Copyright 2026');
    expect(res.markdown).not.toContain('drop me');
  });

  it('truncates cleanly within maxChars', () => {
    const longPara = '<p>' + 'word '.repeat(200) + '</p>';
    const res = distillHtmlToMarkdown(`<html><body><main>${longPara}</main></body></html>`, {
      maxChars: 120,
    });
    expect(res.markdown.length).toBeLessThanOrEqual(140);
    expect(res.markdown).toContain('[truncated]');
    expect(res.textLength).toBeGreaterThan(500);
  });

  it('extracts YouTube caption tracks preferring manual English over ASR', () => {
    const fakeYtHtml = `
      <html>
        <head><title>Demo Video - YouTube</title></head>
        <body>
          <script>
            var ytInitialPlayerResponse = {
              "videoDetails": {"title": "Rust WebGPU Deep Dive"},
              "captions": {
                "playerCaptionsTracklistRenderer": {
                  "captionTracks": [
                    {"baseUrl": "https://www.youtube.com/api/timedtext?lang=es", "languageCode": "es"},
                    {"baseUrl": "https://www.youtube.com/api/timedtext?lang=en&kind=asr", "languageCode": "en", "kind": "asr"},
                    {"baseUrl": "https://www.youtube.com/api/timedtext?lang=en\\u0026fmt=srv1", "languageCode": "en"}
                  ]
                }
              }
            };
          </script>
        </body>
      </html>
    `;

    const track = extractYouTubeCaptionTrack(fakeYtHtml);
    expect(track).toEqual({
      title: 'Rust WebGPU Deep Dive',
      captionBaseUrl: 'https://www.youtube.com/api/timedtext?lang=en&fmt=srv1',
      languageCode: 'en',
    });

    const xml = `
      <transcript>
        <text start="5.2" dur="2.1">Hello &amp;#39;world&amp;#39;</text>
        <text start="65.9" dur="3.0">Second\nline here</text>
      </transcript>
    `;
    const transcript = parseYouTubeTranscriptXml(xml, 500);
    expect(transcript).toBe("[00:05] Hello 'world'\n[01:05] Second line here");
  });

  it('formats interactive elements into compact token-efficient lines', () => {
    const out = formatInteractiveElements(
      [
        { id: 1, role: 'button', name: 'Submit', state: 'enabled', cx: 120.4, cy: 44.8 },
        { id: 2, role: 'input:text', name: 'Email', cx: 80, cy: 100 },
        { id: 3, role: 'link', name: 'Cancel', cx: 200, cy: 100 },
      ],
      2,
    );
    expect(out).toBe(
      '[1 | button | Submit | enabled | (120,45)]\n[2 | input:text | Email | - | (80,100)]\n... (+1 more elements)',
    );
  });
});

describe('bm25Ranker', () => {
  it('ranks relevant chunks higher, preserves intro chunk, and restores reading order', () => {
    const intro = '# Architecture Overview\n\nThis document describes the TinyGPT browser runtime and local execution model.';
    const filler1 = '## Styling Tokens\n\n' + 'Buttons use rounded borders and subtle shadows for light and dark themes. '.repeat(6);
    const relevant = '## WebGPU Quantization\n\n' + 'INT8 block quantization reduces KV cache memory bandwidth on WebGPU compute shaders. '.repeat(5);
    const filler2 = '## Icons and Assets\n\n' + 'Extension icons are stored as 16px, 48px, and 128px PNG assets in public. '.repeat(6);

    const doc = [intro, filler1, relevant, filler2].join('\n\n');
    const ranked = rankPassagesBm25(doc, 'WebGPU INT8 quantization memory', 900);

    expect(ranked.excerpt).toContain('# Architecture Overview');
    expect(ranked.excerpt).toContain('## WebGPU Quantization');
    expect(ranked.excerpt.indexOf('# Architecture Overview')).toBeLessThan(
      ranked.excerpt.indexOf('## WebGPU Quantization'),
    );
    expect(ranked.excerpt.length).toBeLessThanOrEqual(900);
    expect(ranked.topScore).toBeGreaterThan(0);
    expect(ranked.totalChunks).toBeGreaterThanOrEqual(4);
  });
});

describe('jsCalc', () => {
  it('evaluates arithmetic, scientific notation, constants, and array functions without eval', () => {
    expect(runJsCalc({ expression: '2 + 3 * 4 ^ 2 - (10 / 2)' }).result).toBe(45);
    expect(runJsCalc({ expression: '-2^2 + 1.5e-3 * 1000' }).result).toBeCloseTo(-2.5);
    expect(runJsCalc({ expression: 'round(pi, 4)' }).result).toBe(3.1416);
    expect(runJsCalc({ expression: 'round(e, 2)' }).result).toBe(2.72);
    expect(runJsCalc({ expression: 'sum([1, 2, 3, 4])' }).result).toBe(10);
    expect(runJsCalc({ expression: 'mean([10, 20, 30])' }).result).toBe(20);
    expect(runJsCalc({ expression: 'median([10, 2, 8, 4])' }).result).toBe(6);
    expect(runJsCalc({ expression: 'std([2, 4, 4, 4, 5, 5, 7, 9])' }).result).toBe(2);
    expect(runJsCalc({ expression: 'count([5, 6, 7])' }).result).toBe(3);
    expect(runJsCalc({ expression: 'sqrt(81) + pow(2, 5)' }).result).toBe(41);
  });

  it('supports unit conversions in both expr and unit modes', () => {
    expect(runJsCalc({ expression: 'convert(100, "C", "F")' }).result).toBe(212);
    expect(runJsCalc({ mode: 'unit', value: 1, from: 'KiB', to: 'B' }).result).toBe(1024);
    expect(runJsCalc({ mode: 'unit', value: 1, from: 'KB', to: 'B' }).result).toBe(1000);
    expect(runJsCalc({ mode: 'unit', value: 5, from: 'km', to: 'm' }).result).toBe(5000);
    expect(runJsCalc({ mode: 'unit', value: 2, from: 'hr', to: 'min' }).result).toBe(120);
    expect(runJsCalc({ mode: 'unit', value: 1, from: 'kg', to: 'g' }).result).toBe(1000);
    expect(runJsCalc({ mode: 'unit', value: 1, from: 'km', to: 'kg' }).ok).toBe(false);
  });

  it('supports date math, JSON path queries, and CSV summaries', () => {
    expect(runJsCalc({ expression: 'daysBetween("2026-01-01", "2026-01-15")' }).result).toBe(14);
    expect(runJsCalc({ expression: 'addDays("2026-01-30", 3)' }).result).toBe('2026-02-02');
    expect(runJsCalc({ mode: 'date', dateA: '2026-03-01', dateB: '2026-03-11' }).result).toBe(10);
    expect(runJsCalc({ mode: 'date', dateA: '2026-03-01', days: 5 }).result).toBe('2026-03-06');

    const jsonRes = runJsCalc({
      mode: 'json',
      data: JSON.stringify({ a: { b: [{ c: 'found' }] } }),
      path: 'a.b[0].c',
    });
    expect(jsonRes.ok).toBe(true);
    expect(jsonRes.result).toBe('found');

    const csvRes = runJsCalc({
      mode: 'csv',
      data: 'name,score,latency\nrun1,10,20\nrun2,20,40',
    });
    expect(csvRes.ok).toBe(true);
    expect(csvRes.formatted).toContain('Rows: 2');
    expect(csvRes.formatted).toContain('score: min=10, max=20, mean=15');
  });

  it('rejects division by zero and unknown identifiers cleanly', () => {
    expect(runJsCalc({ expression: '10 / 0' }).ok).toBe(false);
    expect(runJsCalc({ expression: 'window.location' }).ok).toBe(false);
  });
});

describe('toolProtocol', () => {
  it('defines all 6 Tier-1 tools', () => {
    expect(TIER1_TOOL_DEFINITIONS.map((t) => t.name)).toEqual([
      'read_active_page',
      'web_search',
      'fetch_and_read_url',
      'list_open_tabs',
      'read_tab',
      'js_calc',
    ]);
  });

  it('parses <tool_call> blocks including fenced json, trailing commas, and unescaped newlines', () => {
    const modelOutput = `Let me check that for you.
<tool_call>
\`\`\`json
{
  "name": "web_search",
  "arguments": {
    "query": "rust webgpu
tutorial",
    "limit": 25,
  }
}
\`\`\`
</tool_call>
One moment.`;

    const parsed = parseToolCalls(modelOutput);
    expect(parsed.parseErrors).toEqual([]);
    expect(parsed.toolCalls).toHaveLength(1);
    expect(parsed.toolCalls[0].name).toBe('web_search');
    expect(parsed.toolCalls[0].arguments.limit).toBe(8); // clamped to 1..8
    expect(parsed.cleanText).toBe('Let me check that for you.\n\nOne moment.');
  });

  it('detects pending streaming tool calls with friendly labels', () => {
    expect(detectPendingToolCall('Thinking... <tool_call>{"name": "web_search", "arg')).toEqual({
      isOpen: true,
      toolName: 'web_search',
      label: 'Searching the web...',
    });
    expect(detectPendingToolCall('<tool_call>{"name": "js_calc"')).toEqual({
      isOpen: true,
      toolName: 'js_calc',
      label: 'Calculating...',
    });
    expect(
      detectPendingToolCall('<tool_call>{"name": "js_calc", "arguments": {"expression": "1+1"}}</tool_call>'),
    ).toBeNull();
  });

  it('validates and clamps bounds strictly', () => {
    const tooManyUrls = validateToolCall({
      name: 'fetch_and_read_url',
      arguments: {
        urls: [
          'https://a.com',
          'https://b.com',
          'https://c.com',
          'https://d.com',
          'https://e.com',
          'https://f.com',
        ],
      },
    });
    expect(tooManyUrls.valid).toBe(true);
    expect((tooManyUrls.normalized?.arguments.urls as string[]).length).toBe(5);

    expect(validateToolCall({ name: 'read_tab', arguments: { tabId: -1 } }).valid).toBe(false);
  });

  it('formats compact <tool_result> and validates JSON prefixes via ToolCallJsonGrammar', () => {
    const formatted = formatToolResultMessage('js_calc', {
      ok: true,
      toolName: 'js_calc',
      formatted: '42',
    });
    expect(formatted).toBe('<tool_result name="js_calc" ok="true">\n42\n</tool_result>');

    expect(ToolCallJsonGrammar.isValidToolJsonPrefix('{"name": "web_search", "arguments": {"qu')).toBe(true);
    expect(ToolCallJsonGrammar.isValidToolJsonPrefix('{"name": "js_calc"}')).toBe(true);
    expect(ToolCallJsonGrammar.isValidToolJsonPrefix('{"name": "js_calc"}{')).toBe(false);
    expect(ToolCallJsonGrammar.isValidToolJsonPrefix('not json')).toBe(false);
    expect(ToolCallJsonGrammar.allowedNextTokenHint('{"name"')).toBe(':');
  });
});

describe('webSearch & urlFetcher', () => {
  it('parses DuckDuckGo HTML & Lite results and unwraps /l/?uddg= redirects', () => {
    const ddgHtml = `
      <div class="result">
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F">The Rust Book</a>
        <a class="result__snippet">Learn Rust programming &amp; ownership.</a>
      </div>
      <div class="result">
        <a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=foo">Ad Result</a>
      </div>
      <div class="result">
        <a class="result__a" href="https://www.w3.org/TR/webgpu/">WebGPU Spec</a>
        <div class="result__snippet">W3C WebGPU specification.</div>
      </div>
    `;

    const results = parseDuckDuckGoHtmlResults(ddgHtml, 5);
    expect(results).toEqual([
      {
        title: 'The Rust Book',
        url: 'https://doc.rust-lang.org/book/',
        snippet: 'Learn Rust programming & ownership.',
        domain: 'doc.rust-lang.org',
      },
      {
        title: 'WebGPU Spec',
        url: 'https://www.w3.org/TR/webgpu/',
        snippet: 'W3C WebGPU specification.',
        domain: 'w3.org',
      },
    ]);
  });

  it('detects CAPTCHA / Turnstile / 429 / 403 challenge responses and aborts cleanly', async () => {
    expect(detectAntiBotChallenge('<div class="cf-turnstile" data-sitekey="abc"></div>', 200)).toBe(true);
    expect(detectAntiBotChallenge('Please verify you are human before continuing', 403)).toBe(true);
    expect(detectAntiBotChallenge('Rate limit exceeded', 429)).toBe(true);
    expect(detectAntiBotChallenge('<html><body>Normal article</body></html>', 200)).toBe(false);

    const searchRes = await runWebSearch('webgpu', 5, {
      hasPermission: async () => true,
      fetchFn: async () =>
        new Response('<html><body>Please verify you are human <div class="g-recaptcha"></div></body></html>', {
          status: 403,
        }),
    });
    expect(searchRes.ok).toBe(false);
    expect(searchRes.blockedByCaptcha).toBe(true);
    expect(searchRes.results).toEqual([]);
    expect(searchRes.manualUrl).toContain('duckduckgo.com');
  });

  it('fails closed immediately on Option A permission denial with zero network calls', async () => {
    const fetchSpy = vi.fn();
    const hiddenTabSpy = vi.fn();

    const searchRes = await runWebSearch('local llm', 5, {
      hasPermission: async () => false,
      fetchFn: fetchSpy as unknown as typeof fetch,
      openHiddenTabAndExtract: hiddenTabSpy,
    });
    expect(searchRes.ok).toBe(false);
    expect(searchRes.needsPermission).toBe(true);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(hiddenTabSpy).not.toHaveBeenCalled();

    const fetchRes = await runFetchAndReadUrls(['https://example.com'], {
      deps: {
        hasPermission: async () => false,
        fetchFn: fetchSpy as unknown as typeof fetch,
        openHiddenTabAndExtract: hiddenTabSpy,
      },
    });
    expect(fetchRes).toHaveLength(1);
    expect(fetchRes[0].error).toContain('permission');
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(hiddenTabSpy).not.toHaveBeenCalled();
  });

  it('streams onSourceReady, validates http/https only, and uses hidden tab only when static text < 200 chars', async () => {
    const ready: FetchedSourceResult[] = [];
    const hiddenSpy = vi.fn(async (url: string) => ({
      html: `<article><h1>SPA Loaded</h1><p>${'Hydrated content from background tab. '.repeat(8)}</p></article>`,
      title: 'SPA Loaded',
      url,
    }));

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('static.example')) {
        return new Response(
          `<article><h1>Static Doc</h1><p>${'Detailed static documentation about BM25 ranking. '.repeat(8)}</p></article>`,
          { status: 200, headers: { 'content-type': 'text/html' } },
        );
      }
      return new Response('<html><body><div id="root"></div></body></html>', {
        status: 200,
        headers: { 'content-type': 'text/html' },
      });
    });

    const results = await runFetchAndReadUrls(
      ['file:///etc/passwd', 'https://static.example/doc', 'https://spa.example/app'],
      {
        query: 'BM25 ranking',
        maxCharsPerSource: 2000,
        onSourceReady: (src) => ready.push(src),
        deps: {
          hasPermission: async () => true,
          fetchFn: fetchMock as unknown as typeof fetch,
          openHiddenTabAndExtract: hiddenSpy,
        },
      },
    );

    expect(ready).toHaveLength(3);
    expect(results[0].error).toContain('Invalid URL');
    expect(results[1].fromHiddenTab).toBe(false);
    expect(results[1].excerpt).toContain('Static Doc');
    expect(results[2].fromHiddenTab).toBe(true);
    expect(results[2].excerpt).toContain('SPA Loaded');
    expect(hiddenSpy).toHaveBeenCalledTimes(1);
    expect(hiddenSpy).toHaveBeenCalledWith('https://spa.example/app');
  });
});

describe('browserTools Option A & offline execution', () => {
  const originalChrome = (globalThis as { chrome?: unknown }).chrome;

  afterEach(() => {
    (globalThis as { chrome?: unknown }).chrome = originalChrome;
  });

  it('fails closed on web_search and fetch_and_read_url when chrome.permissions denies https://*/*', async () => {
    const fetchSpy = vi.fn();
    (globalThis as { chrome?: unknown }).chrome = {
      permissions: {
        contains: vi.fn(async () => false),
      },
      tabs: {
        query: vi.fn(async () => [{ id: 7, title: 'Docs', url: 'https://example.com', active: true, windowId: 1 }]),
      },
    };

    const searchResult = await executeToolCall(
      { name: 'web_search', arguments: { query: 'test' } },
      { fetchFn: fetchSpy as unknown as typeof fetch },
    );
    expect(searchResult.ok).toBe(false);
    expect(searchResult.needsPermission).toBe(true);

    const fetchResult = await executeToolCall(
      { name: 'fetch_and_read_url', arguments: { urls: ['https://example.com'] } },
      { fetchFn: fetchSpy as unknown as typeof fetch },
    );
    expect(fetchResult.ok).toBe(false);
    expect(fetchResult.needsPermission).toBe(true);
    expect(fetchSpy).not.toHaveBeenCalled();

    // Meanwhile offline tools work seamlessly without host permission
    const tabsResult = await executeToolCall({ name: 'list_open_tabs', arguments: {} });
    expect(tabsResult.ok).toBe(true);
    expect(tabsResult.formatted).toContain('[7] *active* Docs');

    const calcResult = await executeToolCall({
      name: 'js_calc',
      arguments: { expression: 'sum([10, 20, 30])' },
    });
    expect(calcResult.ok).toBe(true);
    expect(calcResult.formatted).toBe('60');
  });

  it('parses Bing/Startpage SERPs, avoids false-positive CAPTCHA on 200 articles, and supports zero-key Chromium SERP search + fused scrape', async () => {
    const encodedTarget = Buffer.from('https://www.w3.org/TR/webgpu/?v=2', 'utf8')
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    const bingHref = `https://www.bing.com/ck/a?!&&p=abc&u=a1${encodedTarget}&ntb=1`;
    expect(unwrapBingRedirectUrl(bingHref)).toBe('https://www.w3.org/TR/webgpu/?v=2');

    const bingHtml = `
      <ul>
        <li class="b_algo">
          <h2><a href="${bingHref}">WebGPU Specification</a></h2>
          <div class="b_caption"><p>W3C WebGPU candidate recommendation.</p></div>
        </li>
      </ul>
    `;
    const bingItems = parseBingWebHtmlResults(bingHtml, 5);
    expect(bingItems).toHaveLength(1);
    expect(bingItems[0]).toMatchObject({
      title: 'WebGPU Specification',
      url: 'https://www.w3.org/TR/webgpu/?v=2',
      domain: 'w3.org',
    });

    const spHtml = `
      <div class="w-gl__result">
        <a class="w-gl__result-title" href="https://webgpu.org/">WebGPU Hub</a>
        <p class="w-gl__description">Community docs and samples.</p>
      </div>
    `;
    expect(parseStartpageHtmlResults(spHtml, 5)[0]?.url).toBe('https://webgpu.org/');

    // False-positive guard: rich 200 OK article with footer reCAPTCHA must NOT trigger CAPTCHA block
    const richArticleWithRecaptchaFooter = `
      <html><body>
        <article><h1>Deep Technical Article</h1><p>${'Detailed technical explanation about WebGPU shaders and DeltaNet memory layouts. '.repeat(14)}</p></article>
        <div class="newsletter"><div class="g-recaptcha" data-sitekey="footer-key"></div></div>
      </body></html>
    `;
    expect(detectAntiBotChallenge(richArticleWithRecaptchaFooter, 200)).toBe(false);

    // Zero-key direct Chromium SERP + fused parallel URL markdown extraction via executeToolCall
    const serpResult = await executeToolCall(
      { name: 'web_search', arguments: { query: 'webgpu wgsl', limit: 3, readTopN: 1 } },
      {
        hasPermission: async () => true,
        searchViaChromiumSerp: async (q) => ({
          ok: true,
          engine: 'bing',
          query: q,
          elapsedMs: 420,
          results: [
            {
              title: 'WebGPU Spec',
              url: 'https://www.w3.org/TR/webgpu/',
              snippet: 'W3C WebGPU spec summary.',
              domain: 'w3.org',
            },
          ],
        }),
        fetchFn: async () =>
          new Response(
            `<article><h1>WebGPU Standard</h1><p>${'WGSL defines compute and fragment shader execution on WebGPU devices. '.repeat(8)}</p></article>`,
            { status: 200, headers: { 'content-type': 'text/html' } },
          ),
      },
    );
    expect(serpResult.ok).toBe(true);
    expect(serpResult.formatted).toContain('WebGPU Spec');
    expect(serpResult.formatted).toContain('## Top Source Excerpts');
    expect(serpResult.formatted).toContain('WGSL defines compute and fragment shader execution');
  });

  it('supports query-targeted BM25 search inside read_active_page over fullContent', async () => {
    const longBody = [
      '# Overview',
      'Intro section about installation. '.repeat(40),
      '## Enterprise Pricing',
      'The Enterprise plan includes 10,000 req/min and dedicated GPU workers at $490/mo.',
    ].join('\n\n');

    (globalThis as { chrome?: unknown }).chrome = {
      permissions: { contains: vi.fn(async () => true) },
      tabs: { query: vi.fn(async () => [{ id: 9, title: 'Pricing Docs', url: 'https://docs.test/api', active: true, windowId: 1 }]) },
      scripting: {
        executeScript: vi.fn(async () => [
          {
            result: {
              mode: 'article',
              title: 'Pricing Docs',
              url: 'https://docs.test/api',
              content: longBody.slice(0, 500),
              fullContent: longBody,
              contentHash: 'deadbeef',
              charCount: longBody.length,
              truncated: true,
            },
          },
        ]),
      },
    };

    const pageSearch = await executeToolCall({
      name: 'read_active_page',
      arguments: { query: 'Enterprise plan $490/mo req/min', maxChars: 1200 },
    });
    expect(pageSearch.ok).toBe(true);
    expect(pageSearch.formatted).toContain('MATCHED FOR "Enterprise plan $490/mo req/min"');
    expect(pageSearch.formatted).toContain('$490/mo');
  });
});
