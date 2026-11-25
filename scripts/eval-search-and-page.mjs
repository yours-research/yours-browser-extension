import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { build as esbuild } from 'esbuild';
import { chromium } from 'playwright';

const DIST_DIR = path.resolve('dist');
const PROFILE_DIR = path.resolve('.bench-profile-firecrawl-eval');
const RUNS_DIR = path.resolve('benchmarks', 'runs');

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

async function bundleTsModule(entryPath) {
  const out = await esbuild({
    entryPoints: [entryPath],
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
  });
  const code = out.outputFiles[0].text;
  const b64 = Buffer.from(code, 'utf8').toString('base64');
  return import(`data:text/javascript;base64,${b64}`);
}

async function cleanProfileDir() {
  for (let i = 0; i < 8; i++) {
    try {
      if (fs.existsSync(PROFILE_DIR)) {
        fs.rmSync(PROFILE_DIR, { recursive: true, force: true, maxRetries: 4, retryDelay: 150 });
      }
      fs.mkdirSync(PROFILE_DIR, { recursive: true });
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
}

function buildFixtures() {
  const deepFiller = Array.from(
    { length: 75 },
    (_, i) =>
      `<section><h2>Section ${i + 1}: Architecture Notes</h2><p>Detailed documentation paragraph ${i + 1} covering memory alignment, worker queues, and extension messaging lifecycles in modern Chromium environments.</p></section>`,
  ).join('\n');

  const deepDocHtml = `<!doctype html><html><head><title>Deep API Reference</title></head><body>
    <main>
      <h1>Deep API Reference</h1>
      <p>Introduction to the TinyGPT WebGPU SDK.</p>
      ${deepFiller}
      <section>
        <h2>Enterprise Pricing &amp; Quotas</h2>
        <p>The Enterprise rate limit is 10,000 requests/min under code TITAN-QUOTA-490.</p>
        <table>
          <tr><th>Tier</th><th>Rate Limit</th><th>Price</th></tr>
          <tr><td>Starter</td><td>100 req/min</td><td>$0/mo</td></tr>
          <tr><td>Enterprise</td><td>10,000 req/min</td><td>$490/mo</td></tr>
        </table>
      </section>
    </main>
  </body></html>`;

  const multiArticleHtml = `<!doctype html><html><head><title>Community Thread</title></head><body>
    <div class="feed">
      ${Array.from(
        { length: 12 },
        (_, i) =>
          `<article><h3>Thread Post #${i + 1}</h3><p>Member contribution #${i + 1} discussing local WebGPU quantization trade-offs and KV cache reuse benchmarks.</p></article>`,
      ).join('\n')}
    </div>
  </body></html>`;

  const shadowHtml = `<!doctype html><html><head><title>Web Component App</title></head><body>
    <wc-article id="host"></wc-article>
    <script>
      const host = document.getElementById('host');
      const sr = host.attachShadow({ mode: 'open' });
      sr.innerHTML = '<main><h1>Shadow Root Report</h1><p>This critical content lives strictly inside an open Shadow DOM root and must be extracted accurately.</p></main>';
    </script>
  </body></html>`;

  const jsonLdHiddenHtml = `<!doctype html><html><head>
    <title>UltraBook Pro X</title>
    <meta name="description" content="14-inch developer laptop with unified memory." />
    <script type="application/ld+json">
      {"@context":"https://schema.org","@type":"Product","name":"UltraBook Pro X","author":{"name":"Ada Lovelace"},"offers":{"price":"1999","priceCurrency":"USD"},"aggregateRating":{"ratingValue":"4.9"}}
    </script>
    <style>.hidden-modal { display: none; }</style>
  </head><body>
    <div id="onetrust-consent-sdk">Accept all tracking cookies noise text!</div>
    <div class="hidden-modal">SECRET-HIDDEN-MODAL-DO-NOT-EXTRACT</div>
    <article><h1>UltraBook Pro X Overview</h1><p>Built for local 4B WebGPU inference with high unified memory bandwidth and cool thermals.</p></article>
    <footer>Copyright 2026 Storefront Inc. All rights reserved.</footer>
  </body></html>`;

  const iframeChildHtml = `<!doctype html><html><body><p>Embedded iframe spec note: IFRAME-SIGNAL-772.</p></body></html>`;
  const scrollIframeHtml = `<!doctype html><html><head><title>Scrolled Spec + Iframe</title></head><body>
    <main>
      <h1>Top Banner Section</h1>
      ${Array.from({ length: 45 }, (_, i) => `<p>Filler spacer line ${i + 1} before scrolled target.</p>`).join('\n')}
      <h2 id="target">Scrolled Target Section</h2>
      <p>VIEWPORT-SCROLL-MARKER-331: You scrolled directly to the runtime configuration section.</p>
      <iframe src="/iframe-child" style="width:400px;height:120px;"></iframe>
    </main>
  </body></html>`;

  const spaSelectionHtml = `<!doctype html><html><head><title>SPA App</title></head><body>
    <main id="app">
      <h1>Initial Dashboard Route</h1>
      <p id="sel-target">Highlight this exact phrase SELECTION-ALPHA-99 for verification while testing WebGPU local reading.</p>
    </main>
  </body></html>`;

  const recaptchaFooterHtml = `<!doctype html><html><head><title>Healthy Docs Page</title></head><body>
    <article>
      <h1>WebGPU Compute Pipelines</h1>
      ${Array.from({ length: 12 }, (_, i) => `<p>Paragraph ${i + 1}: Explaining workgroup shared memory, subgroup matrix operations, and storage buffer binding alignments.</p>`).join('\n')}
    </article>
    <section class="footer-newsletter">
      <div class="g-recaptcha" data-sitekey="6LcExampleKey12345"></div>
    </section>
  </body></html>`;

  const linksCodeHtml = `<!doctype html><html><head><title>Code & Links Doc</title></head><body>
    <article>
      <h1>WGSL Dispatch Guide</h1>
      <p>Read the full <a href="/docs/wgsl-spec">WGSL Spec Reference</a> before changing bindings.</p>
      <pre><code class="language-rust">fn dispatch_gemv(k: u32) -> u32 { k / 32 }</code></pre>
    </article>
  </body></html>`;

  return {
    deepDocHtml,
    multiArticleHtml,
    shadowHtml,
    jsonLdHiddenHtml,
    iframeChildHtml,
    scrollIframeHtml,
    spaSelectionHtml,
    recaptchaFooterHtml,
    linksCodeHtml,
  };
}

async function runLiveSerpEval(browserCtx, webSearchMod, htmlDistillerMod, bm25Mod) {
  const queries = [
    'W3C WebGPU specification',
    'Gemma 4 E4B architecture',
    'Chrome MV3 offscreen document',
  ];

  const engines = [
    {
      name: 'bing',
      url: (q) => `https://www.bing.com/search?q=${encodeURIComponent(q)}`,
      readySel: 'li.b_algo h2 a',
      extract: (limit) => {
        const items = [];
        for (const node of document.querySelectorAll('li.b_algo')) {
          if (items.length >= limit) break;
          if (node.closest('.b_ad') || node.classList.contains('b_ad')) continue;
          const a = node.querySelector('h2 a');
          if (!a) continue;
          const sn = node.querySelector('.b_caption p, .b_algoSlug, p');
          items.push({
            title: (a.textContent || '').replace(/\s+/g, ' ').trim(),
            href: a.href || a.getAttribute('href') || '',
            snippet: (sn?.textContent || '').replace(/\s+/g, ' ').trim(),
          });
        }
        return items;
      },
    },
    {
      name: 'ddg',
      url: (q) => `https://duckduckgo.com/?q=${encodeURIComponent(q)}`,
      readySel: 'a[data-testid="result-title-a"]',
      extract: (limit) => {
        const items = [];
        for (const node of document.querySelectorAll('article[data-testid="result"], [data-nrn="result"]')) {
          if (items.length >= limit) break;
          const a = node.querySelector('a[data-testid="result-title-a"], h2 a');
          if (!a) continue;
          const sn = node.querySelector('[data-result="snippet"]');
          items.push({
            title: (a.textContent || '').replace(/\s+/g, ' ').trim(),
            href: a.href || a.getAttribute('href') || '',
            snippet: (sn?.textContent || '').replace(/\s+/g, ' ').trim(),
          });
        }
        return items;
      },
    },
    {
      name: 'startpage',
      url: (q) => `https://www.startpage.com/do/search?q=${encodeURIComponent(q)}`,
      readySel: '.w-gl__result-title, a.result-title',
      extract: (limit) => {
        const items = [];
        for (const node of document.querySelectorAll('.w-gl__result, .result')) {
          if (items.length >= limit) break;
          const a = node.querySelector('a.w-gl__result-title, a.result-title, h3 a');
          if (!a) continue;
          const sn = node.querySelector('.w-gl__description, p');
          items.push({
            title: (a.textContent || '').replace(/\s+/g, ' ').trim(),
            href: a.href || a.getAttribute('href') || '',
            snippet: (sn?.textContent || '').replace(/\s+/g, ' ').trim(),
          });
        }
        return items;
      },
    },
  ];

  const serpPage = await browserCtx.newPage();
  const queryResults = [];

  try {
    for (const q of queries) {
      const qStart = performance.now();
      let pickedEngine = null;
      let normalized = [];
      let blocked = false;

      for (const eng of engines) {
        await serpPage.goto(eng.url(q), { waitUntil: 'domcontentloaded', timeout: 10000 }).catch(() => {});
        await serpPage.waitForSelector(eng.readySel, { timeout: 4000 }).catch(() => {});
        const snap = await serpPage.evaluate(
          ({ fnStr, limit }) => {
            const fn = eval(`(${fnStr})`);
            const items = fn(limit);
            const head = `${document.title} ${(document.body?.innerText || '').slice(0, 900)} ${location.href}`;
            const captcha =
              items.length === 0 &&
              /unusual traffic|verify you are human|are you a robot|anomaly-modal|sorry\/index/i.test(head);
            return { items, captcha };
          },
          { fnStr: eng.extract.toString(), limit: 6 },
        );

        if (snap.captcha) {
          blocked = true;
          continue;
        }
        const cleaned = [];
        const seen = new Set();
        for (const r of snap.items) {
          const resolved =
            eng.name === 'bing' ? webSearchMod.unwrapBingRedirectUrl(r.href) ?? r.href : r.href;
          try {
            const u = new URL(resolved);
            if ((u.protocol === 'http:' || u.protocol === 'https:') && !seen.has(u.toString())) {
              seen.add(u.toString());
              cleaned.push({ title: r.title, url: u.toString(), domain: u.hostname.replace(/^www\./, ''), snippet: r.snippet });
            }
          } catch {}
        }
        if (cleaned.length > 0) {
          pickedEngine = eng.name;
          normalized = cleaned.slice(0, 5);
          break;
        }
      }

      const serpMs = Math.round(performance.now() - qStart);

      // Parallel Firecrawl-style top-2 unique-domain scrape + BM25 rank
      const topUrls = [];
      const doms = new Set();
      for (const item of normalized) {
        if (topUrls.length >= 2) break;
        if (!doms.has(item.domain)) {
          doms.add(item.domain);
          topUrls.push(item.url);
        }
      }

      const scrapeStart = performance.now();
      const scraped = await Promise.all(
        topUrls.map(async (u) => {
          try {
            const ctrl = new AbortController();
            const t = setTimeout(() => ctrl.abort(), 5000);
            const res = await fetch(u, {
              signal: ctrl.signal,
              headers: {
                'User-Agent':
                  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                Accept: 'text/html,application/xhtml+xml',
              },
            });
            clearTimeout(t);
            const html = await res.text();
            if (webSearchMod.detectAntiBotChallenge(html, res.status)) {
              return { url: u, status: res.status, blockedByCaptcha: true, excerptChars: 0 };
            }
            const md = htmlDistillerMod.distillHtmlToMarkdown(html, { baseUrl: u, maxChars: 12000 });
            const ranked = bm25Mod.rankPassagesBm25(md.markdown, q, 2400);
            return {
              url: u,
              status: res.status,
              blockedByCaptcha: false,
              excerptChars: ranked.excerpt.length,
              topScore: Number(ranked.topScore.toFixed(2)),
            };
          } catch (e) {
            return { url: u, status: 0, error: e.message, excerptChars: 0 };
          }
        }),
      );
      const scrapeMs = Math.round(performance.now() - scrapeStart);

      console.log(
        `  [SERP+Scrape] "${q}" -> engine=${pickedEngine ?? 'none'} (${serpMs}ms), results=${normalized.length}, scraped=${scraped.filter((s) => s.excerptChars > 200).length}/${topUrls.length} (${scrapeMs}ms), blocked=${blocked}`,
      );
      queryResults.push({
        query: q,
        engine: pickedEngine,
        serpMs,
        scrapeMs,
        totalMs: serpMs + scrapeMs,
        resultCount: normalized.length,
        blockedByCaptcha: blocked && normalized.length === 0,
        topUrls: normalized.slice(0, 3).map((r) => r.url),
        scraped,
      });
    }
  } finally {
    await serpPage.close().catch(() => {});
  }

  return queryResults;
}

async function main() {
  assert(fs.existsSync(path.join(DIST_DIR, 'manifest.json')), 'dist/manifest.json missing. Run npm run build first.');
  await cleanProfileDir();

  const [pageExtMod, bm25Mod, webSearchMod, htmlDistillerMod] = await Promise.all([
    bundleTsModule(path.resolve('src/tools/pageExtractor.ts')),
    bundleTsModule(path.resolve('src/tools/bm25Ranker.ts')),
    bundleTsModule(path.resolve('src/tools/webSearch.ts')),
    bundleTsModule(path.resolve('src/tools/htmlDistiller.ts')),
  ]);

  const fixtures = buildFixtures();
  const routes = {
    '/deep': fixtures.deepDocHtml,
    '/multi': fixtures.multiArticleHtml,
    '/shadow': fixtures.shadowHtml,
    '/jsonld': fixtures.jsonLdHiddenHtml,
    '/scroll': fixtures.scrollIframeHtml,
    '/iframe-child': fixtures.iframeChildHtml,
    '/spa': fixtures.spaSelectionHtml,
    '/recaptcha-200': fixtures.recaptchaFooterHtml,
    '/links-code': fixtures.linksCodeHtml,
  };

  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url, 'http://127.0.0.1').pathname;
    const html = routes[pathname] ?? '<!doctype html><html><body>Not found</body></html>';
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;

  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    viewport: { width: 1280, height: 840 },
    args: [
      `--disable-extensions-except=${DIST_DIR}`,
      `--load-extension=${DIST_DIR}`,
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });

  try {
    let sw = context.serviceWorkers()[0];
    if (!sw) sw = await context.waitForEvent('serviceworker', { timeout: 20000 });
    const extId = new URL(sw.url()).hostname;

    console.log('\n=== [27-firecrawl-local-search-page-eval] Page Reader V2 & Zero-Key Chromium SERP Eval ===');

    // 0. Verify Option A fail-closed EXEC_TOOL search gating when permission isn't granted
    const sidePanel = await context.newPage();
    await sidePanel.goto(`chrome-extension://${extId}/sidepanel.html?noAutoLoad=1`, { waitUntil: 'domcontentloaded' });
    const optionAGate = await sidePanel.evaluate(async () =>
      chrome.runtime.sendMessage({ type: 'EXEC_TOOL', call: { name: 'web_search', arguments: { query: 'webgpu' } } }),
    );
    assert(optionAGate?.needsPermission === true, `Expected fail-closed needsPermission=true, got ${JSON.stringify(optionAGate)}`);
    await sidePanel.close();

    // 1. Run 8 Page Reader V2 fixtures in real Chromium
    const page = context.pages()[0] ?? (await context.newPage());
    const extractFnStr = pageExtMod.inPageExtractScript.toString();
    const runExtract = async (urlPath, mode = 'article', maxChars = 5000) => {
      await page.goto(`${baseUrl}${urlPath}`, { waitUntil: 'load' });
      const t0 = performance.now();
      const extracted = await page.evaluate(
        ({ fn, m, lim }) => eval(`(${fn})`)(m, lim),
        { fn: extractFnStr, m: mode, lim: maxChars },
      );
      return { extracted, extractMs: Number((performance.now() - t0).toFixed(2)) };
    };

    // Fixture 1: Deep 28K doc + BM25 recall
    const f1 = await runExtract('/deep', 'article', 5000);
    const f1Recall = bm25Mod.rankPassagesBm25(f1.extracted.fullContent, 'Enterprise rate limit price $490 TITAN-QUOTA-490', 3200);
    assert(f1.extracted.truncated === true && f1.extracted.charCount > 12000, 'Expected truncated deep doc');
    assert(f1.extracted.content.includes('## Page Outline'), 'Missing Page Outline on truncated doc');
    assert(f1Recall.excerpt.includes('TITAN-QUOTA-490') && f1Recall.excerpt.includes('| Enterprise | 10,000 req/min | $490/mo |'), 'Deep BM25 recall missed tail pricing table');

    // Fixture 2: Multi-<article> 12-post thread
    const f2 = await runExtract('/multi', 'article', 5000);
    assert(
      f2.extracted.fullContent.includes('Post #1') &&
        f2.extracted.fullContent.includes('Post #6') &&
        f2.extracted.fullContent.includes('Post #12'),
      'Multi-article thread lost posts',
    );

    // Fixture 3: Open Shadow DOM
    const f3 = await runExtract('/shadow', 'article', 5000);
    assert(f3.extracted.content.includes('Shadow Root Report'), 'Shadow DOM text missed');

    // Fixture 4: JSON-LD + Hidden modal / cookie noise stripped
    const f4 = await runExtract('/jsonld', 'article', 5000);
    assert(f4.extracted.metadata?.author === 'Ada Lovelace' && f4.extracted.metadata?.price === 'USD 1999', 'JSON-LD metadata missed');
    assert(!f4.extracted.content.includes('SECRET-HIDDEN-MODAL') && !f4.extracted.content.includes('tracking cookies noise'), 'Hidden/cookie noise leaked');

    // Fixture 5: Scrolled Viewport + Same-origin Iframe
    await page.goto(`${baseUrl}/scroll`, { waitUntil: 'load' });
    await page.evaluate(() => window.scrollTo(0, 1400));
    const t5Start = performance.now();
    const f5Ex = await page.evaluate(
      ({ fn, m, lim }) => eval(`(${fn})`)(m, lim),
      { fn: extractFnStr, m: 'article', lim: 5000 },
    );
    const f5Ms = Number((performance.now() - t5Start).toFixed(2));
    assert(f5Ex.viewportExcerpt?.includes('VIEWPORT-SCROLL-MARKER-331'), `Viewport marker missed: ${f5Ex.viewportExcerpt}`);
    assert(f5Ex.fullContent?.includes('IFRAME-SIGNAL-772'), 'Same-origin iframe missed');

    // Fixture 6: Highlight Selection + SPA pushState update
    await page.goto(`${baseUrl}/spa`, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => {
      const el = document.getElementById('sel-target');
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    });
    const f6Sel = await page.evaluate(({ fn }) => eval(`(${fn})`)('auto', 5000), { fn: extractFnStr });
    assert(f6Sel.mode === 'selection' && f6Sel.content.includes('SELECTION-ALPHA-99'), 'Selection mode missed');
    await page.evaluate(() => {
      window.getSelection()?.removeAllRanges();
      history.pushState({}, '', '/spa/v2-route');
      document.getElementById('app').innerHTML = '<h1>SPA Route Two</h1><p>Hydrated route two payload SPA-ROUTE-551.</p>';
    });
    const f6Spa = await page.evaluate(({ fn }) => eval(`(${fn})`)('auto', 5000), { fn: extractFnStr });
    assert(f6Spa.url.endsWith('/spa/v2-route') && f6Spa.content.includes('SPA-ROUTE-551'), 'SPA pushState extraction failed');

    // Fixture 7: Healthy 200 OK article with footer g-recaptcha (0% false positive)
    const fpCheck = webSearchMod.detectAntiBotChallenge(fixtures.recaptchaFooterHtml, 200);
    assert(fpCheck === false, 'False-positive CAPTCHA on 200 OK article with g-recaptcha footer!');
    const f7 = await runExtract('/recaptcha-200', 'article', 8000);
    assert(f7.extracted.content.includes('WebGPU Compute Pipelines'), '200 article extraction failed');

    // Fixture 8: Resolved Markdown links + fenced code language
    const f8 = await runExtract('/links-code', 'article', 5000);
    assert(
      f8.extracted.content.includes(`[WGSL Spec Reference](${baseUrl}/docs/wgsl-spec)`) &&
        f8.extracted.content.includes('```rust\nfn dispatch_gemv'),
      `Links/code Markdown failed: ${f8.extracted.content}`,
    );

    const fixtureMetrics = [
      { id: 'deep_doc_bm25_table', extractMs: f1.extractMs, fullChars: f1.extracted.charCount, previewChars: f1.extracted.content.length, needleRecalled: true },
      { id: 'multi_article_thread_12x', extractMs: f2.extractMs, fullChars: f2.extracted.charCount, all12Posts: true },
      { id: 'open_shadow_dom', extractMs: f3.extractMs, fullChars: f3.extracted.charCount, extracted: true },
      { id: 'jsonld_and_hidden_noise_strip', extractMs: f4.extractMs, zeroNoiseLeak: true, metadataKeys: Object.keys(f4.extracted.metadata || {}) },
      { id: 'scrolled_viewport_and_iframe', extractMs: f5Ms, viewportCaptured: true, iframeCaptured: true },
      { id: 'selection_and_spa_pushstate', spaUrl: f6Spa.url, selectionOk: true, spaRouteOk: true },
      { id: 'recaptcha_footer_200_no_false_positive', extractMs: f7.extractMs, falsePositiveCaptcha: false },
      { id: 'resolved_links_and_fenced_code', extractMs: f8.extractMs, preservedLinksAndLang: true },
    ];
    console.log(`  [Page Reader V2] 8/8 archetypes PASSED (median extract ${f1.extractMs}ms, zero false positives)`);

    // 2. Live Zero-Key Direct-Chromium SERP + Parallel Scrape Eval
    const serpResults = await runLiveSerpEval(context, webSearchMod, htmlDistillerMod, bm25Mod);
    const serpSuccessCount = serpResults.filter((r) => r.resultCount >= 3).length;
    const scrapedSuccessCount = serpResults.filter((r) => r.scraped.some((s) => s.excerptChars > 200)).length;
    assert(serpSuccessCount >= 2, `Expected >= 2/3 live Chromium SERP queries to succeed, got ${serpSuccessCount}`);
    assert(scrapedSuccessCount >= 2, `Expected >= 2/3 live queries to produce distilled source excerpts, got ${scrapedSuccessCount}`);

    const payload = {
      runId: '27-firecrawl-local-search-page-eval',
      track: 'track6-firecrawl-local-eval',
      recordedAt: new Date().toISOString(),
      optionAFailClosedVerified: true,
      pageReaderV2Fixtures: fixtureMetrics,
      liveChromiumSerpAndScrape: serpResults,
      summary: {
        pageFixturesPassed: '8/8',
        serpQueriesSucceeded: `${serpSuccessCount}/3`,
        queriesWithDistilledSources: `${scrapedSuccessCount}/3`,
        falsePositiveCaptchaRatePct: 0,
        apiKeysRequired: 0,
      },
      status: 'ok',
    };
    fs.mkdirSync(RUNS_DIR, { recursive: true });
    const outFile = path.join(RUNS_DIR, '27-firecrawl-local-search-page-eval.json');
    fs.writeFileSync(outFile, JSON.stringify(payload, null, 2));
    console.log(`Saved ${outFile}\nALL FIRECRAWL-LOCAL EVALS PASSED.`);
  } finally {
    await context.close().catch(() => {});
    server.close();
    fs.rmSync(PROFILE_DIR, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('Firecrawl-local eval failed:', err);
  process.exit(1);
});
