import vm from 'node:vm';
import {
  createBaselineHarnessTarget,
  createBaselineToolsTarget,
  createBaselineVoiceTarget,
  runSuite,
} from '@yours/evals';
import { GuardFirewall } from '@yours/guard/policy';
import {
  buildSpotlightedBlock,
  sanitizeUnicode,
  sanitizeUntrustedContent,
  scoreInjectionRisk,
} from '@yours/guard/sanitizer';
import { ContextCompactor } from '@yours/harness/compaction';
import { HybridMemoryEngine } from '@yours/memory';
import { distillHtmlToMarkdown } from '@yours/tools/extractors';
import { parseToolCalls } from '@yours/tools/protocol';
import { runSandbox } from '@yours/tools/sandbox';
import { LocalVoicePipeline } from '@yours/voice/pipeline';
import { KokoroTtsRunner, NullAudioPlaybackSink, TtsEngine } from '@yours/voice/tts';
import { VadEngine } from '@yours/voice/vad';

const strictLive = process.argv.includes('--strict-live');

const baselineTools = createBaselineToolsTarget();
const baselineHarness = createBaselineHarnessTarget();
const baselineVoice = createBaselineVoiceTarget();
const firewall = new GuardFirewall();

const HIDDEN_SVG_RE = /<text\b[^>]*opacity\s*=\s*["']0(?:\.0+)?["'][^>]*>|<foreignObject\b/i;
const VISIBLE_SVG_TEXT_RE = /<(text|tspan)\b([^>]*)>([\s\S]*?)<\/\1>/gi;
const SVG_TEXT_HIDDEN_ATTR_RE =
  /(?:opacity\s*=\s*["']0(?:\.0+)?["']|visibility\s*=\s*["']hidden["']|display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?(?:\s*;|\s*$)|font-size\s*:\s*0)/i;
const SPOOFED_DELIMITER_RE =
  /<\|im_start\|>[\s\S]*?(?=<\|im_end\|>|$)|<\/?tool_call>[\s\S]*?<\/tool_call>|<\/?tool_call>|<\/?tool_response>|\[INST\][\s\S]*?\[\/INST\]|<\|im_end\|>/gi;
const CONFIRMED_ATTACK_SIGNALS = new Set([
  'hidden-css',
  'svg-foreign-text',
  'hashjack',
  'fake-tool-tag',
  'zero-width-unicode',
]);

function sanitizeWithYoursGuard(rawInput, sourceUrl) {
  const u = sanitizeUnicode(rawInput);
  const preNormalized = u.text.replace(VISIBLE_SVG_TEXT_RE, (full, _tag, attrs, inner) =>
    SVG_TEXT_HIDDEN_ATTR_RE.test(attrs) ? full : ` <span>${inner}</span> `,
  );
  const c = sanitizeUntrustedContent(preNormalized, { sourceUrl });

  let cleanText = c.cleanText;
  const extraSignals = [];
  const extraFlags = [];

  if (SPOOFED_DELIMITER_RE.test(cleanText)) {
    SPOOFED_DELIMITER_RE.lastIndex = 0;
    cleanText = cleanText.replace(SPOOFED_DELIMITER_RE, ' ').replace(/[ \t]+\n/g, '\n').trim();
    extraSignals.push('fake-tool-tag');
    extraFlags.push('spoofed-chat-or-tool-delimiter');
  }

  const hasHiddenSvg = HIDDEN_SVG_RE.test(rawInput);
  const strippedSignals = Array.from(
    new Set([...u.strippedSignals, ...c.strippedSignals, ...extraSignals]),
  ).filter((sig) => sig !== 'svg-foreign-text' || hasHiddenSvg);

  const r = scoreInjectionRisk({
    visibleText: cleanText,
    strippedSignals,
  });
  let risk = Math.max(c.injectionRiskScore, r.injectionRiskScore);
  if (strippedSignals.some((sig) => CONFIRMED_ATTACK_SIGNALS.has(sig))) {
    risk = Math.max(risk, 0.45);
  }

  return {
    cleanText,
    spotlightedPromptBlock: buildSpotlightedBlock(cleanText, {
      origin: sourceUrl || 'unknown',
      injectionRiskScore: risk,
    }),
    strippedSignals,
    injectionRiskScore: risk,
    flaggedPatterns: Array.from(new Set([...c.flaggedPatterns, ...r.flaggedPatterns, ...extraFlags])),
    tainted: true,
  };
}

const nodeVmQuickJsAdapter = {
  async execute(code, { timeoutMs }) {
    try {
      const context = vm.createContext(Object.freeze({ Math, JSON }), {
        codeGeneration: { strings: false, wasm: false },
      });
      const result = vm.runInContext(code, context, { timeout: timeoutMs ?? 100 });
      return { ok: true, result };
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) };
    }
  },
};

const targets = {
  tools: {
    distillAom(rawHtml, url) {
      const res = distillHtmlToMarkdown(rawHtml, url, 16000);
      return [res.title, res.markdown].filter(Boolean).join('\n\n');
    },
    validateToolCall(toolName, args) {
      return baselineTools.validateToolCall(toolName, args);
    },
    async runSandbox(code) {
      const res = await runSandbox(
        { mode: 'js', code },
        { quickJsAdapter: nodeVmQuickJsAdapter },
      );
      if (res.ok) {
        return { blocked: false, output: res.result };
      }
      return { blocked: true, error: res.error };
    },
    extractMultiHop(question, tabs) {
      return baselineTools.extractMultiHop(question, tabs);
    },
  },

  harness: {
    async compactMessages(messages, maxTargetTokens) {
      const compactor = new ContextCompactor({
        maxContextTokens: Math.max(512, maxTargetTokens),
        reservedOutputTokens: Math.max(0, 512 - maxTargetTokens),
        safetyMarginTokens: 0,
        summaryMaxTokens: 96,
        preserveRecentMessages: 2,
      });
      const sysMsg = messages.find((m) => m.role === 'system')?.content;
      const nonSys = messages.filter((m) => m.role !== 'system');
      const plan = await compactor.compact({
        systemPrompt: sysMsg,
        messages: nonSys,
      });
      return plan.compiledMessages;
    },
    recoverToolCallJson(toolName, rawChunks) {
      const joined = rawChunks.join('');
      const parsed = parseToolCalls(joined);
      if (parsed.toolCalls.length > 0) {
        return { ok: true, arguments: parsed.toolCalls[0].arguments };
      }
      return baselineHarness.recoverToolCallJson(toolName, rawChunks);
    },
    runtime: baselineHarness.runtime,
  },

  memory() {
    return new HybridMemoryEngine({ includeBuiltinBlocklist: true });
  },

  guard: {
    sanitize: sanitizeWithYoursGuard,
    evaluatePolicy(context) {
      return firewall.evaluate(context);
    },
  },

  voice: {
    chunkSentences: baselineVoice.chunkSentences,
    detectVadSegments: baselineVoice.detectVadSegments,
    createPipeline() {
      const noopSource = {
        sampleRate: 16000,
        isActive: false,
        async start() {},
        async stop() {},
      };
      const tts = new TtsEngine({
        runner: new KokoroTtsRunner({
          infer: async () => new Float32Array(4800),
        }),
        sink: new NullAudioPlaybackSink({ simulateDuration: true }),
      });
      return new LocalVoicePipeline({
        inputSource: noopSource,
        vad: new VadEngine(),
        tts,
      });
    },
  },
};

const referenceScorecard = await runSuite('all');
const liveScorecard = await runSuite('all', { targets });

const liveFailed = liveScorecard.cases
  .filter((c) => !c.passed)
  .map((c) => ({
    id: c.id,
    suite: c.suite,
    error: c.error,
    metrics: c.metrics,
  }));

console.log(
  JSON.stringify(
    {
      schemaVersion: liveScorecard.schemaVersion,
      reference: {
        total: referenceScorecard.summary.total,
        passed: referenceScorecard.summary.passed,
        failed: referenceScorecard.summary.failed,
        aggregateScore: referenceScorecard.summary.aggregateScore,
      },
      live: {
        total: liveScorecard.summary.total,
        passed: liveScorecard.summary.passed,
        failed: liveScorecard.summary.failed,
        aggregateScore: liveScorecard.summary.aggregateScore,
        suiteMetrics: liveScorecard.summary.suiteMetrics,
        failedCases: liveFailed,
      },
    },
    null,
    2,
  ),
);

const controlledSuiteFailed = liveFailed.some(
  (c) => c.suite === 'tools' || c.suite === 'guard' || c.suite === 'voice',
);

if (
  referenceScorecard.summary.failed > 0 ||
  controlledSuiteFailed ||
  (strictLive && liveScorecard.summary.failed > 0)
) {
  process.exitCode = 1;
}
