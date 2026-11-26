import type { GuardDecision, GuardEngine, GuardPolicyContext, SanitizedContent } from '@yours/harness';
import {
  formatApprovalPrompt,
  GuardFirewall,
  inspectOriginSafety,
  type ApprovalPromptRequest,
  type OriginSafetyResult,
} from '@yours/guard/policy';
import {
  buildSpotlightedBlock,
  sanitizeUnicode,
  sanitizeUntrustedContent,
  scoreInjectionRisk,
} from '@yours/guard/sanitizer';

export interface ExtensionGuardEngine extends GuardEngine {
  readonly firewall: GuardFirewall;
  formatApproval(context: GuardPolicyContext, decision: GuardDecision): ApprovalPromptRequest;
  inspectOrigin(rawUrlOrOrigin: string): OriginSafetyResult;
  recordTrustedPrivateData(value: string, origin: string, toolName?: string): void;
  recordUntrustedOutput(toolName: string, text: string, origin: string, injectionRiskScore?: number): void;
}

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

function unwrapBenignVisibleSvgText(html: string): string {
  return html.replace(VISIBLE_SVG_TEXT_RE, (full, _tag: string, attrs: string, inner: string) => {
    if (SVG_TEXT_HIDDEN_ATTR_RE.test(attrs)) {
      return full;
    }
    return ` <span>${inner}</span> `;
  });
}

/**
 * Multi-pass IDPI sanitizer combining Unicode zero-width/bidi/tag stripping,
 * HTML/CSS/SVG/#HashJack neutralization, heuristic injection risk scoring, and
 * structural spotlighting delimiters.
 */
export function sanitizeUntrustedText(rawInput: string, sourceUrl?: string): SanitizedContent {
  const unicodeResult = sanitizeUnicode(rawInput);
  const preNormalized = unwrapBenignVisibleSvgText(unicodeResult.text);
  const contentResult = sanitizeUntrustedContent(preNormalized, {
    sourceUrl,
  });

  let cleanText = contentResult.cleanText;
  const extraSignals: string[] = [];
  const extraFlags: string[] = [];

  if (SPOOFED_DELIMITER_RE.test(cleanText)) {
    SPOOFED_DELIMITER_RE.lastIndex = 0;
    cleanText = cleanText.replace(SPOOFED_DELIMITER_RE, ' ').replace(/[ \t]+\n/g, '\n').trim();
    extraSignals.push('fake-tool-tag');
    extraFlags.push('spoofed-chat-or-tool-delimiter');
  }

  const hasHiddenSvg = HIDDEN_SVG_RE.test(rawInput);
  const combinedSignals = Array.from(
    new Set([...unicodeResult.strippedSignals, ...contentResult.strippedSignals, ...extraSignals]),
  ).filter((sig) => sig !== 'svg-foreign-text' || hasHiddenSvg);

  const risk = scoreInjectionRisk({
    visibleText: cleanText,
    strippedSignals: combinedSignals,
  });
  let combinedRisk = Math.max(contentResult.injectionRiskScore, risk.injectionRiskScore);
  if (combinedSignals.some((sig) => CONFIRMED_ATTACK_SIGNALS.has(sig))) {
    combinedRisk = Math.max(combinedRisk, 0.45);
  }

  const combinedFlags = Array.from(
    new Set([...contentResult.flaggedPatterns, ...risk.flaggedPatterns, ...extraFlags]),
  );
  const spotlightedPromptBlock = buildSpotlightedBlock(cleanText, {
    origin: sourceUrl ?? 'unknown',
    injectionRiskScore: combinedRisk,
  });

  return {
    cleanText,
    spotlightedPromptBlock,
    strippedSignals: combinedSignals,
    injectionRiskScore: combinedRisk,
    flaggedPatterns: combinedFlags,
    tainted: true,
  };
}

/**
 * Instantiates a `@yours/guard` `GuardFirewall` implementing `@yours/harness`'s `GuardEngine`
 * contract for IDPI policy evaluation and untrusted output spotlighting.
 */
export function createExtensionGuardEngine(options?: {
  customBlockedOrigins?: ReadonlyArray<string | RegExp>;
  customAllowedOrigins?: ReadonlyArray<string | RegExp>;
}): ExtensionGuardEngine {
  const firewall = new GuardFirewall({
    customBlockedOrigins: options?.customBlockedOrigins,
    customAllowedOrigins: options?.customAllowedOrigins,
  });

  return {
    firewall,
    evaluateToolCall(context: GuardPolicyContext): GuardDecision {
      // Align argument types cleanly between @yours/harness and @yours/guard contracts
      return firewall.evaluate(context as unknown as Parameters<GuardFirewall['evaluate']>[0]);
    },
    sanitizeToolOutput(_toolName: string, rawText: string, sourceOrigin?: string): SanitizedContent {
      const sanitized = sanitizeUntrustedText(rawText, sourceOrigin);
      firewall.taintTracker.trackUntrustedContent(sanitized.cleanText, {
        origin: sourceOrigin ?? 'unknown',
        sourceKind: 'web-page',
        toolName: _toolName,
        injectionRiskScore: sanitized.injectionRiskScore,
      });
      return sanitized;
    },
    formatApproval(context: GuardPolicyContext, decision: GuardDecision): ApprovalPromptRequest {
      return formatApprovalPrompt(
        context as unknown as Parameters<typeof formatApprovalPrompt>[0],
        decision,
      );
    },
    inspectOrigin(rawUrlOrOrigin: string): OriginSafetyResult {
      return inspectOriginSafety(rawUrlOrOrigin, {
        customBlockedOrigins: options?.customBlockedOrigins,
        customAllowedOrigins: options?.customAllowedOrigins,
      });
    },
    recordTrustedPrivateData(value: string, origin: string, toolName?: string): void {
      firewall.taintTracker.trackPrivateData(value, origin, toolName);
    },
    recordUntrustedOutput(
      toolName: string,
      text: string,
      origin: string,
      injectionRiskScore?: number,
    ): void {
      firewall.taintTracker.trackUntrustedContent(text, {
        origin,
        sourceKind: 'tool-output',
        toolName,
        injectionRiskScore,
      });
    },
  };
}

export { formatApprovalPrompt, GuardFirewall, inspectOriginSafety };
