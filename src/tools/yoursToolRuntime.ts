import type {
  ToolDefinition as HarnessToolDefinition,
  ToolExecutionResult as HarnessToolExecutionResult,
  ToolExecutor,
  ToolInvocationContext as HarnessToolInvocationContext,
} from '@yours/harness';
import {
  ChromeExtensionAdapter,
  createToolRuntime,
  type ChromeExtensionAdapterOptions,
  type ToolRuntime,
  type ToolRuntimeConfig,
} from '@yours/tools/adapters';
import {
  detectPendingToolCall,
  formatToolResultMessage,
  parseOpenAIToolCalls,
  parseToolCalls,
  ToolCallJsonGrammar,
} from '@yours/tools/protocol';
import {
  CANONICAL_TOOL_DEFINITIONS,
  CANONICAL_TOOL_MAP,
  formatToolsForSystemPrompt,
  resolveToolAlias,
  toPragmaToolBundle,
  toQwenToolSchemas,
} from '@yours/tools/schemas';
import type {
  CanonicalToolName,
  FetchedSourceResult,
  MemoryProvider,
  ToolExecutionResult as CanonicalToolExecutionResult,
  ToolInvocationContext,
  WebSearchResult,
} from '@yours/tools';
import {
  isSensitiveDomainBlockedError,
  type ExtensionMemoryClient,
} from '../memory/memoryClient.ts';
import {
  createExtensionGuardEngine,
  sanitizeUntrustedText,
  type ExtensionGuardEngine,
} from './guardBridge.ts';

export {
  CANONICAL_TOOL_DEFINITIONS,
  CANONICAL_TOOL_MAP,
  detectPendingToolCall,
  formatToolResultMessage,
  formatToolsForSystemPrompt,
  parseOpenAIToolCalls,
  parseToolCalls,
  resolveToolAlias,
  ToolCallJsonGrammar,
  toPragmaToolBundle,
  toQwenToolSchemas,
};

export interface ExtensionToolRuntimeOptions {
  adapter?: ChromeExtensionAdapterOptions;
  memoryProvider?: ExtensionMemoryClient | MemoryProvider;
  guardEngine?: ExtensionGuardEngine;
  evaluatePolicy?: ToolRuntimeConfig['evaluatePolicy'];
  onApprovalRequired?: ToolRuntimeConfig['onApprovalRequired'];
}

export interface SanitizedCanonicalToolExecutionResult extends CanonicalToolExecutionResult {
  injectionRiskScore?: number;
  strippedSignals?: string[];
  flaggedPatterns?: string[];
}

export interface ExtensionToolRuntime {
  readonly adapter: ChromeExtensionAdapter;
  readonly runtime: ToolRuntime;
  readonly guardEngine: ExtensionGuardEngine;
  execute(
    rawCall: { name: string; arguments?: Record<string, unknown> },
    partialCtx?: Partial<ToolInvocationContext> & { skipSpotlight?: boolean },
  ): Promise<SanitizedCanonicalToolExecutionResult>;
}

const DEFAULT_INJECTION_SPOTLIGHT_MARKER = '[BEGIN UNTRUSTED CONTENT';

function sanitizeToolExecutionResult(
  result: CanonicalToolExecutionResult,
  guardEngine: ExtensionGuardEngine,
): SanitizedCanonicalToolExecutionResult {
  if (!result.tainted || !result.formatted) {
    return result;
  }

  // Avoid double-wrapping content that is already wrapped by GuardEngine/Sanitizer
  if (result.formatted.includes(DEFAULT_INJECTION_SPOTLIGHT_MARKER)) {
    return result;
  }

  const sanitized = sanitizeUntrustedText(result.formatted, result.sourceOrigin);
  guardEngine.recordUntrustedOutput(
    result.toolName,
    sanitized.cleanText,
    result.sourceOrigin ?? 'unknown',
    sanitized.injectionRiskScore,
  );

  return {
    ...result,
    formatted: sanitized.spotlightedPromptBlock,
    injectionRiskScore: sanitized.injectionRiskScore,
    strippedSignals: sanitized.strippedSignals,
    flaggedPatterns: sanitized.flaggedPatterns,
  };
}

function resolveCanonicalOrFallback(name: string): CanonicalToolName {
  return resolveToolAlias(name) ?? (name as CanonicalToolName);
}

/**
 * Instantiates the canonical `@yours/tools` MV3 runtime backed by `ChromeExtensionAdapter`,
 * wrapped with `@yours/guard` IDPI sanitization/spotlighting and `@yours/memory` sensitive-domain protection.
 */
export function createExtensionToolRuntime(
  options?: ExtensionToolRuntimeOptions,
): ExtensionToolRuntime {
  const guardEngine = options?.guardEngine ?? createExtensionGuardEngine();
  const adapter = new ChromeExtensionAdapter(options?.adapter);

  const memoryClient =
    options?.memoryProvider &&
    'isUrlBlocked' in options.memoryProvider &&
    typeof options.memoryProvider.isUrlBlocked === 'function'
      ? options.memoryProvider
      : undefined;

  const runtime = createToolRuntime(adapter, {
    memoryProvider: options?.memoryProvider as unknown as MemoryProvider | undefined,
    quickJsAdapter: options?.adapter?.quickJsAdapter,
    evaluatePolicy: options?.evaluatePolicy,
    onApprovalRequired: options?.onApprovalRequired,
  });

  return {
    adapter,
    runtime,
    guardEngine,
    async execute(
      rawCall: { name: string; arguments?: Record<string, unknown> },
      partialCtx?: Partial<ToolInvocationContext> & { skipSpotlight?: boolean },
    ): Promise<SanitizedCanonicalToolExecutionResult> {
      const canonical = resolveCanonicalOrFallback(rawCall.name);
      const args = rawCall.arguments ?? {};

      // Enforce @yours/memory sensitive-domain rules proactively when URLs are supplied
      if (memoryClient) {
        const candidateUrls: string[] = [];
        if (typeof args.source_url === 'string') candidateUrls.push(args.source_url);
        if (typeof args.url === 'string') candidateUrls.push(args.url);
        if (Array.isArray(args.urls)) {
          for (const u of args.urls) {
            if (typeof u === 'string') candidateUrls.push(u);
          }
        }
        if (typeof partialCtx?.origin === 'string' && /^https?:/i.test(partialCtx.origin)) {
          candidateUrls.push(partialCtx.origin);
        }

        if (canonical === 'memory_save') {
          for (const url of candidateUrls) {
            const check = await memoryClient.isUrlBlocked(url);
            if (check.blocked) {
              return {
                ok: false,
                toolName: canonical,
                formatted: `Memory save blocked: sensitive domain (${check.reason ?? check.rule ?? url}).`,
                tainted: false,
                error: check.reason ?? 'Sensitive domain blocked by @yours/memory privacy policy.',
              };
            }
          }
        }
      }

      try {
        const rawResult = await runtime.execute(
          { name: rawCall.name, arguments: args },
          {
            environment: partialCtx?.environment ?? 'chrome-mv3',
            sessionId: partialCtx?.sessionId ?? 'extension-session',
            origin: partialCtx?.origin,
            abortSignal: partialCtx?.abortSignal,
          },
        );
        if (partialCtx?.skipSpotlight) {
          return rawResult;
        }
        return sanitizeToolExecutionResult(rawResult, guardEngine);
      } catch (err: unknown) {
        if (isSensitiveDomainBlockedError(err)) {
          const message = err instanceof Error ? err.message : String(err);
          return {
            ok: false,
            toolName: canonical,
            formatted: `Blocked by local privacy rules: ${message}`,
            tainted: false,
            error: message,
          };
        }
        const message = err instanceof Error ? err.message : String(err);
        return {
          ok: false,
          toolName: canonical,
          formatted: `Tool ${canonical} failed: ${message}`,
          tainted: false,
          error: message,
        };
      }
    },
  };
}

export interface BackgroundToolExecResponse {
  ok: boolean;
  toolName?: string;
  data?: unknown;
  formatted?: string;
  tainted?: boolean;
  sourceOrigin?: string;
  error?: string;
  needsPermission?: boolean;
  requiresApproval?: boolean;
  blockedByCaptcha?: boolean;
  manualUrl?: string;
}

export interface SidepanelToolExecutorOptions {
  execBackgroundTool: (call: {
    name: string;
    arguments?: Record<string, unknown>;
  }) => Promise<BackgroundToolExecResponse>;
  getMemoryClient?: () => ExtensionMemoryClient | Promise<ExtensionMemoryClient>;
  guardEngine?: ExtensionGuardEngine;
  /**
   * When `@yours/harness` `runHarnessTurn` is configured with `guard`, it already runs
   * `guard.sanitizeToolOutput` on tainted tool results. Set `true` to avoid double-wrapping
   * `[BEGIN UNTRUSTED CONTENT ...]` markers inside themselves.
   */
  skipGuardSanitize?: boolean;
  onSourcesExtracted?: (sources: FetchedSourceResult[]) => void;
  onResearchPermissionNeeded?: () => void;
  localAdapterOptions?: ChromeExtensionAdapterOptions;
}

function extractFetchedSourcesFromData(
  data: unknown,
  canonicalToolName: CanonicalToolName,
): FetchedSourceResult[] {
  if (canonicalToolName !== 'search_web_tab' && canonicalToolName !== 'fetch_background_tab') {
    return [];
  }
  if (!data || typeof data !== 'object') {
    return [];
  }
  if (Array.isArray(data)) {
    return data.filter(
      (item): item is FetchedSourceResult =>
        Boolean(item) &&
        typeof item === 'object' &&
        typeof (item as { url?: unknown }).url === 'string' &&
        typeof (item as { excerpt?: unknown }).excerpt === 'string',
    );
  }
  const maybeSearch = data as Partial<WebSearchResult>;
  if (Array.isArray(maybeSearch.scrapedSources)) {
    return maybeSearch.scrapedSources.filter(
      (item): item is FetchedSourceResult =>
        Boolean(item) &&
        typeof item === 'object' &&
        typeof item.url === 'string' &&
        typeof item.excerpt === 'string',
    );
  }
  return [];
}

/**
 * Sidepanel-side `@yours/harness` `ToolExecutor`:
 * - Executes `memory_recall`, `memory_save`, and `run_sandbox` locally with worker-backed
 *   `@yours/memory` / QuickJS-safe `@yours/tools` sandbox (plus IDPI/sensitive-domain checks).
 * - Dispatches Chrome tab/web/document tools (`read_active_tab`, `list_tabs`, `read_tab`,
 *   `interact_page`, `read_document`, `search_web_tab`, `fetch_background_tab`) over the MV3
 *   service worker `EXEC_TOOL` bridge.
 */
export function createSidepanelToolExecutor(options: SidepanelToolExecutorOptions): ToolExecutor {
  const guardEngine = options.guardEngine ?? createExtensionGuardEngine();
  const localSandboxRuntime = createExtensionToolRuntime({
    adapter: options.localAdapterOptions,
    guardEngine,
  });

  return {
    listTools(): HarnessToolDefinition[] {
      return CANONICAL_TOOL_DEFINITIONS as unknown as HarnessToolDefinition[];
    },

    getTool(name: string): HarnessToolDefinition | undefined {
      const canonical = resolveToolAlias(name);
      if (!canonical) return undefined;
      return CANONICAL_TOOL_MAP.get(canonical) as unknown as HarnessToolDefinition | undefined;
    },

    async execute(
      name: string,
      args: Record<string, unknown>,
      context: HarnessToolInvocationContext,
    ): Promise<HarnessToolExecutionResult> {
      const canonical = resolveToolAlias(name) ?? (name as CanonicalToolName);

      if (canonical === 'memory_recall' || canonical === 'memory_save') {
        if (!options.getMemoryClient) {
          return {
            ok: false,
            toolName: canonical,
            formatted: 'Local memory provider is not available in this session.',
            tainted: false,
            error: 'Missing getMemoryClient handler.',
          };
        }

        const memoryClient = await options.getMemoryClient();
        await memoryClient.init();
        const memoryRuntime = createExtensionToolRuntime({
          adapter: options.localAdapterOptions,
          memoryProvider: memoryClient,
          guardEngine,
        });
        const res = await memoryRuntime.execute({ name: canonical, arguments: args }, context);
        return res;
      }

      if (canonical === 'run_sandbox') {
        return localSandboxRuntime.execute({ name: canonical, arguments: args }, context);
      }

      // If `read_document` is invoked purely on inline content (no URL), execute locally.
      if (canonical === 'read_document' && typeof args.url !== 'string') {
        return localSandboxRuntime.execute({ name: canonical, arguments: args }, context);
      }

      const bgRes = await options.execBackgroundTool({
        name: canonical,
        arguments: args,
      });

      if (bgRes.needsPermission) {
        options.onResearchPermissionNeeded?.();
      }

      if (bgRes.ok && bgRes.data) {
        const sources = extractFetchedSourcesFromData(bgRes.data, canonical);
        if (sources.length > 0) {
          options.onSourcesExtracted?.(sources);
        }
      }

      const canonicalDef = CANONICAL_TOOL_MAP.get(canonical);
      const isTainted =
        typeof bgRes.tainted === 'boolean'
          ? bgRes.tainted
          : Boolean(canonicalDef?.safety.untrustedContentHint);

      const formattedRaw =
        typeof bgRes.formatted === 'string' && bgRes.formatted.length > 0
          ? bgRes.formatted
          : bgRes.ok
            ? JSON.stringify(bgRes.data ?? {})
            : bgRes.error ?? `Tool ${canonical} failed.`;

      const result: CanonicalToolExecutionResult = {
        ok: bgRes.ok,
        toolName: canonical,
        data: bgRes.data,
        formatted: formattedRaw,
        tainted: isTainted,
        sourceOrigin: bgRes.sourceOrigin,
        error: bgRes.error,
        needsPermission: bgRes.needsPermission,
        requiresApproval: bgRes.requiresApproval,
        blockedByCaptcha: bgRes.blockedByCaptcha,
        manualUrl: bgRes.manualUrl,
      };

      if (options.skipGuardSanitize) {
        return result;
      }

      return sanitizeToolExecutionResult(result, guardEngine);
    },
  };
}
