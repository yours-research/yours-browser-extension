import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import type { GenerateRequest, HarnessMessage, MemoryRecord, RuntimeAdapter, ToolExecutor } from '@yours/harness';
import { buildSlot0Envelope, ContextCompactor, type CompactionInput } from '@yours/harness/compaction';
import { ActiveTabSpeculativePrefillController } from '@yours/harness/engine';
import { useYoursChat } from '@yours/harness/react';
import { IndexedDbStore, InMemoryStore, type HarnessStore } from '@yours/harness/storage';
import { CANONICAL_TOOL_DEFINITIONS } from '@yours/tools/schemas';
import type { FetchedSourceResult } from '@yours/tools';
import {
  FloatingComposer,
  FloatingComposerProps,
  GlobeIcon,
  HardwareStatusStrip,
  HookSidebarList,
  MemoryInspectorList,
  MicIcon,
  ModelDownloadGate,
  NotchDrawerSearch,
  PageIcon,
  QuotedContextBanner,
  SidePanelShell,
  SparkTrendIcon,
  ToastPill,
  TopNotch,
  VoiceDictationPill,
  type HookSidebarItem,
  type MemoryRecord as UiMemoryRecord,
  type MemoryTier as UiMemoryTier,
  type ModelCatalogChoice as UiModelChoice,
  type ModelDownloadProgress as UiDownloadProgress,
  type PromptStarterItem,
  type SourceCitation,
  type TopNotchItem,
} from '@yours/ui';
import {
  BUNDLED_MODEL_IDS,
  DEFAULT_ENGINE_CONFIG,
  MODEL_PRESETS,
  type EngineConfig,
} from '../shared/protocol.ts';
import {
  buildActivePageContextMessage,
  buildHarnessSystemPrompt,
} from '../shared/systemPrompt.ts';
import {
  getExtensionMemoryClient,
  type ExtensionMemoryClient,
} from '../memory/memoryClient.ts';
import {
  createExtensionGuardEngine,
  sanitizeUntrustedText,
  type ExtensionGuardEngine,
} from '../tools/guardBridge.ts';
import { createSidepanelToolExecutor } from '../tools/yoursToolRuntime.ts';
import type { ExtractedPageContent } from '../tools/types.ts';
import { shouldAutoPreloadModel } from './agentController.ts';
import { ChatThread } from './components/ChatThread.tsx';
import { DiagnosticsDrawer } from './components/DiagnosticsDrawer.tsx';
import { measureDisplayRefreshRate } from './displayHz.ts';
import { OffscreenEngineClient } from './engineClient.ts';
import {
  DelegatingRuntimeAdapter,
  createChromeAiRuntimeAdapter,
  createOpenAiRuntimeAdapter,
  createWgslRuntimeAdapter,
  type OpenAiRuntimeConfig,
  type RuntimeBackendKind,
} from './runtimeFactory.ts';
import {
  SidepanelVoiceController,
  type VoiceStatusSnapshot,
} from './voiceController.ts';

declare global {
  interface Window {
    __TINYGPT_CLIENT__?: OffscreenEngineClient;
    __TINYGPT_UI__?: {
      sendPrompt: (text: string) => Promise<void>;
      setPageEnabled: (enabled: boolean) => void;
      setResearchEnabled: (enabled: boolean) => void;
      setEnableThinking: (enabled: boolean) => void;
      abortTurn: () => void;
      approveTool: () => void;
      denyTool: () => void;
      refreshActiveTab: () => void;
      getMetrics: () => {
        status: string;
        modelId: string;
        messageCount: number;
        lastAssistantText: string;
        ttftMs: number | null;
        prefillTokPerSec: number | null;
        tokensPerSec: number | null;
        activeContextTokens: number;
        prefixReused: boolean | null;
        backgroundPrefillState: string;
        activePrefixSlot: string | null;
        lastSlotRestoreMs: number | null;
        basePromptCached: boolean;
        basePromptTokens: number;
        allocatedKvCapacity: number;
        computeTier: string | null;
        displayHz: number;
        pageEnabled: boolean;
        researchEnabled: boolean;
        enableThinking: boolean;
        backendKind: RuntimeBackendKind;
        turnState: string;
        isGenerating: boolean;
        chatError: string | null;
        turnTelemetry: unknown;
        lastCompaction: unknown;
        enabledToolNames: string[];
        pendingApproval: unknown;
        sourceCount: number;
        activePage: {
          title: string;
          url: string;
          mode: string;
          chars: number;
          truncated: boolean;
          contentHash: string;
        } | null;
      };
      getTrajectory: () => {
        threadId: string | null;
        turnState: string;
        isGenerating: boolean;
        chatError: string | null;
        streamingText: string;
        streamingReasoning: string;
        messages: readonly unknown[];
        pendingToolCalls: readonly unknown[];
        pendingApproval: unknown;
        turnTelemetry: unknown;
        lastCompaction: unknown;
        sources: readonly unknown[];
        enabledToolNames: string[];
        activePage: {
          title: string;
          url: string;
          contentHash: string;
          chars: number;
        } | null;
      };
      getMemoryStats: () => Promise<{ stats: unknown; records: readonly unknown[] }>;
      rememberFact: (
        content: string,
        tags?: string[],
        tier?: 'working' | 'profile' | 'episodic' | 'document',
        sourceUrl?: string,
      ) => Promise<unknown>;
      setCompactionBudget: (maxContextTokens: number | null) => void;
      wipeMemory: () => Promise<void>;
      clearChat: () => Promise<void>;
    };
  }
}

const STARTER_PROMPTS: PromptStarterItem[] = [
  {
    id: 'summarize-page',
    label: 'Summarize active page',
    prompt: 'Summarize the key points from the active browser tab.',
    icon: 'page-scan',
  },
  {
    id: 'compare-options',
    label: 'Extract comparison table',
    prompt: 'Extract key tradeoffs on this page into a concise comparison table.',
    icon: 'compare',
  },
  {
    id: 'local-research',
    label: 'Research & cite sources',
    prompt: 'Search the web for the latest benchmarks and summarize with sources.',
    icon: 'globe',
  },
];

function computeEnabledTools(opts: {
  pageEnabled: boolean;
  pageAttached: boolean;
  pageTruncated: boolean;
  researchEnabled: boolean;
}) {
  return CANONICAL_TOOL_DEFINITIONS.filter((tool) => {
    if (tool.name === 'run_sandbox') return true;
    if (tool.name === 'read_active_tab') {
      return opts.pageEnabled && (!opts.pageAttached || opts.pageTruncated);
    }
    if (tool.name === 'list_tabs' || tool.name === 'read_tab') {
      return opts.pageEnabled;
    }
    if (tool.name === 'search_web_tab' || tool.name === 'fetch_background_tab') {
      return opts.researchEnabled;
    }
    return false;
  });
}

function createHarnessStore(): HarnessStore {
  if (typeof indexedDB !== 'undefined') {
    return new IndexedDbStore({ dbName: 'yours-extension-harness' });
  }
  return new InMemoryStore();
}

export function App() {
  const client = useMemo(() => new OffscreenEngineClient(), []);
  const engineSnapshot = useSyncExternalStore(client.subscribe, client.getSnapshot);

  const harnessStore = useMemo(() => createHarnessStore(), []);
  const memoryClient = useMemo<ExtensionMemoryClient>(() => getExtensionMemoryClient(), []);
  const guardEngine = useMemo<ExtensionGuardEngine>(() => createExtensionGuardEngine(), []);
  const voiceController = useMemo(() => new SidepanelVoiceController(), []);

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      const saved = window.localStorage.getItem('yours-theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch {
      // Ignore storage access errors
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem('yours-theme', theme);
    } catch {
      // Ignore storage access errors
    }
  }, [theme]);

  const [expandedNotchId, setExpandedNotchId] = useState<string | null>(null);
  const [composerText, setComposerText] = useState('');
  const [pageEnabled, setPageEnabled] = useState(true);
  const [researchEnabled, setResearchEnabled] = useState(false);
  const [hasResearchPermission, setHasResearchPermission] = useState(false);
  const [activePage, setActivePage] = useState<ExtractedPageContent | null>(null);
  const [quotedSelection, setQuotedSelection] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceCitation[]>([]);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const [backendKind, setBackendKind] = useState<RuntimeBackendKind>('wgsl');
  const [selectedModelId, setSelectedModelId] = useState<string>(DEFAULT_ENGINE_CONFIG.modelId);
  const [openAiConfig, setOpenAiConfig] = useState<OpenAiRuntimeConfig>({
    baseUrl: 'http://127.0.0.1:11434/v1',
    model: 'qwen2.5:7b',
    apiKey: '',
  });
  const [chromeAiAvailable, setChromeAiAvailable] = useState(false);

  const [threads, setThreads] = useState<HookSidebarItem[]>([]);
  const [threadSearch, setThreadSearch] = useState('');
  const [memoryRecords, setMemoryRecords] = useState<MemoryRecord[]>([]);
  const [memorySearch, setMemorySearch] = useState('');
  const [memoryTierFilter, setMemoryTierFilter] = useState<UiMemoryTier | 'all'>('all');
  const [voiceState, setVoiceState] = useState<VoiceStatusSnapshot>(() =>
    voiceController.getSnapshot(),
  );

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
  }, []);

  useEffect(() => {
    if (!toastMsg) return;
    const t = setTimeout(() => setToastMsg(null), 2600);
    return () => clearTimeout(t);
  }, [toastMsg]);

  const autoPreloadTriggeredRef = useRef(false);
  const userUnloadedRef = useRef(false);
  const statusRef = useRef(engineSnapshot.telemetry.status);
  statusRef.current = engineSnapshot.telemetry.status;
  const bgStateRef = useRef(engineSnapshot.telemetry.backgroundPrefillState);
  bgStateRef.current = engineSnapshot.telemetry.backgroundPrefillState;
  const profileMemoriesRef = useRef<MemoryRecord[]>([]);

  // Keep latest dynamic flags in refs so `ChatSession`'s single mount sees live changes
  const configRef = useRef<EngineConfig>(engineSnapshot.config);
  configRef.current = engineSnapshot.config;

  const pageEnabledRef = useRef(pageEnabled);
  pageEnabledRef.current = pageEnabled;

  const researchEnabledRef = useRef(researchEnabled);
  researchEnabledRef.current = researchEnabled;

  const activePageRef = useRef(activePage);
  activePageRef.current = activePage;
  const lastExtractedTruncatedRef = useRef(false);

  const backendKindRef = useRef<RuntimeBackendKind>(backendKind);
  backendKindRef.current = backendKind;

  const hasAttachedPage = Boolean(
    pageEnabled && activePage && activePage.content.trim().length > 0,
  );

  const enabledTools = useMemo(
    () =>
      computeEnabledTools({
        pageEnabled,
        pageAttached: hasAttachedPage,
        pageTruncated: Boolean(activePage?.truncated),
        researchEnabled,
      }),
    [pageEnabled, hasAttachedPage, activePage?.truncated, researchEnabled],
  );

  const modeSystemPrompt = useMemo(
    () =>
      buildHarnessSystemPrompt({
        pageEnabled,
        pageAttached: hasAttachedPage,
        researchEnabled,
        pageTruncated: Boolean(activePage?.truncated),
      }),
    [pageEnabled, hasAttachedPage, researchEnabled, activePage?.truncated],
  );

  const activePageContextMessages = useMemo<HarnessMessage[]>(() => {
    if (!hasAttachedPage || !activePage) {
      return [];
    }
    const spotlighted = sanitizeUntrustedText(activePage.content, activePage.url);
    const pageMsg = buildActivePageContextMessage({
      title: activePage.title,
      url: activePage.url,
      mode: activePage.mode,
      content: spotlighted.spotlightedPromptBlock,
    });
    return [
      {
        id: `page_${activePage.contentHash}`,
        role: 'system',
        content: pageMsg.content,
        tainted: true,
        createdAt: 0,
      },
    ];
  }, [
    hasAttachedPage,
    activePage?.contentHash,
    activePage?.content,
    activePage?.title,
    activePage?.url,
    activePage?.mode,
  ]);

  const wgslAdapter = useMemo(
    () =>
      createWgslRuntimeAdapter(client, {
        modelId: selectedModelId,
        maxContextTokens: engineSnapshot.config.wgslMaxContext,
        tier: engineSnapshot.telemetry.computeTier ?? 'tier1-f16-subgroups',
        pacingMode: engineSnapshot.config.gpuPacingMode,
      }),
    [
      client,
      selectedModelId,
      engineSnapshot.config.wgslMaxContext,
      engineSnapshot.config.gpuPacingMode,
      engineSnapshot.telemetry.computeTier,
    ],
  );

  const chromeAiAdapter = useMemo(() => createChromeAiRuntimeAdapter(), []);
  const openAiAdapter = useMemo(
    () => createOpenAiRuntimeAdapter(openAiConfig),
    [openAiConfig.baseUrl, openAiConfig.model, openAiConfig.apiKey],
  );

  const activeRuntimeRef = useRef<RuntimeAdapter>(wgslAdapter);
  activeRuntimeRef.current =
    backendKind === 'chrome-ai'
      ? chromeAiAdapter
      : backendKind === 'openai'
        ? openAiAdapter
        : wgslAdapter;

  const delegatingAdapter = useMemo(
    () =>
      new DelegatingRuntimeAdapter({
        getAdapter: () => activeRuntimeRef.current,
        transformRequest: (request: GenerateRequest): GenerateRequest => {
          const cfg = configRef.current;
          return {
            ...request,
            maxNewTokens: cfg.maxNewTokens,
            temperature: cfg.temperature,
            topP: cfg.topP,
            topK: cfg.topK,
            enableThinking: cfg.enableThinking,
            pacingMode: cfg.gpuPacingMode,
          };
        },
      }),
    [],
  );

  const handleSourcesExtracted = useCallback((fetched: FetchedSourceResult[]) => {
    setSources((prev) => {
      const seen = new Set(prev.map((s) => s.url));
      const added: SourceCitation[] = [];
      for (const item of fetched) {
        if (item.url && !seen.has(item.url) && !item.error) {
          seen.add(item.url);
          added.push({
            id: item.url,
            title: item.title || item.url,
            subtitle: (item.excerpt ?? item.title ?? item.url).slice(0, 90),
            url: item.url,
          });
        }
      }
      return added.length > 0 ? [...prev, ...added] : prev;
    });
  }, []);

  const toolExecutor = useMemo<ToolExecutor>(
    () =>
      createSidepanelToolExecutor({
        execBackgroundTool: (call) => client.execTool(call, { skipSpotlight: true }),
        getMemoryClient: () => memoryClient,
        guardEngine,
        skipGuardSanitize: true,
        onSourcesExtracted: handleSourcesExtracted,
        onResearchPermissionNeeded: () => {
          setHasResearchPermission(false);
          showToast('Grant https://*/* host access to run live web research.');
        },
      }),
    [client, memoryClient, guardEngine, handleSourcesExtracted, showToast],
  );

  const toolExecutorRef = useRef(toolExecutor);
  toolExecutorRef.current = toolExecutor;

  const stableToolExecutor = useMemo<ToolExecutor>(
    () => ({
      listTools: () => toolExecutorRef.current.listTools?.() ?? [],
      getTool: (name) => toolExecutorRef.current.getTool?.(name),
      execute: (name, args, ctx) => toolExecutorRef.current.execute(name, args, ctx),
    }),
    [],
  );

  const compactionBudgetRef = useRef<number | null>(null);

  const dynamicCompactor = useMemo(
    () =>
      ({
        compact: async (input: CompactionInput) => {
          const overrideBudget = compactionBudgetRef.current;
          const maxContextTokens =
            overrideBudget ?? Math.min(configRef.current.wgslMaxContext, 32768);
          const reservedOutputTokens = overrideBudget
            ? Math.min(256, Math.floor(maxContextTokens * 0.25))
            : Math.min(1024, configRef.current.maxNewTokens);
          const safetyMarginTokens = overrideBudget ? 32 : 128;
          const instance = new ContextCompactor({
            maxContextTokens,
            reservedOutputTokens,
            safetyMarginTokens,
            triggerRatio: overrideBudget ? 0.78 : 0.92,
            preserveRecentMessages: 2,
            summaryMaxTokens: overrideBudget ? 128 : 320,
            countTokens: (msgs) => delegatingAdapter.countTokens(msgs),
          });
          return instance.compact(input);
        },
      }) as unknown as ContextCompactor,
    [delegatingAdapter],
  );

  const chat = useYoursChat({
    adapter: delegatingAdapter,
    store: harnessStore,
    systemPrompt: modeSystemPrompt,
    contextMessages: activePageContextMessages,
    environment: 'chrome-mv3',
    tools: enabledTools as unknown as NonNullable<
      Parameters<typeof useYoursChat>[0]['tools']
    >,
    toolExecutor: stableToolExecutor,
    guard: guardEngine,
    memory: memoryClient,
    compactor: dynamicCompactor,
    enableSpeculativeTyping: true,
    maxToolSteps: 4,
    maxNewTokens: engineSnapshot.config.maxNewTokens,
    temperature: engineSnapshot.config.temperature,
    topP: engineSnapshot.config.topP,
    topK: engineSnapshot.config.topK,
    enableThinking: engineSnapshot.config.enableThinking,
    pacingMode: engineSnapshot.config.gpuPacingMode,
  });

  // Connect offscreen engine, initialize local memory, check Chrome AI & permissions
  useEffect(() => {
    window.__TINYGPT_CLIENT__ = client;
    void client.connect().then(() => {
      client.probeHardwareCatalog();
    }).catch(() => {});

    void memoryClient
      .init()
      .then(() => memoryClient.inspect({ tier: 'profile', limit: 16 }))
      .then((rows) => {
        profileMemoriesRef.current = rows;
      })
      .catch(() => {});

    measureDisplayRefreshRate()
      .then((hz: number) => client.setDisplayHz(hz))
      .catch(() => {});

    void client
      .checkResearchPermission()
      .then((granted) => {
        setHasResearchPermission(granted);
        setResearchEnabled(granted);
      })
      .catch(() => {});

    void chromeAiAdapter.isAvailable().then(setChromeAiAvailable).catch(() => {});

    const unsubVoice = voiceController.subscribe((snap) => {
      setVoiceState(snap);
    });

    return () => {
      unsubVoice();
      void voiceController.dispose();
    };
  }, [client, memoryClient, chromeAiAdapter, voiceController]);

  // Restore automatic model preloading on startup when idle/unloaded
  useEffect(() => {
    const noAutoLoad =
      typeof window !== 'undefined' &&
      new URLSearchParams(window.location.search).get('noAutoLoad') === '1';
    if (
      !noAutoLoad &&
      !autoPreloadTriggeredRef.current &&
      !userUnloadedRef.current &&
      engineSnapshot.connected &&
      backendKind === 'wgsl' &&
      shouldAutoPreloadModel(engineSnapshot.telemetry.status)
    ) {
      autoPreloadTriggeredRef.current = true;
      void client.loadModel({ modelId: selectedModelId });
    }
  }, [
    client,
    backendKind,
    selectedModelId,
    engineSnapshot.connected,
    engineSnapshot.telemetry.status,
  ]);

  // Background active-tab speculative prefill controller matching Slot 0 + contextMessages byte-for-byte
  const tabCtrl = useMemo(
    () =>
      new ActiveTabSpeculativePrefillController({
        extractActiveTab: async () => {
          if (!pageEnabledRef.current) {
            return null;
          }
          const page = await client.extractActiveTabPreview();
          if (!page || page.content.trim().length === 0 || page.error) {
            return null;
          }
          lastExtractedTruncatedRef.current = Boolean(page.truncated);
          setActivePage(page);
          if (page.selectionText && page.selectionText.trim().length > 0) {
            setQuotedSelection(page.selectionText.trim().slice(0, 280));
          }
          return {
            title: page.title,
            url: page.url,
            content: page.content,
            mode: page.mode,
            truncated: page.truncated,
          };
        },
        buildMessages: (tab) => {
          const spotlighted = sanitizeUntrustedText(tab.content, tab.url);
          const ctxMsg = buildActivePageContextMessage({
            title: tab.title,
            url: tab.url,
            mode: tab.mode ?? 'auto',
            content: spotlighted.spotlightedPromptBlock,
          });
          return [
            {
              id: `page_${tab.contentHash}`,
              role: 'system',
              content: ctxMsg.content,
              tainted: Boolean(tab.truncated) || true,
              createdAt: 0,
            },
          ];
        },
        isEngineIdle: () =>
          pageEnabledRef.current &&
          backendKindRef.current === 'wgsl' &&
          statusRef.current === 'ready' &&
          bgStateRef.current !== 'prefilling',
        onPrefill: async (slotKey, messages, meta, signal) => {
          const truncated = lastExtractedTruncatedRef.current;
          const tabTools = computeEnabledTools({
            pageEnabled: true,
            pageAttached: true,
            pageTruncated: truncated,
            researchEnabled: researchEnabledRef.current,
          });
          const slot0 = await buildSlot0Envelope({
            systemPrompt: buildHarnessSystemPrompt({
              pageEnabled: true,
              pageAttached: true,
              researchEnabled: researchEnabledRef.current,
              pageTruncated: truncated,
            }),
            tools: tabTools as unknown as NonNullable<
              NonNullable<Parameters<typeof buildSlot0Envelope>[0]>['tools']
            >,
            profileMemories: profileMemoriesRef.current,
          });
          if (signal.aborted) {
            return;
          }
          client.prefillBackground(
            slotKey,
            [
              {
                id: slot0.messages[0]!.id,
                role: 'system',
                content: slot0.messages[0]!.content,
                createdAt: 0,
              },
              ...messages.map((m) => ({
                id: m.id,
                role: m.role,
                content: m.content,
                createdAt: m.createdAt,
              })),
            ],
            { isDraft: false, url: meta.url, contentHash: meta.contentHash },
          );
        },
        onAbort: () => {
          client.abortBackgroundPrefill();
        },
      }),
    [client],
  );

  const lastWarmedSlot0KeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (pageEnabled && engineSnapshot.telemetry.status === 'ready') {
      if (!activePage || chat.messages.length === 0) {
        tabCtrl.scheduleExtraction();
      }
    } else if (!pageEnabled) {
      tabCtrl.cancelForSubmit();
      setActivePage(null);
      setQuotedSelection(null);
    }
  }, [
    pageEnabled,
    researchEnabled,
    engineSnapshot.telemetry.status,
    activePage,
    chat.messages.length,
    tabCtrl,
  ]);

  // Keep GPU Slot 0 synchronized with the active harness system prompt & tool schemas
  useEffect(() => {
    if (engineSnapshot.telemetry.status !== 'ready' || chat.isGenerating || pageEnabled) {
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      void (async () => {
        const profileMemories = await memoryClient
          .inspect({ tier: 'profile', limit: 16 })
          .catch(() => []);
        if (cancelled) return;
        const slot0 = await buildSlot0Envelope({
          systemPrompt: modeSystemPrompt,
          tools: enabledTools as unknown as NonNullable<
            Parameters<typeof buildSlot0Envelope>[0]
          >['tools'],
          profileMemories,
        });
        if (cancelled || slot0.slotKey === lastWarmedSlot0KeyRef.current) return;
        lastWarmedSlot0KeyRef.current = slot0.slotKey;
        client.prefillBackground('base-system', slot0.messages, { isDraft: false });
      })();
    }, 60);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [
    engineSnapshot.telemetry.status,
    chat.isGenerating,
    pageEnabled,
    activePage,
    modeSystemPrompt,
    enabledTools,
    memoryClient,
    client,
  ]);

  useEffect(() => {
    if (typeof chrome === 'undefined' || !chrome.tabs?.onActivated || !chrome.tabs?.onUpdated) {
      return () => {
        tabCtrl.dispose();
      };
    }
    const onActivated = () => {
      if (pageEnabledRef.current) {
        setActivePage(null);
        setQuotedSelection(null);
        tabCtrl.notifyTabChanged(true);
      }
    };
    const onUpdated = (_tabId: number, changeInfo: { status?: string; url?: string }) => {
      if (
        (changeInfo.status === 'complete' || typeof changeInfo.url === 'string') &&
        pageEnabledRef.current
      ) {
        tabCtrl.notifyTabChanged(true);
      }
    };
    chrome.tabs.onActivated.addListener(onActivated);
    chrome.tabs.onUpdated.addListener(onUpdated);
    return () => {
      chrome.tabs.onActivated.removeListener(onActivated);
      chrome.tabs.onUpdated.removeListener(onUpdated);
      tabCtrl.dispose();
    };
  }, [tabCtrl]);

  // Sync threads list when History drawer opens or active thread updates
  const refreshThreadsList = useCallback(async () => {
    try {
      const list = await (threadSearch.trim()
        ? chat.searchThreads(threadSearch.trim(), { limit: 30 }).then((hits) => hits.map((h) => h.thread))
        : chat.listThreads({ limit: 30 }));
      setThreads(
        list.map((t) => ({
          id: t.id,
          title: t.title || 'Untitled Thread',
          meta: new Date(t.updatedAt).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
          hasLiveDot: t.id === chat.thread?.id && chat.isGenerating,
        })),
      );
    } catch {
      setThreads([]);
    }
  }, [chat, threadSearch]);

  const refreshMemoryList = useCallback(async () => {
    try {
      await memoryClient.init();
      const rows = memorySearch.trim()
        ? (await memoryClient.recall({ query: memorySearch.trim(), limit: 25 })).map((h) => h.record)
        : await memoryClient.inspect({ limit: 30 });
      setMemoryRecords(rows);
    } catch {
      setMemoryRecords([]);
    }
  }, [memoryClient, memorySearch]);

  useEffect(() => {
    if (expandedNotchId === 'history') {
      void refreshThreadsList();
    } else if (expandedNotchId === 'memory') {
      void refreshMemoryList();
    }
  }, [expandedNotchId, refreshThreadsList, refreshMemoryList]);

  const handleSendText = useCallback(
    async (rawText: string) => {
      const trimmed = rawText.trim();
      if (!trimmed || chat.isGenerating) return;

      tabCtrl.cancelForSubmit();
      const composed = quotedSelection
        ? `> ${quotedSelection}\n\n${trimmed}`
        : trimmed;
      setComposerText('');
      setQuotedSelection(null);

      await chat.send(composed, {
        activeOrigin: activePage?.url || undefined,
        enableThinking: engineSnapshot.config.enableThinking,
        contextMessages: activePageContextMessages,
        groundingText:
          pageEnabled && activePage ? (activePage.fullContent ?? activePage.content) : undefined,
      });
    },
    [
      chat,
      tabCtrl,
      quotedSelection,
      activePage,
      activePageContextMessages,
      pageEnabled,
      engineSnapshot.config.enableThinking,
    ],
  );

  // Automation test & eval telemetry hooks preserved + extended for live benchmarks
  useEffect(() => {
    window.__TINYGPT_UI__ = {
      sendPrompt: (text: string) => handleSendText(text),
      setPageEnabled: (enabled: boolean) => setPageEnabled(enabled),
      setResearchEnabled: (enabled: boolean) => setResearchEnabled(enabled),
      setEnableThinking: (enabled: boolean) => {
        client.updateConfig({ enableThinking: enabled });
        chat.updateOptions({ enableThinking: enabled });
      },
      abortTurn: () => chat.abort(),
      approveTool: () => chat.approveTool(),
      denyTool: () => chat.denyTool(),
      refreshActiveTab: () => tabCtrl.notifyTabChanged(true),
      getMetrics: () => {
        const snap = client.getSnapshot();
        const assistants = chat.messages.filter((m) => m.role === 'assistant');
        const lastAssistantText = assistants[assistants.length - 1]?.content ?? '';
        return {
          status: snap.telemetry.status,
          modelId: snap.config.modelId,
          messageCount: chat.messages.length,
          lastAssistantText,
          ttftMs: snap.telemetry.ttftMs,
          prefillTokPerSec: snap.telemetry.prefillTokPerSec,
          tokensPerSec: snap.telemetry.tokensPerSec,
          activeContextTokens: snap.telemetry.activeContextTokens,
          prefixReused: snap.telemetry.prefixReused,
          backgroundPrefillState: snap.telemetry.backgroundPrefillState,
          activePrefixSlot: snap.telemetry.activePrefixSlot,
          lastSlotRestoreMs: snap.telemetry.lastSlotRestoreMs,
          basePromptCached: snap.telemetry.basePromptCached,
          basePromptTokens: snap.telemetry.basePromptTokens,
          allocatedKvCapacity: snap.telemetry.allocatedKvCapacity,
          computeTier: snap.telemetry.computeTier,
          displayHz: snap.telemetry.displayHz,
          pageEnabled,
          researchEnabled,
          enableThinking: snap.config.enableThinking,
          backendKind,
          turnState: chat.state,
          isGenerating: chat.isGenerating,
          chatError: chat.error,
          turnTelemetry: chat.telemetry,
          lastCompaction: chat.lastCompaction,
          enabledToolNames: enabledTools.map((t) => t.name),
          pendingApproval: chat.pendingApproval,
          sourceCount: sources.length,
          activePage: activePage
            ? {
                title: activePage.title,
                url: activePage.url,
                mode: activePage.mode,
                chars: activePage.content.length,
                truncated: Boolean(activePage.truncated),
                contentHash: activePage.contentHash,
              }
            : null,
        };
      },
      getTrajectory: () => ({
        threadId: chat.thread?.id ?? null,
        turnState: chat.state,
        isGenerating: chat.isGenerating,
        chatError: chat.error,
        streamingText: chat.streamingText,
        streamingReasoning: chat.streamingReasoning,
        messages: chat.messages,
        pendingToolCalls: chat.pendingToolCalls,
        pendingApproval: chat.pendingApproval,
        turnTelemetry: chat.telemetry,
        lastCompaction: chat.lastCompaction,
        sources,
        enabledToolNames: enabledTools.map((t) => t.name),
        activePage: activePage
          ? {
              title: activePage.title,
              url: activePage.url,
              contentHash: activePage.contentHash,
              chars: activePage.content.length,
            }
          : null,
      }),
      getMemoryStats: async () => {
        await memoryClient.init();
        const [stats, records] = await Promise.all([
          memoryClient.getStats(),
          memoryClient.inspect({ limit: 50 }),
        ]);
        return { stats, records };
      },
      rememberFact: async (content, tags, tier = 'profile', sourceUrl) => {
        await memoryClient.init();
        const rec = await memoryClient.remember({
          tier,
          content,
          tags,
          ...(sourceUrl ? { sourceUrl } : {}),
        });
        await refreshMemoryList();
        return rec;
      },
      setCompactionBudget: (maxContextTokens: number | null) => {
        compactionBudgetRef.current = maxContextTokens;
      },
      wipeMemory: async () => {
        await memoryClient.init();
        await memoryClient.wipeAll();
        await refreshMemoryList();
      },
      clearChat: async () => {
        await Promise.all([
          chat.createThread({ title: 'New Chat', systemPrompt: modeSystemPrompt }),
          client.resetContext(true),
        ]);
        setSources([]);
      },
    };
  }, [
    client,
    chat,
    tabCtrl,
    memoryClient,
    refreshMemoryList,
    modeSystemPrompt,
    pageEnabled,
    researchEnabled,
    backendKind,
    enabledTools,
    sources,
    activePage,
    handleSendText,
  ]);

  const handleToggleResearch = useCallback(async () => {
    if (!researchEnabled) {
      if (!hasResearchPermission) {
        const granted = await client.requestResearchPermission();
        setHasResearchPermission(granted);
        if (!granted) {
          showToast('Web research permission declined.');
          return;
        }
      }
      setResearchEnabled(true);
    } else {
      setResearchEnabled(false);
    }
  }, [researchEnabled, hasResearchPermission, client, showToast]);

  const handleToggleVoice = useCallback(async () => {
    const res = await voiceController.toggleDictation();
    if (!res.active && res.transcript) {
      setComposerText((prev) => (prev ? `${prev} ${res.transcript}` : res.transcript));
    }
  }, [voiceController]);

  const handleSelectModel = useCallback(
    (modelId: string) => {
      userUnloadedRef.current = false;
      setSelectedModelId(modelId);
      client.updateConfig({ modelId });
      const isBundled = (BUNDLED_MODEL_IDS as readonly string[]).includes(modelId);
      if (isBundled) {
        void client.loadModel({ modelId });
        showToast(`Switching local engine to ${modelId.split('/').pop()}...`);
      }
    },
    [client, showToast],
  );

  // Build catalog items for `@yours/ui` ModelDownloadGate
  const uiModels: UiModelChoice[] = useMemo(() => {
    const hwRecId = engineSnapshot.hardwareProfile?.recommendedModelId;
    if (engineSnapshot.catalogModels.length > 0) {
      return engineSnapshot.catalogModels.map((m) => ({
        id: m.id,
        label: m.label,
        subtitle: `${m.family} · ${m.paramsB}B · ${(m.contextLength / 1024).toFixed(0)}k ctx${m.bundled ? ' · Bundled offline' : ''}`,
        sizeLabel: `${m.estimatedSizeGb} GB`,
        cached: m.cachedInOpfs,
        recommended: m.id === hwRecId || m.bundled,
      }));
    }
    return MODEL_PRESETS.map((preset) => ({
      id: preset.id,
      label: preset.label,
      subtitle: preset.notes,
      sizeLabel: `${preset.estimatedWeightGB[preset.defaultDtype]} GB`,
      cached: (BUNDLED_MODEL_IDS as readonly string[]).includes(preset.id),
      recommended: preset.id === BUNDLED_MODEL_IDS[0],
    }));
  }, [engineSnapshot.catalogModels, engineSnapshot.hardwareProfile?.recommendedModelId]);

  const uiDownloadProgress: UiDownloadProgress | null = useMemo(() => {
    const dp = engineSnapshot.downloadProgress;
    if (!dp) return null;
    const loadedMb = (dp.bytesDownloaded / (1024 * 1024)).toFixed(1);
    const totalMb = (dp.totalBytes / (1024 * 1024)).toFixed(1);
    return {
      percent: dp.pct,
      loadedLabel: `${loadedMb} MB`,
      totalLabel: `${totalMb} MB`,
      verifiedChunks: dp.phase === 'complete' ? 1 : 0,
      totalChunks: 1,
    };
  }, [engineSnapshot.downloadProgress]);

  const hardwareLabel = useMemo(() => {
    const hw = engineSnapshot.hardwareProfile;
    if (!hw) {
      return `WebGPU (${engineSnapshot.telemetry.computeTier ?? 'probing...'})`;
    }
    return `${hw.adapterName} · ${hw.recommendedTier} · Max buf ${hw.maxBufferSizeMb} MB`;
  }, [engineSnapshot.hardwareProfile, engineSnapshot.telemetry.computeTier]);

  const notchItems: TopNotchItem[] = useMemo(
    () => [
      {
        id: 'history',
        icon: 'history',
        label: 'Threads',
        title: 'Conversation history',
      },
      {
        id: 'models',
        icon: 'local-device',
        label: 'Models',
        title: 'Local runtime & model selector',
        badgeDot: engineSnapshot.telemetry.status === 'ready' ? 'green' : 'amber',
        badgePulse:
          engineSnapshot.telemetry.status === 'downloading' ||
          engineSnapshot.telemetry.status === 'compiling',
      },
      {
        id: 'memory',
        icon: 'bookmark',
        label: 'Memory',
        title: 'Local encrypted-domain memory',
      },
      {
        id: 'lab',
        icon: 'sliders',
        label: 'Lab',
        title: 'WebGPU diagnostics & benchmarks',
      },
      {
        id: 'theme',
        icon: 'theme',
        label: theme === 'dark' ? 'Light' : 'Dark',
        title:
          theme === 'dark'
            ? 'Switch to Cotton Paper (Light)'
            : 'Switch to Anodized Obsidian (Dark)',
        onClick: () => {
          setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
        },
      },
    ],
    [engineSnapshot.telemetry.status, theme],
  );

  const drawerContent = useMemo(() => {
    if (expandedNotchId === 'history') {
      return (
        <div className="ext-drawer-panel">
          <NotchDrawerSearch
            value={threadSearch}
            onChange={setThreadSearch}
            placeholder="Filter threads..."
            count={threads.length}
          />
          <HookSidebarList
            items={threads}
            activeId={chat.thread?.id}
            label="Threads"
            dashed
            onSelect={(id) => {
              void chat.loadThread(id);
              setExpandedNotchId(null);
            }}
            emptyText="No threads yet"
          />
          <div className="ext-drawer-actions">
            <button
              type="button"
              className="yui-chip-btn"
              onClick={() => {
                void chat.createThread({ title: 'New Chat', systemPrompt: modeSystemPrompt });
                setSources([]);
                setExpandedNotchId(null);
              }}
            >
              + New chat
            </button>
            {chat.thread?.id && threads.length > 1 ? (
              <button
                type="button"
                className="yui-chip-btn"
                onClick={() => {
                  const currentId = chat.thread?.id;
                  if (currentId) {
                    void harnessStore.deleteThread(currentId).then(() => refreshThreadsList());
                  }
                }}
              >
                Delete current
              </button>
            ) : null}
          </div>
        </div>
      );
    }

    if (expandedNotchId === 'models') {
      return (
        <div className="ext-drawer-panel">
          <div className="ext-drawer-section">
            <div className="ext-drawer-heading">
              <span>Inference Runtime Backend</span>
              <span className="ext-kv-val">{backendKind}</span>
            </div>
            <div className="ext-segment-row">
              <button
                type="button"
                className={`ext-segment-btn ${backendKind === 'wgsl' ? 'is-active' : ''}`}
                onClick={() => setBackendKind('wgsl')}
              >
                Local WGSL GPU
              </button>
              <button
                type="button"
                className={`ext-segment-btn ${backendKind === 'chrome-ai' ? 'is-active' : ''}`}
                onClick={() => {
                  if (!chromeAiAvailable) {
                    showToast('Chrome Built-in LanguageModel API unavailable in this browser.');
                    return;
                  }
                  setBackendKind('chrome-ai');
                }}
              >
                Chrome Built-in AI
              </button>
              <button
                type="button"
                className={`ext-segment-btn ${backendKind === 'openai' ? 'is-active' : ''}`}
                onClick={() => setBackendKind('openai')}
              >
                Local OpenAI / Ollama
              </button>
            </div>
            {backendKind === 'openai' ? (
              <div className="ext-form-row">
                <input
                  className="ext-input"
                  value={openAiConfig.baseUrl}
                  onChange={(e) =>
                    setOpenAiConfig((c) => ({ ...c, baseUrl: e.target.value }))
                  }
                  placeholder="http://127.0.0.1:11434/v1"
                  aria-label="OpenAI compatible base URL"
                />
                <input
                  className="ext-input"
                  value={openAiConfig.model}
                  onChange={(e) =>
                    setOpenAiConfig((c) => ({ ...c, model: e.target.value }))
                  }
                  placeholder="Model ID (e.g. qwen2.5:7b)"
                  aria-label="OpenAI compatible model name"
                />
              </div>
            ) : null}
          </div>

          <HardwareStatusStrip
            tierLabel={engineSnapshot.telemetry.computeTier ?? engineSnapshot.hardwareProfile?.recommendedTier ?? 'probing'}
            adapterLabel={engineSnapshot.hardwareProfile?.adapterName ?? 'WebGPU adapter'}
            memoryLabel={
              engineSnapshot.telemetry.allocatedKvCapacity > 0
                ? `${engineSnapshot.telemetry.activeContextTokens}/${engineSnapshot.telemetry.allocatedKvCapacity} KV`
                : undefined
            }
            status={
              engineSnapshot.telemetry.status === 'error'
                ? 'unsupported'
                : engineSnapshot.telemetry.computeTier
                  ? 'ready'
                  : 'probing'
            }
            sha256Verified={
              engineSnapshot.opfsStatus
                ? engineSnapshot.opfsStatus.modelFilesCount > 0 || Boolean(engineSnapshot.spikeResult?.ok)
                : undefined
            }
          />

          <ModelDownloadGate
            title="Local Weights & Hardware Fit"
            subtitle="Switch offline WGSL models or download verified manifests into OPFS."
            models={uiModels}
            selectedModelId={selectedModelId}
            onSelectModel={handleSelectModel}
            hardwareLabel={hardwareLabel}
            isDownloading={
              engineSnapshot.telemetry.status === 'downloading' ||
              engineSnapshot.downloadProgress !== null
            }
            progress={uiDownloadProgress}
            error={engineSnapshot.downloadError ?? engineSnapshot.telemetry.lastError}
            onStartDownload={() => {
              const isBundled = (BUNDLED_MODEL_IDS as readonly string[]).includes(selectedModelId);
              if (isBundled) {
                void client.loadModel({ modelId: selectedModelId });
              } else {
                client.downloadVerifiedModel(selectedModelId);
              }
            }}
            onCancelDownload={() => {
              userUnloadedRef.current = true;
              client.cancelModelDownload(selectedModelId);
              void client.unloadModel();
            }}
          />
        </div>
      );
    }

    if (expandedNotchId === 'memory') {
      const uiMemoryRecords: UiMemoryRecord[] = memoryRecords.map((rec) => {
        let sourceDomain: string | undefined;
        if (rec.sourceUrl) {
          try {
            sourceDomain = new URL(rec.sourceUrl).hostname;
          } catch {
            sourceDomain = rec.sourceUrl;
          }
        }
        const ageSec = Math.max(0, Math.floor((Date.now() - (rec.updatedAt || rec.createdAt)) / 1000));
        const relativeTime =
          ageSec < 60
            ? 'just now'
            : ageSec < 3600
              ? `${Math.floor(ageSec / 60)}m ago`
              : ageSec < 86400
                ? `${Math.floor(ageSec / 3600)}h ago`
                : `${Math.floor(ageSec / 86400)}d ago`;
        return {
          id: rec.id,
          tier: rec.tier,
          summary: rec.content,
          sourceDomain,
          relativeTime,
          detail: rec.sourceTitle,
        };
      });

      return (
        <div className="ext-drawer-panel">
          <div className="ext-drawer-section">
            <div className="ext-drawer-heading">
              <span>Local Encrypted-Domain Memory</span>
              <span className="ext-kv-val">{memoryRecords.length} entries</span>
            </div>
            <div className="ext-segment-row">
              <input
                className="ext-input"
                value={memorySearch}
                onChange={(e) => setMemorySearch(e.target.value)}
                placeholder="Search local memory (BM25 + recency)..."
                aria-label="Search local memory"
              />
            </div>
            <div className="btn-row">
              <button
                type="button"
                className="btn"
                onClick={() => void refreshMemoryList()}
              >
                Refresh
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  void memoryClient.wipeAll().then(() => {
                    setMemoryRecords([]);
                    showToast('Local memory wiped.');
                  });
                }}
              >
                Wipe All Memory
              </button>
            </div>
          </div>
          <MemoryInspectorList
            records={uiMemoryRecords}
            activeTier={memoryTierFilter}
            onTierChange={setMemoryTierFilter}
            onForget={(id) => {
              void memoryClient.forget(id).then(() => refreshMemoryList());
            }}
            emptyTitle="No local memories stored"
            emptySubtitle="Save a reply or store non-sensitive preferences to populate encrypted local memory."
          />
        </div>
      );
    }

    if (expandedNotchId === 'lab') {
      return (
        <div className="ext-drawer-panel">
          <DiagnosticsDrawer
            config={engineSnapshot.config}
            telemetry={engineSnapshot.telemetry}
            probeInfo={engineSnapshot.probeInfo}
            opfsStatus={engineSnapshot.opfsStatus}
            spikeProgress={engineSnapshot.spikeProgress}
            spikeResult={engineSnapshot.spikeResult}
            benchmarkProgress={engineSnapshot.benchmarkProgress}
            lastBenchmark={engineSnapshot.lastBenchmark}
            niahProgress={engineSnapshot.niahProgress}
            lastNiah={engineSnapshot.lastNiah}
            logs={engineSnapshot.logs}
            onUpdateConfig={(patch) => void client.updateConfig(patch)}
            onProbeWebGpu={() => void client.send({ type: 'PROBE_WEBGPU' })}
            onCheckOpfs={() => void client.send({ type: 'CHECK_OPFS' })}
            onRunOpfsSpike={(totalMiB, shardMiB) =>
              void client.send({ type: 'RUN_OPFS_SPIKE', totalMiB, shardMiB })
            }
            onVerifyOpfsSpike={() => void client.send({ type: 'VERIFY_OPFS_SPIKE' })}
            onClearOpfsSpike={() => void client.send({ type: 'CLEAR_OPFS_SPIKE' })}
            onClearModelCache={() => void client.send({ type: 'CLEAR_MODEL_CACHE' })}
            onSnapshotContext={() => void client.send({ type: 'SNAPSHOT_CONTEXT' })}
            onRestoreContext={() => void client.send({ type: 'RESTORE_CONTEXT' })}
            onResetContext={() => client.resetContext(true)}
            onRunBenchmarkSuite={(contextLengths, decodeTokens) =>
              void client.send({ type: 'RUN_BENCHMARK_SUITE', contextLengths, decodeTokens })
            }
            onRunNiah={(targetTokens, depthRatio) =>
              void client.send({ type: 'RUN_NIAH', targetTokens, depthRatio })
            }
          />
        </div>
      );
    }

    return null;
  }, [
    expandedNotchId,
    threads,
    chat,
    threadSearch,
    harnessStore,
    refreshThreadsList,
    backendKind,
    chromeAiAvailable,
    openAiConfig,
    uiModels,
    selectedModelId,
    handleSelectModel,
    hardwareLabel,
    engineSnapshot,
    uiDownloadProgress,
    client,
    memoryRecords,
    memorySearch,
    memoryTierFilter,
    memoryClient,
    refreshMemoryList,
    showToast,
  ]);

  const voiceBars: Array<'low' | 'mid' | 'high'> = useMemo(() => {
    const lvl = voiceState.micLevel;
    return [
      lvl > 0.2 ? 'mid' : 'low',
      lvl > 0.5 ? 'high' : lvl > 0.25 ? 'mid' : 'low',
      lvl > 0.7 ? 'high' : lvl > 0.35 ? 'mid' : 'low',
      lvl > 0.4 ? 'mid' : 'low',
    ];
  }, [voiceState.micLevel]);

  const composerTopSlot = useMemo(() => {
    if (!quotedSelection) return undefined;
    return (
      <QuotedContextBanner
        quote={quotedSelection}
        onDismiss={() => setQuotedSelection(null)}
      />
    );
  }, [quotedSelection]);

  const composerLeftSlot = (
    <div className="ext-composer-tools">
      <button
        type="button"
        className={`yui-chip-btn ${pageEnabled ? 'is-active' : ''}`}
        onClick={() => setPageEnabled((v) => !v)}
        title="Include active tab content"
      >
        <PageIcon size={13} />
        <span>Page</span>
      </button>
      <button
        type="button"
        className={`yui-chip-btn ${researchEnabled ? 'is-active' : ''}`}
        onClick={() => void handleToggleResearch()}
        title="Allow live web search & multi-tab reading"
      >
        <GlobeIcon size={13} />
        <span>Web</span>
      </button>
      <button
        type="button"
        className={`yui-chip-btn ${engineSnapshot.config.enableThinking ? 'is-active' : ''}`}
        onClick={() => {
          const next = !engineSnapshot.config.enableThinking;
          client.updateConfig({ enableThinking: next });
          chat.updateOptions({ enableThinking: next });
        }}
        title="Toggle thinking reasoning trace"
      >
        <SparkTrendIcon size={13} />
        <span>Think</span>
      </button>
      <button
        type="button"
        className={`yui-chip-btn yui-chip-btn--icon ${voiceState.active ? 'is-recording' : ''}`}
        onClick={() => void handleToggleVoice()}
        title="Local microphone dictation"
        aria-label="Local microphone dictation"
      >
        <MicIcon size={13} />
      </button>
    </div>
  );

  const composerPlaceholder: FloatingComposerProps['placeholder'] = pageEnabled && activePage?.title
    ? `Ask about "${activePage.title.slice(0, 32)}"...`
    : 'Ask locally...';

  return (
    <div className="ext-root">
      <SidePanelShell
        theme={theme}
        bleed
        scrollEdgeFade
        veilOpen={expandedNotchId !== null}
        onVeilClick={() => setExpandedNotchId(null)}
        notch={
          <TopNotch
            items={notchItems}
            expandedId={expandedNotchId}
            onSelectItem={(id) => {
              if (id === 'theme') return;
              setExpandedNotchId((prev) => (prev === id ? null : id));
            }}
            onClose={() => setExpandedNotchId(null)}
            drawerContent={drawerContent}
          />
        }
        footer={
          <div className="ext-footer-stack">
            {toastMsg ? (
              <div className="ext-toast-anchor">
                <ToastPill message={toastMsg} />
              </div>
            ) : null}
            {voiceState.active ? (
              <VoiceDictationPill
                statusLabel={voiceState.partialTranscript || 'Listening locally...'}
                bars={voiceBars}
                onCancel={() => {
                  void voiceController.stopDictation();
                }}
                onConfirm={() => {
                  void voiceController.stopDictation().then((text) => {
                    if (text) {
                      setComposerText((prev) => (prev ? `${prev} ${text}` : text));
                    }
                  });
                }}
              />
            ) : null}
            <FloatingComposer
              value={composerText}
              onChange={(val) => {
                setComposerText(val);
                chat.setComposerText(val);
              }}
              onSubmit={() => void handleSendText(composerText)}
              onStop={chat.abort}
              isGenerating={chat.isGenerating}
              placeholder={composerPlaceholder}
              topSlot={composerTopSlot}
              leftSlot={composerLeftSlot}
              ariaLabel="Message Yours local assistant"
            />
          </div>
        }
      >
        <ChatThread
          nodes={chat.nodes}
          branchSiblings={chat.branchSiblings}
          isGenerating={chat.isGenerating}
          turnState={chat.state}
          streamingText={chat.streamingText}
          streamingReasoning={chat.streamingReasoning}
          pendingToolCalls={chat.pendingToolCalls}
          pendingApproval={chat.pendingApproval}
          turnTelemetry={chat.telemetry}
          engineTelemetry={engineSnapshot.telemetry}
          error={chat.error}
          starters={STARTER_PROMPTS}
          sources={sources}
          onSelectStarter={(item) => {
            if (item.prompt) {
              void handleSendText(item.prompt);
            }
          }}
          onEditAndBranch={(nodeId, next) => void chat.editAndBranch(nodeId, next)}
          onRegenerate={(assistantId) => void chat.regenerate(assistantId)}
          onSwitchBranch={(branchId) => void chat.switchBranch(branchId)}
          onApproveTool={chat.approveTool}
          onDenyTool={chat.denyTool}
          onCopyText={(text) => {
            void navigator.clipboard?.writeText(text).catch(() => {});
            showToast('Copied to clipboard');
          }}
          onSpeakText={(text) => voiceController.speakText(text)}
          onSaveMemory={(text) => {
            void memoryClient
              .remember({
                tier: 'episodic',
                content: text.slice(0, 1200),
                sourceUrl: activePage?.url || undefined,
                sourceTitle: activePage?.title || undefined,
                threadId: chat.thread?.id,
              })
              .then(() => showToast('Saved to local memory'))
              .catch((err) => {
                const msg = err instanceof Error ? err.message : String(err);
                showToast(msg);
              });
          }}
          onSelectChoice={(q, choiceId) => {
            void handleSendText(`Selected option "${choiceId}" for: ${q}`);
          }}
          onRetry={() => void chat.regenerate()}
        />
      </SidePanelShell>
    </div>
  );
}
