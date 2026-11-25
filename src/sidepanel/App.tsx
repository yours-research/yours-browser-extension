import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { buildActivePageContextMessage, buildModeSystemMessage } from '../shared/systemPrompt.ts';
import type { ExtractedPageContent } from '../tools/types.ts';
import {
  ActiveTabSpeculativePrefillController,
  TypingSpeculativePrefillController,
  shouldAutoPreloadModel,
  type ToolStatusUpdate,
} from './agentController.ts';
import { createTinyGptChatAdapter } from './chatAdapter.ts';
import { ChatThread } from './components/ChatThread.tsx';
import { DiagnosticsDrawer } from './components/DiagnosticsDrawer.tsx';
import { TelemetryBar } from './components/TelemetryBar.tsx';
import { measureDisplayRefreshRate } from './displayHz.ts';
import { OffscreenEngineClient } from './engineClient.ts';

const engineClient = new OffscreenEngineClient();
(window as unknown as { __TINYGPT_CLIENT__?: OffscreenEngineClient }).__TINYGPT_CLIENT__ =
  engineClient;

export function App() {
  const snapshot = useSyncExternalStore(engineClient.subscribe, engineClient.getSnapshot);
  const [labOpen, setLabOpen] = useState(false);
  const [threadKey, setThreadKey] = useState(0);
  const [includeActivePage, setIncludeActivePage] = useState(true);
  const [researchGranted, setResearchGranted] = useState(false);
  const [researchEnabled, setResearchEnabled] = useState(false);
  const [activePagePreview, setActivePagePreview] = useState<ExtractedPageContent | null>(null);
  const [activePageError, setActivePageError] = useState<string | null>(null);
  const [toolStatus, setToolStatus] = useState<ToolStatusUpdate | null>(null);
  const [permissionPrompt, setPermissionPrompt] = useState<{
    toolName: string;
    label: string;
  } | null>(null);

  const autoPreloadTriggeredRef = useRef(false);
  const userUnloadedRef = useRef(false);
  const statusRef = useRef(snapshot.telemetry.status);
  statusRef.current = snapshot.telemetry.status;
  const bgStateRef = useRef(snapshot.telemetry.backgroundPrefillState);
  bgStateRef.current = snapshot.telemetry.backgroundPrefillState;
  const activePageRef = useRef<ExtractedPageContent | null>(null);
  activePageRef.current = activePagePreview;
  const activePageErrorRef = useRef<string | null>(null);
  activePageErrorRef.current = activePageError;
  const includeActivePageRef = useRef(includeActivePage);
  includeActivePageRef.current = includeActivePage;
  const researchEnabledRef = useRef(researchEnabled);
  researchEnabledRef.current = researchEnabled;
  const overriddenPageRef = useRef<ExtractedPageContent | null>(null);

  useEffect(() => {
    void engineClient.connect().catch(() => {
      // Connection errors are surfaced through snapshot/logs.
    });
    void measureDisplayRefreshRate().then((hz) => {
      engineClient.setDisplayHz(hz);
    });
    void engineClient
      .checkResearchPermission()
      .then((granted) => {
        setResearchGranted(granted);
        setResearchEnabled(granted);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const noAutoLoad =
      typeof window !== 'undefined' &&
      new URLSearchParams(window.location.search).get('noAutoLoad') === '1';
    if (
      !noAutoLoad &&
      !autoPreloadTriggeredRef.current &&
      !userUnloadedRef.current &&
      snapshot.connected &&
      shouldAutoPreloadModel(snapshot.telemetry.status)
    ) {
      autoPreloadTriggeredRef.current = true;
      void engineClient.send({ type: 'LOAD_MODEL' });
    }
  }, [snapshot.connected, snapshot.telemetry.status]);

  const typingCtrl = useMemo(
    () =>
      new TypingSpeculativePrefillController({
        buildMessages: (stableText) => {
          const page = includeActivePageRef.current ? activePageRef.current : null;
          const hasPage = Boolean(page && page.content.trim().length > 0);
          const msgs = [
            buildModeSystemMessage({
              pageEnabled: includeActivePageRef.current,
              pageAttached: hasPage,
              researchEnabled: researchEnabledRef.current,
              pageTruncated: Boolean(page?.truncated),
            }),
          ];
          if (hasPage && page) {
            msgs.push(
              buildActivePageContextMessage({
                title: page.title,
                url: page.url,
                content: page.content,
                mode: page.mode,
              }),
            );
          }
          msgs.push({ role: 'user', content: stableText });
          return msgs;
        },
        isEngineIdle: () => statusRef.current === 'ready' && bgStateRef.current !== 'prefilling',
        onPrefill: async (slotKey, messages) => {
          engineClient.prefillBackground(slotKey, messages, { isDraft: true });
        },
        onAbort: () => {
          engineClient.abortBackgroundPrefill();
        },
      }),
    [],
  );

  const tabCtrl = useMemo(
    () =>
      new ActiveTabSpeculativePrefillController({
        extractActiveTab: async () => {
          const page = overriddenPageRef.current ?? (await engineClient.extractActiveTabPreview());
          if (page && page.content.trim().length > 0 && !page.error) {
            setActivePageError(null);
            setActivePagePreview(page);
            return {
              title: page.title,
              url: page.url,
              content: page.content,
              mode: page.mode,
              truncated: page.truncated,
            };
          }
          setActivePagePreview(null);
          setActivePageError(page?.error ?? 'Active tab could not be accessed.');
          return null;
        },
        buildMessages: (tab) => [
          buildModeSystemMessage({
            pageEnabled: true,
            pageAttached: true,
            researchEnabled: researchEnabledRef.current,
            pageTruncated: Boolean(tab.truncated),
          }),
          buildActivePageContextMessage({
            title: tab.title,
            url: tab.url,
            content: tab.content,
            mode: tab.mode ?? 'article',
          }),
        ],
        isEngineIdle: () => includeActivePageRef.current && statusRef.current === 'ready',
        onPrefill: async (slotKey, messages, meta) => {
          engineClient.prefillBackground(slotKey, messages, {
            isDraft: false,
            url: meta.url,
            contentHash: meta.contentHash,
          });
        },
        onAbort: () => {
          engineClient.abortBackgroundPrefill();
        },
      }),
    [],
  );

  useEffect(() => {
    if (includeActivePage && snapshot.telemetry.status === 'ready') {
      tabCtrl.scheduleExtraction();
    }
  }, [includeActivePage, researchEnabled, snapshot.telemetry.status, tabCtrl]);

  useEffect(() => {
    if (typeof chrome === 'undefined' || !chrome.tabs?.onActivated || !chrome.tabs?.onUpdated) {
      return;
    }
    const onActivated = () => {
      if (includeActivePageRef.current) {
        setActivePagePreview(null);
        setActivePageError(null);
        tabCtrl.notifyTabChanged(true);
      }
    };
    const onUpdated = (_tabId: number, changeInfo: { status?: string; url?: string }) => {
      if ((changeInfo.status === 'complete' || typeof changeInfo.url === 'string') && includeActivePageRef.current) {
        tabCtrl.notifyTabChanged(true);
      }
    };
    chrome.tabs.onActivated.addListener(onActivated);
    chrome.tabs.onUpdated.addListener(onUpdated);
    return () => {
      chrome.tabs.onActivated.removeListener(onActivated);
      chrome.tabs.onUpdated.removeListener(onUpdated);
      typingCtrl.dispose();
      tabCtrl.dispose();
    };
  }, [tabCtrl, typingCtrl]);

  const adapter = useMemo(
    () =>
      createTinyGptChatAdapter(engineClient, {
        enableAgentTools: true,
        getModeOptions: () => ({
          pageEnabled: includeActivePageRef.current,
          researchEnabled: researchEnabledRef.current,
        }),
        getActivePageContext: () => {
          const page = includeActivePageRef.current ? activePageRef.current : null;
          if (!page || !page.content.trim()) {
            return null;
          }
          return {
            contextMsg: buildActivePageContextMessage({
              title: page.title,
              url: page.url,
              content: page.content,
              mode: page.mode,
            }),
            groundingText: page.fullContent ?? page.content,
            fullContent: page.fullContent ?? page.content,
            truncated: page.truncated,
          };
        },
        refreshActivePage: async () => {
          if (!includeActivePageRef.current) {
            return null;
          }
          if (overriddenPageRef.current) {
            return {
              contextMsg: buildActivePageContextMessage({
                title: overriddenPageRef.current.title,
                url: overriddenPageRef.current.url,
                content: overriddenPageRef.current.content,
                mode: overriddenPageRef.current.mode,
              }),
              groundingText: overriddenPageRef.current.fullContent ?? overriddenPageRef.current.content,
              fullContent: overriddenPageRef.current.fullContent ?? overriddenPageRef.current.content,
              truncated: overriddenPageRef.current.truncated,
            };
          }
          const fresh = await engineClient.extractActiveTabPreview();
          if (fresh && fresh.content.trim().length > 0 && !fresh.error) {
            setActivePageError(null);
            setActivePagePreview(fresh);
            return {
              contextMsg: buildActivePageContextMessage({
                title: fresh.title,
                url: fresh.url,
                content: fresh.content,
                mode: fresh.mode,
              }),
              groundingText: fresh.fullContent ?? fresh.content,
              fullContent: fresh.fullContent ?? fresh.content,
              truncated: fresh.truncated,
            };
          }
          return null;
        },
        onBeforeGenerate: () => {
          typingCtrl.cancelForSubmit();
          tabCtrl.cancelForSubmit();
        },
        onToolStatus: setToolStatus,
        onPermissionRequired: (req) => setPermissionPrompt(req),
      }),
    [tabCtrl, typingCtrl],
  );

  const modelReady =
    snapshot.telemetry.status === 'ready' || snapshot.telemetry.status === 'generating';

  const handleGrantResearchPermission = async () => {
    const granted = researchGranted ? true : await engineClient.requestResearchPermission();
    setResearchGranted(granted);
    if (granted) {
      setResearchEnabled(true);
      setPermissionPrompt(null);
      if (includeActivePageRef.current) {
        tabCtrl.notifyTabChanged(true);
      }
    }
  };

  const handleToggleResearch = async () => {
    if (researchEnabled) {
      setResearchEnabled(false);
      setPermissionPrompt(null);
      return;
    }
    if (researchGranted) {
      setResearchEnabled(true);
      setPermissionPrompt(null);
      if (includeActivePageRef.current) {
        tabCtrl.notifyTabChanged(true);
      }
      return;
    }
    await handleGrantResearchPermission();
  };

  useEffect(() => {
    (
      window as unknown as {
        __TINYGPT_UI__?: {
          getState: () => {
            includeActivePage: boolean;
            researchEnabled: boolean;
            researchGranted: boolean;
            activePageTitle: string | null;
            activePageError: string | null;
            backgroundPrefillState: string;
          };
          setSimulatedActivePage: (page: ExtractedPageContent | null) => void;
        };
      }
    ).__TINYGPT_UI__ = {
      getState: () => ({
        includeActivePage: includeActivePageRef.current,
        researchEnabled: researchEnabledRef.current,
        researchGranted,
        activePageTitle: activePageRef.current?.title ?? null,
        activePageError: activePageErrorRef.current,
        backgroundPrefillState: bgStateRef.current,
      }),
      setSimulatedActivePage: (page: ExtractedPageContent | null) => {
        overriddenPageRef.current = page;
        if (page) {
          setActivePageError(null);
          setActivePagePreview(page);
          tabCtrl.notifyTabChanged(true);
        } else {
          setActivePagePreview(null);
          tabCtrl.notifyTabChanged(false);
        }
      },
    };
  }, [researchGranted, tabCtrl]);

  const handleNewChat = () => {
    typingCtrl.cancelForSubmit();
    tabCtrl.cancelForSubmit();
    setToolStatus(null);
    setPermissionPrompt(null);
    setThreadKey((k) => k + 1);
    engineClient.resetContext(snapshot.telemetry.basePromptCached);
    if (includeActivePageRef.current && statusRef.current === 'ready') {
      tabCtrl.scheduleExtraction();
    }
  };

  return (
    <div className="app-shell">
      <TelemetryBar
        connected={snapshot.connected}
        telemetry={snapshot.telemetry}
        labOpen={labOpen}
        onToggleLab={() => setLabOpen((prev) => !prev)}
        onLoadModel={() => {
          userUnloadedRef.current = false;
          void engineClient.send({ type: 'LOAD_MODEL' });
        }}
        onUnloadModel={() => {
          userUnloadedRef.current = true;
          typingCtrl.cancelForSubmit();
          tabCtrl.cancelForSubmit();
          void engineClient.send({ type: 'UNLOAD_MODEL' });
        }}
        onResetContext={handleNewChat}
      />

      {labOpen && (
        <DiagnosticsDrawer
          config={snapshot.config}
          telemetry={snapshot.telemetry}
          probeInfo={snapshot.probeInfo}
          opfsStatus={snapshot.opfsStatus}
          spikeProgress={snapshot.spikeProgress}
          spikeResult={snapshot.spikeResult}
          benchmarkProgress={snapshot.benchmarkProgress}
          lastBenchmark={snapshot.lastBenchmark}
          niahProgress={snapshot.niahProgress}
          lastNiah={snapshot.lastNiah}
          logs={snapshot.logs}
          onUpdateConfig={(patch) =>
            void engineClient.send({ type: 'UPDATE_CONFIG', config: patch })
          }
          onProbeWebGpu={() => void engineClient.send({ type: 'PROBE_WEBGPU' })}
          onCheckOpfs={() => void engineClient.send({ type: 'CHECK_OPFS' })}
          onRunOpfsSpike={(totalMiB, shardMiB) =>
            void engineClient.send({ type: 'RUN_OPFS_SPIKE', totalMiB, shardMiB })
          }
          onVerifyOpfsSpike={() => void engineClient.send({ type: 'VERIFY_OPFS_SPIKE' })}
          onClearOpfsSpike={() => void engineClient.send({ type: 'CLEAR_OPFS_SPIKE' })}
          onClearModelCache={() => void engineClient.send({ type: 'CLEAR_MODEL_CACHE' })}
          onSnapshotContext={() => void engineClient.send({ type: 'SNAPSHOT_CONTEXT' })}
          onRestoreContext={() => void engineClient.send({ type: 'RESTORE_CONTEXT' })}
          onResetContext={handleNewChat}
          onRunBenchmarkSuite={(contextLengths, decodeTokens) =>
            void engineClient.send({
              type: 'RUN_BENCHMARK_SUITE',
              contextLengths,
              decodeTokens,
            })
          }
          onRunNiah={(targetTokens, depthRatio) =>
            void engineClient.send({
              type: 'RUN_NIAH',
              targetTokens,
              depthRatio,
            })
          }
        />
      )}

      <main className="chat-main">
        <ChatThread
          key={threadKey}
          adapter={adapter}
          modelReady={modelReady}
          includeActivePage={includeActivePage}
          onToggleActivePage={() =>
            setIncludeActivePage((prev) => {
              const next = !prev;
              if (!next) {
                tabCtrl.cancelForSubmit();
                setActivePagePreview(null);
                setActivePageError(null);
              } else {
                tabCtrl.scheduleExtraction();
              }
              return next;
            })
          }
          activePageTitle={activePagePreview?.title ?? null}
          activePageError={activePageError}
          backgroundPrefillState={snapshot.telemetry.backgroundPrefillState}
          researchEnabled={researchEnabled}
          onToggleResearch={() => void handleToggleResearch()}
          toolStatus={toolStatus}
          permissionPrompt={permissionPrompt}
          onGrantResearchPermission={() => void handleGrantResearchPermission()}
          onDismissPermission={() => setPermissionPrompt(null)}
          onComposerTextChange={(txt) => typingCtrl.onTextChanged(txt)}
        />
      </main>
    </div>
  );
}
