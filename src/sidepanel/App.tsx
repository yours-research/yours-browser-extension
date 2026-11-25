import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { buildActivePageContextMessage, buildBaseSystemMessage } from '../shared/systemPrompt.ts';
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
  const [includeActivePage, setIncludeActivePage] = useState(true);
  const [researchGranted, setResearchGranted] = useState(false);
  const [activePagePreview, setActivePagePreview] = useState<ExtractedPageContent | null>(null);
  const [toolStatus, setToolStatus] = useState<ToolStatusUpdate | null>(null);
  const [permissionPrompt, setPermissionPrompt] = useState<{
    toolName: string;
    label: string;
  } | null>(null);

  const autoPreloadTriggeredRef = useRef(false);
  const userUnloadedRef = useRef(false);
  const statusRef = useRef(snapshot.telemetry.status);
  statusRef.current = snapshot.telemetry.status;
  const activePageRef = useRef<ExtractedPageContent | null>(null);
  activePageRef.current = activePagePreview;
  const includeActivePageRef = useRef(includeActivePage);
  includeActivePageRef.current = includeActivePage;

  useEffect(() => {
    void engineClient.connect().catch(() => {
      // Connection errors are surfaced through snapshot/logs.
    });
    void measureDisplayRefreshRate().then((hz) => {
      engineClient.setDisplayHz(hz);
    });
    void engineClient.checkResearchPermission().then(setResearchGranted).catch(() => {});
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
          const msgs = [buildBaseSystemMessage()];
          const page = includeActivePageRef.current ? activePageRef.current : null;
          if (page && page.content.trim().length > 0) {
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
        isEngineIdle: () => statusRef.current === 'ready',
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
          const page = await engineClient.extractActiveTabPreview();
          if (page && page.content.trim().length > 0 && !page.error) {
            setActivePagePreview(page);
            return {
              title: page.title,
              url: page.url,
              content: page.content,
              mode: page.mode,
            };
          }
          return null;
        },
        buildMessages: (tab) => [
          buildBaseSystemMessage(),
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
  }, [includeActivePage, snapshot.telemetry.status, tabCtrl]);

  useEffect(() => {
    if (typeof chrome === 'undefined' || !chrome.tabs?.onActivated || !chrome.tabs?.onUpdated) {
      return;
    }
    const onActivated = () => {
      if (includeActivePageRef.current) {
        tabCtrl.notifyTabChanged(true);
      }
    };
    const onUpdated = (_tabId: number, changeInfo: { status?: string }) => {
      if (changeInfo.status === 'complete' && includeActivePageRef.current) {
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
            groundingText: page.content,
          };
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

  const handleGrantResearch = async () => {
    const granted = await engineClient.requestResearchPermission();
    setResearchGranted(granted);
    if (granted) {
      setPermissionPrompt(null);
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
        onResetContext={() => engineClient.resetContext(snapshot.telemetry.basePromptCached)}
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
          onResetContext={() => engineClient.resetContext(snapshot.telemetry.basePromptCached)}
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
          adapter={adapter}
          modelReady={modelReady}
          includeActivePage={includeActivePage}
          onToggleActivePage={() =>
            setIncludeActivePage((prev) => {
              const next = !prev;
              if (!next) {
                tabCtrl.cancelForSubmit();
              } else {
                tabCtrl.scheduleExtraction();
              }
              return next;
            })
          }
          activePageTitle={activePagePreview?.title ?? null}
          backgroundWarm={snapshot.telemetry.backgroundPrefillState === 'warm'}
          researchGranted={researchGranted}
          onGrantResearch={() => void handleGrantResearch()}
          toolStatus={toolStatus}
          permissionPrompt={permissionPrompt}
          onDismissPermission={() => setPermissionPrompt(null)}
          onComposerTextChange={(txt) => typingCtrl.onTextChanged(txt)}
        />
      </main>
    </div>
  );
}
