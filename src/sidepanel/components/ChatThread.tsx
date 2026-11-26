import { useState } from 'react';
import type {
  BranchSiblingInfo,
  HarnessMessage,
  MessageNode,
  PendingApprovalRequest,
  TurnState,
  TurnTelemetrySnapshot,
} from '@yours/harness';
import {
  ActionBar,
  AssistantMessage,
  BranchPicker,
  CitationFootnoteList,
  EmptyPanelState,
  ErrorRecoveryCard,
  InlineCitationPopover,
  InlineNotice,
  InlineToolStatus,
  OnboardingCard,
  AgentHandoffCard,
  PermissionCard,
  ReasoningTrace,
  SourcesDrawer,
  ToolTraceList,
  UserMessageBubble,
  WebSearchSourcesStrip,
  type CitationFootnoteItem,
  type InlineCitationSource,
  type InlineNoticeTone,
  type PromptStarterItem,
  type SourceCitation,
  type WebSearchSourceItem,
} from '@yours/ui';
import type { TelemetryState } from '../../shared/protocol.ts';
import {
  buildGroupStepsFromToolNodes,
  formatToolChainSummary,
  mergeActiveTurnToolSteps,
  type ActivePageHint,
  type ToolExecutionTelemetry,
} from '../toolPresentation.ts';
import type { ToolTraceRecord } from '@yours/harness/storage';
import { GenUiRenderer } from './GenUiRenderer.tsx';

export interface ChatThreadProps {
  nodes: MessageNode[];
  branchSiblings: Record<string, BranchSiblingInfo>;
  isGenerating: boolean;
  turnState: TurnState;
  streamingText: string;
  streamingReasoning: string;
  pendingToolCalls: Array<{ id: string; name: string }>;
  liveToolExecutions?: readonly ToolExecutionTelemetry[];
  persistedToolTraces?: readonly ToolTraceRecord[];
  activePageHint?: ActivePageHint;
  pendingApproval: PendingApprovalRequest | null;
  turnTelemetry: TurnTelemetrySnapshot | null;
  engineTelemetry: TelemetryState;
  error: string | null;
  starters: readonly PromptStarterItem[];
  sources: readonly SourceCitation[];
  sourcesDrawerVariant?: 'pill' | 'inline';
  isModelReady?: boolean;
  isModelLoading?: boolean;
  selectedModelLabel?: string;
  cacheSizeLabel?: string;
  adapterLabel?: string;
  lastCompaction?: { compacted: boolean; savedTokens: number } | null;
  inlineNotice?: {
    message: string;
    tone?: InlineNoticeTone;
    actionLabel?: string;
    onAction?: () => void;
  } | null;
  diagnosticsText?: string;
  onInitializeModel?: () => void;
  onSwitchModel?: () => void;
  onOpenLab?: () => void;
  onSelectStarter: (item: PromptStarterItem) => void;
  onEditAndBranch: (nodeId: string, newContent: string) => void;
  onRegenerate: (assistantNodeId?: string) => void;
  onSwitchBranch: (branchNodeId: string) => void;
  onApproveTool: () => void;
  onDenyTool: () => void;
  agentHandoff?: {
    title: string;
    reason: string;
    origin: string;
    progressLabel?: string;
    state: 'waiting' | 'user-active';
  } | null;
  onHandoffTakeOver?: () => void;
  onHandoffResume?: () => void;
  onHandoffCancel?: () => void;
  onCopyText: (text: string) => void;
  onSpeakText: (text: string) => void;
  onSaveMemory: (text: string) => void;
  onSelectChoice: (question: string, choiceId: string) => void;
  onRetry: () => void;
}

const OPEN_THINK_BLOCK_RE =
  /(?:<think>|<thought>|<\|thought\|>|<\|channel>thought\r?\n?)([\s\S]*?)(?:<\/think>|<\/thought>|<\/?channel\|>|<\|\/thought\|>|<\|end_of_thought\|>|$)/i;
const ORPHAN_CLOSE_THINK_RE =
  /^([\s\S]*?)(?:<\/think>|<\/thought>|<\/?channel\|>|<\|\/thought\|>|<\|end_of_thought\|>)\s*/i;
const RESIDUAL_TOOL_CALL_RE =
  /(?:<tool_call>[\s\S]*?(?:<\/tool_call>|$)|<\|tool_call>[\s\S]*?(?:<tool_call\|>|$)|<\|call>call:[\s\S]*?(?:<call\|>|$))/gi;

export function sanitizeDisplayBody(raw: string): string {
  return raw.replace(RESIDUAL_TOOL_CALL_RE, '').trim();
}

export function extractReasoningAndBody(raw: string): { reasoning: string; body: string } {
  const openMatch = raw.match(OPEN_THINK_BLOCK_RE);
  if (openMatch) {
    const reasoning = openMatch[1].trim();
    const body = sanitizeDisplayBody(raw.replace(openMatch[0], ''));
    return { reasoning, body };
  }
  const orphanClose = raw.match(ORPHAN_CLOSE_THINK_RE);
  if (orphanClose) {
    const reasoning = orphanClose[1].trim();
    const body = sanitizeDisplayBody(raw.slice(orphanClose[0].length));
    return { reasoning, body };
  }
  return { reasoning: '', body: sanitizeDisplayBody(raw) };
}

function getContextTagFromMessage(msg: HarnessMessage): string | undefined {
  if (msg.tainted) return 'Page Context';
  return undefined;
}

function safeDomainFromUrl(url?: string): string | undefined {
  if (!url) return undefined;
  try {
    return new URL(url).hostname;
  } catch {
    return undefined;
  }
}

function toInlineCitationSource(src: SourceCitation, idx: number): InlineCitationSource {
  return {
    id: src.id,
    index: idx + 1,
    label: `[${idx + 1}]`,
    title: src.title,
    domain: safeDomainFromUrl(src.url),
    snippet: src.subtitle || src.title,
    url: src.url,
  };
}

function toCitationFootnoteItem(src: SourceCitation, idx: number): CitationFootnoteItem {
  return {
    id: src.id,
    index: idx + 1,
    title: src.title,
    domain: safeDomainFromUrl(src.url),
    excerpt: src.subtitle || src.title,
    url: src.url,
    highlight: idx % 2 === 0 ? 'mono' : 'warm',
    source: toInlineCitationSource(src, idx),
  };
}

function toWebSearchSourceItem(src: SourceCitation): WebSearchSourceItem {
  return {
    id: src.id,
    title: src.title,
    url: src.url,
    domain: safeDomainFromUrl(src.url),
    snippet: src.subtitle || src.title,
  };
}

function renderBranchPicker(
  siblingInfo: BranchSiblingInfo | undefined,
  onSwitchBranch: (branchNodeId: string) => void,
) {
  if (!siblingInfo || siblingInfo.total <= 1) return undefined;
  return (
    <BranchPicker
      currentIndex={siblingInfo.activeIndex + 1}
      total={siblingInfo.total}
      onPrevious={() => {
        const prevIdx = (siblingInfo.activeIndex - 1 + siblingInfo.total) % siblingInfo.total;
        const target = siblingInfo.siblingIds[prevIdx];
        if (target) onSwitchBranch(target);
      }}
      onNext={() => {
        const nextIdx = (siblingInfo.activeIndex + 1) % siblingInfo.total;
        const target = siblingInfo.siblingIds[nextIdx];
        if (target) onSwitchBranch(target);
      }}
    />
  );
}

export function ChatThread({
  nodes,
  branchSiblings,
  isGenerating,
  turnState,
  streamingText,
  streamingReasoning,
  pendingToolCalls,
  liveToolExecutions = [],
  persistedToolTraces = [],
  activePageHint,
  pendingApproval,
  turnTelemetry,
  engineTelemetry,
  error,
  starters,
  sources,
  sourcesDrawerVariant = 'inline',
  isModelReady = true,
  isModelLoading = false,
  selectedModelLabel,
  cacheSizeLabel,
  adapterLabel,
  lastCompaction,
  inlineNotice,
  diagnosticsText,
  onInitializeModel,
  onSwitchModel,
  onOpenLab,
  onSelectStarter,
  onEditAndBranch,
  onRegenerate,
  onSwitchBranch,
  onApproveTool,
  onDenyTool,
  agentHandoff,
  onHandoffTakeOver,
  onHandoffResume,
  onHandoffCancel,
  onCopyText,
  onSpeakText,
  onSaveMemory,
  onSelectChoice,
  onRetry,
}: ChatThreadProps) {
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [feedbackByNode, setFeedbackByNode] = useState<Record<string, 'up' | 'down' | null>>({});

  const visibleNodes = nodes.filter((n) => n.message.role !== 'system');
  const isEmpty = visibleNodes.length === 0 && !isGenerating;

  const lastAssistantNodeId = (() => {
    for (let i = visibleNodes.length - 1; i >= 0; i--) {
      if (visibleNodes[i].message.role === 'assistant') {
        return visibleNodes[i].id;
      }
    }
    return null;
  })();

  const activeTurnCompletedSteps: ReturnType<typeof buildGroupStepsFromToolNodes> = [];

  if (isEmpty) {
    return (
      <div className="ext-stream">
        {inlineNotice ? (
          <InlineNotice
            tone={inlineNotice.tone ?? 'info'}
            compact
            message={inlineNotice.message}
            actionLabel={inlineNotice.actionLabel}
            onAction={inlineNotice.onAction}
          />
        ) : null}
        {!isModelReady && onInitializeModel ? (
          <OnboardingCard
            checkpointLabel={selectedModelLabel ?? 'Qwen3.5-4B (Q4F16)'}
            cacheSizeLabel={cacheSizeLabel ?? 'OPFS Verified'}
            adapterLabel={adapterLabel ?? 'WebGPU'}
            adapterReady={Boolean(engineTelemetry.computeTier)}
            loading={isModelLoading}
            onInitialize={onInitializeModel}
            secondaryLabel={onSwitchModel ? 'Switch model' : undefined}
            onSecondaryAction={onSwitchModel}
          />
        ) : (
          <EmptyPanelState
            title="Ask locally on this page."
            subtitle="Private WebGPU inference stays inside your browser."
            starters={starters.map((s) => ({
              id: s.id,
              label: s.label,
              icon: s.icon,
              prompt: s.prompt,
            }))}
            onSelectStarter={(starter) => {
              onSelectStarter({
                id: starter.id,
                label: starter.label,
                icon: starter.icon,
                prompt: starter.prompt,
              });
            }}
            privacyLabel="On-device WebGPU · Zero telemetry"
          />
        )}
        {error ? (
          <ErrorRecoveryCard
            title="Runtime Initialization Notice"
            diagnosis={error}
            badgeLabel={engineTelemetry.status === 'error' ? 'WebGPU Runtime' : 'Recoverable'}
            diagnostics={diagnosticsText}
            retryLabel="Retry"
            switchModelLabel={onSwitchModel ? 'Switch model' : undefined}
            onRetry={onRetry}
            onSwitchModel={onSwitchModel}
            onCopyDiagnostics={onCopyText}
          />
        ) : null}
      </div>
    );
  }

  return (
    <div className="ext-stream">
      {inlineNotice ? (
        <InlineNotice
          tone={inlineNotice.tone ?? 'info'}
          compact
          message={inlineNotice.message}
          actionLabel={inlineNotice.actionLabel}
          onAction={inlineNotice.onAction}
        />
      ) : null}
      {lastCompaction?.compacted && lastCompaction.savedTokens > 0 ? (
        <InlineNotice
          tone="info"
          compact
          icon="sliders"
          message={`Compacted context to preserve local KV headroom (${lastCompaction.savedTokens} tokens saved).`}
          actionLabel={onOpenLab ? 'KV Budget' : undefined}
          onAction={onOpenLab}
        />
      ) : null}
      {(() => {
        const rendered: React.ReactNode[] = [];
        for (let i = 0; i < visibleNodes.length; i += 1) {
          const node = visibleNodes[i]!;
          const msg = node.message;
          const siblingInfo = branchSiblings[node.id];

          if (msg.role === 'user') {
            const isEditing = editingNodeId === node.id;
            rendered.push(
              <div key={node.id} className="ext-tool-group">
                <UserMessageBubble
                  content={msg.content}
                  contextLabel={getContextTagFromMessage(msg)}
                  isEditing={isEditing}
                  onStartEdit={() => setEditingNodeId(node.id)}
                  onCancelEdit={() => setEditingNodeId(null)}
                  onSaveEdit={(next) => {
                    setEditingNodeId(null);
                    if (next.trim() && next !== msg.content) {
                      onEditAndBranch(node.id, next);
                    }
                  }}
                  onCopy={() => onCopyText(msg.content)}
                />
                {renderBranchPicker(siblingInfo, onSwitchBranch)}
              </div>,
            );
            continue;
          }

          if (msg.role === 'tool') {
            const toolNodes: Array<{ node: MessageNode; prevMessage?: HarnessMessage }> = [];
            let j = i;
            while (j < visibleNodes.length) {
              const cand = visibleNodes[j]!;
              if (cand.message.role === 'tool') {
                const prevMessage = j > 0 ? visibleNodes[j - 1]?.message : undefined;
                toolNodes.push({ node: cand, prevMessage });
                j += 1;
              } else if (
                cand.message.role === 'assistant' &&
                cand.message.toolName &&
                !extractReasoningAndBody(cand.message.content).body
              ) {
                j += 1;
              } else {
                break;
              }
            }
            const groupSteps = buildGroupStepsFromToolNodes({
              toolNodes,
              persistedTraces: persistedToolTraces,
              activePage: activePageHint,
            });
            const isTrailingActiveToolGroup = isGenerating && j >= visibleNodes.length;
            if (isTrailingActiveToolGroup) {
              activeTurnCompletedSteps.push(...groupSteps);
            } else {
              const { title, summary } = formatToolChainSummary(groupSteps, false);
              rendered.push(
                <ToolTraceList
                  key={`trace-${node.id}`}
                  variant="inline"
                  title={title}
                  summary={summary}
                  steps={groupSteps}
                  defaultCollapsed={true}
                />,
              );
            }
            i = j - 1;
            continue;
          }

          if (msg.role === 'assistant') {
            const { reasoning: parsedReasoning, body } = extractReasoningAndBody(msg.content);
            const persistedReasoning =
              typeof msg.reasoning === 'string' ? msg.reasoning.trim() : '';
            const combinedReasoning = persistedReasoning || parsedReasoning;

            if (msg.toolName && !body) {
              if (combinedReasoning) {
                rendered.push(
                  <ReasoningTrace key={node.id} status="done" content={combinedReasoning} />,
                );
              }
              continue;
            }

            const isLastAssistant = node.id === lastAssistantNodeId;
            const speedMetrics =
              isLastAssistant && (turnTelemetry || engineTelemetry.tokensPerSec !== null)
                ? {
                    tokensPerSec:
                      turnTelemetry?.tokensPerSec ?? engineTelemetry.tokensPerSec ?? undefined,
                    ttftMs: turnTelemetry?.ttftMs ?? engineTelemetry.ttftMs ?? undefined,
                  }
                : undefined;

            const picker = renderBranchPicker(siblingInfo, onSwitchBranch);

            rendered.push(
              <AssistantMessage
                key={node.id}
                footer={
                  <ActionBar
                    onCopy={() => onCopyText(body)}
                    onRegenerate={() => onRegenerate(node.id)}
                    onSpeak={() => onSpeakText(body)}
                    onSave={() => onSaveMemory(body)}
                    onThumbUp={() =>
                      setFeedbackByNode((prev) => ({
                        ...prev,
                        [node.id]: prev[node.id] === 'up' ? null : 'up',
                      }))
                    }
                    onThumbDown={() =>
                      setFeedbackByNode((prev) => ({
                        ...prev,
                        [node.id]: prev[node.id] === 'down' ? null : 'down',
                      }))
                    }
                    feedback={feedbackByNode[node.id] ?? null}
                    speedMetrics={speedMetrics}
                    rightSlot={picker}
                  />
                }
              >
                {combinedReasoning ? <ReasoningTrace status="done" content={combinedReasoning} /> : null}
                <GenUiRenderer
                  content={body}
                  status="done"
                  onCopyText={onCopyText}
                  onSaveText={onSaveMemory}
                  onSelectChoice={onSelectChoice}
                />
              </AssistantMessage>,
            );
          }
        }
        return rendered;
      })()}

      {agentHandoff ? (
        <AgentHandoffCard
          title={agentHandoff.title}
          reason={agentHandoff.reason}
          origin={agentHandoff.origin}
          progressLabel={agentHandoff.progressLabel}
          state={agentHandoff.state}
          onTakeOver={onHandoffTakeOver}
          onResume={onHandoffResume}
          onCancel={onHandoffCancel}
        />
      ) : null}

      {pendingApproval ? (
        <PermissionCard
          title={`Allow ${pendingApproval.toolName}?`}
          domain={
            safeDomainFromUrl(
              typeof pendingApproval.arguments?.url === 'string'
                ? pendingApproval.arguments.url
                : undefined,
            ) ?? 'local-extension'
          }
          scope={`${pendingApproval.toolName}(${JSON.stringify(pendingApproval.arguments ?? {}).slice(0, 68)})`}
          reason={pendingApproval.decision.reason}
          ruleLabel={pendingApproval.decision.triggeredRule ?? 'IDPI Guard'}
          allowOnceLabel="Allow once"
          denyLabel="Deny"
          onAllowOnce={onApproveTool}
          onDeny={onDenyTool}
        />
      ) : null}

      {(() => {
        const unifiedActiveToolSteps = isGenerating
          ? mergeActiveTurnToolSteps({
              completedTurnSteps: activeTurnCompletedSteps,
              pendingToolCalls,
              liveToolExecutions,
              activePage: activePageHint,
            })
          : [];
        const liveBody = streamingText ? extractReasoningAndBody(streamingText).body : '';
        const showPrefillStatus =
          isGenerating &&
          unifiedActiveToolSteps.length === 0 &&
          !streamingReasoning &&
          !liveBody;
        const hasLiveBlock =
          isGenerating &&
          (streamingReasoning || streamingText || unifiedActiveToolSteps.length > 0 || showPrefillStatus);
        if (!hasLiveBlock) return null;
        const activeChainSummary =
          unifiedActiveToolSteps.length > 0
            ? formatToolChainSummary(unifiedActiveToolSteps, true)
            : null;
        const hasRunningToolStep = unifiedActiveToolSteps.some((step) => step.status === 'running');
        return (
          <div className="ext-tool-group">
            {unifiedActiveToolSteps.length > 0 && activeChainSummary ? (
              <ToolTraceList
                variant="inline"
                timer={hasRunningToolStep}
                title={activeChainSummary.title}
                summary={activeChainSummary.summary}
                defaultCollapsed={false}
                steps={unifiedActiveToolSteps}
              />
            ) : null}
            {showPrefillStatus ? (
              <InlineToolStatus
                state={turnState === 'compacting' ? 'analyzing' : 'thinking'}
                label={turnState === 'compacting' ? 'Compacting context' : 'Thinking'}
                timer
                shimmer
              />
            ) : null}
            {streamingReasoning ? (
              <ReasoningTrace
                status={liveBody || hasRunningToolStep ? 'done' : 'thinking'}
                content={streamingReasoning}
                defaultExpanded={false}
              />
            ) : null}
            {liveBody ? (
              <AssistantMessage>
                <GenUiRenderer
                  content={liveBody}
                  status="streaming"
                  onCopyText={onCopyText}
                  onSaveText={onSaveMemory}
                  onSelectChoice={onSelectChoice}
                />
              </AssistantMessage>
            ) : null}
          </div>
        );
      })()}

      {sources.length > 0 ? (
        <div className="ext-sources-group">
          <WebSearchSourcesStrip
            sources={sources.map(toWebSearchSourceItem)}
            initialVisibleCount={4}
            onSelectSource={(target) => {
              if (
                target.url &&
                /^https?:\/\//i.test(target.url) &&
                typeof chrome !== 'undefined' &&
                chrome.tabs?.create
              ) {
                void chrome.tabs.create({ url: target.url });
              }
            }}
          />
          <div className="ext-citation-row">
            {sources.slice(0, 6).map((src, idx) => (
              <InlineCitationPopover
                key={src.id}
                source={toInlineCitationSource(src, idx)}
                hoverOpen
                onJumpToSource={(target) => {
                  if (
                    target.url &&
                    /^https?:\/\//i.test(target.url) &&
                    typeof chrome !== 'undefined' &&
                    chrome.tabs?.create
                  ) {
                    void chrome.tabs.create({ url: target.url });
                  }
                }}
              />
            ))}
          </div>
          <SourcesDrawer sources={sources} variant={sourcesDrawerVariant} />
          <CitationFootnoteList
            title="Verified Citations"
            citations={sources.slice(0, 6).map((src, idx) => toCitationFootnoteItem(src, idx))}
            onShowOnPage={(cit) => {
              if (
                cit.url &&
                /^https?:\/\//i.test(cit.url) &&
                typeof chrome !== 'undefined' &&
                chrome.tabs?.create
              ) {
                void chrome.tabs.create({ url: cit.url });
              }
            }}
            onCopyQuote={(cit) => onCopyText(`${cit.title}: ${cit.excerpt}`)}
          />
        </div>
      ) : null}

      {error ? (
        <ErrorRecoveryCard
          title="Generation Interrupted"
          diagnosis={error}
          badgeLabel={engineTelemetry.status === 'error' ? 'WebGPU Runtime' : 'Recoverable'}
          diagnostics={diagnosticsText}
          retryLabel="Retry"
          switchModelLabel={onSwitchModel ? 'Switch model' : undefined}
          onRetry={onRetry}
          onSwitchModel={onSwitchModel}
          onCopyDiagnostics={onCopyText}
        />
      ) : null}
    </div>
  );
}
