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
  MessageSkeleton,
  OnboardingCard,
  PermissionCard,
  ReasoningTrace,
  SourcesDrawer,
  ToolTraceList,
  UserMessageBubble,
  type CitationFootnoteItem,
  type InlineCitationSource,
  type InlineNoticeTone,
  type PromptStarterItem,
  type SourceCitation,
  type ToolTraceStep,
} from '@yours/ui';
import type { TelemetryState } from '../../shared/protocol.ts';
import { GenUiRenderer } from './GenUiRenderer.tsx';

export interface ChatThreadProps {
  nodes: MessageNode[];
  branchSiblings: Record<string, BranchSiblingInfo>;
  isGenerating: boolean;
  turnState: TurnState;
  streamingText: string;
  streamingReasoning: string;
  pendingToolCalls: Array<{ id: string; name: string }>;
  pendingApproval: PendingApprovalRequest | null;
  turnTelemetry: TurnTelemetrySnapshot | null;
  engineTelemetry: TelemetryState;
  error: string | null;
  starters: readonly PromptStarterItem[];
  sources: readonly SourceCitation[];
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

function getToolIcon(toolName?: string): ToolTraceStep['icon'] {
  if (!toolName) return 'sliders';
  if (toolName.includes('search') || toolName.includes('fetch')) return 'globe';
  if (toolName.includes('tab')) return 'page-scan';
  if (toolName.includes('memory')) return 'bookmark';
  if (toolName.includes('sandbox') || toolName.includes('calc')) return 'compare';
  return 'sliders';
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

export function ChatThread({
  nodes,
  branchSiblings,
  isGenerating,
  turnState,
  streamingText,
  streamingReasoning,
  pendingToolCalls,
  pendingApproval,
  turnTelemetry,
  engineTelemetry,
  error,
  starters,
  sources,
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
                {siblingInfo && siblingInfo.total > 1 ? (
                  <BranchPicker
                    currentIndex={siblingInfo.activeIndex}
                    total={siblingInfo.total}
                    onPrevious={() => {
                      const prevIdx =
                        (siblingInfo.activeIndex - 1 + siblingInfo.total) % siblingInfo.total;
                      const target = siblingInfo.siblingIds[prevIdx];
                      if (target) onSwitchBranch(target);
                    }}
                    onNext={() => {
                      const nextIdx = (siblingInfo.activeIndex + 1) % siblingInfo.total;
                      const target = siblingInfo.siblingIds[nextIdx];
                      if (target) onSwitchBranch(target);
                    }}
                  />
                ) : null}
              </div>,
            );
            continue;
          }

          if (msg.role === 'tool') {
            const groupSteps: ToolTraceStep[] = [];
            let j = i;
            while (j < visibleNodes.length) {
              const cand = visibleNodes[j]!;
              if (cand.message.role === 'tool') {
                const tm = cand.message;
                const isBlocked = /^blocked|denied/i.test(tm.content);
                const isErr = !isBlocked && /^error\b|failed/i.test(tm.content);
                const prevCall = j > 0 ? visibleNodes[j - 1]?.message : undefined;
                groupSteps.push({
                  id: cand.id,
                  tool: tm.toolName ?? 'tool',
                  icon: getToolIcon(tm.toolName),
                  status: isBlocked ? 'blocked' : isErr ? 'error' : 'done',
                  input:
                    prevCall?.role === 'assistant' && prevCall.toolArgs
                      ? JSON.stringify(prevCall.toolArgs)
                      : undefined,
                  output: tm.content.slice(0, 320),
                });
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
            rendered.push(
              <ToolTraceList
                key={`trace-${node.id}`}
                title="Local Tool Trace"
                summary={`${groupSteps.length} step${groupSteps.length === 1 ? '' : 's'}`}
                steps={groupSteps}
                defaultCollapsed={false}
              />,
            );
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

            const picker =
              siblingInfo && siblingInfo.total > 1 ? (
                <BranchPicker
                  currentIndex={siblingInfo.activeIndex}
                  total={siblingInfo.total}
                  onPrevious={() => {
                    const prevIdx =
                      (siblingInfo.activeIndex - 1 + siblingInfo.total) % siblingInfo.total;
                    const target = siblingInfo.siblingIds[prevIdx];
                    if (target) onSwitchBranch(target);
                  }}
                  onNext={() => {
                    const nextIdx = (siblingInfo.activeIndex + 1) % siblingInfo.total;
                    const target = siblingInfo.siblingIds[nextIdx];
                    if (target) onSwitchBranch(target);
                  }}
                />
              ) : undefined;

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

      {isGenerating && (streamingReasoning || streamingText || pendingToolCalls.length > 0) ? (
        <div className="ext-tool-group">
          {pendingToolCalls.length > 0 ? (
            <ToolTraceList
              title="Active Tool Call"
              summary={`${pendingToolCalls.length} running`}
              defaultCollapsed={false}
              steps={pendingToolCalls.map((tc) => ({
                id: tc.id,
                tool: tc.name,
                icon: getToolIcon(tc.name),
                status: 'running',
                label: 'Executing local MV3 tool',
              }))}
            />
          ) : null}
          {streamingReasoning ? (
            <ReasoningTrace
              status={streamingText ? 'done' : 'thinking'}
              content={streamingReasoning}
              defaultExpanded={false}
            />
          ) : null}
          {(() => {
            const liveBody = streamingText ? extractReasoningAndBody(streamingText).body : '';
            if (liveBody) {
              return (
                <AssistantMessage>
                  <GenUiRenderer
                    content={liveBody}
                    status="streaming"
                    onCopyText={onCopyText}
                    onSaveText={onSaveMemory}
                    onSelectChoice={onSelectChoice}
                  />
                </AssistantMessage>
              );
            }
            if (turnState === 'prefilling') {
              return <MessageSkeleton />;
            }
            return null;
          })()}
        </div>
      ) : null}

      {sources.length > 0 ? (
        <div className="ext-sources-group">
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
          <SourcesDrawer sources={sources} />
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
