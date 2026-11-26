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
  EmptyPanelState,
  ErrorRecoveryBanner,
  InlineCitationPopover,
  MessageSkeleton,
  PermissionApprovalCard,
  ReasoningTrace,
  SourcesDrawer,
  ToolStepCard,
  UserMessageBubble,
  type InlineCitationSource,
  type PromptStarterItem,
  type SourceCitation,
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
      </div>
    );
  }

  return (
    <div className="ext-stream">
      {visibleNodes.map((node) => {
        const msg = node.message;
        const siblingInfo = branchSiblings[node.id];

        if (msg.role === 'user') {
          const isEditing = editingNodeId === node.id;
          return (
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
            </div>
          );
        }

        if (msg.role === 'tool') {
          const isErr = /^error\b|failed|blocked/i.test(msg.content);
          return (
            <ToolStepCard
              key={node.id}
              status={isErr ? 'error' : 'done'}
              title={msg.toolName ?? 'Tool step'}
              summary={msg.content.slice(0, 96)}
              details={msg.content}
            />
          );
        }

        if (msg.role === 'assistant') {
          const { reasoning: parsedReasoning, body } = extractReasoningAndBody(msg.content);
          const persistedReasoning =
            typeof msg.reasoning === 'string' ? msg.reasoning.trim() : '';
          const combinedReasoning = persistedReasoning || parsedReasoning;

          if (msg.toolName && !body) {
            return combinedReasoning ? (
              <ReasoningTrace key={node.id} status="done" content={combinedReasoning} />
            ) : null;
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

          return (
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
            </AssistantMessage>
          );
        }

        return null;
      })}

      {pendingApproval ? (
        <PermissionApprovalCard
          title={`Approve ${pendingApproval.toolName}`}
          description={pendingApproval.decision.reason}
          ruleBadge={pendingApproval.decision.triggeredRule ?? 'IDPI Guard'}
          approveLabel="Approve"
          denyLabel="Deny"
          onApprove={onApproveTool}
          onDeny={onDenyTool}
        />
      ) : null}

      {isGenerating && (streamingReasoning || streamingText || pendingToolCalls.length > 0) ? (
        <div className="ext-tool-group">
          {pendingToolCalls.map((tc) => (
            <ToolStepCard
              key={tc.id}
              status="running"
              title={`Running ${tc.name}`}
              summary="Executing local MV3 tool..."
            />
          ))}
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
        </div>
      ) : null}

      {error ? (
        <ErrorRecoveryBanner
          title="Generation Interrupted"
          description={error}
          retryLabel="Retry"
          onRetry={onRetry}
        />
      ) : null}
    </div>
  );
}
