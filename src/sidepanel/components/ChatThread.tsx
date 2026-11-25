import {
  AssistantRuntimeProvider,
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
  type ChatModelAdapter,
} from '@assistant-ui/react';
import type { ToolStatusUpdate } from '../agentController.ts';

interface ChatThreadProps {
  adapter: ChatModelAdapter;
  modelReady: boolean;
  includeActivePage?: boolean;
  onToggleActivePage?: () => void;
  activePageTitle?: string | null;
  activePageError?: string | null;
  backgroundPrefillState?: 'idle' | 'prefilling' | 'warm';
  researchEnabled?: boolean;
  onToggleResearch?: () => void;
  toolStatus?: ToolStatusUpdate | null;
  permissionPrompt?: { toolName: string; label: string } | null;
  onGrantResearchPermission?: () => void;
  onDismissPermission?: () => void;
  onComposerTextChange?: (text: string) => void;
}

function ChatMessageRow({ role }: { role: string }) {
  const isUser = role === 'user';
  return (
    <MessagePrimitive.Root className={isUser ? 'msg-row msg-row-user' : 'msg-row msg-row-assistant'}>
      <div className={isUser ? 'msg-bubble msg-user' : 'msg-bubble msg-assistant'}>
        <div className="msg-role">{isUser ? 'You' : 'Qwen3.5'}</div>
        <div className="msg-content">
          <MessagePrimitive.Parts />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

export function ChatThread({
  adapter,
  modelReady,
  includeActivePage = true,
  onToggleActivePage,
  activePageTitle,
  activePageError,
  backgroundPrefillState = 'idle',
  researchEnabled = false,
  onToggleResearch,
  toolStatus,
  permissionPrompt,
  onGrantResearchPermission,
  onDismissPermission,
  onComposerTextChange,
}: ChatThreadProps) {
  const runtime = useLocalRuntime(adapter);

  const pageLabel = !includeActivePage
    ? 'Page: Off'
    : activePageTitle
      ? `Page: ${activePageTitle.slice(0, 22)}${
          backgroundPrefillState === 'warm'
            ? ' (warm)'
            : backgroundPrefillState === 'prefilling'
              ? ' (prefilling...)'
              : ''
        }`
      : activePageError
        ? 'Page: No access'
        : 'Page: Reading...';

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <div className="thread-empty">
              <div className="empty-title">TinyGPT Local Assistant</div>
              <div className="empty-subtitle">
                {modelReady
                  ? 'Model is warm in WebGPU VRAM. Toggle Page / Research below or start chatting.'
                  : 'Auto-loading Qwen3.5-4B into WebGPU... You can start typing right away.'}
              </div>
            </div>
          </AuiIf>

          <ThreadPrimitive.Messages>
            {({ message }) => <ChatMessageRow role={message.role} />}
          </ThreadPrimitive.Messages>
        </ThreadPrimitive.Viewport>

        <ThreadPrimitive.ViewportFooter className="composer-footer">
          <div className="btn-row" style={{ marginBottom: 6, alignItems: 'center' }}>
            {onToggleActivePage && (
              <button
                type="button"
                className={includeActivePage && !activePageError ? 'btn btn-active' : 'btn'}
                onClick={onToggleActivePage}
                title={
                  activePageError
                    ? `Active tab unreadable without activeTab gesture or Research grant (${activePageError}). Click to toggle off.`
                    : 'Speculatively extract and ground responses with active tab content'
                }
              >
                {pageLabel}
              </button>
            )}
            {onToggleResearch && (
              <button
                type="button"
                className={researchEnabled ? 'btn btn-active' : 'btn'}
                onClick={onToggleResearch}
                title="Toggle live web_search and URL research (prompts for https://*/* on first enable)"
              >
                {researchEnabled ? 'Research: On' : 'Research: Off'}
              </button>
            )}
            {toolStatus && (
              <span className="badge badge-active" title={toolStatus.toolName}>
                {toolStatus.label}
              </span>
            )}
          </div>

          {permissionPrompt && (onGrantResearchPermission || onToggleResearch) && (
            <div className="progress-banner" style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
              <span>{permissionPrompt.label}</span>
              <span style={{ display: 'flex', gap: 4 }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={onGrantResearchPermission ?? onToggleResearch}
                >
                  Enable Web Research
                </button>
                {onDismissPermission && (
                  <button type="button" className="btn" onClick={onDismissPermission}>
                    Dismiss
                  </button>
                )}
              </span>
            </div>
          )}

          <ComposerPrimitive.Root className="composer-root">
            <ComposerPrimitive.Input
              placeholder={
                modelReady
                  ? 'Ask about this page, calculate, or search locally...'
                  : 'Model loading in background — start typing...'
              }
              className="composer-input"
              rows={2}
              onInput={(e) => onComposerTextChange?.(e.currentTarget.value)}
            />
            <div className="composer-actions">
              <AuiIf condition={(s) => s.thread.isRunning}>
                <ComposerPrimitive.Cancel className="btn btn-danger">
                  Stop
                </ComposerPrimitive.Cancel>
              </AuiIf>
              <AuiIf condition={(s) => !s.thread.isRunning}>
                <ComposerPrimitive.Send className="btn btn-primary">
                  Send
                </ComposerPrimitive.Send>
              </AuiIf>
            </div>
          </ComposerPrimitive.Root>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}
