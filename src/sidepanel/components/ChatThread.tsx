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
  backgroundWarm?: boolean;
  researchGranted?: boolean;
  onGrantResearch?: () => void;
  toolStatus?: ToolStatusUpdate | null;
  permissionPrompt?: { toolName: string; label: string } | null;
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
  backgroundWarm = false,
  researchGranted = false,
  onGrantResearch,
  toolStatus,
  permissionPrompt,
  onDismissPermission,
  onComposerTextChange,
}: ChatThreadProps) {
  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <div className="thread-empty">
              <div className="empty-title">TinyGPT Local Assistant</div>
              <div className="empty-subtitle">
                {modelReady
                  ? 'Model & Base Slot 0 are warm in WebGPU VRAM. Ask about this tab, run calculations, or enable Web Research.'
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
                className={includeActivePage ? 'btn btn-active' : 'btn'}
                onClick={onToggleActivePage}
                title="Speculatively extract and ground responses with active tab content"
              >
                {includeActivePage
                  ? `Page: ${activePageTitle ? activePageTitle.slice(0, 22) : 'Auto'}${backgroundWarm ? ' (warm)' : ''}`
                  : 'Page: Off'}
              </button>
            )}
            {onGrantResearch && (
              <button
                type="button"
                className={researchGranted ? 'btn btn-active' : 'btn'}
                onClick={onGrantResearch}
                title="Optional HTTPS host permission for live web_search & fetch_and_read_url"
              >
                {researchGranted ? 'Research: On' : 'Research: Off'}
              </button>
            )}
            {toolStatus && (
              <span className="badge badge-active" title={toolStatus.toolName}>
                {toolStatus.label}
              </span>
            )}
          </div>

          {permissionPrompt && onGrantResearch && (
            <div className="progress-banner" style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
              <span>{permissionPrompt.label}</span>
              <span style={{ display: 'flex', gap: 4 }}>
                <button type="button" className="btn btn-primary" onClick={onGrantResearch}>
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
