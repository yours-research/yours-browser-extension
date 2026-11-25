import {
  AssistantRuntimeProvider,
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
  type ChatModelAdapter,
} from '@assistant-ui/react';

interface ChatThreadProps {
  adapter: ChatModelAdapter;
  modelReady: boolean;
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

export function ChatThread({ adapter, modelReady }: ChatThreadProps) {
  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <div className="thread-empty">
              <div className="empty-title">TinyGPT WebGPU POC</div>
              <div className="empty-subtitle">
                {modelReady
                  ? 'Model is warm in WebGPU VRAM. Send a message or paste a 32K+ prompt.'
                  : 'Click "Load Model" above to load Qwen3.5-4B into OPFS + WebGPU, or open Lab to run Phase 0 spikes.'}
              </div>
            </div>
          </AuiIf>

          <ThreadPrimitive.Messages>
            {({ message }) => <ChatMessageRow role={message.role} />}
          </ThreadPrimitive.Messages>
        </ThreadPrimitive.Viewport>

        <ThreadPrimitive.ViewportFooter className="composer-footer">
          <ComposerPrimitive.Root className="composer-root">
            <ComposerPrimitive.Input
              placeholder={
                modelReady
                  ? 'Ask Qwen3.5 locally via WebGPU...'
                  : 'Load model first (sending auto-loads Qwen3.5-4B)...'
              }
              className="composer-input"
              rows={2}
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
