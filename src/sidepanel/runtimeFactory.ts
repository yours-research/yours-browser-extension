import type {
  GenerateRequest,
  GpuPacingMode,
  HarnessMessage,
  RuntimeAdapter,
  RuntimeCapabilities,
  StreamDelta,
  WgslComputeTier,
} from '@yours/harness';
import {
  ChromePromptAiAdapter,
  OpenAiCompatibleAdapter,
  WgslWorkerAdapter,
} from '@yours/harness/adapters';
import { toQwenToolSchemas } from '@yours/tools/schemas';
import type { OffscreenEngineClient } from './engineClient.ts';

export type RuntimeBackendKind = 'wgsl' | 'chrome-ai' | 'openai';

export interface OpenAiRuntimeConfig {
  baseUrl: string;
  model: string;
  apiKey?: string;
}

export interface DelegatingRuntimeAdapterOptions {
  getAdapter: () => RuntimeAdapter;
  transformRequest?: (request: GenerateRequest) => GenerateRequest;
}

/**
 * Stable wrapper `RuntimeAdapter` whose underlying backend (`WgslWorkerAdapter`,
 * `ChromePromptAiAdapter`, `OpenAiCompatibleAdapter`) can be swapped live without
 * recreating `useYoursChat`'s `ChatSession`.
 */
export class DelegatingRuntimeAdapter implements RuntimeAdapter {
  private readonly getAdapter: () => RuntimeAdapter;
  private readonly transformRequest?: (request: GenerateRequest) => GenerateRequest;

  constructor(options: DelegatingRuntimeAdapterOptions) {
    this.getAdapter = options.getAdapter;
    this.transformRequest = options.transformRequest;
  }

  capabilities(): RuntimeCapabilities | Promise<RuntimeCapabilities> {
    return this.getAdapter().capabilities();
  }

  countTokens(messages: HarnessMessage[]): Promise<number> | number {
    return this.getAdapter().countTokens(messages);
  }

  async prefillSlot(
    slotKey: string,
    messages: HarnessMessage[],
    signal?: AbortSignal,
  ): Promise<{ tokens: number; elapsedMs: number }> {
    const target = this.getAdapter();
    if (typeof target.prefillSlot === 'function') {
      return target.prefillSlot(slotKey, messages, signal);
    }
    return { tokens: 0, elapsedMs: 0 };
  }

  async evictSlot(slotKey: string): Promise<void> {
    const target = this.getAdapter();
    if (typeof target.evictSlot === 'function') {
      await target.evictSlot(slotKey);
    }
  }

  generate(request: GenerateRequest, signal?: AbortSignal): AsyncIterable<StreamDelta> {
    const target = this.getAdapter();
    const effectiveRequest = this.transformRequest ? this.transformRequest(request) : request;
    return target.generate(effectiveRequest, signal);
  }
}

export function createWgslRuntimeAdapter(
  client: OffscreenEngineClient,
  opts?: {
    modelId?: string;
    maxContextTokens?: number;
    tier?: WgslComputeTier;
    pacingMode?: GpuPacingMode;
  },
): WgslWorkerAdapter {
  return new WgslWorkerAdapter({
    id: opts?.modelId ?? 'wgsl-local',
    transport: client.createWgslPortTransport(),
    maxContextTokens: opts?.maxContextTokens ?? 32768,
    tier: opts?.tier ?? 'tier1-f16-subgroups',
    defaultPacingMode: opts?.pacingMode ?? 'balanced',
    supportsTools: true,
    serializeTools: (tools) =>
      toQwenToolSchemas(tools as unknown as Parameters<typeof toQwenToolSchemas>[0]),
  });
}

export function createChromeAiRuntimeAdapter(): ChromePromptAiAdapter {
  return new ChromePromptAiAdapter({ id: 'chrome-prompt-ai' });
}

export function createOpenAiRuntimeAdapter(cfg: OpenAiRuntimeConfig): OpenAiCompatibleAdapter {
  return new OpenAiCompatibleAdapter({
    id: `openai:${cfg.model}`,
    baseUrl: cfg.baseUrl,
    model: cfg.model,
    apiKey: cfg.apiKey,
    supportsTools: true,
  });
}
