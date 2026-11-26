import {
  HybridMemoryEngine,
  SensitiveDomainBlockedError,
  type BlockedDomainRule,
  type DomainRuleKind,
  type HarnessMessage,
  type MemoryHit,
  type MemoryProvider,
  type MemoryQuery,
  type MemoryRecord,
  type MemoryRecordInput,
  type MemoryStorageStats,
  type MemoryTier,
} from '@yours/memory';
import { createMv3WorkerClient, type MemoryWorkerClient } from '@yours/memory/worker';

export interface ExtensionMemoryClient extends MemoryProvider {
  readonly backendKind: 'worker' | 'in-memory';
  init(): Promise<void>;
  getStats(): Promise<MemoryStorageStats>;
  listBlockedDomains(): Promise<BlockedDomainRule[]>;
  addBlockedDomain(pattern: string, kind?: DomainRuleKind, reason?: string): Promise<BlockedDomainRule>;
  removeBlockedDomain(id: string): Promise<boolean>;
  isUrlBlocked(url: string): Promise<{ blocked: boolean; reason?: string; rule?: string }>;
  rebuildIndex(): Promise<number>;
  close(): Promise<void>;
}

/**
 * Checks whether an error (local or serialized over Worker RPC) represents a
 * blocked sensitive domain violation from `@yours/memory`.
 */
export function isSensitiveDomainBlockedError(err: unknown): boolean {
  if (err instanceof SensitiveDomainBlockedError) {
    return true;
  }
  if (!err || typeof err !== 'object') {
    return typeof err === 'string' && /sensitive domain|blocked domain/i.test(err);
  }
  const rec = err as { name?: string; code?: string; message?: string };
  if (rec.name === 'SensitiveDomainBlockedError' || rec.code === 'SENSITIVE_DOMAIN_BLOCKED') {
    return true;
  }
  return typeof rec.message === 'string' && /sensitive[- ]domain|blocked by .*domain/i.test(rec.message);
}

class WorkerBackedMemoryClient implements ExtensionMemoryClient {
  readonly backendKind = 'worker' as const;
  private readonly worker: Worker;
  private readonly rpc: MemoryWorkerClient;

  constructor(worker: Worker) {
    this.worker = worker;
    this.rpc = createMv3WorkerClient(worker);
  }

  init(): Promise<void> {
    return this.rpc.init();
  }

  remember(input: MemoryRecordInput): Promise<MemoryRecord> {
    return this.rpc.remember(input);
  }

  recall(query: MemoryQuery): Promise<MemoryHit[]> {
    return this.rpc.recall(query);
  }

  consolidate(threadId: string, messages: HarnessMessage[]): Promise<{ added: number; superseded: number }> {
    return this.rpc.consolidate(threadId, messages);
  }

  inspect(filter?: { tier?: MemoryTier; limit?: number; offset?: number }): Promise<MemoryRecord[]> {
    return this.rpc.inspect(filter);
  }

  forget(id: string): Promise<boolean> {
    return this.rpc.forget(id);
  }

  exportEncrypted(passphrase: string): Promise<Uint8Array> {
    return this.rpc.exportEncrypted(passphrase);
  }

  importEncrypted(blob: Uint8Array, passphrase: string): Promise<{ imported: number }> {
    return this.rpc.importEncrypted(blob, passphrase);
  }

  wipeAll(): Promise<void> {
    return this.rpc.wipeAll();
  }

  getStats(): Promise<MemoryStorageStats> {
    return this.rpc.getStats();
  }

  listBlockedDomains(): Promise<BlockedDomainRule[]> {
    return this.rpc.listBlockedDomains();
  }

  addBlockedDomain(pattern: string, kind?: DomainRuleKind, reason?: string): Promise<BlockedDomainRule> {
    return this.rpc.addBlockedDomain(pattern, kind, reason);
  }

  removeBlockedDomain(id: string): Promise<boolean> {
    return this.rpc.removeBlockedDomain(id);
  }

  isUrlBlocked(url: string): Promise<{ blocked: boolean; reason?: string; rule?: string }> {
    return this.rpc.isUrlBlocked(url);
  }

  rebuildIndex(): Promise<number> {
    return this.rpc.rebuildIndex();
  }

  async close(): Promise<void> {
    await this.rpc.close(true);
    this.worker.terminate();
  }
}

class LocalEngineMemoryClient implements ExtensionMemoryClient {
  readonly backendKind = 'in-memory' as const;
  private readonly engine: HybridMemoryEngine;

  constructor() {
    this.engine = new HybridMemoryEngine({
      dbName: 'yours-extension-memory',
      includeBuiltinBlocklist: true,
    });
  }

  init(): Promise<void> {
    return this.engine.init();
  }

  remember(input: MemoryRecordInput): Promise<MemoryRecord> {
    return this.engine.remember(input);
  }

  recall(query: MemoryQuery): Promise<MemoryHit[]> {
    return this.engine.recall(query);
  }

  consolidate(threadId: string, messages: HarnessMessage[]): Promise<{ added: number; superseded: number }> {
    return this.engine.consolidate(threadId, messages);
  }

  inspect(filter?: { tier?: MemoryTier; limit?: number; offset?: number }): Promise<MemoryRecord[]> {
    return this.engine.inspect(filter);
  }

  forget(id: string): Promise<boolean> {
    return this.engine.forget(id);
  }

  exportEncrypted(passphrase: string): Promise<Uint8Array> {
    return this.engine.exportEncrypted(passphrase);
  }

  importEncrypted(blob: Uint8Array, passphrase: string): Promise<{ imported: number }> {
    return this.engine.importEncrypted(blob, passphrase);
  }

  wipeAll(): Promise<void> {
    return this.engine.wipeAll();
  }

  getStats(): Promise<MemoryStorageStats> {
    return this.engine.getStats();
  }

  listBlockedDomains(): Promise<BlockedDomainRule[]> {
    return this.engine.listBlockedDomains();
  }

  addBlockedDomain(pattern: string, kind?: DomainRuleKind, reason?: string): Promise<BlockedDomainRule> {
    return this.engine.addBlockedDomain(pattern, kind, reason);
  }

  removeBlockedDomain(id: string): Promise<boolean> {
    return this.engine.removeBlockedDomain(id);
  }

  async isUrlBlocked(url: string): Promise<{ blocked: boolean; reason?: string; rule?: string }> {
    return this.engine.isUrlBlocked(url);
  }

  rebuildIndex(): Promise<number> {
    return this.engine.rebuildIndex();
  }

  close(): Promise<void> {
    return this.engine.close();
  }
}

/**
 * Creates a local `@yours/memory` provider backed by a Dedicated Web Worker
 * (`@yours/memory/worker`) in browser contexts and `HybridMemoryEngine` in Node tests.
 */
export function createExtensionMemoryClient(): ExtensionMemoryClient {
  if (
    typeof Worker !== 'undefined' &&
    typeof chrome !== 'undefined' &&
    typeof chrome.runtime?.getURL === 'function'
  ) {
    try {
      const worker = new Worker(new URL('./memoryWorker.ts', import.meta.url), {
        type: 'module',
        name: 'yours-memory-worker',
      });
      return new WorkerBackedMemoryClient(worker);
    } catch {
      // Fallback if Dedicated Worker instantiation fails in restricted contexts.
    }
  }
  return new LocalEngineMemoryClient();
}

let sharedClient: ExtensionMemoryClient | null = null;

export function getExtensionMemoryClient(): ExtensionMemoryClient {
  if (!sharedClient) {
    sharedClient = createExtensionMemoryClient();
  }
  return sharedClient;
}

export { SensitiveDomainBlockedError };
