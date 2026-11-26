import type { GpuPacingMode, WgslComputeTier, WgslKvMode } from '../shared/protocol.ts';
import {
  calibratePacingForHardware,
  computeDecodeBurstBreatherMs,
  resolveExtensionAssetUrl,
  sleepMs,
} from './gpuPacing.ts';
import {
  GEMMA4_PLE_ROW_BYTES,
  Gemma4WgslEngine,
  normalizeGemma4PleTokenId,
  type Gemma4CpuSnapshot,
} from './gemma4WgslEngine.ts';
import type { OpfsModelCache } from './opfsCache.ts';
import type { GpuPrefixSlotBackend } from './prefixCacheManager.ts';
import { sampleNextToken, type SamplingOptions } from './sampling.ts';

export interface WgslRuntimeLoadOptions {
  assetBaseUrl?: string;
  bundleBaseUrl?: string | null;
  modelId?: string;
  kvMode?: WgslKvMode;
  maxContext?: number;
  pacingMode?: GpuPacingMode;
  forceComputeTier?: WgslComputeTier;
  deviceMemoryGb?: number;
  onProgress?: (phase: string, pct: number) => void;
}

export interface WgslRuntimeLoadStats {
  computeTier: WgslComputeTier;
  loadSource: 'bundled' | 'opfs' | 'network';
  opfsReadMs: number;
  repackMs: number;
  pipelineCompileMs: number;
  allocatedKvCapacity: number;
  maxSafeKvCapacity: number;
  usingDp4a: boolean;
}

export interface WgslRuntimeEngine extends GpuPrefixSlotBackend {
  readonly activeSeqPos: number;
  readonly maxContextTokens: number;
  load(opfsCache: OpfsModelCache | null, options: WgslRuntimeLoadOptions): Promise<WgslRuntimeLoadStats>;
  getComputeTier(): WgslComputeTier;
  getAllocatedKvCapacity(): number;
  getMaxSafeKvCapacity(): number;
  getMemoryStats(): { estVramMb: number; kvCacheMb: number; deltaNetStateMb: number; weightMb: number };
  setDisplayHz(hz: number): void;
  resetDecodeBurst(): void;
  resetState(): Promise<void>;
  snapshotState(): Promise<{ seqPos: number }>;
  restoreState(snap: { seqPos: number }): Promise<void>;
  prefill(
    tokenIds: number[],
    options: {
      chunkSize: number;
      sampling: SamplingOptions;
      pacingMode?: GpuPacingMode;
      yieldFn: () => Promise<void>;
      abortSignal?: AbortSignal;
      onChunk?: (done: number, total: number) => void;
      skipFinalLogits?: boolean;
    },
  ): Promise<{ nextTokenId: number; prefillMs: number }>;
  decodeStep(
    tokenId: number,
    sampling: SamplingOptions,
    pacingMode?: GpuPacingMode,
  ): Promise<{ nextToken: number; gpuMs: number; breatherMs: number }>;
  decodeSpeculativeStep(
    currentTokenId: number,
    draftTokens: number[],
    sampling: SamplingOptions,
    pacingMode?: GpuPacingMode,
  ): Promise<{ tokens: number[]; acceptedCount: number; draftedCount: number; gpuMs: number; breatherMs: number }>;
  dispose(): void;
}

const PLE_TOKENS_PER_SHARD = 1024;
const PLE_SHARD_CACHE_MAX = 16;

export function mapGemma4PleShard(tokenId: number): {
  shardIndex: number;
  localIndex: number;
  fileName: string;
  byteOffset: number;
} {
  const safeId = normalizeGemma4PleTokenId(tokenId);
  const shardIndex = safeId >>> 10;
  const localIndex = safeId & (PLE_TOKENS_PER_SHARD - 1);
  return {
    shardIndex,
    localIndex,
    fileName: `ple_shards/ple_${String(shardIndex).padStart(3, '0')}.bin`,
    byteOffset: localIndex * GEMMA4_PLE_ROW_BYTES,
  };
}

export function createExtensionPleFetcher(baseUrl: string): (tokenIds: readonly number[]) => Promise<Uint8Array> {
  const normBaseRaw = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const shardCache = new Map<number, Uint8Array>();
  const inflight = new Map<number, Promise<Uint8Array>>();

  const loadShard = (shardIndex: number): Promise<Uint8Array> => {
    const cached = shardCache.get(shardIndex);
    if (cached) {
      shardCache.delete(shardIndex);
      shardCache.set(shardIndex, cached);
      return Promise.resolve(cached);
    }
    const existing = inflight.get(shardIndex);
    if (existing) {
      return existing;
    }
    const fileName = `ple_shards/ple_${String(shardIndex).padStart(3, '0')}.bin`;
    const url = new URL(fileName, normBaseRaw).href;
    const p = fetch(url)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch Gemma 4 PLE shard ${url}: ${res.status}`);
        }
        const bytes = new Uint8Array(await res.arrayBuffer());
        if (shardCache.size >= PLE_SHARD_CACHE_MAX) {
          const oldest = shardCache.keys().next().value;
          if (oldest !== undefined) {
            shardCache.delete(oldest);
          }
        }
        shardCache.set(shardIndex, bytes);
        return bytes;
      })
      .finally(() => {
        inflight.delete(shardIndex);
      });
    inflight.set(shardIndex, p);
    return p;
  };

  return async (tokenIds: readonly number[]): Promise<Uint8Array> => {
    const mappings = tokenIds.map((id) => mapGemma4PleShard(id));
    const uniqueShards = Array.from(new Set(mappings.map((m) => m.shardIndex)));
    const loadedShards = await Promise.all(uniqueShards.map((idx) => loadShard(idx)));
    const shardById = new Map<number, Uint8Array>();
    for (let i = 0; i < uniqueShards.length; i++) {
      shardById.set(uniqueShards[i]!, loadedShards[i]!);
    }
    const out = new Uint8Array(mappings.length * GEMMA4_PLE_ROW_BYTES);
    for (let i = 0; i < mappings.length; i++) {
      const m = mappings[i]!;
      const shard = shardById.get(m.shardIndex)!;
      out.set(
        shard.subarray(m.byteOffset, m.byteOffset + GEMMA4_PLE_ROW_BYTES),
        i * GEMMA4_PLE_ROW_BYTES,
      );
    }
    return out;
  };
}

export class Gemma4ExtensionAdapter implements WgslRuntimeEngine {
  private readonly engine = new Gemma4WgslEngine();
  private displayHz = 60;
  private decodeBurstMs = 0;

  get activeSeqPos(): number {
    return this.engine.getSeqPos();
  }

  get maxContextTokens(): number {
    return this.engine.getMaxContext();
  }

  async load(
    _opfsCache: OpfsModelCache | null,
    options: WgslRuntimeLoadOptions,
  ): Promise<WgslRuntimeLoadStats> {
    const baseUrl = resolveExtensionAssetUrl('models/gemma4-e4b', options.assetBaseUrl);
    const wasmUrl = resolveExtensionAssetUrl('wasm/tinygpt_wgsl_engine.wasm', options.assetBaseUrl);
    const pleRowFetcher = createExtensionPleFetcher(baseUrl);

    const stats = await this.engine.load({
      baseUrl,
      wasmUrl,
      kvMode: options.kvMode ?? 'int8',
      maxContext: options.maxContext ?? 131072,
      pacingMode: options.pacingMode ?? 'balanced',
      displayHz: this.displayHz,
      pleRowFetcher,
      onProgress: (stage, pct) => {
        options.onProgress?.(stage, Math.min(100, Math.round(pct * 100)));
      },
    });

    return {
      computeTier: stats.computeTier,
      loadSource: 'bundled',
      opfsReadMs: stats.streamWeightsMs,
      repackMs: 0,
      pipelineCompileMs: stats.pipelineCompileMs,
      allocatedKvCapacity: stats.maxContext,
      maxSafeKvCapacity: stats.maxContext,
      usingDp4a: stats.hasDp4a,
    };
  }

  getComputeTier(): WgslComputeTier {
    return this.engine.getComputeTier();
  }

  getAllocatedKvCapacity(): number {
    return this.engine.getMaxContext();
  }

  getMaxSafeKvCapacity(): number {
    return this.engine.getMaxContext();
  }

  getMemoryStats(): { estVramMb: number; kvCacheMb: number; deltaNetStateMb: number; weightMb: number } {
    const plan = this.engine.getVramPlan();
    if (!plan) {
      return { estVramMb: 0, kvCacheMb: 0, deltaNetStateMb: 0, weightMb: 0 };
    }
    return {
      estVramMb: plan.totalVramMb,
      kvCacheMb: plan.kvTotalMb,
      deltaNetStateMb: 0,
      weightMb: Number((plan.weightBytes / (1024 * 1024)).toFixed(2)),
    };
  }

  setDisplayHz(hz: number): void {
    this.displayHz = hz;
    this.engine.setDisplayHz(hz);
  }

  resetDecodeBurst(): void {
    this.decodeBurstMs = 0;
  }

  async resetState(): Promise<void> {
    this.decodeBurstMs = 0;
    this.engine.resetContext();
  }

  saveGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; byteSize: number }> {
    return this.engine.saveGpuPrefixSlot(slotId);
  }

  restoreGpuPrefixSlot(slotId: string): Promise<{ seqPos: number; restoreMs: number }> {
    this.decodeBurstMs = 0;
    return this.engine.restoreGpuPrefixSlot(slotId);
  }

  deleteGpuPrefixSlot(slotId: string): boolean {
    return this.engine.deleteGpuPrefixSlot(slotId);
  }

  snapshotState(): Promise<Gemma4CpuSnapshot> {
    return this.engine.snapshotState();
  }

  async restoreState(snap: { seqPos: number }): Promise<void> {
    this.decodeBurstMs = 0;
    await this.engine.restoreState(snap as Gemma4CpuSnapshot);
  }

  async prefill(
    tokenIds: number[],
    options: {
      chunkSize: number;
      sampling: SamplingOptions;
      pacingMode?: GpuPacingMode;
      yieldFn: () => Promise<void>;
      abortSignal?: AbortSignal;
      onChunk?: (done: number, total: number) => void;
      skipFinalLogits?: boolean;
    },
  ): Promise<{ nextTokenId: number; prefillMs: number }> {
    this.decodeBurstMs = 0;
    if (options.pacingMode) {
      this.engine.setPacingMode(options.pacingMode);
    }
    const res = await this.engine.prefillTokens(tokenIds, {
      chunkSize: options.chunkSize,
      abortSignal: options.abortSignal,
      onChunk: async (done, total) => {
        options.onChunk?.(done, total);
        await options.yieldFn();
      },
    });
    if (options.abortSignal?.aborted) {
      throw new Error('__ABORTED__');
    }
    let nextTokenId = res.nextTokenId;
    if (!options.skipFinalLogits && options.sampling.temperature > 0) {
      const logits = await this.engine.readLastLogits();
      nextTokenId = sampleNextToken(logits, options.sampling);
    }
    return { nextTokenId, prefillMs: res.prefillMs };
  }

  async decodeStep(
    tokenId: number,
    sampling: SamplingOptions,
    pacingMode: GpuPacingMode = 'balanced',
  ): Promise<{ nextToken: number; gpuMs: number; breatherMs: number }> {
    this.engine.setPacingMode(pacingMode);
    const step = await this.engine.decodeStep(tokenId);
    const gpuMs = this.engine.getLastDecodeGpuMs() || step.forwardMs;
    let nextToken = step.nextTokenId;
    if (sampling.temperature > 0) {
      const logits = await this.engine.readLastLogits();
      nextToken = sampleNextToken(logits, sampling);
    }
    this.decodeBurstMs += gpuMs;
    const policy = calibratePacingForHardware(pacingMode, this.displayHz, gpuMs);
    const { breatherMs, nextBurstMs } = computeDecodeBurstBreatherMs(
      pacingMode,
      0,
      this.decodeBurstMs,
      policy,
    );
    this.decodeBurstMs = nextBurstMs;
    if (breatherMs > 0) {
      await sleepMs(breatherMs);
    }
    return { nextToken, gpuMs, breatherMs };
  }

  async decodeSpeculativeStep(
    currentTokenId: number,
    draftTokens: number[],
    sampling: SamplingOptions,
    pacingMode: GpuPacingMode = 'balanced',
  ): Promise<{
    tokens: number[];
    acceptedCount: number;
    draftedCount: number;
    gpuMs: number;
    breatherMs: number;
  }> {
    this.engine.setPacingMode(pacingMode);
    const batch = [currentTokenId, ...draftTokens];
    const ver = await this.engine.verifySpeculativeTokens(batch);
    let matchCount = 0;
    if (sampling.temperature <= 0) {
      while (matchCount < draftTokens.length && ver.predictions[matchCount] === draftTokens[matchCount]) {
        matchCount += 1;
      }
    }
    const acceptedSeq = [currentTokenId, ...draftTokens.slice(0, matchCount)];
    this.engine.commitSpeculativeTokens(acceptedSeq);

    this.decodeBurstMs += ver.gpuMs;
    const policy = calibratePacingForHardware(pacingMode, this.displayHz, ver.gpuMs);
    const { breatherMs, nextBurstMs } = computeDecodeBurstBreatherMs(
      pacingMode,
      0,
      this.decodeBurstMs,
      policy,
    );
    this.decodeBurstMs = nextBurstMs;
    if (breatherMs > 0) {
      await sleepMs(breatherMs);
    }

    return {
      tokens: [...draftTokens.slice(0, matchCount), ver.predictions[matchCount]!],
      acceptedCount: matchCount,
      draftedCount: draftTokens.length,
      gpuMs: ver.gpuMs,
      breatherMs,
    };
  }

  dispose(): void {
    this.engine.dispose();
  }
}
