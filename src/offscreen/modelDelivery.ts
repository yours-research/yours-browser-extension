import type { ModelManifestV1 } from '@yours/models/contracts';
import {
  WorkStealingModelDownloader,
  downloadVerifiedModel,
  type DownloadTelemetry,
} from '@yours/models/download';
import {
  DEFAULT_PRAGMA_CATALOG,
  ModelRegistry,
  probeHardware,
  selectComputeTier,
  type HardwareProbeResult,
  type WebGpuNavigatorLike,
} from '@yours/models/hardware';
import {
  RedirectCache,
  computeManifestMerkleRoot,
  sha256Hex,
  verifySignedManifest,
  type Ed25519PublicKeyInput,
} from '@yours/models/manifest';
import {
  MemoryVault,
  OpfsAsyncVault,
  OpfsSyncVault,
  StorageQuotaManager,
  deriveModelKey,
  type ModelVaultBackend,
} from '@yours/models/vault';
import {
  BUNDLED_MODEL_IDS,
  MODEL_PRESETS,
  type HardwareProfileSnapshot,
  type ModelCatalogChoice,
  type ModelDownloadProgressInfo,
  type WgslComputeTier,
} from '../shared/protocol.ts';

export {
  DEFAULT_PRAGMA_CATALOG,
  ModelRegistry,
  RedirectCache,
  StorageQuotaManager,
  WorkStealingModelDownloader,
  computeManifestMerkleRoot,
  deriveModelKey,
  downloadVerifiedModel,
  probeHardware,
  selectComputeTier,
  sha256Hex,
  verifySignedManifest,
};

function createExtensionVault(): ModelVaultBackend {
  const hasOpfs =
    typeof navigator !== 'undefined' &&
    typeof navigator.storage?.getDirectory === 'function';
  if (!hasOpfs) {
    return new MemoryVault();
  }
  const inWorker =
    typeof window === 'undefined' &&
    typeof self !== 'undefined' &&
    typeof (self as { postMessage?: unknown }).postMessage === 'function';
  return inWorker ? new OpfsSyncVault() : new OpfsAsyncVault();
}

function inferParamsB(modelId: string, defaultB = 4): number {
  const match = modelId.match(/(\d+(?:\.\d+)?)\s*B/i);
  if (match) {
    const val = Number.parseFloat(match[1]);
    if (Number.isFinite(val) && val > 0) return val;
  }
  if (/e4b/i.test(modelId)) return 4.5;
  if (/nano/i.test(modelId)) return 1.0;
  return defaultB;
}

function normalizeTier(tier: WgslComputeTier | 'cpu-only'): WgslComputeTier {
  return tier === 'cpu-only' ? 'tier3-f32-universal' : tier;
}

export class ModelDeliveryService {
  readonly vault: ModelVaultBackend;
  readonly quotaManager: StorageQuotaManager;
  readonly registry: ModelRegistry;
  readonly redirectCache: RedirectCache;
  private activeDownloads = new Map<string, AbortController>();
  private lastHardware: HardwareProfileSnapshot | null = null;
  private lastCatalog: ModelCatalogChoice[] = [];

  constructor(options?: {
    vault?: ModelVaultBackend;
    registry?: ModelRegistry;
    redirectCache?: RedirectCache;
  }) {
    this.vault = options?.vault ?? createExtensionVault();
    this.quotaManager = new StorageQuotaManager(this.vault);
    this.registry = options?.registry ?? new ModelRegistry(DEFAULT_PRAGMA_CATALOG);
    this.redirectCache = options?.redirectCache ?? new RedirectCache();
  }

  getLastHardwareProfile(): HardwareProfileSnapshot | null {
    return this.lastHardware;
  }

  getLastCatalog(): ModelCatalogChoice[] {
    return this.lastCatalog;
  }

  async probeHardwareAndCatalog(): Promise<{
    hardware: HardwareProfileSnapshot;
    catalog: ModelCatalogChoice[];
    rawProbe: HardwareProbeResult;
  }> {
    const nav =
      typeof navigator !== 'undefined'
        ? (navigator as unknown as WebGpuNavigatorLike & { deviceMemory?: number })
        : undefined;

    const rawProbe = await probeHardware({
      navigator: nav,
      preferredFamily: 'all',
      registry: this.registry,
    });

    const maxBufferBytes =
      rawProbe.grantedLimits?.maxBufferSize ?? rawProbe.profile.maxBufferSize ?? 0;
    const maxBindingBytes =
      rawProbe.grantedLimits?.maxStorageBufferBindingSize ??
      rawProbe.profile.maxStorageBufferBindingSize ??
      0;

    const hardware: HardwareProfileSnapshot = {
      adapterName:
        rawProbe.adapterInfo?.description ||
        rawProbe.adapterInfo?.device ||
        (rawProbe.profile.hasWebGpu ? 'WebGPU Adapter' : 'CPU Fallback'),
      vendor: rawProbe.adapterInfo?.vendor || 'unknown',
      architecture: rawProbe.adapterInfo?.architecture || 'unknown',
      deviceMemoryGb:
        typeof nav?.deviceMemory === 'number'
          ? nav.deviceMemory
          : (rawProbe.estimatedSystemRamGB ?? null),
      hasF16: rawProbe.hasShaderF16,
      hasSubgroups: rawProbe.hasSubgroups,
      recommendedTier: normalizeTier(rawProbe.profile.tier),
      maxBufferSizeMb: Math.round((maxBufferBytes / (1024 * 1024)) * 10) / 10,
      maxStorageBindingMb: Math.round((maxBindingBytes / (1024 * 1024)) * 10) / 10,
      recommendedModelId: MODEL_PRESETS[0]?.id ?? rawProbe.profile.recommendedModelId,
      probedAt: Date.now(),
    };

    const storedModels = await this.quotaManager.listModels().catch(() => []);
    const storedIds = new Set(storedModels.map((m) => m.modelId));
    const catalogMap = new Map<string, ModelCatalogChoice>();

    // Only expose the offline bundled Pragma-1-4B RC1 preset
    for (const preset of MODEL_PRESETS) {
      const isBundled = (BUNDLED_MODEL_IDS as readonly string[]).includes(preset.id);
      const cached = isBundled || storedIds.has(preset.id);
      catalogMap.set(preset.id, {
        id: preset.id,
        label: preset.label,
        family: preset.id.toLowerCase().includes('pragma') ? 'Pragma' : 'Qwen3.5',
        paramsB: inferParamsB(preset.id, 4),
        estimatedSizeGb: preset.estimatedWeightGB[preset.defaultDtype],
        contextLength: 131072,
        source: isBundled ? 'bundled' : 'custom',
        cachedInOpfs: cached,
        bundled: isBundled,
        manifestVerified: isBundled || storedIds.has(preset.id),
        notes: preset.notes,
      });
    }

    const catalog = Array.from(catalogMap.values());
    this.lastHardware = hardware;
    this.lastCatalog = catalog;
    return { hardware, catalog, rawProbe };
  }

  async downloadVerifiedModelFromManifest(options: {
    modelId: string;
    manifestUrl?: string;
    manifest?: ModelManifestV1;
    pinnedPublicKeys?: readonly Ed25519PublicKeyInput[];
    onProgress?: (progress: ModelDownloadProgressInfo) => void;
    fetchImpl?: typeof fetch;
  }): Promise<ModelCatalogChoice[]> {
    const { modelId, onProgress } = options;
    const isBundled = (BUNDLED_MODEL_IDS as readonly string[]).includes(modelId);
    if (isBundled) {
      onProgress?.({
        modelId,
        phase: 'complete',
        bytesDownloaded: 1,
        totalBytes: 1,
        pct: 100,
        mbPerSec: 0,
      });
      const { catalog } = await this.probeHardwareAndCatalog();
      return catalog;
    }

    const existingAbort = this.activeDownloads.get(modelId);
    if (existingAbort) {
      existingAbort.abort();
    }
    const abortController = new AbortController();
    this.activeDownloads.set(modelId, abortController);

    try {
      onProgress?.({
        modelId,
        phase: 'probing',
        bytesDownloaded: 0,
        totalBytes: 0,
        pct: 2,
        mbPerSec: 0,
      });

      const entry = this.registry
        .list()
        .find((item) => item.modelId === modelId || item.id === modelId);
      const manifestUrl = options.manifestUrl ?? entry?.manifestUrl;
      const fetcher = options.fetchImpl ?? globalThis.fetch.bind(globalThis);

      let manifest = options.manifest;
      if (!manifest) {
        if (!manifestUrl) {
          throw new Error(
            `No signed manifest URL available for "${modelId}". Select a bundled offline model or supply a verified manifest URL.`,
          );
        }
        onProgress?.({
          modelId,
          phase: 'verifying-manifest',
          bytesDownloaded: 0,
          totalBytes: 0,
          pct: 5,
          mbPerSec: 0,
        });
        const resp = await fetcher(manifestUrl, { signal: abortController.signal });
        if (!resp.ok) {
          throw new Error(`Failed to fetch manifest (${resp.status} ${resp.statusText}).`);
        }
        manifest = (await resp.json()) as ModelManifestV1;
      }

      if (!options.pinnedPublicKeys || options.pinnedPublicKeys.length === 0) {
        throw new Error(
          'Missing pinned Ed25519 verification key for remote model manifest.',
        );
      }

      const verifyRes = await verifySignedManifest(manifest, {
        pinnedPublicKeys: options.pinnedPublicKeys,
        expectedModelId: manifest.modelId,
        verifyMerkleRoot: true,
      });
      if (!verifyRes.valid) {
        throw new Error(verifyRes.reason ?? 'Model manifest Ed25519/Merkle verification failed.');
      }

      const quota = await this.quotaManager.ensureSpaceForModel(manifest.totalBytes, {
        safetyHeadroomBytes: 64 * 1024 * 1024,
      });
      if (!quota.ok) {
        throw new Error(quota.reason ?? 'Insufficient OPFS storage space for verified model.');
      }

      const downloader = new WorkStealingModelDownloader({
        vault: this.vault,
        pinnedPublicKeys: options.pinnedPublicKeys,
        redirectCache: this.redirectCache,
        fetchImpl: fetcher,
        signal: abortController.signal,
        onProgress: (dl: DownloadTelemetry) => {
          const pct =
            dl.totalBytes > 0
              ? Math.min(99, Math.round((dl.bytesDownloaded / dl.totalBytes) * 100))
              : 10;
          const phase: ModelDownloadProgressInfo['phase'] =
            dl.phase === 'verifying-manifest'
              ? 'verifying-manifest'
              : dl.phase === 'finalizing'
                ? 'writing-opfs'
                : dl.phase === 'complete'
                  ? 'complete'
                  : 'downloading';
          onProgress?.({
            modelId,
            phase,
            bytesDownloaded: dl.bytesDownloaded,
            totalBytes: dl.totalBytes,
            pct,
            mbPerSec: dl.speedMiBs,
          });
        },
      });

      await downloader.download(manifest);
      onProgress?.({
        modelId,
        phase: 'complete',
        bytesDownloaded: manifest.totalBytes,
        totalBytes: manifest.totalBytes,
        pct: 100,
        mbPerSec: 0,
      });

      const { catalog } = await this.probeHardwareAndCatalog();
      return catalog;
    } finally {
      if (this.activeDownloads.get(modelId) === abortController) {
        this.activeDownloads.delete(modelId);
      }
    }
  }

  cancelDownload(modelId?: string): void {
    if (modelId) {
      const controller = this.activeDownloads.get(modelId);
      if (controller) {
        controller.abort();
        this.activeDownloads.delete(modelId);
      }
      return;
    }
    for (const controller of this.activeDownloads.values()) {
      controller.abort();
    }
    this.activeDownloads.clear();
  }
}
