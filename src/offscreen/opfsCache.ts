import type { OpfsFileSummary, OpfsSpikeResult, OpfsStatusInfo } from '../shared/protocol.ts';

const MODEL_CACHE_DIR = 'tinygpt-model-cache';
const SPIKE_DIR = 'tinygpt-phase0-spike';
const SPIKE_MANIFEST = 'spike-manifest.json';

interface CachedFileMetadata {
  key: string;
  contentType: string;
  contentLength: number;
  cachedAt: string;
  complete: boolean;
}

interface SpikeManifest {
  totalBytes: number;
  shardCount: number;
  shardBytes: number;
  createdAt: string;
}

export interface OpfsCacheOptions {
  getRootDir?: () => Promise<FileSystemDirectoryHandle>;
}

interface IterableDirectoryHandle extends FileSystemDirectoryHandle {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
}

function normalizeCacheKey(request: string | URL | Request): string {
  if (typeof request === 'string') {
    return request;
  }
  if (request instanceof URL) {
    return request.href;
  }
  return request.url;
}

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  if (typeof globalThis.crypto?.subtle?.digest === 'function') {
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
  // Deterministic fallback hash only if subtle crypto is absent.
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < bytes.length; i++) {
    h1 ^= bytes[i];
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= bytes[i] + ((h2 << 6) + (h2 >>> 2));
    h2 = h2 >>> 0;
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

async function getOptionalFileHandle(
  dir: FileSystemDirectoryHandle,
  name: string,
): Promise<FileSystemFileHandle | null> {
  try {
    return await dir.getFileHandle(name);
  } catch {
    return null;
  }
}

async function safeRemoveEntry(dir: FileSystemDirectoryHandle, name: string): Promise<void> {
  try {
    await dir.removeEntry(name, { recursive: true });
  } catch {
    // Ignore missing entry errors during cleanup.
  }
}

async function readMetadata(
  dir: FileSystemDirectoryHandle,
  metaName: string,
): Promise<CachedFileMetadata | null> {
  const handle = await getOptionalFileHandle(dir, metaName);
  if (!handle) {
    return null;
  }
  try {
    const file = await handle.getFile();
    const parsed = JSON.parse(await file.text()) as Partial<CachedFileMetadata>;
    if (
      typeof parsed.key === 'string' &&
      typeof parsed.contentLength === 'number' &&
      parsed.complete === true
    ) {
      return {
        key: parsed.key,
        contentType: parsed.contentType ?? 'application/octet-stream',
        contentLength: parsed.contentLength,
        cachedAt: parsed.cachedAt ?? '',
        complete: true,
      };
    }
    return null;
  } catch {
    return null;
  }
}

function buildDeterministicShardChunk(
  shardIndex: number,
  chunkOffset: number,
  length: number,
): Uint8Array {
  const buf = new Uint8Array(length);
  const seed = (shardIndex * 31 + 17) & 0xff;
  // Write verifiable header bytes at offset 0 and stride markers for fast multi-GB verification.
  for (let i = 0; i < length; i += 4096) {
    buf[i] = seed;
    if (i + 1 < length) buf[i + 1] = ((chunkOffset + i) >>> 12) & 0xff;
    if (i + 2 < length) buf[i + 2] = ((chunkOffset + i) >>> 20) & 0xff;
    if (i + 3 < length) buf[i + 3] = 0xa5;
  }
  if (length > 0) {
    buf[length - 1] = (seed ^ 0x5a) & 0xff;
  }
  return buf;
}

export class OpfsModelCache {
  private readonly getRootDirFn: () => Promise<FileSystemDirectoryHandle>;
  private hitCount = 0;
  private missCount = 0;
  private putCount = 0;
  private matchReadMs = 0;

  constructor(options: OpfsCacheOptions = {}) {
    this.getRootDirFn =
      options.getRootDir ??
      (async () => {
        if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
          throw new Error('OPFS (navigator.storage.getDirectory) is not supported.');
        }
        return navigator.storage.getDirectory();
      });
  }

  getCacheCounters(): { hits: number; misses: number; puts: number; matchReadMs: number } {
    return {
      hits: this.hitCount,
      misses: this.missCount,
      puts: this.putCount,
      matchReadMs: this.matchReadMs,
    };
  }

  private async getCacheDir(): Promise<FileSystemDirectoryHandle> {
    const root = await this.getRootDirFn();
    return root.getDirectoryHandle(MODEL_CACHE_DIR, { create: true });
  }

  async getRootDirectory(): Promise<FileSystemDirectoryHandle> {
    return this.getRootDirFn();
  }

  private async getSpikeDir(create: boolean): Promise<FileSystemDirectoryHandle> {
    const root = await this.getRootDirFn();
    return root.getDirectoryHandle(SPIKE_DIR, { create });
  }

  async requestPersistence(): Promise<boolean | null> {
    try {
      if (typeof navigator !== 'undefined' && typeof navigator.storage?.persist === 'function') {
        return await navigator.storage.persist();
      }
      return null;
    } catch {
      return null;
    }
  }

  private async resolveCachedEntry(
    request: string | URL | Request,
  ): Promise<{ meta: CachedFileMetadata; file: File } | null> {
    const key = normalizeCacheKey(request);
    const hash = await sha256Hex(key);
    const cacheDir = await this.getCacheDir();
    const meta = await readMetadata(cacheDir, `${hash}.meta.json`);
    if (!meta || meta.key !== key) {
      return null;
    }
    const binHandle = await getOptionalFileHandle(cacheDir, `${hash}.bin`);
    if (!binHandle) {
      return null;
    }
    const file = await binHandle.getFile();
    if (file.size !== meta.contentLength) {
      return null;
    }
    return { meta, file };
  }

  /**
   * Returns the raw OPFS File handle without copying bytes into JS memory.
   */
  async getCachedFile(request: string | URL | Request): Promise<File | null> {
    const entry = await this.resolveCachedEntry(request);
    if (!entry) {
      return null;
    }
    this.hitCount += 1;
    return entry.file;
  }

  /**
   * Ensures `url` is stored cleanly in OPFS and returns its raw File handle.
   */
  async ensureCachedFile(
    url: string,
    onProgress?: (data: { loaded: number; total: number; progress: number }) => void,
  ): Promise<File> {
    const existing = await this.getCachedFile(url);
    if (existing) {
      return existing;
    }
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }
    await this.put(url, response, onProgress);
    const stored = await this.getCachedFile(url);
    if (!stored) {
      throw new Error(`Cached OPFS entry missing right after download: ${url}`);
    }
    return stored;
  }

  /**
   * Transformers.js customCache.match implementation backed by OPFS.
   * Rejects incomplete or truncated entries so partial downloads never poison the cache.
   */
  async match(request: string | URL | Request): Promise<Response | undefined> {
    const startMs = performance.now();
    const entry = await this.resolveCachedEntry(request);
    if (!entry) {
      this.missCount += 1;
      return undefined;
    }

    this.hitCount += 1;
    this.matchReadMs += performance.now() - startMs;
    const headers = new Headers();
    headers.set('content-type', entry.meta.contentType);
    headers.set('content-length', String(entry.file.size));

    return new Response(entry.file, {
      status: 200,
      headers,
    });
  }

  /**
   * Transformers.js customCache.put implementation that streams directly to OPFS
   * without duplicating multi-GB weights in JS memory.
   */
  async put(
    request: string | URL | Request,
    response: Response,
    onProgress?: (data: { loaded: number; total: number; progress: number }) => void,
  ): Promise<void> {
    const key = normalizeCacheKey(request);
    const hash = await sha256Hex(key);
    const cacheDir = await this.getCacheDir();
    const metaName = `${hash}.meta.json`;
    const binName = `${hash}.bin`;

    // Remove any stale metadata first so a failure never leaves a stale valid marker.
    await safeRemoveEntry(cacheDir, metaName);

    const binHandle = await cacheDir.getFileHandle(binName, { create: true });
    const writable = await binHandle.createWritable();

    const expectedTotalHeader = response.headers.get('content-length');
    const expectedTotal = expectedTotalHeader ? Number(expectedTotalHeader) : 0;
    let totalWritten = 0;

    try {
      if (response.body) {
        const reader = response.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            if (value && value.byteLength > 0) {
              await writable.write(value as unknown as FileSystemWriteChunkType);
              totalWritten += value.byteLength;
              if (onProgress) {
                const denom = expectedTotal > 0 ? expectedTotal : totalWritten;
                onProgress({
                  loaded: totalWritten,
                  total: denom,
                  progress: denom > 0 ? (totalWritten / denom) * 100 : 100,
                });
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      } else {
        const buf = new Uint8Array(await response.arrayBuffer());
        if (buf.byteLength > 0) {
          await writable.write(buf as unknown as FileSystemWriteChunkType);
        }
        totalWritten = buf.byteLength;
        if (onProgress) {
          onProgress({ loaded: totalWritten, total: totalWritten, progress: 100 });
        }
      }

      await writable.close();

      const meta: CachedFileMetadata = {
        key,
        contentType: response.headers.get('content-type') ?? 'application/octet-stream',
        contentLength: totalWritten,
        cachedAt: new Date().toISOString(),
        complete: true,
      };
      const metaHandle = await cacheDir.getFileHandle(metaName, { create: true });
      const metaWritable = await metaHandle.createWritable();
      await metaWritable.write(JSON.stringify(meta));
      await metaWritable.close();
      this.putCount += 1;
    } catch (err) {
      try {
        await writable.abort();
      } catch {
        // Ignore abort failures.
      }
      await safeRemoveEntry(cacheDir, metaName);
      await safeRemoveEntry(cacheDir, binName);
      throw err;
    }
  }

  async getOpfsStatus(): Promise<OpfsStatusInfo> {
    try {
      const root = await this.getRootDirFn();
      const persisted =
        typeof navigator !== 'undefined' && typeof navigator.storage?.persisted === 'function'
          ? await navigator.storage.persisted()
          : null;
      const estimate =
        typeof navigator !== 'undefined' && typeof navigator.storage?.estimate === 'function'
          ? await navigator.storage.estimate()
          : null;

      let modelFilesCount = 0;
      let modelCacheBytes = 0;
      try {
        const cacheDir = (await root.getDirectoryHandle(MODEL_CACHE_DIR, {
          create: false,
        })) as IterableDirectoryHandle;
        for await (const [name, handle] of cacheDir.entries()) {
          if (handle.kind === 'file' && name.endsWith('.bin')) {
            const file = await (handle as FileSystemFileHandle).getFile();
            modelFilesCount += 1;
            modelCacheBytes += file.size;
          }
        }
      } catch {
        // Cache dir does not exist yet.
      }

      const spikeFiles: OpfsFileSummary[] = [];
      try {
        const spikeDir = (await root.getDirectoryHandle(SPIKE_DIR, {
          create: false,
        })) as IterableDirectoryHandle;
        for await (const [name, handle] of spikeDir.entries()) {
          if (handle.kind === 'file') {
            const file = await (handle as FileSystemFileHandle).getFile();
            spikeFiles.push({ name, sizeBytes: file.size });
          }
        }
        spikeFiles.sort((a, b) => a.name.localeCompare(b.name));
      } catch {
        // Spike dir does not exist yet.
      }

      return {
        supported: true,
        persisted,
        quotaBytes: estimate?.quota ?? null,
        usageBytes: estimate?.usage ?? null,
        modelFilesCount,
        modelCacheBytes,
        spikeFiles,
        error: null,
      };
    } catch (err) {
      return {
        supported: false,
        persisted: null,
        quotaBytes: null,
        usageBytes: null,
        modelFilesCount: 0,
        modelCacheBytes: 0,
        spikeFiles: [],
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  async clearModelCache(): Promise<void> {
    const root = await this.getRootDirFn();
    await safeRemoveEntry(root, MODEL_CACHE_DIR);
  }

  /**
   * Phase 0 OPFS persistence spike: writes sharded binary files totaling `totalMiB`
   * (2560 MiB = 2.5 GiB by default) using streaming 4 MiB chunks to avoid JS RAM spikes.
   */
  async runOpfsSpike(
    totalMiB: number,
    shardMiB: number,
    onProgress?: (msg: string, pct: number) => void,
  ): Promise<OpfsSpikeResult> {
    const start = performance.now();
    await this.requestPersistence();
    const root = await this.getRootDirFn();
    await safeRemoveEntry(root, SPIKE_DIR);
    const spikeDir = await this.getSpikeDir(true);

    const totalBytes = Math.max(1, Math.round(totalMiB)) * 1024 * 1024;
    const shardBytes = Math.max(1, Math.round(shardMiB)) * 1024 * 1024;
    const shardCount = Math.ceil(totalBytes / shardBytes);
    const chunkBytes = Math.min(4 * 1024 * 1024, shardBytes);

    let writtenTotal = 0;

    for (let shardIdx = 0; shardIdx < shardCount; shardIdx++) {
      const thisShardBytes =
        shardIdx === shardCount - 1 ? totalBytes - shardIdx * shardBytes : shardBytes;
      const shardName = `spike-shard-${String(shardIdx).padStart(2, '0')}.bin`;
      const fileHandle = await spikeDir.getFileHandle(shardName, { create: true });
      const writable = await fileHandle.createWritable();

      try {
        let shardOffset = 0;
        while (shardOffset < thisShardBytes) {
          const writeLen = Math.min(chunkBytes, thisShardBytes - shardOffset);
          const chunk = buildDeterministicShardChunk(shardIdx, shardOffset, writeLen);
          await writable.write(chunk as unknown as FileSystemWriteChunkType);
          shardOffset += writeLen;
          writtenTotal += writeLen;
          const pct = Math.min(100, (writtenTotal / totalBytes) * 100);
          onProgress?.(
            `Writing OPFS shard ${shardIdx + 1}/${shardCount} (${(writtenTotal / (1024 * 1024)).toFixed(0)}/${totalMiB} MiB)`,
            pct,
          );
        }
        await writable.close();
      } catch (err) {
        try {
          await writable.abort();
        } catch {
          // Ignore abort failure.
        }
        throw err;
      }
    }

    const manifest: SpikeManifest = {
      totalBytes,
      shardCount,
      shardBytes,
      createdAt: new Date().toISOString(),
    };
    const manifestHandle = await spikeDir.getFileHandle(SPIKE_MANIFEST, { create: true });
    const manifestWritable = await manifestHandle.createWritable();
    await manifestWritable.write(JSON.stringify(manifest));
    await manifestWritable.close();

    const durationMs = Math.max(1, performance.now() - start);
    const throughputMBs = totalBytes / (1024 * 1024) / (durationMs / 1000);

    return {
      action: 'write',
      ok: true,
      totalBytes,
      shardCount,
      durationMs,
      throughputMBs,
      details: `Wrote ${shardCount} shards (${(totalBytes / (1024 * 1024 * 1024)).toFixed(2)} GiB) to OPFS in ${(durationMs / 1000).toFixed(2)}s.`,
    };
  }

  /**
   * Verifies that all Phase 0 spike shards still exist in OPFS with matching byte sizes
   * and deterministic sentinel headers (safe to call after closing and relaunching Chrome).
   */
  async verifyOpfsSpike(
    onProgress?: (msg: string, pct: number) => void,
  ): Promise<OpfsSpikeResult> {
    const start = performance.now();
    let spikeDir: FileSystemDirectoryHandle;
    try {
      spikeDir = await this.getSpikeDir(false);
    } catch {
      return {
        action: 'verify',
        ok: false,
        totalBytes: 0,
        shardCount: 0,
        durationMs: performance.now() - start,
        throughputMBs: 0,
        details: 'No OPFS spike directory found. Run the OPFS Persistence Spike first.',
      };
    }

    const manifestHandle = await getOptionalFileHandle(spikeDir, SPIKE_MANIFEST);
    if (!manifestHandle) {
      return {
        action: 'verify',
        ok: false,
        totalBytes: 0,
        shardCount: 0,
        durationMs: performance.now() - start,
        throughputMBs: 0,
        details: 'OPFS spike manifest missing.',
      };
    }

    const manifest = JSON.parse(await (await manifestHandle.getFile()).text()) as SpikeManifest;
    let verifiedBytes = 0;

    for (let shardIdx = 0; shardIdx < manifest.shardCount; shardIdx++) {
      const expectedBytes =
        shardIdx === manifest.shardCount - 1
          ? manifest.totalBytes - shardIdx * manifest.shardBytes
          : manifest.shardBytes;
      const shardName = `spike-shard-${String(shardIdx).padStart(2, '0')}.bin`;
      const shardHandle = await getOptionalFileHandle(spikeDir, shardName);
      if (!shardHandle) {
        return {
          action: 'verify',
          ok: false,
          totalBytes: verifiedBytes,
          shardCount: shardIdx,
          durationMs: performance.now() - start,
          throughputMBs: 0,
          details: `Missing OPFS spike shard: ${shardName}`,
        };
      }

      const file = await shardHandle.getFile();
      if (file.size !== expectedBytes) {
        return {
          action: 'verify',
          ok: false,
          totalBytes: verifiedBytes,
          shardCount: shardIdx,
          durationMs: performance.now() - start,
          throughputMBs: 0,
          details: `Size mismatch on ${shardName}: expected ${expectedBytes} bytes, got ${file.size}.`,
        };
      }

      const headBytes = new Uint8Array(await file.slice(0, Math.min(4, file.size)).arrayBuffer());
      const expectedSeed = (shardIdx * 31 + 17) & 0xff;
      if (headBytes[0] !== expectedSeed || (file.size >= 4 && headBytes[3] !== 0xa5)) {
        return {
          action: 'verify',
          ok: false,
          totalBytes: verifiedBytes,
          shardCount: shardIdx,
          durationMs: performance.now() - start,
          throughputMBs: 0,
          details: `Corrupt sentinel header in ${shardName}.`,
        };
      }

      const tailBytes = new Uint8Array(
        await file.slice(Math.max(0, file.size - 1), file.size).arrayBuffer(),
      );
      if (tailBytes[0] !== ((expectedSeed ^ 0x5a) & 0xff)) {
        return {
          action: 'verify',
          ok: false,
          totalBytes: verifiedBytes,
          shardCount: shardIdx,
          durationMs: performance.now() - start,
          throughputMBs: 0,
          details: `Corrupt sentinel tail byte in ${shardName}.`,
        };
      }

      verifiedBytes += file.size;
      onProgress?.(
        `Verified OPFS shard ${shardIdx + 1}/${manifest.shardCount}`,
        ((shardIdx + 1) / manifest.shardCount) * 100,
      );
    }

    const durationMs = Math.max(1, performance.now() - start);
    return {
      action: 'verify',
      ok: verifiedBytes === manifest.totalBytes,
      totalBytes: verifiedBytes,
      shardCount: manifest.shardCount,
      durationMs,
      throughputMBs: verifiedBytes / (1024 * 1024) / (durationMs / 1000),
      details: `Verified ${manifest.shardCount} persisted shards (${(verifiedBytes / (1024 * 1024 * 1024)).toFixed(2)} GiB) created ${manifest.createdAt}.`,
    };
  }

  async clearOpfsSpike(): Promise<OpfsSpikeResult> {
    const start = performance.now();
    const root = await this.getRootDirFn();
    await safeRemoveEntry(root, SPIKE_DIR);
    return {
      action: 'clear',
      ok: true,
      totalBytes: 0,
      shardCount: 0,
      durationMs: performance.now() - start,
      throughputMBs: 0,
      details: 'Cleared OPFS Phase 0 spike directory.',
    };
  }
}
