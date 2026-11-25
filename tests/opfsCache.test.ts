import { describe, expect, it } from 'vitest';
import { OpfsModelCache } from '../src/offscreen/opfsCache.ts';

class MemoryWritableStream {
  private chunks: Uint8Array[] = [];
  private closed = false;
  private aborted = false;

  constructor(private readonly onCommit: (data: Uint8Array) => void) {}

  async write(data: string | Uint8Array | ArrayBufferView): Promise<void> {
    if (this.closed || this.aborted) {
      throw new Error('Writable already closed');
    }
    if (typeof data === 'string') {
      this.chunks.push(new TextEncoder().encode(data));
      return;
    }
    if (data instanceof Uint8Array) {
      this.chunks.push(new Uint8Array(data));
      return;
    }
    this.chunks.push(new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice());
  }

  async close(): Promise<void> {
    if (this.aborted) return;
    this.closed = true;
    const total = this.chunks.reduce((acc, c) => acc + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    this.onCommit(out);
  }

  async abort(): Promise<void> {
    this.aborted = true;
    this.chunks = [];
  }
}

class MemoryFileHandle {
  kind = 'file' as const;
  private bytes: Uint8Array = new Uint8Array(0);

  constructor(public readonly name: string) {}

  async getFile(): Promise<File> {
    return new File([new Uint8Array(this.bytes)], this.name, {
      type: 'application/octet-stream',
    });
  }

  async createWritable(): Promise<FileSystemWritableFileStream> {
    return new MemoryWritableStream((data) => {
      this.bytes = data;
    }) as unknown as FileSystemWritableFileStream;
  }

  setRawBytesForTest(data: Uint8Array): void {
    this.bytes = data;
  }
}

class MemoryDirectoryHandle {
  kind = 'directory' as const;
  private readonly subdirs = new Map<string, MemoryDirectoryHandle>();
  private readonly files = new Map<string, MemoryFileHandle>();

  constructor(public readonly name: string) {}

  async getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<FileSystemDirectoryHandle> {
    const existing = this.subdirs.get(name);
    if (existing) {
      return existing as unknown as FileSystemDirectoryHandle;
    }
    if (!options?.create) {
      throw new Error(`Directory not found: ${name}`);
    }
    const created = new MemoryDirectoryHandle(name);
    this.subdirs.set(name, created);
    return created as unknown as FileSystemDirectoryHandle;
  }

  async getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<FileSystemFileHandle> {
    const existing = this.files.get(name);
    if (existing) {
      return existing as unknown as FileSystemFileHandle;
    }
    if (!options?.create) {
      throw new Error(`File not found: ${name}`);
    }
    const created = new MemoryFileHandle(name);
    this.files.set(name, created);
    return created as unknown as FileSystemFileHandle;
  }

  async removeEntry(name: string): Promise<void> {
    this.files.delete(name);
    this.subdirs.delete(name);
  }

  async *entries(): AsyncIterableIterator<[string, FileSystemHandle]> {
    for (const [k, v] of this.files.entries()) {
      yield [k, v as unknown as FileSystemHandle];
    }
    for (const [k, v] of this.subdirs.entries()) {
      yield [k, v as unknown as FileSystemHandle];
    }
  }
}

describe('OpfsModelCache', () => {
  it('streams response bytes to OPFS and hits on subsequent match()', async () => {
    const root = new MemoryDirectoryHandle('root');
    const cache = new OpfsModelCache({
      getRootDir: async () => root as unknown as FileSystemDirectoryHandle,
    });

    const url = 'https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT/resolve/main/test.onnx';
    const payload = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(payload.slice(0, 3));
        controller.enqueue(payload.slice(3));
        controller.close();
      },
    });

    await cache.put(
      url,
      new Response(stream, {
        status: 200,
        headers: {
          'content-type': 'application/octet-stream',
          'content-length': String(payload.byteLength),
        },
      }),
    );

    const hit = await cache.match(url);
    expect(hit).toBeDefined();
    expect(hit?.headers.get('content-length')).toBe(String(payload.byteLength));
    const readBack = new Uint8Array(await hit!.arrayBuffer());
    expect(Array.from(readBack)).toEqual(Array.from(payload));

    const status = await cache.getOpfsStatus();
    expect(status.modelFilesCount).toBe(1);
    expect(status.modelCacheBytes).toBe(payload.byteLength);
  });

  it('rejects partial/incomplete stream writes and cleans up stale cache entries', async () => {
    const root = new MemoryDirectoryHandle('root');
    const cache = new OpfsModelCache({
      getRootDir: async () => root as unknown as FileSystemDirectoryHandle,
    });

    const url = 'https://huggingface.co/test/broken.onnx';
    const brokenStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([9, 9, 9]));
        controller.error(new Error('network reset'));
      },
    });

    await expect(
      cache.put(url, new Response(brokenStream, { status: 200 })),
    ).rejects.toThrow('network reset');

    expect(await cache.match(url)).toBeUndefined();
    const status = await cache.getOpfsStatus();
    expect(status.modelFilesCount).toBe(0);
  });

  it('rejects cache entry if stored binary size mismatches metadata', async () => {
    const root = new MemoryDirectoryHandle('root');
    const cache = new OpfsModelCache({
      getRootDir: async () => root as unknown as FileSystemDirectoryHandle,
    });

    const url = 'https://huggingface.co/test/truncated.onnx';
    await cache.put(
      url,
      new Response(new Uint8Array([10, 20, 30, 40]), {
        status: 200,
        headers: { 'content-length': '4' },
      }),
    );

    // Truncate the .bin file directly inside the mock directory.
    const cacheDir = (await root.getDirectoryHandle(
      'tinygpt-model-cache',
    )) as unknown as MemoryDirectoryHandle;
    for await (const [name, handle] of cacheDir.entries()) {
      if (name.endsWith('.bin')) {
        (handle as unknown as MemoryFileHandle).setRawBytesForTest(new Uint8Array([10]));
      }
    }

    expect(await cache.match(url)).toBeUndefined();
  });

  it('runs and verifies sharded Phase 0 OPFS persistence spike', async () => {
    const root = new MemoryDirectoryHandle('root');
    const cache = new OpfsModelCache({
      getRootDir: async () => root as unknown as FileSystemDirectoryHandle,
    });

    const writeRes = await cache.runOpfsSpike(3, 2);
    expect(writeRes.ok).toBe(true);
    expect(writeRes.shardCount).toBe(2);
    expect(writeRes.totalBytes).toBe(3 * 1024 * 1024);

    const verifyRes = await cache.verifyOpfsSpike();
    expect(verifyRes.ok).toBe(true);
    expect(verifyRes.shardCount).toBe(2);
    expect(verifyRes.totalBytes).toBe(3 * 1024 * 1024);

    await cache.clearOpfsSpike();
    const postClearVerify = await cache.verifyOpfsSpike();
    expect(postClearVerify.ok).toBe(false);
  });
});
