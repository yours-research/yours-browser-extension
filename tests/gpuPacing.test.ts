import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  calibratePacingForHardware,
  computeDecodeBurstBreatherMs,
  computeDecodePostStepBreatherMs,
  computeDecodeSliceBounds,
  computePrefillSliceBreatherMs,
  getGpuPacingPolicy,
  resolveExtensionAssetUrl,
  selectDecodeSliceCount,
  selectPrefillChunkSize,
  sleepMs,
} from '../src/offscreen/gpuPacing.ts';
import {
  loadWgslWasm,
  streamUrlToGpuBuffer,
  type GpuBufferLike,
  type GpuDeviceLike,
} from '../src/offscreen/wgslWebgpu.ts';
import {
  DEFAULT_ENGINE_CONFIG,
  INITIAL_TELEMETRY_STATE,
  MODEL_PRESETS,
  type GpuPacingMode,
} from '../src/shared/protocol.ts';

function assertCompleteContiguousCoverage(
  slices: Array<{ start: number; end: number }>,
  totalSteps: number,
): void {
  expect(slices.length).toBeGreaterThan(0);
  expect(slices[0]?.start).toBe(0);
  expect(slices[slices.length - 1]?.end).toBe(totalSteps);
  for (let i = 0; i < slices.length; i++) {
    expect(slices[i]!.end).toBeGreaterThan(slices[i]!.start);
    if (i > 0) {
      expect(slices[i]!.start).toBe(slices[i - 1]!.end);
    }
  }
}

describe('protocol defaults (Wave 1B)', () => {
  it('sets DEFAULT_ENGINE_CONFIG and INITIAL_TELEMETRY_STATE to the sole optimal configuration', () => {
    expect(DEFAULT_ENGINE_CONFIG.modelId).toBe(MODEL_PRESETS[0].id);
    expect(DEFAULT_ENGINE_CONFIG.dtype).toBe('q4f16');
    expect(DEFAULT_ENGINE_CONFIG.chunkSize).toBe(64);
    expect(DEFAULT_ENGINE_CONFIG.backend).toBe('wgsl');
    expect(DEFAULT_ENGINE_CONFIG.wgslKvMode).toBe('int8');
    expect(DEFAULT_ENGINE_CONFIG.wgslMaxContext).toBe(131072);
    expect(DEFAULT_ENGINE_CONFIG.yieldMode).toBe('none');
    expect(DEFAULT_ENGINE_CONFIG.fastDecodePath).toBe(true);
    expect(DEFAULT_ENGINE_CONFIG.fastOpfsLoad).toBe(true);
    expect(DEFAULT_ENGINE_CONFIG.gpuPacingMode).toBe('balanced');
    expect(DEFAULT_ENGINE_CONFIG.executionHost).toBe('worker');
    expect(DEFAULT_ENGINE_CONFIG.enableSpeculative).toBe(true);
    expect(DEFAULT_ENGINE_CONFIG.maxNewTokens).toBe(1024);

    expect(INITIAL_TELEMETRY_STATE.loadSource).toBeNull();
    expect(INITIAL_TELEMETRY_STATE.executionHost).toBe('worker');
    expect(INITIAL_TELEMETRY_STATE.speculativeStats).toBeNull();
  });
});

describe('gpuPacing', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('returns the exact policy table for all three pacing modes without exposing mutable state', () => {
    expect(getGpuPacingPolicy('max-speed')).toEqual({
      mode: 'max-speed',
      prefillChunkShort: 64,
      prefillChunkLong: 64,
      prefillLongThreshold: 16384,
      prefillBurstBudgetMs: Number.POSITIVE_INFINITY,
      prefillBreatherMs: 0,
      decodeSplitsShort: 1,
      decodeSplitsLong: 1,
      decodeLongThreshold: 16384,
      decodeSlowStepThresholdMs: Number.POSITIVE_INFINITY,
      decodeSliceBreatherMs: 0,
      decodeBurstBudgetMs: Number.POSITIVE_INFINITY,
      decodePostStepRatio: 0,
      decodePostStepMinMs: 0,
      decodePostStepMaxMs: 0,
      loadChunkYieldMs: 0,
    });

    expect(getGpuPacingPolicy('balanced')).toEqual({
      mode: 'balanced',
      prefillChunkShort: 64,
      prefillChunkLong: 64,
      prefillLongThreshold: 16384,
      prefillBurstBudgetMs: 8,
      prefillBreatherMs: 4,
      decodeSplitsShort: 1,
      decodeSplitsLong: 2,
      decodeLongThreshold: 32768,
      decodeSlowStepThresholdMs: 14,
      decodeSliceBreatherMs: 2,
      decodeBurstBudgetMs: 14,
      decodePostStepRatio: 0.22,
      decodePostStepMinMs: 2,
      decodePostStepMaxMs: 6,
      loadChunkYieldMs: 1,
    });

    expect(getGpuPacingPolicy('smooth-ui')).toEqual({
      mode: 'smooth-ui',
      prefillChunkShort: 64,
      prefillChunkLong: 32,
      prefillLongThreshold: 8192,
      prefillBurstBudgetMs: 6,
      prefillBreatherMs: 8,
      decodeSplitsShort: 2,
      decodeSplitsLong: 4,
      decodeLongThreshold: 8192,
      decodeSlowStepThresholdMs: 8,
      decodeSliceBreatherMs: 3,
      decodeBurstBudgetMs: 8,
      decodePostStepRatio: 0.45,
      decodePostStepMinMs: 4,
      decodePostStepMaxMs: 12,
      loadChunkYieldMs: 4,
    });

    const mutated = getGpuPacingPolicy('balanced');
    mutated.prefillChunkShort = 999;
    expect(getGpuPacingPolicy('balanced').prefillChunkShort).toBe(64);
  });

  it('selects prefill chunk size across modes, sequence thresholds, and requested caps', () => {
    expect(selectPrefillChunkSize('max-speed', 0)).toBe(64);
    expect(selectPrefillChunkSize('max-speed', 32768, 0)).toBe(64);
    expect(selectPrefillChunkSize('max-speed', 32768, 256)).toBe(64);
    expect(selectPrefillChunkSize('max-speed', 100, 24)).toBe(24);
    expect(selectPrefillChunkSize('max-speed', 100, -4)).toBe(1);

    expect(selectPrefillChunkSize('balanced', 0)).toBe(64);
    expect(selectPrefillChunkSize('balanced', 16383)).toBe(64);
    expect(selectPrefillChunkSize('balanced', 16384)).toBe(64);
    expect(selectPrefillChunkSize('balanced', 500, 64)).toBe(64);
    expect(selectPrefillChunkSize('balanced', 500, 12)).toBe(12);
    expect(selectPrefillChunkSize('balanced', 500, 0)).toBe(64);
    expect(selectPrefillChunkSize('balanced', 20000, 24)).toBe(24);
    expect(selectPrefillChunkSize('balanced', 20000, 8)).toBe(8);

    expect(selectPrefillChunkSize('smooth-ui', 0)).toBe(64);
    expect(selectPrefillChunkSize('smooth-ui', 8191, 32)).toBe(32);
    expect(selectPrefillChunkSize('smooth-ui', 8192, 64)).toBe(32);
    expect(selectPrefillChunkSize('smooth-ui', 8192, 4)).toBe(4);
    expect(selectPrefillChunkSize('smooth-ui', 100, 0.2)).toBe(1);
  });

  it('selects decode slice count from mode, sequence length, and previous step GPU duration', () => {
    expect(selectDecodeSliceCount('max-speed', 0, 0)).toBe(1);
    expect(selectDecodeSliceCount('max-speed', 100000, 200)).toBe(1);

    expect(selectDecodeSliceCount('balanced', 32767, 13.9)).toBe(1);
    expect(selectDecodeSliceCount('balanced', 32768, 5)).toBe(2);
    expect(selectDecodeSliceCount('balanced', 1000, 14)).toBe(2);

    expect(selectDecodeSliceCount('smooth-ui', 8191, 7.9)).toBe(2);
    expect(selectDecodeSliceCount('smooth-ui', 8192, 4)).toBe(4);
    expect(selectDecodeSliceCount('smooth-ui', 512, 8)).toBe(4);
  });

  it('computes adaptive decode burst breathers only once burst budget is exceeded', () => {
    expect(computeDecodeBurstBreatherMs('max-speed', 50, 100)).toEqual({
      breatherMs: 0,
      nextBurstMs: 0,
    });

    // Balanced budget = 14ms: two 6ms steps burst cleanly without sleeping, third triggers breather
    const b1 = computeDecodeBurstBreatherMs('balanced', 6, 0);
    expect(b1).toEqual({ breatherMs: 0, nextBurstMs: 6 });
    const b2 = computeDecodeBurstBreatherMs('balanced', 6, b1.nextBurstMs);
    expect(b2).toEqual({ breatherMs: 0, nextBurstMs: 12 });
    const b3 = computeDecodeBurstBreatherMs('balanced', 6, b2.nextBurstMs);
    expect(b3).toEqual({ breatherMs: 4, nextBurstMs: 0 });

    // Smooth-UI budget = 8ms
    expect(computeDecodeBurstBreatherMs('smooth-ui', 5, 0)).toEqual({
      breatherMs: 0,
      nextBurstMs: 5,
    });
    expect(computeDecodeBurstBreatherMs('smooth-ui', 5, 5)).toEqual({
      breatherMs: 5,
      nextBurstMs: 0,
    });
  });

  it('computes bounded post-step decode breathers per pacing mode', () => {
    expect(computeDecodePostStepBreatherMs('max-speed', 0)).toBe(0);
    expect(computeDecodePostStepBreatherMs('max-speed', 50)).toBe(0);

    expect(computeDecodePostStepBreatherMs('balanced', 0)).toBe(2);
    expect(computeDecodePostStepBreatherMs('balanced', 10)).toBe(2);
    expect(computeDecodePostStepBreatherMs('balanced', 20)).toBe(4);
    expect(computeDecodePostStepBreatherMs('balanced', 25)).toBe(6);
    expect(computeDecodePostStepBreatherMs('balanced', 100)).toBe(6);
    expect(computeDecodePostStepBreatherMs('balanced', Number.NaN)).toBe(2);

    expect(computeDecodePostStepBreatherMs('smooth-ui', 0)).toBe(4);
    expect(computeDecodePostStepBreatherMs('smooth-ui', 16)).toBe(7);
    expect(computeDecodePostStepBreatherMs('smooth-ui', 20)).toBe(9);
    expect(computeDecodePostStepBreatherMs('smooth-ui', 50)).toBe(12);
  });

  it('computes prefill slice breather decisions and resets burst counter when triggered', () => {
    expect(computePrefillSliceBreatherMs('max-speed', 1000)).toEqual({
      breatherMs: 0,
      nextBurstMs: 1000,
    });

    expect(computePrefillSliceBreatherMs('balanced', 7.99)).toEqual({
      breatherMs: 0,
      nextBurstMs: 7.99,
    });
    expect(computePrefillSliceBreatherMs('balanced', 8)).toEqual({
      breatherMs: 4,
      nextBurstMs: 0,
    });
    expect(computePrefillSliceBreatherMs('balanced', 15)).toEqual({
      breatherMs: 4,
      nextBurstMs: 0,
    });

    expect(computePrefillSliceBreatherMs('smooth-ui', 5.99)).toEqual({
      breatherMs: 0,
      nextBurstMs: 5.99,
    });
    expect(computePrefillSliceBreatherMs('smooth-ui', 6)).toEqual({
      breatherMs: 8,
      nextBurstMs: 0,
    });
  });

  describe('computeDecodeSliceBounds', () => {
    // 32 layers mimicking Qwen3.5-4B hybrid (24 DeltaNet @ 14 steps + 8 FullAttn @ 12 steps)
    const layerEndStepIndices32: number[] = [];
    let runningSteps = 0;
    for (let layer = 0; layer < 32; layer++) {
      runningSteps += layer % 4 === 3 ? 12 : 14;
      layerEndStepIndices32.push(runningSteps);
    }
    const total32Steps = layerEndStepIndices32[31]!;

    it.each<{
      numSlices: 1 | 2 | 4;
      expected: Array<{ start: number; end: number }>;
    }>([
      {
        numSlices: 1,
        expected: [{ start: 0, end: total32Steps }],
      },
      {
        numSlices: 2,
        expected: [
          { start: 0, end: layerEndStepIndices32[15]! },
          { start: layerEndStepIndices32[15]!, end: total32Steps },
        ],
      },
      {
        numSlices: 4,
        expected: [
          { start: 0, end: layerEndStepIndices32[7]! },
          { start: layerEndStepIndices32[7]!, end: layerEndStepIndices32[15]! },
          { start: layerEndStepIndices32[15]!, end: layerEndStepIndices32[23]! },
          { start: layerEndStepIndices32[23]!, end: total32Steps },
        ],
      },
    ])('splits 32 layers into $numSlices contiguous slices with zero gaps or overlaps', ({
      numSlices,
      expected,
    }) => {
      const slices = computeDecodeSliceBounds(layerEndStepIndices32, numSlices);
      expect(slices).toEqual(expected);
      assertCompleteContiguousCoverage(slices, total32Steps);
    });

    it('handles small layer counts, empty tables, and duplicate boundaries cleanly', () => {
      expect(computeDecodeSliceBounds([], 4)).toEqual([]);
      expect(computeDecodeSliceBounds([0, 0], 2)).toEqual([]);
      expect(computeDecodeSliceBounds([12], 4)).toEqual([{ start: 0, end: 12 }]);

      const twoLayer = computeDecodeSliceBounds([10, 22], 4);
      expect(twoLayer).toEqual([
        { start: 0, end: 10 },
        { start: 10, end: 22 },
      ]);
      assertCompleteContiguousCoverage(twoLayer, 22);

      const threeLayer = computeDecodeSliceBounds([10, 20, 30], 4);
      expect(threeLayer).toEqual([
        { start: 0, end: 10 },
        { start: 10, end: 20 },
        { start: 20, end: 30 },
      ]);
      assertCompleteContiguousCoverage(threeLayer, 30);

      // Deduplicates when intermediate layers contribute 0 steps
      const repeated = computeDecodeSliceBounds([10, 10, 10, 25], 4);
      expect(repeated).toEqual([
        { start: 0, end: 10 },
        { start: 10, end: 25 },
      ]);
      assertCompleteContiguousCoverage(repeated, 25);
    });
  });

  describe('resolveExtensionAssetUrl', () => {
    it('strips leading slashes and resolves against explicit assetBaseUrl', () => {
      expect(
        resolveExtensionAssetUrl(
          '/wasm/tinygpt_wgsl_engine.wasm',
          'chrome-extension://abcdefghijklmnop/',
        ),
      ).toBe('chrome-extension://abcdefghijklmnop/wasm/tinygpt_wgsl_engine.wasm');

      expect(
        resolveExtensionAssetUrl(
          'wasm/tinygpt_wgsl_engine.wasm',
          'chrome-extension://abcdefghijklmnop',
        ),
      ).toBe('chrome-extension://abcdefghijklmnop/wasm/tinygpt_wgsl_engine.wasm');

      expect(
        resolveExtensionAssetUrl('/models/qwen/weights.bin', 'http://127.0.0.1:4173/ext'),
      ).toBe('http://127.0.0.1:4173/ext/models/qwen/weights.bin');

      expect(resolveExtensionAssetUrl('wasm/engine.wasm', '/assets')).toBe(
        '/assets/wasm/engine.wasm',
      );
    });

    it('falls back to chrome.runtime.getURL, then self.location.origin, then root-relative path', () => {
      vi.stubGlobal('chrome', {
        runtime: {
          getURL: (p: string) => `chrome-extension://mocked-id/${p}`,
        },
      });
      expect(resolveExtensionAssetUrl('/wasm/tinygpt_wgsl_engine.wasm')).toBe(
        'chrome-extension://mocked-id/wasm/tinygpt_wgsl_engine.wasm',
      );

      vi.stubGlobal('chrome', undefined);
      vi.stubGlobal('self', { location: { origin: 'https://worker.example.test' } });
      expect(resolveExtensionAssetUrl('/wasm/tinygpt_wgsl_engine.wasm')).toBe(
        'https://worker.example.test/wasm/tinygpt_wgsl_engine.wasm',
      );

      vi.stubGlobal('self', { location: { origin: 'null' } });
      expect(resolveExtensionAssetUrl('/wasm/tinygpt_wgsl_engine.wasm')).toBe(
        '/wasm/tinygpt_wgsl_engine.wasm',
      );

      vi.stubGlobal('self', undefined);
      expect(resolveExtensionAssetUrl('wasm/tinygpt_wgsl_engine.wasm')).toBe(
        '/wasm/tinygpt_wgsl_engine.wasm',
      );
    });
  });

  it('calibrates pacing policy dynamically for 60/120/144Hz monitors and slow iGPUs', () => {
    // max-speed stays unthrottled regardless of Hz or step time
    expect(calibratePacingForHardware('max-speed', 144, 150)).toEqual(getGpuPacingPolicy('max-speed'));

    // 60Hz fast dGPU matches balanced baseline
    const bal60 = calibratePacingForHardware('balanced', 60, 4);
    expect(bal60.decodeBurstBudgetMs).toBe(14);
    expect(bal60.decodeSplitsShort).toBe(1);
    expect(bal60.prefillChunkShort).toBe(64);

    // 120Hz and 144Hz tighten burst budget to fit frame cadence
    const bal120 = calibratePacingForHardware('balanced', 120, 4);
    expect(bal120.decodeBurstBudgetMs).toBeLessThanOrEqual(8);
    const bal144 = calibratePacingForHardware('balanced', 144, 4);
    expect(bal144.decodeBurstBudgetMs).toBeLessThanOrEqual(7);
    const smooth144 = calibratePacingForHardware('smooth-ui', 144, 4);
    expect(smooth144.decodeBurstBudgetMs).toBeLessThanOrEqual(4);

    // Slow iGPU (>= 100ms per decode step) increases decode splits and shrinks prefill chunks for TDR safety
    const slowBalanced = calibratePacingForHardware('balanced', 60, 120);
    expect(slowBalanced.decodeSplitsShort).toBe(2);
    expect(slowBalanced.decodeSplitsLong).toBe(4);
    expect(slowBalanced.prefillChunkShort).toBe(32);
    expect(slowBalanced.prefillChunkLong).toBe(16);

    const slowSmooth = calibratePacingForHardware('smooth-ui', 60, 120);
    expect(slowSmooth.decodeSplitsShort).toBe(4);
    expect(slowSmooth.decodeSplitsLong).toBe(4);
    expect(slowSmooth.prefillChunkShort).toBe(32);
    expect(slowSmooth.prefillChunkLong).toBe(16);
  });

  it('sleepMs resolves immediately for non-positive durations and waits for positive ms', async () => {
    const spy = vi.spyOn(globalThis, 'setTimeout');
    await sleepMs(0);
    await sleepMs(-10);
    expect(spy).not.toHaveBeenCalled();

    await sleepMs(1);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

describe('wgslWebgpu URL asset loading & streaming', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function createMockGpuDevice(targetBytes: number): {
    device: GpuDeviceLike;
    gpuBuf: GpuBufferLike;
    written: Uint8Array;
    writeCalls: Array<{ offset: number; size: number }>;
  } {
    const written = new Uint8Array(targetBytes);
    const writeCalls: Array<{ offset: number; size: number }> = [];
    const gpuBuf = { size: targetBytes } as GpuBufferLike;
    const device = {
      queue: {
        writeBuffer: vi.fn(
          (
            _buf: GpuBufferLike,
            bufferOffset: number,
            data: BufferSource,
            dataOffset = 0,
            size?: number,
          ) => {
            const view = ArrayBuffer.isView(data)
              ? new Uint8Array(data.buffer, data.byteOffset + dataOffset, size ?? data.byteLength)
              : new Uint8Array(data, dataOffset, size ?? data.byteLength);
            written.set(view, bufferOffset);
            writeCalls.push({ offset: bufferOffset, size: view.byteLength });
          },
        ),
        submit: vi.fn(),
        onSubmittedWorkDone: vi.fn(async () => {}),
      },
    } as unknown as GpuDeviceLike;
    return { device, gpuBuf, written, writeCalls };
  }

  it('streams chunks smaller and larger than stagingBytes with trailing aligned flush and yields', async () => {
    const total = 20;
    const expected = new Uint8Array(total);
    for (let i = 0; i < total; i++) expected[i] = (i * 17 + 3) & 0xff;

    const chunks = [
      expected.subarray(0, 3),
      expected.subarray(3, 18), // spans across two 8-byte staging flushes
      expected.subarray(18, 20), // leaves 4 trailing bytes (16..19)
    ];
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(body, { status: 200 })),
    );

    const { device, gpuBuf, written, writeCalls } = createMockGpuDevice(total);
    const progress: number[] = [];
    const modes: GpuPacingMode[] = ['balanced'];
    expect(modes[0]).toBe('balanced');

    await streamUrlToGpuBuffer(device, 'https://example.test/weights.bin', gpuBuf, total, {
      stagingBytes: 8,
      yieldMs: 1,
      onProgress: (loaded) => progress.push(loaded),
    });

    expect(written).toEqual(expected);
    expect(writeCalls).toEqual([
      { offset: 0, size: 8 },
      { offset: 8, size: 8 },
      { offset: 16, size: 4 },
    ]);
    expect(progress).toEqual([8, 16, 20]);
  });

  it('handles !res.body fallback cleanly using stagingBytes chunks', async () => {
    const total = 12;
    const payload = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        statusText: 'OK',
        body: null,
        arrayBuffer: async () => payload.buffer.slice(0),
      })),
    );

    const { device, gpuBuf, written, writeCalls } = createMockGpuDevice(total);
    const progress: number[] = [];

    await streamUrlToGpuBuffer(device, 'https://example.test/fallback.bin', gpuBuf, total, {
      stagingBytes: 8,
      onProgress: (loaded) => progress.push(loaded),
    });

    expect(written).toEqual(payload);
    expect(writeCalls).toEqual([
      { offset: 0, size: 8 },
      { offset: 8, size: 4 },
    ]);
    expect(progress).toEqual([8, 12]);
  });

  it('streams two sequential shards with dstByteOffset into one GPU buffer', async () => {
    const shard0 = new Uint8Array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
    const shard1 = new Uint8Array([30, 31, 32, 33, 34, 35, 36, 37]);
    const expected = new Uint8Array([...shard0, ...shard1]);

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string) => {
        const bytes = input.endsWith('part00.bin') ? shard0 : shard1;
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(bytes.subarray(0, 5));
              controller.enqueue(bytes.subarray(5));
              controller.close();
            },
          }),
          { status: 200 },
        );
      }),
    );

    const { device, gpuBuf, written, writeCalls } = createMockGpuDevice(expected.byteLength);
    const p0: number[] = [];
    const p1: number[] = [];

    await streamUrlToGpuBuffer(device, 'https://example.test/slab.part00.bin', gpuBuf, shard0.byteLength, {
      dstByteOffset: 0,
      stagingBytes: 8,
      onProgress: (loaded) => p0.push(loaded),
    });
    await streamUrlToGpuBuffer(device, 'https://example.test/slab.part01.bin', gpuBuf, shard1.byteLength, {
      dstByteOffset: shard0.byteLength,
      stagingBytes: 8,
      onProgress: (loaded) => p1.push(loaded),
    });

    expect(written).toEqual(expected);
    expect(p0).toEqual([8, 12]);
    expect(p1).toEqual([8]);
    expect(writeCalls).toEqual([
      { offset: 0, size: 8 },
      { offset: 8, size: 4 },
      { offset: 12, size: 8 },
    ]);
  });

  it('rejects HTTP errors, unaligned trailing bytes, and byte-count mismatches', async () => {
    const { device, gpuBuf } = createMockGpuDevice(16);

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 404, statusText: 'Not Found' })),
    );
    await expect(
      streamUrlToGpuBuffer(device, 'https://example.test/missing.bin', gpuBuf, 8),
    ).rejects.toThrow(/404/);

    const unalignedBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3, 4, 5, 6]));
        controller.close();
      },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(unalignedBody, { status: 200 })),
    );
    await expect(
      streamUrlToGpuBuffer(device, 'https://example.test/unaligned.bin', gpuBuf, 6, {
        stagingBytes: 4,
      }),
    ).rejects.toThrow(/4-byte aligned/);

    const shortBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3, 4]));
        controller.close();
      },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(shortBody, { status: 200 })),
    );
    await expect(
      streamUrlToGpuBuffer(device, 'https://example.test/short.bin', gpuBuf, 8, {
        stagingBytes: 4,
      }),
    ).rejects.toThrow(/mismatch/i);
  });

  it('loadWgslWasm resolves wasm path via assetBaseUrl', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      arrayBuffer: async () => new ArrayBuffer(8),
    }));
    vi.stubGlobal('fetch', fetchMock);
    const instantiateMock = vi
      .spyOn(WebAssembly, 'instantiate')
      .mockImplementation((async () => ({
        instance: { exports: { ok: 1 } },
      })) as unknown as typeof WebAssembly.instantiate);

    const exports = await loadWgslWasm('chrome-extension://worker-id');
    expect(fetchMock).toHaveBeenCalledWith(
      'chrome-extension://worker-id/wasm/tinygpt_wgsl_engine.wasm',
    );
    expect(instantiateMock).toHaveBeenCalledTimes(1);
    expect(exports).toEqual({ ok: 1 });
  });
});
