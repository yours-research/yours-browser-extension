import { describe, expect, it } from 'vitest';
import {
  buildGemma4DonorRoutes,
  computeGemma4KvMemoryPlan,
  extractTopKLogits,
  GEMMA4_FULL_ATTN_LAYERS,
  GEMMA4_GLOBAL_DONOR_LAYER,
  GEMMA4_NUM_LAYERS,
  GEMMA4_SLIDING_DONOR_LAYER,
  GEMMA4_SLIDING_RING_SLOTS,
  GEMMA4_SLIDING_WINDOW,
  normalizeGemma4PleTokenId,
} from '../src/offscreen/gemma4WgslEngine';

describe('Gemma 4 E4B WGSL architecture & KV routing', () => {
  it('routes 42 layers cleanly across 20 sliding donors, 4 full donors, and 18 shared layers', () => {
    const routes = buildGemma4DonorRoutes();
    expect(routes).toHaveLength(GEMMA4_NUM_LAYERS);

    const ownSliding = routes.filter((r) => !r.isKvShared && !r.isFullAttn);
    const ownGlobal = routes.filter((r) => !r.isKvShared && r.isFullAttn);
    const sharedSliding = routes.filter((r) => r.isKvShared && !r.isFullAttn);
    const sharedGlobal = routes.filter((r) => r.isKvShared && r.isFullAttn);

    expect(ownSliding).toHaveLength(20);
    expect(ownGlobal).toHaveLength(4);
    expect(ownGlobal.map((r) => r.layer)).toEqual([5, 11, 17, 23]);
    expect(sharedSliding).toHaveLength(15);
    expect(sharedGlobal).toHaveLength(3);
    expect(sharedGlobal.map((r) => r.layer)).toEqual([29, 35, 41]);

    // Sliding donors occupy localDonorIdx 0..19 cleanly
    expect(ownSliding.map((r) => r.localDonorIdx)).toEqual(
      Array.from({ length: 20 }, (_, i) => i),
    );
    expect(routes[GEMMA4_SLIDING_DONOR_LAYER].localDonorIdx).toBe(19);
    for (const r of sharedSliding) {
      expect(r.donorLayer).toBe(GEMMA4_SLIDING_DONOR_LAYER);
      expect(r.localDonorIdx).toBe(19);
      expect(r.globalBufIdx).toBe(-1);
    }

    // Global donors split evenly across kvGlobal0 (5, 11) and kvGlobal1 (17, 23)
    expect(routes[5]).toMatchObject({ localDonorIdx: 0, globalBufIdx: 0 });
    expect(routes[11]).toMatchObject({ localDonorIdx: 1, globalBufIdx: 0 });
    expect(routes[17]).toMatchObject({ localDonorIdx: 0, globalBufIdx: 1 });
    expect(routes[23]).toMatchObject({ localDonorIdx: 1, globalBufIdx: 1 });
    for (const r of sharedGlobal) {
      expect(r.donorLayer).toBe(GEMMA4_GLOBAL_DONOR_LAYER);
      expect(r.localDonorIdx).toBe(1);
      expect(r.globalBufIdx).toBe(1);
      expect(GEMMA4_FULL_ATTN_LAYERS.has(r.layer)).toBe(true);
    }
  });

  it('maintains ring-buffer non-collision invariant across max chunk prefill', () => {
    // 512 sliding window + max 64 token batch < 1024 ring slots (cannot overwrite unread keys)
    expect(GEMMA4_SLIDING_WINDOW + 64).toBeLessThan(GEMMA4_SLIDING_RING_SLOTS);
  });

  it('computes exact VRAM footprints for FP16 32K and INT8 128K', () => {
    const weightsBytes = 1011906048 + 990897664 + 168296448 + 56170664; // 2,227,270,824 B
    const fp16Plan32k = computeGemma4KvMemoryPlan(32768, 'fp16', weightsBytes);
    expect(fp16Plan32k.slidingKvBytes).toBe(40 * 1024 * 1024);
    expect(fp16Plan32k.globalKvPerBufBytes).toBe(256 * 1024 * 1024);
    expect(fp16Plan32k.kvTotalMb).toBe(552);

    const int8Plan128k = computeGemma4KvMemoryPlan(131072, 'int8', weightsBytes);
    expect(int8Plan128k.slidingKvBytes).toBe(40 * 1024 * 1024);
    expect(int8Plan128k.globalKvPerBufBytes).toBe(516 * 1024 * 1024);
    expect(int8Plan128k.kvTotalMb).toBe(1072);
    // Fits comfortably below 3.4 GiB total VRAM on 8GB/16GB/24GB consumer GPUs
    expect(int8Plan128k.totalVramMb).toBeLessThan(3300);
  });

  it('clamps sentinel image/audio and out-of-vocab PLE token IDs to 0', () => {
    expect(normalizeGemma4PleTokenId(2)).toBe(2);
    expect(normalizeGemma4PleTokenId(236812)).toBe(236812);
    expect(normalizeGemma4PleTokenId(258880)).toBe(0);
    expect(normalizeGemma4PleTokenId(258881)).toBe(0);
    expect(normalizeGemma4PleTokenId(262144)).toBe(0);
    expect(normalizeGemma4PleTokenId(-1)).toBe(0);
  });

  it('extracts sorted top-K logits cleanly', () => {
    const arr = new Float32Array([-5, 28.5, 19.2, 22.7969, 3.1, 21.3281]);
    expect(extractTopKLogits(arr, 3)).toEqual([
      { id: 1, val: 28.5 },
      { id: 3, val: 22.7969 },
      { id: 5, val: 21.3281 },
    ]);
  });
});
