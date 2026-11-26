import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  computeAuxLayout,
  QWEN35_HIDDEN,
  QWEN35_INTERMEDIATE,
  QWEN35_NUM_LAYERS,
  QWEN35_VOCAB_SIZE,
} from '../src/offscreen/onnxWeightLoader.ts';
import {
  computeKvSliceByteOffset,
  computeLayerWeightByteSizes,
  computeTieredKvPlan,
  getDeltaNetConvDim,
  getDeltaNetLayerCount,
  getDeltaNetQkvzAbRows,
  getFullAttnKvHeadU32s,
  getFullAttnLayerCount,
  getFullAttnQkvRows,
  partitionVocabRowsIntoSlabs,
  partitionWeightLayersIntoSlabs,
  q4MatrixBytes,
  QWEN35_4B_DEFAULT_ARCH,
  resolveModelArch,
  type ModelArchConfig,
} from '../src/offscreen/modelConfig.ts';
import { WgslQwen35Engine } from '../src/offscreen/wgslRealEngine.ts';
import type { WgslComputeTier } from '../src/offscreen/wgslWebgpu.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WASM_PATH = path.resolve(
  __dirname,
  '../crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
);

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

const QWEN35_2B_ARCH: ModelArchConfig = {
  hiddenSize: 2048,
  intermediateSize: 6144,
  numLayers: 24,
  vocabSize: 248320,
  fullAttnLayers: [3, 7, 11, 15, 19, 23],
  dnKeyHeads: 16,
  dnValHeads: 16,
  dnHeadDim: 128,
  dnConvKernel: 4,
  attnQHeads: 8,
  attnKvHeads: 2,
  attnHeadDim: 256,
  ropeDim: 64,
  ropeTheta: 10000000,
};

describe('modelConfig geometry & partitioning', () => {
  it('matches legacy Qwen3.5-4B defaults and backward-compatible v1 manifest resolution', () => {
    expect(QWEN35_4B_DEFAULT_ARCH.hiddenSize).toBe(QWEN35_HIDDEN);
    expect(QWEN35_4B_DEFAULT_ARCH.intermediateSize).toBe(QWEN35_INTERMEDIATE);
    expect(QWEN35_4B_DEFAULT_ARCH.numLayers).toBe(QWEN35_NUM_LAYERS);
    expect(QWEN35_4B_DEFAULT_ARCH.vocabSize).toBe(QWEN35_VOCAB_SIZE);
    expect(QWEN35_4B_DEFAULT_ARCH.fullAttnLayers).toEqual([3, 7, 11, 15, 19, 23, 27, 31]);

    const resolvedV1 = resolveModelArch();
    expect(resolvedV1).toEqual(QWEN35_4B_DEFAULT_ARCH);
    resolvedV1.fullAttnLayers.push(99);
    expect(QWEN35_4B_DEFAULT_ARCH.fullAttnLayers).toHaveLength(8);

    const pruned4B = resolveModelArch({ arch: { vocabSize: 49152 } });
    expect(pruned4B.vocabSize).toBe(49152);
    expect(pruned4B.hiddenSize).toBe(2560);
    expect(pruned4B.numLayers).toBe(32);

    const resolved2B = resolveModelArch({ arch: QWEN35_2B_ARCH });
    expect(resolved2B).toEqual(QWEN35_2B_ARCH);

    const exp003cManifest = JSON.parse(
      fs.readFileSync(
        'D:/yours-pragma-lab/05_wgsl_bundles/EXP-003c-tool-qad/manifest.json',
        'utf8',
      ),
    );
    const resolvedExp003c = resolveModelArch(exp003cManifest);
    expect(resolvedExp003c.vocabSize).toBe(81920);
    expect(resolvedExp003c.hiddenSize).toBe(2560);
    expect(resolvedExp003c.intermediateSize).toBe(9216);
    expect(resolvedExp003c.numLayers).toBe(32);
    expect(resolvedExp003c.fullAttnLayers).toEqual([3, 7, 11, 15, 19, 23, 27, 31]);

    const exp003cVocabSlices = partitionVocabRowsIntoSlabs(
      resolvedExp003c.vocabSize,
      resolvedExp003c.hiddenSize,
      256 * MIB,
    );
    expect(exp003cVocabSlices).toHaveLength(1);
    expect(exp003cVocabSlices[0]?.byteSize).toBe(exp003cManifest.slab2Bytes);
    expect(exp003cVocabSlices[0]?.byteSize).toBe(131072000);
    const exp003cLayers = computeLayerWeightByteSizes(resolvedExp003c).reduce((a, b) => a + b, 0);
    expect(exp003cLayers).toBe(exp003cManifest.slab0Bytes + exp003cManifest.slab1Bytes);
    expect(computeAuxLayout(resolvedExp003c).totalF32s * 4).toBe(exp003cManifest.auxBytes);
  });

  it('computes derived geometry helpers accurately for 4B, pruned 4B, and 2B', () => {
    expect(getFullAttnLayerCount(QWEN35_4B_DEFAULT_ARCH)).toBe(8);
    expect(getDeltaNetLayerCount(QWEN35_4B_DEFAULT_ARCH)).toBe(24);
    expect(getDeltaNetConvDim(QWEN35_4B_DEFAULT_ARCH)).toBe(8192);
    expect(getDeltaNetQkvzAbRows(QWEN35_4B_DEFAULT_ARCH)).toBe(12352);
    expect(getFullAttnQkvRows(QWEN35_4B_DEFAULT_ARCH)).toBe(10240);
    expect(getFullAttnKvHeadU32s(QWEN35_4B_DEFAULT_ARCH, 'int8')).toBe(65);
    expect(getFullAttnKvHeadU32s(QWEN35_4B_DEFAULT_ARCH, 'fp16')).toBe(128);

    expect(getFullAttnLayerCount(QWEN35_2B_ARCH)).toBe(6);
    expect(getDeltaNetLayerCount(QWEN35_2B_ARCH)).toBe(18);
    expect(getDeltaNetConvDim(QWEN35_2B_ARCH)).toBe(6144); // 16*128*2 + 16*128
    expect(getDeltaNetQkvzAbRows(QWEN35_2B_ARCH)).toBe(8224); // 6144 + 2048 + 32
    expect(getFullAttnQkvRows(QWEN35_2B_ARCH)).toBe(5120); // 8*256*2 + 2*256*2
  });

  it('computes exact computeAuxLayout offsets for 4B, pruned-vocab 4B, and 2B', () => {
    const aux4B = computeAuxLayout(QWEN35_4B_DEFAULT_ARCH);
    expect(aux4B.totalF32s * 4).toBe(3846272); // exact manifest.auxBytes in bundled 4B

    const pruned4B = resolveModelArch({ arch: { vocabSize: 49152 } });
    const auxPruned = computeAuxLayout(pruned4B);
    expect(auxPruned.totalF32s).toBe(aux4B.totalF32s);

    const aux2B = computeAuxLayout(QWEN35_2B_ARCH);
    expect(aux2B.layers).toHaveLength(24);
    const expected2BTotal =
      32 + // inv_freq (ropeDim / 2)
      2048 + // final_norm
      24 * (2 * 2048) + // input + post norms
      6 * (256 + 256) + // 6 FullAttn q_norm + k_norm
      18 * (16 + 16 + 128 + 6144 * 4); // 18 DeltaNet (dnAux + conv1d)
    expect(aux2B.totalF32s).toBe(expected2BTotal);
    expect(aux2B.layers[3]!.isFullAttn).toBe(true);
    expect(aux2B.layers[0]!.isFullAttn).toBe(false);
    expect(aux2B.layers[0]!.convWOff - aux2B.layers[0]!.dnAuxOff).toBe(160);
  });

  it('partitions transformer layers into >= 2 contiguous GPU slabs under 256MiB, 512MiB, 1GiB, and 2GiB caps', () => {
    const sizes4B = computeLayerWeightByteSizes(QWEN35_4B_DEFAULT_ARCH);
    expect(sizes4B).toHaveLength(32);
    const total4BBytes = sizes4B.reduce((a, b) => a + b, 0);
    // Exact bundled 4B slab0 + slab1 byte count: 1115340800 * 2
    expect(total4BBytes).toBe(1115340800 * 2);

    for (const maxBinding of [256 * MIB, 512 * MIB, 1 * GIB, 2 * GIB]) {
      const cap = Math.min(1 * GIB, maxBinding);
      const parts = partitionWeightLayersIntoSlabs(sizes4B, maxBinding);
      expect(parts.length).toBeGreaterThanOrEqual(2);
      expect(parts[0]!.startLayer).toBe(0);
      expect(parts[parts.length - 1]!.endLayer).toBe(32);

      let sumBytes = 0;
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i]!;
        expect(p.slabIdx).toBe(i);
        expect(p.endLayer).toBeGreaterThan(p.startLayer);
        expect(p.totalBytes).toBeLessThanOrEqual(cap);
        if (i > 0) {
          expect(p.startLayer).toBe(parts[i - 1]!.endLayer);
        }
        sumBytes += p.totalBytes;
      }
      expect(sumBytes).toBe(total4BBytes);
    }

    // 256 MiB needs 16 slabs (2 layers/slab ~141 MiB), 512 MiB needs 8 slabs, 1 GiB/2 GiB needs 4 slabs
    expect(partitionWeightLayersIntoSlabs(sizes4B, 256 * MIB)).toHaveLength(16);
    expect(partitionWeightLayersIntoSlabs(sizes4B, 512 * MIB)).toHaveLength(8);
    expect(partitionWeightLayersIntoSlabs(sizes4B, 1 * GIB)).toHaveLength(4);
    expect(partitionWeightLayersIntoSlabs(sizes4B, 2 * GIB)).toHaveLength(4);

    // 2B fits cleanly in 2 slabs under 1 GiB
    const sizes2B = computeLayerWeightByteSizes(QWEN35_2B_ARCH);
    expect(partitionWeightLayersIntoSlabs(sizes2B, 1 * GIB)).toHaveLength(2);
  });

  it('row-slices lm_head across multiple buffers when maxBindingBytes < slab2Bytes', () => {
    const fullBytes = q4MatrixBytes(248320, 2560);
    expect(fullBytes).toBe(397312000);

    const single = partitionVocabRowsIntoSlabs(248320, 2560, 1 * GIB);
    expect(single).toHaveLength(1);
    expect(single[0]!.byteSize).toBe(397312000);

    const sliced256 = partitionVocabRowsIntoSlabs(248320, 2560, 256 * MIB);
    expect(sliced256).toHaveLength(2);
    expect(sliced256[0]!.startRow).toBe(0);
    expect(sliced256[0]!.endRow).toBe(sliced256[1]!.startRow);
    expect(sliced256[1]!.endRow).toBe(248320);
    expect(sliced256[0]!.byteSize).toBeLessThanOrEqual(256 * MIB);
    expect(sliced256[1]!.byteSize).toBeLessThanOrEqual(256 * MIB);
    expect(sliced256[0]!.byteSize + sliced256[1]!.byteSize).toBe(397312000);
  });

  it('computes tiered KV cache plans across 4GB, 8GB, 16GB, and 24GB hardware limits', () => {
    const plan24 = computeTieredKvPlan(
      131072,
      QWEN35_4B_DEFAULT_ARCH,
      'int8',
      { maxBufferSize: 4 * GIB, maxStorageBufferBindingSize: 2 * GIB },
      24,
    );
    expect(plan24.initialCapacity).toBe(32768 + 512);
    expect(plan24.maxSafeCapacity).toBe(131072 + 512);
    expect(plan24.tiers).toEqual([33280, 66048, 131584]);

    const plan16 = computeTieredKvPlan(
      131072,
      QWEN35_4B_DEFAULT_ARCH,
      'int8',
      { maxBufferSize: 2 * GIB, maxStorageBufferBindingSize: 1 * GIB },
      16,
    );
    expect(plan16.initialCapacity).toBe(33280);
    expect(plan16.maxSafeCapacity).toBeLessThanOrEqual(Math.floor((1 * GIB) / (4 * 2 * 4 * 65 * 4)));
    expect(plan16.tiers[0]).toBe(plan16.initialCapacity);
    expect(plan16.tiers[plan16.tiers.length - 1]).toBe(plan16.maxSafeCapacity);

    const plan8 = computeTieredKvPlan(
      131072,
      QWEN35_4B_DEFAULT_ARCH,
      'int8',
      { maxBufferSize: 1 * GIB, maxStorageBufferBindingSize: 1 * GIB },
      8,
    );
    expect(plan8.initialCapacity).toBe(16384 + 512);
    expect(plan8.tiers[0]).toBe(16896);

    const plan4 = computeTieredKvPlan(
      131072,
      QWEN35_4B_DEFAULT_ARCH,
      'int8',
      { maxBufferSize: 256 * MIB, maxStorageBufferBindingSize: 256 * MIB },
      4,
    );
    expect(plan4.initialCapacity).toBe(16384 + 512);
    expect(plan4.maxSafeCapacity).toBeLessThan(65536);
    for (let i = 1; i < plan4.tiers.length; i++) {
      expect(plan4.tiers[i]!).toBeGreaterThan(plan4.tiers[i - 1]!);
    }
  });

  it('computes non-overlapping KV slice byte offsets for GPU growth and prefix slots', () => {
    const cap = 16896;
    const seqLen = 256;
    const sliceBytes = seqLen * 65 * 4;
    const spans: Array<{ start: number; end: number }> = [];
    for (let l = 0; l < 4; l++) {
      for (const p of [0, 1] as const) {
        for (let h = 0; h < 4; h++) {
          const start = computeKvSliceByteOffset(l, p, h, cap, QWEN35_4B_DEFAULT_ARCH, 'int8');
          spans.push({ start, end: start + sliceBytes });
        }
      }
    }
    spans.sort((a, b) => a.start - b.start);
    for (let i = 1; i < spans.length; i++) {
      expect(spans[i]!.start).toBeGreaterThanOrEqual(spans[i - 1]!.end);
    }
  });

  it('validates all 43 compiled WASM WGSL shaders across Tier 1, Tier 2, and Tier 3 feature constraints', async () => {
    const wasmBytes = fs.readFileSync(WASM_PATH);
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    const e = instance.exports as {
      memory: WebAssembly.Memory;
      wgsl_build_shader(id: number): number;
      wgsl_get_shader_ptr(): number;
    };
    const dec = new TextDecoder();
    const readShader = (id: number) => {
      const len = e.wgsl_build_shader(id);
      expect(len).toBeGreaterThan(0);
      expect(len).toBeLessThan(65536);
      return dec.decode(new Uint8Array(e.memory.buffer, e.wgsl_get_shader_ptr(), len));
    };

    for (let id = 0; id <= 42; id++) {
      const src = readShader(id);
      expect(src).toContain('fn main(');
    }

    const tier2Ids = [10, 34, 12, 25, 29, 27, 31, 13, 14, 15, 41, 42, 16, 35, 37, 9, 38, 40, 7, 8];
    for (const id of tier2Ids) {
      const src = readShader(id);
      expect(src).not.toContain('enable subgroups');
      expect(src).not.toMatch(/\bsubgroup/);
    }

    const tier3Ids = [10, 34, 12, 26, 30, 28, 32, 33, 14, 15, 41, 42, 16, 36, 37, 9, 39, 40, 7, 8];
    for (const id of tier3Ids) {
      const src = readShader(id);
      expect(src).not.toContain('enable f16');
      expect(src).not.toContain('<f16>');
      expect(src).not.toContain('f16(');
      expect(src).not.toContain('enable subgroups');
      expect(src).not.toMatch(/\bsubgroup/);
    }
  });
});

describe('WgslQwen35Engine fallback, KV growth, and GPU prefix slots', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('self-heals from tier1 -> tier2 -> tier3 and manages KV growth + GPU prefix slots', async () => {
    const wasmBytes = fs.readFileSync(WASM_PATH);
    const tinyArch: ModelArchConfig = {
      hiddenSize: 64,
      intermediateSize: 128,
      numLayers: 4,
      vocabSize: 64,
      fullAttnLayers: [3],
      dnKeyHeads: 1,
      dnValHeads: 1,
      dnHeadDim: 128,
      dnConvKernel: 4,
      attnQHeads: 4,
      attnKvHeads: 1,
      attnHeadDim: 256,
      ropeDim: 64,
      ropeTheta: 10000000,
    };
    const layerSizes = computeLayerWeightByteSizes(tinyArch);
    const halfBytes0 = layerSizes[0]! + layerSizes[1]!;
    const halfBytes1 = layerSizes[2]! + layerSizes[3]!;
    const lmBytes = q4MatrixBytes(tinyArch.vocabSize, tinyArch.hiddenSize);
    const auxLayout = computeAuxLayout(tinyArch);
    const auxBytes = auxLayout.totalF32s * 4;

    const manifest = {
      version: 2,
      modelId: 'test/tiny-qwen35',
      complete: true,
      arch: tinyArch,
      slab0Bytes: halfBytes0,
      slab1Bytes: halfBytes1,
      slab2Bytes: lmBytes,
      auxBytes,
      createdAt: '2026-11-25T00:00:00Z',
    };

    const compiledTiers: string[] = [];
    const copyCalls: Array<{ srcSize: number; dstSize: number; size: number }> = [];
    const destroyedBuffers: number[] = [];

    const mockDevice = {
      limits: {
        maxBufferSize: 512 * MIB,
        maxStorageBufferBindingSize: 256 * MIB,
      },
      features: new Set(['shader-f16', 'subgroups']),
      queue: {
        writeBuffer: vi.fn(),
        submit: vi.fn(),
        onSubmittedWorkDone: vi.fn(async () => {}),
      },
      createBuffer: vi.fn((desc: { size: number; usage: number }) => ({
        size: desc.size,
        mapAsync: vi.fn(async () => {}),
        getMappedRange: vi.fn(() => new ArrayBuffer(desc.size)),
        unmap: vi.fn(),
        destroy: vi.fn(() => {
          destroyedBuffers.push(desc.size);
        }),
      })),
      createShaderModule: vi.fn(({ label }: { label?: string; code: string }) => ({
        label,
        getCompilationInfo: async () => ({ messages: [] }),
      })),
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createComputePipelineAsync: vi.fn(async ({ label }: { label?: string }) => {
        if (label?.startsWith('tier1-f16-subgroups') || label?.startsWith('tier2-f16-shared')) {
          compiledTiers.push(label);
          throw new Error(`Simulated driver failure for ${label}`);
        }
        compiledTiers.push(label ?? 'unknown');
        return { label };
      }),
      createBindGroup: vi.fn(() => ({})),
      createCommandEncoder: vi.fn(() => ({
        beginComputePass: () => ({
          setPipeline: vi.fn(),
          setBindGroup: vi.fn(),
          dispatchWorkgroups: vi.fn(),
          end: vi.fn(),
        }),
        copyBufferToBuffer: vi.fn(
          (src: { size: number }, _so: number, dst: { size: number }, _do: number, size: number) => {
            copyCalls.push({ srcSize: src.size, dstSize: dst.size, size });
          },
        ),
        clearBuffer: vi.fn(),
        finish: vi.fn(() => ({})),
      })),
      pushErrorScope: vi.fn(),
      popErrorScope: vi.fn(async () => null),
      destroy: vi.fn(),
    };

    vi.stubGlobal('navigator', {
      gpu: {
        wgslLanguageFeatures: { has: () => false },
        requestAdapter: vi.fn(async () => ({
          features: new Set(['shader-f16', 'subgroups']),
          limits: mockDevice.limits,
          requestDevice: vi.fn(async () => mockDevice),
        })),
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url.endsWith('tinygpt_wgsl_engine.wasm')) {
          return { ok: true, status: 200, arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) };
        }
        if (url.endsWith('manifest.json')) {
          return { ok: true, status: 200, json: async () => manifest };
        }
        const sizeMap: Record<string, number> = {
          'slab0.bin': halfBytes0,
          'slab1.bin': halfBytes1,
          'slab2.bin': lmBytes,
          'aux_params.bin': auxBytes,
        };
        const fileName = url.split('/').pop()!;
        const byteLen = sizeMap[fileName] ?? 0;
        return {
          ok: true,
          status: 200,
          body: null,
          arrayBuffer: async () => new ArrayBuffer(byteLen),
        };
      }),
    );

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new WgslQwen35Engine();
    // Skip verifyRealKernels math tolerance failure on zero-filled mock readbacks by stubbing private verifyRealKernels
    vi.spyOn(
      engine as unknown as { verifyRealKernels: () => Promise<void> },
      'verifyRealKernels',
    ).mockResolvedValue(undefined);

    const stats = await engine.load(null, {
      modelId: 'test/tiny-qwen35',
      kvMode: 'int8',
      maxContext: 65536,
      deviceMemoryGb: 8,
    });

    expect(stats.computeTier).toBe('tier3-f32-universal');
    expect(engine.getComputeTier()).toBe('tier3-f32-universal');
    expect(warnSpy).toHaveBeenCalledTimes(2);
    expect(compiledTiers.some((l) => l.startsWith('tier1-f16-subgroups'))).toBe(true);
    expect(compiledTiers.some((l) => l.startsWith('tier2-f16-shared'))).toBe(true);
    expect(compiledTiers.some((l) => l.startsWith('tier3-f32-universal'))).toBe(true);
    expect(stats.allocatedKvCapacity).toBe(16896);
    expect(stats.maxSafeKvCapacity).toBe(66048);

    // Save GPU prefix slot -> grow KV capacity -> restore GPU prefix slot -> delete slot
    const slotInfo = await engine.saveGpuPrefixSlot('turn-0');
    expect(slotInfo.seqPos).toBe(0);
    expect(slotInfo.byteSize).toBeGreaterThan(0);
    expect(engine.listGpuPrefixSlots()).toEqual([
      { slotId: 'turn-0', seqPos: 0, byteSize: slotInfo.byteSize },
    ]);

    await engine.ensureKvCapacity(20000);
    expect(engine.getAllocatedKvCapacity()).toBe(33280);

    const restored = await engine.restoreGpuPrefixSlot('turn-0');
    expect(restored.seqPos).toBe(0);
    expect(copyCalls.length).toBeGreaterThan(0);

    expect(engine.deleteGpuPrefixSlot('turn-0')).toBe(true);
    expect(engine.deleteGpuPrefixSlot('turn-0')).toBe(false);
    expect(engine.listGpuPrefixSlots()).toEqual([]);

    // Verify auto-eviction of oldest non-base slot when exceeding VRAM cap (no throw)
    await engine.saveGpuPrefixSlot('base-system');
    await engine.saveGpuPrefixSlot('tab:first');
    await engine.saveGpuPrefixSlot('tab:second');
    await engine.saveGpuPrefixSlot('draft:third');
    const slotIdsAfterFill = engine.listGpuPrefixSlots().map((s) => s.slotId);
    expect(slotIdsAfterFill).toContain('base-system');
    expect(slotIdsAfterFill).toContain('draft:third');

    const forcedEngine = new WgslQwen35Engine();
    vi.spyOn(
      forcedEngine as unknown as { verifyRealKernels: () => Promise<void> },
      'verifyRealKernels',
    ).mockResolvedValue(undefined);
    const forcedStats = await forcedEngine.load(null, {
      modelId: 'test/tiny-qwen35',
      forceComputeTier: 'tier3-f32-universal' satisfies WgslComputeTier,
    });
    expect(forcedStats.computeTier).toBe('tier3-f32-universal');

    engine.dispose();
    forcedEngine.dispose();
    expect(destroyedBuffers.length).toBeGreaterThan(0);
  });
});
