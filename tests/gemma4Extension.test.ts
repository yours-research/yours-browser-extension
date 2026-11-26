import { describe, expect, it } from 'vitest';
import { mapGemma4PleShard } from '../src/offscreen/gemma4ExtensionAdapter.ts';
import { computeGemma4KvMemoryPlan } from '../src/offscreen/gemma4WgslEngine.ts';
import { buildIncrementalTurnPrompt } from '../src/offscreen/inferenceEngine.ts';
import { BUNDLED_MODEL_IDS, isGemma4ModelId } from '../src/shared/protocol.ts';

describe('Gemma 4 Extension Integration & Phase 3 Helpers', () => {
  it('identifies bundled model IDs accurately', () => {
    expect(BUNDLED_MODEL_IDS).toEqual([
      'onnx-community/Qwen3.5-4B-ONNX-OPT',
      'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
    ]);
    expect(isGemma4ModelId('onnx-community/gemma-4-E4B-it-qat-mobile-ONNX')).toBe(true);
    expect(isGemma4ModelId('onnx-community/Qwen3.5-4B-ONNX-OPT')).toBe(false);
  });

  it('computes INT4 global KV VRAM plan with 68 u32s/tok-head', () => {
    const int4Plan = computeGemma4KvMemoryPlan(131072, 'int4', 2_254_817_952, 25_200_000);
    const int8Plan = computeGemma4KvMemoryPlan(131072, 'int8', 2_254_817_952, 25_200_000);
    // 2 global buffers * 8 head-planes * 131072 * 68 * 4 = 570,425,344 B (+ 41,943,040 sliding = 612,368,384 B)
    expect(int4Plan.globalKvPerBufBytes).toBe(2 * 4 * 131072 * 68 * 4);
    expect(int4Plan.kvTotalBytes).toBeLessThan(int8Plan.kvTotalBytes);
    expect(int8Plan.totalVramMb - int4Plan.totalVramMb).toBeGreaterThan(480);
  });

  it('maps Gemma 4 PLE token IDs to shards ple_000..ple_255 cleanly', () => {
    expect(mapGemma4PleShard(0)).toEqual({
      shardIndex: 0,
      localIndex: 0,
      fileName: 'ple_shards/ple_000.bin',
      byteOffset: 0,
    });
    expect(mapGemma4PleShard(2050)).toEqual({
      shardIndex: 2,
      localIndex: 2,
      fileName: 'ple_shards/ple_002.bin',
      byteOffset: 2 * 2772,
    });
    expect(mapGemma4PleShard(258880).shardIndex).toBe(0);
    expect(mapGemma4PleShard(262143).fileName).toBe('ple_shards/ple_255.bin');
  });

  it('formats incremental turn prompts for both Qwen3.5 ChatML and Gemma 4 turn syntax', () => {
    const qwenSuffix = buildIncrementalTurnPrompt('<|im_start|>user\n...', [
      { role: 'user', content: 'Hello!' },
    ]);
    expect(qwenSuffix).toBe(
      '<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
    );

    const gemmaSuffix = buildIncrementalTurnPrompt('<|turn>user\n...', [
      { role: 'user', content: '  Follow up  ' },
    ]);
    expect(gemmaSuffix).toBe('<turn|>\n<|turn>user\nFollow up<turn|>\n<|turn>model\n');

    expect(buildIncrementalTurnPrompt('unknown template', [])).toBeNull();
  });
});
