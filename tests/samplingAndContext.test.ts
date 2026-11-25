import { describe, expect, it } from 'vitest';
import {
  buildNeedleInHaystackPrompt,
  buildSyntheticTokenIds,
  verifyNeedleAnswer,
  type MinimalTokenizer,
} from '../src/offscreen/benchmarks.ts';
import {
  argmaxFloat16,
  argmaxFloat32,
  argmaxNumeric,
  buildPositionIdsData,
  computeBandwidthStats,
  decodeFloat16,
  decodeFloat16SliceToFloat32,
  extractCacheStats,
  isMessagePrefix,
  sampleNextToken,
} from '../src/offscreen/sampling.ts';

function makeWordTokenizer(): MinimalTokenizer {
  const vocab = new Map<string, number>();
  const rev = new Map<number, string>();
  let nextId = 10;

  const getOrAdd = (tok: string): number => {
    const existing = vocab.get(tok);
    if (existing !== undefined) return existing;
    const id = nextId++;
    vocab.set(tok, id);
    rev.set(id, tok);
    return id;
  };

  return {
    encode(text: string) {
      return text
        .split(/\s+/)
        .filter(Boolean)
        .map((w) => getOrAdd(w));
    },
    decode(ids: number[]) {
      return ids.map((id) => rev.get(id) ?? '').join(' ');
    },
  };
}

describe('sampling, position_ids, cache telemetry, and benchmarks', () => {
  it('builds contiguous 3D MRoPE [3, 1, seqLen] position_ids across prefill and decode offsets', () => {
    const prefillPos = buildPositionIdsData(512, 4, true);
    expect(prefillPos.dims).toEqual([3, 1, 4]);
    expect(Array.from(prefillPos.data)).toEqual([
      512n, 513n, 514n, 515n,
      512n, 513n, 514n, 515n,
      512n, 513n, 514n, 515n,
    ]);

    const decodePos = buildPositionIdsData(32768, 1, true);
    expect(decodePos.dims).toEqual([3, 1, 1]);
    expect(Array.from(decodePos.data)).toEqual([32768n, 32768n, 32768n]);

    const rope2D = buildPositionIdsData(10, 3, false);
    expect(rope2D.dims).toEqual([1, 3]);
    expect(Array.from(rope2D.data)).toEqual([10n, 11n, 12n]);
  });

  it('samples greedily at temperature 0 and respects top-k/top-p when temperature > 0', () => {
    const logits = new Float32Array([-2.0, 0.5, 4.25, 1.0]);
    expect(sampleNextToken(logits, { temperature: 0, topK: 20, topP: 0.95 })).toBe(2);

    const picked = sampleNextToken(new Float32Array([-100, -100, 10, -100]), {
      temperature: 0.7,
      topK: 1,
      topP: 0.95,
    });
    expect(picked).toBe(2);

    // Verify float16 LUT decoding & greedy argmax helpers
    // 0x3c00 = 1.0, 0x4000 = 2.0, 0xc000 = -2.0, 0x3555 ~ 0.33325
    expect(decodeFloat16(0x3c00)).toBe(1.0);
    expect(decodeFloat16(0xc000)).toBe(-2.0);
    const f16Buf = new Uint16Array([0xc000, 0x3555, 0x4000, 0x3c00, 0x0000, 0x4200, 0x3800]);
    expect(argmaxFloat16(f16Buf, 0, 4)).toBe(2);
    expect(argmaxFloat16(f16Buf, 4, 3)).toBe(1); // 0x4200 = 3.0
    const decoded = decodeFloat16SliceToFloat32(f16Buf, 0, 4);
    expect(Array.from(decoded)).toEqual([-2, decodeFloat16(0x3555), 2, 1]);
    expect(argmaxFloat32(new Float32Array([-5, 1.5, 9.25, 3.0]), 0, 4)).toBe(2);
    expect(argmaxNumeric([-3.5, 0.25, 4.5, -1.0], 0, 4)).toBe(2);
  });

  it('computes Qwen3.5 hybrid KV + DeltaNet cache memory accurately at 32K context', () => {
    const entries: Record<string, { dims: number[]; type: string }> = {};
    // 8 full attention layers: [1, 4 kv heads, 32768 seq, 256 head_dim] float16
    for (const layer of [3, 7, 11, 15, 19, 23, 27, 31]) {
      entries[`past_key_values.${layer}.key`] = { dims: [1, 4, 32768, 256], type: 'float16' };
      entries[`past_key_values.${layer}.value`] = { dims: [1, 4, 32768, 256], type: 'float16' };
    }
    // 24 DeltaNet linear attention layers: conv [1, 8192, 4] + recurrent [1, 32, 128, 128] float32
    for (let i = 0; i < 32; i++) {
      if ([3, 7, 11, 15, 19, 23, 27, 31].includes(i)) continue;
      entries[`past_conv.${i}`] = { dims: [1, 8192, 4], type: 'float32' };
      entries[`past_recurrent.${i}`] = { dims: [1, 32, 128, 128], type: 'float32' };
    }

    const stats = extractCacheStats(entries);
    expect(stats.seqLength).toBe(32768);
    expect(stats.kvLayerCount).toBe(8);
    expect(stats.deltaNetLayerCount).toBe(24);
    expect(stats.kvCacheBytes).toBe(1024 * 1024 * 1024); // exactly 1 GiB
    expect(stats.deltaNetBytes).toBe(24 * (8192 * 4 + 32 * 128 * 128) * 4);
  });

  it('computes Phase 2 Decision Gate bandwidth utilization and prefix detection', () => {
    const lowBw = computeBandwidthStats(40, 2.67, 936);
    expect(lowBw.utilizationPct).toBeCloseTo((40 * 2.67 * 100) / 936, 2);
    expect(lowBw.recommendation).toContain('Below 40%');

    const highBw = computeBandwidthStats(160, 2.67, 936);
    expect(highBw.utilizationPct).toBeGreaterThan(40);
    expect(highBw.recommendation).toContain('>= 40%');

    expect(
      isMessagePrefix(
        [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: 'hello' },
        ],
        [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: 'hello' },
          { role: 'user', content: 'next' },
        ],
      ),
    ).toBe(true);

    expect(
      isMessagePrefix(
        [{ role: 'user', content: 'hi' }],
        [{ role: 'user', content: 'edited' }],
      ),
    ).toBe(false);
  });

  it('builds synthetic prompts and Needle-in-a-Haystack prompts at exact target scale', () => {
    const tok = makeWordTokenizer();
    const synth = buildSyntheticTokenIds(tok, 1024);
    expect(synth.length).toBe(1024);

    const niah = buildNeedleInHaystackPrompt(tok, 500, 0.5, 'VELVET-ORBIT-7492');
    expect(niah.actualTokens).toBe(500);
    expect(niah.needleTokenIndex).toBeGreaterThan(150);
    expect(niah.needleTokenIndex).toBeLessThan(350);

    const decoded = tok.decode(niah.tokenIds);
    expect(decoded).toContain('VELVET-ORBIT-7492');
    expect(verifyNeedleAnswer('The passkey is velvet-orbit-7492.', 'VELVET-ORBIT-7492')).toBe(true);
    expect(verifyNeedleAnswer('I do not know.', 'VELVET-ORBIT-7492')).toBe(false);
  });
});
