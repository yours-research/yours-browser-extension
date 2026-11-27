import { describe, expect, it, beforeEach } from 'vitest';
import { AutoTokenizer } from '@huggingface/transformers';
import path from 'node:path';
import { decodeGeneratedText } from '../src/offscreen/inferenceEngine.ts';
import {
  IncrementalTokenDecoder,
  isValidSpecCalibration,
  loadSpecCalibration,
  saveSpecCalibration,
  shouldEnableEngineB,
} from '../src/offscreen/specCalibration.ts';

describe('specCalibration & break-even guard', () => {
  beforeEach(() => {
    const store = new Map<string, string>();
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => {
          store.set(k, v);
        },
        removeItem: (k: string) => {
          store.delete(k);
        },
      },
    });
  });

  it('enables Engine B only when headMs <= 0.35 * t1Ms', () => {
    expect(shouldEnableEngineB(15.1, 10.65)).toBe(false);
    expect(shouldEnableEngineB(3.8, 10.65)).toBe(false);
    expect(shouldEnableEngineB(3.72, 10.65)).toBe(true);
    expect(shouldEnableEngineB(0.4, 10.65)).toBe(true);
    expect(shouldEnableEngineB(null, 10.65)).toBe(false);
  });

  it('persists and reloads calibration scoped by (adapterName, dropinSha256)', async () => {
    await saveSpecCalibration({
      adapterName: 'nvidia|ampere|rtx3090',
      dropinSha256: 'sha-007b',
      t1Ms: 10.4,
      stepMs: [10.4, 13.0, 15.6, 18.2, 20.8, 23.4, 26.0, 28.6],
      headMs: 15.1,
      rejectPenaltyMs: 0.7,
      measuredAtIso: '2026-11-27T19:00:00.000Z',
    });

    const hit = await loadSpecCalibration('nvidia|ampere|rtx3090', 'sha-007b');
    expect(hit).not.toBeNull();
    expect(hit?.t1Ms).toBe(10.4);
    expect(hit?.headMs).toBe(15.1);
    expect(await loadSpecCalibration('nvidia|ampere|rtx3090', 'other-sha')).toBeNull();
    expect(isValidSpecCalibration({ adapterName: 'a' }, 'a', 'b')).toBe(false);
    // Reject stale pre-M11 calibration entries lacking current packRevision
    expect(
      isValidSpecCalibration(
        {
          adapterName: 'nvidia|ampere|rtx3090',
          dropinSha256: 'sha-007b',
          t1Ms: 10.65,
          stepMs: [10.65],
          headMs: 16.8,
          rejectPenaltyMs: 2.35,
          measuredAtIso: '2026-11-27T18:00:00.000Z',
        },
        'nvidia|ampere|rtx3090',
        'sha-007b',
      ),
    ).toBe(false);
  });

  it('IncrementalTokenDecoder matches decodeGeneratedText byte-for-byte on every token', async () => {
    const tokDir = path.resolve('dist/models/pragma-1-4b-rc1');
    const tok = await AutoTokenizer.from_pretrained(tokDir, { local_files_only: true });

    const cases = [
      {
        preOpenedThink: false,
        checkCumulative: true,
        text:
          'Comparing WebGPU vs Vulkan: 👋🏾 café naïve 日本語 🚀✨. ' +
          'Let us examine memory bandwidth bounds, subgroup matrix operations, and buffer binding alignment. '.repeat(3) +
          '<tool_call>\n{"name": "search_web_tab", "arguments": {"query": "Hugging Face Pro pricing"}}\n</tool_call>',
      },
      {
        preOpenedThink: true,
        checkCumulative: true,
        text: 'Step 1: check cache hits.\n</think>\nFinal TypeScript snippet:\n```ts\nexport const fast = (x: number) => x * 2;\n```<|im_end|>',
      },
      {
        preOpenedThink: true,
        checkCumulative: false,
        text: '   <think>\nExplicit think tag after spaces.\n</think>\nDone.',
      },
    ];

    for (const c of cases) {
      const ids = tok.encode(c.text, { add_special_tokens: false });
      const inc = new IncrementalTokenDecoder(tok, c.preOpenedThink);
      let emitted = '';
      for (let i = 0; i < ids.length; i++) {
        inc.appendToken(ids[i]!);
        const safeInc = inc.getSafeEmittedText();
        const full = decodeGeneratedText(tok, ids.slice(0, i + 1), {
          preOpenedThink: c.preOpenedThink,
        });
        const expectedSafe = full.endsWith('\uFFFD') ? full.slice(0, full.length - 1) : full;
        expect(safeInc).toBe(expectedSafe);
        if (safeInc.length > emitted.length) {
          emitted += safeInc.slice(emitted.length);
        }
      }
      if (c.checkCumulative) {
        expect(emitted).toBe(
          decodeGeneratedText(tok, ids, { preOpenedThink: c.preOpenedThink }).replace(/\uFFFD$/, ''),
        );
      }
    }
  });
});
