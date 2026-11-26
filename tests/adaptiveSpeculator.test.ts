import { describe, expect, it } from 'vitest';
import {
  AdaptiveVerifyCostModel,
  EngineASpeculator,
  SuffixAutomaton,
  buildToolCorpus,
  chooseBudget,
  detectDialect,
  grammarContinuations,
  parseToolsFromPrompt,
  type EngineATokenizer,
} from '../src/offscreen/adaptiveSpeculator.ts';

function makeCharWordTokenizer(): EngineATokenizer {
  const strToId = new Map<string, number>();
  const idToStr: string[] = [];
  const encodePiece = (s: string): number => {
    let id = strToId.get(s);
    if (id === undefined) {
      id = idToStr.length;
      strToId.set(s, id);
      idToStr.push(s);
    }
    return id;
  };
  return {
    encode: (text: string) => {
      const pieces = text.match(/<\|?\/?[a-zA-Z0-9_]+>|<\/?[a-zA-Z0-9_=]+>|[A-Za-z0-9_]+|\s+|[^\sA-Za-z0-9_]/g) ?? [];
      return pieces.map(encodePiece);
    },
    decode: (tokens: readonly number[]) => tokens.map((t) => idToStr[t] ?? '').join(''),
  };
}

describe('EngineASpeculator & AdaptiveVerifyCostModel', () => {
  it('detects XML and JSON tool call dialects and parses <tools> schemas', () => {
    const prompt = `<tools>
{"type":"function","function":{"name":"read_tab","parameters":{"type":"object","properties":{"tab_id":{"type":"integer"},"mode":{"type":"string","enum":["auto","article"]}},"required":["tab_id"]}}}
</tools>`;
    const parsed = parseToolsFromPrompt(prompt);
    expect(parsed).toHaveLength(1);
    expect(parsed[0]!.name).toBe('read_tab');
    expect(parsed[0]!.params[0]!.required).toBe(true);
    expect(detectDialect(prompt)).toBe('xml');
    expect(detectDialect('Emit <tool_call>{"name": "foo", "arguments": {}}</tool_call>')).toBe('json');
  });

  it('produces forced grammar continuations for XML and JSON tool calls', () => {
    const tools = [
      {
        name: 'web_search',
        params: [{ name: 'query', type: 'string' as const, required: true }],
      },
    ];
    const xmlConts = grammarContinuations('<tool_call>\n<function=web', tools, 'xml', true);
    expect(xmlConts[0]!.text).toContain('_search>\n<parameter=');
    const jsonConts = grammarContinuations('<tool_call>{"name": "web', tools, 'json', true);
    expect(jsonConts[0]!.text).toContain('_search", "arguments": {"');
  });

  it('returns K = 0 when path probabilities are below cost-model breakeven', () => {
    const model = new AdaptiveVerifyCostModel({ initialT1Ms: 10.9, initialSlopeMs: 4.5, initialRejectPenaltyMs: 1.0 });
    // Cold / low probability: p = 0.2 -> extra verify cost cannot pay
    const low = chooseBudget([0.2, 0.1, 0.05], model, 7, 0.04);
    expect(low.nodes).toBe(0);

    // High confidence structural / copy span: p = 0.94..0.88 -> multi-token draft pays
    const high = chooseBudget([0.96, 0.94, 0.92, 0.9, 0.88, 0.86, 0.84], model, 7, 0.04);
    expect(high.nodes).toBeGreaterThanOrEqual(4);
    expect(high.nodes).toBeLessThanOrEqual(7);
  });

  it('adapts online verify cost slope from live GPU step timings', () => {
    const model = new AdaptiveVerifyCostModel({ initialT1Ms: 10, initialSlopeMs: 7.5 });
    const before = model.stepMs(5);
    for (let i = 0; i < 40; i++) {
      model.recordStepTiming(5, 22, false); // slope ~3ms/tok
    }
    const after = model.stepMs(5);
    expect(after).toBeLessThan(before);
    expect(Math.abs(after - 22)).toBeLessThan(0.25);
  });

  it('proposes copy + corpus + grammar drafts up to MAX_SPEC_DRAFT = 7 cleanly', () => {
    const tok = makeCharWordTokenizer();
    const cost = new AdaptiveVerifyCostModel({ initialT1Ms: 10.9, initialSlopeMs: 3.5 });
    const engine = new EngineASpeculator({
      tokenizer: tok,
      corpus: buildToolCorpus(),
      adaptiveProfile: cost,
    });

    const sourcePassage = 'WebGPU speculative verification eliminates per-row argmax copies and writes 64-byte results. ';
    const prompt = `<|im_start|>user\nQuote this verbatim:\n${sourcePassage}<|im_end|>\n<|im_start|>assistant\n`;
    engine.beginSample(prompt, false);
    engine.observe(tok.encode(prompt), false);

    // Empty repeated suffix -> K = 0 when cold
    const cold = engine.proposeChain(7);
    expect(cold.tokens.length).toBeLessThanOrEqual(4);

    // Once assistant begins echoing the passage, SAM+Anchor immediately proposes a multi-token chain
    engine.observe(tok.encode('WebGPU speculative verification eliminates'), true);
    const copyDraft = engine.proposeChain(7);
    expect(copyDraft.tokens.length).toBeGreaterThanOrEqual(2);
    expect(copyDraft.tokens.length).toBeLessThanOrEqual(7);
    const draftedStr = tok.decode(copyDraft.tokens);
    expect(draftedStr).toContain('per');

    // Online Beta-Bernoulli calibration scales K(t) up as accepted copy spans grow
    let cur = copyDraft;
    for (let step = 0; step < 12; step++) {
      engine.feedback(cur, cur.tokens.length);
      cur = engine.proposeChain(7);
    }
    expect(cur.tokens.length).toBeGreaterThan(copyDraft.tokens.length);
    expect(cur.tokens.length).toBeLessThanOrEqual(7);
    expect(tok.decode(cur.tokens)).toContain(' per-');

    engine.endSample();
    expect(engine.calibrator.snapshot()).not.toEqual({});
  });

  it('SuffixAutomaton finds longest repeated suffix accurately', () => {
    const sam = new SuffixAutomaton(64);
    sam.extendAll([1, 2, 3, 4, 5, 9, 1, 2, 3]);
    const match = sam.longestRepeatedSuffix(2);
    expect(match).not.toBeNull();
    expect(match!.len).toBe(3);
    const conts = sam.continuations(match!.state, 2);
    expect(conts[0]![0]).toBe(4);
  });
});
