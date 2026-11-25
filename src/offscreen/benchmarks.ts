export interface MinimalTokenizer {
  encode: (text: string, options?: { add_special_tokens?: boolean }) => number[];
  decode: (ids: number[], options?: { skip_special_tokens?: boolean }) => string;
  apply_chat_template?: (
    messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
    options: { tokenize: false; add_generation_prompt?: boolean; [key: string]: unknown },
  ) => unknown;
}

const HAYSTACK_PARAGRAPHS = [
  'Modern web browsers execute GPU compute workloads through WebGPU command encoders, separating buffer staging from storage bindings while enforcing adapter-level memory limits across compute passes.',
  'Hybrid sequence architectures interleave recurrent gated linear attention blocks with periodic full-attention layers so memory growth stays bounded over tens of thousands of tokens.',
  'Origin Private File System storage exposes random-access file handles inside extension offscreen documents, allowing multi-gigabyte model weight shards to persist across browser restarts.',
  'Streaming tokenizers convert integer vocabulary identifiers into incremental UTF-8 character sequences while handling multi-byte boundaries across consecutive decoding steps.',
  'Chunked prefill partitions long input prompts into bounded compute slices to avoid OS GPU watchdog resets such as Windows Timeout Detection and Recovery during large matrix multiplications.',
];

/**
 * Generates an exact-length synthetic token ID array for prefill/decode benchmarking.
 */
export function buildSyntheticTokenIds(
  tokenizer: MinimalTokenizer,
  targetTokenCount: number,
): number[] {
  const seedCorpus = `${HAYSTACK_PARAGRAPHS.join(' ')} `;
  const baseIds = tokenizer.encode(seedCorpus, { add_special_tokens: false });
  if (baseIds.length === 0) {
    throw new Error('Tokenizer returned 0 tokens for benchmark corpus.');
  }

  const out = new Array<number>(targetTokenCount);
  for (let i = 0; i < targetTokenCount; i++) {
    out[i] = baseIds[i % baseIds.length];
  }
  return out;
}

export interface NeedlePromptBuildResult {
  promptText: string;
  tokenIds: number[];
  passkey: string;
  actualTokens: number;
  needleTokenIndex: number;
}

/**
 * Builds a Needle-in-a-Haystack prompt targeting `targetContextTokens` with a deterministic
 * passkey embedded at `depthRatio` (0.0 .. 1.0).
 */
export function buildNeedleInHaystackPrompt(
  tokenizer: MinimalTokenizer,
  targetContextTokens: number,
  depthRatio: number,
  passkey = 'VELVET-ORBIT-7492',
): NeedlePromptBuildResult {
  const clampedDepth = Math.max(0, Math.min(1, depthRatio));
  const needleSentence = `\n[CONFIDENTIAL AUDIT RECORD] The secret verification passkey for TinyGPT is ${passkey}. Remember this exact passkey.\n`;
  const questionText =
    'Read the entire log above carefully. What is the exact secret verification passkey for TinyGPT? Reply with only the passkey.';

  let prefixText = '';
  let suffixText = '';
  if (typeof tokenizer.apply_chat_template === 'function') {
    const sentinel = '__NIAH_BODY__';
    const templated = String(
      tokenizer.apply_chat_template(
        [{ role: 'user', content: `${sentinel}\n\n${questionText}` }],
        {
          tokenize: false,
          add_generation_prompt: true,
          enable_thinking: false,
        },
      ),
    );
    const splitIdx = templated.indexOf(sentinel);
    if (splitIdx !== -1) {
      prefixText = templated.slice(0, splitIdx);
      suffixText = templated.slice(splitIdx + sentinel.length);
    } else {
      prefixText = templated;
    }
  } else {
    prefixText = '<|im_start|>user\n';
    suffixText = `\n\n${questionText}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  }

  const prefixIds = tokenizer.encode(prefixText, { add_special_tokens: false });
  const needleIds = tokenizer.encode(needleSentence, { add_special_tokens: false });
  const suffixIds = tokenizer.encode(suffixText, { add_special_tokens: false });

  const overhead = prefixIds.length + needleIds.length + suffixIds.length;
  const fillerBudget = Math.max(64, targetContextTokens - overhead);
  const beforeCount = Math.floor(fillerBudget * clampedDepth);
  const afterCount = fillerBudget - beforeCount;

  const fillerIds = buildSyntheticTokenIds(tokenizer, fillerBudget);
  const beforeIds = fillerIds.slice(0, beforeCount);
  const afterIds = fillerIds.slice(beforeCount, beforeCount + afterCount);

  const tokenIds = [...prefixIds, ...beforeIds, ...needleIds, ...afterIds, ...suffixIds];
  const needleTokenIndex = prefixIds.length + beforeIds.length;

  return {
    promptText: `${prefixText}[...${beforeIds.length} tokens...]${needleSentence}[...${afterIds.length} tokens...]${suffixText}`,
    tokenIds,
    passkey,
    actualTokens: tokenIds.length,
    needleTokenIndex,
  };
}

export function verifyNeedleAnswer(responseText: string, passkey: string): boolean {
  const normalizedOutput = responseText.toUpperCase().replace(/\s+/g, '');
  const normalizedNeedle = passkey.toUpperCase().replace(/\s+/g, '');
  return normalizedOutput.includes(normalizedNeedle);
}
