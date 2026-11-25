import { MODEL_PRESETS, type ChatMessage, type ModelDtype } from '../shared/protocol.ts';

export interface SamplingOptions {
  temperature: number;
  topK: number;
  topP: number;
}

let float16Lut: Float32Array | null = null;

function getFloat16Lut(): Float32Array {
  if (float16Lut) {
    return float16Lut;
  }
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) {
      lut[h] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 0x1f) {
      lut[h] = f ? NaN : s ? -Infinity : Infinity;
    } else {
      lut[h] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
  }
  float16Lut = lut;
  return lut;
}

export function decodeFloat16(u16: number): number {
  return getFloat16Lut()[u16 & 0xffff];
}

export function argmaxFloat16(u16Data: Uint16Array, offset: number, length: number): number {
  if (length <= 0) {
    throw new Error('Cannot compute argmax on empty float16 slice.');
  }
  const lut = getFloat16Lut();
  let bestIdx = 0;
  let bestVal = lut[u16Data[offset]];
  for (let i = 1; i < length; i++) {
    const v = lut[u16Data[offset + i]];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function argmaxNumeric(data: ArrayLike<number>, offset: number, length: number): number {
  if (length <= 0) {
    throw new Error('Cannot compute argmax on empty numeric slice.');
  }
  let bestIdx = 0;
  let bestVal = data[offset];
  for (let i = 1; i < length; i++) {
    const v = data[offset + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function argmaxFloat32(f32Data: Float32Array, offset: number, length: number): number {
  return argmaxNumeric(f32Data, offset, length);
}

export function decodeFloat16SliceToFloat32(
  u16Data: Uint16Array,
  offset: number,
  length: number,
  out: Float32Array = new Float32Array(length),
): Float32Array {
  const lut = getFloat16Lut();
  for (let i = 0; i < length; i++) {
    out[i] = lut[u16Data[offset + i]];
  }
  return out;
}

/**
 * Samples the next token ID from a 1D Float32Array of vocab logits.
 * Uses argmax when temperature <= 0, otherwise top-k + top-p nucleus sampling.
 */
export function sampleNextToken(logits: Float32Array, options: SamplingOptions): number {
  const vocabSize = logits.length;
  if (vocabSize === 0) {
    throw new Error('Cannot sample from empty logits array.');
  }

  if (options.temperature <= 0) {
    let bestIdx = 0;
    let bestVal = logits[0];
    for (let i = 1; i < vocabSize; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  const invTemp = 1 / options.temperature;
  const k = options.topK > 0 && options.topK < vocabSize ? options.topK : vocabSize;

  // Collect candidate indices (full sort only if topK < vocabSize, or partial top-k selection).
  let candidates: Array<{ id: number; scaled: number }>;
  if (k < vocabSize) {
    candidates = new Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      candidates[i] = { id: i, scaled: logits[i] * invTemp };
    }
    candidates.sort((a, b) => b.scaled - a.scaled);
    candidates.length = k;
  } else {
    candidates = new Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      candidates[i] = { id: i, scaled: logits[i] * invTemp };
    }
    candidates.sort((a, b) => b.scaled - a.scaled);
  }

  const maxLogit = candidates[0].scaled;
  const probs = new Float64Array(candidates.length);
  let sumExp = 0;
  for (let i = 0; i < candidates.length; i++) {
    const e = Math.exp(candidates[i].scaled - maxLogit);
    probs[i] = e;
    sumExp += e;
  }

  const topP = options.topP > 0 && options.topP < 1 ? options.topP : 1;
  let cutoff = candidates.length;
  let cumulative = 0;
  for (let i = 0; i < candidates.length; i++) {
    probs[i] /= sumExp;
    cumulative += probs[i];
    if (cumulative >= topP) {
      cutoff = i + 1;
      break;
    }
  }

  let keptSum = 0;
  for (let i = 0; i < cutoff; i++) {
    keptSum += probs[i];
  }

  let r = Math.random() * keptSum;
  for (let i = 0; i < cutoff; i++) {
    r -= probs[i];
    if (r <= 0) {
      return candidates[i].id;
    }
  }
  return candidates[cutoff - 1].id;
}

/**
 * Constructs contiguous position_ids for either 3D MRoPE (Qwen3.5 [3, 1, seqLen])
 * or standard 2D RoPE ([1, seqLen]), avoiding the upstream +dims[1] offset bug.
 */
export function buildPositionIdsData(
  startPos: number,
  seqLen: number,
  is3DRoPE: boolean,
): { data: BigInt64Array; dims: number[] } {
  if (is3DRoPE) {
    const data = new BigInt64Array(3 * seqLen);
    for (let j = 0; j < seqLen; j++) {
      const pos = BigInt(startPos + j);
      data[j] = pos;
      data[seqLen + j] = pos;
      data[2 * seqLen + j] = pos;
    }
    return { data, dims: [3, 1, seqLen] };
  }

  const data = new BigInt64Array(seqLen);
  for (let j = 0; j < seqLen; j++) {
    data[j] = BigInt(startPos + j);
  }
  return { data, dims: [1, seqLen] };
}

export interface CacheTensorDescriptor {
  dims: readonly number[];
  type: string;
}

export interface CacheStats {
  seqLength: number;
  kvCacheBytes: number;
  deltaNetBytes: number;
  totalCacheBytes: number;
  kvLayerCount: number;
  deltaNetLayerCount: number;
}

function bytesPerElement(dtype: string): number {
  switch (dtype) {
    case 'float32':
    case 'int32':
    case 'uint32':
      return 4;
    case 'float16':
    case 'bfloat16':
    case 'int16':
    case 'uint16':
      return 2;
    case 'int64':
    case 'uint64':
    case 'float64':
      return 8;
    default:
      return 1;
  }
}

export function extractCacheStats(
  cacheEntries: Record<string, CacheTensorDescriptor> | null | undefined,
): CacheStats {
  if (!cacheEntries) {
    return {
      seqLength: 0,
      kvCacheBytes: 0,
      deltaNetBytes: 0,
      totalCacheBytes: 0,
      kvLayerCount: 0,
      deltaNetLayerCount: 0,
    };
  }

  let seqLength = 0;
  let kvCacheBytes = 0;
  let deltaNetBytes = 0;
  const kvLayers = new Set<string>();
  const deltaNetLayers = new Set<string>();

  for (const [name, tensor] of Object.entries(cacheEntries)) {
    if (!tensor || !Array.isArray(tensor.dims)) {
      continue;
    }
    const elements = tensor.dims.reduce((acc, d) => acc * d, 1);
    const bytes = elements * bytesPerElement(tensor.type);

    if (name.startsWith('past_key_values.')) {
      kvCacheBytes += bytes;
      const layerIdx = name.split('.')[1] ?? name;
      kvLayers.add(layerIdx);
      if (seqLength === 0 && tensor.dims.length >= 2) {
        seqLength = tensor.dims[tensor.dims.length - 2];
      }
    } else if (
      name.startsWith('past_conv.') ||
      name.startsWith('past_recurrent.') ||
      name.startsWith('past_ssm.')
    ) {
      deltaNetBytes += bytes;
      const layerIdx = name.split('.')[1] ?? name;
      deltaNetLayers.add(layerIdx);
    }
  }

  return {
    seqLength,
    kvCacheBytes,
    deltaNetBytes,
    totalCacheBytes: kvCacheBytes + deltaNetBytes,
    kvLayerCount: kvLayers.size,
    deltaNetLayerCount: deltaNetLayers.size,
  };
}

export function getEstimatedWeightGB(modelId: string, dtype: ModelDtype): number {
  const preset = MODEL_PRESETS.find((p) => p.id === modelId);
  if (preset) {
    return preset.estimatedWeightGB[dtype] ?? preset.estimatedWeightGB[preset.defaultDtype];
  }
  return dtype === 'fp16' ? 8.0 : 2.67;
}

export function computeBandwidthStats(
  decodeTokPerSec: number,
  modelWeightGB: number,
  gpuBandwidthGBs: number,
): {
  achievedGBs: number;
  utilizationPct: number;
  recommendation: string;
} {
  const achievedGBs = Math.max(0, decodeTokPerSec) * Math.max(0, modelWeightGB);
  const utilizationPct = gpuBandwidthGBs > 0 ? (achievedGBs / gpuBandwidthGBs) * 100 : 0;
  const recommendation =
    utilizationPct < 40
      ? `Below 40% theoretical bandwidth (${utilizationPct.toFixed(1)}% of ${gpuBandwidthGBs} GB/s): Phase 2 Decision Gate triggers custom Rust/WASM + WGSL engine.`
      : `At ${utilizationPct.toFixed(1)}% of ${gpuBandwidthGBs} GB/s theoretical bandwidth (>= 40%): ONNX Runtime WebGPU baseline meets threshold.`;
  return { achievedGBs, utilizationPct, recommendation };
}

/**
 * Checks whether `next` extends `prev` monotonically (same prior role/content entries)
 * so cached hybrid DeltaNet + KV state can be reused incrementally without re-prefill.
 */
export function isMessagePrefix(prev: readonly ChatMessage[], next: readonly ChatMessage[]): boolean {
  if (prev.length === 0 || next.length <= prev.length) {
    return false;
  }
  for (let i = 0; i < prev.length; i++) {
    if (prev[i].role !== next[i].role || prev[i].content !== next[i].content) {
      return false;
    }
  }
  return true;
}
