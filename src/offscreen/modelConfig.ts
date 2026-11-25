import type { WgslKvMode } from '../shared/protocol.ts';

export interface ModelArchConfig {
  hiddenSize: number;
  intermediateSize: number;
  numLayers: number;
  vocabSize: number;
  fullAttnLayers: number[];
  dnKeyHeads: number;
  dnValHeads: number;
  dnHeadDim: number;
  dnConvKernel: number;
  attnQHeads: number;
  attnKvHeads: number;
  attnHeadDim: number;
  ropeDim: number;
  ropeTheta: number;
}

export const QWEN35_4B_DEFAULT_ARCH: ModelArchConfig = {
  hiddenSize: 2560,
  intermediateSize: 9216,
  numLayers: 32,
  vocabSize: 248320,
  fullAttnLayers: [3, 7, 11, 15, 19, 23, 27, 31],
  dnKeyHeads: 16,
  dnValHeads: 32,
  dnHeadDim: 128,
  dnConvKernel: 4,
  attnQHeads: 16,
  attnKvHeads: 4,
  attnHeadDim: 256,
  ropeDim: 64,
  ropeTheta: 10000000,
};

export function resolveModelArch(manifest?: { arch?: Partial<ModelArchConfig> }): ModelArchConfig {
  const partial = manifest?.arch;
  return {
    hiddenSize: partial?.hiddenSize ?? QWEN35_4B_DEFAULT_ARCH.hiddenSize,
    intermediateSize: partial?.intermediateSize ?? QWEN35_4B_DEFAULT_ARCH.intermediateSize,
    numLayers: partial?.numLayers ?? QWEN35_4B_DEFAULT_ARCH.numLayers,
    vocabSize: partial?.vocabSize ?? QWEN35_4B_DEFAULT_ARCH.vocabSize,
    fullAttnLayers: partial?.fullAttnLayers
      ? [...partial.fullAttnLayers]
      : [...QWEN35_4B_DEFAULT_ARCH.fullAttnLayers],
    dnKeyHeads: partial?.dnKeyHeads ?? QWEN35_4B_DEFAULT_ARCH.dnKeyHeads,
    dnValHeads: partial?.dnValHeads ?? QWEN35_4B_DEFAULT_ARCH.dnValHeads,
    dnHeadDim: partial?.dnHeadDim ?? QWEN35_4B_DEFAULT_ARCH.dnHeadDim,
    dnConvKernel: partial?.dnConvKernel ?? QWEN35_4B_DEFAULT_ARCH.dnConvKernel,
    attnQHeads: partial?.attnQHeads ?? QWEN35_4B_DEFAULT_ARCH.attnQHeads,
    attnKvHeads: partial?.attnKvHeads ?? QWEN35_4B_DEFAULT_ARCH.attnKvHeads,
    attnHeadDim: partial?.attnHeadDim ?? QWEN35_4B_DEFAULT_ARCH.attnHeadDim,
    ropeDim: partial?.ropeDim ?? QWEN35_4B_DEFAULT_ARCH.ropeDim,
    ropeTheta: partial?.ropeTheta ?? QWEN35_4B_DEFAULT_ARCH.ropeTheta,
  };
}

export function getFullAttnLayerSet(arch: ModelArchConfig): Set<number> {
  const set = new Set<number>();
  for (const l of arch.fullAttnLayers) {
    if (Number.isInteger(l) && l >= 0 && l < arch.numLayers) {
      set.add(l);
    }
  }
  return set;
}

export function getFullAttnLayerCount(arch: ModelArchConfig): number {
  return getFullAttnLayerSet(arch).size;
}

export function getDeltaNetLayerCount(arch: ModelArchConfig): number {
  return Math.max(0, arch.numLayers - getFullAttnLayerCount(arch));
}

export function getDeltaNetConvDim(arch: ModelArchConfig): number {
  return arch.dnKeyHeads * arch.dnHeadDim * 2 + arch.dnValHeads * arch.dnHeadDim;
}

export function getDeltaNetQkvzAbRows(arch: ModelArchConfig): number {
  return getDeltaNetConvDim(arch) + arch.dnValHeads * arch.dnHeadDim + arch.dnValHeads * 2;
}

export function getFullAttnQkvRows(arch: ModelArchConfig): number {
  return arch.attnQHeads * arch.attnHeadDim * 2 + arch.attnKvHeads * arch.attnHeadDim * 2;
}

export function getFullAttnKvHeadU32s(arch: ModelArchConfig, kvMode: WgslKvMode): number {
  return kvMode === 'int8' ? arch.attnHeadDim / 4 + 1 : arch.attnHeadDim / 2;
}

export function q4RowU32s(cols: number): number {
  return Math.floor(cols / 32) * 5;
}

export function q4MatrixBytes(rows: number, cols: number): number {
  return rows * q4RowU32s(cols) * 4;
}

export function computeLayerWeightByteSizes(arch: ModelArchConfig): number[] {
  const fullAttnSet = getFullAttnLayerSet(arch);
  const dnConvDim = getDeltaNetConvDim(arch);
  const dnValDim = arch.dnValHeads * arch.dnHeadDim;
  const attnQDim = arch.attnQHeads * arch.attnHeadDim * 2;
  const attnKvDim = arch.attnKvHeads * arch.attnHeadDim;
  const attnODim = arch.attnQHeads * arch.attnHeadDim;
  const mlpBytes =
    q4MatrixBytes(arch.intermediateSize, arch.hiddenSize) * 2 +
    q4MatrixBytes(arch.hiddenSize, arch.intermediateSize);

  const sizes: number[] = [];
  for (let l = 0; l < arch.numLayers; l++) {
    if (!fullAttnSet.has(l)) {
      const dnBytes =
        q4MatrixBytes(dnConvDim, arch.hiddenSize) +
        q4MatrixBytes(dnValDim, arch.hiddenSize) +
        q4MatrixBytes(arch.dnValHeads * 2, arch.hiddenSize) +
        q4MatrixBytes(arch.hiddenSize, dnValDim);
      sizes.push(dnBytes + mlpBytes);
    } else {
      const attnBytes =
        q4MatrixBytes(attnQDim, arch.hiddenSize) +
        q4MatrixBytes(attnKvDim, arch.hiddenSize) * 2 +
        q4MatrixBytes(arch.hiddenSize, attnODim);
      sizes.push(attnBytes + mlpBytes);
    }
  }
  return sizes;
}

export interface WeightSlabPartition {
  slabIdx: number;
  startLayer: number;
  endLayer: number; // exclusive
  totalBytes: number;
}

const ONE_GIB_BYTES = 1024 * 1024 * 1024;

/**
 * Partitions transformer layers `0..numLayers-1` into contiguous GPU buffers (`>= 2` slabs
 * when `numLayers >= 2`, scaling dynamically 2, 4, 8, ... as needed) so every slab stays
 * `<= Math.min(1024 * 1024 * 1024, maxBindingBytes)`.
 */
export function partitionWeightLayersIntoSlabs(
  layerByteSizes: number[],
  maxBindingBytes: number,
): WeightSlabPartition[] {
  const n = layerByteSizes.length;
  if (n === 0) {
    return [];
  }
  const safeMaxBinding = Number.isFinite(maxBindingBytes) && maxBindingBytes > 0
    ? maxBindingBytes
    : ONE_GIB_BYTES;
  const cap = Math.min(ONE_GIB_BYTES, safeMaxBinding);

  for (let i = 0; i < n; i++) {
    if (layerByteSizes[i]! > cap) {
      throw new Error(
        `Layer ${i} weight size (${layerByteSizes[i]} bytes) exceeds max binding cap (${cap} bytes).`,
      );
    }
  }

  if (n === 1) {
    return [{ slabIdx: 0, startLayer: 0, endLayer: 1, totalBytes: layerByteSizes[0]! }];
  }

  // Prefer balanced power-of-2 slab counts (2, 4, 8, ...) so GPU slabs align cleanly
  // with half-model disk shard boundaries when possible.
  for (let numSlabs = 2; numSlabs <= n; numSlabs *= 2) {
    const candidate: WeightSlabPartition[] = [];
    let fits = true;
    for (let s = 0; s < numSlabs; s++) {
      const startLayer = Math.floor((s * n) / numSlabs);
      const endLayer = Math.floor(((s + 1) * n) / numSlabs);
      let totalBytes = 0;
      for (let l = startLayer; l < endLayer; l++) {
        totalBytes += layerByteSizes[l]!;
      }
      if (endLayer <= startLayer || totalBytes > cap) {
        fits = false;
        break;
      }
      candidate.push({ slabIdx: s, startLayer, endLayer, totalBytes });
    }
    if (fits) {
      return candidate;
    }
  }

  // Fallback greedy contiguous packing when layer sizes are uneven and even splits don't fit
  const greedy: WeightSlabPartition[] = [];
  let start = 0;
  while (start < n) {
    let total = 0;
    let end = start;
    while (end < n && total + layerByteSizes[end]! <= cap) {
      total += layerByteSizes[end]!;
      end++;
    }
    greedy.push({ slabIdx: greedy.length, startLayer: start, endLayer: end, totalBytes: total });
    start = end;
  }

  if (greedy.length === 1 && n >= 2) {
    const mid = Math.floor(n / 2);
    let b0 = 0;
    for (let l = 0; l < mid; l++) b0 += layerByteSizes[l]!;
    let b1 = 0;
    for (let l = mid; l < n; l++) b1 += layerByteSizes[l]!;
    return [
      { slabIdx: 0, startLayer: 0, endLayer: mid, totalBytes: b0 },
      { slabIdx: 1, startLayer: mid, endLayer: n, totalBytes: b1 },
    ];
  }

  return greedy;
}

export interface VocabSlabSlice {
  slabIdx: number;
  startRow: number;
  endRow: number; // exclusive
  rowCount: number;
  byteSize: number;
}

/**
 * Slices `lm_head` / tied embedding rows into contiguous GPU buffers when `slab2Bytes`
 * exceeds `Math.min(1 GiB, maxBindingBytes)` (e.g. 256 MiB storage buffer binding limit).
 */
export function partitionVocabRowsIntoSlabs(
  vocabSize: number,
  hiddenSize: number,
  maxBindingBytes: number,
): VocabSlabSlice[] {
  const rowBytes = q4RowU32s(hiddenSize) * 4;
  const safeMaxBinding = Number.isFinite(maxBindingBytes) && maxBindingBytes > 0
    ? maxBindingBytes
    : ONE_GIB_BYTES;
  const cap = Math.min(ONE_GIB_BYTES, safeMaxBinding);
  const totalBytes = vocabSize * rowBytes;

  if (totalBytes <= cap) {
    return [
      {
        slabIdx: 0,
        startRow: 0,
        endRow: vocabSize,
        rowCount: vocabSize,
        byteSize: totalBytes,
      },
    ];
  }

  const rawMaxRows = Math.floor(cap / rowBytes);
  const maxRowsAligned = rawMaxRows >= 16 ? rawMaxRows & ~15 : Math.max(1, rawMaxRows);
  if (maxRowsAligned <= 0) {
    throw new Error(`Single vocab row (${rowBytes} bytes) exceeds binding cap (${cap} bytes).`);
  }

  const slices: VocabSlabSlice[] = [];
  for (let startRow = 0; startRow < vocabSize; startRow += maxRowsAligned) {
    const endRow = Math.min(vocabSize, startRow + maxRowsAligned);
    const rowCount = endRow - startRow;
    slices.push({
      slabIdx: slices.length,
      startRow,
      endRow,
      rowCount,
      byteSize: rowCount * rowBytes,
    });
  }
  return slices;
}

export interface TieredKvPlan {
  initialCapacity: number;
  maxSafeCapacity: number;
  tiers: number[];
}

const KV_HEADROOM_TOKENS = 512;
const KV_BASE_TIERS = [16384, 32768, 65536, 131072];

/**
 * Computes an initial compact KV allocation and monotonic growth tiers bounded by
 * hardware `maxBufferSize`, `maxStorageBufferBindingSize`, and optional `deviceMemoryGb`.
 */
export function computeTieredKvPlan(
  requestedMaxContext: number,
  arch: ModelArchConfig,
  kvMode: WgslKvMode,
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number },
  deviceMemoryGb?: number,
): TieredKvPlan {
  const faLayers = Math.max(1, getFullAttnLayerCount(arch));
  const maxLayersPerSlab = Math.max(1, Math.ceil(faLayers / 2));
  const rowBytes = getFullAttnKvHeadU32s(arch, kvMode) * 4;
  const bytesPerTokenPerSlab = maxLayersPerSlab * 2 * arch.attnKvHeads * rowBytes;

  const maxBuffer = limits.maxBufferSize > 0 ? limits.maxBufferSize : 256 * 1024 * 1024;
  const maxBinding =
    limits.maxStorageBufferBindingSize > 0 ? limits.maxStorageBufferBindingSize : 128 * 1024 * 1024;
  const slabLimitBytes = Math.min(maxBuffer, maxBinding);

  let hardwareMaxCap = Math.max(512, Math.floor(slabLimitBytes / bytesPerTokenPerSlab));
  if (deviceMemoryGb !== undefined && Number.isFinite(deviceMemoryGb) && deviceMemoryGb > 0) {
    // Chrome caps navigator.deviceMemory at 8 GiB even on >=16 GiB workstations:
    // keep a strict 25% KV ceiling on <8 GiB laptops/tablets and 30% on 8+ GiB systems
    // so >=2 GiB GPUs can still unlock full 128K context while starting lean at 16K.
    const ratio = deviceMemoryGb < 8 ? 0.25 : 0.30;
    const kvBudgetBytes = deviceMemoryGb * 1024 * 1024 * 1024 * ratio;
    const memCap = Math.max(512, Math.floor(kvBudgetBytes / (2 * bytesPerTokenPerSlab)));
    hardwareMaxCap = Math.min(hardwareMaxCap, memCap);
  }

  const safeReq = Math.max(512, requestedMaxContext || 131072);
  const requestedWithHeadroom = safeReq + KV_HEADROOM_TOKENS;
  const maxSafeCapacity = Math.min(requestedWithHeadroom, hardwareMaxCap);

  const constrainedMemory =
    (deviceMemoryGb !== undefined && deviceMemoryGb <= 8) ||
    maxBuffer <= 1.2 * ONE_GIB_BYTES;
  const initialBase = constrainedMemory ? 16384 : 32768;
  const initialTarget = initialBase + KV_HEADROOM_TOKENS;
  const initialCapacity = Math.min(maxSafeCapacity, initialTarget);

  const tiers: number[] = [initialCapacity];
  for (const baseTier of KV_BASE_TIERS) {
    const withHeadroom = baseTier + KV_HEADROOM_TOKENS;
    if (withHeadroom > initialCapacity && withHeadroom <= maxSafeCapacity) {
      tiers.push(withHeadroom);
    }
  }
  if (tiers[tiers.length - 1]! < maxSafeCapacity) {
    tiers.push(maxSafeCapacity);
  }

  return {
    initialCapacity,
    maxSafeCapacity,
    tiers,
  };
}

/**
 * Computes the exact byte offset of a single `(localLayerIdx, planeIdx, kvHeadIdx)`
 * sequence buffer within `kvSlab0` or `kvSlab1` for a given `maxSeqCapacity`.
 */
export function computeKvSliceByteOffset(
  localLayerIdx: number,
  planeIdx: 0 | 1,
  kvHeadIdx: number,
  maxSeqCapacity: number,
  arch: ModelArchConfig,
  kvMode: WgslKvMode,
): number {
  const rowBytes = getFullAttnKvHeadU32s(arch, kvMode) * 4;
  const headStrideBytes = maxSeqCapacity * rowBytes;
  const planeBytes = arch.attnKvHeads * headStrideBytes;
  const layerBytes = 2 * planeBytes;
  return localLayerIdx * layerBytes + planeIdx * planeBytes + kvHeadIdx * headStrideBytes;
}
