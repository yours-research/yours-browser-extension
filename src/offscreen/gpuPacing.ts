import type { GpuPacingMode } from '../shared/protocol.ts';

/**
 * Policy parameters governing WebGPU command pacing, prefill chunking,
 * per-decode layer splitting, and UI breathers to prevent desktop frame freezes.
 */
export interface GpuPacingPolicy {
  mode: GpuPacingMode;
  prefillChunkShort: number;
  prefillChunkLong: number;
  prefillLongThreshold: number;
  prefillBurstBudgetMs: number;
  prefillBreatherMs: number;
  decodeSplitsShort: 1 | 2 | 4;
  decodeSplitsLong: 1 | 2 | 4;
  decodeLongThreshold: number;
  decodeSlowStepThresholdMs: number;
  decodeSliceBreatherMs: number;
  decodeBurstBudgetMs: number;
  decodePostStepRatio: number;
  decodePostStepMinMs: number;
  decodePostStepMaxMs: number;
  loadChunkYieldMs: number;
}

const GPU_PACING_POLICIES: Record<GpuPacingMode, GpuPacingPolicy> = {
  'max-speed': {
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
  },
  balanced: {
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
  },
  'smooth-ui': {
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
  },
};

/** Returns a shallow copy of the GPU pacing policy for the given mode. */
export function getGpuPacingPolicy(mode: GpuPacingMode): GpuPacingPolicy {
  const policy = GPU_PACING_POLICIES[mode] ?? GPU_PACING_POLICIES.balanced;
  return { ...policy };
}

/**
 * Calibrates a pacing policy for the active monitor refresh rate (`displayHz`) and
 * observed GPU decode step duration (`measuredStepMs`), preserving smooth compositor
 * frames on 60/120/144Hz displays and preventing Windows WDDM 2s TDR hangs on slow iGPUs.
 */
export function calibratePacingForHardware(
  mode: GpuPacingMode,
  displayHz: number,
  measuredStepMs: number,
): GpuPacingPolicy {
  const base = getGpuPacingPolicy(mode);
  if (base.mode === 'max-speed') {
    return base;
  }

  const safeHz =
    Number.isFinite(displayHz) && displayHz > 0 ? Math.max(30, Math.min(240, displayHz)) : 60;
  const frameMs = 1000 / safeHz;
  const safeStepMs = Number.isFinite(measuredStepMs) && measuredStepMs > 0 ? measuredStepMs : 0;

  // In balanced mode on Dedicated Worker, single-submit decode runs at ~80 tok/s while keeping
  // the browser compositor at 140+ FPS / 0% jank. Never tighten the 14 ms threshold below baseline
  // on 120/144Hz monitors when individual GPU steps are already fast (< 22 ms).
  let decodeBurstBudgetMs =
    base.mode === 'balanced'
      ? base.decodeBurstBudgetMs
      : Math.min(base.decodeBurstBudgetMs, Math.max(4, Math.round(frameMs * 0.75)));
  let decodeSlowStepThresholdMs = base.decodeSlowStepThresholdMs;

  let decodeSplitsShort: 1 | 2 | 4 = base.decodeSplitsShort;
  let decodeSplitsLong: 1 | 2 | 4 = base.decodeSplitsLong;
  let prefillChunkShort = base.prefillChunkShort;
  let prefillChunkLong = base.prefillChunkLong;
  let prefillBurstBudgetMs =
    base.mode === 'balanced' && safeStepMs > 0 && safeStepMs < 22 ? 24 : base.prefillBurstBudgetMs;
  let prefillBreatherMs =
    base.mode === 'balanced' && safeStepMs > 0 && safeStepMs < 22 ? 1 : base.prefillBreatherMs;

  if (safeStepMs >= 100) {
    // Very slow iGPU / Tier 3 fallback: split decode aggressively and shrink prefill chunks
    // to stay well below Windows WDDM 2-second TDR timeouts.
    decodeSplitsShort = base.mode === 'smooth-ui' ? 4 : 2;
    decodeSplitsLong = 4;
    prefillChunkShort = 32;
    prefillChunkLong = 16;
    decodeBurstBudgetMs = Math.min(decodeBurstBudgetMs, Math.max(2, Math.round(frameMs * 0.6)));
    decodeSlowStepThresholdMs = Math.min(decodeSlowStepThresholdMs, 8);
  } else if (safeStepMs >= 22) {
    decodeSplitsShort = safeStepMs >= 45 ? (base.mode === 'smooth-ui' ? 4 : 2) : base.decodeSplitsShort;
    decodeSplitsLong = 4;
    prefillChunkShort = base.mode === 'smooth-ui' ? 32 : base.prefillChunkShort;
    prefillChunkLong = base.mode === 'smooth-ui' ? 16 : 32;
    decodeBurstBudgetMs = Math.min(
      decodeBurstBudgetMs,
      base.mode === 'balanced' ? 10 : Math.max(3, Math.round(frameMs * 0.6)),
    );
  }

  return {
    ...base,
    prefillChunkShort,
    prefillChunkLong,
    prefillBurstBudgetMs,
    prefillBreatherMs,
    decodeSplitsShort,
    decodeSplitsLong,
    decodeSlowStepThresholdMs,
    decodeBurstBudgetMs,
  };
}

/** Selects the prefill chunk token limit (clamped to `[1, 64]`) for the active pacing mode. */
export function selectPrefillChunkSize(
  mode: GpuPacingMode,
  currentSeqPos: number,
  requestedChunkSize?: number,
  policyOverride?: GpuPacingPolicy,
): number {
  const policy = policyOverride ?? getGpuPacingPolicy(mode);
  if (policy.mode === 'max-speed') {
    return Math.max(1, Math.min(64, requestedChunkSize || 64));
  }
  const limit =
    currentSeqPos >= policy.prefillLongThreshold
      ? policy.prefillChunkLong
      : policy.prefillChunkShort;
  if (requestedChunkSize && requestedChunkSize > 0) {
    return Math.max(1, Math.min(64, Math.min(limit, requestedChunkSize)));
  }
  return Math.max(1, Math.min(64, limit));
}

/** Selects how many GPU command-submission slices a single decode step should be split across. */
export function selectDecodeSliceCount(
  mode: GpuPacingMode,
  currentSeqPos: number,
  lastStepGpuMs: number,
  policyOverride?: GpuPacingPolicy,
): 1 | 2 | 4 {
  const policy = policyOverride ?? getGpuPacingPolicy(mode);
  if (policy.mode === 'max-speed') {
    return 1;
  }
  if (
    currentSeqPos >= policy.decodeLongThreshold ||
    lastStepGpuMs >= policy.decodeSlowStepThresholdMs
  ) {
    return policy.decodeSplitsLong;
  }
  return policy.decodeSplitsShort;
}

/** Computes the post-decode-step breather duration in milliseconds to let the browser composite frames. */
export function computeDecodePostStepBreatherMs(
  mode: GpuPacingMode,
  stepGpuMs: number,
  policyOverride?: GpuPacingPolicy,
): number {
  const policy = policyOverride ?? getGpuPacingPolicy(mode);
  if (policy.mode === 'max-speed' || policy.decodePostStepMaxMs <= 0) {
    return 0;
  }
  const safeStepMs = Number.isFinite(stepGpuMs) && stepGpuMs > 0 ? stepGpuMs : 0;
  return Math.min(
    policy.decodePostStepMaxMs,
    Math.max(policy.decodePostStepMinMs, Math.round(safeStepMs * policy.decodePostStepRatio)),
  );
}

/**
 * Accumulates decode GPU step time into a burst budget and triggers a compositor breather
 * only once `cumulativeBurstMs + stepGpuMs >= decodeBurstBudgetMs`.
 */
export function computeDecodeBurstBreatherMs(
  mode: GpuPacingMode,
  stepGpuMs: number,
  cumulativeBurstMs: number,
  policyOverride?: GpuPacingPolicy,
): { breatherMs: number; nextBurstMs: number } {
  const policy = policyOverride ?? getGpuPacingPolicy(mode);
  if (policy.mode === 'max-speed' || policy.decodePostStepMaxMs <= 0) {
    return { breatherMs: 0, nextBurstMs: 0 };
  }
  const safeStepMs = Number.isFinite(stepGpuMs) && stepGpuMs > 0 ? stepGpuMs : 0;
  const safeCumMs = Number.isFinite(cumulativeBurstMs) && cumulativeBurstMs > 0 ? cumulativeBurstMs : 0;
  const totalBurstMs = safeCumMs + safeStepMs;
  if (totalBurstMs < policy.decodeBurstBudgetMs) {
    return { breatherMs: 0, nextBurstMs: totalBurstMs };
  }
  return {
    breatherMs: computeDecodePostStepBreatherMs(mode, totalBurstMs, policy),
    nextBurstMs: 0,
  };
}

/** Computes whether an accumulated prefill GPU burst should pause for a UI breather and reset its counter. */
export function computePrefillSliceBreatherMs(
  mode: GpuPacingMode,
  cumulativeBurstMs: number,
  policyOverride?: GpuPacingPolicy,
): { breatherMs: number; nextBurstMs: number } {
  const policy = policyOverride ?? getGpuPacingPolicy(mode);
  if (cumulativeBurstMs >= policy.prefillBurstBudgetMs && policy.prefillBreatherMs > 0) {
    return { breatherMs: policy.prefillBreatherMs, nextBurstMs: 0 };
  }
  return { breatherMs: 0, nextBurstMs: cumulativeBurstMs };
}

/**
 * Partitions `[0, totalSteps)` along layer boundaries into contiguous, non-empty
 * decode step slices so each slice aligns cleanly with whole transformer layers.
 */
export function computeDecodeSliceBounds(
  layerEndStepIndices: number[],
  numSlices: 1 | 2 | 4,
): Array<{ start: number; end: number }> {
  const layerCount = layerEndStepIndices.length;
  if (layerCount === 0) {
    return [];
  }
  const totalSteps = layerEndStepIndices[layerCount - 1];
  if (!Number.isFinite(totalSteps) || totalSteps <= 0) {
    return [];
  }
  if (numSlices === 1 || layerCount === 1) {
    return [{ start: 0, end: totalSteps }];
  }

  const splitLayerIndices =
    numSlices === 2
      ? [Math.floor(layerCount / 2) - 1]
      : [
          Math.floor(layerCount / 4) - 1,
          Math.floor(layerCount / 2) - 1,
          Math.floor((3 * layerCount) / 4) - 1,
        ];

  const bounds: Array<{ start: number; end: number }> = [];
  let start = 0;
  for (const layerIdx of splitLayerIndices) {
    if (layerIdx < 0 || layerIdx >= layerCount) {
      continue;
    }
    const end = layerEndStepIndices[layerIdx];
    if (end > start && end < totalSteps) {
      bounds.push({ start, end });
      start = end;
    }
  }
  if (start < totalSteps) {
    bounds.push({ start, end: totalSteps });
  }
  return bounds;
}

/** Async timer helper that avoids scheduling macrotasks when `ms <= 0`. */
export function sleepMs(ms: number): Promise<void> {
  if (!Number.isFinite(ms) || ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Resolves an extension-relative asset path across workers, offscreen documents,
 * explicit asset base URLs, and plain test environments.
 */
export function resolveExtensionAssetUrl(relativePath: string, assetBaseUrl?: string): string {
  const cleanPath = relativePath.replace(/^\/+/, '');
  const trimmedBase = assetBaseUrl?.trim();
  if (trimmedBase) {
    const normalizedBase = trimmedBase.endsWith('/') ? trimmedBase : `${trimmedBase}/`;
    try {
      return new URL(cleanPath, normalizedBase).href;
    } catch {
      return `${normalizedBase}${cleanPath}`;
    }
  }
  if (typeof chrome !== 'undefined' && typeof chrome.runtime?.getURL === 'function') {
    return chrome.runtime.getURL(cleanPath);
  }
  if (typeof self !== 'undefined' && self.location?.origin && self.location.origin !== 'null') {
    return `${self.location.origin}/${cleanPath}`;
  }
  return `/${cleanPath}`;
}
