export const STANDARD_REFRESH_RATES = [60, 75, 90, 120, 144, 165, 240] as const;

const MIN_VALID_HZ = 24;
const MAX_FRAME_DELTA_MS = 100;

/**
 * Snaps a raw measured refresh rate (in Hz) to the nearest standard panel rate.
 * Clamps invalid or backgrounded readings (< 24 Hz or non-finite) to 60 Hz.
 */
export function snapRefreshRate(rawHz: number): number {
  if (!Number.isFinite(rawHz) || rawHz < MIN_VALID_HZ) {
    return 60;
  }

  let bestRate: number = STANDARD_REFRESH_RATES[0];
  let bestDiff = Math.abs(rawHz - bestRate);

  for (let i = 1; i < STANDARD_REFRESH_RATES.length; i += 1) {
    const candidate = STANDARD_REFRESH_RATES[i];
    const diff = Math.abs(rawHz - candidate);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestRate = candidate;
    }
  }

  return bestRate;
}

/**
 * Measures the active display refresh rate using requestAnimationFrame timestamps,
 * filtering out backgrounded frame pauses (> 100 ms) and snapping the median Hz.
 */
export function measureDisplayRefreshRate(sampleFrames = 18): Promise<number> {
  if (typeof requestAnimationFrame !== 'function' || sampleFrames <= 0) {
    return Promise.resolve(60);
  }

  const targetFrames = Math.max(2, Math.floor(sampleFrames));

  return new Promise<number>((resolve) => {
    const deltas: number[] = [];
    let prevTimestamp: number | null = null;
    let totalCallbacks = 0;
    const maxCallbacks = targetFrames + 1;

    const onFrame = (timestamp: number) => {
      totalCallbacks += 1;

      if (prevTimestamp !== null) {
        const deltaMs = timestamp - prevTimestamp;
        if (deltaMs > 0 && deltaMs <= MAX_FRAME_DELTA_MS) {
          deltas.push(deltaMs);
        }
      }
      prevTimestamp = timestamp;

      if (totalCallbacks >= maxCallbacks) {
        if (deltas.length === 0) {
          resolve(60);
          return;
        }
        const sorted = [...deltas].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const medianDeltaMs =
          sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

        if (medianDeltaMs <= 0) {
          resolve(60);
          return;
        }
        resolve(snapRefreshRate(1000 / medianDeltaMs));
        return;
      }

      requestAnimationFrame(onFrame);
    };

    requestAnimationFrame(onFrame);
  });
}
