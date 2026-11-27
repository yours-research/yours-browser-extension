export const CYCLE17_DROPIN_CALIBRATION_REV = 'm11-b7fb760';

export interface PersistedSpecCalibration {
  adapterName: string;
  dropinSha256: string;
  packRevision?: string;
  kernelTier?: string;
  t1Ms: number;
  stepMs: number[];
  headMs: number;
  rejectPenaltyMs: number;
  measuredAtIso: string;
}

const CALIBRATION_STORAGE_PREFIX = 'yours-spec-calibration:';
const CALIBRATION_IDB_NAME = 'tinygpt-spec-calibration';
const CALIBRATION_IDB_STORE = 'calibrations';

export function buildSpecCalibrationKey(adapterName: string, dropinSha256: string): string {
  return `${adapterName || 'webgpu-default'}::${dropinSha256}::${CYCLE17_DROPIN_CALIBRATION_REV}`;
}

export function isValidSpecCalibration(
  raw: unknown,
  adapterName: string,
  dropinSha256: string,
): raw is PersistedSpecCalibration {
  if (!raw || typeof raw !== 'object') return false;
  const r = raw as Partial<PersistedSpecCalibration>;
  return (
    r.adapterName === adapterName &&
    r.dropinSha256 === dropinSha256 &&
    r.packRevision === CYCLE17_DROPIN_CALIBRATION_REV &&
    typeof r.t1Ms === 'number' &&
    Number.isFinite(r.t1Ms) &&
    r.t1Ms > 0 &&
    Array.isArray(r.stepMs) &&
    r.stepMs.length > 0 &&
    r.stepMs.every((v) => typeof v === 'number' && Number.isFinite(v) && v > 0) &&
    typeof r.headMs === 'number' &&
    Number.isFinite(r.headMs) &&
    r.headMs >= 0 &&
    typeof r.rejectPenaltyMs === 'number' &&
    Number.isFinite(r.rejectPenaltyMs) &&
    r.rejectPenaltyMs >= 0
  );
}

/** Break-even gate: only activate Engine B when headMs <= 0.35 * t1Ms. */
export function shouldEnableEngineB(headMs: number | null, t1Ms: number): boolean {
  if (headMs === null || !Number.isFinite(headMs) || !Number.isFinite(t1Ms) || t1Ms <= 0) {
    return false;
  }
  return headMs <= 0.35 * t1Ms;
}

function openCalibrationDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(CALIBRATION_IDB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(CALIBRATION_IDB_STORE)) {
          db.createObjectStore(CALIBRATION_IDB_STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function readFromIdb(key: string): Promise<unknown> {
  const db = await openCalibrationDb();
  if (!db) return null;
  try {
    return await new Promise((resolve) => {
      try {
        const tx = db.transaction(CALIBRATION_IDB_STORE, 'readonly');
        const req = tx.objectStore(CALIBRATION_IDB_STORE).get(key);
        req.onsuccess = () => resolve(req.result ?? null);
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  } finally {
    try {
      db.close();
    } catch {
      // ignore close errors
    }
  }
}

async function writeToIdb(key: string, value: PersistedSpecCalibration): Promise<void> {
  const db = await openCalibrationDb();
  if (!db) return;
  try {
    await new Promise<void>((resolve) => {
      try {
        const tx = db.transaction(CALIBRATION_IDB_STORE, 'readwrite');
        tx.objectStore(CALIBRATION_IDB_STORE).put(value, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      } catch {
        resolve();
      }
    });
  } finally {
    try {
      db.close();
    } catch {
      // ignore close errors
    }
  }
}

export async function loadSpecCalibration(
  adapterName: string,
  dropinSha256: string,
): Promise<PersistedSpecCalibration | null> {
  const key = buildSpecCalibrationKey(adapterName, dropinSha256);
  const storageKey = `${CALIBRATION_STORAGE_PREFIX}${key}`;
  const chromeStorage = (globalThis as { chrome?: { storage?: { local?: chrome.storage.LocalStorageArea } } })
    .chrome?.storage?.local;
  if (chromeStorage) {
    try {
      const hit = await new Promise<unknown>((resolve) => {
        chromeStorage.get(storageKey, (res) => resolve(res?.[storageKey] ?? null));
      });
      if (isValidSpecCalibration(hit, adapterName, dropinSha256)) {
        return hit;
      }
    } catch {
      // ignore chrome.storage errors
    }
  }
  const idbHit = await readFromIdb(key);
  if (isValidSpecCalibration(idbHit, adapterName, dropinSha256)) {
    return idbHit;
  }
  try {
    if (typeof globalThis.localStorage !== 'undefined') {
      const raw = globalThis.localStorage.getItem(storageKey);
      if (raw) {
        const parsed = JSON.parse(raw) as unknown;
        if (isValidSpecCalibration(parsed, adapterName, dropinSha256)) {
          return parsed;
        }
      }
    }
  } catch {
    // ignore localStorage errors
  }
  return null;
}

export async function saveSpecCalibration(entry: PersistedSpecCalibration): Promise<void> {
  const stamped: PersistedSpecCalibration = {
    ...entry,
    packRevision: entry.packRevision ?? CYCLE17_DROPIN_CALIBRATION_REV,
  };
  const key = buildSpecCalibrationKey(stamped.adapterName, stamped.dropinSha256);
  const storageKey = `${CALIBRATION_STORAGE_PREFIX}${key}`;
  const chromeStorage = (globalThis as { chrome?: { storage?: { local?: chrome.storage.LocalStorageArea } } })
    .chrome?.storage?.local;
  if (chromeStorage) {
    try {
      await new Promise<void>((resolve) => {
        chromeStorage.set({ [storageKey]: stamped }, () => resolve());
      });
    } catch {
      // ignore chrome.storage errors
    }
  }
  await writeToIdb(key, stamped);
  try {
    if (typeof globalThis.localStorage !== 'undefined') {
      globalThis.localStorage.setItem(storageKey, JSON.stringify(stamped));
    }
  } catch {
    // ignore localStorage errors
  }
}

const CONTROL_TOKENS_TO_STRIP_RE =
  /<\|im_end\|>|<\|endoftext\|>|<end_of_turn>|<turn\|>|<\|turn>model\r?\n?|<\|im_start\|>assistant\r?\n?|<eos>|<bos>|<pad>/g;

/**
 * Stateful O(1)-per-token incremental tokenizer decoder that matches
 * `decodeGeneratedText(tokenizer, tokenIds, { preOpenedThink })` byte-for-byte
 * while decoding only a fixed context+tail window per token instead of O(N^2) full arrays.
 */
export class IncrementalTokenDecoder {
  private readonly ids: number[] = [];
  private commitIdx = 0;
  private committedCleaned = '';
  private ctxDecoded = '';
  private openedThinkApplied = false;

  private static readonly CONTEXT_OVERLAP = 6;
  private static readonly COMMIT_STRIDE = 8;

  constructor(
    private readonly tokenizer: { decode(ids: number[], opts?: { skip_special_tokens?: boolean }): string },
    private readonly preOpenedThink = false,
  ) {}

  private decodeWindow(fromIdx: number, toIdx: number): string {
    if (toIdx <= fromIdx) return '';
    return this.tokenizer.decode(this.ids.slice(fromIdx, toIdx), { skip_special_tokens: false });
  }

  private cleanTail(rawTail: string, hasPrefix: boolean): string {
    let cleaned = rawTail.replace(CONTROL_TOKENS_TO_STRIP_RE, '');
    if (
      this.preOpenedThink &&
      !this.openedThinkApplied &&
      !hasPrefix &&
      cleaned.length > 0
    ) {
      if (!/^\s*(?:<think>|<thought>|<\|channel>thought)/.test(cleaned)) {
        cleaned = `<think>\n${cleaned}`;
      }
      this.openedThinkApplied = true;
    }
    return cleaned;
  }

  private maybeAdvanceCommit(): void {
    if (this.ids.length - this.commitIdx < IncrementalTokenDecoder.COMMIT_STRIDE * 2) {
      return;
    }
    const targetCommit = this.ids.length - IncrementalTokenDecoder.COMMIT_STRIDE;
    const ctxStart = Math.max(0, this.commitIdx - IncrementalTokenDecoder.CONTEXT_OVERLAP);
    const windowRaw = this.decodeWindow(ctxStart, targetCommit);
    const rawSlice = windowRaw.slice(this.ctxDecoded.length);
    // Avoid freezing commit boundary on incomplete UTF-8 runes, partial/adjacent control tags,
    // or leading whitespace before <think> when preOpenedThink is enabled.
    if (
      rawSlice.endsWith('\uFFFD') ||
      /<(?:\|im_start\|>|\|turn>|[^>\r\n]{0,24})$/.test(rawSlice) ||
      (this.preOpenedThink && this.committedCleaned.length === 0 && /^\s*$/.test(rawSlice))
    ) {
      return;
    }
    const cleanedSlice = this.cleanTail(rawSlice, this.committedCleaned.length > 0);
    this.committedCleaned += cleanedSlice;
    this.commitIdx = targetCommit;
    const nextCtxStart = Math.max(0, this.commitIdx - IncrementalTokenDecoder.CONTEXT_OVERLAP);
    this.ctxDecoded = this.decodeWindow(nextCtxStart, this.commitIdx);
  }

  appendToken(tokenId: number): string {
    this.ids.push(tokenId);
    const ctxStart = Math.max(0, this.commitIdx - IncrementalTokenDecoder.CONTEXT_OVERLAP);
    const windowRaw = this.decodeWindow(ctxStart, this.ids.length);
    const rawTail = windowRaw.slice(this.ctxDecoded.length);
    let cleanedTail: string;
    if (this.committedCleaned.length === 0) {
      cleanedTail = rawTail.replace(CONTROL_TOKENS_TO_STRIP_RE, '');
      if (
        this.preOpenedThink &&
        cleanedTail.length > 0 &&
        !/^\s*(?:<think>|<thought>|<\|channel>thought)/.test(cleanedTail)
      ) {
        cleanedTail = `<think>\n${cleanedTail}`;
      }
    } else {
      cleanedTail = rawTail.replace(CONTROL_TOKENS_TO_STRIP_RE, '');
    }
    const fullSoFar = this.committedCleaned + cleanedTail;
    this.maybeAdvanceCommit();
    return fullSoFar;
  }

  getSafeEmittedText(): string {
    const ctxStart = Math.max(0, this.commitIdx - IncrementalTokenDecoder.CONTEXT_OVERLAP);
    const windowRaw = this.decodeWindow(ctxStart, this.ids.length);
    const rawTail = windowRaw.slice(this.ctxDecoded.length);
    let cleanedTail = rawTail.replace(CONTROL_TOKENS_TO_STRIP_RE, '');
    if (
      this.committedCleaned.length === 0 &&
      this.preOpenedThink &&
      cleanedTail.length > 0 &&
      !/^\s*(?:<think>|<thought>|<\|channel>thought)/.test(cleanedTail)
    ) {
      cleanedTail = `<think>\n${cleanedTail}`;
    }
    const full = this.committedCleaned + cleanedTail;
    return full.endsWith('\uFFFD') ? full.slice(0, full.length - 1) : full;
  }
}
