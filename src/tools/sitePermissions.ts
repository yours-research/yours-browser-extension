import { inspectOriginSafety, normalizeOrigin } from '@yours/guard/policy';
import type { SitePermissionItem, SitePermissionMode } from '@yours/ui';

export const SITE_PERMISSIONS_STORAGE_KEY = 'yours-site-permissions';

const CHECKOUT_PAYMENT_PATH_OR_HOST_RE =
  /(?:^|[./-])(?:checkout|payment|payments|pay|billing|card|wallet|transfer|wire|bank|banking)(?:[./-]|$|\?)/i;

export interface StoredSitePermission extends SitePermissionItem {
  updatedAt: number;
}

/**
 * Evaluates whether an origin is locked from persistent `'site'` access by
 * `@yours/guard` sensitive-origin rules or checkout/payment heuristics.
 */
export function resolveSiteLockReason(rawOriginOrUrl: string): string | undefined {
  const safety = inspectOriginSafety(rawOriginOrUrl);
  if (!safety.allowed) {
    return safety.reason ?? 'Sensitive banking, auth, or privileged origin locked by @yours/guard.';
  }
  try {
    const parsed = new URL(rawOriginOrUrl);
    if (CHECKOUT_PAYMENT_PATH_OR_HOST_RE.test(`${parsed.hostname}${parsed.pathname}`)) {
      return 'Payment/checkout flow detected — persistent site access disabled.';
    }
  } catch {
    if (CHECKOUT_PAYMENT_PATH_OR_HOST_RE.test(rawOriginOrUrl)) {
      return 'Payment/checkout flow detected — persistent site access disabled.';
    }
  }
  return undefined;
}

export function normalizePermissionOrigin(rawUrlOrOrigin: string): string | null {
  const trimmed = rawUrlOrOrigin.trim();
  if (!trimmed) return null;
  const normalized = normalizeOrigin(trimmed);
  if (normalized) return normalized;
  try {
    const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
    return normalizeOrigin(withScheme);
  } catch {
    return null;
  }
}

export function clampPermissionMode(
  requestedMode: SitePermissionMode,
  lockedReason?: string,
): SitePermissionMode {
  if (lockedReason && requestedMode === 'site') {
    return 'once';
  }
  return requestedMode;
}

export function createSitePermissionEntry(
  rawUrlOrOrigin: string,
  opts?: {
    mode?: SitePermissionMode;
    scope?: string;
    lastUsedLabel?: string;
    lockedReason?: string;
  },
): StoredSitePermission | null {
  const origin = normalizePermissionOrigin(rawUrlOrOrigin);
  if (!origin) return null;
  const lockedReason = opts?.lockedReason ?? resolveSiteLockReason(rawUrlOrOrigin) ?? resolveSiteLockReason(origin);
  const mode = clampPermissionMode(opts?.mode ?? 'off', lockedReason);
  return {
    id: origin,
    origin,
    scope: opts?.scope ?? 'Page read & tab tools',
    mode,
    lastUsedLabel: opts?.lastUsedLabel ?? 'Just now',
    ...(lockedReason ? { lockedReason } : {}),
    updatedAt: Date.now(),
  };
}

export function sanitizeStoredPermissions(raw: unknown): StoredSitePermission[] {
  if (!Array.isArray(raw)) return [];
  const deduped = new Map<string, StoredSitePermission>();
  for (const row of raw) {
    if (!row || typeof row !== 'object') continue;
    const rec = row as Partial<StoredSitePermission>;
    if (typeof rec.origin !== 'string') continue;
    const mode: SitePermissionMode =
      rec.mode === 'off' || rec.mode === 'once' || rec.mode === 'site' ? rec.mode : 'off';
    const entry = createSitePermissionEntry(rec.origin, {
      mode,
      scope: typeof rec.scope === 'string' && rec.scope.trim() ? rec.scope : 'Page read & tab tools',
      lastUsedLabel: typeof rec.lastUsedLabel === 'string' ? rec.lastUsedLabel : undefined,
      lockedReason: typeof rec.lockedReason === 'string' ? rec.lockedReason : undefined,
    });
    if (entry) {
      deduped.set(entry.id, {
        ...entry,
        updatedAt: typeof rec.updatedAt === 'number' ? rec.updatedAt : entry.updatedAt,
      });
    }
  }
  return Array.from(deduped.values()).sort((a, b) => b.updatedAt - a.updatedAt);
}

export function loadLocalSitePermissions(): StoredSitePermission[] {
  try {
    const raw = window.localStorage.getItem(SITE_PERMISSIONS_STORAGE_KEY);
    if (!raw) return [];
    return sanitizeStoredPermissions(JSON.parse(raw));
  } catch {
    return [];
  }
}

export function persistSitePermissions(items: readonly StoredSitePermission[]): void {
  const clean = sanitizeStoredPermissions(items);
  try {
    window.localStorage.setItem(SITE_PERMISSIONS_STORAGE_KEY, JSON.stringify(clean));
  } catch {
    // Ignore localStorage quota/restrictions
  }
  if (typeof chrome !== 'undefined' && chrome.storage?.local) {
    try {
      chrome.storage.local.set({ [SITE_PERMISSIONS_STORAGE_KEY]: clean });
    } catch {
      // Ignore extension storage errors
    }
  }
}
