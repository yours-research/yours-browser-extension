import type { WebGpuProbeInfo } from '../shared/protocol.ts';

export interface WebGpuLimitsLike {
  maxBufferSize?: number;
  maxStorageBufferBindingSize?: number;
  maxComputeWorkgroupStorageSize?: number;
  maxComputeInvocationsPerWorkgroup?: number;
  maxStorageBuffersPerShaderStage?: number;
  [key: string]: number | undefined;
}

export interface WebGpuAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

export interface WebGpuDeviceLostInfoLike {
  reason?: string;
  message?: string;
}

export interface WebGpuDeviceLike {
  limits: WebGpuLimitsLike;
  features: Iterable<string>;
  lost: Promise<WebGpuDeviceLostInfoLike>;
  destroy?: () => void;
}

export interface WebGpuDeviceDescriptorLike {
  requiredFeatures?: Iterable<string>;
  requiredLimits?: Record<string, number>;
  [key: string]: unknown;
}

export interface WebGpuAdapterLike {
  limits: WebGpuLimitsLike;
  features: Iterable<string>;
  info?: WebGpuAdapterInfoLike;
  requestAdapterInfo?: () => Promise<WebGpuAdapterInfoLike>;
  requestDevice: (descriptor?: WebGpuDeviceDescriptorLike) => Promise<WebGpuDeviceLike>;
}

export interface WebGpuNavigatorLike {
  gpu?: {
    requestAdapter: (options?: {
      powerPreference?: 'low-power' | 'high-performance';
      forceFallbackAdapter?: boolean;
    }) => Promise<WebGpuAdapterLike | null>;
  };
}

export interface GrantedLimitsSnapshot {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}

let patchInstalled = false;
let lastGrantedLimits: GrantedLimitsSnapshot | null = null;
let deviceLostCallback: ((info: { reason: string; message: string }) => void) | null = null;
const wrappedAdapters = new WeakSet<object>();
const hookedDevices = new WeakSet<object>();
const ignoredLostDevices = new WeakSet<object>();

export function getLastGrantedLimits(): GrantedLimitsSnapshot | null {
  return lastGrantedLimits;
}

/**
 * Compute elevated device limits clamped to what the adapter actually supports.
 * Ensures maxStorageBufferBindingSize <= maxBufferSize <= adapterLimits.maxBufferSize.
 */
export function computeElevatedLimits(
  adapterLimits: WebGpuLimitsLike,
  requestedLimits?: Record<string, number>,
): Record<string, number> {
  const elevated: Record<string, number> = { ...(requestedLimits ?? {}) };

  const adapterMaxBuffer = adapterLimits.maxBufferSize;
  const reqMaxBuffer = requestedLimits?.maxBufferSize;
  if (typeof adapterMaxBuffer === 'number' && adapterMaxBuffer > 0) {
    elevated.maxBufferSize =
      typeof reqMaxBuffer === 'number' && reqMaxBuffer > 0
        ? Math.min(Math.max(reqMaxBuffer, adapterMaxBuffer), adapterMaxBuffer)
        : adapterMaxBuffer;
  }

  const effectiveMaxBuffer = elevated.maxBufferSize ?? adapterMaxBuffer;
  const adapterMaxStorage = adapterLimits.maxStorageBufferBindingSize;
  const reqMaxStorage = requestedLimits?.maxStorageBufferBindingSize;
  if (typeof adapterMaxStorage === 'number' && adapterMaxStorage > 0) {
    const rawTarget =
      typeof reqMaxStorage === 'number' && reqMaxStorage > 0
        ? Math.max(reqMaxStorage, adapterMaxStorage)
        : adapterMaxStorage;
    elevated.maxStorageBufferBindingSize =
      typeof effectiveMaxBuffer === 'number' && effectiveMaxBuffer > 0
        ? Math.min(rawTarget, adapterMaxStorage, effectiveMaxBuffer)
        : Math.min(rawTarget, adapterMaxStorage);
  }

  return elevated;
}

function attachDeviceObservers(device: WebGpuDeviceLike): void {
  if (device.limits) {
    lastGrantedLimits = {
      maxBufferSize: device.limits.maxBufferSize ?? 0,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize ?? 0,
    };
  }

  if (typeof device === 'object' && device !== null && !hookedDevices.has(device)) {
    hookedDevices.add(device);
    if (device.lost && typeof device.lost.then === 'function') {
      device.lost.then(
        (info) => {
          if (ignoredLostDevices.has(device) || info?.reason === 'destroyed') {
            return;
          }
          deviceLostCallback?.({
            reason: info?.reason ?? 'unknown',
            message: info?.message ?? 'WebGPU device lost without details.',
          });
        },
        () => {
          // Ignore promise rejection on lost channel.
        },
      );
    }
  }
}

function wrapAdapter(adapter: WebGpuAdapterLike): WebGpuAdapterLike {
  if (wrappedAdapters.has(adapter)) {
    return adapter;
  }
  wrappedAdapters.add(adapter);

  const originalRequestDevice = adapter.requestDevice.bind(adapter);
  adapter.requestDevice = async (descriptor?: WebGpuDeviceDescriptorLike) => {
    const elevatedLimits = computeElevatedLimits(adapter.limits, descriptor?.requiredLimits);
    const nextDescriptor: WebGpuDeviceDescriptorLike = {
      ...(descriptor ?? {}),
      requiredLimits: elevatedLimits,
    };

    try {
      const device = await originalRequestDevice(nextDescriptor);
      attachDeviceObservers(device);
      return device;
    } catch {
      // Fallback to caller's unmodified descriptor if driver rejects an elevated limit.
      const device = await originalRequestDevice(descriptor);
      attachDeviceObservers(device);
      return device;
    }
  };

  return adapter;
}

/**
 * Installs a one-time wrapper over navigator.gpu.requestAdapter so ONNX Runtime Web
 * automatically receives elevated maxBufferSize and maxStorageBufferBindingSize limits
 * instead of WebGPU's 128 MiB / 256 MiB defaults.
 */
export function installWebGpuLimitPatch(
  onDeviceLost?: (info: { reason: string; message: string }) => void,
  gpuNav: WebGpuNavigatorLike | undefined = typeof navigator !== 'undefined'
    ? (navigator as unknown as WebGpuNavigatorLike)
    : undefined,
): boolean {
  if (onDeviceLost) {
    deviceLostCallback = onDeviceLost;
  }

  if (patchInstalled) {
    return true;
  }

  if (!gpuNav?.gpu || typeof gpuNav.gpu.requestAdapter !== 'function') {
    return false;
  }

  const originalRequestAdapter = gpuNav.gpu.requestAdapter.bind(gpuNav.gpu);
  gpuNav.gpu.requestAdapter = async (options) => {
    const adapter = await originalRequestAdapter(options);
    return adapter ? wrapAdapter(adapter) : null;
  };
  patchInstalled = true;
  return true;
}

export function resetWebGpuLimitPatchForTests(): void {
  patchInstalled = false;
  lastGrantedLimits = null;
  deviceLostCallback = null;
}

async function readAdapterInfo(adapter: WebGpuAdapterLike): Promise<{
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}> {
  let raw: WebGpuAdapterInfoLike | undefined = adapter.info;
  if (!raw && typeof adapter.requestAdapterInfo === 'function') {
    try {
      raw = await adapter.requestAdapterInfo();
    } catch {
      raw = undefined;
    }
  }
  return {
    vendor: raw?.vendor || 'unknown',
    architecture: raw?.architecture || 'unknown',
    device: raw?.device || 'unknown',
    description: raw?.description || 'unknown',
  };
}

export async function probeWebGpu(
  gpuNav: WebGpuNavigatorLike | undefined = typeof navigator !== 'undefined'
    ? (navigator as unknown as WebGpuNavigatorLike)
    : undefined,
): Promise<WebGpuProbeInfo> {
  const probedAt = new Date().toISOString();

  if (!gpuNav?.gpu) {
    return {
      supported: false,
      adapterInfo: null,
      features: [],
      hasShaderF16: false,
      hasSubgroups: false,
      adapterLimits: null,
      grantedLimits: lastGrantedLimits,
      error: 'navigator.gpu is not available in this context.',
      probedAt,
    };
  }

  try {
    const adapter = await gpuNav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return {
        supported: false,
        adapterInfo: null,
        features: [],
        hasShaderF16: false,
        hasSubgroups: false,
        adapterLimits: null,
        grantedLimits: lastGrantedLimits,
        error: 'navigator.gpu.requestAdapter returned null.',
        probedAt,
      };
    }

    const features = Array.from(adapter.features).sort();
    const adapterInfo = await readAdapterInfo(adapter);
    const adapterLimits = {
      maxBufferSize: adapter.limits.maxBufferSize ?? 0,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize ?? 0,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize ?? 0,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup ?? 0,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage ?? 0,
    };

    let grantedLimits = lastGrantedLimits;
    const probeDevice = await adapter.requestDevice({
      requiredLimits: computeElevatedLimits(adapter.limits),
    });
    grantedLimits = {
      maxBufferSize: probeDevice.limits.maxBufferSize ?? 0,
      maxStorageBufferBindingSize: probeDevice.limits.maxStorageBufferBindingSize ?? 0,
    };
    lastGrantedLimits = grantedLimits;
    ignoredLostDevices.add(probeDevice);
    probeDevice.destroy?.();

    return {
      supported: true,
      adapterInfo,
      features,
      hasShaderF16: features.includes('shader-f16'),
      hasSubgroups: features.includes('subgroups'),
      adapterLimits,
      grantedLimits,
      error: null,
      probedAt,
    };
  } catch (err) {
    return {
      supported: false,
      adapterInfo: null,
      features: [],
      hasShaderF16: false,
      hasSubgroups: false,
      adapterLimits: null,
      grantedLimits: lastGrantedLimits,
      error: err instanceof Error ? err.message : String(err),
      probedAt,
    };
  }
}
