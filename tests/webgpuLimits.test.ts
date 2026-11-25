import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  computeElevatedLimits,
  getLastGrantedLimits,
  installWebGpuLimitPatch,
  probeWebGpu,
  resetWebGpuLimitPatchForTests,
  type WebGpuDeviceDescriptorLike,
  type WebGpuNavigatorLike,
} from '../src/offscreen/webgpuLimits.ts';

describe('webgpuLimits', () => {
  beforeEach(() => {
    resetWebGpuLimitPatchForTests();
  });

  it('computes elevated maxBufferSize and maxStorageBufferBindingSize clamped to adapter limits', () => {
    const elevated = computeElevatedLimits(
      {
        maxBufferSize: 4 * 1024 * 1024 * 1024,
        maxStorageBufferBindingSize: 2 * 1024 * 1024 * 1024,
      },
      {
        maxBufferSize: 256 * 1024 * 1024,
        maxComputeWorkgroupStorageSize: 32768,
      },
    );

    expect(elevated.maxBufferSize).toBe(4 * 1024 * 1024 * 1024);
    expect(elevated.maxStorageBufferBindingSize).toBe(2 * 1024 * 1024 * 1024);
    expect(elevated.maxComputeWorkgroupStorageSize).toBe(32768);
  });

  it('clamps maxStorageBufferBindingSize <= maxBufferSize even if adapter reports unusual limits', () => {
    const elevated = computeElevatedLimits({
      maxBufferSize: 512 * 1024 * 1024,
      maxStorageBufferBindingSize: 1024 * 1024 * 1024,
    });

    expect(elevated.maxBufferSize).toBe(512 * 1024 * 1024);
    expect(elevated.maxStorageBufferBindingSize).toBe(512 * 1024 * 1024);
  });

  it('wraps navigator.gpu.requestAdapter to inject elevated limits and track device.lost', async () => {
    let resolveLost!: (info: { reason: string; message: string }) => void;
    const lostPromise = new Promise<{ reason: string; message: string }>((resolve) => {
      resolveLost = resolve;
    });

    const receivedDescriptors: Array<WebGpuDeviceDescriptorLike | undefined> = [];
    const mockAdapter = {
      limits: {
        maxBufferSize: 2147483648,
        maxStorageBufferBindingSize: 1073741824,
      },
      features: new Set(['shader-f16', 'subgroups']),
      info: {
        vendor: 'nvidia',
        architecture: 'ampere',
        device: 'rtx3090',
        description: 'NVIDIA GeForce RTX 3090',
      },
      requestDevice: vi.fn(async (desc?: WebGpuDeviceDescriptorLike) => {
        receivedDescriptors.push(desc);
        return {
          limits: {
            maxBufferSize: desc?.requiredLimits?.maxBufferSize ?? 268435456,
            maxStorageBufferBindingSize:
              desc?.requiredLimits?.maxStorageBufferBindingSize ?? 134217728,
          },
          features: new Set(['shader-f16']),
          lost: lostPromise,
          destroy: vi.fn(),
        };
      }),
    };

    const mockNav: WebGpuNavigatorLike = {
      gpu: {
        requestAdapter: vi.fn(async () => mockAdapter),
      },
    };

    const lostSpy = vi.fn();
    expect(installWebGpuLimitPatch(lostSpy, mockNav)).toBe(true);

    const adapter = await mockNav.gpu!.requestAdapter({ powerPreference: 'high-performance' });
    expect(adapter).not.toBeNull();

    const device = await adapter!.requestDevice({ requiredFeatures: ['shader-f16'] });
    expect(receivedDescriptors[0]?.requiredFeatures).toEqual(['shader-f16']);
    expect(receivedDescriptors[0]?.requiredLimits?.maxBufferSize).toBe(2147483648);
    expect(receivedDescriptors[0]?.requiredLimits?.maxStorageBufferBindingSize).toBe(1073741824);
    expect(device.limits.maxBufferSize).toBe(2147483648);
    expect(getLastGrantedLimits()).toEqual({
      maxBufferSize: 2147483648,
      maxStorageBufferBindingSize: 1073741824,
    });

    resolveLost({ reason: 'unknown', message: 'Simulated TDR' });
    await lostPromise;
    await Promise.resolve();
    expect(lostSpy).toHaveBeenCalledWith({ reason: 'unknown', message: 'Simulated TDR' });

    const probe = await probeWebGpu(mockNav);
    expect(probe.supported).toBe(true);
    expect(probe.hasShaderF16).toBe(true);
    expect(probe.hasSubgroups).toBe(true);
    expect(probe.grantedLimits?.maxBufferSize).toBe(2147483648);
  });

  it('ignores intentional device destruction during probeWebGpu while still reporting unexpected device loss', async () => {
    let resolveProbeLost!: (info: { reason: string; message: string }) => void;
    const probeLostPromise = new Promise<{ reason: string; message: string }>((resolve) => {
      resolveProbeLost = resolve;
    });

    const mockAdapter = {
      limits: {
        maxBufferSize: 2147483648,
        maxStorageBufferBindingSize: 1073741824,
      },
      features: new Set(['shader-f16']),
      info: { vendor: 'nvidia', architecture: 'ampere', device: 'rtx3090', description: 'RTX 3090' },
      requestDevice: vi.fn(async (desc?: WebGpuDeviceDescriptorLike) => ({
        limits: {
          maxBufferSize: desc?.requiredLimits?.maxBufferSize ?? 268435456,
          maxStorageBufferBindingSize:
            desc?.requiredLimits?.maxStorageBufferBindingSize ?? 134217728,
        },
        features: new Set(['shader-f16']),
        lost: probeLostPromise,
        destroy: vi.fn(() => {
          resolveProbeLost({ reason: 'destroyed', message: 'Device was destroyed.' });
        }),
      })),
    };

    const mockNav: WebGpuNavigatorLike = {
      gpu: {
        requestAdapter: vi.fn(async () => mockAdapter),
      },
    };

    const lostSpy = vi.fn();
    expect(installWebGpuLimitPatch(lostSpy, mockNav)).toBe(true);

    const probe = await probeWebGpu(mockNav);
    await probeLostPromise;
    await Promise.resolve();

    expect(probe.supported).toBe(true);
    expect(lostSpy).not.toHaveBeenCalled();
  });
});
