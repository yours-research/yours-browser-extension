/**
 * Entry for dist/cycle17-dropin/bench/bench-engine-b-head.mjs (bundled by the
 * pack). In any WebGPU page or the extension's offscreen document:
 *
 *   const m = await import(chrome.runtime.getURL('cycle17/bench/bench-engine-b-head.mjs'));
 *   const report = await m.runInPage({ assetsBase: chrome.runtime.getURL('cycle17/'),
 *     tableUrl: chrome.runtime.getURL('models/pragma-rc1/slab2.part00.bin'),
 *     probe: { measure: (M, reject) => engine.measureSpecStep(M, reject) } });
 *   const opts = m.toAdapterOptions(report, liveChainStepMs, 3.67);
 *
 * Inside the extension, pass `device` and `table` to reuse the live device and
 * the borrowed RC1 tied slabs instead of uploading another copy.
 */
import { loadGdnBundle } from '../draft/gdnEagle3Loader.js';
import { GdnDraftRuntime, shaderAssets, type TiedTable } from '../draft/gdnEagle3Runtime.js';
import { runEngineBBench, type VerifyProbe } from './engineBHeadBench.js';

export { benchHeadLatency, benchRejectSurcharge, quantile, runEngineBBench, toAdapterOptions } from './engineBHeadBench.js';

/** Golden anchor shipped in the pack (fixtures/trained-exp007): RC1 trace taps and root token 271. */
export const GOLDEN_ROOT_TOKEN = 271;

export async function runInPage(o: {
  assetsBase: string | URL;
  tableUrl?: string;
  device?: GPUDevice;
  table?: TiedTable;
  probe?: VerifyProbe;
  budgetMs?: number;
  /** Kernel tier ('auto' default). Run once with 'reference' to compare. */
  mode?: 'auto' | 'reference' | 'fast';
}) {
  const base = new URL(String(o.assetsBase), (globalThis as { location?: { href: string } }).location?.href);
  let device = o.device;
  let label: string | undefined;
  if (!device) {
    const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    label = adapter.info?.description || adapter.info?.vendor;
    // Request subgroups + shader-f16 when available so the fast Tier-1 kernels can be selected.
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
    if (adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16');
    device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
    });
    // Propagate adapter.info onto device.adapterInfo when Chrome leaves it unset so selectDraftTier sees subgroup limits.
    const devWithInfo = device as GPUDevice & { adapterInfo?: GPUAdapterInfo };
    if (adapter.info && (!devWithInfo.adapterInfo || typeof devWithInfo.adapterInfo.subgroupMaxSize !== 'number')) {
      try {
        Object.defineProperty(device, 'adapterInfo', { value: adapter.info, configurable: true });
      } catch {
        // Read-only host object on browsers that already expose device.adapterInfo natively.
      }
    }
  }
  const bundle = await loadGdnBundle(new URL('assets/exp007/manifest.json', base));
  if (!bundle) throw new Error('No EXP-007 draft head under assets/exp007/');
  let owned: GPUBuffer | undefined;
  let table = o.table;
  if (!table) {
    if (!o.tableUrl) throw new Error('Pass table (borrowed slabs) or tableUrl (RC1 slab2)');
    const response = await fetch(o.tableUrl);
    if (!response.ok) throw new Error('Unable to load the tied table (HTTP ' + response.status + ')');
    const bytes = await response.arrayBuffer();
    owned = device.createBuffer({ size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(owned, 0, bytes);
    table = { buffer: owned };
  }
  const tapsResponse = await fetch(new URL('fixtures/trained-exp007/taps.f32', base));
  if (!tapsResponse.ok) throw new Error('Unable to load the golden taps');
  const taps = new Float32Array(await tapsResponse.arrayBuffer());
  const runtime = await GdnDraftRuntime.create(device, bundle, table, shaderAssets(new URL('wgsl/', base)), { mode: o.mode });
  try {
    const report = await runEngineBBench(runtime, { taps, rootToken: GOLDEN_ROOT_TOKEN }, o.probe, { budgetMs: o.budgetMs ?? 10_000, device: label });
    return { ...report, headSha256: bundle.sha256, tier: runtime.tier, chainStats: { ...runtime.chainStats } };
  } finally {
    runtime.dispose();
    owned?.destroy();
  }
}
