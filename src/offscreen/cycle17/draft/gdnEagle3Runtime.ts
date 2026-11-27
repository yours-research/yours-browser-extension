import type { DraftBudget, ScoredDraftTree } from '../engines/types.js';
import type { GdnBundle } from './gdnEagle3Loader.js';

type Access = 'uniform' | 'storage' | 'read-only-storage';
const U = 'uniform',
  R = 'read-only-storage',
  W = 'storage';
const q4Layout = [U, R, R, W] as const;
const q4x3Layout = [U, R, R, W, W, W] as const;
const tiedLayout = [U, R, R, R, W] as const;
const convLayout = [U, R, R, R, W, W, R] as const;
const ioLayout = [U, R, R, R, W, R, R, W] as const;
const dnLayout = [U, R, R, W, W, R] as const;

const shaders = {
  q4: ['q4_gemv_smallm_v2', 'main', q4Layout],
  swiglu: ['q4_gate_up_swiglu_smallm_v2', 'main', q4Layout],
  rms: ['gdn_eagle3_rms', 'main', q4Layout],
  embed: ['gdn_eagle3_tied', 'embed', tiedLayout],
  scores: ['gdn_eagle3_tied', 'subset_logits', tiedLayout],
  sliceScores: ['gdn_eagle3_tied', 'subset_logits_slice', tiedLayout],
  conv: ['DN_CONV_TREE', 'verify', convLayout],
  commitConv: ['DN_CONV_TREE', 'commit', convLayout],
  pack: ['DN_IO', 'pack', ioLayout],
  recur: ['DN_TREE', 'verify', dnLayout],
  commitState: ['DN_TREE', 'commit', dnLayout],
  gate: ['DN_IO', 'gate', ioLayout],
  topK: ['TOPK_BATCH', 'topk_batch', [U, R, W]],
  embedIndexed: ['gdn_eagle3_embed_indexed', 'embed', tiedLayout],
  select: ['gdn_eagle3_select', 'select_top1', [U, R, R, W]],
  // Fast tiers (not bit-identical to the reference kernels above).
  q4Fast: ['q4_gemv_fast', 'gemv', q4Layout],
  swigluFast: ['q4_gemv_fast', 'swiglu', q4Layout],
  scoresFast: ['gdn_eagle3_tied_fast', 'subset_logits', tiedLayout],
  sliceScoresFast: ['gdn_eagle3_tied_fast', 'subset_logits_slice', tiedLayout],
  q4x3Fast: ['q4_gemv_fast', 'gemv3', q4x3Layout],
  q4x3FastSg: ['q4_gemv_fast_sg', 'gemv3', q4x3Layout],
  q4FastSg: ['q4_gemv_fast_sg', 'gemv', q4Layout],
  swigluFastSg: ['q4_gemv_fast_sg', 'swiglu', q4Layout],
  scoresFastSg: ['gdn_eagle3_tied_fast_sg', 'subset_logits', tiedLayout],
  sliceScoresFastSg: ['gdn_eagle3_tied_fast_sg', 'subset_logits_slice', tiedLayout],
} as const satisfies Record<string, readonly [string, string, readonly Access[]]>;
type Kernel = keyof typeof shaders;

/** The assets an offscreen build must deploy alongside the host. */
export const draftShaderEntries = Object.values(shaders).map(([name, entryPoint]) => ({
  file: name + '.wgsl', entryPoint,
}));

/**
 * Kernel tier for the hot projections and subset logits.
 *  - 'reference': the bit-exact scalar kernels (CPU/WARP four-way parity path).
 *  - 'fast': word-parallel f32 kernels with workgroup reductions and native FMA (any WebGPU device).
 *  - 'fast-subgroups': the same with subgroupAdd (device feature 'subgroups', subgroup size 4..64).
 * Fast tiers agree with the reference to float rounding, not bit-for-bit.
 */
export type DraftKernelTier = 'reference' | 'fast' | 'fast-subgroups';
export interface DraftRuntimeOptions {
  /** 'auto' (default) picks the fastest tier the device supports; 'reference' keeps bit-exact kernels. */
  mode?: 'auto' | 'reference' | 'fast';
}
const TIER_KERNELS = {
  reference: { q4: 'q4', swiglu: 'swiglu', scores: 'scores', sliceScores: 'sliceScores', fused3: null, rowsPerGroup: 64 },
  fast: { q4: 'q4Fast', swiglu: 'swigluFast', scores: 'scoresFast', sliceScores: 'sliceScoresFast', fused3: 'q4x3Fast', rowsPerGroup: 4 },
  'fast-subgroups': { q4: 'q4FastSg', swiglu: 'swigluFastSg', scores: 'scoresFastSg', sliceScores: 'sliceScoresFastSg', fused3: 'q4x3FastSg', rowsPerGroup: 4 },
} as const satisfies Record<DraftKernelTier, { q4: Kernel; swiglu: Kernel; scores: Kernel; sliceScores: Kernel; fused3: Kernel | null; rowsPerGroup: number }>;
const SHARED_KERNELS: Kernel[] = ['rms', 'embed', 'conv', 'commitConv', 'pack', 'recur', 'commitState', 'gate', 'topK', 'embedIndexed', 'select'];

/** Choose the tier: subgroups need the device feature and a known subgroup size of at most 64. */
export function selectDraftTier(device: GPUDevice, mode: DraftRuntimeOptions['mode'] = 'auto'): DraftKernelTier {
  if (mode === 'reference') return 'reference';
  const features = (device as { features?: { has(name: string): boolean } }).features;
  const info = (device as { adapterInfo?: { subgroupMinSize?: number; subgroupMaxSize?: number } }).adapterInfo;
  const max = info?.subgroupMaxSize, min = info?.subgroupMinSize;
  const subgroupsOk = !!features?.has('subgroups') && typeof max === 'number' && max <= 64 && (min ?? max) >= 4;
  return subgroupsOk ? 'fast-subgroups' : 'fast';
}

export interface DraftAnchor {
  /** Concatenated post-residual hidden vectors in 3, 15, 31 order, before final norm.
   * These describe the committed prefix immediately BEFORE rootToken. */
  taps: GPUBuffer | Float32Array;
  rootToken: number;
}
interface TableBuffer {
  buffer: GPUBuffer;
  wordOffset?: number;
}
export type TiedSlice = TableBuffer & { startRow: number; endRow: number };
export type TiedTable = TableBuffer | { slices: readonly TiedSlice[] };
interface ResidentSlice extends TiedSlice {
  ids?: GPUBuffer;
  count: number;
}
export type ReadShader = (filename: string) => Promise<string>;

export function shaderAssets(directory: URL): ReadShader {
  return async (filename) => {
    const response = await fetch(new URL(filename, directory));
    if (!response.ok) throw new Error('Unable to load ' + filename + ' (HTTP ' + response.status + ')');
    return response.text();
  };
}

async function compile(device: GPUDevice, read: ReadShader, keys: readonly Kernel[]) {
  const modules = new Map<string, Promise<GPUShaderModule>>();
  const pipes = new Map<Kernel, GPUComputePipeline>();
  await Promise.all(
    keys.map(async (key) => {
      const [file, entryPoint, types] = shaders[key];
      let module = modules.get(file);
      if (!module) {
        module = read(file + '.wgsl').then((code) => device.createShaderModule({ label: file, code }));
        modules.set(file, module);
      }
      const layout = device.createBindGroupLayout({
        entries: types.map((type, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type },
        })),
      });
      pipes.set(
        key,
        await device.createComputePipelineAsync({
          label: file + '.' + entryPoint,
          layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: { module: await module, entryPoint },
        }),
      );
    }),
  );
  return pipes;
}

class Buffers {
  private readonly owned: GPUBuffer[] = [];
  constructor(private readonly device: GPUDevice) {}
  make(label: string, bytes: number, usage: GPUBufferUsageFlags, initial?: ArrayBufferView) {
    const buffer = this.device.createBuffer({ label, size: bytes, usage, mappedAtCreation: !!initial });
    this.owned.push(buffer);
    if (initial) {
      new Uint8Array(buffer.getMappedRange()).set(
        new Uint8Array(initial.buffer, initial.byteOffset, initial.byteLength),
      );
      buffer.unmap();
    }
    return buffer;
  }
  destroy() {
    for (const b of this.owned.splice(0)) b.destroy();
  }
}

/** Single-submit chain: slots[0] = root id, slots[1 + s] = chosen id at step s,
 * slots[CHAIN_RECORDS + 2s .. +1] = (chosen id, f32 top-1 probability). */
export const MAX_CHAIN = 15;
const CHAIN_RECORDS = 16;
const SLOT_WORDS = CHAIN_RECORDS + 2 * (MAX_CHAIN + 1);

export interface ChainProposal extends ScoredDraftTree {
  /** Raw cumulative top-1 path probability per node (product of per-step softmax maxima). */
  probs: number[];
  /** Per-step top-1 softmax probability over the 16,384-token subset. */
  conditionals: number[];
}
type ChainDispatch = { pipe: GPUComputePipeline; group: GPUBindGroup; x: number; y: number };
type ChainOp = { pass: ChainDispatch[] } | { copy: [GPUBuffer, GPUBuffer] } | { clear: GPUBuffer };
type ChainBuffer = 'fused' | 'mixed' | 'norm' | 'raw' | 'activated' | 'z' | 'ab' | 'packed' | 'y' | 'gated' | 'mlp' |
  'logits' | 'feature' | 'matrix' | 'history' | 'slots' | 'staging' | 'tapsHost' | 'subsetGlobal';
interface ChainState {
  buf: Record<ChainBuffer, GPUBuffer>;
  literals: Map<string, GPUBuffer>;
  plans: Map<string, ChainOp[]>;
  tapIds: Map<GPUBuffer, number>;
}

interface Branch {
  feature: GPUBuffer;
  history: GPUBuffer;
  matrix: GPUBuffer;
}
export class GdnDraftRuntime {
  private active = 0;
  private retired = false;
  private chain?: ChainState;
  private chainQueue: Promise<unknown> = Promise.resolve();
  /** Submits and readback bytes issued by the single-submit chain path (diagnostics). */
  readonly chainStats = { proposals: 0, submits: 0, readbackBytes: 0 };
  /** Kernel tier in use (see DraftKernelTier). */
  readonly tier: DraftKernelTier;
  private readonly k: (typeof TIER_KERNELS)[DraftKernelTier];
  private constructor(
    tier: DraftKernelTier,
    private readonly device: GPUDevice,
    private readonly bundle: GdnBundle,
    private readonly slices: ResidentSlice[],
    private readonly kernels: ReadonlyMap<Kernel, GPUComputePipeline>,
    private readonly owner: Buffers,
    private readonly weights: GPUBuffer,
    private readonly aux: GPUBuffer,
    private readonly convWeights: GPUBuffer,
  ) {
    this.tier = tier;
    this.k = TIER_KERNELS[tier];
  }

  static async create(device: GPUDevice, bundle: GdnBundle, table: TiedTable, readShader: ReadShader, options: DraftRuntimeOptions = {}) {
    const { hidden, vocab } = bundle.config;
    let tier = selectDraftTier(device, options.mode);
    // The fast kernels assume whole 32-column Q4 blocks.
    const q4Cols = ['fusion.weight', 'qkv.weight', 'z.weight', 'ab.weight', 'out.weight', 'gate.weight', 'up.weight', 'down.weight'].map((n) => bundle.tensor(n).shape[1]);
    if (tier !== 'reference' && (hidden % 32 || q4Cols.some((c) => c % 32))) {
      if (options.mode === 'fast') throw new Error('Fast draft kernels need Q4 widths that are multiples of 32');
      tier = 'reference';
    }
    const slices = 'slices' in table ? table.slices : [{ ...table, startRow: 0, endRow: vocab }];
    let nextRow = 0;
    for (const slice of slices) {
      const offset = slice.wordOffset ?? 0;
      const rows = slice.endRow - slice.startRow;
      if (slice.startRow !== nextRow || !Number.isSafeInteger(rows) || rows <= 0 ||
          !Number.isSafeInteger(offset) || offset < 0 ||
          offset * 4 + rows * Math.ceil(hidden / 32) * 20 > slice.buffer.size ||
          slice.buffer.size > device.limits.maxStorageBufferBindingSize)
        throw new Error('Invalid tied Q4 storage range');
      nextRow = slice.endRow;
    }
    if (nextRow !== vocab)
      throw new Error('Invalid tied Q4 storage range');
    const t = TIER_KERNELS[tier];
    const kernels = await compile(device, readShader, [...SHARED_KERNELS, t.q4, t.swiglu, t.scores, t.sliceScores, ...(t.fused3 ? [t.fused3] : [])]);
    const owner = new Buffers(device);
    const upload = (label: string, value: ArrayBufferView) =>
      owner.make(label, value.byteLength, GPUBufferUsage.STORAGE, value);
    try {
      const aux = Float32Array.from([
        ...Array.from(bundle.floats('log_rate'), (v) => -Math.exp(v)),
        ...bundle.floats('dt_bias'),
        ...bundle.floats('head_norm.weight'),
      ]);
      if (!aux.every(Number.isFinite)) throw new Error('Invalid draft recurrence auxiliary weights');
      const resident = slices.map((slice): ResidentSlice => {
        if (slices.length === 1)
          return { ...slice, ids: upload('draft subset ids', bundle.subsetIds), count: bundle.subsetIds.length };
        const pairs: number[] = [];
        bundle.subsetIds.forEach((id, index) => {
          if (id >= slice.startRow && id < slice.endRow) pairs.push(id - slice.startRow, index);
        });
        return { ...slice, count: pairs.length / 2, ids: pairs.length ? upload('draft subset slice', Uint32Array.from(pairs)) : undefined };
      });
      return new GdnDraftRuntime(
        tier,
        device,
        bundle,
        resident,
        kernels,
        owner,
        upload('draft weights', new Uint8Array(bundle.binary)),
        upload('draft aux', aux),
        upload('draft conv', bundle.floats('conv')),
      );
    } catch (error) {
      owner.destroy();
      throw error;
    }
  }

  /** Fast tiers: one dispatch for the QKV, Z and AB projections of the same input (same width, no residual). */
  private fused3(): { key: Kernel; params: number[]; groups: number } | null {
    if (!this.k.fused3) return null;
    const [q, z, ab] = ['qkv.weight', 'z.weight', 'ab.weight'].map((n) => this.bundle.tensor(n));
    if (q.shape[1] !== z.shape[1] || q.shape[1] !== ab.shape[1]) return null;
    const rows = q.shape[0] + z.shape[0] + ab.shape[0];
    return {
      key: this.k.fused3,
      params: [q.shape[0], q.shape[1], q.offset / 4, 0, z.shape[0], ab.shape[0], z.offset / 4, ab.offset / 4],
      groups: Math.ceil(rows / this.k.rowsPerGroup),
    };
  }

  private checkAnchor(anchor: DraftAnchor) {
    const { config } = this.bundle;
    if (
      !Number.isInteger(anchor.rootToken) ||
      anchor.rootToken < 0 ||
      anchor.rootToken >= config.vocab ||
      (anchor.taps instanceof Float32Array
        ? anchor.taps.byteLength !== 12 * config.hidden
        : anchor.taps.size < 12 * config.hidden)
    )
      throw new Error('Invalid draft anchor');
  }

  /**
   * Greedy width-1 chain of `k` draft steps in ONE command buffer with ONE
   * 8k-byte readback. The chosen token feeds the next step's embedding on
   * the device (gdn_eagle3_select -> gdn_eagle3_embed_indexed), and each step
   * reports its top-1 softmax probability so the WASM cutoff can trim the
   * chain to k* before target verification. Scratch, state, uniforms and
   * bind groups persist across proposals; proposals on one runtime are
   * serialized. Arithmetic is identical to the multi-submit path at width 1.
   */
  proposeChain(anchor: DraftAnchor, k: number): Promise<ChainProposal> {
    if (this.retired) return Promise.reject(new Error('Draft runtime has been released'));
    if (!Number.isInteger(k) || k < 0 || k > MAX_CHAIN) return Promise.reject(new Error('Invalid draft chain length'));
    try {
      this.checkAnchor(anchor);
    } catch (error) {
      return Promise.reject(error);
    }
    if (!k) return Promise.resolve({ tokens: [], parents: [], probs: [], conditionals: [] });
    const run = this.chainQueue.then(() => this.runChain(anchor, k));
    this.chainQueue = run.catch(() => {});
    return run;
  }

  private allocChain(): ChainState {
    const { hidden: h, heads, dim, intermediate: ff, subset } = this.bundle.config;
    const a = heads * dim;
    const rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label: string, words: number, usage = rw) => this.owner.make('chain ' + label, 4 * words, usage);
    const single = this.slices.length === 1 ? this.slices[0].ids : undefined;
    return {
      buf: {
        fused: make('fused taps', h), mixed: make('mixed', h), norm: make('normalized', h),
        raw: make('raw qkv', 3 * a), activated: make('conv qkv', 3 * a), z: make('z', a), ab: make('ab', 2 * heads),
        packed: make('rec inputs', heads * (3 * dim + 2)), y: make('raw readout', a), gated: make('gated readout', a),
        mlp: make('swiglu', ff), logits: make('draft logits', subset),
        feature: make('feature', h), matrix: make('matrix', heads * dim * dim), history: make('history', 9 * a),
        slots: make('slots', SLOT_WORDS),
        staging: make('readback', 2 * (MAX_CHAIN + 1), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        tapsHost: make('host taps', 3 * h, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
        subsetGlobal: single ??
          this.owner.make('draft subset global', this.bundle.subsetIds.byteLength, GPUBufferUsage.STORAGE, this.bundle.subsetIds),
      },
      literals: new Map(),
      plans: new Map(),
      tapIds: new Map(),
    };
  }

  private buildChain(st: ChainState, taps: GPUBuffer, k: number): ChainOp[] {
    const { hidden: h, heads, dim, subset } = this.bundle.config;
    const channels = 3 * heads * dim;
    const b = st.buf;
    const literal = (data: number[], uniform = false) => {
      const key = (uniform ? 'u' : 's') + data.join(',');
      let buffer = st.literals.get(key);
      if (!buffer) {
        const words = Uint32Array.from(data);
        buffer = this.owner.make('chain ' + key, words.byteLength, uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE, words);
        st.literals.set(key, buffer);
      }
      return buffer;
    };
    const ops: ChainOp[] = [];
    let pass: ChainDispatch[] | undefined;
    const run = (key: Kernel, params: number[], buffers: GPUBuffer[], x: number, y = 1) => {
      const pipe = this.kernels.get(key)!;
      const group = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [literal(params, true), ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      if (!pass) ops.push({ pass: (pass = []) });
      pass.push({ pipe, group, x, y });
    };
    const copy = (src: GPUBuffer, dst: GPUBuffer) => {
      pass = undefined;
      ops.push({ copy: [src, dst] });
    };
    const linear = (name: string, input: GPUBuffer, out: GPUBuffer, residual = 0, up?: string) => {
      const e = this.bundle.tensor(name);
      const [rows, cols] = e.shape;
      run(up ? this.k.swiglu : this.k.q4, [rows, cols, e.offset / 4, 0, 0, residual, 1, up ? this.bundle.tensor(up).offset / 4 : 0],
        [this.weights, input, out], Math.ceil(rows / this.k.rowsPerGroup));
    };
    const rms = (name: string, input: GPUBuffer) =>
      run('rms', [h, 1, this.bundle.tensor(name).offset / 4, 0], [input, this.weights, b.norm], 1);
    const mask = literal([1]),
      choice = literal([0, 0, 0]);
    // head.initial(): zero recurrent matrix and convolution history.
    ops.push({ clear: b.matrix }, { clear: b.history });
    linear('fusion.weight', taps, b.fused);
    for (let step = 0; step < k; step++) {
      for (const slice of this.slices)
        run('embedIndexed', [h, slice.startRow, slice.endRow, slice.wordOffset ?? 0, step, 0, 0, 0],
          [slice.buffer, b.slots, step ? b.feature : b.fused, b.mixed], Math.ceil(h / 64));
      rms('input_norm.weight', b.mixed);
      const f3 = this.fused3();
      if (f3) run(f3.key, f3.params, [this.weights, b.norm, b.raw, b.z, b.ab], f3.groups);
      else {
        linear('qkv.weight', b.norm, b.raw);
        linear('z.weight', b.norm, b.z);
        linear('ab.weight', b.norm, b.ab);
      }
      const conv = [mask, b.raw, this.convWeights, b.history, b.activated, choice];
      const io = [b.activated, b.ab, this.aux, b.packed, b.y, b.z, b.gated];
      const dn = [mask, b.packed, b.matrix, b.y, choice];
      run('conv', [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
      run('pack', [heads, heads, dim, 1], io, heads);
      run('recur', [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
      run('gate', [heads, heads, dim, 1], io, heads);
      run('commitState', [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
      run('commitConv', [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
      copy(b.mixed, b.feature);
      linear('out.weight', b.gated, b.feature, 1);
      rms('ff_norm.weight', b.feature);
      linear('gate.weight', b.norm, b.mlp, 0, 'up.weight');
      linear('down.weight', b.mlp, b.feature, 1);
      rms('final_norm.weight', b.feature);
      for (const slice of this.slices) {
        if (slice.ids)
          run(this.slices.length === 1 ? this.k.scores : this.k.sliceScores, [h, 1, slice.count, slice.wordOffset ?? 0],
            [slice.buffer, slice.ids, b.norm, b.logits], Math.ceil(slice.count / this.k.rowsPerGroup));
      }
      run('select', [subset, step, CHAIN_RECORDS, 0], [b.logits, b.subsetGlobal, b.slots], 1);
    }
    return ops;
  }

  private async runChain(anchor: DraftAnchor, k: number): Promise<ChainProposal> {
    if (this.retired) throw new Error('Draft runtime has been released');
    this.active++;
    try {
      const st = (this.chain ??= this.allocChain());
      const { queue } = this.device;
      let taps = st.buf.tapsHost;
      if (anchor.taps instanceof Float32Array)
        queue.writeBuffer(taps, 0, anchor.taps.buffer as ArrayBuffer, anchor.taps.byteOffset, anchor.taps.byteLength);
      else taps = anchor.taps;
      let id = st.tapIds.get(taps);
      if (id === undefined) st.tapIds.set(taps, (id = st.tapIds.size));
      const key = k + '|' + id;
      let plan = st.plans.get(key);
      if (!plan) st.plans.set(key, (plan = this.buildChain(st, taps, k)));
      queue.writeBuffer(st.buf.slots, 0, Uint32Array.of(anchor.rootToken));
      const encoder = this.device.createCommandEncoder();
      for (const op of plan) {
        if ('clear' in op) encoder.clearBuffer(op.clear);
        else if ('copy' in op) encoder.copyBufferToBuffer(op.copy[0], 0, op.copy[1], 0, op.copy[1].size);
        else {
          const pass = encoder.beginComputePass();
          for (const d of op.pass) {
            pass.setPipeline(d.pipe);
            pass.setBindGroup(0, d.group);
            pass.dispatchWorkgroups(d.x, d.y, 1);
          }
          pass.end();
        }
      }
      const bytes = 8 * k;
      encoder.copyBufferToBuffer(st.buf.slots, 4 * CHAIN_RECORDS, st.buf.staging, 0, bytes);
      queue.submit([encoder.finish()]);
      this.chainStats.proposals++;
      this.chainStats.submits++;
      this.chainStats.readbackBytes += bytes;
      await st.buf.staging.mapAsync(GPUMapMode.READ, 0, bytes);
      const words = new Uint32Array(st.buf.staging.getMappedRange(0, bytes).slice(0, bytes));
      st.buf.staging.unmap();
      const probBits = new Float32Array(words.buffer);
      const out: ChainProposal = { tokens: [], parents: [], probs: [], conditionals: [] };
      let path = 1;
      for (let s = 0; s < k; s++) {
        const token = words[2 * s];
        const cond = probBits[2 * s + 1];
        if (token >= this.bundle.config.vocab || !(cond > 0 && cond <= 1.0000001)) throw new Error('Invalid draft chain record');
        path *= Math.min(1, cond);
        out.tokens.push(token);
        out.parents.push(s - 1);
        out.conditionals.push(cond);
        out.probs.push(path);
      }
      return out;
    } finally {
      this.active--;
      if (this.retired && !this.active) this.owner.destroy();
    }
  }

  /** Each anchor starts at head.initial(), as in the training/evaluation unroll.
   * Only top-k records cross to the host. Siblings inherit independent copies
   * of the parent state. All head scratch dies with this proposal. */
  async propose(anchor: DraftAnchor, budget: DraftBudget, branchWidth = 2, singleSubmit = true): Promise<ScoredDraftTree> {
    if (this.retired) throw new Error('Draft runtime has been released');
    const { maxNodes } = budget,
      { config } = this.bundle;
    if (
      !Number.isInteger(maxNodes) ||
      maxNodes < 0 ||
      maxNodes > 15 ||
      !Number.isInteger(branchWidth) ||
      branchWidth < 1 ||
      branchWidth > 64
    )
      throw new Error('Invalid draft tree budget');
    if (
      !Number.isInteger(anchor.rootToken) ||
      anchor.rootToken < 0 ||
      anchor.rootToken >= config.vocab ||
      (anchor.taps instanceof Float32Array
        ? anchor.taps.byteLength !== 12 * config.hidden
        : anchor.taps.size < 12 * config.hidden)
    ) {
      throw new Error('Invalid draft anchor');
    }
    const tree: ScoredDraftTree = { tokens: [], parents: [] };
    if (!maxNodes) return tree;
    const width = Math.min(budget.tree ? branchWidth : 1, config.subset, maxNodes);
    // Width-1 chains take the single-submit path (one command buffer, one readback).
    if (width === 1 && singleSubmit) return this.proposeChain(anchor, maxNodes);
    const local = new Buffers(this.device);
    this.active++;
    try {
      const { hidden: h, heads, dim, intermediate: ff, subset } = config;
      const a = heads * dim,
        channels = 3 * a,
        storage = GPUBufferUsage.STORAGE;
      const rw = storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const scratch = (label: string, n: number) => local.make(label, 4 * n, rw);
      const constants = new Map<string, GPUBuffer>();
      const literal = (data: number[], uniform = false) => {
        const key = (uniform ? 'u' : 's') + data.join(',');
        let b = constants.get(key);
        if (!b) {
          const words = Uint32Array.from(data);
          b = local.make('draft ' + key, words.byteLength, uniform ? GPUBufferUsage.UNIFORM : storage, words);
          constants.set(key, b);
        }
        return b;
      };
      const taps =
        anchor.taps instanceof Float32Array
          ? local.make('target taps', anchor.taps.byteLength, storage, anchor.taps)
          : anchor.taps;
      const fused = scratch('fused taps', h),
        mixed = scratch('mixed', h),
        norm = scratch('normalized', h);
      const raw = scratch('raw qkv', channels),
        activated = scratch('conv qkv', channels),
        z = scratch('z', a);
      const ab = scratch('ab', 2 * heads),
        packed = scratch('rec inputs', heads * (3 * dim + 2));
      const y = scratch('raw readout', a),
        gated = scratch('gated readout', a),
        mlp = scratch('swiglu', ff);
      const logits = scratch('draft logits', subset),
        ranked = scratch('top-k records', 2 * width);
      const staging = local.make(
        'draft readback',
        8 * width,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
      const mask = literal([1]),
        choice = literal([0, 0, 0]);
      const queue: { token: number; node: number; prior?: Branch }[] = [{ token: anchor.rootToken, node: -1 }];
      for (let i = 0; i < queue.length && tree.tokens.length < maxNodes; i++) {
        const { token, node, prior } = queue[i];
        const encoder = this.device.createCommandEncoder();
        const copy = (src: GPUBuffer, dst: GPUBuffer) => encoder.copyBufferToBuffer(src, 0, dst, 0, dst.size);
        const run = (key: Kernel, params: number[], buffers: GPUBuffer[], x: number, groupsY = 1) => {
          const pipe = this.kernels.get(key)!;
          const group = this.device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [literal(params, true), ...buffers].map((buffer, binding) => ({
              binding,
              resource: { buffer },
            })),
          });
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipe);
          pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(x, groupsY, 1);
          pass.end();
        };
        const linear = (name: string, input: GPUBuffer, out: GPUBuffer, residual = 0, up?: string) => {
          const e = this.bundle.tensor(name);
          const [rows, cols] = e.shape;
          run(
            up ? this.k.swiglu : this.k.q4,
            [rows, cols, e.offset / 4, 0, 0, residual, 1, up ? this.bundle.tensor(up).offset / 4 : 0],
            [this.weights, input, out],
            Math.ceil(rows / this.k.rowsPerGroup),
          );
        };
        const rms = (name: string, input: GPUBuffer) =>
          run('rms', [h, 1, this.bundle.tensor(name).offset / 4, 0], [input, this.weights, norm], 1);
        const next: Branch = {
          feature: scratch('branch feature', h),
          matrix: scratch('branch matrix', heads * dim * dim),
          history: scratch('branch history', channels * 3),
        };
        if (prior) {
          copy(prior.matrix, next.matrix);
          copy(prior.history, next.history);
        } else {
          linear('fusion.weight', taps, fused);
        }
        const embedding = this.slices.find((s) => token >= s.startRow && token < s.endRow)!;
        run(
          'embed',
          [h, 1, subset, embedding.wordOffset ?? 0],
          [embedding.buffer, literal([token - embedding.startRow]), prior?.feature ?? fused, mixed],
          Math.ceil(h / 64),
        );
        rms('input_norm.weight', mixed);
        const f3 = this.fused3();
        if (f3) run(f3.key, f3.params, [this.weights, norm, raw, z, ab], f3.groups);
        else {
          linear('qkv.weight', norm, raw);
          linear('z.weight', norm, z);
          linear('ab.weight', norm, ab);
        }
        const conv = [mask, raw, this.convWeights, next.history, activated, choice];
        const io = [activated, ab, this.aux, packed, y, z, gated];
        const dn = [mask, packed, next.matrix, y, choice];
        run('conv', [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
        run('pack', [heads, heads, dim, 1], io, heads);
        run('recur', [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
        run('gate', [heads, heads, dim, 1], io, heads);
        run('commitState', [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
        run('commitConv', [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
        copy(mixed, next.feature);
        linear('out.weight', gated, next.feature, 1);
        rms('ff_norm.weight', next.feature);
        linear('gate.weight', norm, mlp, 0, 'up.weight');
        linear('down.weight', mlp, next.feature, 1);
        rms('final_norm.weight', next.feature);
        for (const slice of this.slices) {
          if (slice.ids) run(this.slices.length === 1 ? this.k.scores : this.k.sliceScores,
            [h, 1, slice.count, slice.wordOffset ?? 0], [slice.buffer, slice.ids, norm, logits], Math.ceil(slice.count / this.k.rowsPerGroup));
        }
        run('topK', [subset, subset, width, 0], [logits, ranked], 1);
        copy(ranked, staging);
        this.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        try {
          const records = new Uint32Array(staging.getMappedRange());
          for (let r = 0; r < width && tree.tokens.length < maxNodes; r++) {
            const idx = records[2 * r];
            if (idx >= subset) throw new Error('Invalid draft top-k record');
            const child = this.bundle.subsetIds[idx];
            queue.push({ token: child, node: tree.tokens.length, prior: next });
            tree.tokens.push(child);
            tree.parents.push(node);
          }
        } finally {
          staging.unmap();
        }
      }
      return tree;
    } finally {
      local.destroy();
      this.active--;
      if (this.retired && !this.active) this.owner.destroy();
    }
  }

  dispose() {
    this.retired = true;
    if (!this.active) this.owner.destroy();
  }
}
