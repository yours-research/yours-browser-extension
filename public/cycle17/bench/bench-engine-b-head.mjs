// Engine B (EXP-007) head + reject-surcharge microbench. Generated from src/bench/engineBHeadBenchPage.ts; see docs/CYCLE17_ROLLBACK_PATCHES.md.

// src/draft/gdnEagle3Loader.ts
var EXP007_CONFIG = Object.freeze({
  hidden: 2560,
  heads: 8,
  dim: 128,
  intermediate: 3072,
  vocab: 81920,
  subset: 16384,
  taps: 4
});
function record(value) {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid draft manifest object");
  return value;
}
function sameShape(actual, expected) {
  return Array.isArray(actual) && actual.length === expected.length && expected.every((v, i) => actual[i] === v);
}
function validateManifest(raw, expected) {
  const manifest = record(raw);
  if (manifest.format !== "gdn-eagle3-draft" || manifest.format_version !== 1 || manifest.byte_order !== "little" || manifest.file !== "draft_head.bin")
    throw new Error("Unsupported draft bundle format");
  const cfg = record(manifest.config);
  for (const [k, v] of Object.entries(expected)) {
    if (cfg[k] !== v || !Number.isSafeInteger(v) || v < 1)
      throw new Error("Incompatible draft dimension: " + k);
  }
  const { hidden: h, heads, dim, intermediate: ff, subset, vocab, taps } = expected;
  if (taps !== 4 || dim > 128 || subset > vocab || !sameShape(manifest.layer_taps, [3, 15, 31]) || manifest.subset_size !== subset)
    throw new Error("Incompatible draft feature or subset layout");
  const a = heads * dim;
  const shapes = {
    log_rate: [heads],
    dt_bias: [heads],
    conv: [3 * a, taps],
    "fusion.weight": [h, 3 * h],
    "input_norm.weight": [h],
    "qkv.weight": [3 * a, h],
    "z.weight": [a, h],
    "ab.weight": [2 * heads, h],
    "head_norm.weight": [dim],
    "out.weight": [h, a],
    "ff_norm.weight": [h],
    "gate.weight": [ff, h],
    "up.weight": [ff, h],
    "down.weight": [h, ff],
    "final_norm.weight": [h],
    subset_ids: [subset]
  };
  const entries = /* @__PURE__ */ new Map();
  let offset = 0;
  if (!Array.isArray(manifest.tensors)) throw new Error("Missing draft tensor directory");
  for (const item of manifest.tensors) {
    const entry = record(item), name = entry.name;
    if (typeof name !== "string" || !Object.hasOwn(shapes, name) || entries.has(name))
      throw new Error("Unexpected or duplicate draft tensor");
    const shape = shapes[name];
    const kind = name === "subset_ids" ? "u32" : shape.length === 2 && name !== "conv" ? "q4" : "f32";
    const blocks = Math.ceil(shape[1] / 32);
    const length = kind === "q4" ? shape[0] * blocks * 20 : 4 * shape.reduce((v, d) => v * d, 1);
    if (!sameShape(entry.shape, shape) || entry.kind !== kind || entry.offset !== offset || entry.byte_length !== length || kind === "q4" && (entry.blocks_per_row !== blocks || entry.padded_cols !== blocks * 32)) {
      throw new Error("Invalid draft tensor layout: " + name);
    }
    entries.set(name, { shape, kind, offset, byte_length: length });
    offset += length;
  }
  if (entries.size !== Object.keys(shapes).length || offset !== manifest.total_bytes || offset > 40 * 2 ** 20) {
    throw new Error("Incomplete draft bundle or incorrect byte count");
  }
  if (typeof manifest.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.sha256))
    throw new Error("Invalid draft SHA-256");
  return { entries, config: Object.freeze({ ...expected }), size: offset, sha256: manifest.sha256 };
}
async function validatePayload(layout, binary) {
  if (binary.byteLength !== layout.size) throw new Error("Incorrect draft payload length");
  const hash = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", binary));
  const hex = Array.from(hash, (b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== layout.sha256) throw new Error("Draft SHA-256 mismatch");
  function tensor(name) {
    const entry = layout.entries.get(name);
    if (!entry) throw new Error("Unknown draft tensor: " + name);
    return entry;
  }
  function words(name) {
    const e = tensor(name);
    return new Uint32Array(binary, e.offset, e.byte_length / 4);
  }
  function floats(name) {
    const e = tensor(name);
    if (e.kind !== "f32") throw new Error("Expected f32 draft tensor: " + name);
    return new Float32Array(binary, e.offset, e.byte_length / 4);
  }
  const subsetIds = words("subset_ids");
  const seen = /* @__PURE__ */ new Set();
  for (const id of subsetIds) {
    if (id >= layout.config.vocab || seen.has(id)) throw new Error("Invalid draft vocabulary subset");
    seen.add(id);
  }
  return { config: layout.config, binary, sha256: hex, subsetIds, tensor, words, floats };
}
async function loadGdnBundle(manifestUrl, options = {}) {
  const get = options.fetch ?? globalThis.fetch;
  async function optional(url) {
    const response2 = await get(url, { signal: options.signal });
    if (response2.status === 404 || response2.status === 410) return null;
    if (!response2.ok) throw new Error("Draft fetch failed with HTTP " + response2.status);
    return response2;
  }
  const manifest = await optional(manifestUrl);
  if (!manifest) return null;
  const layout = validateManifest(await manifest.json(), options.expectedConfig ?? EXP007_CONFIG);
  const response = await optional(new URL("draft_head.bin", manifest.url || manifestUrl));
  if (!response) return null;
  const binary = new ArrayBuffer(layout.size);
  const dst = new Uint8Array(binary);
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Missing draft response body");
  let offset = 0;
  try {
    for (; ; ) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.byteLength > dst.length) {
        await reader.cancel();
        throw new Error("Draft payload exceeds its declared length");
      }
      dst.set(value, offset);
      offset += value.byteLength;
    }
  } finally {
    reader.releaseLock();
  }
  if (offset !== dst.length) throw new Error("Truncated draft payload");
  return validatePayload(layout, binary);
}

// src/draft/gdnEagle3Runtime.ts
var U = "uniform";
var R = "read-only-storage";
var W = "storage";
var q4Layout = [U, R, R, W];
var q4x3Layout = [U, R, R, W, W, W];
var tiedLayout = [U, R, R, R, W];
var convLayout = [U, R, R, R, W, W, R];
var ioLayout = [U, R, R, R, W, R, R, W];
var dnLayout = [U, R, R, W, W, R];
var shaders = {
  q4: ["q4_gemv_smallm_v2", "main", q4Layout],
  swiglu: ["q4_gate_up_swiglu_smallm_v2", "main", q4Layout],
  rms: ["gdn_eagle3_rms", "main", q4Layout],
  embed: ["gdn_eagle3_tied", "embed", tiedLayout],
  scores: ["gdn_eagle3_tied", "subset_logits", tiedLayout],
  sliceScores: ["gdn_eagle3_tied", "subset_logits_slice", tiedLayout],
  conv: ["DN_CONV_TREE", "verify", convLayout],
  commitConv: ["DN_CONV_TREE", "commit", convLayout],
  pack: ["DN_IO", "pack", ioLayout],
  recur: ["DN_TREE", "verify", dnLayout],
  commitState: ["DN_TREE", "commit", dnLayout],
  gate: ["DN_IO", "gate", ioLayout],
  topK: ["TOPK_BATCH", "topk_batch", [U, R, W]],
  embedIndexed: ["gdn_eagle3_embed_indexed", "embed", tiedLayout],
  select: ["gdn_eagle3_select", "select_top1", [U, R, R, W]],
  // Fast tiers (not bit-identical to the reference kernels above).
  q4Fast: ["q4_gemv_fast", "gemv", q4Layout],
  swigluFast: ["q4_gemv_fast", "swiglu", q4Layout],
  scoresFast: ["gdn_eagle3_tied_fast", "subset_logits", tiedLayout],
  sliceScoresFast: ["gdn_eagle3_tied_fast", "subset_logits_slice", tiedLayout],
  q4x3Fast: ["q4_gemv_fast", "gemv3", q4x3Layout],
  q4x3FastSg: ["q4_gemv_fast_sg", "gemv3", q4x3Layout],
  q4FastSg: ["q4_gemv_fast_sg", "gemv", q4Layout],
  swigluFastSg: ["q4_gemv_fast_sg", "swiglu", q4Layout],
  scoresFastSg: ["gdn_eagle3_tied_fast_sg", "subset_logits", tiedLayout],
  sliceScoresFastSg: ["gdn_eagle3_tied_fast_sg", "subset_logits_slice", tiedLayout]
};
var draftShaderEntries = Object.values(shaders).map(([name, entryPoint]) => ({
  file: name + ".wgsl",
  entryPoint
}));
var TIER_KERNELS = {
  reference: { q4: "q4", swiglu: "swiglu", scores: "scores", sliceScores: "sliceScores", fused3: null, rowsPerGroup: 64 },
  fast: { q4: "q4Fast", swiglu: "swigluFast", scores: "scoresFast", sliceScores: "sliceScoresFast", fused3: "q4x3Fast", rowsPerGroup: 4 },
  "fast-subgroups": { q4: "q4FastSg", swiglu: "swigluFastSg", scores: "scoresFastSg", sliceScores: "sliceScoresFastSg", fused3: "q4x3FastSg", rowsPerGroup: 4 }
};
var SHARED_KERNELS = ["rms", "embed", "conv", "commitConv", "pack", "recur", "commitState", "gate", "topK", "embedIndexed", "select"];
function selectDraftTier(device, mode = "auto") {
  if (mode === "reference") return "reference";
  const features = device.features;
  const info = device.adapterInfo;
  const max = info?.subgroupMaxSize, min = info?.subgroupMinSize;
  const subgroupsOk = !!features?.has("subgroups") && typeof max === "number" && max <= 64 && (min ?? max) >= 4;
  return subgroupsOk ? "fast-subgroups" : "fast";
}
function shaderAssets(directory) {
  return async (filename) => {
    const response = await fetch(new URL(filename, directory));
    if (!response.ok) throw new Error("Unable to load " + filename + " (HTTP " + response.status + ")");
    return response.text();
  };
}
async function compile(device, read, keys) {
  const modules = /* @__PURE__ */ new Map();
  const pipes = /* @__PURE__ */ new Map();
  await Promise.all(
    keys.map(async (key) => {
      const [file, entryPoint, types] = shaders[key];
      let module = modules.get(file);
      if (!module) {
        module = read(file + ".wgsl").then((code) => device.createShaderModule({ label: file, code }));
        modules.set(file, module);
      }
      const layout = device.createBindGroupLayout({
        entries: types.map((type, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type }
        }))
      });
      pipes.set(
        key,
        await device.createComputePipelineAsync({
          label: file + "." + entryPoint,
          layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: { module: await module, entryPoint }
        })
      );
    })
  );
  return pipes;
}
var Buffers = class {
  constructor(device) {
    this.device = device;
  }
  device;
  owned = [];
  make(label, bytes, usage, initial) {
    const buffer = this.device.createBuffer({ label, size: bytes, usage, mappedAtCreation: !!initial });
    this.owned.push(buffer);
    if (initial) {
      new Uint8Array(buffer.getMappedRange()).set(
        new Uint8Array(initial.buffer, initial.byteOffset, initial.byteLength)
      );
      buffer.unmap();
    }
    return buffer;
  }
  destroy() {
    for (const b of this.owned.splice(0)) b.destroy();
  }
};
var MAX_CHAIN = 15;
var CHAIN_RECORDS = 16;
var SLOT_WORDS = CHAIN_RECORDS + 2 * (MAX_CHAIN + 1);
var GdnDraftRuntime = class _GdnDraftRuntime {
  constructor(tier, device, bundle, slices, kernels, owner, weights, aux, convWeights) {
    this.device = device;
    this.bundle = bundle;
    this.slices = slices;
    this.kernels = kernels;
    this.owner = owner;
    this.weights = weights;
    this.aux = aux;
    this.convWeights = convWeights;
    this.tier = tier;
    this.k = TIER_KERNELS[tier];
  }
  device;
  bundle;
  slices;
  kernels;
  owner;
  weights;
  aux;
  convWeights;
  active = 0;
  retired = false;
  chain;
  chainQueue = Promise.resolve();
  /** Submits and readback bytes issued by the single-submit chain path (diagnostics). */
  chainStats = { proposals: 0, submits: 0, readbackBytes: 0 };
  /** Kernel tier in use (see DraftKernelTier). */
  tier;
  k;
  static async create(device, bundle, table, readShader, options = {}) {
    const { hidden, vocab } = bundle.config;
    let tier = selectDraftTier(device, options.mode);
    const q4Cols = ["fusion.weight", "qkv.weight", "z.weight", "ab.weight", "out.weight", "gate.weight", "up.weight", "down.weight"].map((n) => bundle.tensor(n).shape[1]);
    if (tier !== "reference" && (hidden % 32 || q4Cols.some((c) => c % 32))) {
      if (options.mode === "fast") throw new Error("Fast draft kernels need Q4 widths that are multiples of 32");
      tier = "reference";
    }
    const slices = "slices" in table ? table.slices : [{ ...table, startRow: 0, endRow: vocab }];
    let nextRow = 0;
    for (const slice of slices) {
      const offset = slice.wordOffset ?? 0;
      const rows = slice.endRow - slice.startRow;
      if (slice.startRow !== nextRow || !Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(offset) || offset < 0 || offset * 4 + rows * Math.ceil(hidden / 32) * 20 > slice.buffer.size || slice.buffer.size > device.limits.maxStorageBufferBindingSize)
        throw new Error("Invalid tied Q4 storage range");
      nextRow = slice.endRow;
    }
    if (nextRow !== vocab)
      throw new Error("Invalid tied Q4 storage range");
    const t = TIER_KERNELS[tier];
    const kernels = await compile(device, readShader, [...SHARED_KERNELS, t.q4, t.swiglu, t.scores, t.sliceScores, ...t.fused3 ? [t.fused3] : []]);
    const owner = new Buffers(device);
    const upload = (label, value) => owner.make(label, value.byteLength, GPUBufferUsage.STORAGE, value);
    try {
      const aux = Float32Array.from([
        ...Array.from(bundle.floats("log_rate"), (v) => -Math.exp(v)),
        ...bundle.floats("dt_bias"),
        ...bundle.floats("head_norm.weight")
      ]);
      if (!aux.every(Number.isFinite)) throw new Error("Invalid draft recurrence auxiliary weights");
      const resident = slices.map((slice) => {
        if (slices.length === 1)
          return { ...slice, ids: upload("draft subset ids", bundle.subsetIds), count: bundle.subsetIds.length };
        const pairs = [];
        bundle.subsetIds.forEach((id, index) => {
          if (id >= slice.startRow && id < slice.endRow) pairs.push(id - slice.startRow, index);
        });
        return { ...slice, count: pairs.length / 2, ids: pairs.length ? upload("draft subset slice", Uint32Array.from(pairs)) : void 0 };
      });
      return new _GdnDraftRuntime(
        tier,
        device,
        bundle,
        resident,
        kernels,
        owner,
        upload("draft weights", new Uint8Array(bundle.binary)),
        upload("draft aux", aux),
        upload("draft conv", bundle.floats("conv"))
      );
    } catch (error) {
      owner.destroy();
      throw error;
    }
  }
  /** Fast tiers: one dispatch for the QKV, Z and AB projections of the same input (same width, no residual). */
  fused3() {
    if (!this.k.fused3) return null;
    const [q, z, ab] = ["qkv.weight", "z.weight", "ab.weight"].map((n) => this.bundle.tensor(n));
    if (q.shape[1] !== z.shape[1] || q.shape[1] !== ab.shape[1]) return null;
    const rows = q.shape[0] + z.shape[0] + ab.shape[0];
    return {
      key: this.k.fused3,
      params: [q.shape[0], q.shape[1], q.offset / 4, 0, z.shape[0], ab.shape[0], z.offset / 4, ab.offset / 4],
      groups: Math.ceil(rows / this.k.rowsPerGroup)
    };
  }
  checkAnchor(anchor) {
    const { config } = this.bundle;
    if (!Number.isInteger(anchor.rootToken) || anchor.rootToken < 0 || anchor.rootToken >= config.vocab || (anchor.taps instanceof Float32Array ? anchor.taps.byteLength !== 12 * config.hidden : anchor.taps.size < 12 * config.hidden))
      throw new Error("Invalid draft anchor");
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
  proposeChain(anchor, k) {
    if (this.retired) return Promise.reject(new Error("Draft runtime has been released"));
    if (!Number.isInteger(k) || k < 0 || k > MAX_CHAIN) return Promise.reject(new Error("Invalid draft chain length"));
    try {
      this.checkAnchor(anchor);
    } catch (error) {
      return Promise.reject(error);
    }
    if (!k) return Promise.resolve({ tokens: [], parents: [], probs: [], conditionals: [] });
    const run = this.chainQueue.then(() => this.runChain(anchor, k));
    this.chainQueue = run.catch(() => {
    });
    return run;
  }
  allocChain() {
    const { hidden: h, heads, dim, intermediate: ff, subset } = this.bundle.config;
    const a = heads * dim;
    const rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label, words, usage = rw) => this.owner.make("chain " + label, 4 * words, usage);
    const single = this.slices.length === 1 ? this.slices[0].ids : void 0;
    return {
      buf: {
        fused: make("fused taps", h),
        mixed: make("mixed", h),
        norm: make("normalized", h),
        raw: make("raw qkv", 3 * a),
        activated: make("conv qkv", 3 * a),
        z: make("z", a),
        ab: make("ab", 2 * heads),
        packed: make("rec inputs", heads * (3 * dim + 2)),
        y: make("raw readout", a),
        gated: make("gated readout", a),
        mlp: make("swiglu", ff),
        logits: make("draft logits", subset),
        feature: make("feature", h),
        matrix: make("matrix", heads * dim * dim),
        history: make("history", 9 * a),
        slots: make("slots", SLOT_WORDS),
        staging: make("readback", 2 * (MAX_CHAIN + 1), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        tapsHost: make("host taps", 3 * h, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
        subsetGlobal: single ?? this.owner.make("draft subset global", this.bundle.subsetIds.byteLength, GPUBufferUsage.STORAGE, this.bundle.subsetIds)
      },
      literals: /* @__PURE__ */ new Map(),
      plans: /* @__PURE__ */ new Map(),
      tapIds: /* @__PURE__ */ new Map()
    };
  }
  buildChain(st, taps, k) {
    const { hidden: h, heads, dim, subset } = this.bundle.config;
    const channels = 3 * heads * dim;
    const b = st.buf;
    const literal = (data, uniform = false) => {
      const key = (uniform ? "u" : "s") + data.join(",");
      let buffer = st.literals.get(key);
      if (!buffer) {
        const words = Uint32Array.from(data);
        buffer = this.owner.make("chain " + key, words.byteLength, uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE, words);
        st.literals.set(key, buffer);
      }
      return buffer;
    };
    const ops = [];
    let pass;
    const run = (key, params, buffers, x, y = 1) => {
      const pipe = this.kernels.get(key);
      const group = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [literal(params, true), ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } }))
      });
      if (!pass) ops.push({ pass: pass = [] });
      pass.push({ pipe, group, x, y });
    };
    const copy = (src, dst) => {
      pass = void 0;
      ops.push({ copy: [src, dst] });
    };
    const linear = (name, input, out, residual = 0, up) => {
      const e = this.bundle.tensor(name);
      const [rows, cols] = e.shape;
      run(
        up ? this.k.swiglu : this.k.q4,
        [rows, cols, e.offset / 4, 0, 0, residual, 1, up ? this.bundle.tensor(up).offset / 4 : 0],
        [this.weights, input, out],
        Math.ceil(rows / this.k.rowsPerGroup)
      );
    };
    const rms = (name, input) => run("rms", [h, 1, this.bundle.tensor(name).offset / 4, 0], [input, this.weights, b.norm], 1);
    const mask = literal([1]), choice = literal([0, 0, 0]);
    ops.push({ clear: b.matrix }, { clear: b.history });
    linear("fusion.weight", taps, b.fused);
    for (let step = 0; step < k; step++) {
      for (const slice of this.slices)
        run(
          "embedIndexed",
          [h, slice.startRow, slice.endRow, slice.wordOffset ?? 0, step, 0, 0, 0],
          [slice.buffer, b.slots, step ? b.feature : b.fused, b.mixed],
          Math.ceil(h / 64)
        );
      rms("input_norm.weight", b.mixed);
      const f3 = this.fused3();
      if (f3) run(f3.key, f3.params, [this.weights, b.norm, b.raw, b.z, b.ab], f3.groups);
      else {
        linear("qkv.weight", b.norm, b.raw);
        linear("z.weight", b.norm, b.z);
        linear("ab.weight", b.norm, b.ab);
      }
      const conv = [mask, b.raw, this.convWeights, b.history, b.activated, choice];
      const io = [b.activated, b.ab, this.aux, b.packed, b.y, b.z, b.gated];
      const dn = [mask, b.packed, b.matrix, b.y, choice];
      run("conv", [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
      run("pack", [heads, heads, dim, 1], io, heads);
      run("recur", [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
      run("gate", [heads, heads, dim, 1], io, heads);
      run("commitState", [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
      run("commitConv", [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
      copy(b.mixed, b.feature);
      linear("out.weight", b.gated, b.feature, 1);
      rms("ff_norm.weight", b.feature);
      linear("gate.weight", b.norm, b.mlp, 0, "up.weight");
      linear("down.weight", b.mlp, b.feature, 1);
      rms("final_norm.weight", b.feature);
      for (const slice of this.slices) {
        if (slice.ids)
          run(
            this.slices.length === 1 ? this.k.scores : this.k.sliceScores,
            [h, 1, slice.count, slice.wordOffset ?? 0],
            [slice.buffer, slice.ids, b.norm, b.logits],
            Math.ceil(slice.count / this.k.rowsPerGroup)
          );
      }
      run("select", [subset, step, CHAIN_RECORDS, 0], [b.logits, b.subsetGlobal, b.slots], 1);
    }
    return ops;
  }
  async runChain(anchor, k) {
    if (this.retired) throw new Error("Draft runtime has been released");
    this.active++;
    try {
      const st = this.chain ??= this.allocChain();
      const { queue } = this.device;
      let taps = st.buf.tapsHost;
      if (anchor.taps instanceof Float32Array)
        queue.writeBuffer(taps, 0, anchor.taps.buffer, anchor.taps.byteOffset, anchor.taps.byteLength);
      else taps = anchor.taps;
      let id = st.tapIds.get(taps);
      if (id === void 0) st.tapIds.set(taps, id = st.tapIds.size);
      const key = k + "|" + id;
      let plan = st.plans.get(key);
      if (!plan) st.plans.set(key, plan = this.buildChain(st, taps, k));
      queue.writeBuffer(st.buf.slots, 0, Uint32Array.of(anchor.rootToken));
      const encoder = this.device.createCommandEncoder();
      for (const op of plan) {
        if ("clear" in op) encoder.clearBuffer(op.clear);
        else if ("copy" in op) encoder.copyBufferToBuffer(op.copy[0], 0, op.copy[1], 0, op.copy[1].size);
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
      const out = { tokens: [], parents: [], probs: [], conditionals: [] };
      let path = 1;
      for (let s = 0; s < k; s++) {
        const token = words[2 * s];
        const cond = probBits[2 * s + 1];
        if (token >= this.bundle.config.vocab || !(cond > 0 && cond <= 1.0000001)) throw new Error("Invalid draft chain record");
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
  async propose(anchor, budget, branchWidth = 2, singleSubmit = true) {
    if (this.retired) throw new Error("Draft runtime has been released");
    const { maxNodes } = budget, { config } = this.bundle;
    if (!Number.isInteger(maxNodes) || maxNodes < 0 || maxNodes > 15 || !Number.isInteger(branchWidth) || branchWidth < 1 || branchWidth > 64)
      throw new Error("Invalid draft tree budget");
    if (!Number.isInteger(anchor.rootToken) || anchor.rootToken < 0 || anchor.rootToken >= config.vocab || (anchor.taps instanceof Float32Array ? anchor.taps.byteLength !== 12 * config.hidden : anchor.taps.size < 12 * config.hidden)) {
      throw new Error("Invalid draft anchor");
    }
    const tree = { tokens: [], parents: [] };
    if (!maxNodes) return tree;
    const width = Math.min(budget.tree ? branchWidth : 1, config.subset, maxNodes);
    if (width === 1 && singleSubmit) return this.proposeChain(anchor, maxNodes);
    const local = new Buffers(this.device);
    this.active++;
    try {
      const { hidden: h, heads, dim, intermediate: ff, subset } = config;
      const a = heads * dim, channels = 3 * a, storage = GPUBufferUsage.STORAGE;
      const rw = storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const scratch = (label, n) => local.make(label, 4 * n, rw);
      const constants = /* @__PURE__ */ new Map();
      const literal = (data, uniform = false) => {
        const key = (uniform ? "u" : "s") + data.join(",");
        let b = constants.get(key);
        if (!b) {
          const words = Uint32Array.from(data);
          b = local.make("draft " + key, words.byteLength, uniform ? GPUBufferUsage.UNIFORM : storage, words);
          constants.set(key, b);
        }
        return b;
      };
      const taps = anchor.taps instanceof Float32Array ? local.make("target taps", anchor.taps.byteLength, storage, anchor.taps) : anchor.taps;
      const fused = scratch("fused taps", h), mixed = scratch("mixed", h), norm = scratch("normalized", h);
      const raw = scratch("raw qkv", channels), activated = scratch("conv qkv", channels), z = scratch("z", a);
      const ab = scratch("ab", 2 * heads), packed = scratch("rec inputs", heads * (3 * dim + 2));
      const y = scratch("raw readout", a), gated = scratch("gated readout", a), mlp = scratch("swiglu", ff);
      const logits = scratch("draft logits", subset), ranked = scratch("top-k records", 2 * width);
      const staging = local.make(
        "draft readback",
        8 * width,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      );
      const mask = literal([1]), choice = literal([0, 0, 0]);
      const queue = [{ token: anchor.rootToken, node: -1 }];
      for (let i = 0; i < queue.length && tree.tokens.length < maxNodes; i++) {
        const { token, node, prior } = queue[i];
        const encoder = this.device.createCommandEncoder();
        const copy = (src, dst) => encoder.copyBufferToBuffer(src, 0, dst, 0, dst.size);
        const run = (key, params, buffers, x, groupsY = 1) => {
          const pipe = this.kernels.get(key);
          const group = this.device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [literal(params, true), ...buffers].map((buffer, binding) => ({
              binding,
              resource: { buffer }
            }))
          });
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipe);
          pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(x, groupsY, 1);
          pass.end();
        };
        const linear = (name, input, out, residual = 0, up) => {
          const e = this.bundle.tensor(name);
          const [rows, cols] = e.shape;
          run(
            up ? this.k.swiglu : this.k.q4,
            [rows, cols, e.offset / 4, 0, 0, residual, 1, up ? this.bundle.tensor(up).offset / 4 : 0],
            [this.weights, input, out],
            Math.ceil(rows / this.k.rowsPerGroup)
          );
        };
        const rms = (name, input) => run("rms", [h, 1, this.bundle.tensor(name).offset / 4, 0], [input, this.weights, norm], 1);
        const next = {
          feature: scratch("branch feature", h),
          matrix: scratch("branch matrix", heads * dim * dim),
          history: scratch("branch history", channels * 3)
        };
        if (prior) {
          copy(prior.matrix, next.matrix);
          copy(prior.history, next.history);
        } else {
          linear("fusion.weight", taps, fused);
        }
        const embedding = this.slices.find((s) => token >= s.startRow && token < s.endRow);
        run(
          "embed",
          [h, 1, subset, embedding.wordOffset ?? 0],
          [embedding.buffer, literal([token - embedding.startRow]), prior?.feature ?? fused, mixed],
          Math.ceil(h / 64)
        );
        rms("input_norm.weight", mixed);
        const f3 = this.fused3();
        if (f3) run(f3.key, f3.params, [this.weights, norm, raw, z, ab], f3.groups);
        else {
          linear("qkv.weight", norm, raw);
          linear("z.weight", norm, z);
          linear("ab.weight", norm, ab);
        }
        const conv = [mask, raw, this.convWeights, next.history, activated, choice];
        const io = [activated, ab, this.aux, packed, y, z, gated];
        const dn = [mask, packed, next.matrix, y, choice];
        run("conv", [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
        run("pack", [heads, heads, dim, 1], io, heads);
        run("recur", [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
        run("gate", [heads, heads, dim, 1], io, heads);
        run("commitState", [1, heads, dim, dim], dn, Math.ceil(dim / 32), heads);
        run("commitConv", [channels, 1, 0, 0], conv, Math.ceil(channels / 64));
        copy(mixed, next.feature);
        linear("out.weight", gated, next.feature, 1);
        rms("ff_norm.weight", next.feature);
        linear("gate.weight", norm, mlp, 0, "up.weight");
        linear("down.weight", mlp, next.feature, 1);
        rms("final_norm.weight", next.feature);
        for (const slice of this.slices) {
          if (slice.ids) run(
            this.slices.length === 1 ? this.k.scores : this.k.sliceScores,
            [h, 1, slice.count, slice.wordOffset ?? 0],
            [slice.buffer, slice.ids, norm, logits],
            Math.ceil(slice.count / this.k.rowsPerGroup)
          );
        }
        run("topK", [subset, subset, width, 0], [logits, ranked], 1);
        copy(ranked, staging);
        this.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        try {
          const records = new Uint32Array(staging.getMappedRange());
          for (let r = 0; r < width && tree.tokens.length < maxNodes; r++) {
            const idx = records[2 * r];
            if (idx >= subset) throw new Error("Invalid draft top-k record");
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
};

// src/bench/engineBHeadBench.ts
function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
var clock = () => globalThis.performance?.now() ?? Date.now();
async function benchHeadLatency(runtime, anchor, o = {}) {
  const ks = o.ks ?? [1, 2, 3, 4, 5];
  const now = o.now ?? clock;
  const reps = o.reps ?? 40, warmup = o.warmup ?? 3, budget = o.budgetMs ?? 1e4;
  for (let w = 0; w < warmup; w++) for (const k of ks) await runtime.proposeChain(anchor, k);
  const samples = new Map(ks.map((k) => [k, []]));
  const start = now();
  for (let r = 0; r < reps; r++) {
    for (const k of ks) {
      const t0 = now();
      const out = await runtime.proposeChain(anchor, k);
      const dt = now() - t0;
      if (out.tokens.length !== k) throw new Error(`Expected ${k} draft tokens, got ${out.tokens.length}`);
      samples.get(k).push(dt);
    }
    if (now() - start > budget && r >= 4) break;
  }
  return ks.map((k) => {
    const s = [...samples.get(k)].sort((a, b) => a - b);
    return { k, medianMs: quantile(s, 0.5), p10Ms: quantile(s, 0.1), p90Ms: quantile(s, 0.9), samples: s.length };
  });
}
async function benchRejectSurcharge(probe, o = {}) {
  const M = o.M ?? 5, reps = o.reps ?? 15;
  for (let w = 0; w < (o.warmup ?? 2); w++) {
    await probe.measure(M, false);
    await probe.measure(M, true);
  }
  const accept = [], reject = [];
  for (let r = 0; r < reps; r++) {
    const first = r % 2 === 0;
    const a = first ? await probe.measure(M, false) : NaN;
    const b = await probe.measure(M, true);
    const c = first ? a : await probe.measure(M, false);
    accept.push(c);
    reject.push(b);
  }
  const sa = [...accept].sort((x, y) => x - y), sr = [...reject].sort((x, y) => x - y);
  const acceptMs = quantile(sa, 0.5), rejectMs = quantile(sr, 0.5);
  return { M, acceptMs, rejectMs, rejectPenaltyMs: Math.max(0, rejectMs - acceptMs), samples: reps };
}
async function runEngineBBench(runtime, anchor, probe, o = {}) {
  const now = o.now ?? clock;
  const t0 = now();
  const head = await benchHeadLatency(runtime, anchor, o);
  const reject = probe ? await benchRejectSurcharge(probe) : void 0;
  const k5 = head.find((h) => h.k === 5) ?? head[head.length - 1];
  return { format: "engine-b-head-bench", version: 1, device: o.device, head, headMs: k5.medianMs, reject, elapsedMs: now() - t0 };
}
function toAdapterOptions(report, stepMs, fallbackRejectPenaltyMs) {
  if (!stepMs.length || stepMs.some((v) => !(v > 0))) throw new Error("stepMs must be positive wall times for M = 1..");
  return {
    profile: { stepMs: [...stepMs], rejectPenaltyMs: report.reject?.rejectPenaltyMs ?? fallbackRejectPenaltyMs },
    cutoff: { headMs: report.headMs }
  };
}

// src/bench/engineBHeadBenchPage.ts
var GOLDEN_ROOT_TOKEN = 271;
async function runInPage(o) {
  const base = new URL(String(o.assetsBase), globalThis.location?.href);
  let device = o.device;
  let label;
  if (!device) {
    const gpu = globalThis.navigator?.gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPU adapter unavailable");
    label = adapter.info?.description || adapter.info?.vendor;
    const requiredFeatures = adapter.features.has("subgroups") ? ["subgroups"] : [];
    device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize }
    });
  }
  const bundle = await loadGdnBundle(new URL("assets/exp007/manifest.json", base));
  if (!bundle) throw new Error("No EXP-007 draft head under assets/exp007/");
  let owned;
  let table = o.table;
  if (!table) {
    if (!o.tableUrl) throw new Error("Pass table (borrowed slabs) or tableUrl (RC1 slab2)");
    const response = await fetch(o.tableUrl);
    if (!response.ok) throw new Error("Unable to load the tied table (HTTP " + response.status + ")");
    const bytes = await response.arrayBuffer();
    owned = device.createBuffer({ size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(owned, 0, bytes);
    table = { buffer: owned };
  }
  const tapsResponse = await fetch(new URL("fixtures/trained-exp007/taps.f32", base));
  if (!tapsResponse.ok) throw new Error("Unable to load the golden taps");
  const taps = new Float32Array(await tapsResponse.arrayBuffer());
  const runtime = await GdnDraftRuntime.create(device, bundle, table, shaderAssets(new URL("wgsl/", base)), { mode: o.mode });
  try {
    const report = await runEngineBBench(runtime, { taps, rootToken: GOLDEN_ROOT_TOKEN }, o.probe, { budgetMs: o.budgetMs ?? 1e4, device: label });
    return { ...report, headSha256: bundle.sha256, tier: runtime.tier, chainStats: { ...runtime.chainStats } };
  } finally {
    runtime.dispose();
    owned?.destroy();
  }
}
export {
  GOLDEN_ROOT_TOKEN,
  benchHeadLatency,
  benchRejectSurcharge,
  quantile,
  runEngineBBench,
  runInPage,
  toAdapterOptions
};
