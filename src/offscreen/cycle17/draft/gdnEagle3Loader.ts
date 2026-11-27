/** Wire format v1 from python/train_gdn_eagle3.py. All offsets below are bytes. */
export interface GdnConfig {
  hidden: number;
  heads: number;
  dim: number;
  intermediate: number;
  vocab: number;
  subset: number;
  taps: number;
}
export const EXP007_CONFIG: Readonly<GdnConfig> = Object.freeze({
  hidden: 2560,
  heads: 8,
  dim: 128,
  intermediate: 3072,
  vocab: 81920,
  subset: 16384,
  taps: 4,
});
export const EXP007_BYTES = 35_397_184;

interface TensorEntry {
  kind: 'u32' | 'f32' | 'q4';
  shape: readonly number[];
  offset: number;
  byte_length: number;
}

export interface GdnBundle {
  config: Readonly<GdnConfig>;
  sha256: string;
  /** Little-endian, validated payload for upload as one storage buffer. */
  binary: ArrayBuffer;
  subsetIds: Uint32Array;
  tensor(name: string): TensorEntry;
  words(name: string): Uint32Array;
  floats(name: string): Float32Array;
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid draft manifest object');
  return value as Record<string, unknown>;
}

function sameShape(actual: unknown, expected: readonly number[]) {
  return (
    Array.isArray(actual) && actual.length === expected.length && expected.every((v, i) => actual[i] === v)
  );
}

function validateManifest(raw: unknown, expected: Readonly<GdnConfig>) {
  const manifest = record(raw);
  if (
    manifest.format !== 'gdn-eagle3-draft' ||
    manifest.format_version !== 1 ||
    manifest.byte_order !== 'little' ||
    manifest.file !== 'draft_head.bin'
  )
    throw new Error('Unsupported draft bundle format');
  const cfg = record(manifest.config);
  for (const [k, v] of Object.entries(expected)) {
    if (cfg[k] !== v || !Number.isSafeInteger(v) || v < 1)
      throw new Error('Incompatible draft dimension: ' + k);
  }
  const { hidden: h, heads, dim, intermediate: ff, subset, vocab, taps } = expected;
  if (
    taps !== 4 ||
    dim > 128 ||
    subset > vocab ||
    !sameShape(manifest.layer_taps, [3, 15, 31]) ||
    manifest.subset_size !== subset
  )
    throw new Error('Incompatible draft feature or subset layout');
  const a = heads * dim;
  const shapes: Record<string, readonly number[]> = {
    log_rate: [heads],
    dt_bias: [heads],
    conv: [3 * a, taps],
    'fusion.weight': [h, 3 * h],
    'input_norm.weight': [h],
    'qkv.weight': [3 * a, h],
    'z.weight': [a, h],
    'ab.weight': [2 * heads, h],
    'head_norm.weight': [dim],
    'out.weight': [h, a],
    'ff_norm.weight': [h],
    'gate.weight': [ff, h],
    'up.weight': [ff, h],
    'down.weight': [h, ff],
    'final_norm.weight': [h],
    subset_ids: [subset],
  };
  const entries = new Map<string, TensorEntry>();
  let offset = 0;
  if (!Array.isArray(manifest.tensors)) throw new Error('Missing draft tensor directory');
  for (const item of manifest.tensors) {
    const entry = record(item),
      name = entry.name;
    if (typeof name !== 'string' || !Object.hasOwn(shapes, name) || entries.has(name))
      throw new Error('Unexpected or duplicate draft tensor');
    const shape = shapes[name];
    const kind = name === 'subset_ids' ? 'u32' : shape.length === 2 && name !== 'conv' ? 'q4' : 'f32';
    const blocks = Math.ceil(shape[1] / 32);
    const length = kind === 'q4' ? shape[0] * blocks * 20 : 4 * shape.reduce((v, d) => v * d, 1);
    if (
      !sameShape(entry.shape, shape) ||
      entry.kind !== kind ||
      entry.offset !== offset ||
      entry.byte_length !== length ||
      (kind === 'q4' && (entry.blocks_per_row !== blocks || entry.padded_cols !== blocks * 32))
    ) {
      throw new Error('Invalid draft tensor layout: ' + name);
    }
    entries.set(name, { shape, kind, offset, byte_length: length });
    offset += length;
  }
  if (entries.size !== Object.keys(shapes).length || offset !== manifest.total_bytes || offset > 40 * 2 ** 20) {
    throw new Error('Incomplete draft bundle or incorrect byte count');
  }
  if (typeof manifest.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(manifest.sha256))
    throw new Error('Invalid draft SHA-256');
  return { entries, config: Object.freeze({ ...expected }), size: offset, sha256: manifest.sha256 };
}

async function validatePayload(
  layout: ReturnType<typeof validateManifest>,
  binary: ArrayBuffer,
): Promise<GdnBundle> {
  if (binary.byteLength !== layout.size) throw new Error('Incorrect draft payload length');
  const hash = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', binary));
  const hex = Array.from(hash, (b) => b.toString(16).padStart(2, '0')).join('');
  if (hex !== layout.sha256) throw new Error('Draft SHA-256 mismatch');
  function tensor(name: string) {
    const entry = layout.entries.get(name);
    if (!entry) throw new Error('Unknown draft tensor: ' + name);
    return entry;
  }
  function words(name: string) {
    const e = tensor(name);
    return new Uint32Array(binary, e.offset, e.byte_length / 4);
  }
  function floats(name: string) {
    const e = tensor(name);
    if (e.kind !== 'f32') throw new Error('Expected f32 draft tensor: ' + name);
    return new Float32Array(binary, e.offset, e.byte_length / 4);
  }
  const subsetIds = words('subset_ids');
  const seen = new Set<number>();
  for (const id of subsetIds) {
    if (id >= layout.config.vocab || seen.has(id)) throw new Error('Invalid draft vocabulary subset');
    seen.add(id);
  }
  return { config: layout.config, binary, sha256: hex, subsetIds, tensor, words, floats };
}

/** The caller transfers ownership of binary. Treat validated data and its views as read-only. */
export async function parseGdnBundle(manifest: unknown, binary: ArrayBuffer, expectedConfig = EXP007_CONFIG) {
  return validatePayload(validateManifest(manifest, expectedConfig), binary);
}

/** Use a manifest URL from the same deployment as the frozen target. A missing
 * manifest or binary is an optional head; malformed or corrupt assets fail closed. */
export async function loadGdnBundle(
  manifestUrl: URL,
  options: {
    fetch?: typeof fetch;
    signal?: AbortSignal;
    expectedConfig?: Readonly<GdnConfig>;
  } = {},
): Promise<GdnBundle | null> {
  const get = options.fetch ?? globalThis.fetch;
  async function optional(url: URL) {
    const response = await get(url, { signal: options.signal });
    if (response.status === 404 || response.status === 410) return null;
    if (!response.ok) throw new Error('Draft fetch failed with HTTP ' + response.status);
    return response;
  }
  const manifest = await optional(manifestUrl);
  if (!manifest) return null;
  const layout = validateManifest(await manifest.json(), options.expectedConfig ?? EXP007_CONFIG);
  const response = await optional(new URL('draft_head.bin', manifest.url || manifestUrl));
  if (!response) return null;
  const binary = new ArrayBuffer(layout.size);
  const dst = new Uint8Array(binary);
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Missing draft response body');
  let offset = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.byteLength > dst.length) {
        await reader.cancel();
        throw new Error('Draft payload exceeds its declared length');
      }
      dst.set(value, offset);
      offset += value.byteLength;
    }
  } finally {
    reader.releaseLock();
  }
  if (offset !== dst.length) throw new Error('Truncated draft payload');
  return validatePayload(layout, binary);
}
