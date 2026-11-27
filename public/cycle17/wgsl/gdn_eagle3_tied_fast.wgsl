// Fast subset logits for the EXP-007 draft head (universal tier: f32, workgroup
// reduction, native FMA). Same bindings and parameters as gdn_eagle3_tied's
// subset_logits / subset_logits_slice, for one anchor row (rows = 1) and a
// hidden size that is a multiple of 32.
//
// Word-parallel: 64 lanes share one tied-table row and load consecutive packed
// words (coalesced); four subset rows per 256-lane workgroup. Each word adds
// scale * dot(q, x) + low * sum(x). Not bit-identical to gdn_eagle3_tied, which
// remains the bit-exact parity path.
struct Params { hidden: u32, rows: u32, subset: u32, slab_word_offset: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> slab: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> features: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const LANES = 64u;
const ROWS = 4u;
var<workgroup> partial: array<f32, 256>;

fn nibbles(q: u32, shift: u32) -> vec4<f32> {
  return vec4<f32>((vec4<u32>(q) >> vec4<u32>(shift, shift + 4u, shift + 8u, shift + 12u)) & vec4<u32>(15u));
}

fn row_dot(table_row: u32, lane: u32) -> f32 {
  let blocks = p.hidden / 32u;
  let base = p.slab_word_offset + table_row * blocks * 5u;
  var acc = 0.0;
  for (var i = lane; i < blocks * 4u; i += LANES) {
    let block = base + (i >> 2u) * 5u;
    let affine = unpack2x16float(slab[block + 4u]);
    let q = slab[block + (i & 3u)];
    let c = (i >> 2u) * 32u + (i & 3u) * 8u;
    let x0 = vec4<f32>(features[c], features[c + 1u], features[c + 2u], features[c + 3u]);
    let x1 = vec4<f32>(features[c + 4u], features[c + 5u], features[c + 6u], features[c + 7u]);
    let qx = dot(nibbles(q, 0u), x0) + dot(nibbles(q, 16u), x1);
    acc = fma(affine.x, qx, fma(affine.y, dot(x0 + x1, vec4<f32>(1.0)), acc));
  }
  return acc;
}

fn reduce(lane: u32, slot: u32) {
  for (var span = 32u; span > 0u; span >>= 1u) {
    if (lane < span) { partial[slot] += partial[slot + span]; }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn subset_logits(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let lane = local % LANES;
  let i = wg.x * ROWS + local / LANES;
  var acc = 0.0;
  if (i < p.subset) { acc = row_dot(ids[i], lane); }
  partial[local] = acc;
  workgroupBarrier();
  reduce(lane, local);
  if (lane == 0u && i < p.subset) { output[i] = partial[local]; }
}

// Pairs are (token row within this slab, global subset index).
@compute @workgroup_size(256)
fn subset_logits_slice(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let lane = local % LANES;
  let i = wg.x * ROWS + local / LANES;
  var acc = 0.0;
  if (i < p.subset) { acc = row_dot(ids[2u * i], lane); }
  partial[local] = acc;
  workgroupBarrier();
  reduce(lane, local);
  if (lane == 0u && i < p.subset) { output[ids[2u * i + 1u]] = partial[local]; }
}
