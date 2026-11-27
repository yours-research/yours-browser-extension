// Fast subset logits for the EXP-007 draft head (Tier 2 universal, f32 accumulation, native FMA).
// Coalesced word-parallel + 2 KiB workgroup-staged features (16 blocks = 512 cols = 1 word/lane/tile),
// 8 subset rows per 256-lane workgroup (4 teams x 64 lanes x 2 rows/team), zero inner loop.
struct Params { hidden: u32, rows: u32, subset: u32, slab_word_offset: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> slab: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> features: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const ROWS_PER_WG = 8u;
const TILE_BLOCKS = 16u;

var<workgroup> x_sh: array<vec4<f32>, 128>; // 2 KiB
var<workgroup> red_a: array<f32, 256>;
var<workgroup> red_b: array<f32, 256>;

fn nibbles(q: u32, shift: u32) -> vec4<f32> {
  return vec4<f32>((vec4<u32>(q) >> vec4<u32>(shift, shift + 4u, shift + 8u, shift + 12u)) & vec4<u32>(15u));
}

fn q4_word_fma(base: u32, word_in_block: u32, x0: vec4<f32>, x1: vec4<f32>, sx: f32, acc: f32) -> f32 {
  let affine = unpack2x16float(slab[base + 4u]);
  let q = slab[base + word_in_block];
  let qx = dot(nibbles(q, 0u), x0) + dot(nibbles(q, 16u), x1);
  return fma(affine.x, qx, fma(affine.y, sx, acc));
}

fn eval_pair(tid: u32, lane: u32, v0: bool, v1: bool, o0: u32, o1: u32) -> vec2<f32> {
  let blocks = p.hidden >> 5u;
  let lane_blk = lane >> 2u;
  let wib = lane & 3u;
  var acc0 = 0.0;
  var acc1 = 0.0;
  for (var k_tile = 0u; k_tile < blocks; k_tile += TILE_BLOCKS) {
    let tb = min(TILE_BLOCKS, blocks - k_tile);
    let load_vec4s = tb << 3u;
    if (tid < load_vec4s) {
      let c = (k_tile << 5u) + (tid << 2u);
      x_sh[tid] = vec4<f32>(features[c], features[c + 1u], features[c + 2u], features[c + 3u]);
    }
    workgroupBarrier();
    if (lane < (tb << 2u)) {
      let x0 = x_sh[lane << 1u];
      let x1 = x_sh[(lane << 1u) + 1u];
      let sx = dot(x0 + x1, vec4<f32>(1.0));
      let b5 = (k_tile + lane_blk) * 5u;
      if (v0) { acc0 = q4_word_fma(o0 + b5, wib, x0, x1, sx, acc0); }
      if (v1) { acc1 = q4_word_fma(o1 + b5, wib, x0, x1, sx, acc1); }
    }
    if (k_tile + TILE_BLOCKS < blocks) { workgroupBarrier(); }
  }
  red_a[tid] = acc0;
  red_b[tid] = acc1;
  workgroupBarrier();
  for (var span = 32u; span > 0u; span >>= 1u) {
    if (lane < span) {
      red_a[tid] += red_a[tid + span];
      red_b[tid] += red_b[tid + span];
    }
    workgroupBarrier();
  }
  return vec2<f32>(red_a[tid], red_b[tid]);
}

@compute @workgroup_size(256)
fn subset_logits(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let team = tid >> 6u;
  let lane = tid & 63u;
  let i0 = wg.x * ROWS_PER_WG + team * 2u;
  let i1 = i0 + 1u;
  let v0 = i0 < p.subset;
  let v1 = i1 < p.subset;
  let row_stride = (p.hidden >> 5u) * 5u;
  let o0 = select(0u, p.slab_word_offset + ids[i0] * row_stride, v0);
  let o1 = select(0u, p.slab_word_offset + ids[i1] * row_stride, v1);
  let res = eval_pair(tid, lane, v0, v1, o0, o1);
  if (lane == 0u) {
    if (v0) { output[i0] = res.x; }
    if (v1) { output[i1] = res.y; }
  }
}

@compute @workgroup_size(256)
fn subset_logits_slice(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let team = tid >> 6u;
  let lane = tid & 63u;
  let i0 = wg.x * ROWS_PER_WG + team * 2u;
  let i1 = i0 + 1u;
  let v0 = i0 < p.subset;
  let v1 = i1 < p.subset;
  let row_stride = (p.hidden >> 5u) * 5u;
  let o0 = select(0u, p.slab_word_offset + ids[2u * i0] * row_stride, v0);
  let o1 = select(0u, p.slab_word_offset + ids[2u * i1] * row_stride, v1);
  let res = eval_pair(tid, lane, v0, v1, o0, o1);
  if (lane == 0u) {
    if (v0) { output[ids[2u * i0 + 1u]] = res.x; }
    if (v1) { output[ids[2u * i1 + 1u]] = res.y; }
  }
}
