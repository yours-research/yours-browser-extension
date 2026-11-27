enable subgroups;
// Tier 1 variant (subgroupAdd; subgroup size 4..64 so no subgroup spans two 64-lane teams).
// Coalesced word-parallel + 2 KiB workgroup-staged x_sh (16 blocks = 512 cols = 1 word/lane/tile),
// 8 rows per 256-lane workgroup (2 rows per 64-lane team), zero inner loop, single-barrier subgroupAdd.
struct Params {
  rows: u32, cols: u32, weight_offset: u32, input_offset: u32,
  output_offset: u32, residual: u32, tokens: u32, arg: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> weights: array<u32>;
@group(0) @binding(2) var<storage,read> input: array<f32>;
@group(0) @binding(3) var<storage,read_write> output: array<f32>;
@group(0) @binding(4) var<storage,read_write> output_b: array<f32>;
@group(0) @binding(5) var<storage,read_write> output_c: array<f32>;

const LANES = 64u;
const ROWS_PER_WG = 8u;
const TILE_BLOCKS = 16u;

var<workgroup> x_sh: array<vec4<f32>, 128>; // 2 KiB
var<workgroup> sg_a: array<f32, 64>;
var<workgroup> sg_b: array<f32, 64>;
var<workgroup> sg_c: array<f32, 64>;
var<workgroup> sg_d: array<f32, 64>;

fn nibbles(q: u32, shift: u32) -> vec4<f32> {
  return vec4<f32>((vec4<u32>(q) >> vec4<u32>(shift, shift + 4u, shift + 8u, shift + 12u)) & vec4<u32>(15u));
}

fn q4_word_fma(base: u32, word_in_block: u32, x0: vec4<f32>, x1: vec4<f32>, sx: f32, acc: f32) -> f32 {
  let affine = unpack2x16float(weights[base + 4u]);
  let q = weights[base + word_in_block];
  let qx = dot(nibbles(q, 0u), x0) + dot(nibbles(q, 16u), x1);
  return fma(affine.x, qx, fma(affine.y, sx, acc));
}

fn stage_tile(tid: u32, x_base: u32, tile_blocks: u32) {
  let load_vec4s = tile_blocks << 3u;
  if (tid < load_vec4s) {
    let c = x_base + (tid << 2u);
    x_sh[tid] = vec4<f32>(input[c], input[c + 1u], input[c + 2u], input[c + 3u]);
  }
  workgroupBarrier();
}

fn team_sum_pair(team: u32, sg_size: u32) -> vec2<f32> {
  let per_team = LANES / sg_size;
  let first = team * per_team;
  var ta = 0.0;
  var tb = 0.0;
  for (var k = 0u; k < per_team; k++) {
    ta += sg_a[first + k];
    tb += sg_b[first + k];
  }
  return vec2<f32>(ta, tb);
}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  if (p.rows == 0u) { return; }
  let team = tid >> 6u;
  let lane = tid & 63u;
  let r0 = wg.x * ROWS_PER_WG + team * 2u;
  let r1 = r0 + 1u;
  let v0 = r0 < p.rows;
  let v1 = r1 < p.rows;
  let blocks = p.cols >> 5u;
  let row_stride = blocks * 5u;
  let o0 = p.weight_offset + min(r0, p.rows - 1u) * row_stride;
  let o1 = p.weight_offset + min(r1, p.rows - 1u) * row_stride;
  let lane_blk = lane >> 2u;
  let wib = lane & 3u;
  var acc0 = 0.0;
  var acc1 = 0.0;
  for (var k_tile = 0u; k_tile < blocks; k_tile += TILE_BLOCKS) {
    let tb = min(TILE_BLOCKS, blocks - k_tile);
    stage_tile(tid, p.input_offset + (k_tile << 5u), tb);
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
  let s0 = subgroupAdd(acc0);
  let s1 = subgroupAdd(acc1);
  if (sg_lane == 0u) {
    let idx = tid / sg_size;
    sg_a[idx] = s0;
    sg_b[idx] = s1;
  }
  workgroupBarrier();
  if (lane == 0u) {
    let res = team_sum_pair(team, sg_size);
    if (v0) {
      let d0 = p.output_offset + r0;
      output[d0] = select(res.x, output[d0] + res.x, p.residual != 0u);
    }
    if (v1) {
      let d1 = p.output_offset + r1;
      output[d1] = select(res.y, output[d1] + res.y, p.residual != 0u);
    }
  }
}

@compute @workgroup_size(256)
fn gemv_add(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32,
            @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  if (p.rows == 0u) { return; }
  let team = tid >> 6u;
  let lane = tid & 63u;
  let r0 = wg.x * ROWS_PER_WG + team * 2u;
  let r1 = r0 + 1u;
  let v0 = r0 < p.rows;
  let v1 = r1 < p.rows;
  let blocks = p.cols >> 5u;
  let row_stride = blocks * 5u;
  let o0 = p.weight_offset + min(r0, p.rows - 1u) * row_stride;
  let o1 = p.weight_offset + min(r1, p.rows - 1u) * row_stride;
  let lane_blk = lane >> 2u;
  let wib = lane & 3u;
  var acc0 = 0.0;
  var acc1 = 0.0;
  for (var k_tile = 0u; k_tile < blocks; k_tile += TILE_BLOCKS) {
    let tb = min(TILE_BLOCKS, blocks - k_tile);
    stage_tile(tid, p.input_offset + (k_tile << 5u), tb);
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
  let s0 = subgroupAdd(acc0);
  let s1 = subgroupAdd(acc1);
  if (sg_lane == 0u) {
    let idx = tid / sg_size;
    sg_a[idx] = s0;
    sg_b[idx] = s1;
  }
  workgroupBarrier();
  if (lane == 0u) {
    let res = team_sum_pair(team, sg_size);
    if (v0) {
      let d0 = p.output_offset + r0;
      output[d0] = output_b[d0] + res.x;
    }
    if (v1) {
      let d1 = p.output_offset + r1;
      output[d1] = output_b[d1] + res.y;
    }
  }
}

@compute @workgroup_size(256)
fn swiglu(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32,
          @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  if (p.rows == 0u) { return; }
  let team = tid >> 6u;
  let lane = tid & 63u;
  let r0 = wg.x * ROWS_PER_WG + team * 2u;
  let r1 = r0 + 1u;
  let v0 = r0 < p.rows;
  let v1 = r1 < p.rows;
  let blocks = p.cols >> 5u;
  let row_stride = blocks * 5u;
  let rel0 = min(r0, p.rows - 1u) * row_stride;
  let rel1 = min(r1, p.rows - 1u) * row_stride;
  let lane_blk = lane >> 2u;
  let wib = lane & 3u;
  var g0 = 0.0; var u0 = 0.0;
  var g1 = 0.0; var u1 = 0.0;
  for (var k_tile = 0u; k_tile < blocks; k_tile += TILE_BLOCKS) {
    let tb = min(TILE_BLOCKS, blocks - k_tile);
    stage_tile(tid, p.input_offset + (k_tile << 5u), tb);
    if (lane < (tb << 2u)) {
      let x0 = x_sh[lane << 1u];
      let x1 = x_sh[(lane << 1u) + 1u];
      let sx = dot(x0 + x1, vec4<f32>(1.0));
      let b5 = (k_tile + lane_blk) * 5u;
      if (v0) {
        g0 = q4_word_fma(p.weight_offset + rel0 + b5, wib, x0, x1, sx, g0);
        u0 = q4_word_fma(p.arg + rel0 + b5, wib, x0, x1, sx, u0);
      }
      if (v1) {
        g1 = q4_word_fma(p.weight_offset + rel1 + b5, wib, x0, x1, sx, g1);
        u1 = q4_word_fma(p.arg + rel1 + b5, wib, x0, x1, sx, u1);
      }
    }
    if (k_tile + TILE_BLOCKS < blocks) { workgroupBarrier(); }
  }
  let sg0 = subgroupAdd(g0);
  let su0 = subgroupAdd(u0);
  let sg1 = subgroupAdd(g1);
  let su1 = subgroupAdd(u1);
  if (sg_lane == 0u) {
    let idx = tid / sg_size;
    sg_a[idx] = sg0; sg_b[idx] = su0;
    sg_c[idx] = sg1; sg_d[idx] = su1;
  }
  workgroupBarrier();
  if (lane == 0u) {
    let per_team = LANES / sg_size;
    let first = team * per_team;
    var tg0 = 0.0; var tu0 = 0.0;
    var tg1 = 0.0; var tu1 = 0.0;
    for (var k = 0u; k < per_team; k++) {
      tg0 += sg_a[first + k]; tu0 += sg_b[first + k];
      tg1 += sg_c[first + k]; tu1 += sg_d[first + k];
    }
    if (v0) {
      let d0 = p.output_offset + r0;
      let y0 = (tg0 / (1.0 + exp(-tg0))) * tu0;
      output[d0] = select(y0, output[d0] + y0, p.residual != 0u);
    }
    if (v1) {
      let d1 = p.output_offset + r1;
      let y1 = (tg1 / (1.0 + exp(-tg1))) * tu1;
      output[d1] = select(y1, output[d1] + y1, p.residual != 0u);
    }
  }
}

fn gemv3_row_base(r: u32, row_stride: u32) -> u32 {
  if (r < p.rows) { return p.weight_offset + r * row_stride; }
  if (r < p.rows + p.output_offset) { return p.tokens + (r - p.rows) * row_stride; }
  return p.arg + (r - p.rows - p.output_offset) * row_stride;
}

fn gemv3_store(r: u32, y: f32) {
  if (r < p.rows) { output[r] = y; }
  else if (r < p.rows + p.output_offset) { output_b[r - p.rows] = y; }
  else { output_c[r - p.rows - p.output_offset] = y; }
}

@compute @workgroup_size(256)
fn gemv3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32,
         @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  let total = p.rows + p.output_offset + p.residual;
  if (total == 0u) { return; }
  let team = tid >> 6u;
  let lane = tid & 63u;
  let r0 = wg.x * ROWS_PER_WG + team * 2u;
  let r1 = r0 + 1u;
  let v0 = r0 < total;
  let v1 = r1 < total;
  let blocks = p.cols >> 5u;
  let row_stride = blocks * 5u;
  let o0 = select(0u, gemv3_row_base(r0, row_stride), v0);
  let o1 = select(0u, gemv3_row_base(r1, row_stride), v1);
  let lane_blk = lane >> 2u;
  let wib = lane & 3u;
  var acc0 = 0.0;
  var acc1 = 0.0;
  for (var k_tile = 0u; k_tile < blocks; k_tile += TILE_BLOCKS) {
    let tb = min(TILE_BLOCKS, blocks - k_tile);
    stage_tile(tid, k_tile << 5u, tb);
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
  let s0 = subgroupAdd(acc0);
  let s1 = subgroupAdd(acc1);
  if (sg_lane == 0u) {
    let idx = tid / sg_size;
    sg_a[idx] = s0;
    sg_b[idx] = s1;
  }
  workgroupBarrier();
  if (lane == 0u) {
    let res = team_sum_pair(team, sg_size);
    if (v0) { gemv3_store(r0, res.x); }
    if (v1) { gemv3_store(r1, res.y); }
  }
}
