enable subgroups;
// Tier 1 variant (subgroupAdd; subgroup size 4..64 so no subgroup spans two rows).
// Fast M = 1 Q4 GEMV for the EXP-007 draft head (Tier 1: subgroups, f32 accumulation, native FMA). Same bindings and parameters as q4_gemv_smallm_v2 /
// q4_gate_up_swiglu_smallm_v2 (tokens must be 1; cols a multiple of 32).
//
// Word-parallel: 64 lanes share one row and step through its packed nibble
// words, so consecutive lanes load consecutive words (coalesced). A word holds 8
// columns; with w = q * scale + low, its contribution is
//   scale * dot(q, x) + low * sum(x).
// Four rows per 256-lane workgroup. Not bit-identical to the reference kernels
// (different summation order and fused multiply-adds); the reference shaders
// stay the bit-exact parity path.
struct Params {
  rows: u32, cols: u32, weight_offset: u32, input_offset: u32,
  output_offset: u32, residual: u32, tokens: u32, arg: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> weights: array<u32>;
@group(0) @binding(2) var<storage,read> input: array<f32>;
@group(0) @binding(3) var<storage,read_write> output: array<f32>;
// gemv3 only: second and third outputs.
@group(0) @binding(4) var<storage,read_write> output_b: array<f32>;
@group(0) @binding(5) var<storage,read_write> output_c: array<f32>;

const LANES = 64u;
const ROWS = 4u;

fn nibbles(q: u32, shift: u32) -> vec4<f32> {
  return vec4<f32>((vec4<u32>(q) >> vec4<u32>(shift, shift + 4u, shift + 8u, shift + 12u)) & vec4<u32>(15u));
}

// Partial dot of one lane over its words of the row starting at word `base`.
fn row_dot(base: u32, lane: u32) -> f32 {
  let words = (p.cols / 32u) * 4u;
  var acc = 0.0;
  for (var i = lane; i < words; i += LANES) {
    let block = base + (i >> 2u) * 5u;
    let affine = unpack2x16float(weights[block + 4u]);
    let q = weights[block + (i & 3u)];
    let c = p.input_offset + (i >> 2u) * 32u + (i & 3u) * 8u;
    let x0 = vec4<f32>(input[c], input[c + 1u], input[c + 2u], input[c + 3u]);
    let x1 = vec4<f32>(input[c + 4u], input[c + 5u], input[c + 6u], input[c + 7u]);
    let qx = dot(nibbles(q, 0u), x0) + dot(nibbles(q, 16u), x1);
    let sx = dot(x0 + x1, vec4<f32>(1.0));
    acc = fma(affine.x, qx, fma(affine.y, sx, acc));
  }
  return acc;
}

// Sum the 64 lanes of each row segment; returns the total in lane 0 of the row.
// Row totals: subgroupAdd inside each subgroup, then lane 0 of the row adds the
// subgroup totals that cover its 64 lanes.
var<workgroup> sg_total: array<f32, 256>;
var<workgroup> sg_total_up: array<f32, 256>;
fn row_sum(local: u32, size: u32) -> f32 {
  let first = (local / LANES) * LANES / size;
  var total = 0.0;
  for (var k = 0u; k < LANES / size; k++) { total += sg_total[first + k]; }
  return total;
}
fn row_sum_up(local: u32, size: u32) -> f32 {
  let first = (local / LANES) * LANES / size;
  var total = 0.0;
  for (var k = 0u; k < LANES / size; k++) { total += sg_total_up[first + k]; }
  return total;
}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  let lane = local % LANES;
  let row = wg.x * ROWS + local / LANES;
  var acc = 0.0;
  if (row < p.rows) { acc = row_dot(p.weight_offset + row * (p.cols / 32u) * 5u, lane); }
  let s = subgroupAdd(acc);
  if (sg_lane == 0u) { sg_total[local / sg_size] = s; }
  workgroupBarrier();
  if (lane == 0u && row < p.rows) {
    let dst = p.output_offset + row;
    var y = row_sum(local, sg_size);
    if (p.residual != 0u) { y += output[dst]; }
    output[dst] = y;
  }
}

@compute @workgroup_size(256)
fn swiglu(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32,
          @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  let lane = local % LANES;
  let row = wg.x * ROWS + local / LANES;
  var g = 0.0;
  var u = 0.0;
  if (row < p.rows) {
    let relative = row * (p.cols / 32u) * 5u;
    g = row_dot(p.weight_offset + relative, lane);
    u = row_dot(p.arg + relative, lane);
  }
  let sg = subgroupAdd(g);
  let su = subgroupAdd(u);
  if (sg_lane == 0u) {
    sg_total[local / sg_size] = sg;
    sg_total_up[local / sg_size] = su;
  }
  workgroupBarrier();
  if (lane == 0u && row < p.rows) {
    let dst = p.output_offset + row;
    let gate = row_sum(local, sg_size);
    var y = gate / (1.0 + exp(-gate)) * row_sum_up(local, sg_size);
    if (p.residual != 0u) { y += output[dst]; }
    output[dst] = y;
  }
}

// Three projections of one input in one dispatch (the draft head's QKV, Z and AB).
// Params reuse: rows = rows_a, weight_offset = offset_a, output_offset = rows_b,
// residual = rows_c, tokens = offset_b, arg = offset_c; no residual, outputs at 0.
@compute @workgroup_size(256)
fn gemv3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32,
         @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  let lane = local % LANES;
  let total = p.rows + p.output_offset + p.residual;
  let r = wg.x * ROWS + local / LANES;
  let row_words = (p.cols / 32u) * 5u;
  var acc = 0.0;
  if (r < p.rows) {
    acc = row_dot(p.weight_offset + r * row_words, lane);
  } else if (r < p.rows + p.output_offset) {
    acc = row_dot(p.tokens + (r - p.rows) * row_words, lane);
  } else if (r < total) {
    acc = row_dot(p.arg + (r - p.rows - p.output_offset) * row_words, lane);
  }
  let s = subgroupAdd(acc);
  if (sg_lane == 0u) { sg_total[local / sg_size] = s; }
  workgroupBarrier();
  if (lane != 0u || r >= total) { return; }
  let y = row_sum(local, sg_size);
  if (r < p.rows) {
    output[r] = y;
  } else if (r < p.rows + p.output_offset) {
    output_b[r - p.rows] = y;
  } else {
    output_c[r - p.rows - p.output_offset] = y;
  }
}
