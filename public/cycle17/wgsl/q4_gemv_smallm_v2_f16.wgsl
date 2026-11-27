// Q4_GEMV_SMALLM_V2_F16: Tier-1 (shader-f16 + subgroups) cooperative small-M
// Q4 projection, M = 1..16. Same bindings and uniform words as
// q4_gemv_smallm_v2.wgsl; dispatch (ceil(rows / 16), 1, 1).
//
// Work split. A workgroup of 256 lanes is 8 teams of 32 lanes; a team owns 2
// rows. Lane l of a team owns column l of every 32-column block, dequantizes
// that weight ONCE per step and reuses it for all M tokens (4 tokens per vec4).
// Activations are staged 128 columns x 16 tokens at a time and shared by all
// 16 rows of the workgroup.
//
// Fixed reduction order, independent of M and of the subgroup hardware.
// Lane partial: ascending blocks, one rounded product (`product`) and one add
// per block. Team total: xor butterfly over the 32 lanes with masks
// 16, 8, 4, 2, 1, each step x = x + x[lane ^ mask]. Addition is commutative,
// so both lanes of a pair hold identical bits and every lane ends with the same
// total. Subgroup shuffles are used only when the team occupies an aligned 32-lane slice
// of one hardware subgroup in team order; otherwise the same butterfly runs
// through workgroup memory. Both paths produce identical bits, so a token's
// output depends only on its own activations: M = 1 == M > 1.
//
// Precision tier: weights are dequantized in f16 (q * scale, then + low, each
// rounded to f16 with a flush of f16 subnormals); products and sums are f32.
// The WARP conformance twin emulates the two half roundings in f32 words and
// selects the shared butterfly. The scheduling benchmark uses full f32 dequant.
enable f16;
enable subgroups;

struct Params {
  rows: u32, cols: u32, weight_offset: u32, input_offset: u32,
  output_offset: u32, residual: u32, tokens: u32, output_stride: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> weights: array<u32>;
@group(0) @binding(2) var<storage,read> input: array<f32>;
@group(0) @binding(3) var<storage,read_write> output: array<f32>;

// Debug/conformance switch: force the workgroup-memory butterfly.
override force_shared: bool = false;

const TEAMS: u32 = 8u;
const ROWS: u32 = 2u;

var<workgroup> activation: array<vec4<f32>, 512>; // [pack][128 columns], 8 KiB
var<workgroup> exchange: array<vec4<f32>, 256>;   // butterfly staging, 4 KiB
var<workgroup> team_tags: array<u32, 8>;
var<workgroup> linear_flag: atomic<u32>;
var<workgroup> use_subgroup: u32;

// BEGIN-PRECISION f16
// Materialize each half result and flush its subnormal bit patterns to signed zero.
fn dq_keep(x: f16) -> f16 {
  let bits = bitcast<u32>(f32(x));
  let zero = f16(bitcast<f32>(bits & 0x80000000u));
  return select(x, zero, (bits & 0x7fffffffu) < 0x38800000u);
}
fn dequant(q: u32, scale: f32, low: f32) -> f32 {
  return f32(dq_keep(dq_keep(f16(q) * dq_keep(f16(scale))) + dq_keep(f16(low))));
}
// END-PRECISION

// Rounded f32 product with the portable flush-to-zero convention (as V2).
fn product(w: f32, x: vec4<f32>) -> vec4<f32> {
  let bits = bitcast<vec4<u32>>(x * w);
  return bitcast<vec4<f32>>(select(bits, bits & vec4<u32>(0x80000000u),
    (bits & vec4<u32>(0x7f800000u)) == vec4<u32>(0u)));
}

// exact_mac = true (default): separately rounded product with flush-to-zero, so
// the finite conformance corpus uses the oracle's arithmetic. exact_mac = false: one fma per
// weight and token. Its rounding is implementation-defined, but every token runs
// the same instruction sequence, so M = 1 == M > 1 still holds on each device.
override exact_mac: bool = true;
fn mac(acc: vec4<f32>, w: f32, x: vec4<f32>) -> vec4<f32> {
  if (exact_mac) { return acc + product(w, x); }
  return fma(vec4<f32>(w), x, acc);
}

// Team butterfly. Must be called in workgroup-uniform control flow.
fn reduce(v: vec4<f32>, lane: u32, sg: bool) -> vec4<f32> {
  var x = v;
  let team_base = lane & ~31u;
  let tl = lane & 31u;
  if (sg) {
    for (var m = 16u; m > 0u; m >>= 1u) { x = x + subgroupShuffleXor(x, m); }
  } else {
    for (var m = 16u; m > 0u; m >>= 1u) {
      exchange[lane] = x;
      workgroupBarrier();
      x = x + exchange[team_base | (tl ^ m)];
      workgroupBarrier();
    }
  }
  return x;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_invocation_id) sid: u32,
        @builtin(subgroup_size) sg_size: u32) {
  // BEGIN-SUBGROUP-DETECT
  // Each logical team must fill an aligned 32-lane slice of one physical
  // subgroup, even when local invocation order differs from subgroup order.
  if (lane == 0u) { atomicStore(&linear_flag, 1u); }
  workgroupBarrier();
  if (sg_size < 32u) { atomicStore(&linear_flag, 0u); }
  workgroupBarrier();
  if (lane == 0u) { use_subgroup = select(atomicLoad(&linear_flag), 0u, force_shared); }
  if (workgroupUniformLoad(&use_subgroup) != 0u) {
    let tag = (subgroupBroadcastFirst(lane) << 16u) | (sid >> 5u);
    if ((lane & 31u) == 0u) { team_tags[lane >> 5u] = tag; }
    workgroupBarrier();
    if ((sid & 31u) != (lane & 31u) || tag != team_tags[lane >> 5u]) {
      atomicStore(&linear_flag, 0u);
    }
    workgroupBarrier();
    if (lane == 0u) { use_subgroup = atomicLoad(&linear_flag); }
  }
  let sg = workgroupUniformLoad(&use_subgroup) != 0u;
  // END-SUBGROUP-DETECT

  let team = lane / 32u;
  let tl = lane & 31u;
  let packs = (p.tokens + 3u) / 4u;
  let blocks = (p.cols + 31u) / 32u;
  var acc: array<vec4<f32>, 8>; // [row r][pack] at r*4 + pack
  for (var first = 0u; first < p.cols; first += 128u) {
    let cols = min(128u, p.cols - first);
    for (var flat = lane; flat < packs * cols; flat += 256u) {
      let pack = flat / cols;
      let c = flat % cols;
      var x = vec4<f32>(0.0);
      for (var j = 0u; j < min(4u, p.tokens - 4u * pack); j++) {
        x[j] = input[p.input_offset + (4u * pack + j) * p.cols + first + c];
      }
      activation[pack * 128u + c] = x;
    }
    workgroupBarrier();
    for (var r = 0u; r < ROWS; r++) {
      let row = (wg.x * TEAMS + team) * ROWS + r;
      if (row < p.rows) {
        for (var b = 0u; b * 32u < cols; b++) {
          let c = b * 32u + tl;
          if (c < cols) {
            let base = p.weight_offset + (row * blocks + first / 32u + b) * 5u;
            let affine = unpack2x16float(weights[base + 4u]);
            let q = extractBits(weights[base + tl / 8u], (tl % 8u) * 4u, 4u);
            let w = dequant(q, affine.x, affine.y);
            for (var pack = 0u; pack < packs; pack++) {
              acc[r * 4u + pack] = mac(acc[r * 4u + pack], w, activation[pack * 128u + c]);
            }
          }
        }
      }
    }
    workgroupBarrier();
  }
  let stride = select(p.output_stride, p.rows, p.output_stride == 0u);
  for (var r = 0u; r < ROWS; r++) {
    let row = (wg.x * TEAMS + team) * ROWS + r;
    for (var pack = 0u; pack < packs; pack++) {
      let total = reduce(acc[r * 4u + pack], lane, sg);
      let token = 4u * pack + tl;
      if (row < p.rows && tl < 4u && token < p.tokens) {
        let dst = p.output_offset + token * stride + row;
        var y = total[tl];
        if (p.residual != 0u) { y += output[dst]; }
        output[dst] = y;
      }
    }
  }
}
