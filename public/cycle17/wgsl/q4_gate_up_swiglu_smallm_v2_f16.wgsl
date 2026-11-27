// Q4_GATE_UP_SWIGLU_SMALLM_V2_F16: Tier-1 (shader-f16 + subgroups) cooperative
// fused gate/up projection + SwiGLU, M = 1..16. Same bindings and uniform
// words as q4_gate_up_swiglu_smallm_v2.wgsl (last word = up-weight offset,
// dense output); dispatch (ceil(rows / 8), 1, 1).
//
// Identical structure to q4_gemv_smallm_v2_f16.wgsl with one row per team and
// two weight streams (gate, up) sharing the staged activations. Each stream is
// reduced with the same fixed 32-lane xor butterfly, then SwiGLU uses V2's
// staged sigmoid. See that file for the determinism argument.
enable f16;
enable subgroups;

struct Params {
  rows: u32, cols: u32, weight_offset: u32, input_offset: u32,
  output_offset: u32, residual: u32, tokens: u32, up_offset: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> weights: array<u32>;
@group(0) @binding(2) var<storage,read> input: array<f32>;
@group(0) @binding(3) var<storage,read_write> output: array<f32>;

override force_shared: bool = false;

const TEAMS: u32 = 8u;

var<workgroup> activation: array<vec4<f32>, 512>;
var<workgroup> exchange: array<vec4<f32>, 256>;
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

fn product(w: f32, x: vec4<f32>) -> vec4<f32> {
  let bits = bitcast<vec4<u32>>(x * w);
  return bitcast<vec4<f32>>(select(bits, bits & vec4<u32>(0x80000000u),
    (bits & vec4<u32>(0x7f800000u)) == vec4<u32>(0u)));
}

fn stage(x: f32) -> f32 {
  let bits = bitcast<u32>(x);
  return bitcast<f32>(select(bits, bits & 0x80000000u, (bits & 0x7f800000u) == 0u));
}

// exp for x <= 0, same reduced polynomial as V2 and the EXP-007 f32 oracle.
fn exp_neg(x: f32) -> f32 {
  if (x <= -87.33654) { return 0.0; }
  let n = i32(round(x * 1.4426950408889634));
  let r = (x - f32(n)*0.693359375) - f32(n)*-0.00021219444005469058;
  var y = stage(r*0.0013888889) + 0.0083333338;
  y = stage(r*y) + 0.0416666679;
  y = stage(r*y) + 0.1666666716;
  y = stage(r*y) + 0.5;
  y = stage(r*y) + 1.0;
  y = stage(r*y) + 1.0;
  return bitcast<f32>(u32(n+127)<<23u)*y;
}
fn swiglu(gate: f32, up: f32) -> f32 {
  let e = exp_neg(-abs(gate));
  let sigmoid = stage(select(1.0,e,gate < 0.0) * stage(1.0/(1.0+e)));
  return stage(stage(gate*sigmoid)*up);
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
  let row = wg.x * TEAMS + team;
  let packs = (p.tokens + 3u) / 4u;
  let blocks = (p.cols + 31u) / 32u;
  var gate_acc: array<vec4<f32>, 4>;
  var up_acc: array<vec4<f32>, 4>;
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
    if (row < p.rows) {
      for (var b = 0u; b * 32u < cols; b++) {
        let c = b * 32u + tl;
        if (c < cols) {
          let relative = (row * blocks + first / 32u + b) * 5u;
          let base = p.weight_offset + relative;
          let up_base = p.up_offset + relative;
          let affine = unpack2x16float(weights[base + 4u]);
          let up_affine = unpack2x16float(weights[up_base + 4u]);
          let w = dequant(extractBits(weights[base + tl / 8u], (tl % 8u) * 4u, 4u), affine.x, affine.y);
          let up_w = dequant(extractBits(weights[up_base + tl / 8u], (tl % 8u) * 4u, 4u), up_affine.x, up_affine.y);
          for (var pack = 0u; pack < packs; pack++) {
            let x = activation[pack * 128u + c];
            gate_acc[pack] = mac(gate_acc[pack], w, x);
            up_acc[pack] = mac(up_acc[pack], up_w, x);
          }
        }
      }
    }
    workgroupBarrier();
  }
  for (var pack = 0u; pack < packs; pack++) {
    let gate = reduce(gate_acc[pack], lane, sg);
    let up = reduce(up_acc[pack], lane, sg);
    let token = 4u * pack + tl;
    if (row < p.rows && tl < 4u && token < p.tokens) {
      let dst = p.output_offset + token * p.rows + row;
      var y = swiglu(gate[tl], up[tl]);
      if (p.residual != 0u) { y += output[dst]; }
      output[dst] = y;
    }
  }
}
