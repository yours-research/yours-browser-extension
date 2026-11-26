pub fn get_shader(shader_id: u32) -> Option<&'static [u8]> {
    match shader_id {
        134 => Some(WGSL_G4_Q4_GEMV_VEC4_SG_F16.as_bytes()),
        135 => Some(WGSL_G4_Q4_GATE_UP_GEGLU_VEC4_SG_F16.as_bytes()),
        136 => Some(WGSL_G4_Q4_GEMV_WIDE_K_VEC4_SG_F16.as_bytes()),
        137 => Some(WGSL_G4_Q2_LM_HEAD_VEC4_SG_F16.as_bytes()),
        138 => Some(WGSL_G4_GQA_DECODE_GLOBAL_SPLIT_DP4A.as_bytes()),
        139 => Some(WGSL_G4_Q8_GEMV_DP4A.as_bytes()),
        140 => Some(WGSL_G4_GQA_PREFILL_GLOBAL_FLASH_DP4A.as_bytes()),
        141 => Some(WGSL_G4_GQA_PREFILL_SLIDING_FLASH.as_bytes()),
        142 => Some(WGSL_G4_GQA_PREFILL_REDUCE.as_bytes()),
        143 => Some(WGSL_G4_PLE_HOT_DEQUANT.as_bytes()),
        144 => Some(WGSL_G4_Q4_GEMV_DP4A_W4A8.as_bytes()),
        145 => Some(WGSL_G4_Q2_LM_HEAD_DP4A_W2A8.as_bytes()),
        146 => Some(WGSL_G4_GQA_DECODE_GLOBAL_WARP_INT4.as_bytes()),
        147 => Some(WGSL_G4_GQA_PREFILL_GLOBAL_FLASH_INT4.as_bytes()),
        148 => Some(WGSL_G4_GQA_DECODE_SLIDING_WARP.as_bytes()),
        149 => Some(WGSL_G4_GQA_DECODE_GLOBAL_WARP_DP4A.as_bytes()),
        _ => None,
    }
}

// ============================================================================
// 134: 128-Bit Coalesced vec4<u32> 32-Lane Factored Pure-Q4 GEMV (K <= 2560, T=1)
// Dispatch: (ceil(rows / 4), 1, 1) @workgroup_size(128) -> 1 row per 32-lane warp
// params.q_u32_offset holds q_vec4_offset (qOffset >> 4)
// params.s_u32_offset holds s_vec4_offset (sOffset >> 4)
// ============================================================================
const WGSL_G4_Q4_GEMV_VEC4_SG_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_vec4_offset: u32,
  s_vec4_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<f32, 128>;

fn q4_word_f16(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn read_scale_vec4(s_vec4_off: u32, row: u32) -> f32 {
  let sv = weights[s_vec4_off + (row >> 3u)];
  let wi = (row >> 1u) & 3u;
  let w = select(select(sv.x, sv.y, (wi & 1u) != 0u), select(sv.z, sv.w, (wi & 1u) != 0u), (wi & 2u) != 0u);
  let sp = unpack2x16float(w);
  return select(sp.x, sp.y, (row & 1u) != 0u);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let warp = tid >> 5u;
  let lane = tid & 31u;
  let row = wid.x * 4u + warp;
  let safe_row = min(row, max(params.rows, 1u) - 1u);

  let vec4_x = params.cols >> 2u; // <= 640
  let ones = vec4<f16>(1.0h);
  var sx_loc: f32 = 0.0;
  for (var i: u32 = tid; i < vec4_x; i = i + 128u) {
    let kb = params.in_offset + (i << 2u);
    let xv = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    x_sh[i] = xv;
    sx_loc += f32(dot(ones, xv));
  }
  let sg_sx = subgroupAdd(sx_loc);
  sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
  workgroupBarrier();

  let sg_step = max(sg_size, 1u);
  var total_sum_x: f32 = 0.0;
  for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
    total_sum_x += sumx_sh[s];
  }

  let row_vec4s = params.cols >> 5u; // 32 Q4 weights per vec4<u32>
  let q_base = params.q_vec4_offset + safe_row * row_vec4s;
  var acc: f32 = 0.0;
  for (var w: u32 = lane; w < row_vec4s; w = w + 32u) {
    let wv = weights[q_base + w];
    let xb = w << 3u;
    acc += q4_word_f16(wv.x, x_sh[xb + 0u], x_sh[xb + 1u])
      + q4_word_f16(wv.y, x_sh[xb + 2u], x_sh[xb + 3u])
      + q4_word_f16(wv.z, x_sh[xb + 4u], x_sh[xb + 5u])
      + q4_word_f16(wv.w, x_sh[xb + 6u], x_sh[xb + 7u]);
  }

  if (sg_size == 32u) {
    acc = subgroupAdd(acc);
  } else if (sg_size > 32u) {
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
  } else {
    red_sh[tid] = acc;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc: f32 = 0.0;
      let base = warp * 32u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        s_acc += red_sh[base + l];
      }
      acc = s_acc;
    }
  }

  if (lane == 0u && row < params.rows) {
    let scale = read_scale_vec4(params.s_vec4_offset, safe_row);
    let val = scale * (acc - 8.0 * total_sum_x);
    let out_i = params.out_offset + row;
    y[out_i] = select(val, y[out_i] + val, params.mode != 0u);
  }
}
"#;

// ============================================================================
// 135: 128-Bit Coalesced vec4<u32> Fused Q4 gate_up_proj + GeGLU (K <= 2560, T=1)
// Dispatch: (ceil(rows / 4), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GATE_UP_GEGLU_VEC4_SG_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_vec4_offset: u32,
  s_vec4_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn q4_word_f16(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn read_scale_vec4(s_vec4_off: u32, row: u32) -> f32 {
  let sv = weights[s_vec4_off + (row >> 3u)];
  let wi = (row >> 1u) & 3u;
  let w = select(select(sv.x, sv.y, (wi & 1u) != 0u), select(sv.z, sv.w, (wi & 1u) != 0u), (wi & 2u) != 0u);
  let sp = unpack2x16float(w);
  return select(sp.x, sp.y, (row & 1u) != 0u);
}

fn gelu_tanh(v: f32) -> f32 {
  let arg = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(arg));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let warp = tid >> 5u;
  let lane = tid & 31u;
  let row = wid.x * 4u + warp;
  let rg = min(row, max(params.rows, 1u) - 1u);
  let ru = rg + params.rows;

  let vec4_x = params.cols >> 2u;
  let ones = vec4<f16>(1.0h);
  var sx_loc: f32 = 0.0;
  for (var i: u32 = tid; i < vec4_x; i = i + 128u) {
    let kb = params.in_offset + (i << 2u);
    let xv = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    x_sh[i] = xv;
    sx_loc += f32(dot(ones, xv));
  }
  let sg_sx = subgroupAdd(sx_loc);
  sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
  workgroupBarrier();

  let sg_step = max(sg_size, 1u);
  var total_sum_x: f32 = 0.0;
  for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
    total_sum_x += sumx_sh[s];
  }

  let row_vec4s = params.cols >> 5u;
  let og = params.q_vec4_offset + rg * row_vec4s;
  let ou = params.q_vec4_offset + ru * row_vec4s;
  var acc_g: f32 = 0.0;
  var acc_u: f32 = 0.0;

  for (var w: u32 = lane; w < row_vec4s; w = w + 32u) {
    let xb = w << 3u;
    let x0 = x_sh[xb + 0u]; let x1 = x_sh[xb + 1u];
    let x2 = x_sh[xb + 2u]; let x3 = x_sh[xb + 3u];
    let x4 = x_sh[xb + 4u]; let x5 = x_sh[xb + 5u];
    let x6 = x_sh[xb + 6u]; let x7 = x_sh[xb + 7u];

    let wg = weights[og + w];
    acc_g += q4_word_f16(wg.x, x0, x1) + q4_word_f16(wg.y, x2, x3)
      + q4_word_f16(wg.z, x4, x5) + q4_word_f16(wg.w, x6, x7);

    let wu = weights[ou + w];
    acc_u += q4_word_f16(wu.x, x0, x1) + q4_word_f16(wu.y, x2, x3)
      + q4_word_f16(wu.z, x4, x5) + q4_word_f16(wu.w, x6, x7);
  }

  var sum2 = vec2<f32>(acc_g, acc_u);
  if (sg_size == 32u) {
    sum2 = subgroupAdd(sum2);
  } else if (sg_size > 32u) {
    sum2 += subgroupShuffleXor(sum2, 16u);
    sum2 += subgroupShuffleXor(sum2, 8u);
    sum2 += subgroupShuffleXor(sum2, 4u);
    sum2 += subgroupShuffleXor(sum2, 2u);
    sum2 += subgroupShuffleXor(sum2, 1u);
  } else {
    red_sh[tid] = sum2;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc = vec2<f32>(0.0);
      let base = warp * 32u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        s_acc += red_sh[base + l];
      }
      sum2 = s_acc;
    }
  }

  if (lane == 0u && row < params.rows) {
    let sg = read_scale_vec4(params.s_vec4_offset, rg);
    let su = read_scale_vec4(params.s_vec4_offset, ru);
    let g_val = sg * (sum2.x - 8.0 * total_sum_x);
    let u_val = su * (sum2.y - 8.0 * total_sum_x);
    y[params.out_offset + row] = gelu_tanh(g_val) * u_val;
  }
}
"#;

// ============================================================================
// 136: Double-Buffered vec4<u32> Wide-K GEMV (K = 4096 / 10240, T=1)
// Dispatch: (ceil(rows / 4), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GEMV_WIDE_K_VEC4_SG_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_vec4_offset: u32,
  s_vec4_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 1024>; // 2 ping-pong tiles of 512 vec4<f16> (2048 floats = 64 vec4<u32> weights)
var<workgroup> sumx_sh: array<f32, 256>;
var<workgroup> red_sh: array<f32, 128>;

fn q4_word_f16(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn read_scale_vec4(s_vec4_off: u32, row: u32) -> f32 {
  let sv = weights[s_vec4_off + (row >> 3u)];
  let wi = (row >> 1u) & 3u;
  let w = select(select(sv.x, sv.y, (wi & 1u) != 0u), select(sv.z, sv.w, (wi & 1u) != 0u), (wi & 2u) != 0u);
  let sp = unpack2x16float(w);
  return select(sp.x, sp.y, (row & 1u) != 0u);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let warp = tid >> 5u;
  let lane = tid & 31u;
  let row = wid.x * 4u + warp;
  let safe_row = min(row, max(params.rows, 1u) - 1u);

  let row_vec4s = params.cols >> 5u;
  let q_base = params.q_vec4_offset + safe_row * row_vec4s;
  let ones = vec4<f16>(1.0h);
  let sg_step = max(sg_size, 1u);

  var acc: f32 = 0.0;
  var total_sum_x: f32 = 0.0;
  var buf_idx: u32 = 0u;

  for (var k_tile: u32 = 0u; k_tile < row_vec4s; k_tile = k_tile + 64u) {
    let tile_w = min(64u, row_vec4s - k_tile);
    let tile_v4 = tile_w << 3u; // <= 512
    let sh_base = buf_idx * 512u;
    let sx_base = buf_idx * 128u;
    let x_base = params.in_offset + (k_tile << 5u);

    var tile_sx_loc: f32 = 0.0;
    for (var i: u32 = tid; i < tile_v4; i = i + 128u) {
      let kb = x_base + (i << 2u);
      let xv = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      x_sh[sh_base + i] = xv;
      tile_sx_loc += f32(dot(ones, xv));
    }
    let sg_sx = subgroupAdd(tile_sx_loc);
    sumx_sh[sx_base + tid] = select(0.0, sg_sx, sg_inv == 0u);
    workgroupBarrier();

    var tile_sx: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      tile_sx += sumx_sh[sx_base + s];
    }
    total_sum_x += tile_sx;

    for (var w: u32 = lane; w < tile_w; w = w + 32u) {
      let wv = weights[q_base + k_tile + w];
      let xb = sh_base + (w << 3u);
      acc += q4_word_f16(wv.x, x_sh[xb + 0u], x_sh[xb + 1u])
        + q4_word_f16(wv.y, x_sh[xb + 2u], x_sh[xb + 3u])
        + q4_word_f16(wv.z, x_sh[xb + 4u], x_sh[xb + 5u])
        + q4_word_f16(wv.w, x_sh[xb + 6u], x_sh[xb + 7u]);
    }
    buf_idx = buf_idx ^ 1u;
  }

  if (sg_size == 32u) {
    acc = subgroupAdd(acc);
  } else if (sg_size > 32u) {
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
  } else {
    red_sh[tid] = acc;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc: f32 = 0.0;
      let base = warp * 32u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        s_acc += red_sh[base + l];
      }
      acc = s_acc;
    }
  }

  if (lane == 0u && row < params.rows) {
    let scale = read_scale_vec4(params.s_vec4_offset, safe_row);
    let val = scale * (acc - 8.0 * total_sum_x);
    let out_i = params.out_offset + row;
    y[out_i] = select(val, y[out_i] + val, params.mode != 0u);
  }
}
"#;

// ============================================================================
// 137: 128-Bit Coalesced vec4<u32> 32-Lane Pure-Q2 LM Head + Softcap(30.0)
// 8 rows/WG (2 rows/warp -> 262144 / 8 = 32768 WGs <= 65535 WebGPU limit!)
// Dispatch: (ceil(rows / 8), T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q2_LM_HEAD_VEC4_SG_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_vec4_offset: u32,
  s_vec4_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn q2_word_f16(p: u32, xb: u32) -> f32 {
  let x0 = x_sh[xb + 0u]; let x1 = x_sh[xb + 1u];
  let x2 = x_sh[xb + 2u]; let x3 = x_sh[xb + 3u];
  let q0 = vec4<f16>(f16(p & 3u), f16((p >> 2u) & 3u), f16((p >> 4u) & 3u), f16((p >> 6u) & 3u));
  let q1 = vec4<f16>(f16((p >> 8u) & 3u), f16((p >> 10u) & 3u), f16((p >> 12u) & 3u), f16((p >> 14u) & 3u));
  let q2 = vec4<f16>(f16((p >> 16u) & 3u), f16((p >> 18u) & 3u), f16((p >> 20u) & 3u), f16((p >> 22u) & 3u));
  let q3 = vec4<f16>(f16((p >> 24u) & 3u), f16((p >> 26u) & 3u), f16((p >> 28u) & 3u), f16((p >> 30u) & 3u));
  return f32(dot(q0, x0)) + f32(dot(q1, x1)) + f32(dot(q2, x2)) + f32(dot(q3, x3));
}

fn read_scale_vec4(s_vec4_off: u32, row: u32) -> f32 {
  let sv = weights[s_vec4_off + (row >> 3u)];
  let wi = (row >> 1u) & 3u;
  let w = select(select(sv.x, sv.y, (wi & 1u) != 0u), select(sv.z, sv.w, (wi & 1u) != 0u), (wi & 2u) != 0u);
  let sp = unpack2x16float(w);
  return select(sp.x, sp.y, (row & 1u) != 0u);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let warp = tid >> 5u;
  let lane = tid & 31u;
  let tok = wid.y;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 8u + warp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, max_r);
  let r1 = min(rB, max_r);

  let vec4_x = params.cols >> 2u; // 640
  let x_tok_base = params.in_offset + tok * params.cols;
  let ones = vec4<f16>(1.0h);
  var sx_loc: f32 = 0.0;
  for (var i: u32 = tid; i < vec4_x; i = i + 128u) {
    let kb = x_tok_base + (i << 2u);
    let xv = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    x_sh[i] = xv;
    sx_loc += f32(dot(ones, xv));
  }
  let sg_sx = subgroupAdd(sx_loc);
  sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
  workgroupBarrier();

  let sg_step = max(sg_size, 1u);
  var total_sum_x: f32 = 0.0;
  for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
    total_sum_x += sumx_sh[s];
  }

  let row_vec4s = params.cols >> 6u; // 40 vec4<u32>s for cols=2560
  let o0 = params.q_vec4_offset + r0 * row_vec4s;
  let o1 = params.q_vec4_offset + r1 * row_vec4s;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;

  for (var w: u32 = lane; w < row_vec4s; w = w + 32u) {
    let xb = w << 4u;
    let w0 = weights[o0 + w];
    let w1 = weights[o1 + w];
    acc0 += q2_word_f16(w0.x, xb + 0u) + q2_word_f16(w0.y, xb + 4u)
      + q2_word_f16(w0.z, xb + 8u) + q2_word_f16(w0.w, xb + 12u);
    acc1 += q2_word_f16(w1.x, xb + 0u) + q2_word_f16(w1.y, xb + 4u)
      + q2_word_f16(w1.z, xb + 8u) + q2_word_f16(w1.w, xb + 12u);
  }

  var sum2 = vec2<f32>(acc0, acc1);
  if (sg_size == 32u) {
    sum2 = subgroupAdd(sum2);
  } else if (sg_size > 32u) {
    sum2 += subgroupShuffleXor(sum2, 16u);
    sum2 += subgroupShuffleXor(sum2, 8u);
    sum2 += subgroupShuffleXor(sum2, 4u);
    sum2 += subgroupShuffleXor(sum2, 2u);
    sum2 += subgroupShuffleXor(sum2, 1u);
  } else {
    red_sh[tid] = sum2;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc = vec2<f32>(0.0);
      let base = warp * 32u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        s_acc += red_sh[base + l];
      }
      sum2 = s_acc;
    }
  }

  if (lane == 0u) {
    let s0 = read_scale_vec4(params.s_vec4_offset, r0);
    let s1 = read_scale_vec4(params.s_vec4_offset, r1);
    var v0 = s0 * (sum2.x - 2.0 * total_sum_x);
    var v1 = s1 * (sum2.y - 2.0 * total_sum_x);
    if (params.mode != 0u) {
      v0 = 30.0 * tanh(clamp(v0 / 30.0, -15.0, 15.0));
      v1 = 30.0 * tanh(clamp(v1 / 30.0, -15.0, 15.0));
    }
    let tok_out = params.out_offset + tok * params.rows;
    if (rA < params.rows) { y[tok_out + rA] = v0; }
    if (rB < params.rows) { y[tok_out + rB] = v1; }
  }
}
"#;

// ============================================================================
// 138: Hardware DP4A GQA Decode Global Split-K (D=512, INT8 KV, 1 barrier/tok!)
// Dispatch: (2, num_splits, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_DECODE_GLOBAL_SPLIT_DP4A: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  in_seq_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_global: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> q_max_sh: array<vec4<f32>, 128>;
var<workgroup> red_sh: array<vec4<f32>, 256>;

fn pack4_i8(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

fn unpack4_i8(word: u32) -> vec4<f32> {
  let sw = bitcast<i32>(word);
  return vec4<f32>(
    f32(extractBits(sw, 0u, 8u)),
    f32(extractBits(sw, 8u, 8u)),
    f32(extractBits(sw, 16u, 8u)),
    f32(extractBits(sw, 24u, 8u)),
  );
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits, 1u);
  let qh0 = kv_head * 4u;
  let tid = lid.x;
  let d = tid * 4u;
  let sg_step = max(sg_size, 1u);

  let b0 = (qh0 + 0u) * 512u + d;
  let b1 = (qh0 + 1u) * 512u + d;
  let b2 = (qh0 + 2u) * 512u + d;
  let b3 = (qh0 + 3u) * 512u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let qm0 = subgroupMax(max(max(abs(q0.x), abs(q0.y)), max(abs(q0.z), abs(q0.w))));
  let qm1 = subgroupMax(max(max(abs(q1.x), abs(q1.y)), max(abs(q1.z), abs(q1.w))));
  let qm2 = subgroupMax(max(max(abs(q2.x), abs(q2.y)), max(abs(q2.z), abs(q2.w))));
  let qm3 = subgroupMax(max(max(abs(q3.x), abs(q3.y)), max(abs(q3.z), abs(q3.w))));
  q_max_sh[tid] = select(vec4<f32>(0.0), vec4<f32>(qm0, qm1, qm2, qm3), sg_inv == 0u);
  workgroupBarrier();

  var qm_all = vec4<f32>(0.0);
  for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
    qm_all = max(qm_all, q_max_sh[s]);
  }
  let q_scales = max(qm_all / 127.0, vec4<f32>(1e-6));
  let q0_i8 = pack4_i8(q0 / q_scales.x);
  let q1_i8 = pack4_i8(q1 / q_scales.y);
  let q2_i8 = pack4_i8(q2 / q_scales.z);
  let q3_i8 = pack4_i8(q3 / q_scales.w);

  let head_u32s = params.max_seq * 129u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let total_toks = params.seq_pos + 1u;
  let chunk = (total_toks + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, total_toks);
  let t_end = min(t_start + chunk, total_toks);

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let ko = k_base + t * 129u;
    let ks = bitcast<f32>(kv_global[ko + 128u]);
    let kw = kv_global[ko + tid];

    let p4 = vec4<f32>(
      f32(dot4I8Packed(q0_i8, kw)),
      f32(dot4I8Packed(q1_i8, kw)),
      f32(dot4I8Packed(q2_i8, kw)),
      f32(dot4I8Packed(q3_i8, kw)),
    );
    let sh_slot = ((t - t_start) & 1u) * 128u;
    let sg_dot = subgroupAdd(p4);
    red_sh[sh_slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var full_dot = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      full_dot += red_sh[sh_slot + s];
    }

    let score = full_dot * (q_scales * ks);
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let vo = v_base + t * 129u;
    let vs = bitcast<f32>(kv_global[vo + 128u]);
    let v4 = unpack4_i8(kv_global[vo + tid]) * vs;
    acc0 = acc0 * alpha.x + v4 * beta.x;
    acc1 = acc1 * alpha.y + v4 * beta.y;
    acc2 = acc2 * alpha.z + v4 * beta.z;
    acc3 = acc3 * alpha.w + v4 * beta.w;
    m4 = m_new;
  }

  if (num_splits == 1u) {
    let o0 = acc0 / max(l4.x, 1e-9);
    let o1 = acc1 / max(l4.y, 1e-9);
    let o2 = acc2 / max(l4.z, 1e-9);
    let o3 = acc3 / max(l4.w, 1e-9);
    let s0 = (qh0 + 0u) * 512u + d;
    let s1 = (qh0 + 1u) * 512u + d;
    let s2 = (qh0 + 2u) * 512u + d;
    let s3 = (qh0 + 3u) * 512u + d;
    split_out[s0] = o0.x; split_out[s0 + 1u] = o0.y; split_out[s0 + 2u] = o0.z; split_out[s0 + 3u] = o0.w;
    split_out[s1] = o1.x; split_out[s1 + 1u] = o1.y; split_out[s1 + 2u] = o1.z; split_out[s1 + 3u] = o1.w;
    split_out[s2] = o2.x; split_out[s2 + 1u] = o2.y; split_out[s2 + 2u] = o2.z; split_out[s2 + 3u] = o2.w;
    split_out[s3] = o3.x; split_out[s3 + 1u] = o3.y; split_out[s3 + 2u] = o3.z; split_out[s3 + 3u] = o3.w;
    return;
  }

  let ob0 = ((qh0 + 0u) * num_splits + split_id) * 516u;
  split_out[ob0 + d] = acc0.x; split_out[ob0 + d + 1u] = acc0.y; split_out[ob0 + d + 2u] = acc0.z; split_out[ob0 + d + 3u] = acc0.w;
  if (tid == 0u) { split_out[ob0 + 512u] = m4.x; split_out[ob0 + 513u] = l4.x; }

  let ob1 = ((qh0 + 1u) * num_splits + split_id) * 516u;
  split_out[ob1 + d] = acc1.x; split_out[ob1 + d + 1u] = acc1.y; split_out[ob1 + d + 2u] = acc1.z; split_out[ob1 + d + 3u] = acc1.w;
  if (tid == 0u) { split_out[ob1 + 512u] = m4.y; split_out[ob1 + 513u] = l4.y; }

  let ob2 = ((qh0 + 2u) * num_splits + split_id) * 516u;
  split_out[ob2 + d] = acc2.x; split_out[ob2 + d + 1u] = acc2.y; split_out[ob2 + d + 2u] = acc2.z; split_out[ob2 + d + 3u] = acc2.w;
  if (tid == 0u) { split_out[ob2 + 512u] = m4.z; split_out[ob2 + 513u] = l4.z; }

  let ob3 = ((qh0 + 3u) * num_splits + split_id) * 516u;
  split_out[ob3 + d] = acc3.x; split_out[ob3 + d + 1u] = acc3.y; split_out[ob3 + d + 2u] = acc3.z; split_out[ob3 + d + 3u] = acc3.w;
  if (tid == 0u) { split_out[ob3 + 512u] = m4.w; split_out[ob3 + 513u] = l4.w; }
}
"#;

// ============================================================================
// 139: Hardware DP4A Pure-Q8 GEMV (ple_gate 2560->256, ple_proj 256->2560)
// Exact bit trick: signed_w_i8 = (u8 ^ 0x80) -> zero sum(x) correction!
// Dispatch: (ceil(rows / 8), T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q8_GEMV_DP4A: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> extra: array<f32>;

var<workgroup> x_i8_sh: array<u32, 640>;
var<workgroup> x_gscale_sh: array<f32, 40>;
var<workgroup> x_max_sh: array<f32, 128>;
var<workgroup> red_sh: array<f32, 128>;

fn pack4_i8(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

fn gelu_tanh(v: f32) -> f32 {
  let arg = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(arg));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let grp = tid >> 4u;  // 0..7 (8 groups of 16 lanes = 64 floats per group!)
  let lane = tid & 15u; // 0..15
  let row = wid.x * 8u + grp;
  let tok = wid.y;
  let safe_row = min(row, max(params.rows, 1u) - 1u);

  let u32_words = params.cols >> 2u; // 64 or 640
  let x_tok_base = params.in_offset + tok * params.cols;

  for (var base: u32 = 0u; base < u32_words; base = base + 128u) {
    let idx = base + tid;
    var xv = vec4<f32>(0.0);
    var ma: f32 = 0.0;
    if (idx < u32_words) {
      let kb = x_tok_base + (idx << 2u);
      xv = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
      ma = max(max(abs(xv.x), abs(xv.y)), max(abs(xv.z), abs(xv.w)));
    }

    if (sg_size >= 16u) {
      ma = max(ma, subgroupShuffleXor(ma, 8u));
      ma = max(ma, subgroupShuffleXor(ma, 4u));
      ma = max(ma, subgroupShuffleXor(ma, 2u));
      ma = max(ma, subgroupShuffleXor(ma, 1u));
    } else {
      x_max_sh[tid] = ma;
      workgroupBarrier();
      var gm: f32 = 0.0;
      let gb = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        gm = max(gm, x_max_sh[gb + l]);
      }
      ma = gm;
      workgroupBarrier();
    }

    if (idx < u32_words) {
      let gs = max(ma / 127.0, 1e-6);
      x_i8_sh[idx] = pack4_i8(xv / gs);
      if (lane == 0u) {
        x_gscale_sh[idx >> 4u] = gs;
      }
    }
  }
  workgroupBarrier();

  let w_base = params.q_u32_offset + safe_row * u32_words;
  var acc: f32 = 0.0;
  for (var w_rel: u32 = lane; w_rel < u32_words; w_rel = w_rel + 16u) {
    let w = weights[w_base + w_rel] ^ 0x80808080u;
    let dp = dot4I8Packed(w, x_i8_sh[w_rel]);
    acc += f32(dp) * x_gscale_sh[w_rel >> 4u];
  }

  if (sg_size >= 16u) {
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
  } else {
    red_sh[tid] = acc;
    workgroupBarrier();
    if (lane == 0u) {
      var sa: f32 = 0.0;
      let gb = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        sa += red_sh[gb + l];
      }
      acc = sa;
    }
  }

  if (lane == 0u && row < params.rows) {
    let sp = unpack2x16float(weights[params.s_u32_offset + (safe_row >> 1u)]);
    let row_scale = select(sp.x, sp.y, (safe_row & 1u) != 0u);
    var val: f32 = row_scale * acc;
    if (params.mode == 1u) {
      val = gelu_tanh(val) * extra[params.extra_offset + tok * 10752u + row];
    } else if (params.mode == 2u) {
      val = val * 0.01976423537605237;
    }
    y[params.out_offset + tok * params.rows + row] = val;
  }
}
"#;

// ============================================================================
// 140: FlashAttention-2 Q-Block Tiled DP4A Prefill Global (B_q=4 x 2 Q heads, B_kv=16, Split-K)
// 8x fewer kv_global DRAM reads + 16x fewer barriers + Split-K full SM saturation!
// Dispatch: (4, ceil(tokens / 4), num_splits) @workgroup_size(128)
// params.in_seq_offset holds `tokens`
// ============================================================================
const WGSL_G4_GQA_PREFILL_GLOBAL_FLASH_DP4A: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_global: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;

var<workgroup> q_i8_sh: array<u32, 1024>;    // 8 Q vecs * 128 u32s = 4096 B
var<workgroup> q_scale_sh: array<f32, 8>;
var<workgroup> qmax_sh: array<f32, 128>;
var<workgroup> kv_sh: array<u32, 2048>;      // 16 KV vecs * 128 u32s = 8192 B
var<workgroup> kv_scale_sh: array<f32, 16>;
var<workgroup> s_sh: array<f32, 128>;        // 8 Q vecs * 16 KV tokens = 512 B
var<workgroup> m_sh: array<f32, 8>;
var<workgroup> l_sh: array<f32, 8>;
var<workgroup> alpha_sh: array<f32, 8>;

fn pack4_i8(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

fn unpack4_i8(word: u32) -> vec4<f32> {
  let sw = bitcast<i32>(word);
  return vec4<f32>(
    f32(extractBits(sw, 0u, 8u)),
    f32(extractBits(sw, 8u, 8u)),
    f32(extractBits(sw, 16u, 8u)),
    f32(extractBits(sw, 24u, 8u)),
  );
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let pair_id = wid.x; // 0..3 -> (kv_head, half_pair)
  let kv_head = pair_id >> 1u;
  let qh_base = pair_id * 2u;
  let tok_tile = wid.y * 4u;
  let split_id = wid.z;
  let num_splits = max(params.num_splits, 1u);
  let tid = lid.x; // 0..127 -> d = tid*4..tid*4+3
  let sg_step = max(sg_size, 1u);

  // 1. Load & quantize all 8 Q vectors (4 tokens x 2 Q heads)
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    let qt = r >> 1u;
    let qh_loc = r & 1u;
    let tok = tok_tile + qt;
    let qh = qh_base + qh_loc;
    var qv = vec4<f32>(0.0);
    if (tok < params.tokens) {
      let qb = (tok * 8u + qh) * 512u + tid * 4u;
      qv = vec4<f32>(q_buf[qb], q_buf[qb + 1u], q_buf[qb + 2u], q_buf[qb + 3u]);
    }
    let qm = subgroupMax(max(max(abs(qv.x), abs(qv.y)), max(abs(qv.z), abs(qv.w))));
    qmax_sh[tid] = select(0.0, qm, sg_inv == 0u);
    workgroupBarrier();

    var max_q: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      max_q = max(max_q, qmax_sh[s]);
    }
    let gs = max(max_q / 127.0, 1e-6);
    q_i8_sh[r * 128u + tid] = pack4_i8(qv / gs);
    if (tid == 0u) {
      q_scale_sh[r] = gs;
    }
    workgroupBarrier();
  }
  if (tid < 8u) {
    m_sh[tid] = -1e30;
    l_sh[tid] = 0.0;
    alpha_sh[tid] = 1.0;
  }
  workgroupBarrier();

  let head_u32s = params.max_seq * 129u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let max_k_len = params.seq_pos + min(tok_tile + 4u, params.tokens);
  let chunk = (max_k_len + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, max_k_len);
  let t_end = min(t_start + chunk, max_k_len);

  var acc: array<vec4<f32>, 8>;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    acc[r] = vec4<f32>(0.0);
  }

  let team = tid >> 4u;  // 0..7 (corresponds to Q vector r!)
  let lane16 = tid & 15u;

  for (var t0: u32 = t_start; t0 < t_end; t0 = t0 + 16u) {
    let k_cnt = min(16u, t_end - t0);

    // Load K block [k_cnt, 128] + scales coalesced
    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let ko = k_base + (t0 + j) * 129u;
      kv_sh[j * 128u + tid] = kv_global[ko + tid];
      if (tid == 0u) {
        kv_scale_sh[j] = bitcast<f32>(kv_global[ko + 128u]);
      }
    }
    workgroupBarrier();

    // Compute 8 x k_cnt DP4A scores in parallel across 8 16-lane teams
    let qt_team = team >> 1u;
    let tok_team = tok_tile + qt_team;
    let pos_q_team = params.seq_pos + tok_team;
    let q_off = team * 128u;
    let q_sc = q_scale_sh[team];

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let k_off = j * 128u;
      var d_sum: i32 = 0;
      for (var w: u32 = lane16; w < 128u; w = w + 16u) {
        d_sum += dot4I8Packed(q_i8_sh[q_off + w], kv_sh[k_off + w]);
      }
      if (sg_size >= 16u) {
        d_sum += subgroupShuffleXor(d_sum, 8u);
        d_sum += subgroupShuffleXor(d_sum, 4u);
        d_sum += subgroupShuffleXor(d_sum, 2u);
        d_sum += subgroupShuffleXor(d_sum, 1u);
      } else {
        qmax_sh[tid] = f32(d_sum);
        workgroupBarrier();
        if (lane16 == 0u) {
          var s: f32 = 0.0;
          let gb = team * 16u;
          for (var l: u32 = 0u; l < 16u; l = l + 1u) { s += qmax_sh[gb + l]; }
          d_sum = i32(s);
        }
        workgroupBarrier();
      }

      if (lane16 == 0u) {
        let key_pos = t0 + j;
        var sc: f32 = -1e30;
        if (tok_team < params.tokens && key_pos <= pos_q_team) {
          sc = f32(d_sum) * q_sc * kv_scale_sh[j];
        }
        s_sh[team * 16u + j] = sc;
      }
    }
    workgroupBarrier();

    // Overlap softmax weight update on s_sh (tid < 8) with V block load on kv_sh (all 128)!
    if (tid < 8u) {
      let sb = tid * 16u;
      var m_block: f32 = -1e30;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        m_block = max(m_block, s_sh[sb + j]);
      }
      let m_old = m_sh[tid];
      let m_new = max(m_old, m_block);
      let alpha = exp(m_old - m_new);
      var l_block: f32 = 0.0;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        let w_exp = select(0.0, exp(s_sh[sb + j] - m_new), s_sh[sb + j] > -1e29);
        s_sh[sb + j] = w_exp;
        l_block += w_exp;
      }
      l_sh[tid] = l_sh[tid] * alpha + l_block;
      m_sh[tid] = m_new;
      alpha_sh[tid] = alpha;
    }

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let vo = v_base + (t0 + j) * 129u;
      kv_sh[j * 128u + tid] = kv_global[vo + tid];
      if (tid == 0u) {
        kv_scale_sh[j] = bitcast<f32>(kv_global[vo + 128u]);
      }
    }
    workgroupBarrier();

    // Update all 8 Q accumulators using P_sh * V_sh
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
      var v_acc = vec4<f32>(0.0);
      let sb = r * 16u;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        let w_p = s_sh[sb + j];
        if (w_p > 0.0) {
          v_acc += unpack4_i8(kv_sh[j * 128u + tid]) * (kv_scale_sh[j] * w_p);
        }
      }
      acc[r] = acc[r] * alpha_sh[r] + v_acc;
    }
    if (t0 + 16u < t_end) {
      workgroupBarrier();
    }
  }

  let d = tid * 4u;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    let qt = r >> 1u;
    let tok = tok_tile + qt;
    if (tok >= params.tokens) {
      continue;
    }
    let qh = qh_base + (r & 1u);
    if (num_splits == 1u) {
      let o4 = acc[r] / max(l_sh[r], 1e-9);
      let ob = (tok * 8u + qh) * 512u + d;
      out_buf[ob] = o4.x; out_buf[ob + 1u] = o4.y; out_buf[ob + 2u] = o4.z; out_buf[ob + 3u] = o4.w;
    } else {
      let ob = ((tok * 8u + qh) * num_splits + split_id) * 516u;
      let a4 = acc[r];
      out_buf[ob + d] = a4.x; out_buf[ob + d + 1u] = a4.y; out_buf[ob + d + 2u] = a4.z; out_buf[ob + d + 3u] = a4.w;
      if (tid == 0u) {
        out_buf[ob + 512u] = m_sh[r];
        out_buf[ob + 513u] = l_sh[r];
      }
    }
  }
}
"#;

// ============================================================================
// 141: FlashAttention-2 Q-Block Tiled Prefill Sliding (B_q=4 x 2 Q heads, B_kv=16, D=256)
// Dispatch: (4, ceil(tokens / 4), 1) @workgroup_size(64)
// ============================================================================
const WGSL_G4_GQA_PREFILL_SLIDING_FLASH: &str = r#"
enable f16;
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_sliding: array<u32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

var<workgroup> q_sh: array<vec4<f16>, 512>;  // 8 Q vecs * 64 vec4<f16> = 4096 B
var<workgroup> k_sh: array<vec4<f16>, 1024>; // 16 KV vecs * 64 vec4<f16> = 8192 B
var<workgroup> s_sh: array<f32, 128>;
var<workgroup> red_sh: array<f32, 64>;
var<workgroup> m_sh: array<f32, 8>;
var<workgroup> l_sh: array<f32, 8>;
var<workgroup> alpha_sh: array<f32, 8>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let pair_id = wid.x;
  let kv_head = pair_id >> 1u;
  let qh_base = pair_id * 2u;
  let tok_tile = wid.y * 4u;
  let tid = lid.x; // 0..63 -> d = tid*4..tid*4+3 (D=256)

  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    let qt = r >> 1u;
    let tok = tok_tile + qt;
    let qh = qh_base + (r & 1u);
    var qv = vec4<f16>(0.0h);
    if (tok < params.tokens) {
      let qb = (tok * 8u + qh) * 256u + tid * 4u;
      qv = vec4<f16>(f16(q_buf[qb]), f16(q_buf[qb + 1u]), f16(q_buf[qb + 2u]), f16(q_buf[qb + 3u]));
    }
    q_sh[r * 64u + tid] = qv;
  }
  if (tid < 8u) {
    m_sh[tid] = -1e30;
    l_sh[tid] = 0.0;
    alpha_sh[tid] = 1.0;
  }
  workgroupBarrier();

  let head_u32s = 1024u * 128u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let win_m1 = max(params.window_size, 1u) - 1u;
  let pos0 = params.seq_pos + tok_tile;
  let t_start = pos0 - min(pos0, win_m1);
  let t_end = params.seq_pos + min(tok_tile + 4u, params.tokens);

  var acc: array<vec4<f32>, 8>;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    acc[r] = vec4<f32>(0.0);
  }

  let team = tid >> 3u;  // 0..7 (8 teams of 8 lanes!)
  let lane8 = tid & 7u;

  for (var t0: u32 = t_start; t0 < t_end; t0 = t0 + 16u) {
    let k_cnt = min(16u, t_end - t0);

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let slot = ((t0 + j) & 1023u) * 128u + tid * 2u;
      let ka = unpack2x16float(kv_sliding[k_base + slot]);
      let kb = unpack2x16float(kv_sliding[k_base + slot + 1u]);
      k_sh[j * 64u + tid] = vec4<f16>(f16(ka.x), f16(ka.y), f16(kb.x), f16(kb.y));
    }
    workgroupBarrier();

    let qt_team = team >> 1u;
    let tok_team = tok_tile + qt_team;
    let pos_q_team = params.seq_pos + tok_team;
    let q_off = team * 64u;

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let k_off = j * 64u;
      var d_sum: f32 = 0.0;
      for (var w: u32 = lane8; w < 64u; w = w + 8u) {
        d_sum += f32(dot(q_sh[q_off + w], k_sh[k_off + w]));
      }
      if (sg_size >= 8u) {
        d_sum += subgroupShuffleXor(d_sum, 4u);
        d_sum += subgroupShuffleXor(d_sum, 2u);
        d_sum += subgroupShuffleXor(d_sum, 1u);
      } else {
        red_sh[tid] = d_sum;
        workgroupBarrier();
        if (lane8 == 0u) {
          var s: f32 = 0.0;
          let gb = team * 8u;
          for (var l: u32 = 0u; l < 8u; l = l + 1u) { s += red_sh[gb + l]; }
          d_sum = s;
        }
        workgroupBarrier();
      }
      if (lane8 == 0u) {
        let key_pos = t0 + j;
        var sc: f32 = -1e30;
        if (tok_team < params.tokens && key_pos <= pos_q_team && (pos_q_team - key_pos) <= win_m1) {
          sc = d_sum;
        }
        s_sh[team * 16u + j] = sc;
      }
    }
    workgroupBarrier();

    if (tid < 8u) {
      let sb = tid * 16u;
      var m_block: f32 = -1e30;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        m_block = max(m_block, s_sh[sb + j]);
      }
      let m_old = m_sh[tid];
      let m_new = max(m_old, m_block);
      let alpha = exp(m_old - m_new);
      var l_block: f32 = 0.0;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        let w_exp = select(0.0, exp(s_sh[sb + j] - m_new), s_sh[sb + j] > -1e29);
        s_sh[sb + j] = w_exp;
        l_block += w_exp;
      }
      l_sh[tid] = l_sh[tid] * alpha + l_block;
      m_sh[tid] = m_new;
      alpha_sh[tid] = alpha;
    }

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let slot = ((t0 + j) & 1023u) * 128u + tid * 2u;
      let va = unpack2x16float(kv_sliding[v_base + slot]);
      let vb = unpack2x16float(kv_sliding[v_base + slot + 1u]);
      k_sh[j * 64u + tid] = vec4<f16>(f16(va.x), f16(va.y), f16(vb.x), f16(vb.y));
    }
    workgroupBarrier();

    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
      var v_acc = vec4<f32>(0.0);
      let sb = r * 16u;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        let w_p = s_sh[sb + j];
        if (w_p > 0.0) {
          v_acc += vec4<f32>(k_sh[j * 64u + tid]) * w_p;
        }
      }
      acc[r] = acc[r] * alpha_sh[r] + v_acc;
    }
    if (t0 + 16u < t_end) {
      workgroupBarrier();
    }
  }

  let d = tid * 4u;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    let tok = tok_tile + (r >> 1u);
    if (tok >= params.tokens) {
      continue;
    }
    let qh = qh_base + (r & 1u);
    let o4 = acc[r] / max(l_sh[r], 1e-9);
    let ob = (tok * 8u + qh) * 256u + d;
    attn_out[ob] = o4.x; attn_out[ob + 1u] = o4.y; attn_out[ob + 2u] = o4.z; attn_out[ob + 3u] = o4.w;
  }
}
"#;

// ============================================================================
// 142: Prefill Global Split-K Merge Reduce (8 heads x T tokens x D=512)
// Dispatch: (8, T, 1) @workgroup_size(128) -> 4 dims/thread
// ============================================================================
const WGSL_G4_GQA_PREFILL_REDUCE: &str = r#"
struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> split_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> attn_out: array<f32>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let qh = wid.x;
  let tok = wid.y;
  let d = lid.x * 4u;
  let num_splits = max(params.num_splits, 1u);
  if (tok >= params.tokens) {
    return;
  }

  var m_g: f32 = -1e30;
  var l_g: f32 = 0.0;
  var acc = vec4<f32>(0.0);

  for (var s: u32 = 0u; s < num_splits; s = s + 1u) {
    let base = ((tok * 8u + qh) * num_splits + s) * 516u;
    let l_s = split_out[base + 513u];
    if (l_s <= 0.0) {
      continue;
    }
    let m_s = split_out[base + 512u];
    let m_new = max(m_g, m_s);
    let alpha = exp(m_g - m_new);
    let beta = exp(m_s - m_new);
    l_g = l_g * alpha + l_s * beta;
    acc = acc * alpha + vec4<f32>(split_out[base + d], split_out[base + d + 1u], split_out[base + d + 2u], split_out[base + d + 3u]) * beta;
    m_g = m_new;
  }

  let o4 = acc / max(l_g, 1e-9);
  let ob = (tok * 8u + qh) * 512u + d;
  attn_out[ob] = o4.x;
  attn_out[ob + 1u] = o4.y;
  attn_out[ob + 2u] = o4.z;
  attn_out[ob + 3u] = o4.w;
}
"#;

// ============================================================================
// 148: Zero-Barrier Split-K Warp Sliding Decode (D=256, 4 Q-heads/kv_head)
// Dispatch: (2, num_splits, 1) @workgroup_size(128) -> ZERO workgroup barriers!
// ============================================================================
const WGSL_G4_GQA_DECODE_SLIDING_WARP: &str = r#"
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  in_seq_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_sliding: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits, 1u);
  let tid = lid.x;
  let warp = tid >> 5u; // 0..3 -> 4 Q heads per KV head
  let lane = tid & 31u; // 0..31 -> 8 dims/lane (32 * 8 = 256)
  let qh = kv_head * 4u + warp;
  let d = lane * 8u;

  let qb = qh * 256u + d;
  let qa = vec4<f32>(q_buf[qb], q_buf[qb + 1u], q_buf[qb + 2u], q_buf[qb + 3u]);
  let qb_v = vec4<f32>(q_buf[qb + 4u], q_buf[qb + 5u], q_buf[qb + 6u], q_buf[qb + 7u]);

  let head_v4s = 1024u * 32u; // 128 u32s = 32 vec4<u32>s per token-head
  let donor_v4s = 4u * head_v4s;
  let k_base = params.local_donor_idx * donor_v4s + kv_head * head_v4s;
  let v_base = params.local_donor_idx * donor_v4s + (2u + kv_head) * head_v4s;

  let win_minus_1 = max(params.window_size, 1u) - 1u;
  let win_start = params.seq_pos - min(params.seq_pos, win_minus_1);
  let win_len = (params.seq_pos + 1u) - win_start;
  let chunk = (win_len + num_splits - 1u) / num_splits;
  let rel_start = min(split_id * chunk, win_len);
  let rel_end = min(rel_start + chunk, win_len);
  let t_start = win_start + rel_start;
  let t_end = win_start + rel_end;

  var m: f32 = -1e30;
  var l: f32 = 0.0;
  var acc_a = vec4<f32>(0.0);
  var acc_b = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let slot_v4 = (t & 1023u) * 32u + lane;
    let kv = kv_sliding[k_base + slot_v4];
    let k0 = unpack2x16float(kv.x);
    let k1 = unpack2x16float(kv.y);
    let k2 = unpack2x16float(kv.z);
    let k3 = unpack2x16float(kv.w);
    var score = dot(qa, vec4<f32>(k0.x, k0.y, k1.x, k1.y))
      + dot(qb_v, vec4<f32>(k2.x, k2.y, k3.x, k3.y));

    if (sg_size == 32u) {
      score = subgroupAdd(score);
    } else {
      score += subgroupShuffleXor(score, 16u);
      score += subgroupShuffleXor(score, 8u);
      score += subgroupShuffleXor(score, 4u);
      score += subgroupShuffleXor(score, 2u);
      score += subgroupShuffleXor(score, 1u);
    }

    let m_new = max(m, score);
    let alpha = exp(m - m_new);
    let beta = exp(score - m_new);
    l = l * alpha + beta;

    let vv = kv_sliding[v_base + slot_v4];
    let v0 = unpack2x16float(vv.x);
    let v1 = unpack2x16float(vv.y);
    let v2 = unpack2x16float(vv.z);
    let v3 = unpack2x16float(vv.w);
    acc_a = acc_a * alpha + vec4<f32>(v0.x, v0.y, v1.x, v1.y) * beta;
    acc_b = acc_b * alpha + vec4<f32>(v2.x, v2.y, v3.x, v3.y) * beta;
    m = m_new;
  }

  if (num_splits == 1u) {
    let inv_l = 1.0 / max(l, 1e-9);
    let fa = acc_a * inv_l;
    let fb = acc_b * inv_l;
    split_out[qb] = fa.x; split_out[qb + 1u] = fa.y; split_out[qb + 2u] = fa.z; split_out[qb + 3u] = fa.w;
    split_out[qb + 4u] = fb.x; split_out[qb + 5u] = fb.y; split_out[qb + 6u] = fb.z; split_out[qb + 7u] = fb.w;
    return;
  }

  let ob = (qh * num_splits + split_id) * 260u;
  split_out[ob + d] = acc_a.x; split_out[ob + d + 1u] = acc_a.y; split_out[ob + d + 2u] = acc_a.z; split_out[ob + d + 3u] = acc_a.w;
  split_out[ob + d + 4u] = acc_b.x; split_out[ob + d + 5u] = acc_b.y; split_out[ob + d + 6u] = acc_b.z; split_out[ob + d + 7u] = acc_b.w;
  if (lane == 0u) {
    split_out[ob + 256u] = m;
    split_out[ob + 257u] = l;
  }
}
"#;

// ============================================================================
// 149: Zero-Barrier Warp DP4A Global Split-K Decode (D=512, INT8 KV, 1 warp/Q-head)
// Dispatch: (2, num_splits, 1) @workgroup_size(128) -> 4 warps = 4 Q heads of kv_head
// Coalesced 128B cache-line KV reads, ZERO workgroup barriers in entire kernel!
// ============================================================================
const WGSL_G4_GQA_DECODE_GLOBAL_WARP_DP4A: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  in_seq_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_global: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

fn pack4_i8(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

fn unpack4_i8(word: u32) -> vec4<f32> {
  let sw = bitcast<i32>(word);
  return vec4<f32>(
    f32(extractBits(sw, 0u, 8u)),
    f32(extractBits(sw, 8u, 8u)),
    f32(extractBits(sw, 16u, 8u)),
    f32(extractBits(sw, 24u, 8u)),
  );
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits, 1u);
  let tid = lid.x;
  let warp = tid >> 5u; // 0..3 -> 4 Q heads per KV head
  let lane = tid & 31u; // 0..31 -> 4 coalesced vec4s (dims 0..127, 128..255, 256..383, 384..511)
  let qh = kv_head * 4u + warp;

  let qh_base = qh * 512u;
  let w0 = lane;
  let w1 = 32u + lane;
  let w2 = 64u + lane;
  let w3 = 96u + lane;
  let b0 = qh_base + (w0 << 2u);
  let b1 = qh_base + (w1 << 2u);
  let b2 = qh_base + (w2 << 2u);
  let b3 = qh_base + (w3 << 2u);

  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let mx0 = max(max(abs(q0.x), abs(q0.y)), max(abs(q0.z), abs(q0.w)));
  let mx1 = max(max(abs(q1.x), abs(q1.y)), max(abs(q1.z), abs(q1.w)));
  let mx2 = max(max(abs(q2.x), abs(q2.y)), max(abs(q2.z), abs(q2.w)));
  let mx3 = max(max(abs(q3.x), abs(q3.y)), max(abs(q3.z), abs(q3.w)));
  var qm = max(max(mx0, mx1), max(mx2, mx3));
  if (sg_size == 32u) {
    qm = subgroupMax(qm);
  } else {
    qm = max(qm, subgroupShuffleXor(qm, 16u));
    qm = max(qm, subgroupShuffleXor(qm, 8u));
    qm = max(qm, subgroupShuffleXor(qm, 4u));
    qm = max(qm, subgroupShuffleXor(qm, 2u));
    qm = max(qm, subgroupShuffleXor(qm, 1u));
  }

  let q_scale = max(qm / 127.0, 1e-6);
  let qi0 = pack4_i8(q0 / q_scale);
  let qi1 = pack4_i8(q1 / q_scale);
  let qi2 = pack4_i8(q2 / q_scale);
  let qi3 = pack4_i8(q3 / q_scale);

  let head_u32s = params.max_seq * 129u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let total_toks = params.seq_pos + 1u;
  let chunk = (total_toks + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, total_toks);
  let t_end = min(t_start + chunk, total_toks);

  var m: f32 = -1e30;
  var l: f32 = 0.0;
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let ko = k_base + t * 129u;
    let ks = bitcast<f32>(kv_global[ko + 128u]);
    var dp: i32 = dot4I8Packed(qi0, kv_global[ko + w0])
      + dot4I8Packed(qi1, kv_global[ko + w1])
      + dot4I8Packed(qi2, kv_global[ko + w2])
      + dot4I8Packed(qi3, kv_global[ko + w3]);
    if (sg_size == 32u) {
      dp = subgroupAdd(dp);
    } else {
      dp += subgroupShuffleXor(dp, 16u);
      dp += subgroupShuffleXor(dp, 8u);
      dp += subgroupShuffleXor(dp, 4u);
      dp += subgroupShuffleXor(dp, 2u);
      dp += subgroupShuffleXor(dp, 1u);
    }

    let score = f32(dp) * (q_scale * ks);
    let m_new = max(m, score);
    let alpha = exp(m - m_new);
    let beta = exp(score - m_new);
    l = l * alpha + beta;

    let vo = v_base + t * 129u;
    let vs = bitcast<f32>(kv_global[vo + 128u]) * beta;
    acc0 = acc0 * alpha + unpack4_i8(kv_global[vo + w0]) * vs;
    acc1 = acc1 * alpha + unpack4_i8(kv_global[vo + w1]) * vs;
    acc2 = acc2 * alpha + unpack4_i8(kv_global[vo + w2]) * vs;
    acc3 = acc3 * alpha + unpack4_i8(kv_global[vo + w3]) * vs;
    m = m_new;
  }

  if (num_splits == 1u) {
    let inv_l = 1.0 / max(l, 1e-9);
    let o0 = acc0 * inv_l;
    let o1 = acc1 * inv_l;
    let o2 = acc2 * inv_l;
    let o3 = acc3 * inv_l;
    split_out[b0] = o0.x; split_out[b0 + 1u] = o0.y; split_out[b0 + 2u] = o0.z; split_out[b0 + 3u] = o0.w;
    split_out[b1] = o1.x; split_out[b1 + 1u] = o1.y; split_out[b1 + 2u] = o1.z; split_out[b1 + 3u] = o1.w;
    split_out[b2] = o2.x; split_out[b2 + 1u] = o2.y; split_out[b2 + 2u] = o2.z; split_out[b2 + 3u] = o2.w;
    split_out[b3] = o3.x; split_out[b3 + 1u] = o3.y; split_out[b3 + 2u] = o3.z; split_out[b3 + 3u] = o3.w;
    return;
  }

  let ob = (qh * num_splits + split_id) * 516u;
  let o0 = ob + (w0 << 2u);
  let o1 = ob + (w1 << 2u);
  let o2 = ob + (w2 << 2u);
  let o3 = ob + (w3 << 2u);
  split_out[o0] = acc0.x; split_out[o0 + 1u] = acc0.y; split_out[o0 + 2u] = acc0.z; split_out[o0 + 3u] = acc0.w;
  split_out[o1] = acc1.x; split_out[o1 + 1u] = acc1.y; split_out[o1 + 2u] = acc1.z; split_out[o1 + 3u] = acc1.w;
  split_out[o2] = acc2.x; split_out[o2 + 1u] = acc2.y; split_out[o2 + 2u] = acc2.z; split_out[o2 + 3u] = acc2.w;
  split_out[o3] = acc3.x; split_out[o3 + 1u] = acc3.y; split_out[o3 + 2u] = acc3.z; split_out[o3 + 3u] = acc3.w;
  if (lane == 0u) {
    split_out[ob + 512u] = m;
    split_out[ob + 513u] = l;
  }
}
"#;

// ============================================================================
// 143: On-GPU Q2 Hot PLE Dequant (32K hot tokens -> [T, 42, 256] f32)
// Matches g4_dequant_ple_rows bit-for-bit (PLE_SCALE_FACTOR = 16.0, 693 u32s/row)
// Dispatch: (42, T, 1) @workgroup_size(64)
// Bindings (lay5 compatible): 0 uniform, 1 token_ids, 2 hot_map, 3 ple_raw_out (rw), 4 hot_data
// ============================================================================
const WGSL_G4_PLE_HOT_DEQUANT: &str = r#"
struct G4PleHotParams {
  tokens: u32,
  num_layers: u32,
  ple_dim: u32,
  hot_count: u32,
};

@group(0) @binding(0) var<uniform> params: G4PleHotParams;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read> hot_map: array<u32>;
@group(0) @binding(3) var<storage, read_write> ple_raw_out: array<f32>;
@group(0) @binding(4) var<storage, read> hot_data: array<u32>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let layer = wid.x;
  let tok = wid.y;
  let b = lid.x; // 0..63 byte inside layer (4 weights per byte -> 256 dims)
  if (tok >= params.tokens || layer >= params.num_layers) {
    return;
  }

  let raw_id = token_ids[tok];
  let safe_id = select(0u, raw_id, raw_id < 262144u && raw_id != 258880u && raw_id != 258881u);
  let mapped = hot_map[safe_id];
  let slot = select(0u, mapped, mapped < params.hot_count);

  let row_base = slot * 693u;
  let sp = unpack2x16float(hot_data[row_base + 672u + (layer >> 1u)]);
  let scale = select(sp.x, sp.y, (layer & 1u) != 0u) * 16.0;

  let word = hot_data[row_base + layer * 16u + (b >> 2u)];
  let packed = (word >> ((b & 3u) * 8u)) & 0xFFu;

  let o = (tok * params.num_layers + layer) * params.ple_dim + b * 4u;
  ple_raw_out[o + 0u] = f32(i32(packed & 3u) - 2) * scale;
  ple_raw_out[o + 1u] = f32(i32((packed >> 2u) & 3u) - 2) * scale;
  ple_raw_out[o + 2u] = f32(i32((packed >> 4u) & 3u) - 2) * scale;
  ple_raw_out[o + 3u] = f32(i32((packed >> 6u) & 3u) - 2) * scale;
}
"#;

// ============================================================================
// 144: Hardware DP4A Zero-Repack W4A8 Factored Q4 GEMV (K = 2048..10240)
// Dispatch: (ceil(rows / 4), 1, 1) @workgroup_size(128)
// Uses vec4<u32> weight offset: q_vec4_offset (qOffset >> 4), s_vec4_offset (sOffset >> 4)
// ============================================================================
const WGSL_G4_Q4_GEMV_DP4A_W4A8: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_vec4_offset: u32,
  s_vec4_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> xe_sh: array<u32, 320>;
var<workgroup> xo_sh: array<u32, 320>;
var<workgroup> gx_sh: array<f32, 40>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<f32, 128>;

fn pack4_i8(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

fn read_scale_vec4(s_vec4_off: u32, row: u32) -> f32 {
  let sv = weights[s_vec4_off + (row >> 3u)];
  let wi = (row >> 1u) & 3u;
  let w = select(select(sv.x, sv.y, (wi & 1u) != 0u), select(sv.z, sv.w, (wi & 1u) != 0u), (wi & 2u) != 0u);
  let sp = unpack2x16float(w);
  return select(sp.x, sp.y, (row & 1u) != 0u);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let warp = tid >> 5u;
  let lane = tid & 31u;
  let row = wid.x * 4u + warp;
  let safe_row = min(row, max(params.rows, 1u) - 1u);

  let row_vec4s = params.cols >> 5u;
  let q_base = params.q_vec4_offset + safe_row * row_vec4s;
  let sg_step = max(sg_size, 1u);

  var acc: f32 = 0.0;
  var total_sum_x: f32 = 0.0;

  for (var k_tile: u32 = 0u; k_tile < row_vec4s; k_tile = k_tile + 80u) {
    let tile_v4 = min(80u, row_vec4s - k_tile);
    let tile_words = tile_v4 << 2u; // <= 320 (8 weights/word)
    let x_base = params.in_offset + (k_tile << 5u);

    var tile_sx_loc: f32 = 0.0;
    for (var base: u32 = 0u; base < tile_words; base = base + 128u) {
      let i = base + tid;
      var xv0 = vec4<f32>(0.0);
      var xv1 = vec4<f32>(0.0);
      var ma: f32 = 0.0;
      if (i < tile_words) {
        let kb = x_base + (i << 3u);
        xv0 = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
        xv1 = vec4<f32>(x[kb + 4u], x[kb + 5u], x[kb + 6u], x[kb + 7u]);
        tile_sx_loc += xv0.x + xv0.y + xv0.z + xv0.w + xv1.x + xv1.y + xv1.z + xv1.w;
        let a0 = max(max(abs(xv0.x), abs(xv0.y)), max(abs(xv0.z), abs(xv0.w)));
        let a1 = max(max(abs(xv1.x), abs(xv1.y)), max(abs(xv1.z), abs(xv1.w)));
        ma = max(a0, a1);
      }
      if (sg_size >= 8u) {
        ma = max(ma, subgroupShuffleXor(ma, 4u));
        ma = max(ma, subgroupShuffleXor(ma, 2u));
        ma = max(ma, subgroupShuffleXor(ma, 1u));
      } else {
        sumx_sh[tid] = ma;
        workgroupBarrier();
        var gm: f32 = 0.0;
        let gb = (tid >> 3u) * 8u;
        for (var l: u32 = 0u; l < 8u; l = l + 1u) { gm = max(gm, sumx_sh[gb + l]); }
        ma = gm;
        workgroupBarrier();
      }
      if (i < tile_words) {
        let gs = max(ma / 127.0, 1e-6);
        xe_sh[i] = pack4_i8(vec4<f32>(xv0.x, xv0.z, xv1.x, xv1.z) / gs);
        xo_sh[i] = pack4_i8(vec4<f32>(xv0.y, xv0.w, xv1.y, xv1.w) / gs);
        if ((i & 7u) == 0u) {
          gx_sh[i >> 3u] = gs;
        }
      }
    }
    let sg_sx = subgroupAdd(tile_sx_loc);
    sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
    workgroupBarrier();

    var tile_sx: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      tile_sx += sumx_sh[s];
    }
    total_sum_x += tile_sx;

    for (var w: u32 = lane; w < tile_v4; w = w + 32u) {
      let wv = weights[q_base + k_tile + w];
      let xb = w << 2u;
      let gs = gx_sh[w >> 1u];
      var dp: i32 = dot4I8Packed(wv.x & 0x0F0F0F0Fu, xe_sh[xb + 0u])
        + dot4I8Packed((wv.x >> 4u) & 0x0F0F0F0Fu, xo_sh[xb + 0u])
        + dot4I8Packed(wv.y & 0x0F0F0F0Fu, xe_sh[xb + 1u])
        + dot4I8Packed((wv.y >> 4u) & 0x0F0F0F0Fu, xo_sh[xb + 1u])
        + dot4I8Packed(wv.z & 0x0F0F0F0Fu, xe_sh[xb + 2u])
        + dot4I8Packed((wv.z >> 4u) & 0x0F0F0F0Fu, xo_sh[xb + 2u])
        + dot4I8Packed(wv.w & 0x0F0F0F0Fu, xe_sh[xb + 3u])
        + dot4I8Packed((wv.w >> 4u) & 0x0F0F0F0Fu, xo_sh[xb + 3u]);
      acc += f32(dp) * gs;
    }

    if (k_tile + 80u < row_vec4s) {
      workgroupBarrier();
    }
  }

  if (sg_size == 32u) {
    acc = subgroupAdd(acc);
  } else if (sg_size > 32u) {
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
  } else {
    red_sh[tid] = acc;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc: f32 = 0.0;
      let base = warp * 32u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) { s_acc += red_sh[base + l]; }
      acc = s_acc;
    }
  }

  if (lane == 0u && row < params.rows) {
    let scale = read_scale_vec4(params.s_vec4_offset, safe_row);
    let val = scale * (acc - 8.0 * total_sum_x);
    let out_i = params.out_offset + row;
    y[out_i] = select(val, y[out_i] + val, params.mode != 0u);
  }
}
"#;

// ============================================================================
// 145: Hardware DP4A Zero-Repack W2A8 Q2 LM Head (K = 2560, 16 rows/WG, 16 lanes/row)
// Dispatch: (ceil(rows / 16), T, 1) @workgroup_size(128) -> 10 balanced words/lane!
// ============================================================================
const WGSL_G4_Q2_LM_HEAD_DP4A_W2A8: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  extra_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x0_sh: array<u32, 160>;
var<workgroup> x1_sh: array<u32, 160>;
var<workgroup> x2_sh: array<u32, 160>;
var<workgroup> x3_sh: array<u32, 160>;
var<workgroup> gx_sh: array<f32, 40>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn pack4_i8(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let tok = wid.y;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, max_r);
  let r1 = min(rB, max_r);

  let u32_words = params.cols >> 4u; // 160 words for K=2560 (16 weights/word)
  let x_tok_base = params.in_offset + tok * params.cols;
  let sg_step = max(sg_size, 1u);

  var sx_loc: f32 = 0.0;
  for (var base: u32 = 0u; base < u32_words; base = base + 128u) {
    let i = base + tid;
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var ma: f32 = 0.0;
    if (i < u32_words) {
      let kb = x_tok_base + (i << 4u);
      a0 = vec4<f32>(x[kb + 0u], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
      a1 = vec4<f32>(x[kb + 4u], x[kb + 5u], x[kb + 6u], x[kb + 7u]);
      a2 = vec4<f32>(x[kb + 8u], x[kb + 9u], x[kb + 10u], x[kb + 11u]);
      a3 = vec4<f32>(x[kb + 12u], x[kb + 13u], x[kb + 14u], x[kb + 15u]);
      sx_loc += dot(vec4<f32>(1.0), a0 + a1 + a2 + a3);
      let m0 = max(max(abs(a0.x), abs(a0.y)), max(abs(a0.z), abs(a0.w)));
      let m1 = max(max(abs(a1.x), abs(a1.y)), max(abs(a1.z), abs(a1.w)));
      let m2 = max(max(abs(a2.x), abs(a2.y)), max(abs(a2.z), abs(a2.w)));
      let m3 = max(max(abs(a3.x), abs(a3.y)), max(abs(a3.z), abs(a3.w)));
      ma = max(max(m0, m1), max(m2, m3));
    }
    if (sg_size >= 4u) {
      ma = max(ma, subgroupShuffleXor(ma, 2u));
      ma = max(ma, subgroupShuffleXor(ma, 1u));
    }
    if (i < u32_words) {
      let gs = max(ma / 127.0, 1e-6);
      x0_sh[i] = pack4_i8(vec4<f32>(a0.x, a1.x, a2.x, a3.x) / gs);
      x1_sh[i] = pack4_i8(vec4<f32>(a0.y, a1.y, a2.y, a3.y) / gs);
      x2_sh[i] = pack4_i8(vec4<f32>(a0.z, a1.z, a2.z, a3.z) / gs);
      x3_sh[i] = pack4_i8(vec4<f32>(a0.w, a1.w, a2.w, a3.w) / gs);
      if ((i & 3u) == 0u) {
        gx_sh[i >> 2u] = gs;
      }
    }
  }

  let sg_sx = subgroupAdd(sx_loc);
  sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
  workgroupBarrier();

  var total_sum_x: f32 = 0.0;
  for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
    total_sum_x += sumx_sh[s];
  }

  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;

  for (var w: u32 = lane; w < u32_words; w = w + 16u) {
    let gs = gx_sh[w >> 2u];
    let q0_p = weights[o0 + w];
    let q1_p = weights[o1 + w];
    let xv0 = x0_sh[w]; let xv1 = x1_sh[w]; let xv2 = x2_sh[w]; let xv3 = x3_sh[w];
    let dp0 = dot4I8Packed(q0_p & 0x03030303u, xv0)
      + dot4I8Packed((q0_p >> 2u) & 0x03030303u, xv1)
      + dot4I8Packed((q0_p >> 4u) & 0x03030303u, xv2)
      + dot4I8Packed((q0_p >> 6u) & 0x03030303u, xv3);
    let dp1 = dot4I8Packed(q1_p & 0x03030303u, xv0)
      + dot4I8Packed((q1_p >> 2u) & 0x03030303u, xv1)
      + dot4I8Packed((q1_p >> 4u) & 0x03030303u, xv2)
      + dot4I8Packed((q1_p >> 6u) & 0x03030303u, xv3);
    acc0 += f32(dp0) * gs;
    acc1 += f32(dp1) * gs;
  }

  var sum2 = vec2<f32>(acc0, acc1);
  if (sg_size >= 16u) {
    sum2 += subgroupShuffleXor(sum2, 8u);
    sum2 += subgroupShuffleXor(sum2, 4u);
    sum2 += subgroupShuffleXor(sum2, 2u);
    sum2 += subgroupShuffleXor(sum2, 1u);
  } else {
    red_sh[tid] = sum2;
    workgroupBarrier();
    if (lane == 0u) {
      var sa = vec2<f32>(0.0);
      let gb = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) { sa += red_sh[gb + l]; }
      sum2 = sa;
    }
  }

  if (lane == 0u) {
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    var v0 = s0 * (sum2.x - 2.0 * total_sum_x);
    var v1 = s1 * (sum2.y - 2.0 * total_sum_x);
    if (params.mode != 0u) {
      let lo15 = vec2<f32>(-15.0);
      let hi15 = vec2<f32>(15.0);
      let capped = 30.0 * tanh(clamp(vec2<f32>(v0, v1) / 30.0, lo15, hi15));
      v0 = capped.x;
      v1 = capped.y;
    }
    let tok_out = params.out_offset + tok * params.rows;
    if (rA < params.rows) { y[tok_out + rA] = v0; }
    if (rB < params.rows) { y[tok_out + rB] = v1; }
  }
}
"#;

// ============================================================================
// 146: Zero-Barrier Warp Split-K Global Decode INT4 KV (D=512, 68 u32s/head)
// Dispatch: (2, num_splits, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_DECODE_GLOBAL_WARP_INT4: &str = r#"
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  in_seq_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_global: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

fn unpack_lo4(w: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(i32(w & 0xFu) - 8),
    f32(i32((w >> 4u) & 0xFu) - 8),
    f32(i32((w >> 8u) & 0xFu) - 8),
    f32(i32((w >> 12u) & 0xFu) - 8),
  );
}

fn unpack_hi4(w: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(i32((w >> 16u) & 0xFu) - 8),
    f32(i32((w >> 20u) & 0xFu) - 8),
    f32(i32((w >> 24u) & 0xFu) - 8),
    f32(i32((w >> 28u) & 0xFu) - 8),
  );
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits, 1u);
  let tid = lid.x;
  let warp = tid >> 5u; // 0..3 -> 4 Q heads
  let lane = tid & 31u; // 0..31 -> words lane (dims lane*8..+7) and 32+lane (dims 256+lane*8..+7)
  let qh = kv_head * 4u + warp;
  let qh_base = qh * 512u;

  let b0 = qh_base + lane * 8u;
  let b1 = qh_base + (32u + lane) * 8u;
  let q0a = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q0b = vec4<f32>(q_buf[b0 + 4u], q_buf[b0 + 5u], q_buf[b0 + 6u], q_buf[b0 + 7u]);
  let q1a = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q1b = vec4<f32>(q_buf[b1 + 4u], q_buf[b1 + 5u], q_buf[b1 + 6u], q_buf[b1 + 7u]);

  let g0 = lane >> 3u; // 0..3
  let head_u32s = params.max_seq * 68u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let total_toks = params.seq_pos + 1u;
  let chunk = (total_toks + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, total_toks);
  let t_end = min(t_start + chunk, total_toks);

  var m: f32 = -1e30;
  var l: f32 = 0.0;
  var acc0a = vec4<f32>(0.0);
  var acc0b = vec4<f32>(0.0);
  var acc1a = vec4<f32>(0.0);
  var acc1b = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let ko = k_base + t * 68u;
    let kw0 = kv_global[ko + lane];
    let kw1 = kv_global[ko + 32u + lane];
    let ks0p = unpack2x16float(kv_global[ko + 64u + (g0 >> 1u)]);
    let ks1p = unpack2x16float(kv_global[ko + 66u + (g0 >> 1u)]);
    let ks0 = select(ks0p.x, ks0p.y, (g0 & 1u) != 0u);
    let ks1 = select(ks1p.x, ks1p.y, (g0 & 1u) != 0u);

    var score = (dot(q0a, unpack_lo4(kw0)) + dot(q0b, unpack_hi4(kw0))) * ks0
      + (dot(q1a, unpack_lo4(kw1)) + dot(q1b, unpack_hi4(kw1))) * ks1;
    if (sg_size == 32u) {
      score = subgroupAdd(score);
    } else {
      score += subgroupShuffleXor(score, 16u);
      score += subgroupShuffleXor(score, 8u);
      score += subgroupShuffleXor(score, 4u);
      score += subgroupShuffleXor(score, 2u);
      score += subgroupShuffleXor(score, 1u);
    }

    let m_new = max(m, score);
    let alpha = exp(m - m_new);
    let beta = exp(score - m_new);
    l = l * alpha + beta;

    let vo = v_base + t * 68u;
    let vw0 = kv_global[vo + lane];
    let vw1 = kv_global[vo + 32u + lane];
    let vs0p = unpack2x16float(kv_global[vo + 64u + (g0 >> 1u)]);
    let vs1p = unpack2x16float(kv_global[vo + 66u + (g0 >> 1u)]);
    let vs0 = select(vs0p.x, vs0p.y, (g0 & 1u) != 0u) * beta;
    let vs1 = select(vs1p.x, vs1p.y, (g0 & 1u) != 0u) * beta;

    acc0a = acc0a * alpha + unpack_lo4(vw0) * vs0;
    acc0b = acc0b * alpha + unpack_hi4(vw0) * vs0;
    acc1a = acc1a * alpha + unpack_lo4(vw1) * vs1;
    acc1b = acc1b * alpha + unpack_hi4(vw1) * vs1;
    m = m_new;
  }

  if (num_splits == 1u) {
    let inv_l = 1.0 / max(l, 1e-9);
    let o0a = acc0a * inv_l; let o0b = acc0b * inv_l;
    let o1a = acc1a * inv_l; let o1b = acc1b * inv_l;
    split_out[b0] = o0a.x; split_out[b0 + 1u] = o0a.y; split_out[b0 + 2u] = o0a.z; split_out[b0 + 3u] = o0a.w;
    split_out[b0 + 4u] = o0b.x; split_out[b0 + 5u] = o0b.y; split_out[b0 + 6u] = o0b.z; split_out[b0 + 7u] = o0b.w;
    split_out[b1] = o1a.x; split_out[b1 + 1u] = o1a.y; split_out[b1 + 2u] = o1a.z; split_out[b1 + 3u] = o1a.w;
    split_out[b1 + 4u] = o1b.x; split_out[b1 + 5u] = o1b.y; split_out[b1 + 6u] = o1b.z; split_out[b1 + 7u] = o1b.w;
    return;
  }

  let ob = (qh * num_splits + split_id) * 516u;
  let o0 = ob + lane * 8u;
  let o1 = ob + (32u + lane) * 8u;
  split_out[o0] = acc0a.x; split_out[o0 + 1u] = acc0a.y; split_out[o0 + 2u] = acc0a.z; split_out[o0 + 3u] = acc0a.w;
  split_out[o0 + 4u] = acc0b.x; split_out[o0 + 5u] = acc0b.y; split_out[o0 + 6u] = acc0b.z; split_out[o0 + 7u] = acc0b.w;
  split_out[o1] = acc1a.x; split_out[o1 + 1u] = acc1a.y; split_out[o1 + 2u] = acc1a.z; split_out[o1 + 3u] = acc1a.w;
  split_out[o1 + 4u] = acc1b.x; split_out[o1 + 5u] = acc1b.y; split_out[o1 + 6u] = acc1b.z; split_out[o1 + 7u] = acc1b.w;
  if (lane == 0u) {
    split_out[ob + 512u] = m;
    split_out[ob + 513u] = l;
  }
}
"#;

// ============================================================================
// 147: FlashAttention-2 Q-Block Tiled Prefill Global INT4 KV (B_q=4 x 2 Q heads, B_kv=16, Split-K)
// Dispatch: (4, ceil(tokens / 4), num_splits) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_PREFILL_GLOBAL_FLASH_INT4: &str = r#"
enable f16;
enable subgroups;

struct G4GqaParams {
  local_donor_idx: u32,
  seq_pos: u32,
  head_dim: u32,
  max_seq: u32,
  kv_mode: u32,
  num_splits: u32,
  window_size: u32,
  tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GqaParams;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> kv_global: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;

var<workgroup> q_sh: array<vec4<f16>, 1024>;    // 8 Q vecs * 128 vec4<f16> = 8192 B
var<workgroup> kv_sh: array<u32, 1024>;         // 16 KV vecs * 64 u32s = 4096 B
var<workgroup> kv_scale_sh: array<u32, 64>;     // 16 KV vecs * 4 packed scale u32s
var<workgroup> qmax_sh: array<f32, 128>;
var<workgroup> s_sh: array<f32, 128>;
var<workgroup> m_sh: array<f32, 8>;
var<workgroup> l_sh: array<f32, 8>;
var<workgroup> alpha_sh: array<f32, 8>;

fn unpack_lo4_h(w: u32) -> vec4<f16> {
  return vec4<f16>(
    f16(i32(w & 0xFu) - 8),
    f16(i32((w >> 4u) & 0xFu) - 8),
    f16(i32((w >> 8u) & 0xFu) - 8),
    f16(i32((w >> 12u) & 0xFu) - 8),
  );
}

fn unpack_hi4_h(w: u32) -> vec4<f16> {
  return vec4<f16>(
    f16(i32((w >> 16u) & 0xFu) - 8),
    f16(i32((w >> 20u) & 0xFu) - 8),
    f16(i32((w >> 24u) & 0xFu) - 8),
    f16(i32((w >> 28u) & 0xFu) - 8),
  );
}

fn read_blk_scale(j: u32, g: u32) -> f32 {
  let sp = unpack2x16float(kv_scale_sh[j * 4u + (g >> 1u)]);
  return select(sp.x, sp.y, (g & 1u) != 0u);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let pair_id = wid.x;
  let kv_head = pair_id >> 1u;
  let qh_base = pair_id * 2u;
  let tok_tile = wid.y * 4u;
  let split_id = wid.z;
  let num_splits = max(params.num_splits, 1u);
  let tid = lid.x; // 0..127 -> d = tid*4..tid*4+3

  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    let tok = tok_tile + (r >> 1u);
    let qh = qh_base + (r & 1u);
    var qv = vec4<f16>(0.0h);
    if (tok < params.tokens) {
      let qb = (tok * 8u + qh) * 512u + tid * 4u;
      qv = vec4<f16>(f16(q_buf[qb]), f16(q_buf[qb + 1u]), f16(q_buf[qb + 2u]), f16(q_buf[qb + 3u]));
    }
    q_sh[r * 128u + tid] = qv;
  }
  if (tid < 8u) {
    m_sh[tid] = -1e30;
    l_sh[tid] = 0.0;
    alpha_sh[tid] = 1.0;
  }
  workgroupBarrier();

  let head_u32s = params.max_seq * 68u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let max_k_len = params.seq_pos + min(tok_tile + 4u, params.tokens);
  let chunk = (max_k_len + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, max_k_len);
  let t_end = min(t_start + chunk, max_k_len);

  var acc: array<vec4<f32>, 8>;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    acc[r] = vec4<f32>(0.0);
  }

  let team = tid >> 4u;
  let lane16 = tid & 15u;

  for (var t0: u32 = t_start; t0 < t_end; t0 = t0 + 16u) {
    let k_cnt = min(16u, t_end - t0);
    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let ko = k_base + (t0 + j) * 68u;
      if (tid < 64u) {
        kv_sh[j * 64u + tid] = kv_global[ko + tid];
      }
      if (tid < 4u) {
        kv_scale_sh[j * 4u + tid] = kv_global[ko + 64u + tid];
      }
    }
    workgroupBarrier();

    let qt_team = team >> 1u;
    let tok_team = tok_tile + qt_team;
    let pos_q_team = params.seq_pos + tok_team;
    let q_off = team * 128u;

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let k_off = j * 64u;
      var d_sum: f32 = 0.0;
      for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let w = lane16 * 4u + k; // 0..63
        let kw = kv_sh[k_off + w];
        let gs = read_blk_scale(j, w >> 3u);
        let ds = f32(dot(q_sh[q_off + w * 2u], unpack_lo4_h(kw)))
          + f32(dot(q_sh[q_off + w * 2u + 1u], unpack_hi4_h(kw)));
        d_sum += ds * gs;
      }
      if (sg_size >= 16u) {
        d_sum += subgroupShuffleXor(d_sum, 8u);
        d_sum += subgroupShuffleXor(d_sum, 4u);
        d_sum += subgroupShuffleXor(d_sum, 2u);
        d_sum += subgroupShuffleXor(d_sum, 1u);
      } else {
        qmax_sh[tid] = d_sum;
        workgroupBarrier();
        if (lane16 == 0u) {
          var s: f32 = 0.0;
          let gb = team * 16u;
          for (var l: u32 = 0u; l < 16u; l = l + 1u) { s += qmax_sh[gb + l]; }
          d_sum = s;
        }
        workgroupBarrier();
      }

      if (lane16 == 0u) {
        let key_pos = t0 + j;
        var sc: f32 = -1e30;
        if (tok_team < params.tokens && key_pos <= pos_q_team) {
          sc = d_sum;
        }
        s_sh[team * 16u + j] = sc;
      }
    }
    workgroupBarrier();

    if (tid < 8u) {
      let sb = tid * 16u;
      var m_block: f32 = -1e30;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        m_block = max(m_block, s_sh[sb + j]);
      }
      let m_old = m_sh[tid];
      let m_new = max(m_old, m_block);
      let alpha = exp(m_old - m_new);
      var l_block: f32 = 0.0;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        let w_exp = select(0.0, exp(s_sh[sb + j] - m_new), s_sh[sb + j] > -1e29);
        s_sh[sb + j] = w_exp;
        l_block += w_exp;
      }
      l_sh[tid] = l_sh[tid] * alpha + l_block;
      m_sh[tid] = m_new;
      alpha_sh[tid] = alpha;
    }

    for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
      let vo = v_base + (t0 + j) * 68u;
      if (tid < 64u) {
        kv_sh[j * 64u + tid] = kv_global[vo + tid];
      }
      if (tid < 4u) {
        kv_scale_sh[j * 4u + tid] = kv_global[vo + 64u + tid];
      }
    }
    workgroupBarrier();

    let vw_idx = tid >> 1u;
    let vw_hi = (tid & 1u) != 0u;
    let v_grp = tid >> 4u;
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
      var v_acc = vec4<f32>(0.0);
      let sb = r * 16u;
      for (var j: u32 = 0u; j < k_cnt; j = j + 1u) {
        let w_p = s_sh[sb + j];
        if (w_p > 0.0) {
          let vw = kv_sh[j * 64u + vw_idx];
          let v4h = select(unpack_lo4_h(vw), unpack_hi4_h(vw), vw_hi);
          let vs = read_blk_scale(j, v_grp) * w_p;
          v_acc += vec4<f32>(v4h) * vs;
        }
      }
      acc[r] = acc[r] * alpha_sh[r] + v_acc;
    }
    if (t0 + 16u < t_end) {
      workgroupBarrier();
    }
  }

  let d = tid * 4u;
  for (var r: u32 = 0u; r < 8u; r = r + 1u) {
    let qt = r >> 1u;
    let tok = tok_tile + qt;
    if (tok >= params.tokens) {
      continue;
    }
    let qh = qh_base + (r & 1u);
    if (num_splits == 1u) {
      let o4 = acc[r] / max(l_sh[r], 1e-9);
      let ob = (tok * 8u + qh) * 512u + d;
      out_buf[ob] = o4.x; out_buf[ob + 1u] = o4.y; out_buf[ob + 2u] = o4.z; out_buf[ob + 3u] = o4.w;
    } else {
      let ob = ((tok * 8u + qh) * num_splits + split_id) * 516u;
      let a4 = acc[r];
      out_buf[ob + d] = a4.x; out_buf[ob + d + 1u] = a4.y; out_buf[ob + d + 2u] = a4.z; out_buf[ob + d + 3u] = a4.w;
      if (tid == 0u) {
        out_buf[ob + 512u] = m_sh[r];
        out_buf[ob + 513u] = l_sh[r];
      }
    }
  }
}
"#;
