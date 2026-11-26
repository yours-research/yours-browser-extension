use crate::f16_to_f32;

const PLE_SCALE_FACTOR: f32 = 16.0;
const PLE_NUM_LAYERS: usize = 42;
const PLE_DIM: usize = 256;
const PLE_Q_BYTES: usize = 2688; // 42 * 64 bytes (256 weights/layer * 2 bits)
const PLE_S_BYTES: usize = 84; // 42 * 2 bytes (1 f16 scale/layer)
pub const PLE_PACKED_ROW_BYTES: usize = PLE_Q_BYTES + PLE_S_BYTES; // 2772 bytes/token

/// Unpacks `num_tokens` raw PLE rows (`[2688B q2, 84B f16 scales]` each = `2772B/tok`)
/// into contiguous `[num_tokens, 42, 256]` f32 values scaled by `16.0` (`sqrt(256)`).
#[no_mangle]
pub unsafe extern "C" fn g4_dequant_ple_rows(
    num_tokens: u32,
    raw_ptr: *const u8,
    out_f32_ptr: *mut f32,
) {
    let n = num_tokens as usize;
    let src = core::slice::from_raw_parts(raw_ptr, n * PLE_PACKED_ROW_BYTES);
    let dst = core::slice::from_raw_parts_mut(out_f32_ptr, n * PLE_NUM_LAYERS * PLE_DIM);

    for t in 0..n {
        let row_src = &src[t * PLE_PACKED_ROW_BYTES..(t + 1) * PLE_PACKED_ROW_BYTES];
        let q_bytes = &row_src[..PLE_Q_BYTES];
        let s_bytes = &row_src[PLE_Q_BYTES..];
        let tok_dst_base = t * PLE_NUM_LAYERS * PLE_DIM;

        for l in 0..PLE_NUM_LAYERS {
            let h_scale = u16::from_le_bytes([s_bytes[l * 2], s_bytes[l * 2 + 1]]);
            let scale = f16_to_f32(h_scale) * PLE_SCALE_FACTOR;
            let q_base = l * 64;
            let d_base = tok_dst_base + l * PLE_DIM;

            for b in 0..64 {
                let packed = q_bytes[q_base + b];
                let o = d_base + b * 4;
                dst[o] = (((packed & 0x03) as i32 - 2) as f32) * scale;
                dst[o + 1] = ((((packed >> 2) & 0x03) as i32 - 2) as f32) * scale;
                dst[o + 2] = ((((packed >> 4) & 0x03) as i32 - 2) as f32) * scale;
                dst[o + 3] = ((((packed >> 6) & 0x03) as i32 - 2) as f32) * scale;
            }
        }
    }
}

pub fn get_shader(shader_id: u32) -> Option<&'static [u8]> {
    match shader_id {
        100 => Some(WGSL_G4_Q4_GEMV_SUBGROUPS_F16.as_bytes()),
        101 => Some(WGSL_G4_Q4_GEMV_T3_F32.as_bytes()),
        102 => Some(WGSL_G4_Q4_GEMM_F16.as_bytes()),
        103 => Some(WGSL_G4_Q4_GEMM_T3_F32.as_bytes()),
        104 => Some(WGSL_G4_Q4_GATE_UP_GEGLU_SUBGROUPS_F16.as_bytes()),
        105 => Some(WGSL_G4_Q4_GATE_UP_GEGLU_T3_F32.as_bytes()),
        106 => Some(WGSL_G4_Q8_GEMV_BATCH.as_bytes()),
        107 => Some(WGSL_G4_Q2_LM_HEAD_SUBGROUPS_F16.as_bytes()),
        108 => Some(WGSL_G4_Q2_LM_HEAD_T3_F32.as_bytes()),
        109 => Some(WGSL_G4_Q2_EMBED_LOOKUP.as_bytes()),
        110 => Some(WGSL_G4_F16_PLE_PROJ.as_bytes()),
        111 => Some(WGSL_G4_PLE_NORM_COMBINE.as_bytes()),
        112 => Some(WGSL_G4_RMSNORM_RESIDUAL_BATCH.as_bytes()),
        113 => Some(WGSL_G4_GEGLU_BATCH.as_bytes()),
        114 => Some(WGSL_G4_QKV_NORM_ROPE_KVWRITE.as_bytes()),
        115 => Some(WGSL_G4_GQA_DECODE_SLIDING_FP16.as_bytes()),
        116 => Some(WGSL_G4_GQA_DECODE_GLOBAL_SPLIT.as_bytes()),
        117 => Some(WGSL_G4_GQA_DECODE_REDUCE.as_bytes()),
        118 => Some(WGSL_G4_GQA_PREFILL_SLIDING_FP16.as_bytes()),
        119 => Some(WGSL_G4_GQA_PREFILL_GLOBAL.as_bytes()),
        120 => Some(WGSL_G4_GQA_DECODE_SLIDING_SPLIT_SG.as_bytes()),
        121 => Some(WGSL_G4_GQA_DECODE_SLIDING_REDUCE.as_bytes()),
        122 => Some(WGSL_G4_GQA_DECODE_GLOBAL_SPLIT_SG.as_bytes()),
        123 => Some(WGSL_G4_GQA_PREFILL_SLIDING_SG.as_bytes()),
        124 => Some(WGSL_G4_GQA_PREFILL_GLOBAL_SG.as_bytes()),
        125 => Some(WGSL_G4_RMSNORM_RESIDUAL_BATCH_SG.as_bytes()),
        126 => Some(WGSL_G4_Q8_GEMV_BATCH_SG.as_bytes()),
        127 => Some(WGSL_G4_RMSNORM_FUSED_SG.as_bytes()),
        128 => Some(WGSL_G4_QKV_NORM_ROPE_KVWRITE_SG.as_bytes()),
        129 => Some(WGSL_G4_Q4_GEMV_BATCH_F16.as_bytes()),
        130 => Some(WGSL_G4_Q4_GATE_UP_GEGLU_BATCH_F16.as_bytes()),
        131 => Some(WGSL_G4_Q2_LM_HEAD_BATCH_F16.as_bytes()),
        132 => Some(WGSL_G4_ARGMAX_BATCH_PASS1.as_bytes()),
        133 => Some(WGSL_G4_ARGMAX_BATCH_PASS2.as_bytes()),
        _ => crate::gemma4_opt::get_shader(shader_id),
    }
}

// ============================================================================
// 100: Factored Pure-Q4 GEMV (Subgroups + F16, T=1, K up to 10240)
// Zero block scales in inner loop: w = (q - 8) * row_scale
// Dispatch: (ceil(rows / 16), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GEMV_SUBGROUPS_F16: &str = r#"
enable f16;
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

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q4_word_f16(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
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
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, max_r);
  let r1 = min(rB, max_r);

  let u32_words = params.cols >> 3u;
  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;

  var acc_qx0: f32 = 0.0;
  var acc_qx1: f32 = 0.0;
  var total_sum_x: f32 = 0.0;
  let ones = vec4<f16>(1.0h);
  let sg_step = max(sg_size, 1u);

  for (var k_tile: u32 = 0u; k_tile < u32_words; k_tile = k_tile + 320u) {
    let tile_u32s = min(320u, u32_words - k_tile);
    let tile_vec4s = tile_u32s << 1u;
    let x_base = params.in_offset + (k_tile << 3u);
    var tile_sx_loc: f32 = 0.0;
    for (var i: u32 = tid; i < tile_vec4s; i = i + 128u) {
      let kb = x_base + (i << 2u);
      let xv = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      x_sh[i] = xv;
      tile_sx_loc += f32(dot(ones, xv));
    }
    let sg_sx = subgroupAdd(tile_sx_loc);
    sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
    workgroupBarrier();

    var tile_sx: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      tile_sx += sumx_sh[s];
    }
    total_sum_x += tile_sx;

    for (var w_rel: u32 = lane; w_rel < tile_u32s; w_rel = w_rel + 16u) {
      let xb = w_rel << 1u;
      let xa = x_sh[xb];
      let xb_v = x_sh[xb + 1u];
      let gw = k_tile + w_rel;
      acc_qx0 += factored_q4_word_f16(weights[o0 + gw], xa, xb_v);
      acc_qx1 += factored_q4_word_f16(weights[o1 + gw], xa, xb_v);
    }

    if (k_tile + 320u < u32_words) {
      workgroupBarrier();
    }
  }

  var sum4 = vec4<f32>(acc_qx0, acc_qx1, 0.0, 0.0);
  if (sg_size >= 16u) {
    sum4 += subgroupShuffleXor(sum4, 8u);
    sum4 += subgroupShuffleXor(sum4, 4u);
    sum4 += subgroupShuffleXor(sum4, 2u);
    sum4 += subgroupShuffleXor(sum4, 1u);
  } else {
    red_sh[tid] = sum4;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc = vec4<f32>(0.0);
      let g_base = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        s_acc += red_sh[g_base + l];
      }
      sum4 = s_acc;
    }
  }

  if (lane == 0u) {
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    let v0 = s0 * (sum4.x - 8.0 * total_sum_x);
    let v1 = s1 * (sum4.y - 8.0 * total_sum_x);
    if (rA < params.rows) {
      let out0 = params.out_offset + rA;
      y[out0] = select(v0, y[out0] + v0, params.mode != 0u);
    }
    if (rB < params.rows) {
      let out1 = params.out_offset + rB;
      y[out1] = select(v1, y[out1] + v1, params.mode != 0u);
    }
  }
}
"#;

// ============================================================================
// 101: Factored Pure-Q4 GEMV (Universal F32 Fallback, T=1)
// Dispatch: (ceil(rows / 16), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GEMV_T3_F32: &str = r#"
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

var<workgroup> x_sh: array<vec4<f32>, 640>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q4_word_f32(p: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let q0 = vec4<f32>(f32(p & 0xFu), f32((p >> 4u) & 0xFu), f32((p >> 8u) & 0xFu), f32((p >> 12u) & 0xFu));
  let q1 = vec4<f32>(f32((p >> 16u) & 0xFu), f32((p >> 20u) & 0xFu), f32((p >> 24u) & 0xFu), f32((p >> 28u) & 0xFu));
  return dot(q0, xa) + dot(q1, xb);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, max_r);
  let r1 = min(rB, max_r);

  let u32_words = params.cols >> 3u;
  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;

  var acc_qx0: f32 = 0.0;
  var acc_qx1: f32 = 0.0;
  var acc_x: f32 = 0.0;
  let ones = vec4<f32>(1.0);

  for (var k_tile: u32 = 0u; k_tile < u32_words; k_tile = k_tile + 320u) {
    let tile_u32s = min(320u, u32_words - k_tile);
    let tile_vec4s = tile_u32s << 1u;
    let x_base = params.in_offset + (k_tile << 3u);
    for (var i: u32 = tid; i < tile_vec4s; i = i + 128u) {
      let kb = x_base + (i << 2u);
      x_sh[i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
    }
    workgroupBarrier();

    for (var w_rel: u32 = lane; w_rel < tile_u32s; w_rel = w_rel + 16u) {
      let xb = w_rel << 1u;
      let xa = x_sh[xb];
      let xb_v = x_sh[xb + 1u];
      acc_x += dot(ones, xa) + dot(ones, xb_v);
      let gw = k_tile + w_rel;
      acc_qx0 += factored_q4_word_f32(weights[o0 + gw], xa, xb_v);
      acc_qx1 += factored_q4_word_f32(weights[o1 + gw], xa, xb_v);
    }

    if (k_tile + 320u < u32_words) {
      workgroupBarrier();
    }
  }

  red_sh[tid] = vec4<f32>(acc_qx0, acc_qx1, acc_x, 0.0);
  workgroupBarrier();
  for (var s: u32 = 8u; s > 0u; s = s >> 1u) {
    if (lane < s) {
      red_sh[tid] += red_sh[tid + s];
    }
    workgroupBarrier();
  }

  if (lane == 0u) {
    let sum4 = red_sh[tid];
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    let v0 = s0 * (sum4.x - 8.0 * sum4.z);
    let v1 = s1 * (sum4.y - 8.0 * sum4.z);
    if (rA < params.rows) {
      let out0 = params.out_offset + rA;
      y[out0] = select(v0, y[out0] + v0, params.mode != 0u);
    }
    if (rB < params.rows) {
      let out1 = params.out_offset + rB;
      y[out1] = select(v1, y[out1] + v1, params.mode != 0u);
    }
  }
}
"#;

// ============================================================================
// 102: Tiled Batched Pure-Q4 GEMM (F16 Shared Mem, M>=1 Prefill / Spec Verify)
// Stores centered exact integers (q - 8) in f16 w_sh -> multiplies row_scale once at store!
// Dispatch: (ceil(N / 64), ceil(M / 64), 1) @workgroup_size(16, 16)
// ============================================================================
const WGSL_G4_Q4_GEMM_F16: &str = r#"
enable f16;

struct G4GemmParams {
  M: u32,
  N: u32,
  K: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemmParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 512>;
var<workgroup> w_sh: array<vec4<f16>, 512>;

fn unpack_centered_q4_f16(word: u32, shift_base: u32) -> vec4<f16> {
  return vec4<f16>(
    f16((word >> shift_base) & 0xFu) - 8.0h,
    f16((word >> (shift_base + 4u)) & 0xFu) - 8.0h,
    f16((word >> (shift_base + 8u)) & 0xFu) - 8.0h,
    f16((word >> (shift_base + 12u)) & 0xFu) - 8.0h,
  );
}

fn store_scaled_cell(tok: u32, row: u32, raw_acc: f32) {
  if (tok < params.M && row < params.N) {
    let sp = unpack2x16float(weights[params.s_u32_offset + (row >> 1u)]);
    let row_scale = select(sp.x, sp.y, (row & 1u) != 0u);
    let val = raw_acc * row_scale;
    let idx = params.out_offset + tok * params.N + row;
    if (params.add_residual != 0u) {
      y[idx] = y[idx] + val;
    } else {
      y[idx] = val;
    }
  }
}

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row_tile = wid.x * 64u;
  let tok_tile = wid.y * 64u;
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * 16u + lx;
  let u32s_per_row = params.K >> 3u;
  let blocks = params.K >> 5u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let loc_idx = tid >> 2u;
  let qtr = tid & 3u;
  let k4_0 = qtr * 2u;
  let k4_1 = k4_0 + 1u;

  for (var b: u32 = 0u; b < blocks; b = b + 1u) {
    let gt = tok_tile + loc_idx;
    if (gt < params.M) {
      let xb = params.in_offset + gt * params.K + (b << 5u) + (qtr << 3u);
      x_sh[k4_0 * 64u + loc_idx] = vec4<f16>(f16(x[xb + 0u]), f16(x[xb + 1u]), f16(x[xb + 2u]), f16(x[xb + 3u]));
      x_sh[k4_1 * 64u + loc_idx] = vec4<f16>(f16(x[xb + 4u]), f16(x[xb + 5u]), f16(x[xb + 6u]), f16(x[xb + 7u]));
    } else {
      x_sh[k4_0 * 64u + loc_idx] = vec4<f16>(0.0h);
      x_sh[k4_1 * 64u + loc_idx] = vec4<f16>(0.0h);
    }

    let gr = row_tile + loc_idx;
    if (gr < params.N) {
      let pw = weights[params.q_u32_offset + gr * u32s_per_row + (b << 2u) + qtr];
      w_sh[k4_0 * 64u + loc_idx] = unpack_centered_q4_f16(pw, 0u);
      w_sh[k4_1 * 64u + loc_idx] = unpack_centered_q4_f16(pw, 16u);
    } else {
      w_sh[k4_0 * 64u + loc_idx] = vec4<f16>(0.0h);
      w_sh[k4_1 * 64u + loc_idx] = vec4<f16>(0.0h);
    }
    workgroupBarrier();

    for (var k4: u32 = 0u; k4 < 8u; k4 = k4 + 1u) {
      let kb = k4 * 64u;
      let wv0 = w_sh[kb + ly + 0u];
      let wv1 = w_sh[kb + ly + 16u];
      let wv2 = w_sh[kb + ly + 32u];
      let wv3 = w_sh[kb + ly + 48u];
      let xv0 = x_sh[kb + lx + 0u];
      let xv1 = x_sh[kb + lx + 16u];
      let xv2 = x_sh[kb + lx + 32u];
      let xv3 = x_sh[kb + lx + 48u];
      acc0 += vec4<f32>(f32(dot(wv0, xv0)), f32(dot(wv0, xv1)), f32(dot(wv0, xv2)), f32(dot(wv0, xv3)));
      acc1 += vec4<f32>(f32(dot(wv1, xv0)), f32(dot(wv1, xv1)), f32(dot(wv1, xv2)), f32(dot(wv1, xv3)));
      acc2 += vec4<f32>(f32(dot(wv2, xv0)), f32(dot(wv2, xv1)), f32(dot(wv2, xv2)), f32(dot(wv2, xv3)));
      acc3 += vec4<f32>(f32(dot(wv3, xv0)), f32(dot(wv3, xv1)), f32(dot(wv3, xv2)), f32(dot(wv3, xv3)));
    }
    workgroupBarrier();
  }

  let out_t = tok_tile + lx;
  let out_r = row_tile + ly;
  store_scaled_cell(out_t + 0u, out_r + 0u, acc0.x);
  store_scaled_cell(out_t + 16u, out_r + 0u, acc0.y);
  store_scaled_cell(out_t + 32u, out_r + 0u, acc0.z);
  store_scaled_cell(out_t + 48u, out_r + 0u, acc0.w);

  store_scaled_cell(out_t + 0u, out_r + 16u, acc1.x);
  store_scaled_cell(out_t + 16u, out_r + 16u, acc1.y);
  store_scaled_cell(out_t + 32u, out_r + 16u, acc1.z);
  store_scaled_cell(out_t + 48u, out_r + 16u, acc1.w);

  store_scaled_cell(out_t + 0u, out_r + 32u, acc2.x);
  store_scaled_cell(out_t + 16u, out_r + 32u, acc2.y);
  store_scaled_cell(out_t + 32u, out_r + 32u, acc2.z);
  store_scaled_cell(out_t + 48u, out_r + 32u, acc2.w);

  store_scaled_cell(out_t + 0u, out_r + 48u, acc3.x);
  store_scaled_cell(out_t + 16u, out_r + 48u, acc3.y);
  store_scaled_cell(out_t + 32u, out_r + 48u, acc3.z);
  store_scaled_cell(out_t + 48u, out_r + 48u, acc3.w);
}
"#;

// ============================================================================
// 103: Tiled Batched Pure-Q4 GEMM (Universal F32 Fallback, K=16 tiles)
// Dispatch: (ceil(N / 64), ceil(M / 64), 1) @workgroup_size(16, 16)
// ============================================================================
const WGSL_G4_Q4_GEMM_T3_F32: &str = r#"
struct G4GemmParams {
  M: u32,
  N: u32,
  K: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemmParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f32>, 256>;
var<workgroup> w_sh: array<vec4<f32>, 256>;

fn unpack_centered_q4_f32(word: u32, shift_base: u32) -> vec4<f32> {
  return vec4<f32>(
    f32((word >> shift_base) & 0xFu) - 8.0,
    f32((word >> (shift_base + 4u)) & 0xFu) - 8.0,
    f32((word >> (shift_base + 8u)) & 0xFu) - 8.0,
    f32((word >> (shift_base + 12u)) & 0xFu) - 8.0,
  );
}

fn store_scaled_cell(tok: u32, row: u32, raw_acc: f32) {
  if (tok < params.M && row < params.N) {
    let sp = unpack2x16float(weights[params.s_u32_offset + (row >> 1u)]);
    let row_scale = select(sp.x, sp.y, (row & 1u) != 0u);
    let val = raw_acc * row_scale;
    let idx = params.out_offset + tok * params.N + row;
    if (params.add_residual != 0u) {
      y[idx] = y[idx] + val;
    } else {
      y[idx] = val;
    }
  }
}

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row_tile = wid.x * 64u;
  let tok_tile = wid.y * 64u;
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * 16u + lx;
  let u32s_per_row = params.K >> 3u;
  let k_tiles = params.K >> 4u; // 16 weights = 2 u32s per tile

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let loc_idx = tid >> 2u; // 0..63
  let k4 = tid & 3u;       // 0..3 (4 weights)
  let u_off = k4 >> 1u;
  let n_shift = (k4 & 1u) * 16u;

  for (var t: u32 = 0u; t < k_tiles; t = t + 1u) {
    let gt = tok_tile + loc_idx;
    if (gt < params.M) {
      let xb = params.in_offset + gt * params.K + (t << 4u) + (k4 << 2u);
      x_sh[k4 * 64u + loc_idx] = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
    } else {
      x_sh[k4 * 64u + loc_idx] = vec4<f32>(0.0);
    }

    let gr = row_tile + loc_idx;
    if (gr < params.N) {
      let pw = weights[params.q_u32_offset + gr * u32s_per_row + (t << 1u) + u_off];
      w_sh[k4 * 64u + loc_idx] = unpack_centered_q4_f32(pw, n_shift);
    } else {
      w_sh[k4 * 64u + loc_idx] = vec4<f32>(0.0);
    }
    workgroupBarrier();

    for (var s: u32 = 0u; s < 4u; s = s + 1u) {
      let kb = s * 64u;
      let wv0 = w_sh[kb + ly + 0u];
      let wv1 = w_sh[kb + ly + 16u];
      let wv2 = w_sh[kb + ly + 32u];
      let wv3 = w_sh[kb + ly + 48u];
      let xv0 = x_sh[kb + lx + 0u];
      let xv1 = x_sh[kb + lx + 16u];
      let xv2 = x_sh[kb + lx + 32u];
      let xv3 = x_sh[kb + lx + 48u];
      acc0 += vec4<f32>(dot(wv0, xv0), dot(wv0, xv1), dot(wv0, xv2), dot(wv0, xv3));
      acc1 += vec4<f32>(dot(wv1, xv0), dot(wv1, xv1), dot(wv1, xv2), dot(wv1, xv3));
      acc2 += vec4<f32>(dot(wv2, xv0), dot(wv2, xv1), dot(wv2, xv2), dot(wv2, xv3));
      acc3 += vec4<f32>(dot(wv3, xv0), dot(wv3, xv1), dot(wv3, xv2), dot(wv3, xv3));
    }
    workgroupBarrier();
  }

  let out_t = tok_tile + lx;
  let out_r = row_tile + ly;
  store_scaled_cell(out_t + 0u, out_r + 0u, acc0.x);
  store_scaled_cell(out_t + 16u, out_r + 0u, acc0.y);
  store_scaled_cell(out_t + 32u, out_r + 0u, acc0.z);
  store_scaled_cell(out_t + 48u, out_r + 0u, acc0.w);

  store_scaled_cell(out_t + 0u, out_r + 16u, acc1.x);
  store_scaled_cell(out_t + 16u, out_r + 16u, acc1.y);
  store_scaled_cell(out_t + 32u, out_r + 16u, acc1.z);
  store_scaled_cell(out_t + 48u, out_r + 16u, acc1.w);

  store_scaled_cell(out_t + 0u, out_r + 32u, acc2.x);
  store_scaled_cell(out_t + 16u, out_r + 32u, acc2.y);
  store_scaled_cell(out_t + 32u, out_r + 32u, acc2.z);
  store_scaled_cell(out_t + 48u, out_r + 32u, acc2.w);

  store_scaled_cell(out_t + 0u, out_r + 48u, acc3.x);
  store_scaled_cell(out_t + 16u, out_r + 48u, acc3.y);
  store_scaled_cell(out_t + 32u, out_r + 48u, acc3.z);
  store_scaled_cell(out_t + 48u, out_r + 48u, acc3.w);
}
"#;

// ============================================================================
// 104: Fused Q4 gate_up_proj + GeGLU (gelu_tanh(gate) * up) for T=1 (F16 + Subgroups)
// params.rows = 10240 (intermediate size); gate = row r, up = row (r + 10240)
// Dispatch: (ceil(rows / 8), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GATE_UP_GEGLU_SUBGROUPS_F16: &str = r#"
enable f16;
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

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q4_word_f16(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
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
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 8u + grp;
  let rg = min(rA, max_r);
  let ru = rg + params.rows;

  let u32_words = params.cols >> 3u;
  let og = params.q_u32_offset + rg * u32_words;
  let ou = params.q_u32_offset + ru * u32_words;

  var acc_g: f32 = 0.0;
  var acc_u: f32 = 0.0;
  var total_sum_x: f32 = 0.0;
  let ones = vec4<f16>(1.0h);
  let sg_step = max(sg_size, 1u);

  for (var k_tile: u32 = 0u; k_tile < u32_words; k_tile = k_tile + 320u) {
    let tile_u32s = min(320u, u32_words - k_tile);
    let tile_vec4s = tile_u32s << 1u;
    let x_base = params.in_offset + (k_tile << 3u);
    var tile_sx_loc: f32 = 0.0;
    for (var i: u32 = tid; i < tile_vec4s; i = i + 128u) {
      let kb = x_base + (i << 2u);
      let xv = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      x_sh[i] = xv;
      tile_sx_loc += f32(dot(ones, xv));
    }
    let sg_sx = subgroupAdd(tile_sx_loc);
    sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
    workgroupBarrier();

    var tile_sx: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      tile_sx += sumx_sh[s];
    }
    total_sum_x += tile_sx;

    for (var w_rel: u32 = lane; w_rel < tile_u32s; w_rel = w_rel + 16u) {
      let xb = w_rel << 1u;
      let xa = x_sh[xb];
      let xb_v = x_sh[xb + 1u];
      let gw = k_tile + w_rel;
      acc_g += factored_q4_word_f16(weights[og + gw], xa, xb_v);
      acc_u += factored_q4_word_f16(weights[ou + gw], xa, xb_v);
    }

    if (k_tile + 320u < u32_words) {
      workgroupBarrier();
    }
  }

  var sum4 = vec4<f32>(acc_g, acc_u, 0.0, 0.0);
  if (sg_size >= 16u) {
    sum4 += subgroupShuffleXor(sum4, 8u);
    sum4 += subgroupShuffleXor(sum4, 4u);
    sum4 += subgroupShuffleXor(sum4, 2u);
    sum4 += subgroupShuffleXor(sum4, 1u);
  } else {
    red_sh[tid] = sum4;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc = vec4<f32>(0.0);
      let g_base = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        s_acc += red_sh[g_base + l];
      }
      sum4 = s_acc;
    }
  }

  if (lane == 0u && rA < params.rows) {
    let spg = unpack2x16float(weights[params.s_u32_offset + (rg >> 1u)]);
    let sg = select(spg.x, spg.y, (rg & 1u) != 0u);
    let spu = unpack2x16float(weights[params.s_u32_offset + (ru >> 1u)]);
    let su = select(spu.x, spu.y, (ru & 1u) != 0u);
    let g_val = sg * (sum4.x - 8.0 * total_sum_x);
    let u_val = su * (sum4.y - 8.0 * total_sum_x);
    y[params.out_offset + rA] = gelu_tanh(g_val) * u_val;
  }
}
"#;

// ============================================================================
// 105: Fused Q4 gate_up_proj + GeGLU for T=1 (Universal F32 Fallback)
// Dispatch: (ceil(rows / 8), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GATE_UP_GEGLU_T3_F32: &str = r#"
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

var<workgroup> x_sh: array<vec4<f32>, 640>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q4_word_f32(p: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let q0 = vec4<f32>(f32(p & 0xFu), f32((p >> 4u) & 0xFu), f32((p >> 8u) & 0xFu), f32((p >> 12u) & 0xFu));
  let q1 = vec4<f32>(f32((p >> 16u) & 0xFu), f32((p >> 20u) & 0xFu), f32((p >> 24u) & 0xFu), f32((p >> 28u) & 0xFu));
  return dot(q0, xa) + dot(q1, xb);
}

fn gelu_tanh(v: f32) -> f32 {
  let arg = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(arg));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 8u + grp;
  let rg = min(rA, max_r);
  let ru = rg + params.rows;

  let u32_words = params.cols >> 3u;
  let og = params.q_u32_offset + rg * u32_words;
  let ou = params.q_u32_offset + ru * u32_words;

  var acc_g: f32 = 0.0;
  var acc_u: f32 = 0.0;
  var acc_x: f32 = 0.0;
  let ones = vec4<f32>(1.0);

  for (var k_tile: u32 = 0u; k_tile < u32_words; k_tile = k_tile + 320u) {
    let tile_u32s = min(320u, u32_words - k_tile);
    let tile_vec4s = tile_u32s << 1u;
    let x_base = params.in_offset + (k_tile << 3u);
    for (var i: u32 = tid; i < tile_vec4s; i = i + 128u) {
      let kb = x_base + (i << 2u);
      x_sh[i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
    }
    workgroupBarrier();

    for (var w_rel: u32 = lane; w_rel < tile_u32s; w_rel = w_rel + 16u) {
      let xb = w_rel << 1u;
      let xa = x_sh[xb];
      let xb_v = x_sh[xb + 1u];
      acc_x += dot(ones, xa) + dot(ones, xb_v);
      let gw = k_tile + w_rel;
      acc_g += factored_q4_word_f32(weights[og + gw], xa, xb_v);
      acc_u += factored_q4_word_f32(weights[ou + gw], xa, xb_v);
    }

    if (k_tile + 320u < u32_words) {
      workgroupBarrier();
    }
  }

  red_sh[tid] = vec4<f32>(acc_g, acc_u, acc_x, 0.0);
  workgroupBarrier();
  for (var s: u32 = 8u; s > 0u; s = s >> 1u) {
    if (lane < s) {
      red_sh[tid] += red_sh[tid + s];
    }
    workgroupBarrier();
  }

  if (lane == 0u && rA < params.rows) {
    let sum4 = red_sh[tid];
    let spg = unpack2x16float(weights[params.s_u32_offset + (rg >> 1u)]);
    let sg = select(spg.x, spg.y, (rg & 1u) != 0u);
    let spu = unpack2x16float(weights[params.s_u32_offset + (ru >> 1u)]);
    let su = select(spu.x, spu.y, (ru & 1u) != 0u);
    let g_val = sg * (sum4.x - 8.0 * sum4.z);
    let u_val = su * (sum4.y - 8.0 * sum4.z);
    y[params.out_offset + rA] = gelu_tanh(g_val) * u_val;
  }
}
"#;

// ============================================================================
// 106: Batched Pure-Q8 GEMV for PLE Gate (2560->256) & PLE Proj (256->2560)
// w = (u8 - 128) * row_scale; mode==1 fuses gelu_tanh(val) * extra[tok * rows + row]
// Dispatch: (ceil(rows / 8), T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q8_GEMV_BATCH: &str = r#"
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

var<workgroup> x_sh: array<vec4<f32>, 640>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn gelu_tanh(v: f32) -> f32 {
  let arg = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(arg));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let row = wid.x * 8u + grp;
  let tok = wid.y;
  let safe_row = min(row, max(params.rows, 1u) - 1u);

  let u32_words = params.cols >> 2u; // cols up to 2560 -> 640 u32s
  let vec4_count = u32_words;
  let x_tok_base = params.in_offset + tok * params.cols;
  for (var i: u32 = tid; i < vec4_count; i = i + 128u) {
    let kb = x_tok_base + (i << 2u);
    x_sh[i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
  }
  workgroupBarrier();

  let w_base = params.q_u32_offset + safe_row * u32_words;
  var acc_qx: f32 = 0.0;
  var acc_x: f32 = 0.0;
  let ones = vec4<f32>(1.0);

  for (var w_rel: u32 = lane; w_rel < u32_words; w_rel = w_rel + 16u) {
    let p = weights[w_base + w_rel];
    let q = vec4<f32>(
      f32(p & 0xFFu),
      f32((p >> 8u) & 0xFFu),
      f32((p >> 16u) & 0xFFu),
      f32((p >> 24u) & 0xFFu),
    );
    let xv = x_sh[w_rel];
    acc_qx += dot(q, xv);
    acc_x += dot(ones, xv);
  }

  red_sh[tid] = vec2<f32>(acc_qx, acc_x);
  workgroupBarrier();
  for (var s: u32 = 8u; s > 0u; s = s >> 1u) {
    if (lane < s) {
      red_sh[tid] += red_sh[tid + s];
    }
    workgroupBarrier();
  }

  if (lane == 0u && row < params.rows) {
    let sum2 = red_sh[tid];
    let sp = unpack2x16float(weights[params.s_u32_offset + (safe_row >> 1u)]);
    let row_scale = select(sp.x, sp.y, (safe_row & 1u) != 0u);
    var val: f32 = row_scale * (sum2.x - 128.0 * sum2.y);
    if (params.mode != 0u) {
      val = gelu_tanh(val) * extra[params.extra_offset + tok * 10752u + row];
    }
    y[params.out_offset + tok * params.rows + row] = val;
  }
}
"#;

// ============================================================================
// 107: Batched Pure-Q2 LM Head GEMV + Softcap(30.0) (F16 + Subgroups)
// w = (q2 - 2) * row_scale; 16 weights per u32 (160 u32s per K=2560 row)
// Dispatch: (ceil(rows / 16), T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q2_LM_HEAD_SUBGROUPS_F16: &str = r#"
enable f16;
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

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q2_word_f16(p: u32, x0: vec4<f16>, x1: vec4<f16>, x2: vec4<f16>, x3: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 3u), f16((p >> 2u) & 3u), f16((p >> 4u) & 3u), f16((p >> 6u) & 3u));
  let q1 = vec4<f16>(f16((p >> 8u) & 3u), f16((p >> 10u) & 3u), f16((p >> 12u) & 3u), f16((p >> 14u) & 3u));
  let q2 = vec4<f16>(f16((p >> 16u) & 3u), f16((p >> 18u) & 3u), f16((p >> 20u) & 3u), f16((p >> 22u) & 3u));
  let q3 = vec4<f16>(f16((p >> 24u) & 3u), f16((p >> 26u) & 3u), f16((p >> 28u) & 3u), f16((p >> 30u) & 3u));
  return f32(dot(q0, x0)) + f32(dot(q1, x1)) + f32(dot(q2, x2)) + f32(dot(q3, x3));
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

  let u32_words = params.cols >> 4u; // cols=2560 -> 160 u32s
  let vec4_count = params.cols >> 2u; // 640 vec4<f16>
  let x_tok_base = params.in_offset + tok * params.cols;
  let ones = vec4<f16>(1.0h);
  var sx_loc: f32 = 0.0;
  for (var i: u32 = tid; i < vec4_count; i = i + 128u) {
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

  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;
  var acc_qx0: f32 = 0.0;
  var acc_qx1: f32 = 0.0;

  for (var w: u32 = lane; w < u32_words; w = w + 16u) {
    let xb = w << 2u;
    let x0 = x_sh[xb + 0u];
    let x1 = x_sh[xb + 1u];
    let x2 = x_sh[xb + 2u];
    let x3 = x_sh[xb + 3u];
    acc_qx0 += factored_q2_word_f16(weights[o0 + w], x0, x1, x2, x3);
    acc_qx1 += factored_q2_word_f16(weights[o1 + w], x0, x1, x2, x3);
  }

  var sum4 = vec4<f32>(acc_qx0, acc_qx1, 0.0, 0.0);
  if (sg_size >= 16u) {
    sum4 += subgroupShuffleXor(sum4, 8u);
    sum4 += subgroupShuffleXor(sum4, 4u);
    sum4 += subgroupShuffleXor(sum4, 2u);
    sum4 += subgroupShuffleXor(sum4, 1u);
  } else {
    red_sh[tid] = sum4;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc = vec4<f32>(0.0);
      let g_base = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        s_acc += red_sh[g_base + l];
      }
      sum4 = s_acc;
    }
  }

  if (lane == 0u) {
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    var v0 = s0 * (sum4.x - 2.0 * total_sum_x);
    var v1 = s1 * (sum4.y - 2.0 * total_sum_x);
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
// 108: Batched Pure-Q2 LM Head GEMV + Softcap(30.0) (Universal F32 Fallback)
// Dispatch: (ceil(rows / 16), T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q2_LM_HEAD_T3_F32: &str = r#"
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

var<workgroup> x_sh: array<vec4<f32>, 640>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q2_word_f32(p: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>) -> f32 {
  let q0 = vec4<f32>(f32(p & 3u), f32((p >> 2u) & 3u), f32((p >> 4u) & 3u), f32((p >> 6u) & 3u));
  let q1 = vec4<f32>(f32((p >> 8u) & 3u), f32((p >> 10u) & 3u), f32((p >> 12u) & 3u), f32((p >> 14u) & 3u));
  let q2 = vec4<f32>(f32((p >> 16u) & 3u), f32((p >> 18u) & 3u), f32((p >> 20u) & 3u), f32((p >> 22u) & 3u));
  let q3 = vec4<f32>(f32((p >> 24u) & 3u), f32((p >> 26u) & 3u), f32((p >> 28u) & 3u), f32((p >> 30u) & 3u));
  return dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
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

  let u32_words = params.cols >> 4u;
  let vec4_count = params.cols >> 2u;
  let x_tok_base = params.in_offset + tok * params.cols;
  for (var i: u32 = tid; i < vec4_count; i = i + 128u) {
    let kb = x_tok_base + (i << 2u);
    x_sh[i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
  }
  workgroupBarrier();

  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;
  var acc_qx0: f32 = 0.0;
  var acc_qx1: f32 = 0.0;
  var acc_x: f32 = 0.0;
  let ones = vec4<f32>(1.0);

  for (var w: u32 = lane; w < u32_words; w = w + 16u) {
    let xb = w << 2u;
    let x0 = x_sh[xb + 0u];
    let x1 = x_sh[xb + 1u];
    let x2 = x_sh[xb + 2u];
    let x3 = x_sh[xb + 3u];
    acc_x += dot(ones, x0) + dot(ones, x1) + dot(ones, x2) + dot(ones, x3);
    acc_qx0 += factored_q2_word_f32(weights[o0 + w], x0, x1, x2, x3);
    acc_qx1 += factored_q2_word_f32(weights[o1 + w], x0, x1, x2, x3);
  }

  red_sh[tid] = vec4<f32>(acc_qx0, acc_qx1, acc_x, 0.0);
  workgroupBarrier();
  for (var s: u32 = 8u; s > 0u; s = s >> 1u) {
    if (lane < s) {
      red_sh[tid] += red_sh[tid + s];
    }
    workgroupBarrier();
  }

  if (lane == 0u) {
    let sum4 = red_sh[tid];
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    var v0 = s0 * (sum4.x - 2.0 * sum4.z);
    var v1 = s1 * (sum4.y - 2.0 * sum4.z);
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
// 109: Tied Q2 Token Embedding Lookup from slab2 (lm_head)
// Multiplies by 50.59644256269407 (sqrt(2560)) — Zero extra embedding VRAM!
// Dispatch: (T, 1, 1) @workgroup_size(160) -> 1 u32 word (16 dims) per thread
// ============================================================================
const WGSL_G4_Q2_EMBED_LOOKUP: &str = r#"
struct G4EmbedParams {
  tokens: u32,
  hidden: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4EmbedParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> token_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_hidden: array<f32>;

@compute @workgroup_size(160)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tok = wid.x;
  if (tok >= params.tokens) {
    return;
  }
  let raw_id = token_ids[tok];
  let safe_id = select(raw_id, 0u, raw_id >= 262144u);
  let sp = unpack2x16float(weights[params.s_u32_offset + (safe_id >> 1u)]);
  let row_scale = select(sp.x, sp.y, (safe_id & 1u) != 0u) * 50.59644256269407;

  let w_idx = lid.x;
  let u32_words = params.hidden >> 4u; // 160
  if (w_idx >= u32_words) {
    return;
  }
  let packed = weights[params.q_u32_offset + safe_id * u32_words + w_idx];
  let out_base = tok * params.hidden + (w_idx << 4u);
  for (var j: u32 = 0u; j < 16u; j = j + 1u) {
    let q = f32((packed >> (j * 2u)) & 3u) - 2.0;
    out_hidden[out_base + j] = q * row_scale;
  }
}
"#;

// ============================================================================
// 110: Coalesced Vectorized FP16 Per-Layer Projection ([T, 2560] x [2560, 10752])
// Each thread computes 2 adjacent columns from a single u32 fp16-pair read!
// Dispatch: (84, T, 1) @workgroup_size(64) -> 5376 pairs = 10752 output cols
// ============================================================================
const WGSL_G4_F16_PLE_PROJ: &str = r#"
struct G4PleProjParams {
  tokens: u32,
  K: u32,
  N: u32,
  w_u32_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4PleProjParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read> inputs_embeds: array<f32>;
@group(0) @binding(3) var<storage, read_write> ple_proj_out: array<f32>;

var<workgroup> x_sh: array<f32, 2560>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let col_pair = wid.x * 64u + lid.x;
  let tok = wid.y;
  let x_base = tok * params.K;
  for (var i: u32 = lid.x; i < params.K; i = i + 64u) {
    x_sh[i] = inputs_embeds[x_base + i];
  }
  workgroupBarrier();

  let row_u32_stride = params.N >> 1u; // 5376 u32s
  if (tok >= params.tokens || col_pair >= row_u32_stride) {
    return;
  }

  var acc = vec2<f32>(0.0);
  var w_ptr = params.w_u32_offset + col_pair;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    acc += unpack2x16float(aux[w_ptr]) * x_sh[k];
    w_ptr += row_u32_stride;
  }

  let scaled = acc * 0.01976423537605237;
  let out_base = tok * params.N + col_pair * 2u;
  ple_proj_out[out_base] = scaled.x;
  ple_proj_out[out_base + 1u] = scaled.y;
}
"#;

// ============================================================================
// 111: Batched Per-Layer Projection RMSNorm(256) + Add ple_raw * sqrt(0.5)
// Dispatch: (42, T, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_PLE_NORM_COMBINE: &str = r#"
struct G4PleNormParams {
  tokens: u32,
  num_layers: u32,
  ple_dim: u32,
  norm_u32_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4PleNormParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read> ple_proj: array<f32>;
@group(0) @binding(3) var<storage, read> ple_raw: array<f32>;
@group(0) @binding(4) var<storage, read_write> ple_combined: array<f32>;

var<workgroup> red_sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let layer = wid.x;
  let tok = wid.y;
  let d = lid.x;
  let idx = (tok * params.num_layers + layer) * params.ple_dim + d;
  let val = ple_proj[idx];

  red_sh[d] = val * val;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (d < s) {
      red_sh[d] += red_sh[d + s];
    }
    workgroupBarrier();
  }

  let inv_rms = inverseSqrt(red_sh[0] / f32(params.ple_dim) + 1e-6);
  let wp = unpack2x16float(aux[params.norm_u32_offset + (d >> 1u)]);
  let nw = select(wp.x, wp.y, (d & 1u) != 0u);
  let normed = val * inv_rms * nw;
  ple_combined[idx] = (ple_raw[idx] + normed) * 0.7071067811865476;
}
"#;

// ============================================================================
// 112: Batched RMSNorm / Fused Residual + RMSNorm / + Layer Scalar
// mode 0: out[out_off] = RMSNorm(in[in_off], norm_w)
// mode 1: out_resid[resid_off] += RMSNorm(in[in_off], norm_w)
// mode 2: out_resid[resid_off] = (out_resid[resid_off] + RMSNorm(in[in_off], norm_w)) * layer_scalar
// Dispatch: (T, 1, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_RMSNORM_RESIDUAL_BATCH: &str = r#"
struct G4NormParams {
  tokens: u32,
  dim: u32,
  norm_u32_offset: u32,
  in_offset: u32,
  resid_offset: u32,
  out_offset: u32,
  mode: u32,
  scalar_u32_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4NormParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read> in_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_resid: array<f32>;

var<workgroup> red_sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tok = wid.x;
  let tid = lid.x;
  if (tok >= params.tokens) {
    return;
  }

  let in_base = params.in_offset + tok * params.dim;
  var ss: f32 = 0.0;
  for (var i: u32 = tid; i < params.dim; i = i + 256u) {
    let v = in_buf[in_base + i];
    ss += v * v;
  }
  red_sh[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      red_sh[tid] += red_sh[tid + s];
    }
    workgroupBarrier();
  }

  let inv_rms = inverseSqrt(red_sh[0] / f32(params.dim) + 1e-6);
  let layer_scale = select(1.0, unpack2x16float(aux[params.scalar_u32_offset]).x, params.mode == 2u);
  let out_base = select(params.resid_offset, params.out_offset, params.mode == 0u) + tok * params.dim;

  for (var i: u32 = tid; i < params.dim; i = i + 256u) {
    let wp = unpack2x16float(aux[params.norm_u32_offset + (i >> 1u)]);
    let nw = select(wp.x, wp.y, (i & 1u) != 0u);
    let normed = in_buf[in_base + i] * inv_rms * nw;
    if (params.mode == 0u) {
      out_resid[out_base + i] = normed;
    } else if (params.mode == 1u) {
      out_resid[out_base + i] = out_resid[out_base + i] + normed;
    } else {
      out_resid[out_base + i] = (out_resid[out_base + i] + normed) * layer_scale;
    }
  }
}
"#;

// ============================================================================
// 113: Batched GeGLU (gelu_tanh(gate) * up) for Prefill T>1
// Dispatch: (ceil(inter_dim / 256), T, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_GEGLU_BATCH: &str = r#"
struct G4ElemParams {
  tokens: u32,
  inter_dim: u32,
  in_offset: u32,
  out_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4ElemParams;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;
@group(0) @binding(2) var<storage, read_write> act_out: array<f32>;

fn gelu_tanh(v: f32) -> f32 {
  let arg = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(arg));
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let idx = wid.x * 256u + lid.x;
  let tok = wid.y;
  if (tok >= params.tokens || idx >= params.inter_dim) {
    return;
  }
  let base = params.in_offset + tok * (params.inter_dim * 2u);
  let g = gate_up[base + idx];
  let u = gate_up[base + params.inter_dim + idx];
  act_out[params.out_offset + tok * params.inter_dim + idx] = gelu_tanh(g) * u;
}
"#;

// ============================================================================
// 114: Fused Per-Head QK RMSNorm + V Unit-RMSNorm + RoPE (Sliding/Global Proportional) + KV Write
// Handles BOTH Sliding (D=256, 1024-ring FP16) & Global (D=512, FP16 or Symmetric INT8)
// Dispatch: (is_shared ? 8 : 12, T, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_QKV_NORM_ROPE_KVWRITE: &str = r#"
struct G4RopeParams {
  tokens: u32,
  seq_pos: u32,
  head_dim: u32,
  num_q_heads: u32,
  num_kv_heads: u32,
  is_full: u32,
  is_shared: u32,
  kv_mode: u32,
  local_donor_idx: u32,
  max_seq: u32,
  q_norm_u32_off: u32,
  k_norm_u32_off: u32,
};

@group(0) @binding(0) var<uniform> params: G4RopeParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read_write> q_buf: array<f32>;
@group(0) @binding(3) var<storage, read> k_buf: array<f32>;
@group(0) @binding(4) var<storage, read> v_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> kv_cache: array<u32>;

var<workgroup> red_sh: array<f32, 256>;
var<workgroup> tmp_sh: array<f32, 512>;

fn pack4_i8_vec(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h_idx = wid.x;
  let tok = wid.y;
  let i = lid.x;
  let half = params.head_dim >> 1u; // 128 or 256
  let pos = params.seq_pos + tok;

  let is_q = h_idx < params.num_q_heads;
  let is_k = h_idx >= params.num_q_heads && h_idx < (params.num_q_heads + params.num_kv_heads);
  let kvh = select(h_idx - params.num_q_heads - params.num_kv_heads, h_idx - params.num_q_heads, is_k);

  var x0: f32 = 0.0;
  var x1: f32 = 0.0;
  if (i < half && tok < params.tokens) {
    if (is_q) {
      let base = (tok * params.num_q_heads + h_idx) * params.head_dim;
      x0 = q_buf[base + i];
      x1 = q_buf[base + i + half];
    } else if (is_k) {
      let base = (tok * params.num_kv_heads + kvh) * params.head_dim;
      x0 = k_buf[base + i];
      x1 = k_buf[base + i + half];
    } else {
      let base = (tok * params.num_kv_heads + kvh) * params.head_dim;
      x0 = v_buf[base + i];
      x1 = v_buf[base + i + half];
    }
  }

  red_sh[i] = x0 * x0 + x1 * x1;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (i < s) {
      red_sh[i] += red_sh[i + s];
    }
    workgroupBarrier();
  }
  let inv_rms = inverseSqrt(red_sh[0] / f32(params.head_dim) + 1e-6);

  var w0: f32 = 1.0;
  var w1: f32 = 1.0;
  if (i < half && (is_q || is_k)) {
    let n_off = select(params.k_norm_u32_off, params.q_norm_u32_off, is_q);
    let idx0 = i;
    let wp0 = unpack2x16float(aux[n_off + (idx0 >> 1u)]);
    w0 = select(wp0.x, wp0.y, (idx0 & 1u) != 0u);
    let idx1 = i + half;
    let wp1 = unpack2x16float(aux[n_off + (idx1 >> 1u)]);
    w1 = select(wp1.x, wp1.y, (idx1 & 1u) != 0u);
  }

  let nx0 = x0 * inv_rms * w0;
  let nx1 = x1 * inv_rms * w1;

  var out0: f32 = nx0;
  var out1: f32 = nx1;
  if (is_q || is_k) {
    let rot_pairs = select(128u, 64u, params.is_full != 0u);
    if (i < rot_pairs) {
      let theta = select(10000.0, 1000000.0, params.is_full != 0u);
      let inv_freq = 1.0 / pow(theta, f32(2u * i) / f32(params.head_dim));
      let ang = f32(pos) * inv_freq;
      let c = cos(ang);
      let sn = sin(ang);
      out0 = nx0 * c - nx1 * sn;
      out1 = nx0 * sn + nx1 * c;
    }
  }

  if (is_q) {
    if (i < half && tok < params.tokens) {
      let base = (tok * params.num_q_heads + h_idx) * params.head_dim;
      q_buf[base + i] = out0;
      q_buf[base + i + half] = out1;
    }
    return;
  }

  if (i < half) {
    tmp_sh[i] = out0;
    tmp_sh[i + half] = out1;
  }
  workgroupBarrier();

  if (tok >= params.tokens || params.is_shared != 0u) {
    return;
  }

  let is_v = !is_k;
  let plane_idx = select(0u, 1u, is_v);

  if (params.is_full == 0u) {
    // Sliding FP16 1024-slot ring buffer: D=256 -> 128 u32s per token-head
    let head_u32s = 1024u * 128u;
    let donor_u32s = 4u * head_u32s;
    let dst_base = params.local_donor_idx * donor_u32s
      + (plane_idx * 2u + kvh) * head_u32s
      + (pos & 1023u) * 128u;
    if (i < 128u) {
      kv_cache[dst_base + i] = pack2x16float(vec2<f32>(tmp_sh[2u * i], tmp_sh[2u * i + 1u]));
    }
    return;
  }

  if (params.kv_mode == 0u) {
    // Global FP16: D=512 -> 256 u32s per token-head
    let head_u32s = params.max_seq * 256u;
    let donor_u32s = 4u * head_u32s;
    let dst_base = params.local_donor_idx * donor_u32s
      + (plane_idx * 2u + kvh) * head_u32s
      + pos * 256u;
    kv_cache[dst_base + i] = pack2x16float(vec2<f32>(tmp_sh[2u * i], tmp_sh[2u * i + 1u]));
    return;
  }

  if (params.kv_mode == 2u) {
    // Global Symmetric INT4 (8 groups x 64 dims -> 64 u32s packed nibbles + 4 u32s f16 scales = 68 u32s/head)
    let grp = i >> 3u;
    let head_u32s = params.max_seq * 68u;
    let donor_u32s = 4u * head_u32s;
    let dst_base = params.local_donor_idx * donor_u32s + (plane_idx * 2u + kvh) * head_u32s + pos * 68u;
    if (i < 64u) {
      let b = i * 8u;
      var max64: f32 = 0.0;
      for (var k: u32 = grp * 64u; k < (grp + 1u) * 64u; k = k + 1u) {
        max64 = max(max64, abs(tmp_sh[k]));
      }
      let s_g = max(max64 / 7.0, 1e-6);
      var pw: u32 = 0u;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let q = u32(clamp(i32(round(tmp_sh[b + k] / s_g)) + 8, 1, 15));
        pw = pw | (q << (k * 4u));
      }
      kv_cache[dst_base + i] = pw;
    }
    if (i < 4u) {
      let g0 = i * 2u;
      let g1 = g0 + 1u;
      var m0: f32 = 0.0;
      var m1: f32 = 0.0;
      for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        m0 = max(m0, abs(tmp_sh[g0 * 64u + k]));
        m1 = max(m1, abs(tmp_sh[g1 * 64u + k]));
      }
      kv_cache[dst_base + 64u + i] = pack2x16float(vec2<f32>(max(m0 / 7.0, 1e-6), max(m1 / 7.0, 1e-6)));
    }
    return;
  }

  // Global Symmetric INT8: D=512 -> 128 u32s packed int8 + 1 f32 scale u32 (129 u32s/head)
  let ma = max(abs(tmp_sh[i]), abs(tmp_sh[i + 256u]));
  red_sh[i] = ma;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (i < s) {
      red_sh[i] = max(red_sh[i], red_sh[i + s]);
    }
    workgroupBarrier();
  }
  let scale = max(red_sh[0] / 127.0, 1e-6);
  let head_u32s = params.max_seq * 129u;
  let donor_u32s = 4u * head_u32s;
  let dst_base = params.local_donor_idx * donor_u32s
    + (plane_idx * 2u + kvh) * head_u32s
    + pos * 129u;
  if (i < 128u) {
    let b = i * 4u;
    let v4 = vec4<f32>(tmp_sh[b], tmp_sh[b + 1u], tmp_sh[b + 2u], tmp_sh[b + 3u]) / scale;
    kv_cache[dst_base + i] = pack4_i8_vec(v4);
  }
  if (i == 0u) {
    kv_cache[dst_base + 128u] = bitcast<u32>(scale);
  }
}
"#;

// ============================================================================
// 115: GQA Decode Sliding Attention (FP16 1024-ring, window=512, D=256, scale=1.0)
// 1 workgroup per KV head -> shares K/V loads across 4 Q heads!
// Dispatch: (2, 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_DECODE_SLIDING_FP16: &str = r#"
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
@group(0) @binding(2) var<storage, read> kv_sliding: array<u32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

var<workgroup> red_sh: array<vec4<f32>, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let kv_head = wid.x;
  let qh0 = kv_head * 4u;
  let tid = lid.x; // 0..127 -> 2 dims per thread (D=256)
  let d = tid * 2u;

  let q0 = vec2<f32>(q_buf[(qh0 + 0u) * 256u + d], q_buf[(qh0 + 0u) * 256u + d + 1u]);
  let q1 = vec2<f32>(q_buf[(qh0 + 1u) * 256u + d], q_buf[(qh0 + 1u) * 256u + d + 1u]);
  let q2 = vec2<f32>(q_buf[(qh0 + 2u) * 256u + d], q_buf[(qh0 + 2u) * 256u + d + 1u]);
  let q3 = vec2<f32>(q_buf[(qh0 + 3u) * 256u + d], q_buf[(qh0 + 3u) * 256u + d + 1u]);

  let head_u32s = 1024u * 128u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let win_minus_1 = max(params.window_size, 1u) - 1u;
  let t_start = params.seq_pos - min(params.seq_pos, win_minus_1);
  let t_end = params.seq_pos + 1u;

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec2<f32>(0.0);
  var acc1 = vec2<f32>(0.0);
  var acc2 = vec2<f32>(0.0);
  var acc3 = vec2<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let slot = (t & 1023u) * 128u + tid;
    let k2 = unpack2x16float(kv_sliding[k_base + slot]);
    red_sh[tid] = vec4<f32>(dot(q0, k2), dot(q1, k2), dot(q2, k2), dot(q3, k2));
    workgroupBarrier();

    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_sh[tid] += red_sh[tid + s];
      }
      workgroupBarrier();
    }

    let score = red_sh[0]; // scale = 1.0 in Gemma 4!
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let v2 = unpack2x16float(kv_sliding[v_base + slot]);
    acc0 = acc0 * alpha.x + v2 * beta.x;
    acc1 = acc1 * alpha.y + v2 * beta.y;
    acc2 = acc2 * alpha.z + v2 * beta.z;
    acc3 = acc3 * alpha.w + v2 * beta.w;
    m4 = m_new;
    workgroupBarrier();
  }

  let o0 = acc0 / max(l4.x, 1e-9);
  let o1 = acc1 / max(l4.y, 1e-9);
  let o2 = acc2 / max(l4.z, 1e-9);
  let o3 = acc3 / max(l4.w, 1e-9);

  attn_out[(qh0 + 0u) * 256u + d] = o0.x; attn_out[(qh0 + 0u) * 256u + d + 1u] = o0.y;
  attn_out[(qh0 + 1u) * 256u + d] = o1.x; attn_out[(qh0 + 1u) * 256u + d + 1u] = o1.y;
  attn_out[(qh0 + 2u) * 256u + d] = o2.x; attn_out[(qh0 + 2u) * 256u + d + 1u] = o2.y;
  attn_out[(qh0 + 3u) * 256u + d] = o3.x; attn_out[(qh0 + 3u) * 256u + d + 1u] = o3.y;
}
"#;

// ============================================================================
// 116: GQA Decode Global Split-K (D=512, scale=1.0, FP16 or Symmetric INT8)
// Dispatch: (2, num_splits, 1) @workgroup_size(128) -> 4 dims/thread
// ============================================================================
const WGSL_G4_GQA_DECODE_GLOBAL_SPLIT: &str = r#"
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

var<workgroup> red_sh: array<vec4<f32>, 128>;

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
) {
  let kv_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits, 1u);
  let qh0 = kv_head * 4u;
  let tid = lid.x;
  let d = tid * 4u;

  let b0 = (qh0 + 0u) * 512u + d;
  let b1 = (qh0 + 1u) * 512u + d;
  let b2 = (qh0 + 2u) * 512u + d;
  let b3 = (qh0 + 3u) * 512u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let tok_u32_stride = select(256u, 129u, params.kv_mode != 0u);
  let head_u32s = params.max_seq * tok_u32_stride;
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
    var k4: vec4<f32>;
    var v4: vec4<f32>;
    if (params.kv_mode == 0u) {
      let ko = k_base + t * 256u + tid * 2u;
      let ka = unpack2x16float(kv_global[ko]);
      let kb = unpack2x16float(kv_global[ko + 1u]);
      k4 = vec4<f32>(ka.x, ka.y, kb.x, kb.y);
      let vo = v_base + t * 256u + tid * 2u;
      let va = unpack2x16float(kv_global[vo]);
      let vb = unpack2x16float(kv_global[vo + 1u]);
      v4 = vec4<f32>(va.x, va.y, vb.x, vb.y);
    } else {
      let ko = k_base + t * 129u;
      let ks = bitcast<f32>(kv_global[ko + 128u]);
      k4 = unpack4_i8(kv_global[ko + tid]) * ks;
      let vo = v_base + t * 129u;
      let vs = bitcast<f32>(kv_global[vo + 128u]);
      v4 = unpack4_i8(kv_global[vo + tid]) * vs;
    }

    red_sh[tid] = vec4<f32>(dot(q0, k4), dot(q1, k4), dot(q2, k4), dot(q3, k4));
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_sh[tid] += red_sh[tid + s];
      }
      workgroupBarrier();
    }

    let score = red_sh[0]; // scale = 1.0 in Gemma 4!
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    acc0 = acc0 * alpha.x + v4 * beta.x;
    acc1 = acc1 * alpha.y + v4 * beta.y;
    acc2 = acc2 * alpha.z + v4 * beta.z;
    acc3 = acc3 * alpha.w + v4 * beta.w;
    m4 = m_new;
    workgroupBarrier();
  }

  // Split partial stride: 516 floats (512 acc + m + l + 2 pad)
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
// 117: GQA Decode Global Split-K Merge Reduce (8 heads x D=512)
// Dispatch: (8, 1, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_GQA_DECODE_REDUCE: &str = r#"
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
@group(0) @binding(1) var<storage, read> split_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> attn_out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let qh = wid.x;
  let d = lid.x;
  let num_splits = max(params.num_splits, 1u);

  var m_g: f32 = -1e30;
  var l_g: f32 = 0.0;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;

  for (var s: u32 = 0u; s < num_splits; s = s + 1u) {
    let base = (qh * num_splits + s) * 516u;
    let l_s = split_out[base + 513u];
    if (l_s <= 0.0) {
      continue;
    }
    let m_s = split_out[base + 512u];
    let m_new = max(m_g, m_s);
    let alpha = exp(m_g - m_new);
    let beta = exp(m_s - m_new);
    l_g = l_g * alpha + l_s * beta;
    acc0 = acc0 * alpha + split_out[base + d] * beta;
    acc1 = acc1 * alpha + split_out[base + d + 256u] * beta;
    m_g = m_new;
  }

  let inv_l = 1.0 / max(l_g, 1e-9);
  attn_out[qh * 512u + d] = acc0 * inv_l;
  attn_out[qh * 512u + d + 256u] = acc1 * inv_l;
}
"#;

// ============================================================================
// 118: Batched Prefill Sliding Attention (D=256, window=512, ring=1024, scale=1.0)
// Dispatch: (2, T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_PREFILL_SLIDING_FP16: &str = r#"
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
@group(0) @binding(2) var<storage, read> kv_sliding: array<u32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

var<workgroup> red_sh: array<vec4<f32>, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let kv_head = wid.x;
  let tok = wid.y;
  let qh0 = kv_head * 4u;
  let tid = lid.x;
  let d = tid * 2u;
  let pos_q = params.seq_pos + tok;

  let q_tok_base = tok * 2048u; // 8 * 256
  let q0 = vec2<f32>(q_buf[q_tok_base + (qh0 + 0u) * 256u + d], q_buf[q_tok_base + (qh0 + 0u) * 256u + d + 1u]);
  let q1 = vec2<f32>(q_buf[q_tok_base + (qh0 + 1u) * 256u + d], q_buf[q_tok_base + (qh0 + 1u) * 256u + d + 1u]);
  let q2 = vec2<f32>(q_buf[q_tok_base + (qh0 + 2u) * 256u + d], q_buf[q_tok_base + (qh0 + 2u) * 256u + d + 1u]);
  let q3 = vec2<f32>(q_buf[q_tok_base + (qh0 + 3u) * 256u + d], q_buf[q_tok_base + (qh0 + 3u) * 256u + d + 1u]);

  let head_u32s = 1024u * 128u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let win_minus_1 = max(params.window_size, 1u) - 1u;
  let t_start = pos_q - min(pos_q, win_minus_1);
  let t_end = pos_q + 1u;

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec2<f32>(0.0);
  var acc1 = vec2<f32>(0.0);
  var acc2 = vec2<f32>(0.0);
  var acc3 = vec2<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let slot = (t & 1023u) * 128u + tid;
    let k2 = unpack2x16float(kv_sliding[k_base + slot]);
    red_sh[tid] = vec4<f32>(dot(q0, k2), dot(q1, k2), dot(q2, k2), dot(q3, k2));
    workgroupBarrier();

    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_sh[tid] += red_sh[tid + s];
      }
      workgroupBarrier();
    }

    let score = red_sh[0];
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let v2 = unpack2x16float(kv_sliding[v_base + slot]);
    acc0 = acc0 * alpha.x + v2 * beta.x;
    acc1 = acc1 * alpha.y + v2 * beta.y;
    acc2 = acc2 * alpha.z + v2 * beta.z;
    acc3 = acc3 * alpha.w + v2 * beta.w;
    m4 = m_new;
    workgroupBarrier();
  }

  let o0 = acc0 / max(l4.x, 1e-9);
  let o1 = acc1 / max(l4.y, 1e-9);
  let o2 = acc2 / max(l4.z, 1e-9);
  let o3 = acc3 / max(l4.w, 1e-9);

  attn_out[q_tok_base + (qh0 + 0u) * 256u + d] = o0.x; attn_out[q_tok_base + (qh0 + 0u) * 256u + d + 1u] = o0.y;
  attn_out[q_tok_base + (qh0 + 1u) * 256u + d] = o1.x; attn_out[q_tok_base + (qh0 + 1u) * 256u + d + 1u] = o1.y;
  attn_out[q_tok_base + (qh0 + 2u) * 256u + d] = o2.x; attn_out[q_tok_base + (qh0 + 2u) * 256u + d + 1u] = o2.y;
  attn_out[q_tok_base + (qh0 + 3u) * 256u + d] = o3.x; attn_out[q_tok_base + (qh0 + 3u) * 256u + d + 1u] = o3.y;
}
"#;

// ============================================================================
// 119: Batched Prefill Global Attention (D=512, scale=1.0, FP16 or Symmetric INT8)
// Dispatch: (2, T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_PREFILL_GLOBAL: &str = r#"
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
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

var<workgroup> red_sh: array<vec4<f32>, 128>;

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
) {
  let kv_head = wid.x;
  let tok = wid.y;
  let qh0 = kv_head * 4u;
  let tid = lid.x;
  let d = tid * 4u;
  let pos_q = params.seq_pos + tok;

  let q_tok_base = tok * 4096u; // 8 * 512
  let b0 = q_tok_base + (qh0 + 0u) * 512u + d;
  let b1 = q_tok_base + (qh0 + 1u) * 512u + d;
  let b2 = q_tok_base + (qh0 + 2u) * 512u + d;
  let b3 = q_tok_base + (qh0 + 3u) * 512u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let tok_u32_stride = select(256u, 129u, params.kv_mode != 0u);
  let head_u32s = params.max_seq * tok_u32_stride;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let t_end = pos_q + 1u;
  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = 0u; t < t_end; t = t + 1u) {
    var k4: vec4<f32>;
    var v4: vec4<f32>;
    if (params.kv_mode == 0u) {
      let ko = k_base + t * 256u + tid * 2u;
      let ka = unpack2x16float(kv_global[ko]);
      let kb = unpack2x16float(kv_global[ko + 1u]);
      k4 = vec4<f32>(ka.x, ka.y, kb.x, kb.y);
      let vo = v_base + t * 256u + tid * 2u;
      let va = unpack2x16float(kv_global[vo]);
      let vb = unpack2x16float(kv_global[vo + 1u]);
      v4 = vec4<f32>(va.x, va.y, vb.x, vb.y);
    } else {
      let ko = k_base + t * 129u;
      let ks = bitcast<f32>(kv_global[ko + 128u]);
      k4 = unpack4_i8(kv_global[ko + tid]) * ks;
      let vo = v_base + t * 129u;
      let vs = bitcast<f32>(kv_global[vo + 128u]);
      v4 = unpack4_i8(kv_global[vo + tid]) * vs;
    }

    red_sh[tid] = vec4<f32>(dot(q0, k4), dot(q1, k4), dot(q2, k4), dot(q3, k4));
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_sh[tid] += red_sh[tid + s];
      }
      workgroupBarrier();
    }

    let score = red_sh[0];
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    acc0 = acc0 * alpha.x + v4 * beta.x;
    acc1 = acc1 * alpha.y + v4 * beta.y;
    acc2 = acc2 * alpha.z + v4 * beta.z;
    acc3 = acc3 * alpha.w + v4 * beta.w;
    m4 = m_new;
    workgroupBarrier();
  }

  let o0 = acc0 / max(l4.x, 1e-9);
  let o1 = acc1 / max(l4.y, 1e-9);
  let o2 = acc2 / max(l4.z, 1e-9);
  let o3 = acc3 / max(l4.w, 1e-9);

  attn_out[b0] = o0.x; attn_out[b0 + 1u] = o0.y; attn_out[b0 + 2u] = o0.z; attn_out[b0 + 3u] = o0.w;
  attn_out[b1] = o1.x; attn_out[b1 + 1u] = o1.y; attn_out[b1 + 2u] = o1.z; attn_out[b1 + 3u] = o1.w;
  attn_out[b2] = o2.x; attn_out[b2 + 1u] = o2.y; attn_out[b2 + 2u] = o2.z; attn_out[b2 + 3u] = o2.w;
  attn_out[b3] = o3.x; attn_out[b3 + 1u] = o3.y; attn_out[b3 + 2u] = o3.z; attn_out[b3 + 3u] = o3.w;
}
"#;

// ============================================================================
// 120: GQA Decode Sliding Split-K + Subgroups (D=256, 16 splits, 1 barrier/tok!)
// Dispatch: (2, 16, 1) @workgroup_size(64) -> 4 dims/thread
// ============================================================================
const WGSL_G4_GQA_DECODE_SLIDING_SPLIT_SG: &str = r#"
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
@group(0) @binding(2) var<storage, read> kv_sliding: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> red_sh: array<vec4<f32>, 128>;

@compute @workgroup_size(64)
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
  let tid = lid.x; // 0..63 -> 4 dims/thread (D=256)
  let d = tid * 4u;

  let b0 = (qh0 + 0u) * 256u + d;
  let b1 = (qh0 + 1u) * 256u + d;
  let b2 = (qh0 + 2u) * 256u + d;
  let b3 = (qh0 + 3u) * 256u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let head_u32s = 1024u * 128u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let win_minus_1 = max(params.window_size, 1u) - 1u;
  let win_start = params.seq_pos - min(params.seq_pos, win_minus_1);
  let win_len = (params.seq_pos + 1u) - win_start;
  let chunk = (win_len + num_splits - 1u) / num_splits;
  let t_start = win_start + min(split_id * chunk, win_len);
  let t_end = win_start + min((split_id + 1u) * chunk, win_len);
  let sg_step = max(sg_size, 1u);

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let slot = (t & 1023u) * 128u + tid * 2u;
    let ka = unpack2x16float(kv_sliding[k_base + slot]);
    let kb = unpack2x16float(kv_sliding[k_base + slot + 1u]);
    let k4 = vec4<f32>(ka.x, ka.y, kb.x, kb.y);

    let p4 = vec4<f32>(dot(q0, k4), dot(q1, k4), dot(q2, k4), dot(q3, k4));
    let sh_slot = ((t - t_start) & 1u) * 64u;
    let sg_dot = subgroupAdd(p4);
    red_sh[sh_slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var score = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      score += red_sh[sh_slot + s];
    }

    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let va = unpack2x16float(kv_sliding[v_base + slot]);
    let vb = unpack2x16float(kv_sliding[v_base + slot + 1u]);
    let v4 = vec4<f32>(va.x, va.y, vb.x, vb.y);
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
    let s0 = (qh0 + 0u) * 256u + d;
    let s1 = (qh0 + 1u) * 256u + d;
    let s2 = (qh0 + 2u) * 256u + d;
    let s3 = (qh0 + 3u) * 256u + d;
    split_out[s0] = o0.x; split_out[s0 + 1u] = o0.y; split_out[s0 + 2u] = o0.z; split_out[s0 + 3u] = o0.w;
    split_out[s1] = o1.x; split_out[s1 + 1u] = o1.y; split_out[s1 + 2u] = o1.z; split_out[s1 + 3u] = o1.w;
    split_out[s2] = o2.x; split_out[s2 + 1u] = o2.y; split_out[s2 + 2u] = o2.z; split_out[s2 + 3u] = o2.w;
    split_out[s3] = o3.x; split_out[s3 + 1u] = o3.y; split_out[s3 + 2u] = o3.z; split_out[s3 + 3u] = o3.w;
    return;
  }

  let ob0 = ((qh0 + 0u) * num_splits + split_id) * 260u;
  split_out[ob0 + d] = acc0.x; split_out[ob0 + d + 1u] = acc0.y; split_out[ob0 + d + 2u] = acc0.z; split_out[ob0 + d + 3u] = acc0.w;
  if (tid == 0u) { split_out[ob0 + 256u] = m4.x; split_out[ob0 + 257u] = l4.x; }

  let ob1 = ((qh0 + 1u) * num_splits + split_id) * 260u;
  split_out[ob1 + d] = acc1.x; split_out[ob1 + d + 1u] = acc1.y; split_out[ob1 + d + 2u] = acc1.z; split_out[ob1 + d + 3u] = acc1.w;
  if (tid == 0u) { split_out[ob1 + 256u] = m4.y; split_out[ob1 + 257u] = l4.y; }

  let ob2 = ((qh0 + 2u) * num_splits + split_id) * 260u;
  split_out[ob2 + d] = acc2.x; split_out[ob2 + d + 1u] = acc2.y; split_out[ob2 + d + 2u] = acc2.z; split_out[ob2 + d + 3u] = acc2.w;
  if (tid == 0u) { split_out[ob2 + 256u] = m4.z; split_out[ob2 + 257u] = l4.z; }

  let ob3 = ((qh0 + 3u) * num_splits + split_id) * 260u;
  split_out[ob3 + d] = acc3.x; split_out[ob3 + d + 1u] = acc3.y; split_out[ob3 + d + 2u] = acc3.z; split_out[ob3 + d + 3u] = acc3.w;
  if (tid == 0u) { split_out[ob3 + 256u] = m4.w; split_out[ob3 + 257u] = l4.w; }
}
"#;

// ============================================================================
// 121: GQA Decode Sliding Split-K Merge Reduce (8 heads x D=256)
// Dispatch: (8, 1, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_GQA_DECODE_SLIDING_REDUCE: &str = r#"
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
@group(0) @binding(1) var<storage, read> split_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> attn_out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let qh = wid.x;
  let d = lid.x;
  let num_splits = max(params.num_splits, 1u);

  var m_g: f32 = -1e30;
  var l_g: f32 = 0.0;
  var acc: f32 = 0.0;

  for (var s: u32 = 0u; s < num_splits; s = s + 1u) {
    let base = (qh * num_splits + s) * 260u;
    let l_s = split_out[base + 257u];
    if (l_s <= 0.0) {
      continue;
    }
    let m_s = split_out[base + 256u];
    let m_new = max(m_g, m_s);
    let alpha = exp(m_g - m_new);
    let beta = exp(m_s - m_new);
    l_g = l_g * alpha + l_s * beta;
    acc = acc * alpha + split_out[base + d] * beta;
    m_g = m_new;
  }

  attn_out[qh * 256u + d] = acc / max(l_g, 1e-9);
}
"#;

// ============================================================================
// 122: GQA Decode Global Split-K + Subgroups (D=512, 64 splits, 1 barrier/tok!)
// Dispatch: (2, num_splits, 1) @workgroup_size(128) -> 4 dims/thread
// ============================================================================
const WGSL_G4_GQA_DECODE_GLOBAL_SPLIT_SG: &str = r#"
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

var<workgroup> red_sh: array<vec4<f32>, 256>;

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

  let b0 = (qh0 + 0u) * 512u + d;
  let b1 = (qh0 + 1u) * 512u + d;
  let b2 = (qh0 + 2u) * 512u + d;
  let b3 = (qh0 + 3u) * 512u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let tok_u32_stride = select(256u, 129u, params.kv_mode != 0u);
  let head_u32s = params.max_seq * tok_u32_stride;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let total_toks = params.seq_pos + 1u;
  let chunk = (total_toks + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, total_toks);
  let t_end = min(t_start + chunk, total_toks);
  let sg_step = max(sg_size, 1u);

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    var k4: vec4<f32>;
    var v4: vec4<f32>;
    if (params.kv_mode == 0u) {
      let ko = k_base + t * 256u + tid * 2u;
      let ka = unpack2x16float(kv_global[ko]);
      let kb = unpack2x16float(kv_global[ko + 1u]);
      k4 = vec4<f32>(ka.x, ka.y, kb.x, kb.y);
      let vo = v_base + t * 256u + tid * 2u;
      let va = unpack2x16float(kv_global[vo]);
      let vb = unpack2x16float(kv_global[vo + 1u]);
      v4 = vec4<f32>(va.x, va.y, vb.x, vb.y);
    } else {
      let ko = k_base + t * 129u;
      let ks = bitcast<f32>(kv_global[ko + 128u]);
      k4 = unpack4_i8(kv_global[ko + tid]) * ks;
      let vo = v_base + t * 129u;
      let vs = bitcast<f32>(kv_global[vo + 128u]);
      v4 = unpack4_i8(kv_global[vo + tid]) * vs;
    }

    let p4 = vec4<f32>(dot(q0, k4), dot(q1, k4), dot(q2, k4), dot(q3, k4));
    let sh_slot = ((t - t_start) & 1u) * 128u;
    let sg_dot = subgroupAdd(p4);
    red_sh[sh_slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var score = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      score += red_sh[sh_slot + s];
    }

    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

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
// 123: Batched Prefill Sliding Attention + Subgroups (D=256, 1 barrier/tok!)
// Dispatch: (2, T, 1) @workgroup_size(64)
// ============================================================================
const WGSL_G4_GQA_PREFILL_SLIDING_SG: &str = r#"
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
@group(0) @binding(2) var<storage, read> kv_sliding: array<u32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

var<workgroup> red_sh: array<vec4<f32>, 128>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let tok = wid.y;
  let qh0 = kv_head * 4u;
  let tid = lid.x;
  let d = tid * 4u;
  let pos_q = params.seq_pos + tok;

  let q_tok_base = tok * 2048u;
  let b0 = q_tok_base + (qh0 + 0u) * 256u + d;
  let b1 = q_tok_base + (qh0 + 1u) * 256u + d;
  let b2 = q_tok_base + (qh0 + 2u) * 256u + d;
  let b3 = q_tok_base + (qh0 + 3u) * 256u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let head_u32s = 1024u * 128u;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let win_minus_1 = max(params.window_size, 1u) - 1u;
  let t_start = pos_q - min(pos_q, win_minus_1);
  let t_end = pos_q + 1u;
  let sg_step = max(sg_size, 1u);

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let slot = (t & 1023u) * 128u + tid * 2u;
    let ka = unpack2x16float(kv_sliding[k_base + slot]);
    let kb = unpack2x16float(kv_sliding[k_base + slot + 1u]);
    let k4 = vec4<f32>(ka.x, ka.y, kb.x, kb.y);

    let p4 = vec4<f32>(dot(q0, k4), dot(q1, k4), dot(q2, k4), dot(q3, k4));
    let sh_slot = ((t - t_start) & 1u) * 64u;
    let sg_dot = subgroupAdd(p4);
    red_sh[sh_slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var score = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      score += red_sh[sh_slot + s];
    }

    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let va = unpack2x16float(kv_sliding[v_base + slot]);
    let vb = unpack2x16float(kv_sliding[v_base + slot + 1u]);
    let v4 = vec4<f32>(va.x, va.y, vb.x, vb.y);
    acc0 = acc0 * alpha.x + v4 * beta.x;
    acc1 = acc1 * alpha.y + v4 * beta.y;
    acc2 = acc2 * alpha.z + v4 * beta.z;
    acc3 = acc3 * alpha.w + v4 * beta.w;
    m4 = m_new;
  }

  let o0 = acc0 / max(l4.x, 1e-9);
  let o1 = acc1 / max(l4.y, 1e-9);
  let o2 = acc2 / max(l4.z, 1e-9);
  let o3 = acc3 / max(l4.w, 1e-9);

  attn_out[b0] = o0.x; attn_out[b0 + 1u] = o0.y; attn_out[b0 + 2u] = o0.z; attn_out[b0 + 3u] = o0.w;
  attn_out[b1] = o1.x; attn_out[b1 + 1u] = o1.y; attn_out[b1 + 2u] = o1.z; attn_out[b1 + 3u] = o1.w;
  attn_out[b2] = o2.x; attn_out[b2 + 1u] = o2.y; attn_out[b2 + 2u] = o2.z; attn_out[b2 + 3u] = o2.w;
  attn_out[b3] = o3.x; attn_out[b3 + 1u] = o3.y; attn_out[b3 + 2u] = o3.z; attn_out[b3 + 3u] = o3.w;
}
"#;

// ============================================================================
// 124: Batched Prefill Global Attention + Subgroups (D=512, 1 barrier/tok!)
// Dispatch: (2, T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_GQA_PREFILL_GLOBAL_SG: &str = r#"
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
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

var<workgroup> red_sh: array<vec4<f32>, 256>;

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
  let tok = wid.y;
  let qh0 = kv_head * 4u;
  let tid = lid.x;
  let d = tid * 4u;
  let pos_q = params.seq_pos + tok;

  let q_tok_base = tok * 4096u;
  let b0 = q_tok_base + (qh0 + 0u) * 512u + d;
  let b1 = q_tok_base + (qh0 + 1u) * 512u + d;
  let b2 = q_tok_base + (qh0 + 2u) * 512u + d;
  let b3 = q_tok_base + (qh0 + 3u) * 512u + d;
  let q0 = vec4<f32>(q_buf[b0], q_buf[b0 + 1u], q_buf[b0 + 2u], q_buf[b0 + 3u]);
  let q1 = vec4<f32>(q_buf[b1], q_buf[b1 + 1u], q_buf[b1 + 2u], q_buf[b1 + 3u]);
  let q2 = vec4<f32>(q_buf[b2], q_buf[b2 + 1u], q_buf[b2 + 2u], q_buf[b2 + 3u]);
  let q3 = vec4<f32>(q_buf[b3], q_buf[b3 + 1u], q_buf[b3 + 2u], q_buf[b3 + 3u]);

  let tok_u32_stride = select(256u, 129u, params.kv_mode != 0u);
  let head_u32s = params.max_seq * tok_u32_stride;
  let donor_u32s = 4u * head_u32s;
  let k_base = params.local_donor_idx * donor_u32s + kv_head * head_u32s;
  let v_base = params.local_donor_idx * donor_u32s + (2u + kv_head) * head_u32s;

  let t_end = pos_q + 1u;
  let sg_step = max(sg_size, 1u);
  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var t: u32 = 0u; t < t_end; t = t + 1u) {
    var k4: vec4<f32>;
    var v4: vec4<f32>;
    if (params.kv_mode == 0u) {
      let ko = k_base + t * 256u + tid * 2u;
      let ka = unpack2x16float(kv_global[ko]);
      let kb = unpack2x16float(kv_global[ko + 1u]);
      k4 = vec4<f32>(ka.x, ka.y, kb.x, kb.y);
      let vo = v_base + t * 256u + tid * 2u;
      let va = unpack2x16float(kv_global[vo]);
      let vb = unpack2x16float(kv_global[vo + 1u]);
      v4 = vec4<f32>(va.x, va.y, vb.x, vb.y);
    } else {
      let ko = k_base + t * 129u;
      let ks = bitcast<f32>(kv_global[ko + 128u]);
      k4 = unpack4_i8(kv_global[ko + tid]) * ks;
      let vo = v_base + t * 129u;
      let vs = bitcast<f32>(kv_global[vo + 128u]);
      v4 = unpack4_i8(kv_global[vo + tid]) * vs;
    }

    let p4 = vec4<f32>(dot(q0, k4), dot(q1, k4), dot(q2, k4), dot(q3, k4));
    let sh_slot = (t & 1u) * 128u;
    let sg_dot = subgroupAdd(p4);
    red_sh[sh_slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var score = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      score += red_sh[sh_slot + s];
    }

    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    acc0 = acc0 * alpha.x + v4 * beta.x;
    acc1 = acc1 * alpha.y + v4 * beta.y;
    acc2 = acc2 * alpha.z + v4 * beta.z;
    acc3 = acc3 * alpha.w + v4 * beta.w;
    m4 = m_new;
  }

  let o0 = acc0 / max(l4.x, 1e-9);
  let o1 = acc1 / max(l4.y, 1e-9);
  let o2 = acc2 / max(l4.z, 1e-9);
  let o3 = acc3 / max(l4.w, 1e-9);

  attn_out[b0] = o0.x; attn_out[b0 + 1u] = o0.y; attn_out[b0 + 2u] = o0.z; attn_out[b0 + 3u] = o0.w;
  attn_out[b1] = o1.x; attn_out[b1 + 1u] = o1.y; attn_out[b1 + 2u] = o1.z; attn_out[b1 + 3u] = o1.w;
  attn_out[b2] = o2.x; attn_out[b2 + 1u] = o2.y; attn_out[b2 + 2u] = o2.z; attn_out[b2 + 3u] = o2.w;
  attn_out[b3] = o3.x; attn_out[b3 + 1u] = o3.y; attn_out[b3 + 2u] = o3.z; attn_out[b3 + 3u] = o3.w;
}
"#;

// ============================================================================
// 125: Batched RMSNorm / Residual + Subgroups (1 single barrier instead of 8!)
// Dispatch: (T, 1, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_RMSNORM_RESIDUAL_BATCH_SG: &str = r#"
enable subgroups;

struct G4NormParams {
  tokens: u32,
  dim: u32,
  norm_u32_offset: u32,
  in_offset: u32,
  resid_offset: u32,
  out_offset: u32,
  mode: u32,
  scalar_u32_offset: u32,
};

@group(0) @binding(0) var<uniform> params: G4NormParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read> in_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_resid: array<f32>;

var<workgroup> red_sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tok = wid.x;
  let tid = lid.x;
  let valid = tok < params.tokens;
  let in_base = params.in_offset + tok * params.dim;

  var ss: f32 = 0.0;
  if (valid) {
    for (var i: u32 = tid; i < params.dim; i = i + 256u) {
      let v = in_buf[in_base + i];
      ss += v * v;
    }
  }

  let sg_ss = subgroupAdd(ss);
  red_sh[tid] = select(0.0, sg_ss, sg_inv == 0u);
  workgroupBarrier();

  if (!valid) {
    return;
  }

  var total_ss: f32 = 0.0;
  let sg_step = max(sg_size, 1u);
  for (var s: u32 = 0u; s < 256u; s = s + sg_step) {
    total_ss += red_sh[s];
  }

  let inv_rms = inverseSqrt(total_ss / f32(params.dim) + 1e-6);
  let layer_scale = select(1.0, unpack2x16float(aux[params.scalar_u32_offset]).x, params.mode == 2u);
  let out_base = select(params.resid_offset, params.out_offset, params.mode == 0u) + tok * params.dim;

  for (var i: u32 = tid; i < params.dim; i = i + 256u) {
    let wp = unpack2x16float(aux[params.norm_u32_offset + (i >> 1u)]);
    let nw = select(wp.x, wp.y, (i & 1u) != 0u);
    let normed = in_buf[in_base + i] * inv_rms * nw;
    if (params.mode == 0u) {
      out_resid[out_base + i] = normed;
    } else if (params.mode == 1u) {
      out_resid[out_base + i] = out_resid[out_base + i] + normed;
    } else {
      out_resid[out_base + i] = (out_resid[out_base + i] + normed) * layer_scale;
    }
  }
}
"#;

// ============================================================================
// 126: Batched Pure-Q8 GEMV + Subgroups (zero post-loop reduction barriers!)
// Dispatch: (ceil(rows / 8), T, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q8_GEMV_BATCH_SG: &str = r#"
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

var<workgroup> x_sh: array<vec4<f32>, 640>;
var<workgroup> sumx_sh: array<f32, 128>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

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
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let row = wid.x * 8u + grp;
  let tok = wid.y;
  let safe_row = min(row, max(params.rows, 1u) - 1u);

  let u32_words = params.cols >> 2u;
  let x_tok_base = params.in_offset + tok * params.cols;
  let ones = vec4<f32>(1.0);
  var sx_loc: f32 = 0.0;
  for (var i: u32 = tid; i < u32_words; i = i + 128u) {
    let kb = x_tok_base + (i << 2u);
    let xv = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
    x_sh[i] = xv;
    sx_loc += dot(ones, xv);
  }
  let sg_sx = subgroupAdd(sx_loc);
  sumx_sh[tid] = select(0.0, sg_sx, sg_inv == 0u);
  workgroupBarrier();

  let sg_step = max(sg_size, 1u);
  var total_sum_x: f32 = 0.0;
  for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
    total_sum_x += sumx_sh[s];
  }

  let w_base = params.q_u32_offset + safe_row * u32_words;
  var acc_qx: f32 = 0.0;

  for (var w_rel: u32 = lane; w_rel < u32_words; w_rel = w_rel + 16u) {
    let p = weights[w_base + w_rel];
    let q = vec4<f32>(
      f32(p & 0xFFu),
      f32((p >> 8u) & 0xFFu),
      f32((p >> 16u) & 0xFFu),
      f32((p >> 24u) & 0xFFu),
    );
    acc_qx += dot(q, x_sh[w_rel]);
  }

  var sum2 = vec2<f32>(acc_qx, 0.0);
  if (sg_size >= 16u) {
    sum2 += subgroupShuffleXor(sum2, 8u);
    sum2 += subgroupShuffleXor(sum2, 4u);
    sum2 += subgroupShuffleXor(sum2, 2u);
    sum2 += subgroupShuffleXor(sum2, 1u);
  } else {
    red_sh[tid] = sum2;
    workgroupBarrier();
    if (lane == 0u) {
      var s_acc = vec2<f32>(0.0);
      let g_base = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        s_acc += red_sh[g_base + l];
      }
      sum2 = s_acc;
    }
  }

  if (lane == 0u && row < params.rows) {
    let sp = unpack2x16float(weights[params.s_u32_offset + (safe_row >> 1u)]);
    let row_scale = select(sp.x, sp.y, (safe_row & 1u) != 0u);
    var val: f32 = row_scale * (sum2.x - 128.0 * total_sum_x);
    if (params.mode != 0u) {
      val = gelu_tanh(val) * extra[params.extra_offset + tok * 10752u + row];
    }
    y[params.out_offset + tok * params.rows + row] = val;
  }
}
"#;

// ============================================================================
// 127: Fused Double-RMSNorm + Residual + LayerScalar + Subgroups
// mode 0: norm_io = RMSNorm(hidden, w0)
// mode 1: hidden += RMSNorm(norm_io, w0)
// mode 2: hidden = (hidden + RMSNorm(norm_io, w0)) * layer_scale
// mode 3: h_mid = hidden + RMSNorm(norm_io, w0); hidden = h_mid; norm_io = RMSNorm(h_mid, w1)
// mode 4: h_mid = (hidden + RMSNorm(norm_io, w0)) * layer_scale; hidden = h_mid; norm_io = RMSNorm(h_mid, w1)
// Dispatch: (T, 1, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_RMSNORM_FUSED_SG: &str = r#"
enable subgroups;

struct G4FusedNormParams {
  tokens: u32,
  dim: u32,
  w0_u32_off: u32,
  w1_u32_off: u32,
  hidden_offset: u32,
  norm_offset: u32,
  mode: u32,
  scalar_u32_off: u32,
};

@group(0) @binding(0) var<uniform> params: G4FusedNormParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_io: array<f32>;

var<workgroup> red_sh: array<f32, 256>;

fn read_norm_f16(u32_off: u32, idx: u32) -> f32 {
  let wp = unpack2x16float(aux[u32_off + (idx >> 1u)]);
  return select(wp.x, wp.y, (idx & 1u) != 0u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tok = wid.x;
  let tid = lid.x;
  let valid = tok < params.tokens;
  let h_base = params.hidden_offset + tok * params.dim;
  let n_base = params.norm_offset + tok * params.dim;
  let sg_step = max(sg_size, 1u);

  var reg: array<f32, 10>;
  var ss1: f32 = 0.0;
  for (var k: u32 = 0u; k < 10u; k = k + 1u) {
    let i = tid + k * 256u;
    var v: f32 = 0.0;
    if (valid && i < params.dim) {
      if (params.mode == 0u) {
        v = hidden[h_base + i];
      } else {
        v = norm_io[n_base + i];
      }
      ss1 += v * v;
    }
    reg[k] = v;
  }

  let sg_s1 = subgroupAdd(ss1);
  red_sh[tid] = select(0.0, sg_s1, sg_inv == 0u);
  workgroupBarrier();

  var tot1: f32 = 0.0;
  for (var s: u32 = 0u; s < 256u; s = s + sg_step) {
    tot1 += red_sh[s];
  }
  let inv_rms1 = inverseSqrt(tot1 / f32(params.dim) + 1e-6);

  if (params.mode == 0u) {
    if (valid) {
      for (var k: u32 = 0u; k < 10u; k = k + 1u) {
        let i = tid + k * 256u;
        if (i < params.dim) {
          norm_io[n_base + i] = reg[k] * inv_rms1 * read_norm_f16(params.w0_u32_off, i);
        }
      }
    }
    return;
  }

  let use_scale = (params.mode == 2u || params.mode == 4u);
  let layer_scale = select(1.0, unpack2x16float(aux[params.scalar_u32_off]).x, use_scale);
  var ss2: f32 = 0.0;
  for (var k: u32 = 0u; k < 10u; k = k + 1u) {
    let i = tid + k * 256u;
    if (valid && i < params.dim) {
      let normed = reg[k] * inv_rms1 * read_norm_f16(params.w0_u32_off, i);
      let h_new = (hidden[h_base + i] + normed) * layer_scale;
      hidden[h_base + i] = h_new;
      reg[k] = h_new;
      ss2 += h_new * h_new;
    }
  }

  if (params.mode < 3u) {
    return;
  }

  let sg_s2 = subgroupAdd(ss2);
  red_sh[tid] = select(0.0, sg_s2, sg_inv == 0u);
  workgroupBarrier();

  if (!valid) {
    return;
  }

  var tot2: f32 = 0.0;
  for (var s: u32 = 0u; s < 256u; s = s + sg_step) {
    tot2 += red_sh[s];
  }
  let inv_rms2 = inverseSqrt(tot2 / f32(params.dim) + 1e-6);

  for (var k: u32 = 0u; k < 10u; k = k + 1u) {
    let i = tid + k * 256u;
    if (i < params.dim) {
      norm_io[n_base + i] = reg[k] * inv_rms2 * read_norm_f16(params.w1_u32_off, i);
    }
  }
}
"#;

// ============================================================================
// 128: Subgroup Fused QKV Norm + RoPE + KV Cache Write (reads packed qkvBuf!)
// Dispatch: (is_shared != 0 ? 8 : 12, T, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_QKV_NORM_ROPE_KVWRITE_SG: &str = r#"
enable subgroups;

struct G4RopeParams {
  tokens: u32,
  seq_pos: u32,
  head_dim: u32,
  num_q_heads: u32,
  num_kv_heads: u32,
  is_full: u32,
  is_shared: u32,
  kv_mode: u32,
  local_donor_idx: u32,
  max_seq: u32,
  q_norm_u32_off: u32,
  k_norm_u32_off: u32,
};

@group(0) @binding(0) var<uniform> params: G4RopeParams;
@group(0) @binding(1) var<storage, read> aux: array<u32>;
@group(0) @binding(2) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(3) var<storage, read> raw_qkv: array<f32>;
@group(0) @binding(4) var<storage, read_write> kv_cache: array<u32>;

var<workgroup> red_sh: array<f32, 256>;
var<workgroup> tmp_sh: array<f32, 512>;

fn pack4_i8_vec(v: vec4<f32>) -> u32 {
  let b0 = u32(i32(clamp(round(v.x), -127.0, 127.0))) & 0xFFu;
  let b1 = u32(i32(clamp(round(v.y), -127.0, 127.0))) & 0xFFu;
  let b2 = u32(i32(clamp(round(v.z), -127.0, 127.0))) & 0xFFu;
  let b3 = u32(i32(clamp(round(v.w), -127.0, 127.0))) & 0xFFu;
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let h_idx = wid.x;
  let tok = wid.y;
  let i = lid.x;
  let half = params.head_dim >> 1u;
  let pos = params.seq_pos + tok;
  let sg_step = max(sg_size, 1u);

  let is_q = h_idx < params.num_q_heads;
  let is_k = h_idx >= params.num_q_heads && h_idx < (params.num_q_heads + params.num_kv_heads);
  let kvh = select(h_idx - params.num_q_heads - params.num_kv_heads, h_idx - params.num_q_heads, is_k);

  let q_total = params.num_q_heads * params.head_dim;
  let kv_total = params.num_kv_heads * params.head_dim;
  let qkv_stride = select(q_total, q_total + 2u * kv_total, params.is_shared == 0u);
  let tok_off = tok * qkv_stride;

  var x0: f32 = 0.0;
  var x1: f32 = 0.0;
  if (i < half && tok < params.tokens) {
    var src_base: u32;
    if (is_q) {
      src_base = tok_off + h_idx * params.head_dim;
    } else if (is_k) {
      src_base = tok_off + q_total + kvh * params.head_dim;
    } else {
      src_base = tok_off + q_total + kv_total + kvh * params.head_dim;
    }
    x0 = raw_qkv[src_base + i];
    x1 = raw_qkv[src_base + i + half];
  }

  let sg_ss = subgroupAdd(x0 * x0 + x1 * x1);
  red_sh[i] = select(0.0, sg_ss, sg_inv == 0u);
  workgroupBarrier();

  var tot_ss: f32 = 0.0;
  for (var s: u32 = 0u; s < 256u; s = s + sg_step) {
    tot_ss += red_sh[s];
  }
  let inv_rms = inverseSqrt(tot_ss / f32(params.head_dim) + 1e-6);

  var w0: f32 = 1.0;
  var w1: f32 = 1.0;
  if (i < half && (is_q || is_k)) {
    let n_off = select(params.k_norm_u32_off, params.q_norm_u32_off, is_q);
    let idx0 = i;
    let wp0 = unpack2x16float(aux[n_off + (idx0 >> 1u)]);
    w0 = select(wp0.x, wp0.y, (idx0 & 1u) != 0u);
    let idx1 = i + half;
    let wp1 = unpack2x16float(aux[n_off + (idx1 >> 1u)]);
    w1 = select(wp1.x, wp1.y, (idx1 & 1u) != 0u);
  }

  let nx0 = x0 * inv_rms * w0;
  let nx1 = x1 * inv_rms * w1;

  var out0: f32 = nx0;
  var out1: f32 = nx1;
  if (is_q || is_k) {
    let rot_pairs = select(128u, 64u, params.is_full != 0u);
    if (i < rot_pairs) {
      let theta = select(10000.0, 1000000.0, params.is_full != 0u);
      let inv_freq = 1.0 / pow(theta, f32(2u * i) / f32(params.head_dim));
      let ang = f32(pos) * inv_freq;
      let c = cos(ang);
      let sn = sin(ang);
      out0 = nx0 * c - nx1 * sn;
      out1 = nx0 * sn + nx1 * c;
    }
  }

  if (is_q) {
    if (i < half && tok < params.tokens) {
      let base = (tok * params.num_q_heads + h_idx) * params.head_dim;
      q_out[base + i] = out0;
      q_out[base + i + half] = out1;
    }
    return;
  }

  if (i < half) {
    tmp_sh[i] = out0;
    tmp_sh[i + half] = out1;
  }
  workgroupBarrier();

  if (tok >= params.tokens || params.is_shared != 0u) {
    return;
  }

  let plane_idx = select(0u, 1u, !is_k);

  if (params.is_full == 0u) {
    let head_u32s = 1024u * 128u;
    let donor_u32s = 4u * head_u32s;
    let dst_base = params.local_donor_idx * donor_u32s
      + (plane_idx * 2u + kvh) * head_u32s
      + (pos & 1023u) * 128u;
    if (i < 128u) {
      kv_cache[dst_base + i] = pack2x16float(vec2<f32>(tmp_sh[2u * i], tmp_sh[2u * i + 1u]));
    }
    return;
  }

  if (params.kv_mode == 0u) {
    let head_u32s = params.max_seq * 256u;
    let donor_u32s = 4u * head_u32s;
    let dst_base = params.local_donor_idx * donor_u32s
      + (plane_idx * 2u + kvh) * head_u32s
      + pos * 256u;
    kv_cache[dst_base + i] = pack2x16float(vec2<f32>(tmp_sh[2u * i], tmp_sh[2u * i + 1u]));
    return;
  }

  if (params.kv_mode == 2u) {
    let grp = i >> 3u;
    let head_u32s = params.max_seq * 68u;
    let donor_u32s = 4u * head_u32s;
    let dst_base = params.local_donor_idx * donor_u32s + (plane_idx * 2u + kvh) * head_u32s + pos * 68u;
    if (i < 64u) {
      let b = i * 8u;
      var max64: f32 = 0.0;
      for (var k: u32 = grp * 64u; k < (grp + 1u) * 64u; k = k + 1u) {
        max64 = max(max64, abs(tmp_sh[k]));
      }
      let s_g = max(max64 / 7.0, 1e-6);
      var pw: u32 = 0u;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let q = u32(clamp(i32(round(tmp_sh[b + k] / s_g)) + 8, 1, 15));
        pw = pw | (q << (k * 4u));
      }
      kv_cache[dst_base + i] = pw;
    }
    if (i < 4u) {
      let g0 = i * 2u;
      let g1 = g0 + 1u;
      var m0: f32 = 0.0;
      var m1: f32 = 0.0;
      for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        m0 = max(m0, abs(tmp_sh[g0 * 64u + k]));
        m1 = max(m1, abs(tmp_sh[g1 * 64u + k]));
      }
      kv_cache[dst_base + 64u + i] = pack2x16float(vec2<f32>(max(m0 / 7.0, 1e-6), max(m1 / 7.0, 1e-6)));
    }
    return;
  }

  let ma = max(abs(tmp_sh[i]), abs(tmp_sh[i + 256u]));
  let sg_ma = subgroupMax(ma);
  red_sh[i] = select(0.0, sg_ma, sg_inv == 0u);
  workgroupBarrier();

  var max_abs: f32 = 0.0;
  for (var s: u32 = 0u; s < 256u; s = s + sg_step) {
    max_abs = max(max_abs, red_sh[s]);
  }
  let scale = max(max_abs / 127.0, 1e-6);
  let head_u32s = params.max_seq * 129u;
  let donor_u32s = 4u * head_u32s;
  let dst_base = params.local_donor_idx * donor_u32s
    + (plane_idx * 2u + kvh) * head_u32s
    + pos * 129u;
  if (i < 128u) {
    let b = i * 4u;
    let v4 = vec4<f32>(tmp_sh[b], tmp_sh[b + 1u], tmp_sh[b + 2u], tmp_sh[b + 3u]) / scale;
    kv_cache[dst_base + i] = pack4_i8_vec(v4);
  }
  if (i == 0u) {
    kv_cache[dst_base + 128u] = bitcast<u32>(scale);
  }
}
"#;

// ============================================================================
// 129: Small-Batch Pure-Q4 GEMV (M = 1..8 speculative tokens, 1 weight read!)
// Dispatch: (ceil(rows / 16), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GEMV_BATCH_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  num_tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 256>; // up to 8 tokens * 32 vec4<f16> (16 u32s = 128 cols)
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn q4_lo(p: u32) -> vec4<f16> {
  return vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
}

fn q4_hi(p: u32) -> vec4<f16> {
  return vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
}

fn reduce16_vec4(v_in: vec4<f32>, tid: u32, grp: u32, lane: u32, sg_size: u32) -> vec4<f32> {
  var v = v_in;
  if (sg_size >= 16u) {
    v += subgroupShuffleXor(v, 8u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
  }
  red_sh[tid] = v;
  workgroupBarrier();
  var out_v = vec4<f32>(0.0);
  if (lane == 0u) {
    let base = grp * 16u;
    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
      out_v += red_sh[base + l];
    }
  }
  workgroupBarrier();
  return out_v;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let M = min(params.num_tokens, 8u);
  if (params.rows == 0u || M == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, max_r);
  let r1 = min(rB, max_r);

  let u32_words = params.cols >> 3u;
  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;

  var acc0_lo = vec4<f32>(0.0);
  var acc0_hi = vec4<f32>(0.0);
  var acc1_lo = vec4<f32>(0.0);
  var acc1_hi = vec4<f32>(0.0);
  var sx_lo = vec4<f32>(0.0);
  var sx_hi = vec4<f32>(0.0);
  let ones = vec4<f16>(1.0h);

  for (var k_tile: u32 = 0u; k_tile < u32_words; k_tile = k_tile + 16u) {
    let tile_u32s = min(16u, u32_words - k_tile);
    let tile_vec4s = tile_u32s << 1u; // <= 32
    let x_base = params.in_offset + (k_tile << 3u);
    if (tid < tile_vec4s) {
      for (var t: u32 = 0u; t < M; t = t + 1u) {
        let kb = x_base + t * params.cols + (tid << 2u);
        x_sh[t * 32u + tid] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      }
    }
    workgroupBarrier();

    if (lane < tile_u32s) {
      let gw = k_tile + lane;
      let p0 = weights[o0 + gw];
      let p1 = weights[o1 + gw];
      let q0a = q4_lo(p0); let q0b = q4_hi(p0);
      let q1a = q4_lo(p1); let q1b = q4_hi(p1);
      let xb = lane << 1u;

      {
        let xa = x_sh[0u * 32u + xb]; let xb_v = x_sh[0u * 32u + xb + 1u];
        sx_lo.x += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_lo.x += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_lo.x += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 1u) {
        let xa = x_sh[1u * 32u + xb]; let xb_v = x_sh[1u * 32u + xb + 1u];
        sx_lo.y += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_lo.y += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_lo.y += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 2u) {
        let xa = x_sh[2u * 32u + xb]; let xb_v = x_sh[2u * 32u + xb + 1u];
        sx_lo.z += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_lo.z += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_lo.z += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 3u) {
        let xa = x_sh[3u * 32u + xb]; let xb_v = x_sh[3u * 32u + xb + 1u];
        sx_lo.w += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_lo.w += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_lo.w += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 4u) {
        let xa = x_sh[4u * 32u + xb]; let xb_v = x_sh[4u * 32u + xb + 1u];
        sx_hi.x += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_hi.x += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_hi.x += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 5u) {
        let xa = x_sh[5u * 32u + xb]; let xb_v = x_sh[5u * 32u + xb + 1u];
        sx_hi.y += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_hi.y += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_hi.y += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 6u) {
        let xa = x_sh[6u * 32u + xb]; let xb_v = x_sh[6u * 32u + xb + 1u];
        sx_hi.z += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_hi.z += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_hi.z += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
      if (M > 7u) {
        let xa = x_sh[7u * 32u + xb]; let xb_v = x_sh[7u * 32u + xb + 1u];
        sx_hi.w += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        acc0_hi.w += f32(dot(q0a, xa)) + f32(dot(q0b, xb_v));
        acc1_hi.w += f32(dot(q1a, xa)) + f32(dot(q1b, xb_v));
      }
    }

    if (k_tile + 16u < u32_words) {
      workgroupBarrier();
    }
  }

  let r0_lo = reduce16_vec4(acc0_lo, tid, grp, lane, sg_size);
  let r0_hi = reduce16_vec4(acc0_hi, tid, grp, lane, sg_size);
  let r1_lo = reduce16_vec4(acc1_lo, tid, grp, lane, sg_size);
  let r1_hi = reduce16_vec4(acc1_hi, tid, grp, lane, sg_size);
  let rsx_lo = reduce16_vec4(sx_lo, tid, grp, lane, sg_size);
  let rsx_hi = reduce16_vec4(sx_hi, tid, grp, lane, sg_size);

  if (lane == 0u) {
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    let v0_lo = s0 * (r0_lo - 8.0 * rsx_lo);
    let v0_hi = s0 * (r0_hi - 8.0 * rsx_hi);
    let v1_lo = s1 * (r1_lo - 8.0 * rsx_lo);
    let v1_hi = s1 * (r1_hi - 8.0 * rsx_hi);

    for (var t: u32 = 0u; t < M; t = t + 1u) {
      let c0 = select(v0_hi[t - 4u], v0_lo[t], t < 4u);
      let c1 = select(v1_hi[t - 4u], v1_lo[t], t < 4u);
      if (rA < params.rows) {
        let out0 = params.out_offset + t * params.rows + rA;
        y[out0] = select(c0, y[out0] + c0, params.mode != 0u);
      }
      if (rB < params.rows) {
        let out1 = params.out_offset + t * params.rows + rB;
        y[out1] = select(c1, y[out1] + c1, params.mode != 0u);
      }
    }
  }
}
"#;

// ============================================================================
// 130: Small-Batch Fused Q4 gate_up_proj + GeGLU (M = 1..8 speculative tokens)
// Dispatch: (ceil(rows / 8), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q4_GATE_UP_GEGLU_BATCH_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  num_tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 256>;
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn q4_lo(p: u32) -> vec4<f16> {
  return vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
}

fn q4_hi(p: u32) -> vec4<f16> {
  return vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
}

fn gelu_tanh(v: f32) -> f32 {
  let arg = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(arg));
}

fn reduce16_vec4(v_in: vec4<f32>, tid: u32, grp: u32, lane: u32, sg_size: u32) -> vec4<f32> {
  var v = v_in;
  if (sg_size >= 16u) {
    v += subgroupShuffleXor(v, 8u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
  }
  red_sh[tid] = v;
  workgroupBarrier();
  var out_v = vec4<f32>(0.0);
  if (lane == 0u) {
    let base = grp * 16u;
    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
      out_v += red_sh[base + l];
    }
  }
  workgroupBarrier();
  return out_v;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let M = min(params.num_tokens, 8u);
  if (params.rows == 0u || M == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 8u + grp;
  let rg = min(rA, max_r);
  let ru = rg + params.rows;

  let u32_words = params.cols >> 3u;
  let og = params.q_u32_offset + rg * u32_words;
  let ou = params.q_u32_offset + ru * u32_words;

  var accg_lo = vec4<f32>(0.0);
  var accg_hi = vec4<f32>(0.0);
  var accu_lo = vec4<f32>(0.0);
  var accu_hi = vec4<f32>(0.0);
  var sx_lo = vec4<f32>(0.0);
  var sx_hi = vec4<f32>(0.0);
  let ones = vec4<f16>(1.0h);

  for (var k_tile: u32 = 0u; k_tile < u32_words; k_tile = k_tile + 16u) {
    let tile_u32s = min(16u, u32_words - k_tile);
    let tile_vec4s = tile_u32s << 1u;
    let x_base = params.in_offset + (k_tile << 3u);
    if (tid < tile_vec4s) {
      for (var t: u32 = 0u; t < M; t = t + 1u) {
        let kb = x_base + t * params.cols + (tid << 2u);
        x_sh[t * 32u + tid] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      }
    }
    workgroupBarrier();

    if (lane < tile_u32s) {
      let gw = k_tile + lane;
      let pg = weights[og + gw];
      let pu = weights[ou + gw];
      let qga = q4_lo(pg); let qgb = q4_hi(pg);
      let qua = q4_lo(pu); let qub = q4_hi(pu);
      let xb = lane << 1u;

      {
        let xa = x_sh[0u * 32u + xb]; let xb_v = x_sh[0u * 32u + xb + 1u];
        sx_lo.x += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_lo.x += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_lo.x += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 1u) {
        let xa = x_sh[1u * 32u + xb]; let xb_v = x_sh[1u * 32u + xb + 1u];
        sx_lo.y += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_lo.y += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_lo.y += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 2u) {
        let xa = x_sh[2u * 32u + xb]; let xb_v = x_sh[2u * 32u + xb + 1u];
        sx_lo.z += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_lo.z += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_lo.z += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 3u) {
        let xa = x_sh[3u * 32u + xb]; let xb_v = x_sh[3u * 32u + xb + 1u];
        sx_lo.w += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_lo.w += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_lo.w += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 4u) {
        let xa = x_sh[4u * 32u + xb]; let xb_v = x_sh[4u * 32u + xb + 1u];
        sx_hi.x += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_hi.x += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_hi.x += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 5u) {
        let xa = x_sh[5u * 32u + xb]; let xb_v = x_sh[5u * 32u + xb + 1u];
        sx_hi.y += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_hi.y += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_hi.y += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 6u) {
        let xa = x_sh[6u * 32u + xb]; let xb_v = x_sh[6u * 32u + xb + 1u];
        sx_hi.z += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_hi.z += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_hi.z += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
      if (M > 7u) {
        let xa = x_sh[7u * 32u + xb]; let xb_v = x_sh[7u * 32u + xb + 1u];
        sx_hi.w += f32(dot(ones, xa)) + f32(dot(ones, xb_v));
        accg_hi.w += f32(dot(qga, xa)) + f32(dot(qgb, xb_v));
        accu_hi.w += f32(dot(qua, xa)) + f32(dot(qub, xb_v));
      }
    }

    if (k_tile + 16u < u32_words) {
      workgroupBarrier();
    }
  }

  let rg_lo = reduce16_vec4(accg_lo, tid, grp, lane, sg_size);
  let rg_hi = reduce16_vec4(accg_hi, tid, grp, lane, sg_size);
  let ru_lo = reduce16_vec4(accu_lo, tid, grp, lane, sg_size);
  let ru_hi = reduce16_vec4(accu_hi, tid, grp, lane, sg_size);
  let rsx_lo = reduce16_vec4(sx_lo, tid, grp, lane, sg_size);
  let rsx_hi = reduce16_vec4(sx_hi, tid, grp, lane, sg_size);

  if (lane == 0u && rA < params.rows) {
    let spg = unpack2x16float(weights[params.s_u32_offset + (rg >> 1u)]);
    let sg = select(spg.x, spg.y, (rg & 1u) != 0u);
    let spu = unpack2x16float(weights[params.s_u32_offset + (ru >> 1u)]);
    let su = select(spu.x, spu.y, (ru & 1u) != 0u);

    let g_lo = sg * (rg_lo - 8.0 * rsx_lo);
    let g_hi = sg * (rg_hi - 8.0 * rsx_hi);
    let u_lo = su * (ru_lo - 8.0 * rsx_lo);
    let u_hi = su * (ru_hi - 8.0 * rsx_hi);

    for (var t: u32 = 0u; t < M; t = t + 1u) {
      let gv = select(g_hi[t - 4u], g_lo[t], t < 4u);
      let uv = select(u_hi[t - 4u], u_lo[t], t < 4u);
      y[params.out_offset + t * params.rows + rA] = gelu_tanh(gv) * uv;
    }
  }
}
"#;

// ============================================================================
// 131: Small-Batch Pure-Q2 LM Head (M = 1..8 speculative tokens, 1 weight read!)
// Reads the 168 MB LM head ONCE for all M tokens!
// Dispatch: (ceil(rows / 16), 1, 1) @workgroup_size(128)
// ============================================================================
const WGSL_G4_Q2_LM_HEAD_BATCH_F16: &str = r#"
enable f16;
enable subgroups;

struct G4GemvParams {
  rows: u32,
  cols: u32,
  q_u32_offset: u32,
  s_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  mode: u32,
  num_tokens: u32,
};

@group(0) @binding(0) var<uniform> params: G4GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 512>; // 8 tokens * 64 vec4<f16> (16 Q2 u32s = 256 cols)
var<workgroup> red_sh: array<vec4<f32>, 128>;

fn factored_q2_word_f16(p: u32, x0: vec4<f16>, x1: vec4<f16>, x2: vec4<f16>, x3: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 3u), f16((p >> 2u) & 3u), f16((p >> 4u) & 3u), f16((p >> 6u) & 3u));
  let q1 = vec4<f16>(f16((p >> 8u) & 3u), f16((p >> 10u) & 3u), f16((p >> 12u) & 3u), f16((p >> 14u) & 3u));
  let q2 = vec4<f16>(f16((p >> 16u) & 3u), f16((p >> 18u) & 3u), f16((p >> 20u) & 3u), f16((p >> 22u) & 3u));
  let q3 = vec4<f16>(f16((p >> 24u) & 3u), f16((p >> 26u) & 3u), f16((p >> 28u) & 3u), f16((p >> 30u) & 3u));
  return f32(dot(q0, x0)) + f32(dot(q1, x1)) + f32(dot(q2, x2)) + f32(dot(q3, x3));
}

fn reduce16_vec4(v_in: vec4<f32>, tid: u32, grp: u32, lane: u32, sg_size: u32) -> vec4<f32> {
  var v = v_in;
  if (sg_size >= 16u) {
    v += subgroupShuffleXor(v, 8u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
  }
  red_sh[tid] = v;
  workgroupBarrier();
  var out_v = vec4<f32>(0.0);
  if (lane == 0u) {
    let base = grp * 16u;
    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
      out_v += red_sh[base + l];
    }
  }
  workgroupBarrier();
  return out_v;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  let M = min(params.num_tokens, 8u);
  if (params.rows == 0u || M == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let max_r = max(params.rows, 1u) - 1u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, max_r);
  let r1 = min(rB, max_r);

  let u32_words = params.cols >> 4u; // 160 u32s
  let o0 = params.q_u32_offset + r0 * u32_words;
  let o1 = params.q_u32_offset + r1 * u32_words;

  var acc0_lo = vec4<f32>(0.0);
  var acc0_hi = vec4<f32>(0.0);
  var acc1_lo = vec4<f32>(0.0);
  var acc1_hi = vec4<f32>(0.0);
  var sx_lo = vec4<f32>(0.0);
  var sx_hi = vec4<f32>(0.0);
  let ones = vec4<f16>(1.0h);

  for (var w_tile: u32 = 0u; w_tile < u32_words; w_tile = w_tile + 16u) {
    let tile_u32s = min(16u, u32_words - w_tile);
    let tile_vec4s = tile_u32s << 2u; // <= 64
    let x_base = params.in_offset + (w_tile << 4u);
    if (tid < tile_vec4s) {
      for (var t: u32 = 0u; t < M; t = t + 1u) {
        let kb = x_base + t * params.cols + (tid << 2u);
        x_sh[t * 64u + tid] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      }
    }
    workgroupBarrier();

    if (lane < tile_u32s) {
      let gw = w_tile + lane;
      let p0 = weights[o0 + gw];
      let p1 = weights[o1 + gw];
      let xb = lane << 2u;

      {
        let b = 0u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_lo.x += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_lo.x += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_lo.x += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 1u) {
        let b = 1u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_lo.y += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_lo.y += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_lo.y += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 2u) {
        let b = 2u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_lo.z += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_lo.z += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_lo.z += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 3u) {
        let b = 3u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_lo.w += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_lo.w += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_lo.w += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 4u) {
        let b = 4u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_hi.x += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_hi.x += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_hi.x += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 5u) {
        let b = 5u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_hi.y += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_hi.y += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_hi.y += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 6u) {
        let b = 6u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_hi.z += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_hi.z += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_hi.z += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
      if (M > 7u) {
        let b = 7u * 64u + xb;
        let x0 = x_sh[b]; let x1 = x_sh[b + 1u]; let x2 = x_sh[b + 2u]; let x3 = x_sh[b + 3u];
        sx_hi.w += f32(dot(ones, x0)) + f32(dot(ones, x1)) + f32(dot(ones, x2)) + f32(dot(ones, x3));
        acc0_hi.w += factored_q2_word_f16(p0, x0, x1, x2, x3);
        acc1_hi.w += factored_q2_word_f16(p1, x0, x1, x2, x3);
      }
    }

    if (w_tile + 16u < u32_words) {
      workgroupBarrier();
    }
  }

  let r0_lo = reduce16_vec4(acc0_lo, tid, grp, lane, sg_size);
  let r0_hi = reduce16_vec4(acc0_hi, tid, grp, lane, sg_size);
  let r1_lo = reduce16_vec4(acc1_lo, tid, grp, lane, sg_size);
  let r1_hi = reduce16_vec4(acc1_hi, tid, grp, lane, sg_size);
  let rsx_lo = reduce16_vec4(sx_lo, tid, grp, lane, sg_size);
  let rsx_hi = reduce16_vec4(sx_hi, tid, grp, lane, sg_size);

  if (lane == 0u) {
    let sp0 = unpack2x16float(weights[params.s_u32_offset + (r0 >> 1u)]);
    let s0 = select(sp0.x, sp0.y, (r0 & 1u) != 0u);
    let sp1 = unpack2x16float(weights[params.s_u32_offset + (r1 >> 1u)]);
    let s1 = select(sp1.x, sp1.y, (r1 & 1u) != 0u);

    var v0_lo = s0 * (r0_lo - 2.0 * rsx_lo);
    var v0_hi = s0 * (r0_hi - 2.0 * rsx_hi);
    var v1_lo = s1 * (r1_lo - 2.0 * rsx_lo);
    var v1_hi = s1 * (r1_hi - 2.0 * rsx_hi);
    if (params.mode != 0u) {
      let lo15 = vec4<f32>(-15.0);
      let hi15 = vec4<f32>(15.0);
      v0_lo = 30.0 * tanh(clamp(v0_lo / 30.0, lo15, hi15));
      v0_hi = 30.0 * tanh(clamp(v0_hi / 30.0, lo15, hi15));
      v1_lo = 30.0 * tanh(clamp(v1_lo / 30.0, lo15, hi15));
      v1_hi = 30.0 * tanh(clamp(v1_hi / 30.0, lo15, hi15));
    }

    for (var t: u32 = 0u; t < M; t = t + 1u) {
      let c0 = select(v0_hi[t - 4u], v0_lo[t], t < 4u);
      let c1 = select(v1_hi[t - 4u], v1_lo[t], t < 4u);
      let tok_out = params.out_offset + t * params.rows;
      if (rA < params.rows) { y[tok_out + rA] = c0; }
      if (rB < params.rows) { y[tok_out + rB] = c1; }
    }
  }
}
"#;

// ============================================================================
// 132: Batched Argmax Pass 1 (M = 1..8 rows in parallel!)
// Dispatch: (ceil(len / 256), M, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_ARGMAX_BATCH_PASS1: &str = r#"
struct ArgmaxBatchParams {
  len: u32,
  num_tokens: u32,
  blocks_per_tok: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: ArgmaxBatchParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials_val: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials_idx: array<u32>;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let tok = wid.y;
  let idx = wid.x * 256u + tid;
  if (tok < params.num_tokens && idx < params.len) {
    best_v[tid] = logits[tok * params.len + idx];
    best_i[tid] = idx;
  } else {
    best_v[tid] = -1e30;
    best_i[tid] = 0u;
  }
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      if (best_v[tid + s] > best_v[tid]) {
        best_v[tid] = best_v[tid + s];
        best_i[tid] = best_i[tid + s];
      }
    }
    workgroupBarrier();
  }

  if (tid == 0u && tok < params.num_tokens) {
    let out_i = tok * params.blocks_per_tok + wid.x;
    partials_val[out_i] = best_v[0];
    partials_idx[out_i] = best_i[0];
  }
}
"#;

// ============================================================================
// 133: Batched Argmax Pass 2 (M = 1..8 output tokens -> argmaxMultiOutBuf)
// Dispatch: (M, 1, 1) @workgroup_size(256)
// ============================================================================
const WGSL_G4_ARGMAX_BATCH_PASS2: &str = r#"
struct ArgmaxBatchParams {
  len: u32,
  num_tokens: u32,
  blocks_per_tok: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: ArgmaxBatchParams;
@group(0) @binding(1) var<storage, read> partials_val: array<f32>;
@group(0) @binding(2) var<storage, read> partials_idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_tokens: array<u32>;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tok = wid.x;
  let tid = lid.x;
  let base = tok * params.blocks_per_tok;
  var local_v: f32 = -1e30;
  var local_i: u32 = 0u;
  if (tok < params.num_tokens) {
    for (var i: u32 = tid; i < params.blocks_per_tok; i = i + 256u) {
      let v = partials_val[base + i];
      if (v > local_v) {
        local_v = v;
        local_i = partials_idx[base + i];
      }
    }
  }
  best_v[tid] = local_v;
  best_i[tid] = local_i;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      if (best_v[tid + s] > best_v[tid]) {
        best_v[tid] = best_v[tid + s];
        best_i[tid] = best_i[tid + s];
      }
    }
    workgroupBarrier();
  }

  if (tid == 0u && tok < params.num_tokens) {
    out_tokens[tok] = best_i[0];
  }
}
"#;
