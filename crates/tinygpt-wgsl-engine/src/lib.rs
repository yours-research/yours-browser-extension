#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Qwen3.5-4B Architectural Constants
pub const HIDDEN_SIZE: u32 = 2560;
pub const INTERMEDIATE_SIZE: u32 = 9216;
pub const NUM_LAYERS: u32 = 32;
pub const FULL_ATTN_INTERVAL: u32 = 4; // 8 full_attention, 24 linear_attention
pub const VOCAB_SIZE: u32 = 248320;

// GQA Full Attention
pub const ATTN_HEADS: u32 = 16;
pub const ATTN_KV_HEADS: u32 = 4;
pub const ATTN_HEAD_DIM: u32 = 256;

// Gated DeltaNet Linear Attention
pub const DN_KEY_HEADS: u32 = 16;
pub const DN_KEY_DIM: u32 = 128; // Q,K dim = 2048
pub const DN_VAL_HEADS: u32 = 32;
pub const DN_VAL_DIM: u32 = 128; // V dim = 4096
pub const DN_CONV_KERNEL: u32 = 4;
pub const DN_CONV_DIM: u32 = 8192; // 2048 + 2048 + 4096

// Block-32 Q4 packed layout: 5 u32s (20 bytes) per 32 weights
// u32[0..3]: 32 x 4-bit unsigned nibbles (8 nibbles per u32)
// u32[4]:    packed 2x f16 (low 16 bits = scale, high 16 bits = zero)
pub const Q4_BLOCK_WEIGHTS: u32 = 32;
pub const Q4_BLOCK_U32S: u32 = 5;

const SHADER_BUF_CAP: usize = 65536;
const HOST_SCRATCH_CAP: usize = 4 * 1024 * 1024;

static mut SHADER_BUF: [u8; SHADER_BUF_CAP] = [0; SHADER_BUF_CAP];
static mut SHADER_LEN: usize = 0;

fn set_shader_bytes(bytes: &[u8]) -> usize {
    let len = bytes.len().min(SHADER_BUF_CAP);
    unsafe {
        let dst = core::ptr::addr_of_mut!(SHADER_BUF) as *mut u8;
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, len);
        SHADER_LEN = len;
    }
    len
}

#[no_mangle]
pub extern "C" fn wgsl_get_shader_ptr() -> *const u8 {
    core::ptr::addr_of!(SHADER_BUF) as *const u8
}

#[no_mangle]
pub extern "C" fn wgsl_get_shader_len() -> usize {
    unsafe { SHADER_LEN }
}

/// Allocates memory inside WASM linear memory for JS host interop.
static mut HOST_SCRATCH: [u8; HOST_SCRATCH_CAP] = [0; HOST_SCRATCH_CAP];

#[no_mangle]
pub extern "C" fn wasm_scratch_ptr() -> *mut u8 {
    core::ptr::addr_of_mut!(HOST_SCRATCH) as *mut u8
}

#[no_mangle]
pub extern "C" fn wasm_scratch_len() -> usize {
    HOST_SCRATCH_CAP
}

#[no_mangle]
pub extern "C" fn q4_row_u32s(k: u32) -> u32 {
    (k / Q4_BLOCK_WEIGHTS) * Q4_BLOCK_U32S
}

#[no_mangle]
pub extern "C" fn q4_matrix_bytes(rows: u32, cols: u32) -> u64 {
    (rows as u64) * (q4_row_u32s(cols) as u64) * 4
}

#[no_mangle]
pub extern "C" fn deltanet_layer_state_bytes() -> u32 {
    // conv [8192, 4] f32 + recurrent [32, 128, 128] f32
    (DN_CONV_DIM * DN_CONV_KERNEL + DN_VAL_HEADS * DN_KEY_DIM * DN_VAL_DIM) * 4
}

#[no_mangle]
pub extern "C" fn full_attn_layer_kv_bytes(max_seq: u32, int8_kv: u32) -> u64 {
    if int8_kv != 0 {
        // 2 planes (K + V) * 4 heads * max_seq * 65 u32s/token-head (256 i8 + 1 packed scale u32) * 4 bytes
        2u64 * (ATTN_KV_HEADS as u64) * (max_seq as u64) * 65u64 * 4u64
    } else {
        // 2 tensors (K + V) * 4 heads * max_seq * 256 f16 dims * 2 bytes
        2u64 * (ATTN_KV_HEADS as u64) * (max_seq as u64) * (ATTN_HEAD_DIM as u64) * 2u64
    }
}

/// Shader IDs:
/// 0 = q4-scalar
/// 1 = q4-vec4-f16
/// 2 = q4-subgroups-f16
/// 3 = deltanet_step
/// 4 = gqa_fp16_split
/// 5 = gqa_int8_split
/// 6 = rmsnorm_swiglu
/// 7 = argmax_pass1
/// 8 = argmax_pass2
/// 9 = gqa_reduce
/// 10 = embed_lookup
/// 11 = rmsnorm_batch
/// 12 = swiglu_batch
/// 13 = q4_gemm_f16
/// 14 = gdn_conv1d
/// 15 = gdn_recurrent
/// 16 = qk_norm_rope
/// 17 = gqa_fp16_prefill
/// 18 = gqa_int8_prefill
/// 19 = gdn_chunk_prep
/// 20 = gdn_chunk_apply
/// 21 = q4_gate_up_swiglu
/// 22 = q4_gemv_wide_k
/// 23 = gqa_int8_split_dp4a
/// 24 = q4_gemv_batch
/// 25 = q4_gemv_t2_f16
/// 26 = q4_gemv_t3_f32
/// 27 = q4_gate_up_swiglu_t2_f16
/// 28 = q4_gate_up_swiglu_t3_f32
/// 29 = q4_gemv_wide_k_t2_f16
/// 30 = q4_gemv_wide_k_t3_f32
/// 31 = q4_gemv_batch_t2_f16
/// 32 = q4_gemv_batch_t3_f32
/// 33 = q4_gemm_t3_f32
/// 34 = rmsnorm_batch_t2
/// 35 = gqa_fp16_split_t2_f16
/// 36 = gqa_fp16_split_t3_f32
/// 37 = gqa_int8_split_t2
/// 38 = gqa_fp16_prefill_t2_f16
/// 39 = gqa_fp16_prefill_t3_f32
/// 40 = gqa_int8_prefill_t2
/// 41 = gdn_chunk_prep_t2
/// 42 = gdn_chunk_apply_t2
#[no_mangle]
pub extern "C" fn wgsl_build_shader(shader_id: u32) -> usize {
    match shader_id {
        0 => set_shader_bytes(WGSL_Q4_SCALAR.as_bytes()),
        1 => set_shader_bytes(WGSL_Q4_VEC4_F16.as_bytes()),
        2 => set_shader_bytes(WGSL_Q4_SUBGROUPS_F16.as_bytes()),
        3 => set_shader_bytes(WGSL_DELTANET_STEP.as_bytes()),
        4 => set_shader_bytes(WGSL_GQA_FP16_SPLIT.as_bytes()),
        5 => set_shader_bytes(WGSL_GQA_INT8_SPLIT.as_bytes()),
        6 => set_shader_bytes(WGSL_RMSNORM_SWIGLU.as_bytes()),
        7 => set_shader_bytes(WGSL_ARGMAX_PASS1.as_bytes()),
        8 => set_shader_bytes(WGSL_ARGMAX_PASS2.as_bytes()),
        9 => set_shader_bytes(WGSL_GQA_REDUCE.as_bytes()),
        10 => set_shader_bytes(WGSL_EMBED_LOOKUP.as_bytes()),
        11 => set_shader_bytes(WGSL_RMSNORM_BATCH.as_bytes()),
        12 => set_shader_bytes(WGSL_SWIGLU_BATCH.as_bytes()),
        13 => set_shader_bytes(WGSL_Q4_GEMM_F16.as_bytes()),
        14 => set_shader_bytes(WGSL_GDN_CONV1D.as_bytes()),
        15 => set_shader_bytes(WGSL_GDN_RECURRENT.as_bytes()),
        16 => set_shader_bytes(WGSL_QK_NORM_ROPE.as_bytes()),
        17 => set_shader_bytes(WGSL_GQA_FP16_PREFILL.as_bytes()),
        18 => set_shader_bytes(WGSL_GQA_INT8_PREFILL.as_bytes()),
        19 => set_shader_bytes(WGSL_GDN_CHUNK_PREP.as_bytes()),
        20 => set_shader_bytes(WGSL_GDN_CHUNK_APPLY.as_bytes()),
        21 => set_shader_bytes(WGSL_Q4_GATE_UP_SWIGLU.as_bytes()),
        22 => set_shader_bytes(WGSL_Q4_GEMV_WIDE_K.as_bytes()),
        23 => set_shader_bytes(WGSL_GQA_INT8_SPLIT_DP4A.as_bytes()),
        24 => set_shader_bytes(WGSL_Q4_GEMV_BATCH.as_bytes()),
        25 => set_shader_bytes(WGSL_Q4_GEMV_T2_F16.as_bytes()),
        26 => set_shader_bytes(WGSL_Q4_GEMV_T3_F32.as_bytes()),
        27 => set_shader_bytes(WGSL_Q4_GATE_UP_SWIGLU_T2_F16.as_bytes()),
        28 => set_shader_bytes(WGSL_Q4_GATE_UP_SWIGLU_T3_F32.as_bytes()),
        29 => set_shader_bytes(WGSL_Q4_GEMV_WIDE_K_T2_F16.as_bytes()),
        30 => set_shader_bytes(WGSL_Q4_GEMV_WIDE_K_T3_F32.as_bytes()),
        31 => set_shader_bytes(WGSL_Q4_GEMV_BATCH_T2_F16.as_bytes()),
        32 => set_shader_bytes(WGSL_Q4_GEMV_BATCH_T3_F32.as_bytes()),
        33 => set_shader_bytes(WGSL_Q4_GEMM_T3_F32.as_bytes()),
        34 => set_shader_bytes(WGSL_RMSNORM_BATCH_T2.as_bytes()),
        35 => set_shader_bytes(WGSL_GQA_FP16_SPLIT_T2_F16.as_bytes()),
        36 => set_shader_bytes(WGSL_GQA_FP16_SPLIT_T3_F32.as_bytes()),
        37 => set_shader_bytes(WGSL_GQA_INT8_SPLIT_T2.as_bytes()),
        38 => set_shader_bytes(WGSL_GQA_FP16_PREFILL_T2_F16.as_bytes()),
        39 => set_shader_bytes(WGSL_GQA_FP16_PREFILL_T3_F32.as_bytes()),
        40 => set_shader_bytes(WGSL_GQA_INT8_PREFILL_T2.as_bytes()),
        41 => set_shader_bytes(WGSL_GDN_CHUNK_PREP_T2.as_bytes()),
        42 => set_shader_bytes(WGSL_GDN_CHUNK_APPLY_T2.as_bytes()),
        _ => 0,
    }
}

// ============================================================================
// WGSL Kernels (Chrome 159 Tint compliant)
// ============================================================================

const WGSL_Q4_SCALAR: &str = r#"
struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let blocks = params.cols >> 5u;
  let row_base = params.w_u32_offset + row * blocks * 5u;
  var acc: f32 = 0.0;
  for (var b: u32 = 0u; b < blocks; b = b + 1u) {
    let b_idx = row_base + b * 5u;
    let blk_meta = unpack2x16float(weights[b_idx + 4u]);
    let scale = blk_meta.x;
    let zero = blk_meta.y;
    let x_base = params.in_offset + (b << 5u);
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
      let packed = weights[b_idx + w];
      let k_base = x_base + (w << 3u);
      for (var n: u32 = 0u; n < 8u; n = n + 1u) {
        let q = f32((packed >> (n * 4u)) & 0xFu);
        acc += fma(q, scale, zero) * x[k_base + n];
      }
    }
  }
  let out_i = params.out_offset + row;
  if (params.add_residual != 0u) {
    y[out_i] = y[out_i] + acc;
  } else {
    y[out_i] = acc;
  }
}
"#;

const WGSL_Q4_VEC4_F16: &str = r#"
enable f16;

struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  let blocks = params.cols >> 5u;
  let row_base = params.w_u32_offset + row * blocks * 5u;
  var acc: f32 = 0.0;
  for (var b: u32 = 0u; b < blocks; b = b + 1u) {
    let b_idx = row_base + b * 5u;
    let blk_meta = bitcast<vec2<f16>>(weights[b_idx + 4u]);
    let s4 = vec4<f16>(blk_meta.x);
    let z4 = vec4<f16>(blk_meta.y);
    let x_base = params.in_offset + (b << 5u);
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
      let packed = weights[b_idx + w];
      let k_base = x_base + (w << 3u);
      let q0 = vec4<f16>(
        f16(packed & 0xFu),
        f16((packed >> 4u) & 0xFu),
        f16((packed >> 8u) & 0xFu),
        f16((packed >> 12u) & 0xFu),
      );
      let q1 = vec4<f16>(
        f16((packed >> 16u) & 0xFu),
        f16((packed >> 20u) & 0xFu),
        f16((packed >> 24u) & 0xFu),
        f16((packed >> 28u) & 0xFu),
      );
      let w0 = fma(q0, s4, z4);
      let w1 = fma(q1, s4, z4);
      let x0 = vec4<f16>(f16(x[k_base]), f16(x[k_base + 1u]), f16(x[k_base + 2u]), f16(x[k_base + 3u]));
      let x1 = vec4<f16>(f16(x[k_base + 4u]), f16(x[k_base + 5u]), f16(x[k_base + 6u]), f16(x[k_base + 7u]));
      acc += f32(dot(w0, x0) + dot(w1, x1));
    }
  }
  let out_i = params.out_offset + row;
  if (params.add_residual != 0u) {
    y[out_i] = y[out_i] + acc;
  } else {
    y[out_i] = acc;
  }
}
"#;

// Coalesced-X Factored Q4 GEMV v3: 1 workgroup of 128 threads (8 x 16-lane groups) per 16 output rows.
// Stages up to 2560 f16 activations coalesced into x_sh (1 barrier total for K<=2560, 0 K-loop barriers).
const WGSL_Q4_SUBGROUPS_F16: &str = r#"
enable f16;
enable subgroups;

struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn factored_q4_word(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn factored_q4_block(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f16>, x0b: vec4<f16>,
  x1a: vec4<f16>, x1b: vec4<f16>,
  x2a: vec4<f16>, x2b: vec4<f16>,
  x3a: vec4<f16>, x3b: vec4<f16>,
  sum_x: f32,
) -> f32 {
  let blk_meta = unpack2x16float(meta_word);
  let sum_qx = factored_q4_word(w0, x0a, x0b)
    + factored_q4_word(w1, x1a, x1b)
    + factored_q4_word(w2, x2a, x2b)
    + factored_q4_word(w3, x3a, x3b);
  return blk_meta.x * sum_qx + blk_meta.y * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;  // 0..7
  let lane = tid & 15u; // 0..15
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, params.rows - 1u);
  let r1 = min(rB, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let o0 = params.w_u32_offset + r0 * row_stride;
  let o1 = params.w_u32_offset + r1 * row_stride;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  let ones = vec4<f16>(1.0h);

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 80u) {
    let tile_blocks = min(80u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[i] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    }
    workgroupBarrier();

    for (var b_rel: u32 = lane; b_rel < tile_blocks; b_rel = b_rel + 16u) {
      let xb = b_rel << 3u;
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
        + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
        + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
        + f32(dot(ones, x3a)) + f32(dot(ones, x3b));

      let b5 = (k_tile + b_rel) * 5u;
      let i0 = o0 + b5;
      let i1 = o1 + b5;
      acc0 += factored_q4_block(weights[i0], weights[i0 + 1u], weights[i0 + 2u], weights[i0 + 3u], weights[i0 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
      acc1 += factored_q4_block(weights[i1], weights[i1 + 1u], weights[i1 + 2u], weights[i1 + 3u], weights[i1 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
    }

    if (k_tile + 80u < blocks) {
      workgroupBarrier();
    }
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
      var s_acc = vec2<f32>(0.0);
      let g_base = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        s_acc += red_sh[g_base + l];
      }
      sum2 = s_acc;
    }
  }

  if (lane == 0u) {
    if (rA < params.rows) {
      let out0 = params.out_offset + rA;
      y[out0] = select(sum2.x, y[out0] + sum2.x, params.add_residual != 0u);
    }
    if (rB < params.rows) {
      let out1 = params.out_offset + rB;
      y[out1] = select(sum2.y, y[out1] + sum2.y, params.add_residual != 0u);
    }
  }
}
"#;

// In-place Gated DeltaNet recurrent step (32 heads x 128 x 128 state per layer)
// Dispatch 32 workgroups (1 per value head), 128 threads/workgroup (1 per value dim j)
const WGSL_DELTANET_STEP: &str = r#"
struct DeltaNetParams {
  state_f32_offset: u32,
  conv_f32_offset: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  z_offset: u32,
  gate_offset: u32,
  out_offset: u32,
};

@group(0) @binding(0) var<uniform> params: DeltaNetParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> dn_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> dn_out: array<f32>;

fn silu(v: f32) -> f32 {
  return v / (1.0 + exp(-v));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let v_head = wid.x; // 0..31
  let j = lid.x;      // 0..127
  let k_head = v_head >> 1u; // 16 key heads mapped 2:1 to 32 val heads

  // Update 4-tap causal conv1d ring for channels [v_head*128 + j]
  let chan = v_head * 128u + j;
  let c_base = params.conv_f32_offset + chan * 4u;
  let c0 = dn_state[c_base + 1u];
  let c1 = dn_state[c_base + 2u];
  let c2 = dn_state[c_base + 3u];
  let v_raw = act[params.v_offset + chan];
  dn_state[c_base + 0u] = c0;
  dn_state[c_base + 1u] = c1;
  dn_state[c_base + 2u] = c2;
  dn_state[c_base + 3u] = v_raw;
  let v_conv = silu(0.25 * (c0 + c1 + c2 + v_raw));

  // Decay & delta learning rate gates per value head
  let decay = clamp(exp(-abs(act[params.gate_offset + v_head]) * 0.1), 0.0, 1.0);
  let beta = 1.0 / (1.0 + exp(-act[params.gate_offset + 32u + v_head]));

  let s_base = params.state_f32_offset + v_head * (128u * 128u);
  let k_base = params.k_offset + k_head * 128u;
  let q_base = params.q_offset + k_head * 128u;

  // Pass 1: predict S^T * k (decayed) with read-only pass
  var sk: f32 = 0.0;
  for (var i: u32 = 0u; i < 128u; i = i + 1u) {
    let idx = s_base + i * 128u + j;
    sk += (dn_state[idx] * decay) * act[k_base + i];
  }

  let delta = (v_conv - sk) * beta;

  // Pass 2: combined decay + delta update + query readout (single global store per element)
  var out_val: f32 = 0.0;
  for (var i: u32 = 0u; i < 128u; i = i + 1u) {
    let idx = s_base + i * 128u + j;
    let s_new = dn_state[idx] * decay + act[k_base + i] * delta;
    dn_state[idx] = s_new;
    out_val += s_new * act[q_base + i];
  }

  let z_val = act[params.z_offset + chan];
  dn_out[params.out_offset + chan] = out_val * silu(z_val);
}
"#;

// Static FP16 KV Cache Split-K FlashDecoding GQA (16 Q heads, 4 KV heads, head_dim = 256)
// Dispatches (16, num_splits, 1) with 64 threads/workgroup -> 1 barrier per token in history loop.
const WGSL_GQA_FP16_SPLIT: &str = r#"
enable f16;
enable subgroups;

struct GqaSplitParams {
  layer_kv_u32_offset: u32,
  max_seq: u32,
  seq_pos: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  num_splits: u32,
  q_head_stride: u32,
};

@group(0) @binding(0) var<uniform> params: GqaSplitParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>; // packed 2x f16 per u32
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>; // 260 f32s per (q_head, split)
@group(0) @binding(4) var<storage, read> pos_buf: array<u32>;

var<workgroup> red_buf: array<f32, 128>;
var<workgroup> cur_buf: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let q_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let kv_head = q_head / gqa_ratio;
  let tid = lid.x;          // 0..63 -> 4 dims per thread (2 u32s)
  let d2 = tid * 2u;
  let d = tid * 4u;
  let seq_pos = select(params.seq_pos, pos_buf[1], params.seq_pos == 0xFFFFFFFFu);

  let head_u32_stride = params.max_seq * 128u; // 256 f16 = 128 u32 per token
  let kv_plane_u32s = kv_heads * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let q_base = params.q_offset + q_head * q_stride + d;
  let q_vec = vec4<f32>(act[q_base], act[q_base + 1u], act[q_base + 2u], act[q_base + 3u]);
  let inv_sqrt_d: f32 = 0.0625; // 1 / sqrt(256)
  let sg_step = max(sg_size, 1u);

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  var acc_v = vec4<f32>(0.0);

  let chunk = (seq_pos + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, seq_pos);
  let t_end = min(t_start + chunk, seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let partial = dot(q_vec, vec4<f32>(k0.x, k0.y, k1.x, k1.y));

    let slot = ((t - t_start) & 1u) * 64u;
    let sg_dot = subgroupAdd(partial);
    red_buf[slot + tid] = select(0.0, sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var full_dot: f32 = 0.0;
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      full_dot += red_buf[slot + s];
    }

    let score = full_dot * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;

    let v_src = v_head_base + t * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    acc_v = acc_v * alpha + vec4<f32>(v0.x, v0.y, v1.x, v1.y) * beta;
    m_max = m_new;
  }

  // Only split 0 handles current token seq_pos: zero cross-workgroup KV races
  if (split_id == 0u) {
    let k_in = params.k_offset + kv_head * 256u + d;
    let v_in = params.v_offset + kv_head * 256u + d;
    let k0h = vec2<f16>(f16(act[k_in]), f16(act[k_in + 1u]));
    let k1h = vec2<f16>(f16(act[k_in + 2u]), f16(act[k_in + 3u]));
    let v0h = vec2<f16>(f16(act[v_in]), f16(act[v_in + 1u]));
    let v1h = vec2<f16>(f16(act[v_in + 2u]), f16(act[v_in + 3u]));

    if ((q_head % gqa_ratio) == 0u) {
      let k_dst = k_head_base + seq_pos * 128u + d2;
      let v_dst = v_head_base + seq_pos * 128u + d2;
      kv_cache[k_dst] = bitcast<u32>(k0h);
      kv_cache[k_dst + 1u] = bitcast<u32>(k1h);
      kv_cache[v_dst] = bitcast<u32>(v0h);
      kv_cache[v_dst + 1u] = bitcast<u32>(v1h);
    }

    let cur_partial = dot(q_vec, vec4<f32>(f32(k0h.x), f32(k0h.y), f32(k1h.x), f32(k1h.y)));
    let cur_sg = subgroupAdd(cur_partial);
    cur_buf[tid] = select(0.0, cur_sg, sg_inv == 0u);
    workgroupBarrier();

    var cur_dot: f32 = 0.0;
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      cur_dot += cur_buf[s];
    }

    let score = cur_dot * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;
    acc_v = acc_v * alpha + vec4<f32>(f32(v0h.x), f32(v0h.y), f32(v1h.x), f32(v1h.y)) * beta;
    m_max = m_new;
  }

  let out_base = (q_head * num_splits + split_id) * 260u;
  split_out[out_base + d] = acc_v.x;
  split_out[out_base + d + 1u] = acc_v.y;
  split_out[out_base + d + 2u] = acc_v.z;
  split_out[out_base + d + 3u] = acc_v.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_max;
    split_out[out_base + 257u] = l_sum;
  }
}
"#;

// 4-Q-Head GQA-Grouped INT8 KV Split-K FlashDecoding (16 Q heads, 4 KV heads, head_dim = 256)
// Dispatches (4, num_splits, 1) -> 1 workgroup per (kv_head, split), 4x KV bandwidth reuse!
const WGSL_GQA_INT8_SPLIT: &str = r#"
enable subgroups;

struct GqaSplitParams {
  layer_kv_u32_offset: u32,
  max_seq: u32,
  seq_pos: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  num_splits: u32,
  q_head_stride: u32,
};

@group(0) @binding(0) var<uniform> params: GqaSplitParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;
@group(0) @binding(4) var<storage, read> pos_buf: array<u32>;

var<workgroup> red_buf: array<vec4<f32>, 128>;
var<workgroup> cur_kv_max: array<vec2<f32>, 64>;
var<workgroup> cur_dot_buf: array<vec4<f32>, 64>;

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

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let qh0 = kv_head * gqa_ratio;
  let split_id = wid.y;
  let tid = lid.x; // 0..63 -> 1 packed u32 (4 int8 dims) per thread
  let d = tid * 4u;
  let seq_pos = select(params.seq_pos, pos_buf[1], params.seq_pos == 0xFFFFFFFFu);

  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = kv_heads * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let qb0 = params.q_offset + (qh0 + 0u) * q_stride + d;
  let qb1 = params.q_offset + (qh0 + min(1u, gqa_ratio - 1u)) * q_stride + d;
  let qb2 = params.q_offset + (qh0 + min(2u, gqa_ratio - 1u)) * q_stride + d;
  let qb3 = params.q_offset + (qh0 + min(3u, gqa_ratio - 1u)) * q_stride + d;
  let q0 = vec4<f32>(act[qb0], act[qb0 + 1u], act[qb0 + 2u], act[qb0 + 3u]);
  let q1 = vec4<f32>(act[qb1], act[qb1 + 1u], act[qb1 + 2u], act[qb1 + 3u]);
  let q2 = vec4<f32>(act[qb2], act[qb2 + 1u], act[qb2 + 2u], act[qb2 + 3u]);
  let q3 = vec4<f32>(act[qb3], act[qb3 + 1u], act[qb3 + 2u], act[qb3 + 3u]);

  let inv_sqrt_d: f32 = 0.0625;
  let sg_step = max(sg_size, 1u);

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let chunk = (seq_pos + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, seq_pos);
  let t_end = min(t_start + chunk, seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let ki = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    let p4 = vec4<f32>(dot(q0, ki), dot(q1, ki), dot(q2, ki), dot(q3, ki));
    let slot = ((t - t_start) & 1u) * 64u;
    let sg_dot = subgroupAdd(p4);
    red_buf[slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var full_dot = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      full_dot += red_buf[slot + s];
    }

    let score = full_dot * inv_sqrt_d;
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let v_src = v_head_base + t * 65u;
    let vi = unpack4_i8(kv_cache[v_src + tid]) * scales.y;
    acc0 = acc0 * alpha.x + vi * beta.x;
    acc1 = acc1 * alpha.y + vi * beta.y;
    acc2 = acc2 * alpha.z + vi * beta.z;
    acc3 = acc3 * alpha.w + vi * beta.w;
    m4 = m_new;
  }

  if (split_id == 0u) {
    let k_in = params.k_offset + kv_head * 256u + d;
    let v_in = params.v_offset + kv_head * 256u + d;
    let kv4 = vec4<f32>(act[k_in], act[k_in + 1u], act[k_in + 2u], act[k_in + 3u]);
    let vv4 = vec4<f32>(act[v_in], act[v_in + 1u], act[v_in + 2u], act[v_in + 3u]);

    let kmax = subgroupMax(max(max(abs(kv4.x), abs(kv4.y)), max(abs(kv4.z), abs(kv4.w))));
    let vmax = subgroupMax(max(max(abs(vv4.x), abs(vv4.y)), max(abs(vv4.z), abs(vv4.w))));
    cur_kv_max[tid] = select(vec2<f32>(0.0), vec2<f32>(kmax, vmax), sg_inv == 0u);
    workgroupBarrier();

    var km: f32 = 0.0;
    var vm: f32 = 0.0;
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      let p = cur_kv_max[s];
      km = max(km, p.x);
      vm = max(vm, p.y);
    }
    let packed_scales = pack2x16float(vec2<f32>(max(km / 127.0, 1e-6), max(vm / 127.0, 1e-6)));
    let scales = unpack2x16float(packed_scales);
    let qk = pack4_i8(kv4 / scales.x);
    let qv = pack4_i8(vv4 / scales.y);

    let k_dst = k_head_base + seq_pos * 65u;
    let v_dst = v_head_base + seq_pos * 65u;
    kv_cache[k_dst + tid] = qk;
    kv_cache[v_dst + tid] = qv;
    if (tid == 0u) {
      kv_cache[k_dst + 64u] = packed_scales;
    }

    let k_deq = unpack4_i8(qk) * scales.x;
    let cur_p4 = vec4<f32>(dot(q0, k_deq), dot(q1, k_deq), dot(q2, k_deq), dot(q3, k_deq));
    let cur_sg = subgroupAdd(cur_p4);
    cur_dot_buf[tid] = select(vec4<f32>(0.0), cur_sg, sg_inv == 0u);
    workgroupBarrier();

    var cur_dot = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      cur_dot += cur_dot_buf[s];
    }

    let score = cur_dot * inv_sqrt_d;
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;
    let v_deq = unpack4_i8(qv) * scales.y;
    acc0 = acc0 * alpha.x + v_deq * beta.x;
    acc1 = acc1 * alpha.y + v_deq * beta.y;
    acc2 = acc2 * alpha.z + v_deq * beta.z;
    acc3 = acc3 * alpha.w + v_deq * beta.w;
    m4 = m_new;
  }

  let ob0 = ((qh0 + 0u) * num_splits + split_id) * 260u;
  split_out[ob0 + d] = acc0.x; split_out[ob0 + d + 1u] = acc0.y; split_out[ob0 + d + 2u] = acc0.z; split_out[ob0 + d + 3u] = acc0.w;
  if (tid == 0u) { split_out[ob0 + 256u] = m4.x; split_out[ob0 + 257u] = l4.x; }

  if (gqa_ratio > 1u) {
    let ob1 = ((qh0 + 1u) * num_splits + split_id) * 260u;
    split_out[ob1 + d] = acc1.x; split_out[ob1 + d + 1u] = acc1.y; split_out[ob1 + d + 2u] = acc1.z; split_out[ob1 + d + 3u] = acc1.w;
    if (tid == 0u) { split_out[ob1 + 256u] = m4.y; split_out[ob1 + 257u] = l4.y; }
  }
  if (gqa_ratio > 2u) {
    let ob2 = ((qh0 + 2u) * num_splits + split_id) * 260u;
    split_out[ob2 + d] = acc2.x; split_out[ob2 + d + 1u] = acc2.y; split_out[ob2 + d + 2u] = acc2.z; split_out[ob2 + d + 3u] = acc2.w;
    if (tid == 0u) { split_out[ob2 + 256u] = m4.z; split_out[ob2 + 257u] = l4.z; }
  }
  if (gqa_ratio > 3u) {
    let ob3 = ((qh0 + 3u) * num_splits + split_id) * 260u;
    split_out[ob3 + d] = acc3.x; split_out[ob3 + d + 1u] = acc3.y; split_out[ob3 + d + 2u] = acc3.z; split_out[ob3 + d + 3u] = acc3.w;
    if (tid == 0u) { split_out[ob3 + 256u] = m4.w; split_out[ob3 + 257u] = l4.w; }
  }
}
"#;

// Pass 2 of Split-K FlashDecoding: online-softmax merge across num_splits + output sigmoid gate
const WGSL_GQA_REDUCE: &str = r#"
struct GqaReduceParams {
  num_splits: u32,
  gate_offset: u32,
  out_offset: u32,
  g_head_stride: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
  _pad4: u32,
};

@group(0) @binding(0) var<uniform> params: GqaReduceParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read> split_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let q_head = wid.x;
  let tok = wid.y;
  let q_heads = select(params._pad1, 16u, params._pad1 == 0u);
  let head_slot = tok * q_heads + q_head;
  let tid = lid.x;
  let d = tid * 4u;

  var m_global: f32 = -1e30;
  for (var s: u32 = 0u; s < params.num_splits; s = s + 1u) {
    let base = (head_slot * params.num_splits + s) * 260u;
    let ls = split_out[base + 257u];
    if (ls > 0.0) {
      m_global = max(m_global, split_out[base + 256u]);
    }
  }

  var l_global: f32 = 0.0;
  var acc_v = vec4<f32>(0.0);
  for (var s: u32 = 0u; s < params.num_splits; s = s + 1u) {
    let base = (head_slot * params.num_splits + s) * 260u;
    let ls = split_out[base + 257u];
    if (ls > 0.0) {
      let ms = split_out[base + 256u];
      let w = exp(ms - m_global);
      l_global += ls * w;
      let v = vec4<f32>(
        split_out[base + d],
        split_out[base + d + 1u],
        split_out[base + d + 2u],
        split_out[base + d + 3u],
      );
      acc_v += v * w;
    }
  }

  let out_vec = acc_v / max(l_global, 1e-6);
  let g_stride = select(params.g_head_stride, 256u, params.g_head_stride == 0u);
  let g_base = params.gate_offset + tok * (q_heads * g_stride) + q_head * g_stride + d;
  let out_base = params.out_offset + tok * (q_heads * 256u) + q_head * 256u + d;
  attn_out[out_base + 0u] = out_vec.x / (1.0 + exp(-act[g_base + 0u]));
  attn_out[out_base + 1u] = out_vec.y / (1.0 + exp(-act[g_base + 1u]));
  attn_out[out_base + 2u] = out_vec.z / (1.0 + exp(-act[g_base + 2u]));
  attn_out[out_base + 3u] = out_vec.w / (1.0 + exp(-act[g_base + 3u]));
}
"#;

// Mode 0: RMSNorm (src -> dst), Mode 1: SwiGLU (gate + up in src -> dst)
const WGSL_RMSNORM_SWIGLU: &str = r#"
struct ElemParams {
  mode: u32,
  len: u32,
  in_a_offset: u32,
  in_b_offset: u32,
  out_offset: u32,
  eps: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ElemParams;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

var<workgroup> sq_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  if (params.mode == 0u) {
    var sq: f32 = 0.0;
    for (var i: u32 = tid; i < params.len; i = i + 256u) {
      let v = src[params.in_a_offset + i];
      sq += v * v;
    }
    sq_buf[tid] = sq;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sq_buf[tid] += sq_buf[tid + s];
      }
      workgroupBarrier();
    }
    let inv_rms = inverseSqrt(sq_buf[0] / f32(params.len) + params.eps);
    for (var i: u32 = tid; i < params.len; i = i + 256u) {
      let v = src[params.in_a_offset + i];
      dst[params.out_offset + i] = v * inv_rms * gamma[params.in_b_offset + i];
    }
  } else {
    let i = gid.x;
    if (i < params.len) {
      let g = src[params.in_a_offset + i];
      let u = src[params.in_b_offset + i];
      let sig = 1.0 / (1.0 + exp(-g));
      dst[params.out_offset + i] = (g * sig) * u;
    }
  }
}
"#;

const WGSL_ARGMAX_PASS1: &str = r#"
struct ArgmaxParams {
  len: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: ArgmaxParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials_val: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials_idx: array<u32>;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let tid = lid.x;
  let idx = gid.x;
  if (idx < params.len) {
    best_v[tid] = logits[idx];
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

  if (tid == 0u) {
    partials_val[wid.x] = best_v[0];
    partials_idx[wid.x] = best_i[0];
  }
}
"#;

const WGSL_ARGMAX_PASS2: &str = r#"
struct ArgmaxParams {
  len: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: ArgmaxParams;
@group(0) @binding(1) var<storage, read> partials_val: array<f32>;
@group(0) @binding(2) var<storage, read> partials_idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_token: array<u32>;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_v: f32 = -1e30;
  var local_i: u32 = 0u;
  for (var i: u32 = tid; i < params.len; i = i + 256u) {
    let v = partials_val[i];
    if (v > local_v) {
      local_v = v;
      local_i = partials_idx[i];
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

  if (tid == 0u) {
    out_token[0] = best_i[0];
  }
}
"#;

// Token IDs -> hidden_state [num_tokens, 2560] using tied lm_head Q4 rows in weightSlab2
const WGSL_EMBED_LOOKUP: &str = r#"
struct EmbedParams {
  num_tokens: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tok_idx = wid.x;
  let b = wid.y;
  let d = lid.x;
  if (tok_idx >= params.num_tokens) {
    return;
  }
  let tok = token_ids[tok_idx];
  if (tok < params._pad0 || (params._pad1 != 0u && tok >= params._pad1)) {
    return;
  }
  let local_tok = tok - params._pad0;
  let blocks = params.cols >> 5u;
  let bi = (local_tok * blocks + b) * 5u;
  let blk_meta = unpack2x16float(weights[bi + 4u]);
  let word = weights[bi + (d >> 3u)];
  let q = f32((word >> ((d & 7u) * 4u)) & 0xFu);
  out[tok_idx * params.cols + (b << 5u) + d] = fma(q, blk_meta.x, blk_meta.y);
}
"#;

// Batched Subgroup-Accelerated RMSNorm [num_tokens, len] reading gamma from aux[gamma_offset..]
const WGSL_RMSNORM_BATCH: &str = r#"
enable subgroups;

struct BatchNormParams {
  num_tokens: u32,
  len: u32,
  gamma_offset: u32,
  in_offset: u32,
  out_offset: u32,
  eps: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: BatchNormParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

var<workgroup> sq_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tok = wid.x;
  let tid = lid.x;
  let base_in = params.in_offset + tok * params.len;
  let base_out = params.out_offset + tok * params.len;

  var sq: f32 = 0.0;
  for (var i: u32 = tid; i < params.len; i = i + 256u) {
    let v = src[base_in + i];
    sq += v * v;
  }
  let sg_sq = subgroupAdd(sq);
  sq_buf[tid] = select(0.0, sg_sq, sg_inv == 0u);
  workgroupBarrier();
  let sg_step = max(sg_size, 1u);
  var total_sq: f32 = 0.0;
  for (var s: u32 = 0u; s < 256u; s = s + sg_step) {
    total_sq += sq_buf[s];
  }
  let inv_rms = inverseSqrt(total_sq / f32(params.len) + params.eps);
  for (var i: u32 = tid; i < params.len; i = i + 256u) {
    dst[base_out + i] = src[base_in + i] * inv_rms * aux[params.gamma_offset + i];
  }
}
"#;

// Contiguous batched SwiGLU over total_elems = num_tokens * inter_size
const WGSL_SWIGLU_BATCH: &str = r#"
struct SwigluBatchParams {
  total_elems: u32,
  gate_offset: u32,
  up_offset: u32,
  out_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: SwigluBatchParams;
@group(0) @binding(1) var<storage, read> _unused_aux: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx < params.total_elems) {
    let g = src[params.gate_offset + idx];
    let u = src[params.up_offset + idx];
    let sig = 1.0 / (1.0 + exp(-g));
    dst[params.out_offset + idx] = (g * sig) * u;
  }
}
"#;

// Batched Q4-f16 GEMM [M, K] x [N, K]^T -> [M, N] via 64x64x32 shared-memory tiling
const WGSL_Q4_GEMM_F16: &str = r#"
enable f16;

struct GemmParams {
  M: u32,
  N: u32,
  K: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 512>;
var<workgroup> w_sh: array<vec4<f16>, 512>;

fn unpack_nibble_vec(word: u32, shift_base: u32, s4: vec4<f16>, z4: vec4<f16>) -> vec4<f16> {
  let q = vec4<f16>(
    f16((word >> shift_base) & 0xFu),
    f16((word >> (shift_base + 4u)) & 0xFu),
    f16((word >> (shift_base + 8u)) & 0xFu),
    f16((word >> (shift_base + 12u)) & 0xFu),
  );
  return fma(q, s4, z4);
}

fn store_cell(tok: u32, row: u32, val: f32) {
  if (tok < params.M && row < params.N) {
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
  let blocks = params.K >> 5u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let loc_idx = tid >> 2u; // 0..63
  let qtr = tid & 3u;      // 0..3 (8 elements per quarter)
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
      let bi = params.w_u32_offset + gr * blocks * 5u + b * 5u;
      let blk_meta = bitcast<vec2<f16>>(weights[bi + 4u]);
      let s4 = vec4<f16>(blk_meta.x);
      let z4 = vec4<f16>(blk_meta.y);
      let pw = weights[bi + qtr];
      w_sh[k4_0 * 64u + loc_idx] = unpack_nibble_vec(pw, 0u, s4, z4);
      w_sh[k4_1 * 64u + loc_idx] = unpack_nibble_vec(pw, 16u, s4, z4);
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
  store_cell(out_t + 0u, out_r + 0u, acc0.x);
  store_cell(out_t + 16u, out_r + 0u, acc0.y);
  store_cell(out_t + 32u, out_r + 0u, acc0.z);
  store_cell(out_t + 48u, out_r + 0u, acc0.w);

  store_cell(out_t + 0u, out_r + 16u, acc1.x);
  store_cell(out_t + 16u, out_r + 16u, acc1.y);
  store_cell(out_t + 32u, out_r + 16u, acc1.z);
  store_cell(out_t + 48u, out_r + 16u, acc1.w);

  store_cell(out_t + 0u, out_r + 32u, acc2.x);
  store_cell(out_t + 16u, out_r + 32u, acc2.y);
  store_cell(out_t + 32u, out_r + 32u, acc2.z);
  store_cell(out_t + 48u, out_r + 32u, acc2.w);

  store_cell(out_t + 0u, out_r + 48u, acc3.x);
  store_cell(out_t + 16u, out_r + 48u, acc3.y);
  store_cell(out_t + 32u, out_r + 48u, acc3.z);
  store_cell(out_t + 48u, out_r + 48u, acc3.w);
}
"#;

// Causal Depthwise Conv1D (kernel=4, state=3) + SiLU over all 8192 GDN QKV channels
// Dispatch (64, 1, 1) with @workgroup_size(128) -> 1 thread per channel
const WGSL_GDN_CONV1D: &str = r#"
struct ConvParams {
  num_tokens: u32,
  conv_w_offset: u32,
  conv_state_offset: u32,
  qkv_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> dn_conv: array<f32>;
@group(0) @binding(3) var<storage, read_write> work: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let conv_dim = select(params._pad0, 8192u, params._pad0 == 0u);
  let c = gid.x;
  if (c >= conv_dim) {
    return;
  }
  let wb = params.conv_w_offset + c * 4u;
  let w0 = aux[wb + 0u];
  let w1 = aux[wb + 1u];
  let w2 = aux[wb + 2u];
  let w3 = aux[wb + 3u];

  let sb = params.conv_state_offset + c * 3u;
  var s0 = dn_conv[sb + 0u];
  var s1 = dn_conv[sb + 1u];
  var s2 = dn_conv[sb + 2u];

  for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {
    let idx = params.qkv_offset + t * conv_dim + c;
    let x = work[idx];
    let raw = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
    work[idx] = raw / (1.0 + exp(-raw));
    s0 = s1;
    s1 = s2;
    s2 = x;
  }

  dn_conv[sb + 0u] = s0;
  dn_conv[sb + 1u] = s1;
  dn_conv[sb + 2u] = s2;
}
"#;

// Exact Qwen3.5 Gated DeltaNet Recurrent Step + Q/K L2 Norm + Gated RMSNorm (32 heads x 128 x 128)
// Dispatch (32, 1, 1) with @workgroup_size(128)
const WGSL_GDN_RECURRENT: &str = r#"
struct GdnRecParams {
  num_tokens: u32,
  state_f32_offset: u32,
  aux_dn_offset: u32,
  qkv_offset: u32,
  z_offset: u32,
  ab_offset: u32,
  out_offset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: GdnRecParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> work: array<f32>;
@group(0) @binding(3) var<storage, read_write> dn_state: array<f32>;
@group(0) @binding(4) var<storage, read_write> attn_out: array<f32>;

var<workgroup> sh_red_q: array<f32, 128>;
var<workgroup> sh_red_k: array<f32, 128>;
var<workgroup> sh_q: array<f32, 128>;
var<workgroup> sh_k: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let v_heads = select(params._pad0 & 0xFFFFu, 32u, params._pad0 == 0u);
  let k_heads = select(params._pad0 >> 16u, 16u, params._pad0 == 0u);
  let ratio = max(v_heads / max(k_heads, 1u), 1u);
  let qk_dim = k_heads * 128u;
  let v_dim = v_heads * 128u;
  let conv_dim = qk_dim * 2u + v_dim;

  let v_head = wid.x;
  let j = lid.x;      // 0..127
  let k_head = v_head / ratio;

  let a_neg = aux[params.aux_dn_offset + v_head];
  let dt_b = aux[params.aux_dn_offset + v_heads + v_head];
  let gdn_norm_w = aux[params.aux_dn_offset + v_heads * 2u + j];
  let s_base = params.state_f32_offset + v_head * (128u * 128u);

  for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {
    let qkv_t = params.qkv_offset + t * conv_dim;
    let q_raw = work[qkv_t + k_head * 128u + j];
    let k_raw = work[qkv_t + qk_dim + k_head * 128u + j];

    sh_red_q[j] = q_raw * q_raw;
    sh_red_k[j] = k_raw * k_raw;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (j < s) {
        sh_red_q[j] += sh_red_q[j + s];
        sh_red_k[j] += sh_red_k[j + s];
      }
      workgroupBarrier();
    }
    // Q/K L2 normalization + 1/sqrt(128) readout scale
    let inv_q = inverseSqrt(sh_red_q[0] + 1e-6) * 0.08838834764831845;
    let inv_k = inverseSqrt(sh_red_k[0] + 1e-6);
    sh_q[j] = q_raw * inv_q;
    sh_k[j] = k_raw * inv_k;
    workgroupBarrier();

    let v_val = work[qkv_t + qk_dim * 2u + v_head * 128u + j];
    let ab_t = params.ab_offset + t * (v_heads * 2u);
    let b_raw = work[ab_t + v_head];
    let a_raw = work[ab_t + v_heads + v_head];
    let beta = 1.0 / (1.0 + exp(-b_raw));
    let u = a_raw + dt_b;
    let sp = select(log(1.0 + exp(u)), u, u > 20.0);
    let decay = exp(a_neg * sp);

    var sk: f32 = 0.0;
    for (var i: u32 = 0u; i < 128u; i = i + 1u) {
      let idx = s_base + i * 128u + j;
      sk += (dn_state[idx] * decay) * sh_k[i];
    }
    let delta = (v_val - sk) * beta;

    var out_val: f32 = 0.0;
    for (var i: u32 = 0u; i < 128u; i = i + 1u) {
      let idx = s_base + i * 128u + j;
      let s_new = dn_state[idx] * decay + sh_k[i] * delta;
      dn_state[idx] = s_new;
      out_val += s_new * sh_q[i];
    }

    sh_red_q[j] = out_val * out_val;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (j < s) {
        sh_red_q[j] += sh_red_q[j + s];
      }
      workgroupBarrier();
    }
    let inv_rms = inverseSqrt(sh_red_q[0] / 128.0 + 1e-6);
    let z_val = work[params.z_offset + t * v_dim + v_head * 128u + j];
    let z_silu = z_val / (1.0 + exp(-z_val));
    attn_out[params.out_offset + t * v_dim + v_head * 128u + j] = (out_val * inv_rms) * z_silu * gdn_norm_w;
    workgroupBarrier();
  }
}
"#;

// Full Attention QK-RMSNorm + Partial NeoX RoPE (rotary_dim=64) across M tokens
// Dispatch (20, num_tokens, 1) with @workgroup_size(256)
const WGSL_QK_NORM_ROPE: &str = r#"
struct QkRopeParams {
  num_tokens: u32,
  start_pos: u32,
  qg_offset: u32,
  k_offset: u32,
  q_norm_offset: u32,
  k_norm_offset: u32,
  inv_freq_offset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: QkRopeParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> work: array<f32>;
@group(0) @binding(3) var<storage, read> pos_buf: array<u32>;

var<workgroup> sq_buf: array<f32, 256>;
var<workgroup> sh_head: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let q_heads = select(params._pad0 & 0xFFu, 16u, params._pad0 == 0u);
  let kv_heads = select((params._pad0 >> 8u) & 0xFFu, 4u, params._pad0 == 0u);
  let rope_half = select(params._pad0 >> 16u, 32u, params._pad0 == 0u);
  let h_idx = wid.x;
  let tok = wid.y;
  let d = lid.x;

  let is_q = h_idx < q_heads;
  let addr = select(
    params.k_offset + tok * (kv_heads * 256u) + (h_idx - q_heads) * 256u + d,
    params.qg_offset + tok * (q_heads * 512u) + h_idx * 512u + d,
    is_q,
  );
  let nw = select(
    aux[params.k_norm_offset + d],
    aux[params.q_norm_offset + d],
    is_q,
  );

  let val = work[addr];
  sq_buf[d] = val * val;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (d < s) {
      sq_buf[d] += sq_buf[d + s];
    }
    workgroupBarrier();
  }
  let inv_rms = inverseSqrt(sq_buf[0] / 256.0 + 1e-6);
  sh_head[d] = val * inv_rms * nw;
  workgroupBarrier();

  let start_pos_u32 = select(params.start_pos, pos_buf[1], params.start_pos == 0xFFFFFFFFu);
  let pos = f32(start_pos_u32 + tok);
  var out_val = sh_head[d];
  if (d < rope_half) {
    let ang = pos * aux[params.inv_freq_offset + d];
    out_val = sh_head[d] * cos(ang) - sh_head[d + rope_half] * sin(ang);
  } else if (d < rope_half * 2u) {
    let ang = pos * aux[params.inv_freq_offset + (d - rope_half)];
    out_val = sh_head[d] * cos(ang) + sh_head[d - rope_half] * sin(ang);
  }
  work[addr] = out_val;
}
"#;

// Batched FP16 KV Split-K Prefill: mode 0 writes K/V (4, M, 1), mode 1 runs 8-query GQA-tiled FlashPrefill (4, ceil(M/2), num_splits)
const WGSL_GQA_FP16_PREFILL: &str = r#"
enable f16;
enable subgroups;

struct GqaPrefillParams {
  mode: u32,
  layer_kv_u32_offset: u32,
  max_seq: u32,
  start_pos: u32,
  num_tokens: u32,
  buf_a_offset: u32, // mode 0: k_offset, mode 1: qg_offset
  buf_b_offset: u32, // mode 0: v_offset
  num_splits: u32,
};

@group(0) @binding(0) var<uniform> params: GqaPrefillParams;
@group(0) @binding(1) var<storage, read> work: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> red_a: array<vec4<f32>, 128>;
var<workgroup> red_b: array<vec4<f32>, 128>;
var<workgroup> tail_buf: array<vec4<f32>, 64>;

fn write_split_out(
  tok: u32,
  q_head: u32,
  q_heads: u32,
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * q_heads + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, q_heads: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * (q_heads * 512u) + q_head * 512u + d;
  return vec4<f32>(work[b], work[b + 1u], work[b + 2u], work[b + 3u]);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let d2 = tid * 2u;
  let d = tid * 4u;
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let q_heads = kv_heads * gqa_ratio;
  let head_u32_stride = params.max_seq * 128u;
  let kv_plane_u32s = kv_heads * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x;
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let pos = params.start_pos + tok;
    let k_dst = params.layer_kv_u32_offset + kv_head * head_u32_stride + pos * 128u + d2;
    let v_dst = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride + pos * 128u + d2;
    kv_cache[k_dst] = bitcast<u32>(vec2<f16>(f16(work[k_in]), f16(work[k_in + 1u])));
    kv_cache[k_dst + 1u] = bitcast<u32>(vec2<f16>(f16(work[k_in + 2u]), f16(work[k_in + 3u])));
    kv_cache[v_dst] = bitcast<u32>(vec2<f16>(f16(work[v_in]), f16(work[v_in + 1u])));
    kv_cache[v_dst + 1u] = bitcast<u32>(vec2<f16>(f16(work[v_in + 2u]), f16(work[v_in + 3u])));
    return;
  }

  let kv_head = wid.x;
  let qh0 = kv_head * gqa_ratio;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, q_heads, d);
  let q1 = load_q_vec4(tok0, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q2 = load_q_vec4(tok0, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q3 = load_q_vec4(tok0, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, q_heads, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let inv_sqrt_d: f32 = 0.0625;
  let sg_step = max(sg_size, 1u);

  var mA = vec4<f32>(-1e30);
  var lA = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  var mB = vec4<f32>(-1e30);
  var lB = vec4<f32>(0.0);
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);

  let causal0 = params.start_pos + tok0 + 1u;
  let chunk_t = (causal0 + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk_t, causal0);
  let t_end = min(t_start + chunk_t, causal0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let k_vec = vec4<f32>(k0.x, k0.y, k1.x, k1.y);

    let pA = vec4<f32>(dot(q0, k_vec), dot(q1, k_vec), dot(q2, k_vec), dot(q3, k_vec));
    let pB = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));

    let slot = ((t - t_start) & 1u) * 64u;
    let sgA = subgroupAdd(pA);
    let sgB = subgroupAdd(pB);
    red_a[slot + tid] = select(vec4<f32>(0.0), sgA, sg_inv == 0u);
    red_b[slot + tid] = select(vec4<f32>(0.0), sgB, sg_inv == 0u);
    workgroupBarrier();

    var dotA = vec4<f32>(0.0);
    var dotB = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      dotA += red_a[slot + s];
      dotB += red_b[slot + s];
    }

    let v_src = v_head_base + t * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    let v_vec = vec4<f32>(v0.x, v0.y, v1.x, v1.y);

    let scoreA = dotA * inv_sqrt_d;
    let mA_new = max(mA, scoreA);
    let alphaA = exp(mA - mA_new);
    let betaA = exp(scoreA - mA_new);
    lA = lA * alphaA + betaA;
    acc0 = acc0 * alphaA.x + v_vec * betaA.x;
    acc1 = acc1 * alphaA.y + v_vec * betaA.y;
    acc2 = acc2 * alphaA.z + v_vec * betaA.z;
    acc3 = acc3 * alphaA.w + v_vec * betaA.w;
    mA = mA_new;

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  if (has_t1 && split_id == 0u) {
    let t_tail = causal0;
    let k_src = k_head_base + t_tail * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let k_vec = vec4<f32>(k0.x, k0.y, k1.x, k1.y);

    let pB = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    let sgB = subgroupAdd(pB);
    tail_buf[tid] = select(vec4<f32>(0.0), sgB, sg_inv == 0u);
    workgroupBarrier();

    var dotB = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      dotB += tail_buf[s];
    }

    let v_src = v_head_base + t_tail * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    let v_vec = vec4<f32>(v0.x, v0.y, v1.x, v1.y);

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  write_split_out(tok0, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  if (gqa_ratio > 1u) { write_split_out(tok0, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc1, mA.y, lA.y); }
  if (gqa_ratio > 2u) { write_split_out(tok0, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc2, mA.z, lA.z); }
  if (gqa_ratio > 3u) { write_split_out(tok0, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc3, mA.w, lA.w); }
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    if (gqa_ratio > 1u) { write_split_out(tok1, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc5, mB.y, lB.y); }
    if (gqa_ratio > 2u) { write_split_out(tok1, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc6, mB.z, lB.z); }
    if (gqa_ratio > 3u) { write_split_out(tok1, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc7, mB.w, lB.w); }
  }
}
"#;

// Batched INT8 KV Split-K Prefill: mode 0 writes quantized K/V (kv_heads, M, 1), mode 1 runs GQA-tiled FlashPrefill
const WGSL_GQA_INT8_PREFILL: &str = r#"
enable subgroups;

struct GqaPrefillParams {
  mode: u32,
  layer_kv_u32_offset: u32,
  max_seq: u32,
  start_pos: u32,
  num_tokens: u32,
  buf_a_offset: u32,
  buf_b_offset: u32,
  num_splits: u32,
};

@group(0) @binding(0) var<uniform> params: GqaPrefillParams;
@group(0) @binding(1) var<storage, read> work: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> red_a: array<vec4<f32>, 128>;
var<workgroup> red_b: array<vec4<f32>, 128>;
var<workgroup> tail_buf: array<vec4<f32>, 64>;
var<workgroup> kv_max_buf: array<vec2<f32>, 64>;

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

fn write_split_out(
  tok: u32,
  q_head: u32,
  q_heads: u32,
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * q_heads + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, q_heads: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * (q_heads * 512u) + q_head * 512u + d;
  return vec4<f32>(work[b], work[b + 1u], work[b + 2u], work[b + 3u]);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let tid = lid.x;
  let d = tid * 4u;
  let sg_step = max(sg_size, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let q_heads = kv_heads * gqa_ratio;
  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = kv_heads * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x;
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let kv4 = vec4<f32>(work[k_in], work[k_in + 1u], work[k_in + 2u], work[k_in + 3u]);
    let vv4 = vec4<f32>(work[v_in], work[v_in + 1u], work[v_in + 2u], work[v_in + 3u]);

    let kmax = subgroupMax(max(max(abs(kv4.x), abs(kv4.y)), max(abs(kv4.z), abs(kv4.w))));
    let vmax = subgroupMax(max(max(abs(vv4.x), abs(vv4.y)), max(abs(vv4.z), abs(vv4.w))));
    kv_max_buf[tid] = select(vec2<f32>(0.0), vec2<f32>(kmax, vmax), sg_inv == 0u);
    workgroupBarrier();

    var km: f32 = 0.0;
    var vm: f32 = 0.0;
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      let p = kv_max_buf[s];
      km = max(km, p.x);
      vm = max(vm, p.y);
    }
    let packed_scales = pack2x16float(vec2<f32>(max(km / 127.0, 1e-6), max(vm / 127.0, 1e-6)));
    let scales = unpack2x16float(packed_scales);

    let pos = params.start_pos + tok;
    let k_dst = params.layer_kv_u32_offset + kv_head * head_u32_stride + pos * 65u;
    let v_dst = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride + pos * 65u;
    kv_cache[k_dst + tid] = pack4_i8(kv4 / scales.x);
    kv_cache[v_dst + tid] = pack4_i8(vv4 / scales.y);
    if (tid == 0u) {
      kv_cache[k_dst + 64u] = packed_scales;
    }
    return;
  }

  let kv_head = wid.x;
  let qh0 = kv_head * gqa_ratio;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, q_heads, d);
  let q1 = load_q_vec4(tok0, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q2 = load_q_vec4(tok0, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q3 = load_q_vec4(tok0, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, q_heads, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let inv_sqrt_d: f32 = 0.0625;

  var mA = vec4<f32>(-1e30);
  var lA = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  var mB = vec4<f32>(-1e30);
  var lB = vec4<f32>(0.0);
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);

  let causal0 = params.start_pos + tok0 + 1u;
  let chunk_t = (causal0 + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk_t, causal0);
  let t_end = min(t_start + chunk_t, causal0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let k_vec = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    let pA = vec4<f32>(dot(q0, k_vec), dot(q1, k_vec), dot(q2, k_vec), dot(q3, k_vec));
    let pB = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));

    let slot = ((t - t_start) & 1u) * 64u;
    let sgA = subgroupAdd(pA);
    let sgB = subgroupAdd(pB);
    red_a[slot + tid] = select(vec4<f32>(0.0), sgA, sg_inv == 0u);
    red_b[slot + tid] = select(vec4<f32>(0.0), sgB, sg_inv == 0u);
    workgroupBarrier();

    var dotA = vec4<f32>(0.0);
    var dotB = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      dotA += red_a[slot + s];
      dotB += red_b[slot + s];
    }

    let v_src = v_head_base + t * 65u;
    let v_vec = unpack4_i8(kv_cache[v_src + tid]) * scales.y;

    let scoreA = dotA * inv_sqrt_d;
    let mA_new = max(mA, scoreA);
    let alphaA = exp(mA - mA_new);
    let betaA = exp(scoreA - mA_new);
    lA = lA * alphaA + betaA;
    acc0 = acc0 * alphaA.x + v_vec * betaA.x;
    acc1 = acc1 * alphaA.y + v_vec * betaA.y;
    acc2 = acc2 * alphaA.z + v_vec * betaA.z;
    acc3 = acc3 * alphaA.w + v_vec * betaA.w;
    mA = mA_new;

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  if (has_t1 && split_id == 0u) {
    let t_tail = causal0;
    let k_src = k_head_base + t_tail * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let k_vec = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    let pB = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    let sgB = subgroupAdd(pB);
    tail_buf[tid] = select(vec4<f32>(0.0), sgB, sg_inv == 0u);
    workgroupBarrier();

    var dotB = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      dotB += tail_buf[s];
    }

    let v_src = v_head_base + t_tail * 65u;
    let v_vec = unpack4_i8(kv_cache[v_src + tid]) * scales.y;

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  write_split_out(tok0, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  if (gqa_ratio > 1u) { write_split_out(tok0, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc1, mA.y, lA.y); }
  if (gqa_ratio > 2u) { write_split_out(tok0, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc2, mA.z, lA.z); }
  if (gqa_ratio > 3u) { write_split_out(tok0, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc3, mA.w, lA.w); }
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    if (gqa_ratio > 1u) { write_split_out(tok1, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc5, mB.y, lB.y); }
    if (gqa_ratio > 2u) { write_split_out(tok1, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc6, mB.z, lB.z); }
    if (gqa_ratio > 3u) { write_split_out(tok1, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc7, mB.w, lB.w); }
  }
}
"#;

// Chunkwise Parallel Gated DeltaNet WY Prep (C <= 32 tokens per sub-chunk, zero local-array spills)
// Dispatch (32, 1, 1) with @workgroup_size(128) -> 1 workgroup per v_head
const WGSL_GDN_CHUNK_PREP: &str = r#"
enable subgroups;

struct GdnPrepParams {
  num_tokens: u32,
  tok_start: u32,
  aux_dn_offset: u32,
  qkv_offset: u32,
  ab_offset: u32,
  scratch_offset: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GdnPrepParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> work: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

var<workgroup> sh_M: array<f32, 1024>;
var<workgroup> sh_a: array<f32, 1024>;
var<workgroup> sh_b: array<f32, 1024>;
var<workgroup> sh_red_q: array<f32, 256>;
var<workgroup> sh_red_k: array<f32, 256>;
var<workgroup> sh_G: array<f32, 32>;
var<workgroup> sh_beta: array<f32, 32>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let C = params.num_tokens;
  if (C == 0u) {
    return;
  }
  let v_heads = select(params._pad0 & 0xFFFFu, 32u, params._pad0 == 0u);
  let k_heads = select(params._pad0 >> 16u, 16u, params._pad0 == 0u);
  let ratio = max(v_heads / max(k_heads, 1u), 1u);
  let qk_dim = k_heads * 128u;
  let v_dim = v_heads * 128u;
  let conv_dim = qk_dim * 2u + v_dim;

  let v_head = wid.x;
  let j = lid.x;
  let k_head = v_head / ratio;
  let sg_step = max(sg_size, 1u);

  let G_off = params.scratch_offset + 0u;
  let QK_off = params.scratch_offset + 1024u;
  let W_off = params.scratch_offset + 33792u;
  let Q_off = params.scratch_offset + 164864u;
  let K_dec_off = params.scratch_offset + 295936u;
  let U_off = params.scratch_offset + 427008u;

  let a_neg = aux[params.aux_dn_offset + v_head];
  let dt_b = aux[params.aux_dn_offset + v_heads + v_head];
  var g_cum: f32 = 0.0;

  // Step 1: L2-normalize Q/K directly to scratch + compute beta_t, cumulative G_t
  for (var t: u32 = 0u; t < C; t = t + 1u) {
    let qkv_t = params.qkv_offset + (params.tok_start + t) * conv_dim;
    let q_raw = work[qkv_t + k_head * 128u + j];
    let k_raw = work[qkv_t + qk_dim + k_head * 128u + j];

    let slot = (t & 1u) * 128u;
    let sg_q = subgroupAdd(q_raw * q_raw);
    let sg_k = subgroupAdd(k_raw * k_raw);
    sh_red_q[slot + j] = select(0.0, sg_q, sg_inv == 0u);
    sh_red_k[slot + j] = select(0.0, sg_k, sg_inv == 0u);
    workgroupBarrier();

    var sq_q: f32 = 0.0;
    var sq_k: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      sq_q += sh_red_q[slot + s];
      sq_k += sh_red_k[slot + s];
    }
    let inv_q = inverseSqrt(sq_q + 1e-6) * 0.08838834764831845;
    let inv_k = inverseSqrt(sq_k + 1e-6);
    let sc_idx = (v_head * 128u + j) * 32u + t;
    scratch[Q_off + sc_idx] = q_raw * inv_q;
    scratch[K_dec_off + sc_idx] = k_raw * inv_k;

    if (j == 0u) {
      let ab_t = params.ab_offset + (params.tok_start + t) * (v_heads * 2u);
      let b_raw = work[ab_t + v_head];
      let a_raw = work[ab_t + v_heads + v_head];
      sh_beta[t] = 1.0 / (1.0 + exp(-b_raw));
      let u = a_raw + dt_b;
      let sp = select(log(1.0 + exp(u)), u, u > 20.0);
      g_cum += a_neg * sp;
      sh_G[t] = g_cum;
      scratch[G_off + v_head * 32u + t] = g_cum;
    }
  }

  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let idx = k * 128u + j;
    let tr = idx >> 5u;
    let tc = idx & 31u;
    sh_M[idx] = select(0.0, 1.0, tr == tc);
  }
  storageBarrier();
  workgroupBarrier();

  // Step 2: Compute A[t, r] and QK_causal[t, r] across 4 i-tiles (32 dims/tile, 8 barriers total)
  let tx = j & 31u;
  let r0 = (j >> 5u) * 8u;
  var kk0: f32 = 0.0; var kk1: f32 = 0.0; var kk2: f32 = 0.0; var kk3: f32 = 0.0;
  var kk4: f32 = 0.0; var kk5: f32 = 0.0; var kk6: f32 = 0.0; var kk7: f32 = 0.0;
  var qk0: f32 = 0.0; var qk1: f32 = 0.0; var qk2: f32 = 0.0; var qk3: f32 = 0.0;
  var qk4: f32 = 0.0; var qk5: f32 = 0.0; var qk6: f32 = 0.0; var qk7: f32 = 0.0;

  for (var i_tile: u32 = 0u; i_tile < 4u; i_tile = i_tile + 1u) {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let idx = k * 128u + j;
      let i_sub = idx >> 5u;
      let t_sub = idx & 31u;
      if (t_sub < C) {
        let sc_idx = (v_head * 128u + i_tile * 32u + i_sub) * 32u + t_sub;
        sh_a[idx] = scratch[K_dec_off + sc_idx];
        sh_b[idx] = scratch[Q_off + sc_idx];
      } else {
        sh_a[idx] = 0.0;
        sh_b[idx] = 0.0;
      }
    }
    workgroupBarrier();

    for (var ii: u32 = 0u; ii < 32u; ii = ii + 1u) {
      let rb = ii * 32u;
      let k_tx = sh_a[rb + tx];
      let q_tx = sh_b[rb + tx];
      let kr0 = sh_a[rb + r0 + 0u]; let kr1 = sh_a[rb + r0 + 1u];
      let kr2 = sh_a[rb + r0 + 2u]; let kr3 = sh_a[rb + r0 + 3u];
      let kr4 = sh_a[rb + r0 + 4u]; let kr5 = sh_a[rb + r0 + 5u];
      let kr6 = sh_a[rb + r0 + 6u]; let kr7 = sh_a[rb + r0 + 7u];
      kk0 += k_tx * kr0; kk1 += k_tx * kr1; kk2 += k_tx * kr2; kk3 += k_tx * kr3;
      kk4 += k_tx * kr4; kk5 += k_tx * kr5; kk6 += k_tx * kr6; kk7 += k_tx * kr7;
      qk0 += q_tx * kr0; qk1 += q_tx * kr1; qk2 += q_tx * kr2; qk3 += q_tx * kr3;
      qk4 += q_tx * kr4; qk5 += q_tx * kr5; qk6 += q_tx * kr6; qk7 += q_tx * kr7;
    }
    workgroupBarrier();
  }

  if (tx < C) {
    let gt = sh_G[tx];
    let bt = sh_beta[tx];
    let qk_row = (v_head * 32u + tx) * 32u;
    if (r0 + 0u <= tx) { let d = exp(gt - sh_G[r0 + 0u]); scratch[QK_off + qk_row + r0 + 0u] = d * qk0; if (r0 + 0u < tx) { sh_M[tx * 32u + r0 + 0u] = bt * d * kk0; } }
    if (r0 + 1u <= tx) { let d = exp(gt - sh_G[r0 + 1u]); scratch[QK_off + qk_row + r0 + 1u] = d * qk1; if (r0 + 1u < tx) { sh_M[tx * 32u + r0 + 1u] = bt * d * kk1; } }
    if (r0 + 2u <= tx) { let d = exp(gt - sh_G[r0 + 2u]); scratch[QK_off + qk_row + r0 + 2u] = d * qk2; if (r0 + 2u < tx) { sh_M[tx * 32u + r0 + 2u] = bt * d * kk2; } }
    if (r0 + 3u <= tx) { let d = exp(gt - sh_G[r0 + 3u]); scratch[QK_off + qk_row + r0 + 3u] = d * qk3; if (r0 + 3u < tx) { sh_M[tx * 32u + r0 + 3u] = bt * d * kk3; } }
    if (r0 + 4u <= tx) { let d = exp(gt - sh_G[r0 + 4u]); scratch[QK_off + qk_row + r0 + 4u] = d * qk4; if (r0 + 4u < tx) { sh_M[tx * 32u + r0 + 4u] = bt * d * kk4; } }
    if (r0 + 5u <= tx) { let d = exp(gt - sh_G[r0 + 5u]); scratch[QK_off + qk_row + r0 + 5u] = d * qk5; if (r0 + 5u < tx) { sh_M[tx * 32u + r0 + 5u] = bt * d * kk5; } }
    if (r0 + 6u <= tx) { let d = exp(gt - sh_G[r0 + 6u]); scratch[QK_off + qk_row + r0 + 6u] = d * qk6; if (r0 + 6u < tx) { sh_M[tx * 32u + r0 + 6u] = bt * d * kk6; } }
    if (r0 + 7u <= tx) { let d = exp(gt - sh_G[r0 + 7u]); scratch[QK_off + qk_row + r0 + 7u] = d * qk7; if (r0 + 7u < tx) { sh_M[tx * 32u + r0 + 7u] = bt * d * kk7; } }
  }
  workgroupBarrier();

  // Step 3: Forward substitution (I + A) T = I in-place in sh_M
  for (var t: u32 = 1u; t < C; t = t + 1u) {
    var val: f32 = 0.0;
    if (j < t) {
      var acc: f32 = sh_M[t * 32u + j];
      for (var m: u32 = j + 1u; m < t; m = m + 1u) {
        acc += sh_M[t * 32u + m] * sh_M[m * 32u + j];
      }
      val = -acc;
    }
    workgroupBarrier();
    if (j < t) {
      sh_M[t * 32u + j] = val;
    }
    workgroupBarrier();
  }

  // Step 4: Compute W, U, K_dec in scratch via coalesced shared-memory tiles (zero inner-loop exp/global loads)
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let idx = k * 128u + j;
    let t = idx >> 5u;
    let r = idx & 31u;
    if (t < C && r <= t) {
      sh_M[idx] = sh_M[idx] * sh_beta[r];
    }
  }
  workgroupBarrier();
  if (j < C) {
    sh_beta[j] = exp(sh_G[j]);
  }
  workgroupBarrier();

  let t_grp = j >> 5u;
  let j_sub = j & 31u;
  for (var j_tile: u32 = 0u; j_tile < 4u; j_tile = j_tile + 1u) {
    let j_base = j_tile * 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let r = k * 4u + t_grp;
      if (r < C) {
        sh_a[r * 32u + j_sub] = scratch[K_dec_off + (v_head * 128u + j_base + j_sub) * 32u + r] * sh_beta[r];
        sh_b[r * 32u + j_sub] = work[params.qkv_offset + (params.tok_start + r) * conv_dim + qk_dim * 2u + v_head * 128u + j_base + j_sub];
      }
    }
    workgroupBarrier();

    for (var t: u32 = t_grp; t < C; t = t + 4u) {
      var w_acc: f32 = 0.0;
      var u_acc: f32 = 0.0;
      let m_row = t * 32u;
      for (var r: u32 = 0u; r <= t; r = r + 1u) {
        let tr_br = sh_M[m_row + r];
        let s_idx = r * 32u + j_sub;
        w_acc += tr_br * sh_a[s_idx];
        u_acc += tr_br * sh_b[s_idx];
      }
      scratch[W_off + (v_head * 128u + j_base + j_sub) * 32u + t] = w_acc;
      scratch[U_off + (v_head * 32u + t) * 128u + j_base + j_sub] = u_acc;
    }
    workgroupBarrier();
  }

  let g_last = sh_G[C - 1u];
  let k_base_j = K_dec_off + (v_head * 128u + j) * 32u;
  for (var t: u32 = 0u; t < C; t = t + 1u) {
    scratch[k_base_j + t] = exp(g_last - sh_G[t]) * scratch[k_base_j + t];
  }
}
"#;

// Chunkwise Parallel Gated DeltaNet Apply & Gated RMSNorm (zero local-array spills)
// mode 0: Dispatch (32, 8, 1) with @workgroup_size(128) -> 8 i-slices x 16 j-cols, reads & writes dn_state ONCE
// mode 1: Dispatch (32, M, 1) with @workgroup_size(128) -> Gated RMSNorm over attn_out
const WGSL_GDN_CHUNK_APPLY: &str = r#"
enable subgroups;

struct GdnApplyParams {
  mode: u32,
  num_tokens: u32,
  tok_start: u32,
  state_f32_offset: u32,
  aux_dn_offset: u32,
  z_offset: u32,
  out_offset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: GdnApplyParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> work: array<f32>;
@group(0) @binding(3) var<storage, read> scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> dn_state: array<f32>;
@group(0) @binding(5) var<storage, read_write> attn_out: array<f32>;

var<workgroup> sh_a: array<f32, 512>;
var<workgroup> sh_b: array<f32, 512>;
var<workgroup> sh_red: array<f32, 512>;
var<workgroup> sh_delta: array<f32, 512>;
var<workgroup> sh_raw: array<f32, 512>;
var<workgroup> sh_qk: array<f32, 1024>;
var<workgroup> sh_G: array<f32, 32>;

fn dot16_a(base: u32, sv0: vec4<f32>, sv1: vec4<f32>, sv2: vec4<f32>, sv3: vec4<f32>) -> f32 {
  let v0 = vec4<f32>(sh_a[base + 0u], sh_a[base + 1u], sh_a[base + 2u], sh_a[base + 3u]);
  let v1 = vec4<f32>(sh_a[base + 4u], sh_a[base + 5u], sh_a[base + 6u], sh_a[base + 7u]);
  let v2 = vec4<f32>(sh_a[base + 8u], sh_a[base + 9u], sh_a[base + 10u], sh_a[base + 11u]);
  let v3 = vec4<f32>(sh_a[base + 12u], sh_a[base + 13u], sh_a[base + 14u], sh_a[base + 15u]);
  return dot(sv0, v0) + dot(sv1, v1) + dot(sv2, v2) + dot(sv3, v3);
}

fn dot16_b(base: u32, sv0: vec4<f32>, sv1: vec4<f32>, sv2: vec4<f32>, sv3: vec4<f32>) -> f32 {
  let v0 = vec4<f32>(sh_b[base + 0u], sh_b[base + 1u], sh_b[base + 2u], sh_b[base + 3u]);
  let v1 = vec4<f32>(sh_b[base + 4u], sh_b[base + 5u], sh_b[base + 6u], sh_b[base + 7u]);
  let v2 = vec4<f32>(sh_b[base + 8u], sh_b[base + 9u], sh_b[base + 10u], sh_b[base + 11u]);
  let v3 = vec4<f32>(sh_b[base + 12u], sh_b[base + 13u], sh_b[base + 14u], sh_b[base + 15u]);
  return dot(sv0, v0) + dot(sv1, v1) + dot(sv2, v2) + dot(sv3, v3);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_step = max(sg_size, 1u);
  let v_heads = select(params._pad0 & 0xFFFFu, 32u, params._pad0 == 0u);
  let v_dim = v_heads * 128u;

  if (params.mode == 1u) {
    let v_head = wid.x;
    let tok = wid.y;
    let j = lid.x;
    let out_idx = params.out_offset + tok * v_dim + v_head * 128u + j;
    let out_val = attn_out[out_idx];
    let sg_sq = subgroupAdd(out_val * out_val);
    sh_red[j] = select(0.0, sg_sq, sg_inv == 0u);
    workgroupBarrier();
    var sq: f32 = 0.0;
    for (var s: u32 = 0u; s < 128u; s = s + sg_step) {
      sq += sh_red[s];
    }
    let inv_rms = inverseSqrt(sq / 128.0 + 1e-6);
    let z_val = work[params.z_offset + tok * v_dim + v_head * 128u + j];
    let z_silu = z_val / (1.0 + exp(-z_val));
    let gdn_norm_w = aux[params.aux_dn_offset + v_heads * 2u + j];
    attn_out[out_idx] = (out_val * inv_rms) * z_silu * gdn_norm_w;
    return;
  }

  let C = params.num_tokens;
  if (C == 0u) {
    return;
  }
  let v_head = wid.x;
  let j_tile = wid.y;
  let tid = lid.x;
  let j_local = tid & 15u;       // 0..15
  let j = j_tile * 16u + j_local;
  let i_slice = tid >> 4u;       // 0..7
  let i_base = i_slice * 16u;    // 0, 16, ..., 112

  let G_off = 0u;
  let QK_off = 1024u;
  let W_off = 33792u;
  let Q_off = 164864u;
  let K_dec_off = 295936u;
  let U_off = 427008u;

  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let idx = k * 128u + tid;
    sh_qk[idx] = scratch[QK_off + v_head * 1024u + idx];
  }
  if (tid < C) {
    sh_G[tid] = scratch[G_off + v_head * 32u + tid];
  }

  let s_base = params.state_f32_offset + v_head * 16384u + i_base * 128u + j;
  var sv0 = vec4<f32>(dn_state[s_base + 0u * 128u], dn_state[s_base + 1u * 128u], dn_state[s_base + 2u * 128u], dn_state[s_base + 3u * 128u]);
  var sv1 = vec4<f32>(dn_state[s_base + 4u * 128u], dn_state[s_base + 5u * 128u], dn_state[s_base + 6u * 128u], dn_state[s_base + 7u * 128u]);
  var sv2 = vec4<f32>(dn_state[s_base + 8u * 128u], dn_state[s_base + 9u * 128u], dn_state[s_base + 10u * 128u], dn_state[s_base + 11u * 128u]);
  var sv3 = vec4<f32>(dn_state[s_base + 12u * 128u], dn_state[s_base + 13u * 128u], dn_state[s_base + 14u * 128u], dn_state[s_base + 15u * 128u]);

  // Process W*S and Q*S in 4-token blocks (zero private array spills)
  for (var t_blk: u32 = 0u; t_blk < C; t_blk = t_blk + 4u) {
    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
      let t_abs = t_blk + tt;
      let dst_idx = tt * 128u + tid;
      if (t_abs < C) {
        let sc_idx = (v_head * 128u + tid) * 32u + t_abs;
        sh_a[dst_idx] = scratch[W_off + sc_idx];
        sh_b[dst_idx] = scratch[Q_off + sc_idx];
      } else {
        sh_a[dst_idx] = 0.0;
        sh_b[dst_idx] = 0.0;
      }
    }
    workgroupBarrier();

    let ws0 = dot16_a(0u * 128u + i_base, sv0, sv1, sv2, sv3);
    let ws1 = dot16_a(1u * 128u + i_base, sv0, sv1, sv2, sv3);
    let ws2 = dot16_a(2u * 128u + i_base, sv0, sv1, sv2, sv3);
    let ws3 = dot16_a(3u * 128u + i_base, sv0, sv1, sv2, sv3);
    let red_base = (i_slice * 16u + j_local) * 4u;
    sh_red[red_base + 0u] = ws0;
    sh_red[red_base + 1u] = ws1;
    sh_red[red_base + 2u] = ws2;
    sh_red[red_base + 3u] = ws3;
    workgroupBarrier();

    if (i_slice == 0u) {
      for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
        let t_abs = t_blk + tt;
        if (t_abs < C) {
          var w_sum: f32 = 0.0;
          for (var sl: u32 = 0u; sl < 8u; sl = sl + 1u) {
            w_sum += sh_red[(sl * 16u + j_local) * 4u + tt];
          }
          let u_val = scratch[U_off + (v_head * 32u + t_abs) * 128u + j];
          sh_delta[t_abs * 16u + j_local] = u_val - w_sum;
        }
      }
    }
    workgroupBarrier();

    let qs0 = dot16_b(0u * 128u + i_base, sv0, sv1, sv2, sv3);
    let qs1 = dot16_b(1u * 128u + i_base, sv0, sv1, sv2, sv3);
    let qs2 = dot16_b(2u * 128u + i_base, sv0, sv1, sv2, sv3);
    let qs3 = dot16_b(3u * 128u + i_base, sv0, sv1, sv2, sv3);
    sh_red[red_base + 0u] = qs0;
    sh_red[red_base + 1u] = qs1;
    sh_red[red_base + 2u] = qs2;
    sh_red[red_base + 3u] = qs3;
    workgroupBarrier();

    if (i_slice == 0u) {
      for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
        let t_abs = t_blk + tt;
        if (t_abs < C) {
          var q_sum: f32 = 0.0;
          for (var sl: u32 = 0u; sl < 8u; sl = sl + 1u) {
            q_sum += sh_red[(sl * 16u + j_local) * 4u + tt];
          }
          sh_raw[t_abs * 16u + j_local] = q_sum;
        }
      }
    }
    workgroupBarrier();
  }

  if (i_slice == 0u) {
    for (var t: u32 = 0u; t < C; t = t + 1u) {
      var causal_acc: f32 = 0.0;
      let qk_row = t * 32u;
      for (var r: u32 = 0u; r <= t; r = r + 1u) {
        causal_acc += sh_qk[qk_row + r] * sh_delta[r * 16u + j_local];
      }
      let raw_out = exp(sh_G[t]) * sh_raw[t * 16u + j_local] + causal_acc;
      attn_out[params.out_offset + (params.tok_start + t) * v_dim + v_head * 128u + j] = raw_out;
    }
  }

  let g_last = exp(sh_G[C - 1u]);
  sv0 = sv0 * g_last;
  sv1 = sv1 * g_last;
  sv2 = sv2 * g_last;
  sv3 = sv3 * g_last;

  for (var t_blk: u32 = 0u; t_blk < C; t_blk = t_blk + 4u) {
    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
      let t_abs = t_blk + tt;
      let dst_idx = tt * 128u + tid;
      if (t_abs < C) {
        sh_a[dst_idx] = scratch[K_dec_off + (v_head * 128u + tid) * 32u + t_abs];
      } else {
        sh_a[dst_idx] = 0.0;
      }
    }
    workgroupBarrier();

    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
      let t_abs = t_blk + tt;
      if (t_abs < C) {
        let d_t = sh_delta[t_abs * 16u + j_local];
        let kb = tt * 128u + i_base;
        sv0 += vec4<f32>(sh_a[kb + 0u], sh_a[kb + 1u], sh_a[kb + 2u], sh_a[kb + 3u]) * d_t;
        sv1 += vec4<f32>(sh_a[kb + 4u], sh_a[kb + 5u], sh_a[kb + 6u], sh_a[kb + 7u]) * d_t;
        sv2 += vec4<f32>(sh_a[kb + 8u], sh_a[kb + 9u], sh_a[kb + 10u], sh_a[kb + 11u]) * d_t;
        sv3 += vec4<f32>(sh_a[kb + 12u], sh_a[kb + 13u], sh_a[kb + 14u], sh_a[kb + 15u]) * d_t;
      }
    }
    workgroupBarrier();
  }

  dn_state[s_base + 0u * 128u] = sv0.x; dn_state[s_base + 1u * 128u] = sv0.y; dn_state[s_base + 2u * 128u] = sv0.z; dn_state[s_base + 3u * 128u] = sv0.w;
  dn_state[s_base + 4u * 128u] = sv1.x; dn_state[s_base + 5u * 128u] = sv1.y; dn_state[s_base + 6u * 128u] = sv1.z; dn_state[s_base + 7u * 128u] = sv1.w;
  dn_state[s_base + 8u * 128u] = sv2.x; dn_state[s_base + 9u * 128u] = sv2.y; dn_state[s_base + 10u * 128u] = sv2.z; dn_state[s_base + 11u * 128u] = sv2.w;
  dn_state[s_base + 12u * 128u] = sv3.x; dn_state[s_base + 13u * 128u] = sv3.y; dn_state[s_base + 14u * 128u] = sv3.z; dn_state[s_base + 15u * 128u] = sv3.w;
}
"#;

// Shader 21: Fused Q4 Gate + Up + SwiGLU (8 rows/WG, 16 lanes/row, coalesced x_sh staging)
// Passes wGate in w_u32_offset and wUp in _pad0, writing silu(gate) * up directly to attnOutBuf.
const WGSL_Q4_GATE_UP_SWIGLU: &str = r#"
enable f16;
enable subgroups;

struct GateUpParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GateUpParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn factored_q4_word(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn factored_q4_block(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f16>, x0b: vec4<f16>,
  x1a: vec4<f16>, x1b: vec4<f16>,
  x2a: vec4<f16>, x2b: vec4<f16>,
  x3a: vec4<f16>, x3b: vec4<f16>,
  sum_x: f32,
) -> f32 {
  let blk_meta = unpack2x16float(meta_word);
  let sum_qx = factored_q4_word(w0, x0a, x0b)
    + factored_q4_word(w1, x1a, x1b)
    + factored_q4_word(w2, x2a, x2b)
    + factored_q4_word(w3, x3a, x3b);
  return blk_meta.x * sum_qx + blk_meta.y * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;  // 0..7
  let lane = tid & 15u; // 0..15
  let r_raw = wid.x * 8u + grp;
  let r = min(r_raw, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let og = params.w_u32_offset + r * row_stride;
  let ou = params._pad0 + r * row_stride;

  var acc_g: f32 = 0.0;
  var acc_u: f32 = 0.0;
  let ones = vec4<f16>(1.0h);

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 80u) {
    let tile_blocks = min(80u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[i] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    }
    workgroupBarrier();

    for (var b_rel: u32 = lane; b_rel < tile_blocks; b_rel = b_rel + 16u) {
      let xb = b_rel << 3u;
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
        + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
        + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
        + f32(dot(ones, x3a)) + f32(dot(ones, x3b));

      let b5 = (k_tile + b_rel) * 5u;
      let ig = og + b5;
      let iu = ou + b5;
      acc_g += factored_q4_block(weights[ig], weights[ig + 1u], weights[ig + 2u], weights[ig + 3u], weights[ig + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
      acc_u += factored_q4_block(weights[iu], weights[iu + 1u], weights[iu + 2u], weights[iu + 3u], weights[iu + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
    }

    if (k_tile + 80u < blocks) {
      workgroupBarrier();
    }
  }

  var sum2 = vec2<f32>(acc_g, acc_u);
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

  if (lane == 0u && r_raw < params.rows) {
    let g = sum2.x;
    let u = sum2.y;
    let out_val = (g / (1.0 + exp(-g))) * u;
    let out_i = params.out_offset + r_raw;
    y[out_i] = select(out_val, y[out_i] + out_val, params.add_residual != 0u);
  }
}
"#;

// Shader 22: Wide-K 32-Lane Factored Q4 GEMV (4 rows/WG, 32 lanes/row, ping-pong coalesced x_sh)
// Ideal for K=4096 (128 blocks = 32*4) and K=9216 (288 blocks = 32*9): 640 WGs on rows=2560!
const WGSL_Q4_GEMV_WIDE_K: &str = r#"
enable f16;
enable subgroups;

struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 512>;
var<workgroup> red_sh: array<f32, 128>;

fn factored_q4_word(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let row_in_wg = tid >> 5u; // 0..3
  let lane = tid & 31u;      // 0..31
  let r_raw = wid.x * 4u + row_in_wg;
  let r_safe = min(r_raw, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_base = params.w_u32_offset + r_safe * blocks * 5u;

  var acc: f32 = 0.0;
  let ones = vec4<f16>(1.0h);
  var buf_idx: u32 = 0u;

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 32u) {
    let tile_blocks = min(32u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let base_sh = buf_idx * 256u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[base_sh + i] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    }
    workgroupBarrier();

    if (lane < tile_blocks) {
      let xb = base_sh + (lane << 3u);
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
        + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
        + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
        + f32(dot(ones, x3a)) + f32(dot(ones, x3b));

      let bi = row_base + (k_tile + lane) * 5u;
      let blk_meta = unpack2x16float(weights[bi + 4u]);
      let sum_qx = factored_q4_word(weights[bi], x0a, x0b)
        + factored_q4_word(weights[bi + 1u], x1a, x1b)
        + factored_q4_word(weights[bi + 2u], x2a, x2b)
        + factored_q4_word(weights[bi + 3u], x3a, x3b);
      acc += blk_meta.x * sum_qx + blk_meta.y * sum_x;
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
      let base = row_in_wg * 32u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        s_acc += red_sh[base + l];
      }
      acc = s_acc;
    }
  }

  if (lane == 0u && r_raw < params.rows) {
    let out_i = params.out_offset + r_raw;
    y[out_i] = select(acc, y[out_i] + acc, params.add_residual != 0u);
  }
}
"#;

// Shader 23: DP4a 4-Q-Head GQA-Grouped INT8 KV Decode (requires packed_4x8_integer_dot_product)
const WGSL_GQA_INT8_SPLIT_DP4A: &str = r#"
requires packed_4x8_integer_dot_product;
enable subgroups;

struct GqaSplitParams {
  layer_kv_u32_offset: u32,
  max_seq: u32,
  seq_pos: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  num_splits: u32,
  q_head_stride: u32,
};

@group(0) @binding(0) var<uniform> params: GqaSplitParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;
@group(0) @binding(4) var<storage, read> pos_buf: array<u32>;

var<workgroup> q_max_buf: array<vec4<f32>, 64>;
var<workgroup> red_buf: array<vec4<f32>, 128>;
var<workgroup> cur_kv_max: array<vec2<f32>, 64>;
var<workgroup> cur_dot_buf: array<vec4<f32>, 64>;

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

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let kv_head = wid.x;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let qh0 = kv_head * gqa_ratio;
  let split_id = wid.y;
  let tid = lid.x;
  let d = tid * 4u;
  let seq_pos = select(params.seq_pos, pos_buf[1], params.seq_pos == 0xFFFFFFFFu);

  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = kv_heads * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let qb0 = params.q_offset + (qh0 + 0u) * q_stride + d;
  let qb1 = params.q_offset + (qh0 + min(1u, gqa_ratio - 1u)) * q_stride + d;
  let qb2 = params.q_offset + (qh0 + min(2u, gqa_ratio - 1u)) * q_stride + d;
  let qb3 = params.q_offset + (qh0 + min(3u, gqa_ratio - 1u)) * q_stride + d;
  let q0 = vec4<f32>(act[qb0], act[qb0 + 1u], act[qb0 + 2u], act[qb0 + 3u]);
  let q1 = vec4<f32>(act[qb1], act[qb1 + 1u], act[qb1 + 2u], act[qb1 + 3u]);
  let q2 = vec4<f32>(act[qb2], act[qb2 + 1u], act[qb2 + 2u], act[qb2 + 3u]);
  let q3 = vec4<f32>(act[qb3], act[qb3 + 1u], act[qb3 + 2u], act[qb3 + 3u]);

  let inv_sqrt_d: f32 = 0.0625;
  let sg_step = max(sg_size, 1u);

  // Quantize all 4 Q heads once into packed i8
  let qm0 = subgroupMax(max(max(abs(q0.x), abs(q0.y)), max(abs(q0.z), abs(q0.w))));
  let qm1 = subgroupMax(max(max(abs(q1.x), abs(q1.y)), max(abs(q1.z), abs(q1.w))));
  let qm2 = subgroupMax(max(max(abs(q2.x), abs(q2.y)), max(abs(q2.z), abs(q2.w))));
  let qm3 = subgroupMax(max(max(abs(q3.x), abs(q3.y)), max(abs(q3.z), abs(q3.w))));
  q_max_buf[tid] = select(vec4<f32>(0.0), vec4<f32>(qm0, qm1, qm2, qm3), sg_inv == 0u);
  workgroupBarrier();
  var qm_all = vec4<f32>(0.0);
  for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
    qm_all = max(qm_all, q_max_buf[s]);
  }
  let q_scales = max(qm_all / 127.0, vec4<f32>(1e-6));
  let q0_i8 = pack4_i8(q0 / q_scales.x);
  let q1_i8 = pack4_i8(q1 / q_scales.y);
  let q2_i8 = pack4_i8(q2 / q_scales.z);
  let q3_i8 = pack4_i8(q3 / q_scales.w);

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let chunk = (seq_pos + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, seq_pos);
  let t_end = min(t_start + chunk, seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let k_word = kv_cache[k_src + tid];

    let p4 = vec4<f32>(
      f32(dot4I8Packed(q0_i8, k_word)),
      f32(dot4I8Packed(q1_i8, k_word)),
      f32(dot4I8Packed(q2_i8, k_word)),
      f32(dot4I8Packed(q3_i8, k_word)),
    );
    let slot = ((t - t_start) & 1u) * 64u;
    let sg_dot = subgroupAdd(p4);
    red_buf[slot + tid] = select(vec4<f32>(0.0), sg_dot, sg_inv == 0u);
    workgroupBarrier();

    var full_dot = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      full_dot += red_buf[slot + s];
    }

    let score = full_dot * (q_scales * scales.x * inv_sqrt_d);
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let v_src = v_head_base + t * 65u;
    let vi = unpack4_i8(kv_cache[v_src + tid]) * scales.y;
    acc0 = acc0 * alpha.x + vi * beta.x;
    acc1 = acc1 * alpha.y + vi * beta.y;
    acc2 = acc2 * alpha.z + vi * beta.z;
    acc3 = acc3 * alpha.w + vi * beta.w;
    m4 = m_new;
  }

  if (split_id == 0u) {
    let k_in = params.k_offset + kv_head * 256u + d;
    let v_in = params.v_offset + kv_head * 256u + d;
    let kv4 = vec4<f32>(act[k_in], act[k_in + 1u], act[k_in + 2u], act[k_in + 3u]);
    let vv4 = vec4<f32>(act[v_in], act[v_in + 1u], act[v_in + 2u], act[v_in + 3u]);

    let kmax = subgroupMax(max(max(abs(kv4.x), abs(kv4.y)), max(abs(kv4.z), abs(kv4.w))));
    let vmax = subgroupMax(max(max(abs(vv4.x), abs(vv4.y)), max(abs(vv4.z), abs(vv4.w))));
    cur_kv_max[tid] = select(vec2<f32>(0.0), vec2<f32>(kmax, vmax), sg_inv == 0u);
    workgroupBarrier();

    var km: f32 = 0.0;
    var vm: f32 = 0.0;
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      let p = cur_kv_max[s];
      km = max(km, p.x);
      vm = max(vm, p.y);
    }
    let packed_scales = pack2x16float(vec2<f32>(max(km / 127.0, 1e-6), max(vm / 127.0, 1e-6)));
    let scales = unpack2x16float(packed_scales);
    let qk = pack4_i8(kv4 / scales.x);
    let qv = pack4_i8(vv4 / scales.y);

    let k_dst = k_head_base + seq_pos * 65u;
    let v_dst = v_head_base + seq_pos * 65u;
    kv_cache[k_dst + tid] = qk;
    kv_cache[v_dst + tid] = qv;
    if (tid == 0u) {
      kv_cache[k_dst + 64u] = packed_scales;
    }

    let cur_p4 = vec4<f32>(
      f32(dot4I8Packed(q0_i8, qk)),
      f32(dot4I8Packed(q1_i8, qk)),
      f32(dot4I8Packed(q2_i8, qk)),
      f32(dot4I8Packed(q3_i8, qk)),
    );
    let cur_sg = subgroupAdd(cur_p4);
    cur_dot_buf[tid] = select(vec4<f32>(0.0), cur_sg, sg_inv == 0u);
    workgroupBarrier();

    var cur_dot = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      cur_dot += cur_dot_buf[s];
    }

    let score = cur_dot * (q_scales * scales.x * inv_sqrt_d);
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;
    let v_deq = unpack4_i8(qv) * scales.y;
    acc0 = acc0 * alpha.x + v_deq * beta.x;
    acc1 = acc1 * alpha.y + v_deq * beta.y;
    acc2 = acc2 * alpha.z + v_deq * beta.z;
    acc3 = acc3 * alpha.w + v_deq * beta.w;
    m4 = m_new;
  }

  let ob0 = ((qh0 + 0u) * num_splits + split_id) * 260u;
  split_out[ob0 + d] = acc0.x; split_out[ob0 + d + 1u] = acc0.y; split_out[ob0 + d + 2u] = acc0.z; split_out[ob0 + d + 3u] = acc0.w;
  if (tid == 0u) { split_out[ob0 + 256u] = m4.x; split_out[ob0 + 257u] = l4.x; }

  if (gqa_ratio > 1u) {
    let ob1 = ((qh0 + 1u) * num_splits + split_id) * 260u;
    split_out[ob1 + d] = acc1.x; split_out[ob1 + d + 1u] = acc1.y; split_out[ob1 + d + 2u] = acc1.z; split_out[ob1 + d + 3u] = acc1.w;
    if (tid == 0u) { split_out[ob1 + 256u] = m4.y; split_out[ob1 + 257u] = l4.y; }
  }
  if (gqa_ratio > 2u) {
    let ob2 = ((qh0 + 2u) * num_splits + split_id) * 260u;
    split_out[ob2 + d] = acc2.x; split_out[ob2 + d + 1u] = acc2.y; split_out[ob2 + d + 2u] = acc2.z; split_out[ob2 + d + 3u] = acc2.w;
    if (tid == 0u) { split_out[ob2 + 256u] = m4.z; split_out[ob2 + 257u] = l4.z; }
  }
  if (gqa_ratio > 3u) {
    let ob3 = ((qh0 + 3u) * num_splits + split_id) * 260u;
    split_out[ob3 + d] = acc3.x; split_out[ob3 + d + 1u] = acc3.y; split_out[ob3 + d + 2u] = acc3.z; split_out[ob3 + d + 3u] = acc3.w;
    if (tid == 0u) { split_out[ob3 + 256u] = m4.w; split_out[ob3 + 257u] = l4.w; }
  }
}
"#;

// Shader 24: Batched Q4 GEMV for Speculative Verification (M <= 8 tokens, 16 rows/WG, coalesced x_sh)
// 16 rows/WG (8 x 16-lane groups, 2 rows/group), outputs [M, out_stride] at out_offset + t * out_stride + row.
const WGSL_Q4_GEMV_BATCH: &str = r#"
enable f16;
enable subgroups;

struct GemvBatchParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  num_tokens: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvBatchParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 1024>;
var<workgroup> red_sh_a: array<vec4<f32>, 128>;
var<workgroup> red_sh_b: array<vec4<f32>, 128>;

fn unpack_lo(p: u32) -> vec4<f16> {
  return vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
}

fn unpack_hi(p: u32) -> vec4<f16> {
  return vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
}

fn dot_tok_sh(
  sh_base: u32,
  scale: f32,
  zero: f32,
  q0a: vec4<f16>, q0b: vec4<f16>,
  q1a: vec4<f16>, q1b: vec4<f16>,
  q2a: vec4<f16>, q2b: vec4<f16>,
  q3a: vec4<f16>, q3b: vec4<f16>,
) -> f32 {
  let x0a = x_sh[sh_base + 0u]; let x0b = x_sh[sh_base + 1u];
  let x1a = x_sh[sh_base + 2u]; let x1b = x_sh[sh_base + 3u];
  let x2a = x_sh[sh_base + 4u]; let x2b = x_sh[sh_base + 5u];
  let x3a = x_sh[sh_base + 6u]; let x3b = x_sh[sh_base + 7u];

  let ones = vec4<f16>(1.0h);
  let sum_qx = f32(dot(q0a, x0a)) + f32(dot(q0b, x0b))
    + f32(dot(q1a, x1a)) + f32(dot(q1b, x1b))
    + f32(dot(q2a, x2a)) + f32(dot(q2b, x2b))
    + f32(dot(q3a, x3a)) + f32(dot(q3b, x3b));
  let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
    + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
    + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
    + f32(dot(ones, x3a)) + f32(dot(ones, x3b));
  return scale * sum_qx + zero * sum_x;
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
  let grp = tid >> 4u;  // 0..7
  let lane = tid & 15u; // 0..15
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, params.rows - 1u);
  let r1 = min(rB, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let base0 = params.w_u32_offset + r0 * row_stride;
  let base1 = params.w_u32_offset + r1 * row_stride;

  var accA0 = vec4<f32>(0.0);
  var accB0 = vec4<f32>(0.0);
  var accA1 = vec4<f32>(0.0);
  var accB1 = vec4<f32>(0.0);

  for (var b_tile: u32 = 0u; b_tile < blocks; b_tile = b_tile + 16u) {
    let tile_blocks = min(16u, blocks - b_tile);
    let load_vec4s = tile_blocks << 3u;
    let tile_k_off = params.in_offset + (b_tile << 5u);
    if (tid < load_vec4s) {
      for (var t: u32 = 0u; t < M; t = t + 1u) {
        let kb = tile_k_off + t * params.cols + (tid << 2u);
        x_sh[t * 128u + tid] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      }
    }
    workgroupBarrier();

    if (lane < tile_blocks) {
      let b5 = (b_tile + lane) * 5u;
      let xb = lane << 3u;

      let i0 = base0 + b5;
      let w00 = weights[i0 + 0u]; let w01 = weights[i0 + 1u];
      let w02 = weights[i0 + 2u]; let w03 = weights[i0 + 3u];
      let m0 = unpack2x16float(weights[i0 + 4u]);
      let q00a = unpack_lo(w00); let q00b = unpack_hi(w00);
      let q01a = unpack_lo(w01); let q01b = unpack_hi(w01);
      let q02a = unpack_lo(w02); let q02b = unpack_hi(w02);
      let q03a = unpack_lo(w03); let q03b = unpack_hi(w03);

      accA0.x += dot_tok_sh(0u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b);
      if (M > 1u) { accA0.y += dot_tok_sh(1u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 2u) { accA0.z += dot_tok_sh(2u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 3u) { accA0.w += dot_tok_sh(3u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 4u) { accB0.x += dot_tok_sh(4u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 5u) { accB0.y += dot_tok_sh(5u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 6u) { accB0.z += dot_tok_sh(6u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 7u) { accB0.w += dot_tok_sh(7u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }

      let i1 = base1 + b5;
      let w10 = weights[i1 + 0u]; let w11 = weights[i1 + 1u];
      let w12 = weights[i1 + 2u]; let w13 = weights[i1 + 3u];
      let m1 = unpack2x16float(weights[i1 + 4u]);
      let q10a = unpack_lo(w10); let q10b = unpack_hi(w10);
      let q11a = unpack_lo(w11); let q11b = unpack_hi(w11);
      let q12a = unpack_lo(w12); let q12b = unpack_hi(w12);
      let q13a = unpack_lo(w13); let q13b = unpack_hi(w13);

      accA1.x += dot_tok_sh(0u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b);
      if (M > 1u) { accA1.y += dot_tok_sh(1u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 2u) { accA1.z += dot_tok_sh(2u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 3u) { accA1.w += dot_tok_sh(3u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 4u) { accB1.x += dot_tok_sh(4u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 5u) { accB1.y += dot_tok_sh(5u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 6u) { accB1.z += dot_tok_sh(6u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 7u) { accB1.w += dot_tok_sh(7u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
    }

    if (b_tile + 16u < blocks) {
      workgroupBarrier();
    }
  }

  if (sg_size >= 16u) {
    accA0 += subgroupShuffleXor(accA0, 8u); accA0 += subgroupShuffleXor(accA0, 4u);
    accA0 += subgroupShuffleXor(accA0, 2u); accA0 += subgroupShuffleXor(accA0, 1u);
    accA1 += subgroupShuffleXor(accA1, 8u); accA1 += subgroupShuffleXor(accA1, 4u);
    accA1 += subgroupShuffleXor(accA1, 2u); accA1 += subgroupShuffleXor(accA1, 1u);
    if (M > 4u) {
      accB0 += subgroupShuffleXor(accB0, 8u); accB0 += subgroupShuffleXor(accB0, 4u);
      accB0 += subgroupShuffleXor(accB0, 2u); accB0 += subgroupShuffleXor(accB0, 1u);
      accB1 += subgroupShuffleXor(accB1, 8u); accB1 += subgroupShuffleXor(accB1, 4u);
      accB1 += subgroupShuffleXor(accB1, 2u); accB1 += subgroupShuffleXor(accB1, 1u);
    }
  } else {
    red_sh_a[tid] = accA0;
    red_sh_b[tid] = accA1;
    workgroupBarrier();
    if (lane == 0u) {
      var sa0 = vec4<f32>(0.0);
      var sa1 = vec4<f32>(0.0);
      let g_base = grp * 16u;
      for (var l: u32 = 0u; l < 16u; l = l + 1u) {
        sa0 += red_sh_a[g_base + l];
        sa1 += red_sh_b[g_base + l];
      }
      accA0 = sa0;
      accA1 = sa1;
    }
    if (M > 4u) {
      workgroupBarrier();
      red_sh_a[tid] = accB0;
      red_sh_b[tid] = accB1;
      workgroupBarrier();
      if (lane == 0u) {
        var sb0 = vec4<f32>(0.0);
        var sb1 = vec4<f32>(0.0);
        let g_base = grp * 16u;
        for (var l: u32 = 0u; l < 16u; l = l + 1u) {
          sb0 += red_sh_a[g_base + l];
          sb1 += red_sh_b[g_base + l];
        }
        accB0 = sb0;
        accB1 = sb1;
      }
    }
  }

  if (lane == 0u) {
    let add_r = params.add_residual != 0u;
    let out_stride = select(params._pad1, params.rows, params._pad1 == 0u);
    if (rA < params.rows) {
      let i0 = params.out_offset + 0u * out_stride + rA; y[i0] = select(accA0.x, y[i0] + accA0.x, add_r);
      if (M > 1u) { let i1 = params.out_offset + 1u * out_stride + rA; y[i1] = select(accA0.y, y[i1] + accA0.y, add_r); }
      if (M > 2u) { let i2 = params.out_offset + 2u * out_stride + rA; y[i2] = select(accA0.z, y[i2] + accA0.z, add_r); }
      if (M > 3u) { let i3 = params.out_offset + 3u * out_stride + rA; y[i3] = select(accA0.w, y[i3] + accA0.w, add_r); }
      if (M > 4u) { let i4 = params.out_offset + 4u * out_stride + rA; y[i4] = select(accB0.x, y[i4] + accB0.x, add_r); }
      if (M > 5u) { let i5 = params.out_offset + 5u * out_stride + rA; y[i5] = select(accB0.y, y[i5] + accB0.y, add_r); }
      if (M > 6u) { let i6 = params.out_offset + 6u * out_stride + rA; y[i6] = select(accB0.z, y[i6] + accB0.z, add_r); }
      if (M > 7u) { let i7 = params.out_offset + 7u * out_stride + rA; y[i7] = select(accB0.w, y[i7] + accB0.w, add_r); }
    }
    if (rB < params.rows) {
      let i0 = params.out_offset + 0u * out_stride + rB; y[i0] = select(accA1.x, y[i0] + accA1.x, add_r);
      if (M > 1u) { let i1 = params.out_offset + 1u * out_stride + rB; y[i1] = select(accA1.y, y[i1] + accA1.y, add_r); }
      if (M > 2u) { let i2 = params.out_offset + 2u * out_stride + rB; y[i2] = select(accA1.z, y[i2] + accA1.z, add_r); }
      if (M > 3u) { let i3 = params.out_offset + 3u * out_stride + rB; y[i3] = select(accA1.w, y[i3] + accA1.w, add_r); }
      if (M > 4u) { let i4 = params.out_offset + 4u * out_stride + rB; y[i4] = select(accB1.x, y[i4] + accB1.x, add_r); }
      if (M > 5u) { let i5 = params.out_offset + 5u * out_stride + rB; y[i5] = select(accB1.y, y[i5] + accB1.y, add_r); }
      if (M > 6u) { let i6 = params.out_offset + 6u * out_stride + rB; y[i6] = select(accB1.z, y[i6] + accB1.z, add_r); }
      if (M > 7u) { let i7 = params.out_offset + 7u * out_stride + rB; y[i7] = select(accB1.w, y[i7] + accB1.w, add_r); }
    }
  }
}
"#;

// ============================================================================
// Tier 2 (f16 + workgroup shared reduction, NO subgroups) & Tier 3 (pure f32)
// ============================================================================

// Shader 25: Tier 2 Q4 GEMV (f16, no subgroups, tree reduction in red_sh)
const WGSL_Q4_GEMV_T2_F16: &str = r#"
enable f16;

struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn factored_q4_word(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn factored_q4_block(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f16>, x0b: vec4<f16>,
  x1a: vec4<f16>, x1b: vec4<f16>,
  x2a: vec4<f16>, x2b: vec4<f16>,
  x3a: vec4<f16>, x3b: vec4<f16>,
  sum_x: f32,
) -> f32 {
  let blk_meta = unpack2x16float(meta_word);
  let sum_qx = factored_q4_word(w0, x0a, x0b)
    + factored_q4_word(w1, x1a, x1b)
    + factored_q4_word(w2, x2a, x2b)
    + factored_q4_word(w3, x3a, x3b);
  return blk_meta.x * sum_qx + blk_meta.y * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, params.rows - 1u);
  let r1 = min(rB, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let o0 = params.w_u32_offset + r0 * row_stride;
  let o1 = params.w_u32_offset + r1 * row_stride;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  let ones = vec4<f16>(1.0h);

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 80u) {
    let tile_blocks = min(80u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[i] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    }
    workgroupBarrier();

    for (var b_rel: u32 = lane; b_rel < tile_blocks; b_rel = b_rel + 16u) {
      let xb = b_rel << 3u;
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
        + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
        + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
        + f32(dot(ones, x3a)) + f32(dot(ones, x3b));

      let b5 = (k_tile + b_rel) * 5u;
      let i0 = o0 + b5;
      let i1 = o1 + b5;
      acc0 += factored_q4_block(weights[i0], weights[i0 + 1u], weights[i0 + 2u], weights[i0 + 3u], weights[i0 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
      acc1 += factored_q4_block(weights[i1], weights[i1 + 1u], weights[i1 + 2u], weights[i1 + 3u], weights[i1 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
    }

    if (k_tile + 80u < blocks) {
      workgroupBarrier();
    }
  }

  red_sh[tid] = vec2<f32>(acc0, acc1);
  workgroupBarrier();
  if (lane < 8u) { red_sh[tid] += red_sh[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh[tid] += red_sh[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh[tid] += red_sh[tid + 2u]; }
  workgroupBarrier();

  if (lane == 0u) {
    let sum2 = red_sh[tid] + red_sh[tid + 1u];
    if (rA < params.rows) {
      let out0 = params.out_offset + rA;
      y[out0] = select(sum2.x, y[out0] + sum2.x, params.add_residual != 0u);
    }
    if (rB < params.rows) {
      let out1 = params.out_offset + rB;
      y[out1] = select(sum2.y, y[out1] + sum2.y, params.add_residual != 0u);
    }
  }
}
"#;

// Shader 26: Tier 3 Q4 GEMV (pure f32, no f16, no subgroups, 40-block x_sh tile = 5120B + 1024B)
const WGSL_Q4_GEMV_T3_F32: &str = r#"
struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f32>, 320>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn factored_q4_word_f32(p: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let q0 = vec4<f32>(f32(p & 0xFu), f32((p >> 4u) & 0xFu), f32((p >> 8u) & 0xFu), f32((p >> 12u) & 0xFu));
  let q1 = vec4<f32>(f32((p >> 16u) & 0xFu), f32((p >> 20u) & 0xFu), f32((p >> 24u) & 0xFu), f32((p >> 28u) & 0xFu));
  return dot(q0, xa) + dot(q1, xb);
}

fn factored_q4_block_f32(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f32>, x0b: vec4<f32>,
  x1a: vec4<f32>, x1b: vec4<f32>,
  x2a: vec4<f32>, x2b: vec4<f32>,
  x3a: vec4<f32>, x3b: vec4<f32>,
  sum_x: f32,
) -> f32 {
  let blk_meta = unpack2x16float(meta_word);
  let sum_qx = factored_q4_word_f32(w0, x0a, x0b)
    + factored_q4_word_f32(w1, x1a, x1b)
    + factored_q4_word_f32(w2, x2a, x2b)
    + factored_q4_word_f32(w3, x3a, x3b);
  return blk_meta.x * sum_qx + blk_meta.y * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, params.rows - 1u);
  let r1 = min(rB, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let o0 = params.w_u32_offset + r0 * row_stride;
  let o1 = params.w_u32_offset + r1 * row_stride;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  let ones = vec4<f32>(1.0);

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 40u) {
    let tile_blocks = min(40u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
    }
    workgroupBarrier();

    for (var b_rel: u32 = lane; b_rel < tile_blocks; b_rel = b_rel + 16u) {
      let xb = b_rel << 3u;
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = dot(ones, x0a) + dot(ones, x0b)
        + dot(ones, x1a) + dot(ones, x1b)
        + dot(ones, x2a) + dot(ones, x2b)
        + dot(ones, x3a) + dot(ones, x3b);

      let b5 = (k_tile + b_rel) * 5u;
      let i0 = o0 + b5;
      let i1 = o1 + b5;
      acc0 += factored_q4_block_f32(weights[i0], weights[i0 + 1u], weights[i0 + 2u], weights[i0 + 3u], weights[i0 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
      acc1 += factored_q4_block_f32(weights[i1], weights[i1 + 1u], weights[i1 + 2u], weights[i1 + 3u], weights[i1 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
    }

    if (k_tile + 40u < blocks) {
      workgroupBarrier();
    }
  }

  red_sh[tid] = vec2<f32>(acc0, acc1);
  workgroupBarrier();
  if (lane < 8u) { red_sh[tid] += red_sh[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh[tid] += red_sh[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh[tid] += red_sh[tid + 2u]; }
  workgroupBarrier();

  if (lane == 0u) {
    let sum2 = red_sh[tid] + red_sh[tid + 1u];
    if (rA < params.rows) {
      let out0 = params.out_offset + rA;
      y[out0] = select(sum2.x, y[out0] + sum2.x, params.add_residual != 0u);
    }
    if (rB < params.rows) {
      let out1 = params.out_offset + rB;
      y[out1] = select(sum2.y, y[out1] + sum2.y, params.add_residual != 0u);
    }
  }
}
"#;

// Shader 27: Tier 2 Fused Q4 Gate + Up + SwiGLU (f16, no subgroups)
const WGSL_Q4_GATE_UP_SWIGLU_T2_F16: &str = r#"
enable f16;

struct GateUpParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GateUpParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 640>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn factored_q4_word(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

fn factored_q4_block(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f16>, x0b: vec4<f16>,
  x1a: vec4<f16>, x1b: vec4<f16>,
  x2a: vec4<f16>, x2b: vec4<f16>,
  x3a: vec4<f16>, x3b: vec4<f16>,
  sum_x: f32,
) -> f32 {
  let blk_meta = unpack2x16float(meta_word);
  let sum_qx = factored_q4_word(w0, x0a, x0b)
    + factored_q4_word(w1, x1a, x1b)
    + factored_q4_word(w2, x2a, x2b)
    + factored_q4_word(w3, x3a, x3b);
  return blk_meta.x * sum_qx + blk_meta.y * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let r_raw = wid.x * 8u + grp;
  let r = min(r_raw, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let og = params.w_u32_offset + r * row_stride;
  let ou = params._pad0 + r * row_stride;

  var acc_g: f32 = 0.0;
  var acc_u: f32 = 0.0;
  let ones = vec4<f16>(1.0h);

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 80u) {
    let tile_blocks = min(80u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[i] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    }
    workgroupBarrier();

    for (var b_rel: u32 = lane; b_rel < tile_blocks; b_rel = b_rel + 16u) {
      let xb = b_rel << 3u;
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
        + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
        + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
        + f32(dot(ones, x3a)) + f32(dot(ones, x3b));

      let b5 = (k_tile + b_rel) * 5u;
      let ig = og + b5;
      let iu = ou + b5;
      acc_g += factored_q4_block(weights[ig], weights[ig + 1u], weights[ig + 2u], weights[ig + 3u], weights[ig + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
      acc_u += factored_q4_block(weights[iu], weights[iu + 1u], weights[iu + 2u], weights[iu + 3u], weights[iu + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
    }

    if (k_tile + 80u < blocks) {
      workgroupBarrier();
    }
  }

  red_sh[tid] = vec2<f32>(acc_g, acc_u);
  workgroupBarrier();
  if (lane < 8u) { red_sh[tid] += red_sh[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh[tid] += red_sh[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh[tid] += red_sh[tid + 2u]; }
  workgroupBarrier();

  if (lane == 0u && r_raw < params.rows) {
    let sum2 = red_sh[tid] + red_sh[tid + 1u];
    let g = sum2.x;
    let u = sum2.y;
    let out_val = (g / (1.0 + exp(-g))) * u;
    let out_i = params.out_offset + r_raw;
    y[out_i] = select(out_val, y[out_i] + out_val, params.add_residual != 0u);
  }
}
"#;

// Shader 28: Tier 3 Fused Q4 Gate + Up + SwiGLU (pure f32, no f16, no subgroups)
const WGSL_Q4_GATE_UP_SWIGLU_T3_F32: &str = r#"
struct GateUpParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GateUpParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f32>, 320>;
var<workgroup> red_sh: array<vec2<f32>, 128>;

fn factored_q4_word_f32(p: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let q0 = vec4<f32>(f32(p & 0xFu), f32((p >> 4u) & 0xFu), f32((p >> 8u) & 0xFu), f32((p >> 12u) & 0xFu));
  let q1 = vec4<f32>(f32((p >> 16u) & 0xFu), f32((p >> 20u) & 0xFu), f32((p >> 24u) & 0xFu), f32((p >> 28u) & 0xFu));
  return dot(q0, xa) + dot(q1, xb);
}

fn factored_q4_block_f32(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f32>, x0b: vec4<f32>,
  x1a: vec4<f32>, x1b: vec4<f32>,
  x2a: vec4<f32>, x2b: vec4<f32>,
  x3a: vec4<f32>, x3b: vec4<f32>,
  sum_x: f32,
) -> f32 {
  let blk_meta = unpack2x16float(meta_word);
  let sum_qx = factored_q4_word_f32(w0, x0a, x0b)
    + factored_q4_word_f32(w1, x1a, x1b)
    + factored_q4_word_f32(w2, x2a, x2b)
    + factored_q4_word_f32(w3, x3a, x3b);
  return blk_meta.x * sum_qx + blk_meta.y * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let r_raw = wid.x * 8u + grp;
  let r = min(r_raw, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let og = params.w_u32_offset + r * row_stride;
  let ou = params._pad0 + r * row_stride;

  var acc_g: f32 = 0.0;
  var acc_u: f32 = 0.0;
  let ones = vec4<f32>(1.0);

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 40u) {
    let tile_blocks = min(40u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
    }
    workgroupBarrier();

    for (var b_rel: u32 = lane; b_rel < tile_blocks; b_rel = b_rel + 16u) {
      let xb = b_rel << 3u;
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = dot(ones, x0a) + dot(ones, x0b)
        + dot(ones, x1a) + dot(ones, x1b)
        + dot(ones, x2a) + dot(ones, x2b)
        + dot(ones, x3a) + dot(ones, x3b);

      let b5 = (k_tile + b_rel) * 5u;
      let ig = og + b5;
      let iu = ou + b5;
      acc_g += factored_q4_block_f32(weights[ig], weights[ig + 1u], weights[ig + 2u], weights[ig + 3u], weights[ig + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
      acc_u += factored_q4_block_f32(weights[iu], weights[iu + 1u], weights[iu + 2u], weights[iu + 3u], weights[iu + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, sum_x);
    }

    if (k_tile + 40u < blocks) {
      workgroupBarrier();
    }
  }

  red_sh[tid] = vec2<f32>(acc_g, acc_u);
  workgroupBarrier();
  if (lane < 8u) { red_sh[tid] += red_sh[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh[tid] += red_sh[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh[tid] += red_sh[tid + 2u]; }
  workgroupBarrier();

  if (lane == 0u && r_raw < params.rows) {
    let sum2 = red_sh[tid] + red_sh[tid + 1u];
    let g = sum2.x;
    let u = sum2.y;
    let out_val = (g / (1.0 + exp(-g))) * u;
    let out_i = params.out_offset + r_raw;
    y[out_i] = select(out_val, y[out_i] + out_val, params.add_residual != 0u);
  }
}
"#;

// Shader 29: Tier 2 Wide-K 32-Lane Factored Q4 GEMV (f16, no subgroups)
const WGSL_Q4_GEMV_WIDE_K_T2_F16: &str = r#"
enable f16;

struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 512>;
var<workgroup> red_sh: array<f32, 128>;

fn factored_q4_word(p: u32, xa: vec4<f16>, xb: vec4<f16>) -> f32 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return f32(dot(q0, xa)) + f32(dot(q1, xb));
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let row_in_wg = tid >> 5u;
  let lane = tid & 31u;
  let r_raw = wid.x * 4u + row_in_wg;
  let r_safe = min(r_raw, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_base = params.w_u32_offset + r_safe * blocks * 5u;

  var acc: f32 = 0.0;
  let ones = vec4<f16>(1.0h);
  var buf_idx: u32 = 0u;

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 32u) {
    let tile_blocks = min(32u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let base_sh = buf_idx * 256u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[base_sh + i] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    }
    workgroupBarrier();

    if (lane < tile_blocks) {
      let xb = base_sh + (lane << 3u);
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
        + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
        + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
        + f32(dot(ones, x3a)) + f32(dot(ones, x3b));

      let bi = row_base + (k_tile + lane) * 5u;
      let blk_meta = unpack2x16float(weights[bi + 4u]);
      let sum_qx = factored_q4_word(weights[bi], x0a, x0b)
        + factored_q4_word(weights[bi + 1u], x1a, x1b)
        + factored_q4_word(weights[bi + 2u], x2a, x2b)
        + factored_q4_word(weights[bi + 3u], x3a, x3b);
      acc += blk_meta.x * sum_qx + blk_meta.y * sum_x;
    }

    buf_idx = buf_idx ^ 1u;
  }

  red_sh[tid] = acc;
  workgroupBarrier();
  if (lane < 16u) { red_sh[tid] += red_sh[tid + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { red_sh[tid] += red_sh[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh[tid] += red_sh[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh[tid] += red_sh[tid + 2u]; }
  workgroupBarrier();

  if (lane == 0u && r_raw < params.rows) {
    let total = red_sh[tid] + red_sh[tid + 1u];
    let out_i = params.out_offset + r_raw;
    y[out_i] = select(total, y[out_i] + total, params.add_residual != 0u);
  }
}
"#;

// Shader 30: Tier 3 Wide-K 32-Lane Factored Q4 GEMV (pure f32, no f16, no subgroups)
const WGSL_Q4_GEMV_WIDE_K_T3_F32: &str = r#"
struct GemvParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f32>, 512>;
var<workgroup> red_sh: array<f32, 128>;

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
  if (params.rows == 0u) {
    return;
  }
  let tid = lid.x;
  let row_in_wg = tid >> 5u;
  let lane = tid & 31u;
  let r_raw = wid.x * 4u + row_in_wg;
  let r_safe = min(r_raw, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_base = params.w_u32_offset + r_safe * blocks * 5u;

  var acc: f32 = 0.0;
  let ones = vec4<f32>(1.0);
  var buf_idx: u32 = 0u;

  for (var k_tile: u32 = 0u; k_tile < blocks; k_tile = k_tile + 32u) {
    let tile_blocks = min(32u, blocks - k_tile);
    let load_vec4s = tile_blocks << 3u;
    let base_sh = buf_idx * 256u;
    let x_tile_base = params.in_offset + (k_tile << 5u);
    for (var i: u32 = tid; i < load_vec4s; i = i + 128u) {
      let kb = x_tile_base + (i << 2u);
      x_sh[base_sh + i] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
    }
    workgroupBarrier();

    if (lane < tile_blocks) {
      let xb = base_sh + (lane << 3u);
      let x0a = x_sh[xb + 0u]; let x0b = x_sh[xb + 1u];
      let x1a = x_sh[xb + 2u]; let x1b = x_sh[xb + 3u];
      let x2a = x_sh[xb + 4u]; let x2b = x_sh[xb + 5u];
      let x3a = x_sh[xb + 6u]; let x3b = x_sh[xb + 7u];

      let sum_x = dot(ones, x0a) + dot(ones, x0b)
        + dot(ones, x1a) + dot(ones, x1b)
        + dot(ones, x2a) + dot(ones, x2b)
        + dot(ones, x3a) + dot(ones, x3b);

      let bi = row_base + (k_tile + lane) * 5u;
      let blk_meta = unpack2x16float(weights[bi + 4u]);
      let sum_qx = factored_q4_word_f32(weights[bi], x0a, x0b)
        + factored_q4_word_f32(weights[bi + 1u], x1a, x1b)
        + factored_q4_word_f32(weights[bi + 2u], x2a, x2b)
        + factored_q4_word_f32(weights[bi + 3u], x3a, x3b);
      acc += blk_meta.x * sum_qx + blk_meta.y * sum_x;
    }

    buf_idx = buf_idx ^ 1u;
  }

  red_sh[tid] = acc;
  workgroupBarrier();
  if (lane < 16u) { red_sh[tid] += red_sh[tid + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { red_sh[tid] += red_sh[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh[tid] += red_sh[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh[tid] += red_sh[tid + 2u]; }
  workgroupBarrier();

  if (lane == 0u && r_raw < params.rows) {
    let total = red_sh[tid] + red_sh[tid + 1u];
    let out_i = params.out_offset + r_raw;
    y[out_i] = select(total, y[out_i] + total, params.add_residual != 0u);
  }
}
"#;

// Shader 31: Tier 2 Batched Q4 GEMV (f16, no subgroups, 12288B shared mem)
const WGSL_Q4_GEMV_BATCH_T2_F16: &str = r#"
enable f16;

struct GemvBatchParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  num_tokens: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvBatchParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f16>, 1024>;
var<workgroup> red_sh_a: array<vec4<f32>, 128>;
var<workgroup> red_sh_b: array<vec4<f32>, 128>;

fn unpack_lo(p: u32) -> vec4<f16> {
  return vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
}

fn unpack_hi(p: u32) -> vec4<f16> {
  return vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
}

fn dot_tok_sh(
  sh_base: u32,
  scale: f32,
  zero: f32,
  q0a: vec4<f16>, q0b: vec4<f16>,
  q1a: vec4<f16>, q1b: vec4<f16>,
  q2a: vec4<f16>, q2b: vec4<f16>,
  q3a: vec4<f16>, q3b: vec4<f16>,
) -> f32 {
  let x0a = x_sh[sh_base + 0u]; let x0b = x_sh[sh_base + 1u];
  let x1a = x_sh[sh_base + 2u]; let x1b = x_sh[sh_base + 3u];
  let x2a = x_sh[sh_base + 4u]; let x2b = x_sh[sh_base + 5u];
  let x3a = x_sh[sh_base + 6u]; let x3b = x_sh[sh_base + 7u];

  let ones = vec4<f16>(1.0h);
  let sum_qx = f32(dot(q0a, x0a)) + f32(dot(q0b, x0b))
    + f32(dot(q1a, x1a)) + f32(dot(q1b, x1b))
    + f32(dot(q2a, x2a)) + f32(dot(q2b, x2b))
    + f32(dot(q3a, x3a)) + f32(dot(q3b, x3b));
  let sum_x = f32(dot(ones, x0a)) + f32(dot(ones, x0b))
    + f32(dot(ones, x1a)) + f32(dot(ones, x1b))
    + f32(dot(ones, x2a)) + f32(dot(ones, x2b))
    + f32(dot(ones, x3a)) + f32(dot(ones, x3b));
  return scale * sum_qx + zero * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = min(params.num_tokens, 8u);
  if (params.rows == 0u || M == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, params.rows - 1u);
  let r1 = min(rB, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let base0 = params.w_u32_offset + r0 * row_stride;
  let base1 = params.w_u32_offset + r1 * row_stride;

  var accA0 = vec4<f32>(0.0);
  var accB0 = vec4<f32>(0.0);
  var accA1 = vec4<f32>(0.0);
  var accB1 = vec4<f32>(0.0);

  for (var b_tile: u32 = 0u; b_tile < blocks; b_tile = b_tile + 16u) {
    let tile_blocks = min(16u, blocks - b_tile);
    let load_vec4s = tile_blocks << 3u;
    let tile_k_off = params.in_offset + (b_tile << 5u);
    if (tid < load_vec4s) {
      for (var t: u32 = 0u; t < M; t = t + 1u) {
        let kb = tile_k_off + t * params.cols + (tid << 2u);
        x_sh[t * 128u + tid] = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
      }
    }
    workgroupBarrier();

    if (lane < tile_blocks) {
      let b5 = (b_tile + lane) * 5u;
      let xb = lane << 3u;

      let i0 = base0 + b5;
      let w00 = weights[i0 + 0u]; let w01 = weights[i0 + 1u];
      let w02 = weights[i0 + 2u]; let w03 = weights[i0 + 3u];
      let m0 = unpack2x16float(weights[i0 + 4u]);
      let q00a = unpack_lo(w00); let q00b = unpack_hi(w00);
      let q01a = unpack_lo(w01); let q01b = unpack_hi(w01);
      let q02a = unpack_lo(w02); let q02b = unpack_hi(w02);
      let q03a = unpack_lo(w03); let q03b = unpack_hi(w03);

      accA0.x += dot_tok_sh(0u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b);
      if (M > 1u) { accA0.y += dot_tok_sh(1u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 2u) { accA0.z += dot_tok_sh(2u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 3u) { accA0.w += dot_tok_sh(3u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 4u) { accB0.x += dot_tok_sh(4u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 5u) { accB0.y += dot_tok_sh(5u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 6u) { accB0.z += dot_tok_sh(6u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 7u) { accB0.w += dot_tok_sh(7u * 128u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }

      let i1 = base1 + b5;
      let w10 = weights[i1 + 0u]; let w11 = weights[i1 + 1u];
      let w12 = weights[i1 + 2u]; let w13 = weights[i1 + 3u];
      let m1 = unpack2x16float(weights[i1 + 4u]);
      let q10a = unpack_lo(w10); let q10b = unpack_hi(w10);
      let q11a = unpack_lo(w11); let q11b = unpack_hi(w11);
      let q12a = unpack_lo(w12); let q12b = unpack_hi(w12);
      let q13a = unpack_lo(w13); let q13b = unpack_hi(w13);

      accA1.x += dot_tok_sh(0u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b);
      if (M > 1u) { accA1.y += dot_tok_sh(1u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 2u) { accA1.z += dot_tok_sh(2u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 3u) { accA1.w += dot_tok_sh(3u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 4u) { accB1.x += dot_tok_sh(4u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 5u) { accB1.y += dot_tok_sh(5u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 6u) { accB1.z += dot_tok_sh(6u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 7u) { accB1.w += dot_tok_sh(7u * 128u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
    }

    if (b_tile + 16u < blocks) {
      workgroupBarrier();
    }
  }

  red_sh_a[tid] = accA0;
  red_sh_b[tid] = accA1;
  workgroupBarrier();
  if (lane < 8u) { red_sh_a[tid] += red_sh_a[tid + 8u]; red_sh_b[tid] += red_sh_b[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh_a[tid] += red_sh_a[tid + 4u]; red_sh_b[tid] += red_sh_b[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh_a[tid] += red_sh_a[tid + 2u]; red_sh_b[tid] += red_sh_b[tid + 2u]; }
  workgroupBarrier();
  accA0 = red_sh_a[tid] + red_sh_a[tid + 1u];
  accA1 = red_sh_b[tid] + red_sh_b[tid + 1u];

  if (M > 4u) {
    workgroupBarrier();
    red_sh_a[tid] = accB0;
    red_sh_b[tid] = accB1;
    workgroupBarrier();
    if (lane < 8u) { red_sh_a[tid] += red_sh_a[tid + 8u]; red_sh_b[tid] += red_sh_b[tid + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { red_sh_a[tid] += red_sh_a[tid + 4u]; red_sh_b[tid] += red_sh_b[tid + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { red_sh_a[tid] += red_sh_a[tid + 2u]; red_sh_b[tid] += red_sh_b[tid + 2u]; }
    workgroupBarrier();
    accB0 = red_sh_a[tid] + red_sh_a[tid + 1u];
    accB1 = red_sh_b[tid] + red_sh_b[tid + 1u];
  }

  if (lane == 0u) {
    let add_r = params.add_residual != 0u;
    let out_stride = select(params._pad1, params.rows, params._pad1 == 0u);
    if (rA < params.rows) {
      let i0 = params.out_offset + 0u * out_stride + rA; y[i0] = select(accA0.x, y[i0] + accA0.x, add_r);
      if (M > 1u) { let i1 = params.out_offset + 1u * out_stride + rA; y[i1] = select(accA0.y, y[i1] + accA0.y, add_r); }
      if (M > 2u) { let i2 = params.out_offset + 2u * out_stride + rA; y[i2] = select(accA0.z, y[i2] + accA0.z, add_r); }
      if (M > 3u) { let i3 = params.out_offset + 3u * out_stride + rA; y[i3] = select(accA0.w, y[i3] + accA0.w, add_r); }
      if (M > 4u) { let i4 = params.out_offset + 4u * out_stride + rA; y[i4] = select(accB0.x, y[i4] + accB0.x, add_r); }
      if (M > 5u) { let i5 = params.out_offset + 5u * out_stride + rA; y[i5] = select(accB0.y, y[i5] + accB0.y, add_r); }
      if (M > 6u) { let i6 = params.out_offset + 6u * out_stride + rA; y[i6] = select(accB0.z, y[i6] + accB0.z, add_r); }
      if (M > 7u) { let i7 = params.out_offset + 7u * out_stride + rA; y[i7] = select(accB0.w, y[i7] + accB0.w, add_r); }
    }
    if (rB < params.rows) {
      let i0 = params.out_offset + 0u * out_stride + rB; y[i0] = select(accA1.x, y[i0] + accA1.x, add_r);
      if (M > 1u) { let i1 = params.out_offset + 1u * out_stride + rB; y[i1] = select(accA1.y, y[i1] + accA1.y, add_r); }
      if (M > 2u) { let i2 = params.out_offset + 2u * out_stride + rB; y[i2] = select(accA1.z, y[i2] + accA1.z, add_r); }
      if (M > 3u) { let i3 = params.out_offset + 3u * out_stride + rB; y[i3] = select(accA1.w, y[i3] + accA1.w, add_r); }
      if (M > 4u) { let i4 = params.out_offset + 4u * out_stride + rB; y[i4] = select(accB1.x, y[i4] + accB1.x, add_r); }
      if (M > 5u) { let i5 = params.out_offset + 5u * out_stride + rB; y[i5] = select(accB1.y, y[i5] + accB1.y, add_r); }
      if (M > 6u) { let i6 = params.out_offset + 6u * out_stride + rB; y[i6] = select(accB1.z, y[i6] + accB1.z, add_r); }
      if (M > 7u) { let i7 = params.out_offset + 7u * out_stride + rB; y[i7] = select(accB1.w, y[i7] + accB1.w, add_r); }
    }
  }
}
"#;

// Shader 32: Tier 3 Batched Q4 GEMV (pure f32, no f16, no subgroups, b_tile+=8 -> x_sh=8192B + red_sh=4096B)
const WGSL_Q4_GEMV_BATCH_T3_F32: &str = r#"
struct GemvBatchParams {
  rows: u32,
  cols: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  num_tokens: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvBatchParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f32>, 512>;
var<workgroup> red_sh_a: array<vec4<f32>, 128>;
var<workgroup> red_sh_b: array<vec4<f32>, 128>;

fn unpack_lo_f32(p: u32) -> vec4<f32> {
  return vec4<f32>(f32(p & 0xFu), f32((p >> 4u) & 0xFu), f32((p >> 8u) & 0xFu), f32((p >> 12u) & 0xFu));
}

fn unpack_hi_f32(p: u32) -> vec4<f32> {
  return vec4<f32>(f32((p >> 16u) & 0xFu), f32((p >> 20u) & 0xFu), f32((p >> 24u) & 0xFu), f32((p >> 28u) & 0xFu));
}

fn dot_tok_sh_f32(
  sh_base: u32,
  scale: f32,
  zero: f32,
  q0a: vec4<f32>, q0b: vec4<f32>,
  q1a: vec4<f32>, q1b: vec4<f32>,
  q2a: vec4<f32>, q2b: vec4<f32>,
  q3a: vec4<f32>, q3b: vec4<f32>,
) -> f32 {
  let x0a = x_sh[sh_base + 0u]; let x0b = x_sh[sh_base + 1u];
  let x1a = x_sh[sh_base + 2u]; let x1b = x_sh[sh_base + 3u];
  let x2a = x_sh[sh_base + 4u]; let x2b = x_sh[sh_base + 5u];
  let x3a = x_sh[sh_base + 6u]; let x3b = x_sh[sh_base + 7u];

  let ones = vec4<f32>(1.0);
  let sum_qx = dot(q0a, x0a) + dot(q0b, x0b)
    + dot(q1a, x1a) + dot(q1b, x1b)
    + dot(q2a, x2a) + dot(q2b, x2b)
    + dot(q3a, x3a) + dot(q3b, x3b);
  let sum_x = dot(ones, x0a) + dot(ones, x0b)
    + dot(ones, x1a) + dot(ones, x1b)
    + dot(ones, x2a) + dot(ones, x2b)
    + dot(ones, x3a) + dot(ones, x3b);
  return scale * sum_qx + zero * sum_x;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = min(params.num_tokens, 8u);
  if (params.rows == 0u || M == 0u) {
    return;
  }
  let tid = lid.x;
  let grp = tid >> 4u;
  let lane = tid & 15u;
  let rA = wid.x * 16u + grp * 2u;
  let rB = rA + 1u;
  let r0 = min(rA, params.rows - 1u);
  let r1 = min(rB, params.rows - 1u);

  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;
  let base0 = params.w_u32_offset + r0 * row_stride;
  let base1 = params.w_u32_offset + r1 * row_stride;

  var accA0 = vec4<f32>(0.0);
  var accB0 = vec4<f32>(0.0);
  var accA1 = vec4<f32>(0.0);
  var accB1 = vec4<f32>(0.0);

  for (var b_tile: u32 = 0u; b_tile < blocks; b_tile = b_tile + 8u) {
    let tile_blocks = min(8u, blocks - b_tile);
    let load_vec4s = tile_blocks << 3u;
    let tile_k_off = params.in_offset + (b_tile << 5u);
    if (tid < load_vec4s) {
      for (var t: u32 = 0u; t < M; t = t + 1u) {
        let kb = tile_k_off + t * params.cols + (tid << 2u);
        x_sh[t * 64u + tid] = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
      }
    }
    workgroupBarrier();

    if (lane < tile_blocks) {
      let b5 = (b_tile + lane) * 5u;
      let xb = lane << 3u;

      let i0 = base0 + b5;
      let w00 = weights[i0 + 0u]; let w01 = weights[i0 + 1u];
      let w02 = weights[i0 + 2u]; let w03 = weights[i0 + 3u];
      let m0 = unpack2x16float(weights[i0 + 4u]);
      let q00a = unpack_lo_f32(w00); let q00b = unpack_hi_f32(w00);
      let q01a = unpack_lo_f32(w01); let q01b = unpack_hi_f32(w01);
      let q02a = unpack_lo_f32(w02); let q02b = unpack_hi_f32(w02);
      let q03a = unpack_lo_f32(w03); let q03b = unpack_hi_f32(w03);

      accA0.x += dot_tok_sh_f32(0u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b);
      if (M > 1u) { accA0.y += dot_tok_sh_f32(1u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 2u) { accA0.z += dot_tok_sh_f32(2u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 3u) { accA0.w += dot_tok_sh_f32(3u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 4u) { accB0.x += dot_tok_sh_f32(4u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 5u) { accB0.y += dot_tok_sh_f32(5u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 6u) { accB0.z += dot_tok_sh_f32(6u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }
      if (M > 7u) { accB0.w += dot_tok_sh_f32(7u * 64u + xb, m0.x, m0.y, q00a, q00b, q01a, q01b, q02a, q02b, q03a, q03b); }

      let i1 = base1 + b5;
      let w10 = weights[i1 + 0u]; let w11 = weights[i1 + 1u];
      let w12 = weights[i1 + 2u]; let w13 = weights[i1 + 3u];
      let m1 = unpack2x16float(weights[i1 + 4u]);
      let q10a = unpack_lo_f32(w10); let q10b = unpack_hi_f32(w10);
      let q11a = unpack_lo_f32(w11); let q11b = unpack_hi_f32(w11);
      let q12a = unpack_lo_f32(w12); let q12b = unpack_hi_f32(w12);
      let q13a = unpack_lo_f32(w13); let q13b = unpack_hi_f32(w13);

      accA1.x += dot_tok_sh_f32(0u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b);
      if (M > 1u) { accA1.y += dot_tok_sh_f32(1u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 2u) { accA1.z += dot_tok_sh_f32(2u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 3u) { accA1.w += dot_tok_sh_f32(3u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 4u) { accB1.x += dot_tok_sh_f32(4u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 5u) { accB1.y += dot_tok_sh_f32(5u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 6u) { accB1.z += dot_tok_sh_f32(6u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
      if (M > 7u) { accB1.w += dot_tok_sh_f32(7u * 64u + xb, m1.x, m1.y, q10a, q10b, q11a, q11b, q12a, q12b, q13a, q13b); }
    }

    if (b_tile + 8u < blocks) {
      workgroupBarrier();
    }
  }

  red_sh_a[tid] = accA0;
  red_sh_b[tid] = accA1;
  workgroupBarrier();
  if (lane < 8u) { red_sh_a[tid] += red_sh_a[tid + 8u]; red_sh_b[tid] += red_sh_b[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_sh_a[tid] += red_sh_a[tid + 4u]; red_sh_b[tid] += red_sh_b[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_sh_a[tid] += red_sh_a[tid + 2u]; red_sh_b[tid] += red_sh_b[tid + 2u]; }
  workgroupBarrier();
  accA0 = red_sh_a[tid] + red_sh_a[tid + 1u];
  accA1 = red_sh_b[tid] + red_sh_b[tid + 1u];

  if (M > 4u) {
    workgroupBarrier();
    red_sh_a[tid] = accB0;
    red_sh_b[tid] = accB1;
    workgroupBarrier();
    if (lane < 8u) { red_sh_a[tid] += red_sh_a[tid + 8u]; red_sh_b[tid] += red_sh_b[tid + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { red_sh_a[tid] += red_sh_a[tid + 4u]; red_sh_b[tid] += red_sh_b[tid + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { red_sh_a[tid] += red_sh_a[tid + 2u]; red_sh_b[tid] += red_sh_b[tid + 2u]; }
    workgroupBarrier();
    accB0 = red_sh_a[tid] + red_sh_a[tid + 1u];
    accB1 = red_sh_b[tid] + red_sh_b[tid + 1u];
  }

  if (lane == 0u) {
    let add_r = params.add_residual != 0u;
    let out_stride = select(params._pad1, params.rows, params._pad1 == 0u);
    if (rA < params.rows) {
      let i0 = params.out_offset + 0u * out_stride + rA; y[i0] = select(accA0.x, y[i0] + accA0.x, add_r);
      if (M > 1u) { let i1 = params.out_offset + 1u * out_stride + rA; y[i1] = select(accA0.y, y[i1] + accA0.y, add_r); }
      if (M > 2u) { let i2 = params.out_offset + 2u * out_stride + rA; y[i2] = select(accA0.z, y[i2] + accA0.z, add_r); }
      if (M > 3u) { let i3 = params.out_offset + 3u * out_stride + rA; y[i3] = select(accA0.w, y[i3] + accA0.w, add_r); }
      if (M > 4u) { let i4 = params.out_offset + 4u * out_stride + rA; y[i4] = select(accB0.x, y[i4] + accB0.x, add_r); }
      if (M > 5u) { let i5 = params.out_offset + 5u * out_stride + rA; y[i5] = select(accB0.y, y[i5] + accB0.y, add_r); }
      if (M > 6u) { let i6 = params.out_offset + 6u * out_stride + rA; y[i6] = select(accB0.z, y[i6] + accB0.z, add_r); }
      if (M > 7u) { let i7 = params.out_offset + 7u * out_stride + rA; y[i7] = select(accB0.w, y[i7] + accB0.w, add_r); }
    }
    if (rB < params.rows) {
      let i0 = params.out_offset + 0u * out_stride + rB; y[i0] = select(accA1.x, y[i0] + accA1.x, add_r);
      if (M > 1u) { let i1 = params.out_offset + 1u * out_stride + rB; y[i1] = select(accA1.y, y[i1] + accA1.y, add_r); }
      if (M > 2u) { let i2 = params.out_offset + 2u * out_stride + rB; y[i2] = select(accA1.z, y[i2] + accA1.z, add_r); }
      if (M > 3u) { let i3 = params.out_offset + 3u * out_stride + rB; y[i3] = select(accA1.w, y[i3] + accA1.w, add_r); }
      if (M > 4u) { let i4 = params.out_offset + 4u * out_stride + rB; y[i4] = select(accB1.x, y[i4] + accB1.x, add_r); }
      if (M > 5u) { let i5 = params.out_offset + 5u * out_stride + rB; y[i5] = select(accB1.y, y[i5] + accB1.y, add_r); }
      if (M > 6u) { let i6 = params.out_offset + 6u * out_stride + rB; y[i6] = select(accB1.z, y[i6] + accB1.z, add_r); }
      if (M > 7u) { let i7 = params.out_offset + 7u * out_stride + rB; y[i7] = select(accB1.w, y[i7] + accB1.w, add_r); }
    }
  }
}
"#;

// Shader 33: Tier 3 Batched Q4-f32 GEMM [M, K] x [N, K]^T -> [M, N] via 64x64x16 shared-memory tiling (8192B)
const WGSL_Q4_GEMM_T3_F32: &str = r#"
struct GemmParams {
  M: u32,
  N: u32,
  K: u32,
  w_u32_offset: u32,
  in_offset: u32,
  out_offset: u32,
  add_residual: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> x_sh: array<vec4<f32>, 256>;
var<workgroup> w_sh: array<vec4<f32>, 256>;

fn unpack_nibble_vec_f32(word: u32, shift_base: u32, s4: vec4<f32>, z4: vec4<f32>) -> vec4<f32> {
  let q = vec4<f32>(
    f32((word >> shift_base) & 0xFu),
    f32((word >> (shift_base + 4u)) & 0xFu),
    f32((word >> (shift_base + 8u)) & 0xFu),
    f32((word >> (shift_base + 12u)) & 0xFu),
  );
  return fma(q, s4, z4);
}

fn store_cell(tok: u32, row: u32, val: f32) {
  if (tok < params.M && row < params.N) {
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
  let blocks = params.K >> 5u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let loc_idx = tid >> 2u; // 0..63
  let qtr = tid & 3u;      // 0..3 (4 elements per quarter across 16-wide K half-block)

  for (var b: u32 = 0u; b < blocks; b = b + 1u) {
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let gt = tok_tile + loc_idx;
      if (gt < params.M) {
        let xb = params.in_offset + gt * params.K + (b << 5u) + (h << 4u) + (qtr << 2u);
        x_sh[qtr * 64u + loc_idx] = vec4<f32>(x[xb + 0u], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
      } else {
        x_sh[qtr * 64u + loc_idx] = vec4<f32>(0.0);
      }

      let gr = row_tile + loc_idx;
      if (gr < params.N) {
        let bi = params.w_u32_offset + gr * blocks * 5u + b * 5u;
        let blk_meta = unpack2x16float(weights[bi + 4u]);
        let s4 = vec4<f32>(blk_meta.x);
        let z4 = vec4<f32>(blk_meta.y);
        let pw = weights[bi + h * 2u + (qtr >> 1u)];
        w_sh[qtr * 64u + loc_idx] = unpack_nibble_vec_f32(pw, (qtr & 1u) * 16u, s4, z4);
      } else {
        w_sh[qtr * 64u + loc_idx] = vec4<f32>(0.0);
      }
      workgroupBarrier();

      for (var k4: u32 = 0u; k4 < 4u; k4 = k4 + 1u) {
        let kb = k4 * 64u;
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
  }

  let out_t = tok_tile + lx;
  let out_r = row_tile + ly;
  store_cell(out_t + 0u, out_r + 0u, acc0.x);
  store_cell(out_t + 16u, out_r + 0u, acc0.y);
  store_cell(out_t + 32u, out_r + 0u, acc0.z);
  store_cell(out_t + 48u, out_r + 0u, acc0.w);

  store_cell(out_t + 0u, out_r + 16u, acc1.x);
  store_cell(out_t + 16u, out_r + 16u, acc1.y);
  store_cell(out_t + 32u, out_r + 16u, acc1.z);
  store_cell(out_t + 48u, out_r + 16u, acc1.w);

  store_cell(out_t + 0u, out_r + 32u, acc2.x);
  store_cell(out_t + 16u, out_r + 32u, acc2.y);
  store_cell(out_t + 32u, out_r + 32u, acc2.z);
  store_cell(out_t + 48u, out_r + 32u, acc2.w);

  store_cell(out_t + 0u, out_r + 48u, acc3.x);
  store_cell(out_t + 16u, out_r + 48u, acc3.y);
  store_cell(out_t + 32u, out_r + 48u, acc3.z);
  store_cell(out_t + 48u, out_r + 48u, acc3.w);
}
"#;

// Shader 34: Tier 2 & 3 Batched RMSNorm (pure f32, no subgroups)
const WGSL_RMSNORM_BATCH_T2: &str = r#"
struct BatchNormParams {
  num_tokens: u32,
  len: u32,
  gamma_offset: u32,
  in_offset: u32,
  out_offset: u32,
  eps: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: BatchNormParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

var<workgroup> sq_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tok = wid.x;
  let tid = lid.x;
  let base_in = params.in_offset + tok * params.len;
  let base_out = params.out_offset + tok * params.len;

  var sq: f32 = 0.0;
  for (var i: u32 = tid; i < params.len; i = i + 256u) {
    let v = src[base_in + i];
    sq += v * v;
  }
  sq_buf[tid] = sq;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sq_buf[tid] += sq_buf[tid + s];
    }
    workgroupBarrier();
  }
  let inv_rms = inverseSqrt(sq_buf[0] / f32(params.len) + params.eps);
  for (var i: u32 = tid; i < params.len; i = i + 256u) {
    dst[base_out + i] = src[base_in + i] * inv_rms * aux[params.gamma_offset + i];
  }
}
"#;

// Shader 35: Tier 2 FP16 KV Split-K FlashDecoding (f16, no subgroups)
const WGSL_GQA_FP16_SPLIT_T2_F16: &str = r#"
enable f16;

struct GqaSplitParams {
  layer_kv_u32_offset: u32,
  max_seq: u32,
  seq_pos: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  num_splits: u32,
  q_head_stride: u32,
};

@group(0) @binding(0) var<uniform> params: GqaSplitParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;
@group(0) @binding(4) var<storage, read> pos_buf: array<u32>;

var<workgroup> red_buf: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let q_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let kv_head = q_head / gqa_ratio;
  let tid = lid.x;
  let d2 = tid * 2u;
  let d = tid * 4u;
  let seq_pos = select(params.seq_pos, pos_buf[1], params.seq_pos == 0xFFFFFFFFu);

  let head_u32_stride = params.max_seq * 128u;
  let kv_plane_u32s = kv_heads * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let q_base = params.q_offset + q_head * q_stride + d;
  let q_vec = vec4<f32>(act[q_base], act[q_base + 1u], act[q_base + 2u], act[q_base + 3u]);
  let inv_sqrt_d: f32 = 0.0625;

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  var acc_v = vec4<f32>(0.0);

  let chunk = (seq_pos + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, seq_pos);
  let t_end = min(t_start + chunk, seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    red_buf[tid] = dot(q_vec, vec4<f32>(k0.x, k0.y, k1.x, k1.y));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_buf[tid] += red_buf[tid + s]; }
      workgroupBarrier();
    }

    let score = red_buf[0] * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;

    let v_src = v_head_base + t * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    acc_v = acc_v * alpha + vec4<f32>(v0.x, v0.y, v1.x, v1.y) * beta;
    m_max = m_new;
    workgroupBarrier();
  }

  if (split_id == 0u) {
    let k_in = params.k_offset + kv_head * 256u + d;
    let v_in = params.v_offset + kv_head * 256u + d;
    let k0h = vec2<f16>(f16(act[k_in]), f16(act[k_in + 1u]));
    let k1h = vec2<f16>(f16(act[k_in + 2u]), f16(act[k_in + 3u]));
    let v0h = vec2<f16>(f16(act[v_in]), f16(act[v_in + 1u]));
    let v1h = vec2<f16>(f16(act[v_in + 2u]), f16(act[v_in + 3u]));

    if ((q_head % gqa_ratio) == 0u) {
      let k_dst = k_head_base + seq_pos * 128u + d2;
      let v_dst = v_head_base + seq_pos * 128u + d2;
      kv_cache[k_dst] = bitcast<u32>(k0h);
      kv_cache[k_dst + 1u] = bitcast<u32>(k1h);
      kv_cache[v_dst] = bitcast<u32>(v0h);
      kv_cache[v_dst + 1u] = bitcast<u32>(v1h);
    }

    red_buf[tid] = dot(q_vec, vec4<f32>(f32(k0h.x), f32(k0h.y), f32(k1h.x), f32(k1h.y)));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_buf[tid] += red_buf[tid + s]; }
      workgroupBarrier();
    }

    let score = red_buf[0] * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;
    acc_v = acc_v * alpha + vec4<f32>(f32(v0h.x), f32(v0h.y), f32(v1h.x), f32(v1h.y)) * beta;
    m_max = m_new;
  }

  let out_base = (q_head * num_splits + split_id) * 260u;
  split_out[out_base + d] = acc_v.x;
  split_out[out_base + d + 1u] = acc_v.y;
  split_out[out_base + d + 2u] = acc_v.z;
  split_out[out_base + d + 3u] = acc_v.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_max;
    split_out[out_base + 257u] = l_sum;
  }
}
"#;

// Shader 36: Tier 3 FP16 KV Split-K FlashDecoding (pure f32, pack2x16float/unpack2x16float, no f16, no subgroups)
const WGSL_GQA_FP16_SPLIT_T3_F32: &str = r#"
struct GqaSplitParams {
  layer_kv_u32_offset: u32,
  max_seq: u32,
  seq_pos: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  num_splits: u32,
  q_head_stride: u32,
};

@group(0) @binding(0) var<uniform> params: GqaSplitParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;
@group(0) @binding(4) var<storage, read> pos_buf: array<u32>;

var<workgroup> red_buf: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let q_head = wid.x;
  let split_id = wid.y;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let kv_head = q_head / gqa_ratio;
  let tid = lid.x;
  let d2 = tid * 2u;
  let d = tid * 4u;
  let seq_pos = select(params.seq_pos, pos_buf[1], params.seq_pos == 0xFFFFFFFFu);

  let head_u32_stride = params.max_seq * 128u;
  let kv_plane_u32s = kv_heads * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let q_base = params.q_offset + q_head * q_stride + d;
  let q_vec = vec4<f32>(act[q_base], act[q_base + 1u], act[q_base + 2u], act[q_base + 3u]);
  let inv_sqrt_d: f32 = 0.0625;

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  var acc_v = vec4<f32>(0.0);

  let chunk = (seq_pos + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, seq_pos);
  let t_end = min(t_start + chunk, seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    red_buf[tid] = dot(q_vec, vec4<f32>(k0.x, k0.y, k1.x, k1.y));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_buf[tid] += red_buf[tid + s]; }
      workgroupBarrier();
    }

    let score = red_buf[0] * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;

    let v_src = v_head_base + t * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    acc_v = acc_v * alpha + vec4<f32>(v0.x, v0.y, v1.x, v1.y) * beta;
    m_max = m_new;
    workgroupBarrier();
  }

  if (split_id == 0u) {
    let k_in = params.k_offset + kv_head * 256u + d;
    let v_in = params.v_offset + kv_head * 256u + d;
    let k0p = pack2x16float(vec2<f32>(act[k_in], act[k_in + 1u]));
    let k1p = pack2x16float(vec2<f32>(act[k_in + 2u], act[k_in + 3u]));
    let v0p = pack2x16float(vec2<f32>(act[v_in], act[v_in + 1u]));
    let v1p = pack2x16float(vec2<f32>(act[v_in + 2u], act[v_in + 3u]));

    if ((q_head % gqa_ratio) == 0u) {
      let k_dst = k_head_base + seq_pos * 128u + d2;
      let v_dst = v_head_base + seq_pos * 128u + d2;
      kv_cache[k_dst] = k0p;
      kv_cache[k_dst + 1u] = k1p;
      kv_cache[v_dst] = v0p;
      kv_cache[v_dst + 1u] = v1p;
    }

    let k0u = unpack2x16float(k0p);
    let k1u = unpack2x16float(k1p);
    let v0u = unpack2x16float(v0p);
    let v1u = unpack2x16float(v1p);

    red_buf[tid] = dot(q_vec, vec4<f32>(k0u.x, k0u.y, k1u.x, k1u.y));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_buf[tid] += red_buf[tid + s]; }
      workgroupBarrier();
    }

    let score = red_buf[0] * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;
    acc_v = acc_v * alpha + vec4<f32>(v0u.x, v0u.y, v1u.x, v1u.y) * beta;
    m_max = m_new;
  }

  let out_base = (q_head * num_splits + split_id) * 260u;
  split_out[out_base + d] = acc_v.x;
  split_out[out_base + d + 1u] = acc_v.y;
  split_out[out_base + d + 2u] = acc_v.z;
  split_out[out_base + d + 3u] = acc_v.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_max;
    split_out[out_base + 257u] = l_sum;
  }
}
"#;

// Shader 37: Tier 2 & 3 INT8 KV Split-K FlashDecoding (pure f32, no subgroups)
const WGSL_GQA_INT8_SPLIT_T2: &str = r#"
struct GqaSplitParams {
  layer_kv_u32_offset: u32,
  max_seq: u32,
  seq_pos: u32,
  q_offset: u32,
  k_offset: u32,
  v_offset: u32,
  num_splits: u32,
  q_head_stride: u32,
};

@group(0) @binding(0) var<uniform> params: GqaSplitParams;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;
@group(0) @binding(4) var<storage, read> pos_buf: array<u32>;

var<workgroup> red_buf: array<vec4<f32>, 64>;
var<workgroup> cur_kv_max: array<vec2<f32>, 64>;

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

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let kv_head = wid.x;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let qh0 = kv_head * gqa_ratio;
  let split_id = wid.y;
  let tid = lid.x;
  let d = tid * 4u;
  let seq_pos = select(params.seq_pos, pos_buf[1], params.seq_pos == 0xFFFFFFFFu);

  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = kv_heads * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let qb0 = params.q_offset + (qh0 + 0u) * q_stride + d;
  let qb1 = params.q_offset + (qh0 + min(1u, gqa_ratio - 1u)) * q_stride + d;
  let qb2 = params.q_offset + (qh0 + min(2u, gqa_ratio - 1u)) * q_stride + d;
  let qb3 = params.q_offset + (qh0 + min(3u, gqa_ratio - 1u)) * q_stride + d;
  let q0 = vec4<f32>(act[qb0], act[qb0 + 1u], act[qb0 + 2u], act[qb0 + 3u]);
  let q1 = vec4<f32>(act[qb1], act[qb1 + 1u], act[qb1 + 2u], act[qb1 + 3u]);
  let q2 = vec4<f32>(act[qb2], act[qb2 + 1u], act[qb2 + 2u], act[qb2 + 3u]);
  let q3 = vec4<f32>(act[qb3], act[qb3 + 1u], act[qb3 + 2u], act[qb3 + 3u]);

  let inv_sqrt_d: f32 = 0.0625;

  var m4 = vec4<f32>(-1e30);
  var l4 = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let chunk = (seq_pos + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk, seq_pos);
  let t_end = min(t_start + chunk, seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let ki = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    red_buf[tid] = vec4<f32>(dot(q0, ki), dot(q1, ki), dot(q2, ki), dot(q3, ki));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_buf[tid] += red_buf[tid + s]; }
      workgroupBarrier();
    }

    let score = red_buf[0] * inv_sqrt_d;
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;

    let v_src = v_head_base + t * 65u;
    let vi = unpack4_i8(kv_cache[v_src + tid]) * scales.y;
    acc0 = acc0 * alpha.x + vi * beta.x;
    acc1 = acc1 * alpha.y + vi * beta.y;
    acc2 = acc2 * alpha.z + vi * beta.z;
    acc3 = acc3 * alpha.w + vi * beta.w;
    m4 = m_new;
    workgroupBarrier();
  }

  if (split_id == 0u) {
    let k_in = params.k_offset + kv_head * 256u + d;
    let v_in = params.v_offset + kv_head * 256u + d;
    let kv4 = vec4<f32>(act[k_in], act[k_in + 1u], act[k_in + 2u], act[k_in + 3u]);
    let vv4 = vec4<f32>(act[v_in], act[v_in + 1u], act[v_in + 2u], act[v_in + 3u]);

    cur_kv_max[tid] = vec2<f32>(
      max(max(abs(kv4.x), abs(kv4.y)), max(abs(kv4.z), abs(kv4.w))),
      max(max(abs(vv4.x), abs(vv4.y)), max(abs(vv4.z), abs(vv4.w))),
    );
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { cur_kv_max[tid] = max(cur_kv_max[tid], cur_kv_max[tid + s]); }
      workgroupBarrier();
    }
    let km = cur_kv_max[0].x;
    let vm = cur_kv_max[0].y;
    let packed_scales = pack2x16float(vec2<f32>(max(km / 127.0, 1e-6), max(vm / 127.0, 1e-6)));
    let scales = unpack2x16float(packed_scales);
    let qk = pack4_i8(kv4 / scales.x);
    let qv = pack4_i8(vv4 / scales.y);

    let k_dst = k_head_base + seq_pos * 65u;
    let v_dst = v_head_base + seq_pos * 65u;
    kv_cache[k_dst + tid] = qk;
    kv_cache[v_dst + tid] = qv;
    if (tid == 0u) {
      kv_cache[k_dst + 64u] = packed_scales;
    }

    let k_deq = unpack4_i8(qk) * scales.x;
    red_buf[tid] = vec4<f32>(dot(q0, k_deq), dot(q1, k_deq), dot(q2, k_deq), dot(q3, k_deq));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_buf[tid] += red_buf[tid + s]; }
      workgroupBarrier();
    }

    let score = red_buf[0] * inv_sqrt_d;
    let m_new = max(m4, score);
    let alpha = exp(m4 - m_new);
    let beta = exp(score - m_new);
    l4 = l4 * alpha + beta;
    let v_deq = unpack4_i8(qv) * scales.y;
    acc0 = acc0 * alpha.x + v_deq * beta.x;
    acc1 = acc1 * alpha.y + v_deq * beta.y;
    acc2 = acc2 * alpha.z + v_deq * beta.z;
    acc3 = acc3 * alpha.w + v_deq * beta.w;
    m4 = m_new;
  }

  let ob0 = ((qh0 + 0u) * num_splits + split_id) * 260u;
  split_out[ob0 + d] = acc0.x; split_out[ob0 + d + 1u] = acc0.y; split_out[ob0 + d + 2u] = acc0.z; split_out[ob0 + d + 3u] = acc0.w;
  if (tid == 0u) { split_out[ob0 + 256u] = m4.x; split_out[ob0 + 257u] = l4.x; }

  if (gqa_ratio > 1u) {
    let ob1 = ((qh0 + 1u) * num_splits + split_id) * 260u;
    split_out[ob1 + d] = acc1.x; split_out[ob1 + d + 1u] = acc1.y; split_out[ob1 + d + 2u] = acc1.z; split_out[ob1 + d + 3u] = acc1.w;
    if (tid == 0u) { split_out[ob1 + 256u] = m4.y; split_out[ob1 + 257u] = l4.y; }
  }
  if (gqa_ratio > 2u) {
    let ob2 = ((qh0 + 2u) * num_splits + split_id) * 260u;
    split_out[ob2 + d] = acc2.x; split_out[ob2 + d + 1u] = acc2.y; split_out[ob2 + d + 2u] = acc2.z; split_out[ob2 + d + 3u] = acc2.w;
    if (tid == 0u) { split_out[ob2 + 256u] = m4.z; split_out[ob2 + 257u] = l4.z; }
  }
  if (gqa_ratio > 3u) {
    let ob3 = ((qh0 + 3u) * num_splits + split_id) * 260u;
    split_out[ob3 + d] = acc3.x; split_out[ob3 + d + 1u] = acc3.y; split_out[ob3 + d + 2u] = acc3.z; split_out[ob3 + d + 3u] = acc3.w;
    if (tid == 0u) { split_out[ob3 + 256u] = m4.w; split_out[ob3 + 257u] = l4.w; }
  }
}
"#;

// Shader 38: Tier 2 FP16 KV Split-K Prefill (f16, no subgroups)
const WGSL_GQA_FP16_PREFILL_T2_F16: &str = r#"
enable f16;

struct GqaPrefillParams {
  mode: u32,
  layer_kv_u32_offset: u32,
  max_seq: u32,
  start_pos: u32,
  num_tokens: u32,
  buf_a_offset: u32,
  buf_b_offset: u32,
  num_splits: u32,
};

@group(0) @binding(0) var<uniform> params: GqaPrefillParams;
@group(0) @binding(1) var<storage, read> work: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> red_a: array<vec4<f32>, 64>;
var<workgroup> red_b: array<vec4<f32>, 64>;

fn write_split_out(
  tok: u32,
  q_head: u32,
  q_heads: u32,
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * q_heads + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, q_heads: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * (q_heads * 512u) + q_head * 512u + d;
  return vec4<f32>(work[b], work[b + 1u], work[b + 2u], work[b + 3u]);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let d2 = tid * 2u;
  let d = tid * 4u;
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let q_heads = kv_heads * gqa_ratio;
  let head_u32_stride = params.max_seq * 128u;
  let kv_plane_u32s = kv_heads * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x;
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let pos = params.start_pos + tok;
    let k_dst = params.layer_kv_u32_offset + kv_head * head_u32_stride + pos * 128u + d2;
    let v_dst = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride + pos * 128u + d2;
    kv_cache[k_dst] = bitcast<u32>(vec2<f16>(f16(work[k_in]), f16(work[k_in + 1u])));
    kv_cache[k_dst + 1u] = bitcast<u32>(vec2<f16>(f16(work[k_in + 2u]), f16(work[k_in + 3u])));
    kv_cache[v_dst] = bitcast<u32>(vec2<f16>(f16(work[v_in]), f16(work[v_in + 1u])));
    kv_cache[v_dst + 1u] = bitcast<u32>(vec2<f16>(f16(work[v_in + 2u]), f16(work[v_in + 3u])));
    return;
  }

  let kv_head = wid.x;
  let qh0 = kv_head * gqa_ratio;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, q_heads, d);
  let q1 = load_q_vec4(tok0, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q2 = load_q_vec4(tok0, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q3 = load_q_vec4(tok0, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, q_heads, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let inv_sqrt_d: f32 = 0.0625;

  var mA = vec4<f32>(-1e30);
  var lA = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  var mB = vec4<f32>(-1e30);
  var lB = vec4<f32>(0.0);
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);

  let causal0 = params.start_pos + tok0 + 1u;
  let chunk_t = (causal0 + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk_t, causal0);
  let t_end = min(t_start + chunk_t, causal0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let k_vec = vec4<f32>(k0.x, k0.y, k1.x, k1.y);

    red_a[tid] = vec4<f32>(dot(q0, k_vec), dot(q1, k_vec), dot(q2, k_vec), dot(q3, k_vec));
    red_b[tid] = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_a[tid] += red_a[tid + s];
        red_b[tid] += red_b[tid + s];
      }
      workgroupBarrier();
    }

    let dotA = red_a[0];
    let dotB = red_b[0];

    let v_src = v_head_base + t * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    let v_vec = vec4<f32>(v0.x, v0.y, v1.x, v1.y);

    let scoreA = dotA * inv_sqrt_d;
    let mA_new = max(mA, scoreA);
    let alphaA = exp(mA - mA_new);
    let betaA = exp(scoreA - mA_new);
    lA = lA * alphaA + betaA;
    acc0 = acc0 * alphaA.x + v_vec * betaA.x;
    acc1 = acc1 * alphaA.y + v_vec * betaA.y;
    acc2 = acc2 * alphaA.z + v_vec * betaA.z;
    acc3 = acc3 * alphaA.w + v_vec * betaA.w;
    mA = mA_new;

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
    workgroupBarrier();
  }

  if (has_t1 && split_id == 0u) {
    let t_tail = causal0;
    let k_src = k_head_base + t_tail * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let k_vec = vec4<f32>(k0.x, k0.y, k1.x, k1.y);

    red_b[tid] = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_b[tid] += red_b[tid + s]; }
      workgroupBarrier();
    }

    let v_src = v_head_base + t_tail * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    let v_vec = vec4<f32>(v0.x, v0.y, v1.x, v1.y);

    let scoreB = red_b[0] * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  write_split_out(tok0, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  if (gqa_ratio > 1u) { write_split_out(tok0, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc1, mA.y, lA.y); }
  if (gqa_ratio > 2u) { write_split_out(tok0, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc2, mA.z, lA.z); }
  if (gqa_ratio > 3u) { write_split_out(tok0, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc3, mA.w, lA.w); }
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    if (gqa_ratio > 1u) { write_split_out(tok1, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc5, mB.y, lB.y); }
    if (gqa_ratio > 2u) { write_split_out(tok1, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc6, mB.z, lB.z); }
    if (gqa_ratio > 3u) { write_split_out(tok1, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc7, mB.w, lB.w); }
  }
}
"#;

// Shader 39: Tier 3 FP16 KV Split-K Prefill (pure f32, pack2x16float/unpack2x16float, no f16, no subgroups)
const WGSL_GQA_FP16_PREFILL_T3_F32: &str = r#"
struct GqaPrefillParams {
  mode: u32,
  layer_kv_u32_offset: u32,
  max_seq: u32,
  start_pos: u32,
  num_tokens: u32,
  buf_a_offset: u32,
  buf_b_offset: u32,
  num_splits: u32,
};

@group(0) @binding(0) var<uniform> params: GqaPrefillParams;
@group(0) @binding(1) var<storage, read> work: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> red_a: array<vec4<f32>, 64>;
var<workgroup> red_b: array<vec4<f32>, 64>;

fn write_split_out(
  tok: u32,
  q_head: u32,
  q_heads: u32,
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * q_heads + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, q_heads: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * (q_heads * 512u) + q_head * 512u + d;
  return vec4<f32>(work[b], work[b + 1u], work[b + 2u], work[b + 3u]);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let d2 = tid * 2u;
  let d = tid * 4u;
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let q_heads = kv_heads * gqa_ratio;
  let head_u32_stride = params.max_seq * 128u;
  let kv_plane_u32s = kv_heads * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x;
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let pos = params.start_pos + tok;
    let k_dst = params.layer_kv_u32_offset + kv_head * head_u32_stride + pos * 128u + d2;
    let v_dst = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride + pos * 128u + d2;
    kv_cache[k_dst] = pack2x16float(vec2<f32>(work[k_in], work[k_in + 1u]));
    kv_cache[k_dst + 1u] = pack2x16float(vec2<f32>(work[k_in + 2u], work[k_in + 3u]));
    kv_cache[v_dst] = pack2x16float(vec2<f32>(work[v_in], work[v_in + 1u]));
    kv_cache[v_dst + 1u] = pack2x16float(vec2<f32>(work[v_in + 2u], work[v_in + 3u]));
    return;
  }

  let kv_head = wid.x;
  let qh0 = kv_head * gqa_ratio;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, q_heads, d);
  let q1 = load_q_vec4(tok0, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q2 = load_q_vec4(tok0, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q3 = load_q_vec4(tok0, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, q_heads, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let inv_sqrt_d: f32 = 0.0625;

  var mA = vec4<f32>(-1e30);
  var lA = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  var mB = vec4<f32>(-1e30);
  var lB = vec4<f32>(0.0);
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);

  let causal0 = params.start_pos + tok0 + 1u;
  let chunk_t = (causal0 + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk_t, causal0);
  let t_end = min(t_start + chunk_t, causal0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let k_vec = vec4<f32>(k0.x, k0.y, k1.x, k1.y);

    red_a[tid] = vec4<f32>(dot(q0, k_vec), dot(q1, k_vec), dot(q2, k_vec), dot(q3, k_vec));
    red_b[tid] = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_a[tid] += red_a[tid + s];
        red_b[tid] += red_b[tid + s];
      }
      workgroupBarrier();
    }

    let dotA = red_a[0];
    let dotB = red_b[0];

    let v_src = v_head_base + t * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    let v_vec = vec4<f32>(v0.x, v0.y, v1.x, v1.y);

    let scoreA = dotA * inv_sqrt_d;
    let mA_new = max(mA, scoreA);
    let alphaA = exp(mA - mA_new);
    let betaA = exp(scoreA - mA_new);
    lA = lA * alphaA + betaA;
    acc0 = acc0 * alphaA.x + v_vec * betaA.x;
    acc1 = acc1 * alphaA.y + v_vec * betaA.y;
    acc2 = acc2 * alphaA.z + v_vec * betaA.z;
    acc3 = acc3 * alphaA.w + v_vec * betaA.w;
    mA = mA_new;

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
    workgroupBarrier();
  }

  if (has_t1 && split_id == 0u) {
    let t_tail = causal0;
    let k_src = k_head_base + t_tail * 128u + d2;
    let k0 = unpack2x16float(kv_cache[k_src]);
    let k1 = unpack2x16float(kv_cache[k_src + 1u]);
    let k_vec = vec4<f32>(k0.x, k0.y, k1.x, k1.y);

    red_b[tid] = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_b[tid] += red_b[tid + s]; }
      workgroupBarrier();
    }

    let v_src = v_head_base + t_tail * 128u + d2;
    let v0 = unpack2x16float(kv_cache[v_src]);
    let v1 = unpack2x16float(kv_cache[v_src + 1u]);
    let v_vec = vec4<f32>(v0.x, v0.y, v1.x, v1.y);

    let scoreB = red_b[0] * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  write_split_out(tok0, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  if (gqa_ratio > 1u) { write_split_out(tok0, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc1, mA.y, lA.y); }
  if (gqa_ratio > 2u) { write_split_out(tok0, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc2, mA.z, lA.z); }
  if (gqa_ratio > 3u) { write_split_out(tok0, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc3, mA.w, lA.w); }
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    if (gqa_ratio > 1u) { write_split_out(tok1, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc5, mB.y, lB.y); }
    if (gqa_ratio > 2u) { write_split_out(tok1, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc6, mB.z, lB.z); }
    if (gqa_ratio > 3u) { write_split_out(tok1, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc7, mB.w, lB.w); }
  }
}
"#;

// Shader 40: Tier 2 & 3 INT8 KV Split-K Prefill (pure f32, no subgroups)
const WGSL_GQA_INT8_PREFILL_T2: &str = r#"
struct GqaPrefillParams {
  mode: u32,
  layer_kv_u32_offset: u32,
  max_seq: u32,
  start_pos: u32,
  num_tokens: u32,
  buf_a_offset: u32,
  buf_b_offset: u32,
  num_splits: u32,
};

@group(0) @binding(0) var<uniform> params: GqaPrefillParams;
@group(0) @binding(1) var<storage, read> work: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> split_out: array<f32>;

var<workgroup> red_a: array<vec4<f32>, 64>;
var<workgroup> red_b: array<vec4<f32>, 64>;
var<workgroup> kv_max_buf: array<vec2<f32>, 64>;

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

fn write_split_out(
  tok: u32,
  q_head: u32,
  q_heads: u32,
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * q_heads + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, q_heads: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * (q_heads * 512u) + q_head * 512u + d;
  return vec4<f32>(work[b], work[b + 1u], work[b + 2u], work[b + 3u]);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let d = tid * 4u;
  let kv_heads = select((params.num_splits >> 16u) & 0xFFu, 4u, (params.num_splits >> 16u) == 0u);
  let gqa_ratio = select((params.num_splits >> 24u) & 0xFFu, 4u, (params.num_splits >> 24u) == 0u);
  let q_heads = kv_heads * gqa_ratio;
  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = kv_heads * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x;
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * (kv_heads * 256u) + kv_head * 256u + d;
    let kv4 = vec4<f32>(work[k_in], work[k_in + 1u], work[k_in + 2u], work[k_in + 3u]);
    let vv4 = vec4<f32>(work[v_in], work[v_in + 1u], work[v_in + 2u], work[v_in + 3u]);

    kv_max_buf[tid] = vec2<f32>(
      max(max(abs(kv4.x), abs(kv4.y)), max(abs(kv4.z), abs(kv4.w))),
      max(max(abs(vv4.x), abs(vv4.y)), max(abs(vv4.z), abs(vv4.w))),
    );
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { kv_max_buf[tid] = max(kv_max_buf[tid], kv_max_buf[tid + s]); }
      workgroupBarrier();
    }
    let km = kv_max_buf[0].x;
    let vm = kv_max_buf[0].y;
    let packed_scales = pack2x16float(vec2<f32>(max(km / 127.0, 1e-6), max(vm / 127.0, 1e-6)));
    let scales = unpack2x16float(packed_scales);

    let pos = params.start_pos + tok;
    let k_dst = params.layer_kv_u32_offset + kv_head * head_u32_stride + pos * 65u;
    let v_dst = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride + pos * 65u;
    kv_cache[k_dst + tid] = pack4_i8(kv4 / scales.x);
    kv_cache[v_dst + tid] = pack4_i8(vv4 / scales.y);
    if (tid == 0u) {
      kv_cache[k_dst + 64u] = packed_scales;
    }
    return;
  }

  let kv_head = wid.x;
  let qh0 = kv_head * gqa_ratio;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits & 0xFFFFu, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, q_heads, d);
  let q1 = load_q_vec4(tok0, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q2 = load_q_vec4(tok0, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q3 = load_q_vec4(tok0, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, q_heads, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + min(1u, gqa_ratio - 1u), q_heads, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + min(2u, gqa_ratio - 1u), q_heads, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + min(3u, gqa_ratio - 1u), q_heads, d);

  let inv_sqrt_d: f32 = 0.0625;

  var mA = vec4<f32>(-1e30);
  var lA = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  var mB = vec4<f32>(-1e30);
  var lB = vec4<f32>(0.0);
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);

  let causal0 = params.start_pos + tok0 + 1u;
  let chunk_t = (causal0 + num_splits - 1u) / num_splits;
  let t_start = min(split_id * chunk_t, causal0);
  let t_end = min(t_start + chunk_t, causal0);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let k_vec = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    red_a[tid] = vec4<f32>(dot(q0, k_vec), dot(q1, k_vec), dot(q2, k_vec), dot(q3, k_vec));
    red_b[tid] = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        red_a[tid] += red_a[tid + s];
        red_b[tid] += red_b[tid + s];
      }
      workgroupBarrier();
    }

    let dotA = red_a[0];
    let dotB = red_b[0];

    let v_src = v_head_base + t * 65u;
    let v_vec = unpack4_i8(kv_cache[v_src + tid]) * scales.y;

    let scoreA = dotA * inv_sqrt_d;
    let mA_new = max(mA, scoreA);
    let alphaA = exp(mA - mA_new);
    let betaA = exp(scoreA - mA_new);
    lA = lA * alphaA + betaA;
    acc0 = acc0 * alphaA.x + v_vec * betaA.x;
    acc1 = acc1 * alphaA.y + v_vec * betaA.y;
    acc2 = acc2 * alphaA.z + v_vec * betaA.z;
    acc3 = acc3 * alphaA.w + v_vec * betaA.w;
    mA = mA_new;

    let scoreB = dotB * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
    workgroupBarrier();
  }

  if (has_t1 && split_id == 0u) {
    let t_tail = causal0;
    let k_src = k_head_base + t_tail * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let k_vec = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    red_b[tid] = vec4<f32>(dot(q4, k_vec), dot(q5, k_vec), dot(q6, k_vec), dot(q7, k_vec));
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red_b[tid] += red_b[tid + s]; }
      workgroupBarrier();
    }

    let v_src = v_head_base + t_tail * 65u;
    let v_vec = unpack4_i8(kv_cache[v_src + tid]) * scales.y;

    let scoreB = red_b[0] * inv_sqrt_d;
    let mB_new = max(mB, scoreB);
    let alphaB = exp(mB - mB_new);
    let betaB = exp(scoreB - mB_new);
    lB = lB * alphaB + betaB;
    acc4 = acc4 * alphaB.x + v_vec * betaB.x;
    acc5 = acc5 * alphaB.y + v_vec * betaB.y;
    acc6 = acc6 * alphaB.z + v_vec * betaB.z;
    acc7 = acc7 * alphaB.w + v_vec * betaB.w;
    mB = mB_new;
  }

  write_split_out(tok0, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  if (gqa_ratio > 1u) { write_split_out(tok0, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc1, mA.y, lA.y); }
  if (gqa_ratio > 2u) { write_split_out(tok0, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc2, mA.z, lA.z); }
  if (gqa_ratio > 3u) { write_split_out(tok0, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc3, mA.w, lA.w); }
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, q_heads, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    if (gqa_ratio > 1u) { write_split_out(tok1, qh0 + 1u, q_heads, split_id, num_splits, d, tid, acc5, mB.y, lB.y); }
    if (gqa_ratio > 2u) { write_split_out(tok1, qh0 + 2u, q_heads, split_id, num_splits, d, tid, acc6, mB.z, lB.z); }
    if (gqa_ratio > 3u) { write_split_out(tok1, qh0 + 3u, q_heads, split_id, num_splits, d, tid, acc7, mB.w, lB.w); }
  }
}
"#;

// Shader 41: Tier 2 & 3 Chunkwise Parallel Gated DeltaNet WY Prep (pure f32, no subgroups)
const WGSL_GDN_CHUNK_PREP_T2: &str = r#"
struct GdnPrepParams {
  num_tokens: u32,
  tok_start: u32,
  aux_dn_offset: u32,
  qkv_offset: u32,
  ab_offset: u32,
  scratch_offset: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GdnPrepParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> work: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

var<workgroup> sh_M: array<f32, 1024>;
var<workgroup> sh_a: array<f32, 1024>;
var<workgroup> sh_b: array<f32, 1024>;
var<workgroup> sh_red_q: array<f32, 128>;
var<workgroup> sh_red_k: array<f32, 128>;
var<workgroup> sh_G: array<f32, 32>;
var<workgroup> sh_beta: array<f32, 32>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let C = params.num_tokens;
  if (C == 0u) {
    return;
  }
  let v_heads = select(params._pad0 & 0xFFFFu, 32u, params._pad0 == 0u);
  let k_heads = select(params._pad0 >> 16u, 16u, params._pad0 == 0u);
  let ratio = max(v_heads / max(k_heads, 1u), 1u);
  let qk_dim = k_heads * 128u;
  let v_dim = v_heads * 128u;
  let conv_dim = qk_dim * 2u + v_dim;

  let v_head = wid.x;
  let j = lid.x;
  let k_head = v_head / ratio;

  let G_off = params.scratch_offset + 0u;
  let QK_off = params.scratch_offset + 1024u;
  let W_off = params.scratch_offset + 33792u;
  let Q_off = params.scratch_offset + 164864u;
  let K_dec_off = params.scratch_offset + 295936u;
  let U_off = params.scratch_offset + 427008u;

  let a_neg = aux[params.aux_dn_offset + v_head];
  let dt_b = aux[params.aux_dn_offset + v_heads + v_head];
  var g_cum: f32 = 0.0;

  for (var t: u32 = 0u; t < C; t = t + 1u) {
    let qkv_t = params.qkv_offset + (params.tok_start + t) * conv_dim;
    let q_raw = work[qkv_t + k_head * 128u + j];
    let k_raw = work[qkv_t + qk_dim + k_head * 128u + j];

    sh_red_q[j] = q_raw * q_raw;
    sh_red_k[j] = k_raw * k_raw;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (j < s) {
        sh_red_q[j] += sh_red_q[j + s];
        sh_red_k[j] += sh_red_k[j + s];
      }
      workgroupBarrier();
    }
    let inv_q = inverseSqrt(sh_red_q[0] + 1e-6) * 0.08838834764831845;
    let inv_k = inverseSqrt(sh_red_k[0] + 1e-6);
    let sc_idx = (v_head * 128u + j) * 32u + t;
    scratch[Q_off + sc_idx] = q_raw * inv_q;
    scratch[K_dec_off + sc_idx] = k_raw * inv_k;

    if (j == 0u) {
      let ab_t = params.ab_offset + (params.tok_start + t) * (v_heads * 2u);
      let b_raw = work[ab_t + v_head];
      let a_raw = work[ab_t + v_heads + v_head];
      sh_beta[t] = 1.0 / (1.0 + exp(-b_raw));
      let u = a_raw + dt_b;
      let sp = select(log(1.0 + exp(u)), u, u > 20.0);
      g_cum += a_neg * sp;
      sh_G[t] = g_cum;
      scratch[G_off + v_head * 32u + t] = g_cum;
    }
    workgroupBarrier();
  }

  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let idx = k * 128u + j;
    let tr = idx >> 5u;
    let tc = idx & 31u;
    sh_M[idx] = select(0.0, 1.0, tr == tc);
  }
  storageBarrier();
  workgroupBarrier();

  let tx = j & 31u;
  let r0 = (j >> 5u) * 8u;
  var kk0: f32 = 0.0; var kk1: f32 = 0.0; var kk2: f32 = 0.0; var kk3: f32 = 0.0;
  var kk4: f32 = 0.0; var kk5: f32 = 0.0; var kk6: f32 = 0.0; var kk7: f32 = 0.0;
  var qk0: f32 = 0.0; var qk1: f32 = 0.0; var qk2: f32 = 0.0; var qk3: f32 = 0.0;
  var qk4: f32 = 0.0; var qk5: f32 = 0.0; var qk6: f32 = 0.0; var qk7: f32 = 0.0;

  for (var i_tile: u32 = 0u; i_tile < 4u; i_tile = i_tile + 1u) {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let idx = k * 128u + j;
      let i_sub = idx >> 5u;
      let t_sub = idx & 31u;
      if (t_sub < C) {
        let sc_idx = (v_head * 128u + i_tile * 32u + i_sub) * 32u + t_sub;
        sh_a[idx] = scratch[K_dec_off + sc_idx];
        sh_b[idx] = scratch[Q_off + sc_idx];
      } else {
        sh_a[idx] = 0.0;
        sh_b[idx] = 0.0;
      }
    }
    workgroupBarrier();

    for (var ii: u32 = 0u; ii < 32u; ii = ii + 1u) {
      let rb = ii * 32u;
      let k_tx = sh_a[rb + tx];
      let q_tx = sh_b[rb + tx];
      let kr0 = sh_a[rb + r0 + 0u]; let kr1 = sh_a[rb + r0 + 1u];
      let kr2 = sh_a[rb + r0 + 2u]; let kr3 = sh_a[rb + r0 + 3u];
      let kr4 = sh_a[rb + r0 + 4u]; let kr5 = sh_a[rb + r0 + 5u];
      let kr6 = sh_a[rb + r0 + 6u]; let kr7 = sh_a[rb + r0 + 7u];
      kk0 += k_tx * kr0; kk1 += k_tx * kr1; kk2 += k_tx * kr2; kk3 += k_tx * kr3;
      kk4 += k_tx * kr4; kk5 += k_tx * kr5; kk6 += k_tx * kr6; kk7 += k_tx * kr7;
      qk0 += q_tx * kr0; qk1 += q_tx * kr1; qk2 += q_tx * kr2; qk3 += q_tx * kr3;
      qk4 += q_tx * kr4; qk5 += q_tx * kr5; qk6 += q_tx * kr6; qk7 += q_tx * kr7;
    }
    workgroupBarrier();
  }

  if (tx < C) {
    let gt = sh_G[tx];
    let bt = sh_beta[tx];
    let qk_row = (v_head * 32u + tx) * 32u;
    if (r0 + 0u <= tx) { let d = exp(gt - sh_G[r0 + 0u]); scratch[QK_off + qk_row + r0 + 0u] = d * qk0; if (r0 + 0u < tx) { sh_M[tx * 32u + r0 + 0u] = bt * d * kk0; } }
    if (r0 + 1u <= tx) { let d = exp(gt - sh_G[r0 + 1u]); scratch[QK_off + qk_row + r0 + 1u] = d * qk1; if (r0 + 1u < tx) { sh_M[tx * 32u + r0 + 1u] = bt * d * kk1; } }
    if (r0 + 2u <= tx) { let d = exp(gt - sh_G[r0 + 2u]); scratch[QK_off + qk_row + r0 + 2u] = d * qk2; if (r0 + 2u < tx) { sh_M[tx * 32u + r0 + 2u] = bt * d * kk2; } }
    if (r0 + 3u <= tx) { let d = exp(gt - sh_G[r0 + 3u]); scratch[QK_off + qk_row + r0 + 3u] = d * qk3; if (r0 + 3u < tx) { sh_M[tx * 32u + r0 + 3u] = bt * d * kk3; } }
    if (r0 + 4u <= tx) { let d = exp(gt - sh_G[r0 + 4u]); scratch[QK_off + qk_row + r0 + 4u] = d * qk4; if (r0 + 4u < tx) { sh_M[tx * 32u + r0 + 4u] = bt * d * kk4; } }
    if (r0 + 5u <= tx) { let d = exp(gt - sh_G[r0 + 5u]); scratch[QK_off + qk_row + r0 + 5u] = d * qk5; if (r0 + 5u < tx) { sh_M[tx * 32u + r0 + 5u] = bt * d * kk5; } }
    if (r0 + 6u <= tx) { let d = exp(gt - sh_G[r0 + 6u]); scratch[QK_off + qk_row + r0 + 6u] = d * qk6; if (r0 + 6u < tx) { sh_M[tx * 32u + r0 + 6u] = bt * d * kk6; } }
    if (r0 + 7u <= tx) { let d = exp(gt - sh_G[r0 + 7u]); scratch[QK_off + qk_row + r0 + 7u] = d * qk7; if (r0 + 7u < tx) { sh_M[tx * 32u + r0 + 7u] = bt * d * kk7; } }
  }
  workgroupBarrier();

  for (var t: u32 = 1u; t < C; t = t + 1u) {
    var val: f32 = 0.0;
    if (j < t) {
      var acc: f32 = sh_M[t * 32u + j];
      for (var m: u32 = j + 1u; m < t; m = m + 1u) {
        acc += sh_M[t * 32u + m] * sh_M[m * 32u + j];
      }
      val = -acc;
    }
    workgroupBarrier();
    if (j < t) {
      sh_M[t * 32u + j] = val;
    }
    workgroupBarrier();
  }

  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let idx = k * 128u + j;
    let t = idx >> 5u;
    let r = idx & 31u;
    if (t < C && r <= t) {
      sh_M[idx] = sh_M[idx] * sh_beta[r];
    }
  }
  workgroupBarrier();
  if (j < C) {
    sh_beta[j] = exp(sh_G[j]);
  }
  workgroupBarrier();

  let t_grp = j >> 5u;
  let j_sub = j & 31u;
  for (var j_tile: u32 = 0u; j_tile < 4u; j_tile = j_tile + 1u) {
    let j_base = j_tile * 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let r = k * 4u + t_grp;
      if (r < C) {
        sh_a[r * 32u + j_sub] = scratch[K_dec_off + (v_head * 128u + j_base + j_sub) * 32u + r] * sh_beta[r];
        sh_b[r * 32u + j_sub] = work[params.qkv_offset + (params.tok_start + r) * conv_dim + qk_dim * 2u + v_head * 128u + j_base + j_sub];
      }
    }
    workgroupBarrier();

    for (var t: u32 = t_grp; t < C; t = t + 4u) {
      var w_acc: f32 = 0.0;
      var u_acc: f32 = 0.0;
      let m_row = t * 32u;
      for (var r: u32 = 0u; r <= t; r = r + 1u) {
        let tr_br = sh_M[m_row + r];
        let s_idx = r * 32u + j_sub;
        w_acc += tr_br * sh_a[s_idx];
        u_acc += tr_br * sh_b[s_idx];
      }
      scratch[W_off + (v_head * 128u + j_base + j_sub) * 32u + t] = w_acc;
      scratch[U_off + (v_head * 32u + t) * 128u + j_base + j_sub] = u_acc;
    }
    workgroupBarrier();
  }

  let g_last = sh_G[C - 1u];
  let k_base_j = K_dec_off + (v_head * 128u + j) * 32u;
  for (var t: u32 = 0u; t < C; t = t + 1u) {
    scratch[k_base_j + t] = exp(g_last - sh_G[t]) * scratch[k_base_j + t];
  }
}
"#;

// Shader 42: Tier 2 & 3 Chunkwise Parallel Gated DeltaNet Apply & Gated RMSNorm (pure f32, no subgroups)
const WGSL_GDN_CHUNK_APPLY_T2: &str = r#"
struct GdnApplyParams {
  mode: u32,
  num_tokens: u32,
  tok_start: u32,
  state_f32_offset: u32,
  aux_dn_offset: u32,
  z_offset: u32,
  out_offset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: GdnApplyParams;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> work: array<f32>;
@group(0) @binding(3) var<storage, read> scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> dn_state: array<f32>;
@group(0) @binding(5) var<storage, read_write> attn_out: array<f32>;

var<workgroup> sh_a: array<f32, 512>;
var<workgroup> sh_b: array<f32, 512>;
var<workgroup> sh_red: array<f32, 512>;
var<workgroup> sh_delta: array<f32, 512>;
var<workgroup> sh_raw: array<f32, 512>;
var<workgroup> sh_qk: array<f32, 1024>;
var<workgroup> sh_G: array<f32, 32>;

fn dot16_a(base: u32, sv0: vec4<f32>, sv1: vec4<f32>, sv2: vec4<f32>, sv3: vec4<f32>) -> f32 {
  let v0 = vec4<f32>(sh_a[base + 0u], sh_a[base + 1u], sh_a[base + 2u], sh_a[base + 3u]);
  let v1 = vec4<f32>(sh_a[base + 4u], sh_a[base + 5u], sh_a[base + 6u], sh_a[base + 7u]);
  let v2 = vec4<f32>(sh_a[base + 8u], sh_a[base + 9u], sh_a[base + 10u], sh_a[base + 11u]);
  let v3 = vec4<f32>(sh_a[base + 12u], sh_a[base + 13u], sh_a[base + 14u], sh_a[base + 15u]);
  return dot(sv0, v0) + dot(sv1, v1) + dot(sv2, v2) + dot(sv3, v3);
}

fn dot16_b(base: u32, sv0: vec4<f32>, sv1: vec4<f32>, sv2: vec4<f32>, sv3: vec4<f32>) -> f32 {
  let v0 = vec4<f32>(sh_b[base + 0u], sh_b[base + 1u], sh_b[base + 2u], sh_b[base + 3u]);
  let v1 = vec4<f32>(sh_b[base + 4u], sh_b[base + 5u], sh_b[base + 6u], sh_b[base + 7u]);
  let v2 = vec4<f32>(sh_b[base + 8u], sh_b[base + 9u], sh_b[base + 10u], sh_b[base + 11u]);
  let v3 = vec4<f32>(sh_b[base + 12u], sh_b[base + 13u], sh_b[base + 14u], sh_b[base + 15u]);
  return dot(sv0, v0) + dot(sv1, v1) + dot(sv2, v2) + dot(sv3, v3);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let v_heads = select(params._pad0 & 0xFFFFu, 32u, params._pad0 == 0u);
  let v_dim = v_heads * 128u;

  if (params.mode == 1u) {
    let v_head = wid.x;
    let tok = wid.y;
    let j = lid.x;
    let out_idx = params.out_offset + tok * v_dim + v_head * 128u + j;
    let out_val = attn_out[out_idx];
    sh_red[j] = out_val * out_val;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (j < s) { sh_red[j] += sh_red[j + s]; }
      workgroupBarrier();
    }
    let inv_rms = inverseSqrt(sh_red[0] / 128.0 + 1e-6);
    let z_val = work[params.z_offset + tok * v_dim + v_head * 128u + j];
    let z_silu = z_val / (1.0 + exp(-z_val));
    let gdn_norm_w = aux[params.aux_dn_offset + v_heads * 2u + j];
    attn_out[out_idx] = (out_val * inv_rms) * z_silu * gdn_norm_w;
    return;
  }

  let C = params.num_tokens;
  if (C == 0u) {
    return;
  }
  let v_head = wid.x;
  let j_tile = wid.y;
  let tid = lid.x;
  let j_local = tid & 15u;
  let j = j_tile * 16u + j_local;
  let i_slice = tid >> 4u;
  let i_base = i_slice * 16u;

  let G_off = 0u;
  let QK_off = 1024u;
  let W_off = 33792u;
  let Q_off = 164864u;
  let K_dec_off = 295936u;
  let U_off = 427008u;

  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let idx = k * 128u + tid;
    sh_qk[idx] = scratch[QK_off + v_head * 1024u + idx];
  }
  if (tid < C) {
    sh_G[tid] = scratch[G_off + v_head * 32u + tid];
  }

  let s_base = params.state_f32_offset + v_head * 16384u + i_base * 128u + j;
  var sv0 = vec4<f32>(dn_state[s_base + 0u * 128u], dn_state[s_base + 1u * 128u], dn_state[s_base + 2u * 128u], dn_state[s_base + 3u * 128u]);
  var sv1 = vec4<f32>(dn_state[s_base + 4u * 128u], dn_state[s_base + 5u * 128u], dn_state[s_base + 6u * 128u], dn_state[s_base + 7u * 128u]);
  var sv2 = vec4<f32>(dn_state[s_base + 8u * 128u], dn_state[s_base + 9u * 128u], dn_state[s_base + 10u * 128u], dn_state[s_base + 11u * 128u]);
  var sv3 = vec4<f32>(dn_state[s_base + 12u * 128u], dn_state[s_base + 13u * 128u], dn_state[s_base + 14u * 128u], dn_state[s_base + 15u * 128u]);

  for (var t_blk: u32 = 0u; t_blk < C; t_blk = t_blk + 4u) {
    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
      let t_abs = t_blk + tt;
      let dst_idx = tt * 128u + tid;
      if (t_abs < C) {
        let sc_idx = (v_head * 128u + tid) * 32u + t_abs;
        sh_a[dst_idx] = scratch[W_off + sc_idx];
        sh_b[dst_idx] = scratch[Q_off + sc_idx];
      } else {
        sh_a[dst_idx] = 0.0;
        sh_b[dst_idx] = 0.0;
      }
    }
    workgroupBarrier();

    let ws0 = dot16_a(0u * 128u + i_base, sv0, sv1, sv2, sv3);
    let ws1 = dot16_a(1u * 128u + i_base, sv0, sv1, sv2, sv3);
    let ws2 = dot16_a(2u * 128u + i_base, sv0, sv1, sv2, sv3);
    let ws3 = dot16_a(3u * 128u + i_base, sv0, sv1, sv2, sv3);
    let red_base = (i_slice * 16u + j_local) * 4u;
    sh_red[red_base + 0u] = ws0;
    sh_red[red_base + 1u] = ws1;
    sh_red[red_base + 2u] = ws2;
    sh_red[red_base + 3u] = ws3;
    workgroupBarrier();

    if (i_slice == 0u) {
      for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
        let t_abs = t_blk + tt;
        if (t_abs < C) {
          var w_sum: f32 = 0.0;
          for (var sl: u32 = 0u; sl < 8u; sl = sl + 1u) {
            w_sum += sh_red[(sl * 16u + j_local) * 4u + tt];
          }
          let u_val = scratch[U_off + (v_head * 32u + t_abs) * 128u + j];
          sh_delta[t_abs * 16u + j_local] = u_val - w_sum;
        }
      }
    }
    workgroupBarrier();

    let qs0 = dot16_b(0u * 128u + i_base, sv0, sv1, sv2, sv3);
    let qs1 = dot16_b(1u * 128u + i_base, sv0, sv1, sv2, sv3);
    let qs2 = dot16_b(2u * 128u + i_base, sv0, sv1, sv2, sv3);
    let qs3 = dot16_b(3u * 128u + i_base, sv0, sv1, sv2, sv3);
    sh_red[red_base + 0u] = qs0;
    sh_red[red_base + 1u] = qs1;
    sh_red[red_base + 2u] = qs2;
    sh_red[red_base + 3u] = qs3;
    workgroupBarrier();

    if (i_slice == 0u) {
      for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
        let t_abs = t_blk + tt;
        if (t_abs < C) {
          var q_sum: f32 = 0.0;
          for (var sl: u32 = 0u; sl < 8u; sl = sl + 1u) {
            q_sum += sh_red[(sl * 16u + j_local) * 4u + tt];
          }
          sh_raw[t_abs * 16u + j_local] = q_sum;
        }
      }
    }
    workgroupBarrier();
  }

  if (i_slice == 0u) {
    for (var t: u32 = 0u; t < C; t = t + 1u) {
      var causal_acc: f32 = 0.0;
      let qk_row = t * 32u;
      for (var r: u32 = 0u; r <= t; r = r + 1u) {
        causal_acc += sh_qk[qk_row + r] * sh_delta[r * 16u + j_local];
      }
      let raw_out = exp(sh_G[t]) * sh_raw[t * 16u + j_local] + causal_acc;
      attn_out[params.out_offset + (params.tok_start + t) * v_dim + v_head * 128u + j] = raw_out;
    }
  }

  let g_last = exp(sh_G[C - 1u]);
  sv0 = sv0 * g_last;
  sv1 = sv1 * g_last;
  sv2 = sv2 * g_last;
  sv3 = sv3 * g_last;

  for (var t_blk: u32 = 0u; t_blk < C; t_blk = t_blk + 4u) {
    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
      let t_abs = t_blk + tt;
      let dst_idx = tt * 128u + tid;
      if (t_abs < C) {
        sh_a[dst_idx] = scratch[K_dec_off + (v_head * 128u + tid) * 32u + t_abs];
      } else {
        sh_a[dst_idx] = 0.0;
      }
    }
    workgroupBarrier();

    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
      let t_abs = t_blk + tt;
      if (t_abs < C) {
        let d_t = sh_delta[t_abs * 16u + j_local];
        let kb = tt * 128u + i_base;
        sv0 += vec4<f32>(sh_a[kb + 0u], sh_a[kb + 1u], sh_a[kb + 2u], sh_a[kb + 3u]) * d_t;
        sv1 += vec4<f32>(sh_a[kb + 4u], sh_a[kb + 5u], sh_a[kb + 6u], sh_a[kb + 7u]) * d_t;
        sv2 += vec4<f32>(sh_a[kb + 8u], sh_a[kb + 9u], sh_a[kb + 10u], sh_a[kb + 11u]) * d_t;
        sv3 += vec4<f32>(sh_a[kb + 12u], sh_a[kb + 13u], sh_a[kb + 14u], sh_a[kb + 15u]) * d_t;
      }
    }
    workgroupBarrier();
  }

  dn_state[s_base + 0u * 128u] = sv0.x; dn_state[s_base + 1u * 128u] = sv0.y; dn_state[s_base + 2u * 128u] = sv0.z; dn_state[s_base + 3u * 128u] = sv0.w;
  dn_state[s_base + 4u * 128u] = sv1.x; dn_state[s_base + 5u * 128u] = sv1.y; dn_state[s_base + 6u * 128u] = sv1.z; dn_state[s_base + 7u * 128u] = sv1.w;
  dn_state[s_base + 8u * 128u] = sv2.x; dn_state[s_base + 9u * 128u] = sv2.y; dn_state[s_base + 10u * 128u] = sv2.z; dn_state[s_base + 11u * 128u] = sv2.w;
  dn_state[s_base + 12u * 128u] = sv3.x; dn_state[s_base + 13u * 128u] = sv3.y; dn_state[s_base + 14u * 128u] = sv3.z; dn_state[s_base + 15u * 128u] = sv3.w;
}
"#;

// ============================================================================
// ONNX MatMulNBits -> WGSL Packed Q4 Repacker + Float16 Conversions
// ============================================================================

#[no_mangle]
pub extern "C" fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007fffff;
    if exp == 0xff {
        if frac == 0 {
            return sign | 0x7c00;
        }
        let nan_frac = ((frac >> 13) as u16).max(1);
        return sign | 0x7c00 | nan_frac;
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 >= 31 {
        return sign | 0x7c00;
    }
    if exp_f16 <= 0 {
        if exp_f16 < -10 {
            return sign;
        }
        let m = frac | 0x00800000;
        let shift = (14 - exp_f16) as u32;
        let round_bit = 1u32 << (shift - 1);
        let mant = (m + round_bit - 1 + ((m >> shift) & 1)) >> shift;
        return sign | (mant as u16);
    }
    let lsb = (frac >> 13) & 1;
    let rounded = frac + 0x0fff + lsb;
    if (rounded & 0x00800000) != 0 {
        let new_exp = exp_f16 + 1;
        if new_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((new_exp as u16) << 10);
    }
    sign | ((exp_f16 as u16) << 10) | ((rounded >> 13) as u16)
}

#[no_mangle]
pub unsafe extern "C" fn f16_slice_to_f32(
    count: u32,
    src_u16_ptr: *const u16,
    dst_f32_ptr: *mut f32,
) {
    let src = core::slice::from_raw_parts(src_u16_ptr, count as usize);
    let dst = core::slice::from_raw_parts_mut(dst_f32_ptr, count as usize);
    for i in 0..(count as usize) {
        dst[i] = f16_to_f32(src[i]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn repack_q4_rows(
    rows: u32,
    blocks_per_row: u32,
    quant_ptr: *const u8,
    scales_ptr: *const u16,
    zp_ptr: *const u8,
    out_u32_ptr: *mut u32,
) {
    let r_count = rows as usize;
    let b_count = blocks_per_row as usize;
    let zp_stride = (b_count + 1) >> 1;
    let q_bytes = core::slice::from_raw_parts(quant_ptr, r_count * b_count * 16);
    let scales = core::slice::from_raw_parts(scales_ptr, r_count * b_count);
    let zps = core::slice::from_raw_parts(zp_ptr, r_count * zp_stride);
    let out = core::slice::from_raw_parts_mut(out_u32_ptr, r_count * b_count * 5);

    for r in 0..r_count {
        for b in 0..b_count {
            let qb = (r * b_count + b) * 16;
            let dst = (r * b_count + b) * 5;
            out[dst] = u32::from_le_bytes([q_bytes[qb], q_bytes[qb + 1], q_bytes[qb + 2], q_bytes[qb + 3]]);
            out[dst + 1] = u32::from_le_bytes([q_bytes[qb + 4], q_bytes[qb + 5], q_bytes[qb + 6], q_bytes[qb + 7]]);
            out[dst + 2] = u32::from_le_bytes([q_bytes[qb + 8], q_bytes[qb + 9], q_bytes[qb + 10], q_bytes[qb + 11]]);
            out[dst + 3] = u32::from_le_bytes([q_bytes[qb + 12], q_bytes[qb + 13], q_bytes[qb + 14], q_bytes[qb + 15]]);

            let scale_h = scales[r * b_count + b];
            let scale_f = f16_to_f32(scale_h);
            let zp_byte = zps[r * zp_stride + (b >> 1)];
            let zp = if (b & 1) == 0 { zp_byte & 0x0f } else { (zp_byte >> 4) & 0x0f };
            let zero_f = -scale_f * (zp as f32);
            let zero_h = f32_to_f16(zero_f);
            out[dst + 4] = (scale_h as u32) | ((zero_h as u32) << 16);
        }
    }
}

// ============================================================================
// Exact CPU Reference Implementations (Compiled into WASM for Browser Verification)
// ============================================================================

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let f = (h & 0x3ff) as u32;
    if e == 0 {
        if f == 0 {
            return f32::from_bits(s << 31);
        }
        let mut mant = f;
        let mut exp: i32 = -14;
        while (mant & 0x400) == 0 {
            mant <<= 1;
            exp -= 1;
        }
        mant &= 0x3ff;
        let bits = (s << 31) | (((exp + 127) as u32) << 23) | (mant << 13);
        return f32::from_bits(bits);
    }
    if e == 31 {
        let bits = (s << 31) | (0xff << 23) | (f << 13);
        return f32::from_bits(bits);
    }
    let bits = (s << 31) | ((e + 112) << 23) | (f << 13);
    f32::from_bits(bits)
}

#[no_mangle]
pub unsafe extern "C" fn cpu_ref_q4_gemv(
    rows: u32,
    cols: u32,
    weights_ptr: *const u32,
    x_ptr: *const f32,
    out_ptr: *mut f32,
) {
    let blocks = (cols / 32) as usize;
    let w_slice = core::slice::from_raw_parts(weights_ptr, (rows as usize) * blocks * 5);
    let x_slice = core::slice::from_raw_parts(x_ptr, cols as usize);
    let out_slice = core::slice::from_raw_parts_mut(out_ptr, rows as usize);

    for r in 0..(rows as usize) {
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let bi = (r * blocks + b) * 5;
            let meta = w_slice[bi + 4];
            let scale = f16_to_f32((meta & 0xffff) as u16);
            let zero = f16_to_f32((meta >> 16) as u16);
            let xb = b * 32;
            for w in 0..4 {
                let p = w_slice[bi + w];
                for n in 0..8 {
                    let q = ((p >> (n * 4)) & 0xf) as f32;
                    acc += (q * scale + zero) * x_slice[xb + w * 8 + n];
                }
            }
        }
        out_slice[r] = acc;
    }
}

fn silu_f32(v: f32) -> f32 {
    // Pure no_std exp approximation via polynomial or Taylor/libm-free helper
    v / (1.0 + fast_exp_f32(-v))
}

fn fast_exp_f32(x: f32) -> f32 {
    // Clamp to avoid overflow; accurate ~1e-6 for [-20, 20]
    let xc = if x < -80.0 {
        -80.0
    } else if x > 80.0 {
        80.0
    } else {
        x
    };
    // Range reduction: e^x = 2^k * e^r, ln(2) = 0.6931471805599453
    let ln2 = 0.6931471805599453f32;
    let inv_ln2 = 1.4426950408889634f32;
    let kf = xc * inv_ln2;
    let k = if kf >= 0.0 {
        (kf + 0.5) as i32
    } else {
        (kf - 0.5) as i32
    };
    let r = xc - (k as f32) * ln2;
    let mut term = 1.0f32;
    let mut sum = 1.0f32;
    for i in 1..10 {
        term *= r / (i as f32);
        sum += term;
    }
    let scale_bits = ((k + 127).clamp(1, 254) as u32) << 23;
    sum * f32::from_bits(scale_bits)
}

#[no_mangle]
pub unsafe extern "C" fn cpu_ref_deltanet_head_step(
    q_ptr: *const f32,
    k_ptr: *const f32,
    v_ptr: *const f32,
    z_ptr: *const f32,
    gate_a: f32,
    gate_b: f32,
    conv_ptr: *mut f32,
    state_ptr: *mut f32,
    out_ptr: *mut f32,
) {
    let q = core::slice::from_raw_parts(q_ptr, 128);
    let k = core::slice::from_raw_parts(k_ptr, 128);
    let v = core::slice::from_raw_parts(v_ptr, 128);
    let z = core::slice::from_raw_parts(z_ptr, 128);
    let conv = core::slice::from_raw_parts_mut(conv_ptr, 128 * 4);
    let state = core::slice::from_raw_parts_mut(state_ptr, 128 * 128);
    let out = core::slice::from_raw_parts_mut(out_ptr, 128);

    let decay = fast_exp_f32(-gate_a.abs() * 0.1).clamp(0.0, 1.0);
    let beta = 1.0 / (1.0 + fast_exp_f32(-gate_b));

    for j in 0..128 {
        let cb = j * 4;
        let c0 = conv[cb + 1];
        let c1 = conv[cb + 2];
        let c2 = conv[cb + 3];
        let c3 = v[j];
        conv[cb] = c0;
        conv[cb + 1] = c1;
        conv[cb + 2] = c2;
        conv[cb + 3] = c3;
        let v_conv = silu_f32(0.25 * (c0 + c1 + c2 + c3));

        let mut sk = 0.0f32;
        for i in 0..128 {
            let s = state[i * 128 + j] * decay;
            state[i * 128 + j] = s;
            sk += s * k[i];
        }
        let delta = (v_conv - sk) * beta;
        let mut o = 0.0f32;
        for i in 0..128 {
            let sn = state[i * 128 + j] + k[i] * delta;
            state[i * 128 + j] = sn;
            o += sn * q[i];
        }
        out[j] = o * silu_f32(z[j]);
    }
}
