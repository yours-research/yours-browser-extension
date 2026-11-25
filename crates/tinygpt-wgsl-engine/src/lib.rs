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

const SHADER_BUF_CAP: usize = 32768;
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

// Uses 1 workgroup of 32 threads (1 warp on NVIDIA) per 4 output rows (register-held f16 activations + subgroupAdd).
// Works across subgroup_size <= 32 without relying on @builtin(subgroup_id).
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

var<workgroup> sg_tmp: array<vec4<f32>, 32>;

fn dot_q4_word(p: u32, s4: vec4<f16>, z4: vec4<f16>, xa: vec4<f16>, xb: vec4<f16>) -> f16 {
  let q0 = vec4<f16>(f16(p & 0xFu), f16((p >> 4u) & 0xFu), f16((p >> 8u) & 0xFu), f16((p >> 12u) & 0xFu));
  let q1 = vec4<f16>(f16((p >> 16u) & 0xFu), f16((p >> 20u) & 0xFu), f16((p >> 24u) & 0xFu), f16((p >> 28u) & 0xFu));
  return dot(fma(q0, s4, z4), xa) + dot(fma(q1, s4, z4), xb);
}

fn dot_q4_block(
  w0: u32, w1: u32, w2: u32, w3: u32, meta_word: u32,
  x0a: vec4<f16>, x0b: vec4<f16>,
  x1a: vec4<f16>, x1b: vec4<f16>,
  x2a: vec4<f16>, x2b: vec4<f16>,
  x3a: vec4<f16>, x3b: vec4<f16>,
) -> f32 {
  let blk_meta = bitcast<vec2<f16>>(meta_word);
  let s4 = vec4<f16>(blk_meta.x);
  let z4 = vec4<f16>(blk_meta.y);
  let s = dot_q4_word(w0, s4, z4, x0a, x0b)
    + dot_q4_word(w1, s4, z4, x1a, x1b)
    + dot_q4_word(w2, s4, z4, x2a, x2b)
    + dot_q4_word(w3, s4, z4, x3a, x3b);
  return f32(s);
}

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let row_base_idx = wid.x * 4u;
  let tid = lid.x;
  let blocks = params.cols >> 5u;
  let row_stride = blocks * 5u;

  var acc = vec4<f32>(0.0);
  let r0 = row_base_idx;
  let r1 = min(row_base_idx + 1u, params.rows - 1u);
  let r2 = min(row_base_idx + 2u, params.rows - 1u);
  let r3 = min(row_base_idx + 3u, params.rows - 1u);

  let o0 = params.w_u32_offset + r0 * row_stride;
  let o1 = params.w_u32_offset + r1 * row_stride;
  let o2 = params.w_u32_offset + r2 * row_stride;
  let o3 = params.w_u32_offset + r3 * row_stride;

  for (var b: u32 = tid; b < blocks; b = b + 32u) {
    let kb = params.in_offset + (b << 5u);
    let b5 = b * 5u;

    // Load 32 activations once into f16 register vectors and reuse across all 4 rows
    let x0a = vec4<f16>(f16(x[kb]), f16(x[kb + 1u]), f16(x[kb + 2u]), f16(x[kb + 3u]));
    let x0b = vec4<f16>(f16(x[kb + 4u]), f16(x[kb + 5u]), f16(x[kb + 6u]), f16(x[kb + 7u]));
    let x1a = vec4<f16>(f16(x[kb + 8u]), f16(x[kb + 9u]), f16(x[kb + 10u]), f16(x[kb + 11u]));
    let x1b = vec4<f16>(f16(x[kb + 12u]), f16(x[kb + 13u]), f16(x[kb + 14u]), f16(x[kb + 15u]));
    let x2a = vec4<f16>(f16(x[kb + 16u]), f16(x[kb + 17u]), f16(x[kb + 18u]), f16(x[kb + 19u]));
    let x2b = vec4<f16>(f16(x[kb + 20u]), f16(x[kb + 21u]), f16(x[kb + 22u]), f16(x[kb + 23u]));
    let x3a = vec4<f16>(f16(x[kb + 24u]), f16(x[kb + 25u]), f16(x[kb + 26u]), f16(x[kb + 27u]));
    let x3b = vec4<f16>(f16(x[kb + 28u]), f16(x[kb + 29u]), f16(x[kb + 30u]), f16(x[kb + 31u]));

    let i0 = o0 + b5;
    let i1 = o1 + b5;
    let i2 = o2 + b5;
    let i3 = o3 + b5;

    acc.x += dot_q4_block(weights[i0], weights[i0 + 1u], weights[i0 + 2u], weights[i0 + 3u], weights[i0 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b);
    acc.y += dot_q4_block(weights[i1], weights[i1 + 1u], weights[i1 + 2u], weights[i1 + 3u], weights[i1 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b);
    acc.z += dot_q4_block(weights[i2], weights[i2 + 1u], weights[i2 + 2u], weights[i2 + 3u], weights[i2 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b);
    acc.w += dot_q4_block(weights[i3], weights[i3 + 1u], weights[i3 + 2u], weights[i3 + 3u], weights[i3 + 4u], x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b);
  }

  let sg_sum = subgroupAdd(acc);
  sg_tmp[tid] = select(vec4<f32>(0.0), sg_sum, sg_inv == 0u);
  workgroupBarrier();

  if (tid == 0u) {
    var total = vec4<f32>(0.0);
    let step = max(sg_size, 1u);
    for (var s: u32 = 0u; s < 32u; s = s + step) {
      total += sg_tmp[s];
    }
    if (r0 < params.rows) {
      let out0 = params.out_offset + r0;
      y[out0] = select(total.x, y[out0] + total.x, params.add_residual != 0u);
    }
    if (row_base_idx + 1u < params.rows) {
      let out1 = params.out_offset + row_base_idx + 1u;
      y[out1] = select(total.y, y[out1] + total.y, params.add_residual != 0u);
    }
    if (row_base_idx + 2u < params.rows) {
      let out2 = params.out_offset + row_base_idx + 2u;
      y[out2] = select(total.z, y[out2] + total.z, params.add_residual != 0u);
    }
    if (row_base_idx + 3u < params.rows) {
      let out3 = params.out_offset + row_base_idx + 3u;
      y[out3] = select(total.w, y[out3] + total.w, params.add_residual != 0u);
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

var<workgroup> red_buf: array<f32, 128>;
var<workgroup> cur_buf: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let q_head = wid.x;       // 0..15
  let split_id = wid.y;     // 0..num_splits-1
  let kv_head = q_head >> 2u; // 4:1 GQA -> 0..3
  let tid = lid.x;          // 0..63 -> 4 dims per thread (2 u32s)
  let d2 = tid * 2u;
  let d = tid * 4u;

  let head_u32_stride = params.max_seq * 128u; // 256 f16 = 128 u32 per token
  let kv_plane_u32s = 4u * head_u32_stride;
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

  let chunk = (params.seq_pos + params.num_splits - 1u) / params.num_splits;
  let t_start = min(split_id * chunk, params.seq_pos);
  let t_end = min(t_start + chunk, params.seq_pos);

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

    if ((q_head & 3u) == 0u) {
      let k_dst = k_head_base + params.seq_pos * 128u + d2;
      let v_dst = v_head_base + params.seq_pos * 128u + d2;
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

  let out_base = (q_head * params.num_splits + split_id) * 260u;
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

// Static INT8 Quantized KV Cache Split-K FlashDecoding GQA (16 Q heads, 4 KV heads, head_dim = 256)
// 64 u32s per token-head + 1 packed scale u32 = 65 u32s
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

var<workgroup> red_buf: array<f32, 128>;
var<workgroup> cur_kv_max: array<vec2<f32>, 64>;
var<workgroup> cur_dot_buf: array<f32, 64>;

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
  let q_head = wid.x;
  let split_id = wid.y;
  let kv_head = q_head >> 2u;
  let tid = lid.x; // 0..63 -> 1 packed u32 (4 int8 dims) per thread
  let d = tid * 4u;

  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = 4u * head_u32_stride;
  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q_stride = select(params.q_head_stride, 256u, params.q_head_stride == 0u);
  let q_base = params.q_offset + q_head * q_stride + d;
  let q_vec = vec4<f32>(act[q_base], act[q_base + 1u], act[q_base + 2u], act[q_base + 3u]);
  let inv_sqrt_d: f32 = 0.0625;
  let sg_step = max(sg_size, 1u);

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  var acc_v = vec4<f32>(0.0);

  let chunk = (params.seq_pos + params.num_splits - 1u) / params.num_splits;
  let t_start = min(split_id * chunk, params.seq_pos);
  let t_end = min(t_start + chunk, params.seq_pos);

  for (var t: u32 = t_start; t < t_end; t = t + 1u) {
    let k_src = k_head_base + t * 65u;
    let scales = unpack2x16float(kv_cache[k_src + 64u]);
    let ki = unpack4_i8(kv_cache[k_src + tid]) * scales.x;

    let slot = ((t - t_start) & 1u) * 64u;
    let sg_dot = subgroupAdd(dot(q_vec, ki));
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

    let v_src = v_head_base + t * 65u;
    let vi = unpack4_i8(kv_cache[v_src + tid]) * scales.y;
    acc_v = acc_v * alpha + vi * beta;
    m_max = m_new;
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

    if ((q_head & 3u) == 0u) {
      let k_dst = k_head_base + params.seq_pos * 65u;
      let v_dst = v_head_base + params.seq_pos * 65u;
      kv_cache[k_dst + tid] = qk;
      kv_cache[v_dst + tid] = qv;
      if (tid == 0u) {
        kv_cache[k_dst + 64u] = packed_scales;
      }
    }

    let cur_sg = subgroupAdd(dot(q_vec, unpack4_i8(qk) * scales.x));
    cur_dot_buf[tid] = select(0.0, cur_sg, sg_inv == 0u);
    workgroupBarrier();

    var cur_dot: f32 = 0.0;
    for (var s: u32 = 0u; s < 64u; s = s + sg_step) {
      cur_dot += cur_dot_buf[s];
    }

    let score = cur_dot * inv_sqrt_d;
    let m_new = max(m_max, score);
    let alpha = exp(m_max - m_new);
    let beta = exp(score - m_new);
    l_sum = l_sum * alpha + beta;
    acc_v = acc_v * alpha + (unpack4_i8(qv) * scales.y) * beta;
    m_max = m_new;
  }

  let out_base = (q_head * params.num_splits + split_id) * 260u;
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
  let head_slot = tok * 16u + q_head;
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
  let g_base = params.gate_offset + tok * (16u * g_stride) + q_head * g_stride + d;
  let out_base = params.out_offset + tok * 4096u + q_head * 256u + d;
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
  let blocks = params.cols >> 5u;
  let bi = (tok * blocks + b) * 5u;
  let blk_meta = unpack2x16float(weights[bi + 4u]);
  let word = weights[bi + (d >> 3u)];
  let q = f32((word >> ((d & 7u) * 4u)) & 0xFu);
  out[tok_idx * params.cols + (b << 5u) + d] = fma(q, blk_meta.x, blk_meta.y);
}
"#;

// Batched RMSNorm [num_tokens, len] reading gamma from aux[gamma_offset..]
const WGSL_RMSNORM_BATCH: &str = r#"
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

// Batched Q4-f16 GEMM [M, K] x [N, K]^T -> [M, N] via 32x32x32 shared-memory tiling
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

var<workgroup> x_sh: array<vec4<f16>, 256>;
var<workgroup> w_sh: array<vec4<f16>, 256>;

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

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row_tile = wid.x * 32u;
  let tok_tile = wid.y * 32u;
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * 8u + lx;
  let blocks = params.K >> 5u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  let loc_idx = tid >> 1u;
  let half = tid & 1u;
  let sh_base = loc_idx * 8u + half * 4u;

  for (var b: u32 = 0u; b < blocks; b = b + 1u) {
    let gt = tok_tile + loc_idx;
    if (gt < params.M) {
      let xb = params.in_offset + gt * params.K + (b << 5u) + (half << 4u);
      x_sh[sh_base + 0u] = vec4<f16>(f16(x[xb + 0u]), f16(x[xb + 1u]), f16(x[xb + 2u]), f16(x[xb + 3u]));
      x_sh[sh_base + 1u] = vec4<f16>(f16(x[xb + 4u]), f16(x[xb + 5u]), f16(x[xb + 6u]), f16(x[xb + 7u]));
      x_sh[sh_base + 2u] = vec4<f16>(f16(x[xb + 8u]), f16(x[xb + 9u]), f16(x[xb + 10u]), f16(x[xb + 11u]));
      x_sh[sh_base + 3u] = vec4<f16>(f16(x[xb + 12u]), f16(x[xb + 13u]), f16(x[xb + 14u]), f16(x[xb + 15u]));
    } else {
      x_sh[sh_base + 0u] = vec4<f16>(0.0h);
      x_sh[sh_base + 1u] = vec4<f16>(0.0h);
      x_sh[sh_base + 2u] = vec4<f16>(0.0h);
      x_sh[sh_base + 3u] = vec4<f16>(0.0h);
    }

    let gr = row_tile + loc_idx;
    if (gr < params.N) {
      let bi = params.w_u32_offset + gr * blocks * 5u + b * 5u;
      let blk_meta = bitcast<vec2<f16>>(weights[bi + 4u]);
      let s4 = vec4<f16>(blk_meta.x);
      let z4 = vec4<f16>(blk_meta.y);
      let p0 = weights[bi + half * 2u];
      let p1 = weights[bi + half * 2u + 1u];
      w_sh[sh_base + 0u] = unpack_nibble_vec(p0, 0u, s4, z4);
      w_sh[sh_base + 1u] = unpack_nibble_vec(p0, 16u, s4, z4);
      w_sh[sh_base + 2u] = unpack_nibble_vec(p1, 0u, s4, z4);
      w_sh[sh_base + 3u] = unpack_nibble_vec(p1, 16u, s4, z4);
    } else {
      w_sh[sh_base + 0u] = vec4<f16>(0.0h);
      w_sh[sh_base + 1u] = vec4<f16>(0.0h);
      w_sh[sh_base + 2u] = vec4<f16>(0.0h);
      w_sh[sh_base + 3u] = vec4<f16>(0.0h);
    }
    workgroupBarrier();

    let r0 = ly << 2u;
    let t0 = lx << 2u;
    for (var k4: u32 = 0u; k4 < 8u; k4 = k4 + 1u) {
      let wv0 = w_sh[(r0 + 0u) * 8u + k4];
      let wv1 = w_sh[(r0 + 1u) * 8u + k4];
      let wv2 = w_sh[(r0 + 2u) * 8u + k4];
      let wv3 = w_sh[(r0 + 3u) * 8u + k4];
      let xv0 = x_sh[(t0 + 0u) * 8u + k4];
      let xv1 = x_sh[(t0 + 1u) * 8u + k4];
      let xv2 = x_sh[(t0 + 2u) * 8u + k4];
      let xv3 = x_sh[(t0 + 3u) * 8u + k4];
      acc0 += vec4<f32>(f32(dot(wv0, xv0)), f32(dot(wv0, xv1)), f32(dot(wv0, xv2)), f32(dot(wv0, xv3)));
      acc1 += vec4<f32>(f32(dot(wv1, xv0)), f32(dot(wv1, xv1)), f32(dot(wv1, xv2)), f32(dot(wv1, xv3)));
      acc2 += vec4<f32>(f32(dot(wv2, xv0)), f32(dot(wv2, xv1)), f32(dot(wv2, xv2)), f32(dot(wv2, xv3)));
      acc3 += vec4<f32>(f32(dot(wv3, xv0)), f32(dot(wv3, xv1)), f32(dot(wv3, xv2)), f32(dot(wv3, xv3)));
    }
    workgroupBarrier();
  }

  let out_t = tok_tile + (lx << 2u);
  let out_r = row_tile + (ly << 2u);
  store_cell(out_t + 0u, out_r + 0u, acc0.x);
  store_cell(out_t + 1u, out_r + 0u, acc0.y);
  store_cell(out_t + 2u, out_r + 0u, acc0.z);
  store_cell(out_t + 3u, out_r + 0u, acc0.w);

  store_cell(out_t + 0u, out_r + 1u, acc1.x);
  store_cell(out_t + 1u, out_r + 1u, acc1.y);
  store_cell(out_t + 2u, out_r + 1u, acc1.z);
  store_cell(out_t + 3u, out_r + 1u, acc1.w);

  store_cell(out_t + 0u, out_r + 2u, acc2.x);
  store_cell(out_t + 1u, out_r + 2u, acc2.y);
  store_cell(out_t + 2u, out_r + 2u, acc2.z);
  store_cell(out_t + 3u, out_r + 2u, acc2.w);

  store_cell(out_t + 0u, out_r + 3u, acc3.x);
  store_cell(out_t + 1u, out_r + 3u, acc3.y);
  store_cell(out_t + 2u, out_r + 3u, acc3.z);
  store_cell(out_t + 3u, out_r + 3u, acc3.w);
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
  let c = gid.x;
  if (c >= 8192u) {
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
    let idx = params.qkv_offset + t * 8192u + c;
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
  let v_head = wid.x; // 0..31
  let j = lid.x;      // 0..127
  let k_head = v_head >> 1u; // 16 Q/K heads repeated 2:1 across 32 value heads

  let a_neg = aux[params.aux_dn_offset + v_head];
  let dt_b = aux[params.aux_dn_offset + 32u + v_head];
  let gdn_norm_w = aux[params.aux_dn_offset + 64u + j];
  let s_base = params.state_f32_offset + v_head * (128u * 128u);

  for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {
    let qkv_t = params.qkv_offset + t * 8192u;
    let q_raw = work[qkv_t + k_head * 128u + j];
    let k_raw = work[qkv_t + 2048u + k_head * 128u + j];

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

    let v_val = work[qkv_t + 4096u + v_head * 128u + j];
    let ab_t = params.ab_offset + t * 64u;
    let b_raw = work[ab_t + v_head];
    let a_raw = work[ab_t + 32u + v_head];
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
    let z_val = work[params.z_offset + t * 4096u + v_head * 128u + j];
    let z_silu = z_val / (1.0 + exp(-z_val));
    attn_out[params.out_offset + t * 4096u + v_head * 128u + j] = (out_val * inv_rms) * z_silu * gdn_norm_w;
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

var<workgroup> sq_buf: array<f32, 256>;
var<workgroup> sh_head: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h_idx = wid.x; // 0..15 Q heads, 16..19 K heads
  let tok = wid.y;
  let d = lid.x;

  let is_q = h_idx < 16u;
  let addr = select(
    params.k_offset + tok * 1024u + (h_idx - 16u) * 256u + d,
    params.qg_offset + tok * 8192u + h_idx * 512u + d,
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

  let pos = f32(params.start_pos + tok);
  var out_val = sh_head[d];
  if (d < 32u) {
    let ang = pos * aux[params.inv_freq_offset + d];
    out_val = sh_head[d] * cos(ang) - sh_head[d + 32u] * sin(ang);
  } else if (d < 64u) {
    let ang = pos * aux[params.inv_freq_offset + (d - 32u)];
    out_val = sh_head[d] * cos(ang) + sh_head[d - 32u] * sin(ang);
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
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * 16u + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * 8192u + q_head * 512u + d;
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
  let head_u32_stride = params.max_seq * 128u;
  let kv_plane_u32s = 4u * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x; // 0..3
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * 1024u + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * 1024u + kv_head * 256u + d;
    let pos = params.start_pos + tok;
    let k_dst = params.layer_kv_u32_offset + kv_head * head_u32_stride + pos * 128u + d2;
    let v_dst = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride + pos * 128u + d2;
    kv_cache[k_dst] = bitcast<u32>(vec2<f16>(f16(work[k_in]), f16(work[k_in + 1u])));
    kv_cache[k_dst + 1u] = bitcast<u32>(vec2<f16>(f16(work[k_in + 2u]), f16(work[k_in + 3u])));
    kv_cache[v_dst] = bitcast<u32>(vec2<f16>(f16(work[v_in]), f16(work[v_in + 1u])));
    kv_cache[v_dst + 1u] = bitcast<u32>(vec2<f16>(f16(work[v_in + 2u]), f16(work[v_in + 3u])));
    return;
  }

  let kv_head = wid.x; // 0..3
  let qh0 = kv_head * 4u;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, d);
  let q1 = load_q_vec4(tok0, qh0 + 1u, d);
  let q2 = load_q_vec4(tok0, qh0 + 2u, d);
  let q3 = load_q_vec4(tok0, qh0 + 3u, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + 1u, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + 2u, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + 3u, d);

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

  write_split_out(tok0, qh0 + 0u, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  write_split_out(tok0, qh0 + 1u, split_id, num_splits, d, tid, acc1, mA.y, lA.y);
  write_split_out(tok0, qh0 + 2u, split_id, num_splits, d, tid, acc2, mA.z, lA.z);
  write_split_out(tok0, qh0 + 3u, split_id, num_splits, d, tid, acc3, mA.w, lA.w);
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    write_split_out(tok1, qh0 + 1u, split_id, num_splits, d, tid, acc5, mB.y, lB.y);
    write_split_out(tok1, qh0 + 2u, split_id, num_splits, d, tid, acc6, mB.z, lB.z);
    write_split_out(tok1, qh0 + 3u, split_id, num_splits, d, tid, acc7, mB.w, lB.w);
  }
}
"#;

// Batched INT8 KV Split-K Prefill: mode 0 writes quantized K/V (4, M, 1), mode 1 runs 8-query GQA-tiled FlashPrefill (4, ceil(M/2), num_splits)
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
  split_id: u32,
  num_splits: u32,
  d: u32,
  tid: u32,
  acc: vec4<f32>,
  m_val: f32,
  l_val: f32,
) {
  let out_base = ((tok * 16u + q_head) * num_splits + split_id) * 260u;
  split_out[out_base + d + 0u] = acc.x;
  split_out[out_base + d + 1u] = acc.y;
  split_out[out_base + d + 2u] = acc.z;
  split_out[out_base + d + 3u] = acc.w;
  if (tid == 0u) {
    split_out[out_base + 256u] = m_val;
    split_out[out_base + 257u] = l_val;
  }
}

fn load_q_vec4(tok: u32, q_head: u32, d: u32) -> vec4<f32> {
  let b = params.buf_a_offset + tok * 8192u + q_head * 512u + d;
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
  let head_u32_stride = params.max_seq * 65u;
  let kv_plane_u32s = 4u * head_u32_stride;

  if (params.mode == 0u) {
    let kv_head = wid.x; // 0..3
    let tok = wid.y;
    let k_in = params.buf_a_offset + tok * 1024u + kv_head * 256u + d;
    let v_in = params.buf_b_offset + tok * 1024u + kv_head * 256u + d;
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

  let kv_head = wid.x; // 0..3
  let qh0 = kv_head * 4u;
  let tok0 = wid.y * 2u;
  let tok1 = tok0 + 1u;
  let has_t1 = tok1 < params.num_tokens;
  let tok1_safe = select(tok0, tok1, has_t1);
  let split_id = wid.z;
  let num_splits = max(params.num_splits, 1u);

  let k_head_base = params.layer_kv_u32_offset + kv_head * head_u32_stride;
  let v_head_base = params.layer_kv_u32_offset + kv_plane_u32s + kv_head * head_u32_stride;

  let q0 = load_q_vec4(tok0, qh0 + 0u, d);
  let q1 = load_q_vec4(tok0, qh0 + 1u, d);
  let q2 = load_q_vec4(tok0, qh0 + 2u, d);
  let q3 = load_q_vec4(tok0, qh0 + 3u, d);

  let q4 = load_q_vec4(tok1_safe, qh0 + 0u, d);
  let q5 = load_q_vec4(tok1_safe, qh0 + 1u, d);
  let q6 = load_q_vec4(tok1_safe, qh0 + 2u, d);
  let q7 = load_q_vec4(tok1_safe, qh0 + 3u, d);

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

  write_split_out(tok0, qh0 + 0u, split_id, num_splits, d, tid, acc0, mA.x, lA.x);
  write_split_out(tok0, qh0 + 1u, split_id, num_splits, d, tid, acc1, mA.y, lA.y);
  write_split_out(tok0, qh0 + 2u, split_id, num_splits, d, tid, acc2, mA.z, lA.z);
  write_split_out(tok0, qh0 + 3u, split_id, num_splits, d, tid, acc3, mA.w, lA.w);
  if (has_t1) {
    write_split_out(tok1, qh0 + 0u, split_id, num_splits, d, tid, acc4, mB.x, lB.x);
    write_split_out(tok1, qh0 + 1u, split_id, num_splits, d, tid, acc5, mB.y, lB.y);
    write_split_out(tok1, qh0 + 2u, split_id, num_splits, d, tid, acc6, mB.z, lB.z);
    write_split_out(tok1, qh0 + 3u, split_id, num_splits, d, tid, acc7, mB.w, lB.w);
  }
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
