enable subgroups;
// Fast single-slot chain conv + fused DeltaNet (pack + recur + commit + gate) for EXP-007 (Tier 1 subgroups).
// Replaces 6 reference dispatches/step (conv, commitConv, pack, recur, gate, commitState) + 2 clearBuffers
// with 2 in-place dispatches/step and subgroupAdd reductions.
struct Params { a: u32, b: u32, c: u32, d: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src_a: array<f32>;
@group(0) @binding(2) var<storage, read> src_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> aux: array<f32>;
@group(0) @binding(6) var<storage, read> z_in: array<f32>;

var<workgroup> q_sh: array<f32, 128>;
var<workgroup> k_sh: array<f32, 128>;
var<workgroup> sg_xy: array<vec2<f32>, 32>;
var<workgroup> rec_scalars: vec2<f32>;

@compute @workgroup_size(128)
fn conv_step(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= p.a) { return; }
  let h = c * 3u;
  var taps = vec3<f32>(0.0);
  if (p.b > 0u) {
    taps = vec3<f32>(state[h], state[h + 1u], state[h + 2u]);
  }
  let x = src_a[c];
  let w = c * 4u;
  let u = fma(taps.x, src_b[w], fma(taps.y, src_b[w + 1u], fma(taps.z, src_b[w + 2u], x * src_b[w + 3u])));
  out[c] = u / (1.0 + exp(-u));
  if (p.b < p.c) {
    state[h] = taps.y;
    state[h + 1u] = taps.z;
    state[h + 2u] = x;
  }
}

@compute @workgroup_size(128)
fn dn_step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32,
           @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  let head = wg.x;
  let heads = p.a;
  let step = p.b;
  let last_step = p.c;
  let dim = p.d;
  let keys = heads * dim;
  let valid = lane < dim;

  var q_raw = 0.0;
  var k_raw = 0.0;
  var v_val = 0.0;
  if (valid) {
    let pos = head * dim + lane;
    q_raw = src_a[pos];
    k_raw = src_a[keys + pos];
    v_val = src_a[2u * keys + pos];
  }

  let qk_sg = vec2<f32>(subgroupAdd(q_raw * q_raw), subgroupAdd(k_raw * k_raw));
  if (sg_lane == 0u) { sg_xy[lane / sg_size] = qk_sg; }
  workgroupBarrier();
  let n_sg = 128u / sg_size;
  var sq = vec2<f32>(0.0);
  for (var i = 0u; i < n_sg; i++) { sq += sg_xy[i]; }

  if (valid) {
    q_sh[lane] = q_raw * (inverseSqrt(sq.x + 1e-6) * inverseSqrt(f32(dim)));
    k_sh[lane] = k_raw * inverseSqrt(sq.y + 1e-6);
  }
  if (lane == 0u) {
    let u = src_b[head + heads] + aux[heads + head];
    let sp = max(u, 0.0) + log(1.0 + exp(-abs(u)));
    let arg = aux[head] * sp;
    let decay = select(exp(arg), 0.0, arg <= -87.33654);
    let beta = 1.0 / (1.0 + exp(-src_b[head]));
    rec_scalars = vec2<f32>(decay, beta);
  }
  workgroupBarrier();

  var y_val = 0.0;
  if (valid) {
    let decay = rec_scalars.x;
    let beta = rec_scalars.y;
    let state_base = head * dim * dim + lane;
    if (step == 0u) {
      let err = v_val * beta;
      if (step < last_step) {
        for (var i = 0u; i < dim; i++) {
          let si = k_sh[i] * err;
          state[state_base + i * dim] = si;
          y_val = fma(si, q_sh[i], y_val);
        }
      } else {
        var kq = 0.0;
        for (var i = 0u; i < dim; i++) { kq = fma(k_sh[i], q_sh[i], kq); }
        y_val = err * kq;
      }
    } else {
      var s: array<f32, 128>;
      var pred = 0.0;
      for (var i = 0u; i < dim; i++) {
        let si = state[state_base + i * dim] * decay;
        s[i] = si;
        pred = fma(si, k_sh[i], pred);
      }
      let err = (v_val - pred) * beta;
      if (step < last_step) {
        for (var i = 0u; i < dim; i++) {
          let si = fma(k_sh[i], err, s[i]);
          state[state_base + i * dim] = si;
          y_val = fma(si, q_sh[i], y_val);
        }
      } else {
        for (var i = 0u; i < dim; i++) {
          let si = fma(k_sh[i], err, s[i]);
          y_val = fma(si, q_sh[i], y_val);
        }
      }
    }
  }

  let y_sg = subgroupAdd(y_val * y_val);
  if (sg_lane == 0u) { sg_xy[lane / sg_size].x = y_sg; }
  workgroupBarrier();
  var sq_y = 0.0;
  for (var i = 0u; i < n_sg; i++) { sq_y += sg_xy[i].x; }
  if (valid) {
    let pos = head * dim + lane;
    let norm_y = y_val * inverseSqrt(sq_y / f32(dim) + 1e-6);
    let zv = z_in[pos];
    let silu_z = zv / (1.0 + exp(-zv));
    out[pos] = norm_y * silu_z * aux[2u * heads + lane];
  }
}
