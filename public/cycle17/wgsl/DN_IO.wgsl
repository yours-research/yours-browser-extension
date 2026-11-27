// Layout of raw tensors and auxiliary arrays matches the browser GDN layer.
// pack and gate both dispatch (value_heads, slots, 1), with dk = dv = dim.
struct Params { heads: u32, key_heads: u32, dim: u32, slots: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> qkv: array<f32>; // post-convolution
@group(0) @binding(2) var<storage,read> ab: array<f32>; // [slot, beta_raw | rate_raw]
@group(0) @binding(3) var<storage,read> aux: array<f32>; // negative exp(A_log), dt_bias, RMS weights
@group(0) @binding(4) var<storage,read_write> rec: array<f32>; // input to DN_TREE
@group(0) @binding(5) var<storage,read> y: array<f32>; // DN_TREE.verify output
@group(0) @binding(6) var<storage,read> z: array<f32>;
@group(0) @binding(7) var<storage,read_write> out: array<f32>;
var<workgroup> norms: array<vec2<f32>,128>;
// Range-reduced exp for nonpositive inputs. Explicit stages keep the CPU oracle
// and the decode/verify paths on the same polynomial. Flush tiny tails to zero.
fn exp_neg(x: f32) -> f32 {
  if (x <= -87.33654) { return 0.0; }
  let n = i32(round(x * 1.4426950408889634));
  let r = (x - f32(n) * 0.693359375) - f32(n) * -0.00021219444005469058;
  var y = r * 0.0013888889 + 0.0083333338;
  y = r * y + 0.0416666679;
  y = r * y + 0.1666666716;
  y = r * y + 0.5;
  y = r * y + 1.0;
  y = r * y + 1.0;
  return bitcast<f32>(u32(n+127)<<23u)*y;
}
// Restoring square root of a 48-bit significand. Norms include epsilon, so x
// is positive and normal. Round sqrt to f32 before taking its reciprocal.
fn inv_sqrt(x: f32) -> f32 {
  let bits = bitcast<u32>(x);
  let exponent = i32(bits >> 23u)-127;
  let significand = ((bits & 0x7fffffu) | 0x800000u) << (u32(exponent) & 1u);
  let high = significand >> 9u;
  let low = significand << 23u;
  var root = 0u;
  var rem = 0u;
  for (var pair = 23i; pair >= 0; pair--) {
    let shift = 2u*u32(pair);
    let digit = (select(low,high,shift >= 32u) >> (shift & 31u)) & 3u;
    rem = (rem << 2u) | digit;
    let trial = (root << 2u) + 1u;
    root <<= 1u;
    if (rem >= trial) { rem -= trial; root++; }
  }
  if (rem > root) { root++; }
  let result = (u32((exponent >> 1u)+127) << 23u) + (root-0x800000u);
  return 1.0 / bitcast<f32>(result);
}
// log(1+x) for x in [0,1] via 2*atanh(x/(2+x)).
fn log_one_plus(x: f32) -> f32 {
  let v = x/(2.0+x);
  let square = v*v;
  var poly = 1.0/17.0;
  for (var i = 15; i >= 1; i -= 2) { poly = (1.0/f32(i)) + square*poly; }
  return (2.0*v)*poly;
}
// Make a binary32 intermediate observable before the next multiply. The
// exponent test also pins the subnormal convention under fast-math compilers.
fn stage(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  return bitcast<f32>(select(b, b & 0x80000000u, (b & 0x7f800000u) == 0u));
}
fn sigmoid(x: f32) -> f32 {
  let e = exp_neg(-abs(x));
  return select(1.0,e,x < 0.0) * (1.0/(1.0+e));
}
fn sum_norms(lane: u32, x: vec2<f32>) -> vec2<f32> {
  norms[lane] = x*x;
  workgroupBarrier();
  for (var span = 64u; span > 0u; span /= 2u) {
    if (lane < span) { norms[lane] += norms[lane+span]; }
    workgroupBarrier();
  }
  return norms[0];
}
@compute @workgroup_size(128)
fn pack(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let keys = p.key_heads*p.dim;
  let base = wg.y*(2u*keys+p.heads*p.dim);
  let pos = base+(wg.x/(p.heads/p.key_heads))*p.dim+lane;
  var x = vec2<f32>(0.0);
  if (lane < p.dim) { x = vec2<f32>(qkv[pos],qkv[pos+keys]); }
  let sq = sum_norms(lane,x);
  let dst = (wg.y*p.heads+wg.x)*(3u*p.dim+2u);
  if (lane < p.dim) {
    let q_scale = stage(stage(inv_sqrt(sq.x+1e-6))*stage(inv_sqrt(f32(p.dim))));
    rec[dst+lane] = x.x * q_scale;
    rec[dst+p.dim+lane] = x.y * stage(inv_sqrt(sq.y+1e-6));
    rec[dst+2u*p.dim+lane] = qkv[base+2u*keys+wg.x*p.dim+lane];
  }
  if (lane == 0u) {
    let b = wg.y*2u*p.heads+wg.x;
    let u = ab[b+p.heads]+aux[p.heads+wg.x];
    let softplus = max(u,0.0)+log_one_plus(exp_neg(-abs(u)));
    rec[dst+3u*p.dim] = exp_neg(aux[wg.x]*softplus);
    rec[dst+3u*p.dim+1u] = sigmoid(ab[b]);
  }
}
@compute @workgroup_size(128)
fn gate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let pos = (wg.y*p.heads+wg.x)*p.dim+lane;
  var x = 0.0;
  if (lane < p.dim) { x = y[pos]; }
  let sq = sum_norms(lane,vec2<f32>(x,0.0));
  if (lane < p.dim) {
    let normalized = x * stage(inv_sqrt(sq.x/f32(p.dim)+1e-6));
    let silu = stage(z[pos]*stage(sigmoid(z[pos])));
    out[pos] = stage(stage(normalized)*silu) * aux[2u*p.heads+lane];
  }
}
