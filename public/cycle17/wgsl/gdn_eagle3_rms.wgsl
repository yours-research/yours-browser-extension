// Input, FF and final learned RMSNorm. One workgroup reduces one hidden row.
// Striped partial sums and the fixed shared reduction match the f32 CPU oracle.
struct Params { hidden: u32, rows: u32, weight_offset: u32, reserved: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

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
// Make a binary32 intermediate observable before the next multiply. The
// exponent test also pins the subnormal convention under fast-math compilers.
fn stage(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  return bitcast<f32>(select(b, b & 0x80000000u, (b & 0x7f800000u) == 0u));
}

var<workgroup> partial: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  if (wg.x >= p.rows) { return; }
  let base = wg.x * p.hidden;
  var sum = 0.0;
  for (var d = lane; d < p.hidden; d += 64u) {
    let value = x[base+d];
    sum += stage(value*value);
  }
  partial[lane] = sum;
  workgroupBarrier();
  for (var span = 32u; span != 0u; span >>= 1u) {
    if (lane < span) { partial[lane] += partial[lane+span]; }
    workgroupBarrier();
  }
  if (lane == 0u) { partial[0] = stage(inv_sqrt(stage(partial[0] / f32(p.hidden)) + 1e-6)); }
  workgroupBarrier();
  let inv = partial[0];
  for (var d = lane; d < p.hidden; d += 64u) {
    output[base+d] = stage(x[base+d]*inv) * weights[p.weight_offset+d];
  }
}
