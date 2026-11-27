// Fast RMSNorm for the EXP-007 draft head (Tier 2 universal): native hardware inverseSqrt,
// 256-lane coalesced stride, no software bit-by-bit sqrt loop or bitcast fences.
struct Params { hidden: u32, rows: u32, weight_offset: u32, reserved: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  if (wg.x >= p.rows) { return; }
  let base = wg.x * p.hidden;
  var sum = 0.0;
  for (var d = lane; d < p.hidden; d += 256u) {
    let v = x[base + d];
    sum = fma(v, v, sum);
  }
  partial[lane] = sum;
  workgroupBarrier();
  for (var span = 128u; span > 0u; span >>= 1u) {
    if (lane < span) { partial[lane] += partial[lane + span]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.hidden) + 1e-6);
  for (var d = lane; d < p.hidden; d += 256u) {
    output[base + d] = (x[base + d] * inv) * weights[p.weight_offset + d];
  }
}
