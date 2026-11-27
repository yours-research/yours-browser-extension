enable subgroups;
// Fast RMSNorm for the EXP-007 draft head (Tier 1 subgroups): native hardware inverseSqrt,
// 256-lane coalesced stride, subgroupAdd reduction (1 barrier instead of 8).
struct Params { hidden: u32, rows: u32, weight_offset: u32, reserved: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> sg_buf: array<f32, 64>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32, @builtin(subgroup_size) sg_size: u32) {
  if (wg.x >= p.rows) { return; }
  let base = wg.x * p.hidden;
  var sum = 0.0;
  for (var d = lane; d < p.hidden; d += 256u) {
    let v = x[base + d];
    sum = fma(v, v, sum);
  }
  let s = subgroupAdd(sum);
  if (sg_lane == 0u) { sg_buf[lane / sg_size] = s; }
  workgroupBarrier();
  let n_sg = 256u / sg_size;
  var total = 0.0;
  for (var i = 0u; i < n_sg; i++) { total += sg_buf[i]; }
  let inv = inverseSqrt(total / f32(p.hidden) + 1e-6);
  for (var d = lane; d < p.hidden; d += 256u) {
    output[base + d] = (x[base + d] * inv) * weights[p.weight_offset + d];
  }
}
