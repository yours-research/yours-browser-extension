// Depthwise conv4 and SiLU. Committed history is [channel, oldest..newest].
struct Params { channels: u32, slots: u32, reserved0: u32, reserved1: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> masks: array<u32>;
@group(0) @binding(2) var<storage,read> raw: array<f32>;
@group(0) @binding(3) var<storage,read> weights: array<f32>;
@group(0) @binding(4) var<storage,read_write> history: array<f32>;
@group(0) @binding(5) var<storage,read_write> out: array<f32>;
@group(0) @binding(6) var<storage,read> choice: array<u32>;
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
fn scan(c: u32, leaf: u32, committing: bool) {
  if (c >= p.channels) { return; }
  let h = c * 3u;
  var taps = vec3<f32>(history[h], history[h+1u], history[h+2u]);
  for (var slot = 0u; slot <= leaf; slot++) {
    if ((masks[leaf] & (1u << slot)) == 0u) { continue; }
    let x = raw[slot*p.channels+c];
    if (slot == leaf && !committing) {
      let w = c * 4u;
      let u = taps.x*weights[w] + taps.y*weights[w+1u] + taps.z*weights[w+2u] + x*weights[w+3u];
      let e = exp_neg(-abs(u));
      out[slot*p.channels+c] = select(u, u*e, u < 0.0) * (1.0 / (1.0+e));
    }
    taps = vec3<f32>(taps.y, taps.z, x);
  }
  if (committing) { history[h] = taps.x; history[h+1u] = taps.y; history[h+2u] = taps.z; }
}
@compute @workgroup_size(64)
fn verify(@builtin(global_invocation_id) id: vec3<u32>) { scan(id.x, id.y, false); }
@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>) { scan(id.x, choice[2u+choice[0]], true); }
