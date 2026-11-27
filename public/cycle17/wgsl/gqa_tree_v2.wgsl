// GQA_TREE_V2: one grouped-query attention family for scalar decode (M = 1) and
// tree verify (M > 1). Every slot's arithmetic depends only on its logical
// history (committed prefix, then its ancestor chain), never on M, on physical
// sibling slots or on where an ancestor is staged.
//
// Pass 1, entry `partial`: dispatch (q_heads, slots, max_parts).
// Pass 2, entry `combine`: dispatch (q_heads, slots, 1), after pass 1 is visible.
// TS oracle: src/kernels/gqaTreeV2.ts. Keep operation order identical.
//
// Caller contract:
//  * masks[s] = inclusive ancestor bitmask over slots (own bit s set, no higher
//    bit); slot 0 is the pending root. At most 16 slots.
//  * L(s) = prefix + popcount(masks[s]); P(L) = 1 (L <= 512), 4 (L <= 4096), else 16.
//  * max_parts >= P(L(s)) for every slot. Records of partitions >= P are not written.
//  * dim <= 256, even for fp16 K/V, a multiple of 4 for int8 K/V.
//  * q_heads is a multiple of kv_heads; capacity >= prefix + slots.
//  * rotary is even and <= dim; the table covers logical positions < max L(s).
//    Queries are raw (pre-RoPE). K/V are already roped by the append path.
struct Params {
  prefix: u32, capacity: u32, q_heads: u32, kv_heads: u32,
  dim: u32, int8: u32, base: u32, q_stride: u32,
  rotary: u32, max_parts: u32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> masks: array<u32>;
@group(0) @binding(2) var<storage,read> query: array<f32>;
@group(0) @binding(3) var<storage,read> rope: array<f32>;
@group(0) @binding(4) var<storage,read> cache: array<u32>;
// Record of (slot, qhead, part) at ((slot*q_heads + qhead)*max_parts + part)*(dim+2):
// [peak, mass, acc[0..dim)].
@group(0) @binding(5) var<storage,read_write> partials: array<f32>;
@group(0) @binding(6) var<storage,read_write> out: array<f32>;
var<workgroup> lanes: array<f32,64>;
var<workgroup> slot_mask: u32;
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
fn channel(base: u32, d: u32, scale: f32) -> f32 {
  if (p.int8 == 0u) { return unpack2x16float(cache[base+d/2u])[d%2u]; }
  return f32(extractBits(bitcast<i32>(cache[base+d/4u]), (d%4u)*8u, 8u)) * scale;
}
// Thresholds are inclusive and depend on the slot's logical history only.
fn partition_count(len: u32) -> u32 {
  if (len <= 512u) { return 1u; }
  if (len <= 4096u) { return 4u; }
  return 16u;
}

@compute @workgroup_size(64)
fn partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let slot = wg.y;
  let part = wg.z;
  // masks is read-only, so this return is uniform and precedes every barrier.
  if (part >= partition_count(p.prefix + countOneBits(masks[slot]))) { return; }
  if (lane == 0u) { slot_mask = masks[slot]; }
  let mask = workgroupUniformLoad(&slot_mask);
  let len = p.prefix + countOneBits(mask);
  let chunk = (len + partition_count(len) - 1u) / partition_count(len);
  // This partition covers logical positions [lo, hi); it may be empty.
  let lo = part * chunk;
  let hi = min(len, lo + chunk);
  let row = slot*p.q_heads + wg.x;
  let words = select(p.dim/2u, p.dim/4u + 1u, p.int8 != 0u);
  let head = wg.x / (p.q_heads/p.kv_heads);

  // Partial rotate-half RoPE at logical position len-1 for this lane's channels.
  let q_base = row*p.q_stride;
  let half_rotary = p.rotary/2u;
  var q = vec4<f32>(0.0);
  for (var d = lane; d < p.dim; d += 64u) {
    var value = query[q_base+d];
    if (d < half_rotary) {
      let angle = ((len-1u)*half_rotary + d)*2u;
      value = value*rope[angle] - query[q_base+d+half_rotary]*rope[angle+1u];
    } else if (d < p.rotary) {
      let i = d - half_rotary;
      let angle = ((len-1u)*half_rotary + i)*2u;
      value = value*rope[angle] + query[q_base+i]*rope[angle+1u];
    }
    q[d/64u] = value;
  }

  var acc = vec4<f32>(0.0);
  var peak = -3.40282346e38;
  var mass = 0.0;
  // One loop body for prefix and draft positions. t is physical; `logical` counts
  // visible positions, so t == logical throughout the committed prefix.
  var logical = min(lo, p.prefix);
  for (var t = logical; t <= p.prefix + slot; t++) {
    // Masked slots are skipped before any K load, dot product or barrier.
    if (t >= p.prefix) {
      if ((mask & (1u << (t-p.prefix))) == 0u) { continue; }
    }
    let at = logical;
    logical += 1u;
    if (at >= hi) { break; }
    if (at < lo) { continue; }
    let kb = p.base + (head*p.capacity + t)*words;
    let vb = kb + p.kv_heads*p.capacity*words;
    var scales = vec2<f32>(1.0);
    if (p.int8 != 0u) { scales = unpack2x16float(cache[kb+words-1u]); }
    var dot_product = 0.0;
    for (var d = lane; d < p.dim; d += 64u) {
      dot_product += q[d/64u] * channel(kb, d, scales.x);
    }
    lanes[lane] = dot_product;
    workgroupBarrier();
    for (var span = 32u; span != 0u; span /= 2u) {
      if (lane < span) { lanes[lane] += lanes[lane+span]; }
      workgroupBarrier();
    }
    let score = lanes[0] * inverseSqrt(f32(p.dim));
    let high = max(peak, score);
    let a = exp_neg(peak-high);
    let b = exp_neg(score-high);
    mass = mass*a + b;
    for (var d = lane; d < p.dim; d += 64u) {
      acc[d/64u] = acc[d/64u]*a + channel(vb, d, scales.y)*b;
    }
    peak = high;
    workgroupBarrier();
  }

  // An empty partition stores the initial state: [-max_f32, 0, zeros].
  let record = (row*p.max_parts + part)*(p.dim+2u);
  if (lane == 0u) {
    partials[record] = peak;
    partials[record+1u] = mass;
  }
  for (var d = lane; d < p.dim; d += 64u) { partials[record+2u+d] = acc[d/64u]; }
}

@compute @workgroup_size(64)
fn combine(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let slot = wg.y;
  let row = slot*p.q_heads + wg.x;
  let count = partition_count(p.prefix + countOneBits(masks[slot]));
  var acc = vec4<f32>(0.0);
  var peak = -3.40282346e38;
  var mass = 0.0;
  // Ascending partition order. With one partition a == 0 and b == 1, so the
  // merge copies the record exactly and the division matches GQA_TREE.
  for (var part = 0u; part < count; part++) {
    let record = (row*p.max_parts + part)*(p.dim+2u);
    let part_peak = partials[record];
    let part_mass = partials[record+1u];
    let high = max(peak, part_peak);
    let a = exp_neg(peak-high);
    let b = exp_neg(part_peak-high);
    mass = mass*a + part_mass*b;
    for (var d = lane; d < p.dim; d += 64u) {
      acc[d/64u] = acc[d/64u]*a + partials[record+2u+d]*b;
    }
    peak = high;
  }
  for (var d = lane; d < p.dim; d += 64u) { out[row*p.dim+d] = acc[d/64u]*(1.0/mass); }
}
