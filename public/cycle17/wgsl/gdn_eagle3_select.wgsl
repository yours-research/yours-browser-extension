// On-GPU greedy choice for the single-submit EXP-007 chain. One workgroup
// reduces the subset logits to the argmax (ties go to the lowest subset
// index, as in TOPK_BATCH) and the top-1 softmax probability
// 1 / sum_i exp(l_i - max). It writes the chosen global token id to
// slots[p.slot + 1] for the next step's embedding, and an 8-byte readback
// record (global id, f32 probability) at slots[p.records + 2 * p.slot].
struct Params { subset: u32, slot: u32, records: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;
var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;
var<workgroup> sums: array<f32, 256>;
const END = 0xffffffffu;

// Does (v, i) beat (w, j)?
fn better(v: f32, i: u32, w: f32, j: u32) -> bool {
  if (i == END) { return false; }
  if (j == END) { return true; }
  return v > w || (v == w && i < j);
}

@compute @workgroup_size(256)
fn select_top1(@builtin(local_invocation_index) lane: u32) {
  var bv = 0.0;
  var bi = END;
  for (var i = lane; i < p.subset; i += 256u) {
    let v = logits[i];
    if (better(v, i, bv, bi)) { bv = v; bi = i; }
  }
  best_v[lane] = bv;
  best_i[lane] = bi;
  workgroupBarrier();
  for (var span = 128u; span > 0u; span /= 2u) {
    if (lane < span && better(best_v[lane+span], best_i[lane+span], best_v[lane], best_i[lane])) {
      best_v[lane] = best_v[lane+span];
      best_i[lane] = best_i[lane+span];
    }
    workgroupBarrier();
  }
  let top = best_v[0];
  var s = 0.0;
  for (var i = lane; i < p.subset; i += 256u) { s += exp(logits[i] - top); }
  sums[lane] = s;
  workgroupBarrier();
  for (var span = 128u; span > 0u; span /= 2u) {
    if (lane < span) { sums[lane] += sums[lane+span]; }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let id = ids[best_i[0]];
    slots[p.slot + 1u] = id;
    slots[p.records + 2u*p.slot] = id;
    slots[p.records + 2u*p.slot + 1u] = bitcast<u32>(1.0 / sums[0]);
  }
}
