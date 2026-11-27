// One workgroup per logit row. Each lane keeps a sorted list for its stripe.
// Merge stripe heads k times. Ties go to the lowest token id.
struct Params { vocab: u32, stride: u32, k: u32, offset: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> logits: array<f32>;
@group(0) @binding(2) var<storage,read_write> output: array<u32>;
var<workgroup> best: array<u32,64>;
const END = 0xffffffffu;
fn better(row: u32, a: u32, b: u32) -> bool {
  if (a == END) { return false; }
  if (b == END) { return true; }
  let x = logits[row+a];
  let y = logits[row+b];
  return x > y || (x == y && a < b);
}
fn winner(row: u32, id: u32, lane: u32) -> u32 {
  best[lane] = id;
  workgroupBarrier();
  for (var span = 32u; span > 0u; span /= 2u) {
    if (lane < span && better(row, best[lane+span], best[lane])) { best[lane] = best[lane+span]; }
    workgroupBarrier();
  }
  let result = workgroupUniformLoad(&best[0]);
  return result;
}
@compute @workgroup_size(64)
fn argmax_batch(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = p.offset + wg.x*p.stride;
  var first = END;
  for (var id = lane; id < p.vocab; id += 64u) {
    if (better(row, id, first)) { first = id; }
  }
  let selected = winner(row, first, lane);
  if (lane == 0u) { output[wg.x] = selected; }
}
@compute @workgroup_size(64)
fn topk_batch(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = p.offset + wg.x*p.stride;
  var ids: array<u32,64>;
  var count = 0u;
  for (var id = lane; id < p.vocab; id += 64u) {
    if (count == p.k && !better(row, id, ids[count-1u])) { continue; }
    var at = min(count, p.k-1u);
    while (at > 0u) {
      if (!better(row, id, ids[at-1u])) { break; }
      ids[at] = ids[at-1u];
      at--;
    }
    ids[at] = id;
    count = min(p.k, count+1u);
  }
  var cursor = 0u;
  for (var rank = 0u; rank < p.k; rank++) {
    var front = END;
    if (cursor < count) { front = ids[cursor]; }
    let selected = winner(row, front, lane);
    if (lane == 0u) {
      let dst = 2u*(wg.x*p.k+rank);
      output[dst] = selected;
      output[dst+1u] = bitcast<u32>(logits[row+selected]);
    }
    if (front == selected) { cursor++; }
  }
}
