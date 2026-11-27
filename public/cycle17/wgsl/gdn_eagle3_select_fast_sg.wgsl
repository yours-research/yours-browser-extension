enable subgroups;
// Fast on-GPU greedy argmax + top-1 softmax probability for EXP-007 single-submit chains (Tier 1 subgroups).
// Replaces 16 workgroup barriers in gdn_eagle3_select.wgsl with 2 barriers using subgroupMax / subgroupAdd.
struct Params { subset: u32, slot: u32, records: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;

var<workgroup> sg_v: array<f32, 64>;
var<workgroup> sg_i: array<u32, 64>;
var<workgroup> sg_s: array<f32, 64>;
const END = 0xffffffffu;

@compute @workgroup_size(256)
fn select_top1(@builtin(local_invocation_index) lane: u32,
               @builtin(subgroup_invocation_id) sg_lane: u32,
               @builtin(subgroup_size) sg_size: u32) {
  var bv = -3.40282346e+38;
  var bi = END;
  for (var i = lane; i < p.subset; i += 256u) {
    let v = logits[i];
    if (bi == END || v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  let sg_max = subgroupMax(bv);
  let cand_i = select(END, bi, bi != END && bv == sg_max);
  let win_i = subgroupMin(cand_i);
  if (sg_lane == 0u) {
    let idx = lane / sg_size;
    sg_v[idx] = sg_max;
    sg_i[idx] = win_i;
  }
  workgroupBarrier();
  let n_sg = 256u / sg_size;
  var top = -3.40282346e+38;
  var best = END;
  for (var k = 0u; k < n_sg; k++) {
    let v = sg_v[k];
    let idx = sg_i[k];
    if (idx != END && (best == END || v > top || (v == top && idx < best))) {
      top = v;
      best = idx;
    }
  }
  var s = 0.0;
  for (var i = lane; i < p.subset; i += 256u) {
    s += exp(logits[i] - top);
  }
  let s_sg = subgroupAdd(s);
  if (sg_lane == 0u) { sg_s[lane / sg_size] = s_sg; }
  workgroupBarrier();
  if (lane == 0u) {
    var sum = 0.0;
    for (var k = 0u; k < n_sg; k++) { sum += sg_s[k]; }
    let id = ids[best];
    slots[p.slot + 1u] = id;
    slots[p.records + 2u * p.slot] = id;
    slots[p.records + 2u * p.slot + 1u] = bitcast<u32>(1.0 / sum);
  }
}
