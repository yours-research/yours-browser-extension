// Post-normalization recurrence. Each lane owns one value column. Slot 0 is
// the pending root. The inclusive mask selects exactly one topological path.
// verify: (ceil(dv/32), heads, slots); commit: (ceil(dv/32), heads, 1).
struct Params { slots: u32, heads: u32, dk: u32, dv: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> masks: array<u32>;
@group(0) @binding(2) var<storage,read> inputs: array<f32>; // per slot/head: q, k, v, decay, beta
@group(0) @binding(3) var<storage,read_write> state: array<f32>; // [head, key, value]
@group(0) @binding(4) var<storage,read_write> out: array<f32>;
@group(0) @binding(5) var<storage,read> choice: array<u32>; // ACCEPT_TREE result

fn scan(column: u32, head: u32, leaf: u32, committing: bool) {
  if (column >= p.dv) { return; }
  let state_base = head * p.dk * p.dv + column;
  var s: array<f32,128>;
  for (var i = 0u; i < p.dk; i++) { s[i] = state[state_base + i * p.dv]; }
  let path = masks[leaf];
  let stride = 2u * p.dk + p.dv + 2u;
  for (var slot = 0u; slot <= leaf; slot++) {
    if ((path & (1u << slot)) == 0u) { continue; }
    let base = (slot * p.heads + head) * stride;
    let key = base + p.dk;
    let value = key + p.dk;
    let decay = inputs[value + p.dv];
    var prediction = 0.0;
    for (var i = 0u; i < p.dk; i++) {
      s[i] *= decay;
      prediction += s[i] * inputs[key + i];
    }
    let error = (inputs[value + column] - prediction) * inputs[value + p.dv + 1u];
    var y = 0.0;
    for (var i = 0u; i < p.dk; i++) {
      s[i] += inputs[key + i] * error;
      y += s[i] * inputs[base + i];
    }
    if (slot == leaf && !committing) { out[(slot * p.heads + head) * p.dv + column] = y; }
  }
  if (committing) {
    for (var i = 0u; i < p.dk; i++) { state[state_base + i * p.dv] = s[i]; }
  }
}
@compute @workgroup_size(32)
fn verify(@builtin(global_invocation_id) id: vec3<u32>) { scan(id.x, id.y, id.z, false); }
@compute @workgroup_size(32)
fn commit(@builtin(global_invocation_id) id: vec3<u32>) {
  // choice = [number of accepted drafts, next token, root slot, ...accepted slots]
  scan(id.x, id.y, choice[2u + choice[0]], true);
}
