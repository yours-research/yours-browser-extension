// DN_TREE_V2: single-pass shared-ancestor Gated DeltaNet tree recurrence.
//
// Drop-in for DN_TREE (identical bindings, params, input layout and commit).
// V1 gives every (column, head, slot) lane its own rescan of the slot's
// ancestor path: M reads of the committed state and sum(depth + 1) recurrence
// updates (136 for a 16-chain). V2 gives one lane per (column, head) and walks
// the tree once: committed state S0 is read once, every slot is ONE update
// applied to its parent's state, and siblings share that parent state.
//
// Traversal: depth-first, children ordered by (subtree size, slot) ascending.
// The parent's state is saved only while a non-last child is explored and is
// consumed in place by the last (largest) child, so at most
// floor(log2(slots)) states are live: <= 3 for M <= 16, <= 4 for M <= 32.
// Each slot's arithmetic is the exact V1 sequence on the exact V1 operand
// values, so outputs are bit-identical to V1 and to serial M=1 decode.
// Verify never writes `state`; commit replays only the accepted path.
//
// verify: dispatch (ceil(dv / 32), heads, 1). commit: (ceil(dv / 32), heads, 1).
struct Params { slots: u32, heads: u32, dk: u32, dv: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> masks: array<u32>;
@group(0) @binding(2) var<storage,read> inputs: array<f32>; // per slot/head: q, k, v, decay, beta
@group(0) @binding(3) var<storage,read_write> state: array<f32>; // [head, key, value]
@group(0) @binding(4) var<storage,read_write> out: array<f32>;
@group(0) @binding(5) var<storage,read> choice: array<u32>; // ACCEPT_TREE result

const MAX_SLOTS: u32 = 32u;
const STACK: u32 = 4u;

// One recurrence update of `s` (state of the slot's parent) for slot `slot`.
// Operation order is exactly DN_TREE V1's.
fn step(s: ptr<function, array<f32,128>>, column: u32, head: u32, slot: u32) -> f32 {
  let stride = 2u * p.dk + p.dv + 2u;
  let base = (slot * p.heads + head) * stride;
  let key = base + p.dk;
  let value = key + p.dk;
  let decay = inputs[value + p.dv];
  var prediction = 0.0;
  for (var i = 0u; i < p.dk; i++) {
    (*s)[i] *= decay;
    prediction += (*s)[i] * inputs[key + i];
  }
  let error = (inputs[value + column] - prediction) * inputs[value + p.dv + 1u];
  var y = 0.0;
  for (var i = 0u; i < p.dk; i++) {
    (*s)[i] += inputs[key + i] * error;
    y += (*s)[i] * inputs[base + i];
  }
  return y;
}

@compute @workgroup_size(32)
fn verify(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let head = id.y;
  if (column >= p.dv) { return; }
  let n = min(p.slots, MAX_SLOTS);

  // Tree structure from the inclusive ancestor masks (identical in every lane).
  var parent: array<i32, 32>;
  var size: array<u32, 32>;
  var first_child: array<i32, 32>;
  var next_sib: array<i32, 32>;
  var kids: array<u32, 32>;
  for (var i = 0u; i < n; i++) {
    parent[i] = select(i32(firstLeadingBit(masks[i] ^ (1u << i))), -1, i == 0u);
    var c = 0u;
    for (var j = i; j < n; j++) { c += (masks[j] >> i) & 1u; }
    size[i] = c;
    first_child[i] = -1;
    next_sib[i] = -1;
    kids[i] = 0u;
  }
  // Order key (subtree size, slot): unique, so the order is total.
  for (var c = 1u; c < n; c++) {
    let pc = u32(parent[c]);
    kids[pc] += 1u;
    let kc = size[c] * 64u + c;
    var best = -1;
    var best_key = 0xffffffffu;
    for (var d = 1u; d < n; d++) {
      let kd = size[d] * 64u + d;
      if (parent[d] == i32(pc) && kd > kc && kd < best_key) { best = i32(d); best_key = kd; }
    }
    next_sib[c] = best;
    let f = first_child[pc];
    if (f < 0 || kc < size[u32(f)] * 64u + u32(f)) { first_child[pc] = i32(c); }
  }

  var cur: array<f32, 128>;
  var stack: array<array<f32, 128>, 4>;
  var sp = 0u;
  let state_base = head * p.dk * p.dv + column;
  for (var i = 0u; i < p.dk; i++) { cur[i] = state[state_base + i * p.dv]; }

  var u = 0u;
  out[(u * p.heads + head) * p.dv + column] = step(&cur, column, head, u);
  for (var visited = 1u; visited < n; visited++) {
    if (first_child[u] >= 0) {
      if (kids[u] >= 2u && sp < STACK) {
        for (var i = 0u; i < p.dk; i++) { stack[sp][i] = cur[i]; }
        sp += 1u;
      }
      u = u32(first_child[u]);
    } else {
      var a = u;
      while (next_sib[a] < 0 && a != 0u) { a = u32(parent[a]); }
      let nx = u32(next_sib[a]);
      // Restore the shared parent state: peek for a middle sibling, pop for the last.
      for (var i = 0u; i < p.dk; i++) { cur[i] = stack[sp - 1u][i]; }
      if (next_sib[nx] < 0) { sp -= 1u; }
      u = nx;
    }
    out[(u * p.heads + head) * p.dv + column] = step(&cur, column, head, u);
  }
}

// Commit the accepted path (choice = [accepted, next token, root slot, ...slots]).
@compute @workgroup_size(32)
fn commit(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let head = id.y;
  if (column >= p.dv) { return; }
  let leaf = choice[2u + choice[0]];
  let path = masks[leaf];
  var s: array<f32, 128>;
  let state_base = head * p.dk * p.dv + column;
  for (var i = 0u; i < p.dk; i++) { s[i] = state[state_base + i * p.dv]; }
  for (var slot = 0u; slot <= leaf; slot++) {
    if ((path & (1u << slot)) == 0u) { continue; }
    _ = step(&s, column, head, slot);
  }
  for (var i = 0u; i < p.dk; i++) { state[state_base + i * p.dv] = s[i]; }
}
