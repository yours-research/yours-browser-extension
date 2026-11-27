// The existing target Q4 token table serves as both embedding and lm_head.
// IDs are actual target token IDs. Subset order determines output logit order.
struct Params { hidden: u32, rows: u32, subset: u32, slab_word_offset: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> slab: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> features: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Make a binary32 intermediate observable before the next multiply. The
// exponent test also pins the subnormal convention under fast-math compilers.
fn stage(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  return bitcast<f32>(select(b, b & 0x80000000u, (b & 0x7f800000u) == 0u));
}

fn block(row: u32, b: u32) -> u32 {
  return p.slab_word_offset + (row * ((p.hidden+31u)/32u) + b)*5u;
}
fn unpack(qword: u32, d: u32, affine: vec2<f32>) -> f32 {
  return f32(extractBits(qword,(d%8u)*4u,4u)) * affine.x + affine.y;
}

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.hidden || id.y >= p.rows) { return; }
  let at = block(ids[id.y], id.x/32u);
  let w = unpack(slab[at+id.x%32u/8u], id.x, unpack2x16float(slab[at+4u]));
  let dst = id.y*p.hidden+id.x;
  // Freeze the reconstructed embedding before mixing in the independently
  // computed parent feature. Reassociation with the affine bias changes
  // low bits on generated branches.
  output[dst] = features[dst] + stage(w);
}

fn score(row: u32, feature_row: u32) -> f32 {
  var sum = 0.0;
  for (var b = 0u; b < (p.hidden+31u)/32u; b++) {
    let at = block(row,b);
    let affine = unpack2x16float(slab[at+4u]);
    for (var word = 0u; word < 4u; word++) {
      let packed = slab[at+word];
      for (var j = 0u; j < 8u; j++) {
        let d = 32u*b + 8u*word + j;
        if (d < p.hidden) {
          sum += stage(unpack(packed,j,affine) * features[feature_row*p.hidden+d]);
        }
      }
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn subset_logits(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.subset || id.y >= p.rows) { return; }
  output[id.y*p.subset+id.x] = score(ids[id.x], id.y);
}

// One anchor per launch. Each resident target slab owns a disjoint subset
// of output positions. Pairs are (token row within slab, global subset index).
@compute @workgroup_size(64)
fn subset_logits_slice(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.subset) { return; }
  output[ids[2u*id.x+1u]] = score(ids[2u*id.x], 0u);
}
