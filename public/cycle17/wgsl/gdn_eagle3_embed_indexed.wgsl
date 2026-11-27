// GPU-indexed embedding for the single-submit EXP-007 chain. The token id is
// read from slots[p.slot] (the root is written by the host; later ids are
// written by gdn_eagle3_select), so no host round trip separates draft steps.
// The arithmetic is identical to gdn_eagle3_tied.embed with rows = 1. With a
// split tied table each resident slab owns [start_row, end_row); dispatches
// for the other slabs return without writing.
struct Params { hidden: u32, start_row: u32, end_row: u32, slab_word_offset: u32, slot: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> slab: array<u32>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read> features: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Make a binary32 intermediate observable before the next multiply. The
// exponent test also pins the subnormal convention under fast-math compilers.
fn stage(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  return bitcast<f32>(select(b, b & 0x80000000u, (b & 0x7f800000u) == 0u));
}

fn unpack(qword: u32, d: u32, affine: vec2<f32>) -> f32 {
  return f32(extractBits(qword,(d%8u)*4u,4u)) * affine.x + affine.y;
}

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = slots[p.slot];
  if (id.x >= p.hidden || token < p.start_row || token >= p.end_row) { return; }
  let at = p.slab_word_offset + ((token - p.start_row) * ((p.hidden+31u)/32u) + id.x/32u)*5u;
  let w = unpack(slab[at+id.x%32u/8u], id.x, unpack2x16float(slab[at+4u]));
  // Freeze the reconstructed embedding before mixing in the parent feature.
  output[id.x] = features[id.x] + stage(w);
}
