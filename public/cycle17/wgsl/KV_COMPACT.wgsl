// One thread owns a word across all accepted slots within a head/plane.
// Increasing source slots and dst <= src make forward in-place compaction safe.
struct Params { prefix: u32, capacity: u32, words: u32, base: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> choice: array<u32>;
@group(0) @binding(2) var<storage,read_write> cache: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.words) { return; }
  let base = p.base + (id.y*p.capacity + p.prefix)*p.words + id.x;
  for (var dst = 1u; dst <= choice[0]; dst++) {
    let src = choice[2u+dst];
    if (dst != src) { cache[base + dst*p.words] = cache[base + src*p.words]; }
  }
}
