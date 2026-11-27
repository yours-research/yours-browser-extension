// P9c: persist each DeltaNet layer's raw verify projections so a rejected
// speculative step replays only conv + recurrence.
//
// Verify writes, per DeltaNet layer, M rows of three projections into workBuf:
//   qkv at [0, M*conv), z at [M*conv, M*(conv+val)), ab at [M*(conv+val), M*w)
// with w = conv + val + ab. Conv then overwrites the qkv rows in place.
//   stash   (verify, after the ab projection, before conv): cache[cache_off + i] = work[i], i < M*w
//   unstash (rollback, replaces the three Q4 recomputes):     work <- the first `keep`
//           rows of each projection, in the keep-row layout the replayed conv and
//           recurrence read (qkv at 0, z at keep*conv, ab at keep*(conv+val)).
// Bindings match the extension's dynRwStorageBgl (dynamic uniform, read-only
// storage, storage, storage): bind auxBuf (unused here), workBuf, the cache.
// Pure copies: the restored rows are bit-identical to the stashed ones, and the
// V2 Q4 kernel gives row t the same bits at any M (scalar-consistent small-M).
struct Params { m: u32, keep: u32, conv: u32, val: u32, ab: u32, cache_off: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> work: array<f32>;
@group(0) @binding(3) var<storage, read_write> cache: array<f32>;

@compute @workgroup_size(256)
fn stash(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * 65535u * 256u;
  if (i >= p.m * (p.conv + p.val + p.ab)) { return; }
  cache[p.cache_off + i] = work[i];
}

@compute @workgroup_size(256)
fn unstash(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * 65535u * 256u;
  let qkv = p.keep * p.conv;
  let qz = p.keep * (p.conv + p.val);
  if (i >= p.keep * (p.conv + p.val + p.ab)) { return; }
  var src = i;
  if (i >= qz) {
    src = p.m * (p.conv + p.val) + (i - qz);
  } else if (i >= qkv) {
    src = p.m * p.conv + (i - qkv);
  }
  work[i] = cache[p.cache_off + src];
}
