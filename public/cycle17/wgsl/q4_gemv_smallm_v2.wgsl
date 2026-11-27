// One workgroup owns 64 output rows. Each lane decodes a row's Q4 words once
// and reuses the resulting weights across all M tokens. Four adjacent tokens
// occupy the components of each accumulator. Every component visits columns
// in scalar decode order, including for M=1.
struct Params {
  rows: u32, cols: u32, weight_offset: u32, input_offset: u32,
  output_offset: u32, residual: u32, tokens: u32, output_stride: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> weights: array<u32>;
@group(0) @binding(2) var<storage,read> input: array<f32>;
@group(0) @binding(3) var<storage,read_write> output: array<f32>;

// 128 columns x 16 tokens = 8 KiB. The transpose is independent of row.
var<workgroup> activation: array<vec4<f32>,512>;

// Make the rounded product observable so a backend cannot contract the dot
// into differently rounded FMAs. Pin the portable flush-to-zero convention.
fn product(w: f32, x: vec4<f32>) -> vec4<f32> {
  let bits = bitcast<vec4<u32>>(x*w);
  return bitcast<vec4<f32>>(select(bits, bits & vec4<u32>(0x80000000u),
    (bits & vec4<u32>(0x7f800000u)) == vec4<u32>(0u)));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x*64u + lane;
  let packs = (p.tokens+3u)/4u;
  let blocks = (p.cols+31u)/32u;
  var sum: array<vec4<f32>,4>;
  for (var first = 0u; first < p.cols; first += 128u) {
    let cols = min(128u, p.cols-first);
    for (var flat = lane; flat < packs*cols; flat += 64u) {
      let pack = flat / cols;
      let c = flat % cols;
      var x = vec4<f32>(0.0);
      for (var j = 0u; j < min(4u,p.tokens-4u*pack); j++) {
        x[j] = input[p.input_offset+(4u*pack+j)*p.cols+first+c];
      }
      activation[pack*128u+c] = x;
    }
    workgroupBarrier();
    if (row < p.rows) {
      for (var block = 0u; block*32u < cols; block++) {
        let relative = (row*blocks+first/32u+block)*5u;
        let base = p.weight_offset+relative;
        let affine = unpack2x16float(weights[base+4u]);
        for (var word = 0u; word < 4u; word++) {
          let packed = weights[base+word];
          for (var j = 0u; j < 8u; j++) {
            let c = block*32u+word*8u+j;
            if (c >= cols) { break; }
            let w = f32(extractBits(packed,j*4u,4u))*affine.x + affine.y;
            for (var pack = 0u; pack < packs; pack++) {
              let x = activation[pack*128u+c];
              sum[pack] += product(w,x);
            }
          }
        }
      }
    }
    workgroupBarrier();
  }
  if (row < p.rows) {
    for (var token = 0u; token < p.tokens; token++) {
      let stride = select(p.output_stride, p.rows, p.output_stride == 0u);
      let dst = p.output_offset+token*stride+row;
      var y = sum[token/4u][token%4u];
      if (p.residual != 0u) { y += output[dst]; }
      output[dst] = y;
    }
  }
}
