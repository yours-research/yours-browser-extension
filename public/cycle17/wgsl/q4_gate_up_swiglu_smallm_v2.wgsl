// One workgroup owns 64 output rows. Each lane decodes a row's Q4 words once
// and reuses the resulting weights across all M tokens. Four adjacent tokens
// occupy the components of each accumulator. Every component visits columns
// in scalar decode order, including for M=1.
struct Params {
  rows: u32, cols: u32, weight_offset: u32, input_offset: u32,
  output_offset: u32, residual: u32, tokens: u32, up_offset: u32,
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

fn stage(x: f32) -> f32 {
  let bits = bitcast<u32>(x);
  return bitcast<f32>(select(bits,bits & 0x80000000u,(bits & 0x7f800000u) == 0u));
}

// exp for x <= 0, using the same reduced polynomial as the EXP-007 f32 oracle.
fn exp_neg(x: f32) -> f32 {
  if (x <= -87.33654) { return 0.0; }
  let n = i32(round(x * 1.4426950408889634));
  let r = (x - f32(n)*0.693359375) - f32(n)*-0.00021219444005469058;
  var y = stage(r*0.0013888889) + 0.0083333338;
  y = stage(r*y) + 0.0416666679;
  y = stage(r*y) + 0.1666666716;
  y = stage(r*y) + 0.5;
  y = stage(r*y) + 1.0;
  y = stage(r*y) + 1.0;
  return bitcast<f32>(u32(n+127)<<23u)*y;
}
fn swiglu(gate: f32, up: f32) -> f32 {
  let e = exp_neg(-abs(gate));
  let sigmoid = stage(select(1.0,e,gate < 0.0) * stage(1.0/(1.0+e)));
  return stage(stage(gate*sigmoid)*up);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = wg.x*64u + lane;
  let packs = (p.tokens+3u)/4u;
  let blocks = (p.cols+31u)/32u;
  var sum: array<vec4<f32>,4>;
  var up_sum: array<vec4<f32>,4>;
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
        let up_base = p.up_offset+relative;
        let up_affine = unpack2x16float(weights[up_base+4u]);
        for (var word = 0u; word < 4u; word++) {
          let packed = weights[base+word];
          let up_packed = weights[up_base+word];
          for (var j = 0u; j < 8u; j++) {
            let c = block*32u+word*8u+j;
            if (c >= cols) { break; }
            let w = f32(extractBits(packed,j*4u,4u))*affine.x + affine.y;
            let up_w = f32(extractBits(up_packed,j*4u,4u))*up_affine.x + up_affine.y;
            for (var pack = 0u; pack < packs; pack++) {
              let x = activation[pack*128u+c];
              sum[pack] += product(w,x);
              up_sum[pack] += product(up_w,x);
            }
          }
        }
      }
    }
    workgroupBarrier();
  }
  if (row < p.rows) {
    for (var token = 0u; token < p.tokens; token++) {
      let dst = p.output_offset+token*p.rows+row;
      var y = swiglu(sum[token/4u][token%4u], up_sum[token/4u][token%4u]);
      if (p.residual != 0u) { y += output[dst]; }
      output[dst] = y;
    }
  }
}
