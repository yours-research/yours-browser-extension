import { add, div, expNeg, f16ToF32, mul, stage } from './f32.js';

/** Scalar decode reference for the V2 family. Inputs are token-major; only
 * the packed matrix pads its last block. Each word offset is absolute.
 */
export function q4SmallMV2(
  params: ArrayLike<number>,
  weights: ArrayLike<number>,
  input: ArrayLike<number>,
  initial: ArrayLike<number>,
  fused = false,
): Float32Array {
  const [rows, cols, weight, x, y, residual, tokens, strideOrUp] = Array.from(params);
  const stride = fused || !strideOrUp ? rows : strideOrUp;
  const blocks = Math.ceil(cols / 32);
  const output = Float32Array.from(initial);

  function dot(offset: number, token: number, row: number): number {
    let acc = 0;
    for (let c = 0; c < cols; c++) {
      const base = offset + 5 * (row * blocks + (c >>> 5));
      const affine = weights[base + 4];
      const q = (weights[base + ((c & 31) >>> 3)] >>> (4 * (c & 7))) & 15;
      const w = add(mul(q, f16ToF32(affine & 65535)), f16ToF32(affine >>> 16));
      acc = add(acc, stage(mul(w, input[x + token * cols + c])));
    }
    return acc;
  }

  for (let token = 0; token < tokens; token++) {
    for (let row = 0; row < rows; row++) {
      let value = dot(weight, token, row);
      if (fused) {
        const e = expNeg(-Math.abs(value));
        const sigmoid = stage(mul(value < 0 ? e : 1, stage(div(1, add(1, e)))));
        value = stage(mul(stage(mul(value, sigmoid)), dot(strideOrUp, token, row)));
      }
      const dst = y + token * stride + row;
      output[dst] = residual ? add(output[dst], value) : value;
    }
  }
  return output;
}
