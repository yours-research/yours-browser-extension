/**
 * Oracle for the Tier-1 cooperative small-M Q4 kernels
 * (wgsl/q4_gemv_smallm_v2_f16.wgsl, wgsl/q4_gate_up_swiglu_smallm_v2_f16.wgsl).
 *
 * Emulates the kernel's arithmetic exactly. Lane l of a 32-lane team owns
 * column l of every block and accumulates ascending blocks with one rounded
 * product and one f32 add each. The team total is the xor butterfly
 * (masks 16, 8, 4, 2, 1). Dequantization is either the f16 tier (f16 RNE
 * rounding with flush of f16 subnormals) or the f32 twin used to execute the
 * same kernel on devices without shader-f16.
 */
import type { KernelVector } from './golden.js';
import { add, div, expNeg, f16ToF32, f32ToBits, mul, stage } from './f32.js';

export type Tier = 'f16' | 'f32';

/** Nearest binary16 value to x (round to nearest, ties to even, with subnormals). */
export function roundF16(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  if (a >= 65520) return sign * Infinity;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  if (2 ** (e + 1) <= a) e++;
  const quantum = 2 ** (Math.max(e, -14) - 10);
  const n = a / quantum; // exact: division by a power of two
  const f = Math.floor(n);
  const d = n - f;
  const r = d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
  return sign * r * quantum;
}

/** Kernel `dq_keep`: an f16 subnormal (|x| < 2^-14) becomes a signed zero. */
export function keepF16(x: number): number {
  return x !== 0 && Math.abs(x) < 2 ** -14 ? (x < 0 ? -0 : 0) : x;
}

export function dequant(q: number, scale: number, low: number, tier: Tier): number {
  if (tier === 'f32') return add(mul(q, scale), low);
  return keepF16(roundF16(keepF16(roundF16(q * keepF16(scale))) + keepF16(low)));
}

/** 32-lane xor butterfly with the kernel's mask order; every lane ends equal. */
export function butterfly(lanes: readonly number[]): number {
  let v = lanes.slice();
  for (const m of [16, 8, 4, 2, 1]) v = v.map((x, i) => add(x, v[i ^ m]));
  if (v.some((x) => !Object.is(x, v[0]) && !(x === v[0]))) throw new Error('butterfly lanes diverged');
  return v[0];
}

function swiglu(gate: number, up: number): number {
  const e = expNeg(-Math.abs(gate));
  const sigmoid = stage(mul(gate < 0 ? e : 1, stage(div(1, add(1, e)))));
  return stage(mul(stage(mul(gate, sigmoid)), up));
}

/** Same parameter words and layouts as q4SmallMV2 (src/kernels/q4SmallMV2.ts). */
export function q4SmallMV2F16(
  params: ArrayLike<number>,
  weights: ArrayLike<number>,
  input: ArrayLike<number>,
  initial: ArrayLike<number>,
  fused = false,
  tier: Tier = 'f16',
): Float32Array {
  const [rows, cols, wOff, x, y, residual, tokens, strideOrUp] = Array.from(params);
  const stride = fused || !strideOrUp ? rows : strideOrUp;
  const blocks = Math.ceil(cols / 32);
  const out = Float32Array.from(initial);
  const weightAt = (offset: number, row: number, c: number) => {
    const base = offset + 5 * (row * blocks + (c >>> 5));
    const affine = weights[base + 4];
    const lane = c & 31;
    const q = (weights[base + (lane >>> 3)] >>> (4 * (lane & 7))) & 15;
    return dequant(q, f16ToF32(affine & 65535), f16ToF32(affine >>> 16), tier);
  };
  const dot = (offset: number, row: number, token: number) => {
    const lanes = new Array<number>(32).fill(0);
    for (let l = 0; l < 32; l++)
      for (let b = 0; b < blocks; b++) {
        const c = 32 * b + l;
        if (c < cols) lanes[l] = add(lanes[l], stage(mul(weightAt(offset, row, c), input[x + token * cols + c])));
      }
    return butterfly(lanes);
  };
  for (let token = 0; token < tokens; token++)
    for (let row = 0; row < rows; row++) {
      const v = fused ? swiglu(dot(wOff, row, token), dot(strideOrUp, row, token)) : dot(wOff, row, token);
      const dst = y + token * stride + row;
      out[dst] = residual ? add(v, out[dst]) : v;
    }
  return out;
}

export const F16_SHADERS = { linear: 'q4_gemv_smallm_v2_f16', fused: 'q4_gate_up_swiglu_smallm_v2_f16' } as const;
export const ROWS_PER_WORKGROUP = { linear: 16, fused: 8 } as const;

const F32_BLOCK = `// BEGIN-PRECISION f32 (conformance twin)
fn dequant(q: u32, scale: f32, low: f32) -> f32 {
  return f32(q) * scale + low; // q * scale is exact in f32
}
// END-PRECISION`;

// Express binary16 RNE + FTZ in f32 words. WARP's native pack2x16float
// has implementation-dependent rounding and cannot implement this oracle.
const EMULATED_BLOCK = `// BEGIN-PRECISION f32 (binary16 emulation)
fn half(x: f32) -> f32 {
  let bits = bitcast<u32>(x);
  let sign = bits & 0x80000000u;
  let mag = bits & 0x7fffffffu;
  if (mag < 0x38800000u) { // 2^-14: only the top half-ULP rounds to the least normal half
    return bitcast<f32>(sign | select(0u, 0x38800000u, mag >= 0x387fe000u));
  }
  let rounded = (mag + 0xfffu + ((mag >> 13u) & 1u)) & ~0x1fffu;
  return bitcast<f32>(sign | select(rounded, 0x7f800000u, rounded >= 0x47800000u));
}
fn dequant(q: u32, scale: f32, low: f32) -> f32 {
  return half(half(f32(q) * half(scale)) + half(low));
}
// END-PRECISION`;

/** Split a Tier-1 source into [before, precision block, after]. */
export function precisionBlock(src: string): [string, string, string] {
  const s = src.replace(/\r\n/g, '\n');
  const a = s.indexOf('// BEGIN-PRECISION f16');
  const b = s.indexOf('// END-PRECISION');
  if (a < 0 || b < a) throw new Error('missing precision block');
  const end = b + '// END-PRECISION'.length;
  return [s.slice(0, a), s.slice(a, end), s.slice(end)];
}

/**
 * Text naga/wgpu 24 accept: `enable subgroups;` is required by Chrome (Tint)
 * but not yet implemented in naga, so it is removed; subgroup builtins remain.
 */
export function forNaga(src: string): string {
  return src.replace(/\r\n/g, '\n').replace(/^enable subgroups;\n/m, '');
}

const DETECT_TWIN = `  // BEGIN-SUBGROUP-DETECT (conformance twin: workgroup-memory butterfly)
  let sg = false;
  // END-SUBGROUP-DETECT`;

/** Split at the marked subgroup-detection block. */
export function detectBlock(src: string): [string, string, string] {
  const a = src.indexOf('  // BEGIN-SUBGROUP-DETECT');
  const b = src.indexOf('  // END-SUBGROUP-DETECT');
  if (a < 0 || b < a) throw new Error('missing subgroup-detect block');
  const end = b + '  // END-SUBGROUP-DETECT'.length;
  return [src.slice(0, a), src.slice(a, end), src.slice(end)];
}

/**
 * WARP twins pin the workgroup-memory butterfly. The measured CPU wave is
 * 4 wide; the full detector with wave intrinsics timed out on this driver.
 * Select f16 for bit-level emulation of dequantization, or f32 for scheduling
 * benchmarks. The production topology guard is independently exercised with
 * injected wave memberships in the WARP conformance suite.
 */
export function f32Twin(src: string, precision: Tier = 'f32'): string {
  const [before, , after] = precisionBlock(forNaga(src));
  const [b2, , a2] = detectBlock((before + (precision === 'f16' ? EMULATED_BLOCK : F32_BLOCK) + after).replace(/^enable f16;\n/m, ''));
  return b2 + DETECT_TWIN + a2;
}

/** WARP job for a twin written to `<root>/<shader>.wgsl` (see tests/helpers). */
export function f16Job(shader: string, params: number[], weights: number[], input: number[], initial: number[], fused: boolean): KernelVector {
  return {
    name: shader,
    shader,
    entryPoint: 'main',
    dispatch: [Math.ceil(params[0] / (fused ? ROWS_PER_WORKGROUP.fused : ROWS_PER_WORKGROUP.linear)), 1, 1],
    bindings: [
      { type: 'uniform', words: params },
      { type: 'read-only-storage', words: weights },
      { type: 'read-only-storage', words: input.map(f32ToBits) },
      { type: 'storage', words: initial.map(f32ToBits) },
    ],
  };
}

/** Pipeline-constant variant with fma accumulation (exact_mac = false). */
export function fastMac(src: string): string {
  const s = src.replace('override exact_mac: bool = true;', 'override exact_mac: bool = false;');
  if (s === src) throw new Error('missing exact_mac override');
  return s;
}
