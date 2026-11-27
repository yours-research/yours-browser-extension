/**
 * IEEE-754 binary32 emulation helpers for kernel references.
 *
 * The reference rounds each indicated intermediate with Math.fround. Shader
 * parity is checked by comparing the resulting words on the software adapter.
 * Native shader math and compiler contraction can differ across backends.
 */
const f = Math.fround;

export const add = (a: number, b: number): number => f(a + b);
export const sub = (a: number, b: number): number => f(a - b);
export const mul = (a: number, b: number): number => f(a * b);
export const div = (a: number, b: number): number => f(a / b);
export const sqrt = (a: number): number => f(Math.sqrt(a));
export const exp = (a: number): number => f(Math.exp(a));
export const log = (a: number): number => f(Math.log(a));
/** WGSL inverseSqrt reference: correctly rounded 1/sqrt(x). */
export const rsqrt = (a: number): number => f(1 / Math.sqrt(a));
export const fr = f;

export function invSqrt(x: number): number {
  return div(1, sqrt(x));
}

export function logOnePlus(x: number): number {
  const v = div(x, add(2, x)),
    square = mul(v, v);
  let poly = f(1 / 17);
  for (let i = 15; i >= 1; i -= 2) poly = add(f(1 / i), mul(square, poly));
  return mul(mul(2, v), poly);
}

/** Match DN_IO's binary32 stage, including the sign of a flushed subnormal. */
export function stage(x: number): number {
  const bits = f32ToBits(x);
  return (bits & 0x7f800000) === 0 ? bitsToF32(bits & 0x80000000) : x;
}

/** Mirrors the WGSL range reduction and binary32 polynomial. */
export function expNeg(x: number): number {
  if (x <= f(-87.33654)) return 0;
  const n = roundEven(mul(x, f(1.4426950408889634)));
  const r = sub(sub(x, mul(n, 0.693359375)), mul(n, f(-0.00021219444005469058)));
  let y = add(mul(r, f(0.0013888889)), f(0.0083333338));
  for (const c of [0.0416666679, 0.1666666716, 0.5, 1, 1]) y = add(mul(r, y), f(c));
  return mul(2 ** n, y);
}

const buf = new ArrayBuffer(4);
const f32v = new Float32Array(buf);
const u32v = new Uint32Array(buf);

export function f32ToBits(x: number): number {
  f32v[0] = x;
  return u32v[0];
}
export function bitsToF32(u: number): number {
  u32v[0] = u >>> 0;
  return f32v[0];
}

/** Exact IEEE half -> f32 decode (matches WGSL unpack2x16float). */
export function f16ToF32(h: number): number {
  const s = (h >> 15) & 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  let v: number;
  if (e === 0) v = m * 2 ** -24;
  else if (e === 31) v = m ? NaN : Infinity;
  else v = (1 + m / 1024) * 2 ** (e - 15);
  return s ? -v : v;
}

/** Round-to-nearest-even f32 -> f16 bits (generator use only). */
export function f32ToF16(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const s = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a >= 65520) return s | 0x7c00;
  if (a < 2 ** -24 / 2) return s;
  // Find the nearest representable value by brute exponent selection.
  let e = Math.floor(Math.log2(a));
  if (e < -14) {
    // subnormal: step 2^-24
    const m = roundEven(a / 2 ** -24);
    return s | m; // m <= 1024 -> may roll into exponent 1 (0x400), still correct encoding
  }
  let m = roundEven((a / 2 ** e - 1) * 1024);
  if (m === 1024) {
    m = 0;
    e++;
  }
  if (e > 15) return s | 0x7c00;
  return s | ((e + 15) << 10) | m;
}

function roundEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 ? 2 * Math.round(x / 2) : r;
}

/** Order-preserving u32 key for measuring ULP distances. NaNs map below -inf. */
export function orderKey(x: number): number {
  const u = f32ToBits(x);
  if ((u & 0x7fffffff) > 0x7f800000) return 0; // NaN
  return (u & 0x80000000 ? ~u : u | 0x80000000) >>> 0;
}

export function ulpDistance(a: number, b: number): number {
  if (Object.is(a, b)) return 0;
  const ka = orderKey(a);
  const kb = orderKey(b);
  return Math.abs(ka - kb);
}
