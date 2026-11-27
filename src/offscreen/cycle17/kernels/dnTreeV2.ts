/**
 * DN_TREE_V2 reference: single-pass shared-ancestor DeltaNet tree recurrence.
 *
 * Mirrors wgsl/dn_tree_v2.wgsl operation for operation (f32 rounding via the
 * helpers in f32.ts). The committed state is read once per (head, column);
 * slots are visited depth-first with children ordered by (subtree size, slot)
 * ascending; the parent state is saved only while a non-last child is
 * explored. Each slot is one recurrence update of its parent's state with
 * the exact DN_TREE V1 operation order, so outputs equal V1's and serial
 * M = 1 decode's bit for bit.
 */
import type { KernelVector } from './golden.js';
import { add, f32ToBits, mul, sub } from './f32.js';

export interface DnSchedule {
  /** Slot visit order (preorder). */
  order: number[];
  /** Action before computing order[k]: 'none' (parent state is current), 'peek' or 'pop' the saved parent. */
  restore: ('none' | 'peek' | 'pop')[];
  /** Whether the state after order[k] is pushed before descending. */
  push: boolean[];
  /** Peak number of live saved states. */
  maxStack: number;
  parents: number[];
  sizes: number[];
}

export const DN_V2_MAX_SLOTS = 32;
export const DN_V2_STACK = 4;

/** Traversal schedule derived from inclusive ancestor masks (slot 0 = root). */
export function dnTreeV2Schedule(masks: readonly number[]): DnSchedule {
  const n = masks.length;
  if (!n || n > DN_V2_MAX_SLOTS || masks[0] !== 1) throw new Error('expected 1..32 slots with root mask 1');
  const parents = masks.map((m, i) => (i === 0 ? -1 : 31 - Math.clz32((m ^ (1 << i)) >>> 0)));
  const sizes = masks.map((_, i) => masks.reduce((c, m, j) => c + (j >= i ? (m >>> i) & 1 : 0), 0));
  parents.forEach((p, i) => {
    if (i && (p < 0 || p >= i || (masks[i] >>> 0) !== ((masks[p] | (1 << i)) >>> 0))) throw new Error('inconsistent masks');
  });
  const key = (c: number) => sizes[c] * 64 + c;
  const kids: number[][] = Array.from({ length: n }, () => []);
  for (let c = 1; c < n; c++) kids[parents[c]].push(c);
  kids.forEach((k) => k.sort((a, b) => key(a) - key(b)));
  const order: number[] = [];
  const restore: DnSchedule['restore'] = [];
  const push: boolean[] = [];
  let depth = 0,
    maxStack = 0;
  const visit = (u: number, how: DnSchedule['restore'][number]) => {
    order.push(u);
    restore.push(how);
    if (how === 'pop') depth--;
    const k = kids[u];
    push.push(k.length >= 2);
    if (k.length >= 2) maxStack = Math.max(maxStack, ++depth);
    k.forEach((c, i) => visit(c, i === 0 ? 'none' : i === k.length - 1 ? 'pop' : 'peek'));
  };
  visit(0, 'none');
  return { order, restore, push, maxStack, parents, sizes };
}

/**
 * Execute the V2 traversal. `params` = [slots, heads, dk, dv]; `rec` holds per
 * slot/head records [q(dk), k(dk), v(dv), decay, beta]; `initial` is the
 * committed state [head][key][value]. Returns the per-slot readout
 * (slot, head, column) and the number of recurrence updates per (head, column).
 */
export function dnTreeV2(params: readonly number[], masks: readonly number[], rec: readonly number[], initial: readonly number[]) {
  const [slots, heads, dk, dv] = params;
  const sched = dnTreeV2Schedule(masks.slice(0, slots));
  const stride = 2 * dk + dv + 2;
  const output = new Array<number>(slots * heads * dv).fill(0);
  const step = (s: number[], h: number, j: number, slot: number) => {
    const base = (slot * heads + h) * stride,
      key = base + dk,
      value = key + dk,
      decay = rec[value + dv];
    let prediction = 0;
    for (let i = 0; i < dk; i++) {
      s[i] = mul(s[i], decay);
      prediction = add(prediction, mul(s[i], rec[key + i]));
    }
    const error = mul(sub(rec[value + j], prediction), rec[value + dv + 1]);
    let y = 0;
    for (let i = 0; i < dk; i++) {
      s[i] = add(s[i], mul(rec[key + i], error));
      y = add(y, mul(s[i], rec[base + i]));
    }
    return y;
  };
  for (let h = 0; h < heads; h++)
    for (let j = 0; j < dv; j++) {
      let cur = Array.from({ length: dk }, (_, i) => initial[(h * dk + i) * dv + j]);
      const stack: number[][] = [];
      sched.order.forEach((slot, k) => {
        const how = sched.restore[k];
        if (how !== 'none') {
          cur = stack[stack.length - 1].slice();
          if (how === 'pop') stack.pop();
        }
        output[(slot * heads + h) * dv + j] = step(cur, h, j, slot);
        if (sched.push[k]) stack.push(cur.slice());
      });
    }
  return { output, updatesPerColumn: slots, schedule: sched };
}

/** State after committing the accepted path ending at `leaf` (V1/V2 commit semantics). */
export function dnCommitV2(params: readonly number[], masks: readonly number[], rec: readonly number[], initial: readonly number[], leaf: number) {
  const [, heads, dk, dv] = params;
  const stride = 2 * dk + dv + 2;
  const state = initial.slice();
  const path = masks[leaf];
  for (let h = 0; h < heads; h++)
    for (let j = 0; j < dv; j++) {
      const s = Array.from({ length: dk }, (_, i) => state[(h * dk + i) * dv + j]);
      for (let slot = 0; slot <= leaf; slot++) {
        if (!((path >>> slot) & 1)) continue;
        const base = (slot * heads + h) * stride,
          key = base + dk,
          value = key + dk,
          decay = rec[value + dv];
        let prediction = 0;
        for (let i = 0; i < dk; i++) {
          s[i] = mul(s[i], decay);
          prediction = add(prediction, mul(s[i], rec[key + i]));
        }
        const error = mul(sub(rec[value + j], prediction), rec[value + dv + 1]);
        for (let i = 0; i < dk; i++) s[i] = add(s[i], mul(rec[key + i], error));
      }
      s.forEach((v, i) => (state[(h * dk + i) * dv + j] = v));
    }
  return state;
}

/** Per-(head, column) recurrence updates executed by V1 (every slot rescans its path). */
export function dnV1Updates(masks: readonly number[]): number {
  return masks.reduce((a, m) => a + popcount(m), 0);
}

export function popcount(x: number): number {
  let c = 0;
  for (let v = x >>> 0; v; v &= v - 1) c++;
  return c;
}

/** Inclusive ancestor masks for up to 32 slots (ancestorMasks in reference.ts caps at 16). */
export function masksOf(parents: readonly number[]): number[] {
  if (!parents.length || parents[0] !== -1 || parents.length > DN_V2_MAX_SLOTS) throw new Error('bad tree');
  const masks: number[] = [];
  parents.forEach((p, i) => {
    if (i && (p < 0 || p >= i)) throw new Error('bad parent');
    masks.push(((1 << i) | (i ? masks[p] : 0)) >>> 0);
  });
  return masks;
}

/** KernelVector for a DN_TREE (V1) or dn_tree_v2 verify/commit dispatch. */
export function dnVector(
  shader: 'DN_TREE' | 'dn_tree_v2',
  entryPoint: 'verify' | 'commit',
  params: number[],
  masks: number[],
  rec: number[],
  initial: number[],
  choice: number[] = [0, 0, 0],
): KernelVector {
  const [slots, heads, , dv] = params;
  const z = shader === 'DN_TREE' && entryPoint === 'verify' ? slots : 1;
  return {
    name: `${shader}-${entryPoint}-m${slots}`,
    shader,
    entryPoint,
    dispatch: [Math.ceil(dv / 32), heads, z],
    bindings: [
      { type: 'uniform', words: params },
      { type: 'read-only-storage', words: masks },
      { type: 'read-only-storage', words: rec.map(f32ToBits) },
      { type: 'storage', words: initial.map(f32ToBits) },
      { type: 'storage', words: new Array(slots * heads * dv).fill(0) },
      { type: 'read-only-storage', words: choice },
    ],
  };
}
