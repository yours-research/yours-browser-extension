import { Rng } from '../util/rng.js';
import { SamplingParams } from './distribution.js';
import { DraftTree, VerifyResult } from './rejection.js';
import { verifyTopKReadback } from './topKReadback.js';

export interface LogitBatch {
  buffer: GPUBuffer;
  vocab: number;
  stride: number;
  offset?: number;
}

/** Append TOPK_BATCH to the encoder containing the target forward pass. Submit
 * once and map once. The caller retains its pipeline, logits and target state.
 * Transient buffers are per invocation, so overlapping requests cannot mix rows.
 */
export async function finishPathA(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  batch: LogitBatch,
  tree: DraftTree,
  params: SamplingParams,
  rng: Rng,
): Promise<VerifyResult> {
  const { vocab, stride, offset = 0 } = batch;
  const topK = params.topK ?? 0;
  const rows = tree.tokens.length + 1;
  if (
    rows > 16 ||
    !Number.isInteger(topK) ||
    topK < 1 ||
    topK > 64 ||
    !Number.isFinite(params.temperature) ||
    params.temperature <= 0
  ) {
    throw new Error('Path A requires M <= 16, T > 0, and topK in 1..64');
  }
  if (
    ![vocab, stride, offset].every(Number.isSafeInteger) ||
    vocab < 1 ||
    stride < vocab ||
    offset < 0 ||
    4 * (offset + (rows - 1) * stride + vocab) > batch.buffer.size
  )
    throw new Error('bad logit layout');
  const k = Math.min(topK, vocab);
  const size = 8 * k * rows;
  const owned: GPUBuffer[] = [];
  function allocate(desc: GPUBufferDescriptor) {
    const buffer = device.createBuffer(desc);
    owned.push(buffer);
    return buffer;
  }
  try {
    const uniform = allocate({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(uniform.getMappedRange()).set([vocab, stride, k, offset]);
    uniform.unmap();
    const result = allocate({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = allocate({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [uniform, batch.buffer, result].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(rows);
    pass.end();
    encoder.copyBufferToBuffer(result, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return verifyTopKReadback(tree, new Uint32Array(staging.getMappedRange()), vocab, params, rng);
  } finally {
    for (const buffer of owned) buffer.destroy(); // also unmaps any mapped range
  }
}

/** Pack a CPU stochastic decision for DN commit and KV_COMPACT. */
export function commitRecord(result: VerifyResult): Uint32Array {
  return Uint32Array.of(result.accepted, result.emitted[result.accepted], 0, ...result.path.map((i) => i + 1));
}
