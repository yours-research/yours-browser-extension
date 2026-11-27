import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const VECTORS_PATH = 'C:/Users/taylo/.t3/projects/yours-speculative-engine/fixtures/golden-e1-e2-vectors.json';
const WASM_PATH = path.resolve('crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm');
const SHADER_IDS = {
  DN_CONV_TREE: 46,
  DN_IO: 47,
  DN_TREE: 48,
  GQA_TREE: 49,
  KV_COMPACT: 50,
  TOPK_BATCH: 51,
  ACCEPT_TREE: 52,
  Q4_GEMV_SMALLM: 53,
};

async function main() {
  const fixture = JSON.parse(fs.readFileSync(VECTORS_PATH, 'utf8'));
  const wasmBytes = fs.readFileSync(WASM_PATH);
  const wasmBase64 = wasmBytes.toString('base64');

  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end('<!doctype html><html><body></body></html>');
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();

  const browser = await chromium.launch({
    headless: false,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded' });
    const results = await page.evaluate(
      async ({ vectors, wasmB64, shaderMap }) => {
        const binStr = atob(wasmB64);
        const bytes = new Uint8Array(binStr.length);
        for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
        const { instance } = await WebAssembly.instantiate(bytes, {});
        const exp = instance.exports;
        const mem = new Uint8Array(exp.memory.buffer);
        const decoder = new TextDecoder();

        const sources = {};
        for (const [name, id] of Object.entries(shaderMap)) {
          const len = exp.wgsl_build_shader(id);
          const ptr = exp.wgsl_get_shader_ptr();
          sources[name] = decoder.decode(mem.subarray(ptr, ptr + len));
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        const device = await adapter.requestDevice();

        const modCache = new Map();
        const report = [];
        for (const vec of vectors) {
          let mod = modCache.get(vec.shader);
          if (!mod) {
            mod = device.createShaderModule({ code: sources[vec.shader] });
            const info = await mod.getCompilationInfo();
            const errs = info.messages.filter((m) => m.type === 'error');
            if (errs.length > 0) {
              throw new Error(`WGSL compile error (${vec.shader}): ${JSON.stringify(errs)}`);
            }
            modCache.set(vec.shader, mod);
          }

          const bufs = [];
          const bglEntries = [];
          const bgEntries = [];
          for (let i = 0; i < vec.bindings.length; i++) {
            const b = vec.bindings[i];
            const byteSize = Math.max(16, b.words.length * 4);
            let usage = GPUBufferUsage.COPY_DST;
            let bufferType = 'storage';
            if (b.type === 'uniform') {
              usage |= GPUBufferUsage.UNIFORM;
              bufferType = 'uniform';
            } else if (b.type === 'read-only-storage') {
              usage |= GPUBufferUsage.STORAGE;
              bufferType = 'read-only-storage';
            } else {
              usage |= GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
              bufferType = 'storage';
            }
            const buf = device.createBuffer({ size: byteSize, usage });
            device.queue.writeBuffer(buf, 0, new Uint32Array(b.words));
            bufs.push(buf);
            bglEntries.push({
              binding: i,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: bufferType },
            });
            bgEntries.push({ binding: i, resource: { buffer: buf } });
          }

          const bgl = device.createBindGroupLayout({ entries: bglEntries });
          const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
          const pipe = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: mod, entryPoint: vec.entryPoint },
          });
          const bg = device.createBindGroup({
            layout: bgl,
            entries: bgEntries,
          });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipe);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(vec.dispatch[0], vec.dispatch[1], vec.dispatch[2]);
          pass.end();

          const readbacks = [];
          for (let i = 0; i < vec.bindings.length; i++) {
            const b = vec.bindings[i];
            if (b.expected) {
              const stage = device.createBuffer({
                size: b.expected.length * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
              });
              enc.copyBufferToBuffer(bufs[i], 0, stage, 0, b.expected.length * 4);
              readbacks.push({ bindingIdx: i, stage, expected: b.expected });
            }
          }
          device.queue.submit([enc.finish()]);

          let exactMismatches = 0;
          let maxAbsF32Err = 0;
          let firstMismatch = null;
          for (const rb of readbacks) {
            await rb.stage.mapAsync(GPUMapMode.READ);
            const gotU32 = new Uint32Array(rb.stage.getMappedRange().slice(0));
            const gotF32 = new Float32Array(gotU32.buffer);
            const expU32 = new Uint32Array(rb.expected);
            const expF32 = new Float32Array(expU32.buffer);
            rb.stage.unmap();
            rb.stage.destroy();
            for (let w = 0; w < expU32.length; w++) {
              if (gotU32[w] !== expU32[w]) {
                exactMismatches++;
                const absErr =
                  Number.isFinite(gotF32[w]) && Number.isFinite(expF32[w])
                    ? Math.abs(gotF32[w] - expF32[w])
                    : Infinity;
                if (absErr > maxAbsF32Err) maxAbsF32Err = absErr;
                if (!firstMismatch) {
                  firstMismatch = {
                    binding: rb.bindingIdx,
                    word: w,
                    gotU32: gotU32[w],
                    expU32: expU32[w],
                    gotF32: gotF32[w],
                    expF32: expF32[w],
                  };
                }
              }
            }
          }
          for (const buf of bufs) buf.destroy();
          report.push({
            name: vec.name,
            shader: vec.shader,
            entryPoint: vec.entryPoint,
            exactMismatches,
            maxAbsF32Err,
            firstMismatch,
          });
        }
        return report;
      },
      { vectors: fixture.vectors, wasmB64: wasmBase64, shaderMap: SHADER_IDS },
    );

    console.table(results);
    const failed = results.filter((r) => r.exactMismatches > 0);
    if (failed.length > 0) {
      console.error('Golden E1/E2 parity failures:', JSON.stringify(failed, null, 2));
      process.exitCode = 1;
    } else {
      const bitExactCount = results.filter((r) => r.exactMismatches === 0).length;
      console.log(`ALL ${results.length}/${results.length} golden E1/E2 vectors PASSED (${bitExactCount} bit-exact u32 match).`);
    }
  } finally {
    await browser.close();
    await new Promise((r) => server.close(r));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
