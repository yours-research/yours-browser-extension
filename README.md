<img src="assets/yours-icon.png" width="64" height="64" alt="Yours Research">

# TinyGPT Extension

A **100% local, zero-network-permission** Chromium Manifest V3 extension running **Qwen3.5-4B** (`q4f16`, INT8 KV cache, up to **128K** context) inside the browser via a custom **Rust WASM + Native WGSL WebGPU compute engine** hosted in a **Dedicated Web Worker**.

## Architecture

1. **Background Service Worker** ([src/background/service-worker.ts](src/background/service-worker.ts)) — Manages the persistent MV3 Offscreen Document (`ENSURE_OFFSCREEN`) and opens the Side Panel from the browser toolbar action.
2. **Offscreen Host & Dedicated Engine Worker** ([src/offscreen/main.ts](src/offscreen/main.ts), [src/offscreen/engineWorker.ts](src/offscreen/engineWorker.ts), [src/offscreen/inferenceEngine.ts](src/offscreen/inferenceEngine.ts)) — Streams pre-repacked `<= 128 MiB` model weight shards directly from extension storage into WebGPU VRAM, runs inference in a dedicated Web Worker, manages warm prefix reuse and hybrid CPU snapshots, and applies frame-paced GPU submission (`balanced` / `smooth-ui` / `max-speed` via [src/offscreen/gpuPacing.ts](src/offscreen/gpuPacing.ts)).
3. **Custom Rust WASM + Native WGSL Engine** ([crates/tinygpt-wgsl-engine/src/lib.rs](crates/tinygpt-wgsl-engine/src/lib.rs), [src/offscreen/wgslRealEngine.ts](src/offscreen/wgslRealEngine.ts)) — Purpose-built WGSL compute shaders with `shader-f16` and WebGPU `subgroups` covering the hybrid Qwen3.5-4B graph (24 Gated DeltaNet linear-attention layers + 8 GQA full-attention layers, INT8/FP16 KV cache, Split-K FlashPrefill, Q4 MatMul/GEMV, and on-GPU vocab argmax).
4. **Offline Bundled Model Weights** ([scripts/bundle-model.mjs](scripts/bundle-model.mjs)) — Repacks Qwen3.5-4B (`onnx-community/Qwen3.5-4B-ONNX-OPT`) into `model-bundle/qwen35-4b/` (`2.47 GiB` across `<= 128 MiB` binary shards) and syncs them into `dist/models/qwen35-4b/` so the extension runs completely offline with zero `host_permissions`.
5. **Side Panel UI** ([src/sidepanel/App.tsx](src/sidepanel/App.tsx)) — React + `@assistant-ui/react` chat interface with live telemetry (`tok/s`, `TTFT`, `Active Context Tokens`, `Est. VRAM`, GPU pacing selector) and an interactive Lab drawer.

## Quickstart & Commands

```bash
npm install
npm run bundle:model   # Export/repack offline model shards into model-bundle/ & dist/models/
npm run typecheck
npm test
npm run build
```

### Benchmark Suites (Playwright + Chromium WebGPU)

- `npm run e2e:track1` — Legacy ONNX Runtime WebGPU tuning sweep (`benchmarks/runs/01`–`08`)
- `npm run e2e:track2:real` — Real-weight Native WGSL 128K + 32K NIAH sweep (`benchmarks/runs/14`–`16`)
- `npm run e2e:track3` — Offline bundled Worker vs Offscreen Main smoothness & 144Hz FPS sweep (`benchmarks/runs/17`–`20`)

Full cross-track metrics and hardware notes live in [benchmarks/REPORT.md](benchmarks/REPORT.md).

## Load Unpacked in Chrome / Chromium

1. Run `npm run bundle:model` (if not already populated) and `npm run build` to produce `dist/`.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Click **Load unpacked** and select `dist/`.
4. Click the **TinyGPT** extension icon to open the Side Panel.
