<img src="assets/yours-icon.png" width="64" height="64" alt="Yours Research">

# TinyGPT Extension (`yours-browser-extension`)

A **100% local, zero-network-permission** Chromium Manifest V3 extension running **Qwen3.5-4B** (`q4f16`, INT8 KV cache, up to **128K** context) inside the browser via a custom **Rust WASM + Native WGSL WebGPU compute engine** hosted in a **Dedicated Web Worker**.

## Headline Performance (RTX 3090, Offline Bundled `Qwen3.5-4B`)

- **Peak Native WGSL Speed (`21-track4-wgsl-max-speed-128k`, `worker / max-speed`):** **`106.60 tok/s`** decode / **`564.90 tok/s`** prefill at 1K (**`7.83×`** vs. Baseline ORT `13.62 tok/s`, **`27.9%`** bandwidth utilization), **`52.44 / 359.21 tok/s`** at 32K, and **`26.73 / 181.06 tok/s`** at **128K** (`4,641 MiB` VRAM, **`9.48×`** vs. ORT 128K `2.82 tok/s`).
- **Default Production Mode (`22-track4-wgsl-worker-balanced-smoothness`, `worker / balanced`):** **`76.40–80.30 tok/s`** chat decode (`143.4 ms` TTFT), **`293.93 / 37.54 tok/s`** at 32K while maintaining **`143.9 FPS`** (`0.00%` jank >33 ms), **`97.6 ms`** abort latency, warm prefix reuse, CPU snapshot/restore, and **32K NIAH PASS** (`VELVET-ORBIT-7492`) with **`0` external network requests**.

## Architecture

1. **Background Service Worker** ([src/background/service-worker.ts](src/background/service-worker.ts)) — Manages the persistent MV3 Offscreen Document (`ENSURE_OFFSCREEN`) and opens the Side Panel from the browser toolbar action.
2. **Offscreen Host & Dedicated Engine Worker** ([src/offscreen/main.ts](src/offscreen/main.ts), [src/offscreen/engineWorker.ts](src/offscreen/engineWorker.ts), [src/offscreen/inferenceEngine.ts](src/offscreen/inferenceEngine.ts)) — Streams pre-repacked `<= 128 MiB` model weight shards directly from extension storage into WebGPU VRAM, runs inference in a Dedicated Web Worker, manages warm prefix reuse and hybrid CPU snapshots, and applies frame-paced GPU submission (`balanced` / `smooth-ui` / `max-speed` via [src/offscreen/gpuPacing.ts](src/offscreen/gpuPacing.ts)).
3. **Custom Rust WASM + Native WGSL Engine** ([crates/tinygpt-wgsl-engine/src/lib.rs](crates/tinygpt-wgsl-engine/src/lib.rs), [src/offscreen/wgslRealEngine.ts](src/offscreen/wgslRealEngine.ts)) — Purpose-built WGSL compute shaders with `shader-f16` and WebGPU `subgroups`: Chunkwise Parallel WY Gated DeltaNet prefill (`24` linear-attention layers), high-occupancy `16`-row/WG Q4 GEMV (`@workgroup_size(128)`), `64×64` tiled Q4 GEMM, `8`-layer GQA Split-K FlashPrefill with INT8/FP16 KV cache, and GPU vocab argmax.
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
- `npm run e2e:track4` — Chunkwise WY DeltaNet, High-Occupancy Q4 GEMV & Fixed Balanced Pacing (`benchmarks/runs/21`–`22`)

## Documentation & Research

- [benchmarks/REPORT.md](benchmarks/REPORT.md) — Full cross-track benchmark report (Tracks 0–4) & `144Hz` smoothness analysis
- [docs/DOWNLOAD_BENCHMARKS.md](docs/DOWNLOAD_BENCHMARKS.md) — Hugging Face / Xet CAS download throughput benchmarks & secure extension distribution architecture
- [docs/RESEARCH.md](docs/RESEARCH.md) — Deep frontier research synthesis (`Strata`, `magnitude.dev`, `exo`, `ktransformers`, `rapid-mlx`, WebGPU kernels, speculative decoding, quantization)
- [docs/CUSTOM_MODEL_HANDOFF.md](docs/CUSTOM_MODEL_HANDOFF.md) — Standalone specification & training blueprint for a custom fine-tuned/distilled browser-agent model

## Load Unpacked in Chrome / Chromium

1. Run `npm run bundle:model` (if not already populated) and `npm run build` to produce `dist/`.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Click **Load unpacked** and select `dist/`.
4. Click the **TinyGPT** extension icon to open the Side Panel.
