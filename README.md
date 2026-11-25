<img src="assets/yours-icon.png" width="64" height="64" alt="Yours Research">

# TinyGPT Extension (`yours-browser-extension`)

A **100% local, zero-network-permission** Chromium Manifest V3 extension running **Qwen3.5-4B** (`q4f16`, INT8 KV cache, up to **128K** context) inside the browser via a custom **Rust WASM + Native WGSL WebGPU compute engine** hosted in a **Dedicated Web Worker**.

## Headline Performance (RTX 3090, Offline Bundled `Qwen3.5-4B`)

- **Peak Native WGSL Speed (`23-track5-wgsl-max-speed-128k`, `worker / max-speed`):** **`109.55 tok/s`** raw decode / **`506.96 tok/s`** prefill at 1K (**`8.04×`** vs. Baseline ORT `13.62 tok/s`, **`28.7%`** bandwidth utilization; **`114.83 tok/s`** with zero-weight suffix speculation at `94.87%` acceptance & exact bit-identical parity), **`97.18 tok/s`** at 8K, **`74.06 tok/s`** at 32K, **`55.74 tok/s`** at 64K, **`43.71 tok/s`** at 96K, and **`32.32 tok/s`** at **128K** (`4,641 MiB` VRAM, **`11.46×`** vs. ORT 128K `2.82 tok/s`), loaded offline in **`6.97s`**.
- **Default Production Mode (`24-track5-wgsl-worker-balanced-smoothness`, `worker / balanced` + Suffix Speculation):** **`78.29 tok/s`** chat decode (`162.4 ms` TTFT), **`60.41 / 324.90 tok/s`** at 8K, and **`41.20 / 258.80 tok/s`** at 32K while maintaining **`140.25–142.72 FPS`** (`0.00%` jank >33 ms), **`99.3 ms`** abort latency, warm prefix reuse, CPU snapshot/restore, and **32K NIAH PASS** (`VELVET-ORBIT-7492`) with **`0` external network requests**.

## Architecture

1. **Background Service Worker** ([src/background/service-worker.ts](src/background/service-worker.ts)) — Manages the persistent MV3 Offscreen Document (`ENSURE_OFFSCREEN`) and opens the Side Panel from the browser toolbar action.
2. **Offscreen Host & Dedicated Engine Worker** ([src/offscreen/main.ts](src/offscreen/main.ts), [src/offscreen/engineWorker.ts](src/offscreen/engineWorker.ts), [src/offscreen/inferenceEngine.ts](src/offscreen/inferenceEngine.ts)) — Streams pre-repacked `<= 128 MiB` model weight shards directly from extension storage into WebGPU VRAM, runs inference in a Dedicated Web Worker, manages warm prefix reuse, hybrid CPU snapshots, zero-weight suffix speculative decoding ([src/offscreen/suffixSpeculator.ts](src/offscreen/suffixSpeculator.ts)), and adaptive burst-paced GPU submission (`balanced` / `smooth-ui` / `max-speed` via [src/offscreen/gpuPacing.ts](src/offscreen/gpuPacing.ts)).
3. **Custom Rust WASM + Native WGSL Engine** ([crates/tinygpt-wgsl-engine/src/lib.rs](crates/tinygpt-wgsl-engine/src/lib.rs), [src/offscreen/wgslRealEngine.ts](src/offscreen/wgslRealEngine.ts)) — Purpose-built Track 5 WGSL compute shaders with `shader-f16`, WebGPU `subgroups`, and optional `packed_4x8_integer_dot_product` (`dot4I8Packed`): zero-barrier factored Q4 GEMV + `Wide-K` 32-lane GEMV, fused `QKV+Z+AB` / `QG+K+V` / `Gate+Up+SwiGLU` dispatches (`397 → 269` dispatches/token), Chunkwise Parallel WY Gated DeltaNet prefill (`24` linear-attention layers), 4-Q-head grouped GQA INT8/FP16 FlashDecoding (`8` full-attention layers), Split-K FlashPrefill, batched `M <= 5` speculative verify with exact DeltaNet replay rollback, and GPU vocab argmax.
4. **Offline Bundled Model Weights** ([scripts/bundle-model.mjs](scripts/bundle-model.mjs)) — Repacks Qwen3.5-4B (`onnx-community/Qwen3.5-4B-ONNX-OPT`) into `model-bundle/qwen35-4b/` (`2.47 GiB` across `<= 128 MiB` binary shards) and syncs them into `dist/models/qwen35-4b/` so the extension runs completely offline with zero `host_permissions`.
5. **Side Panel UI** ([src/sidepanel/App.tsx](src/sidepanel/App.tsx)) — React + `@assistant-ui/react` chat interface with live telemetry (`tok/s`, `TTFT`, `Active Context Tokens`, `Est. VRAM`, speculative acceptance rate, GPU pacing selector) and an interactive Lab drawer.

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
- `npm run e2e:track5` — Fused `397→269` dispatches, 4-Q-head GQA INT8/DP4a, zero-weight suffix speculation & adaptive decode burst pacing (`benchmarks/runs/23`–`24`)

## Documentation & Research

- [benchmarks/REPORT.md](benchmarks/REPORT.md) — Full cross-track benchmark report (Tracks 0–5) & `144Hz` smoothness analysis
- [docs/DOWNLOAD_BENCHMARKS.md](docs/DOWNLOAD_BENCHMARKS.md) — Hugging Face / Xet CAS download throughput benchmarks & secure extension distribution architecture
- [docs/RESEARCH.md](docs/RESEARCH.md) — Deep frontier research synthesis (`Strata`, `magnitude.dev`, `exo`, `ktransformers`, `rapid-mlx`, WebGPU kernels, speculative decoding, quantization)
- [docs/CUSTOM_MODEL_HANDOFF.md](docs/CUSTOM_MODEL_HANDOFF.md) — Standalone specification & training blueprint for a custom fine-tuned/distilled browser-agent model

## Load Unpacked in Chrome / Chromium

1. Run `npm run bundle:model` (if not already populated) and `npm run build` to produce `dist/`.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Click **Load unpacked** and select `dist/`.
4. Click the **TinyGPT** extension icon to open the Side Panel.
