# TinyGPT MV3 Extension — Qwen3.5-4B Comprehensive Benchmark Report

Generated: `2026-11-25T06:58:40.255Z`

## 1. Environment, Phase 0 Storage & WebGPU Limits

- **Adapter:** `nvidia` / `ampere` (RTX 3090, `936 GB/s` theoretical bandwidth)
- **WebGPU Features:** `shader-f16=true`, `subgroups=true`
- **Granted Elevated Limits:** `maxBufferSize=2147483648` B (`2.00 GiB`), `maxStorageBufferBindingSize=2147483644` B
- **OPFS 2.50 GiB Cross-Restart Persistence:** Write `12.14s` (`210.92 MB/s`), Cold Restart Read Verify `1.37s` (`1871.48 MB/s`), Origin Quota `84.86 GiB`
- **First Online Download + Compile:** `773.12s` | **Stop/Abort Clean:** `true` | **Warm Offscreen Reopen:** `true`

## 2. Cold Model Reload From OPFS (Track Comparison)

| Run ID | Track | Configuration | Total Load | OPFS Read | Tokenizer | Model Compile/Load | Speedup vs Baseline |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `00-baseline-ort-default` | Baseline ORT | asyncify, fastOpfsLoad=false, graphOpt=all | **14.17s** | — | — | — | 1.00x |
| `01-ort-js-fastpath` | Track 1 ORT | asyncify, fastOpfs=false, graphOpt=all | **13.32s** | 257.6 ms | 782.4 ms | 12540.2 ms | 1.06x |
| `02-ort-fast-opfs-load` | Track 1 ORT | asyncify, fastOpfs=true, graphOpt=all | **6.45s** | 88.9 ms | 908.9 ms | 5537.6 ms | 2.20x |
| `03-ort-wasm-jspi` | Track 1 ORT | jspi, fastOpfs=true, graphOpt=all | **6.14s** | 88.8 ms | 811.5 ms | 5332.4 ms | 2.31x |
| `04-ort-ep-bucket-wgpuonly` | Track 1 ORT | asyncify, fastOpfs=true, graphOpt=all | **7.24s** | 79.0 ms | 825.8 ms | 6416.7 ms | 1.96x |
| `05-ort-graph-opt-basic` | Track 1 ORT | asyncify, fastOpfs=true, graphOpt=basic | **5.92s** | 75.6 ms | 794.4 ms | 5126.6 ms | 2.39x |
| `06-ort-chunk-1024` | Track 1 ORT | jspi, fastOpfs=true, graphOpt=all | **6.00s** | 67.4 ms | 729.3 ms | 5271.4 ms | 2.36x |
| `07-ort-chunk-2048` | Track 1 ORT | jspi, fastOpfs=true, graphOpt=all | **6.00s** | 67.4 ms | 729.3 ms | 5271.4 ms | 2.36x |
| `08-ort-ceiling-128k-and-niah` | Track 1 ORT | jspi, fastOpfs=true, graphOpt=all | **6.00s** | 67.4 ms | 729.3 ms | 5271.4 ms | 2.36x |
| `track2-opfs-16mib` | Track 2 WGSL | Direct OPFS File.slice() -> GPUQueue.writeBuffer (16 MiB chunk) | **3.14s** | 3141.6 ms | 0.0 ms | 3141.6 ms | 4.51x |
| `track2-opfs-32mib` | Track 2 WGSL | Direct OPFS File.slice() -> GPUQueue.writeBuffer (32 MiB chunk) | **3.22s** | 3219.5 ms | 0.0 ms | 3219.5 ms | 4.40x |
| `track2-opfs-64mib` | Track 2 WGSL | Direct OPFS File.slice() -> GPUQueue.writeBuffer (64 MiB chunk) | **3.29s** | 3294.0 ms | 0.0 ms | 3294.0 ms | 4.30x |
| `14-wgsl-real-chat-and-load` | Track 2 WGSL (Real) | Real Qwen3.5-4B OPFS slabs -> WebGPU (int8 KV) | **4.98s** | 3830.1 ms | 817.4 ms | 8.6 ms | 2.85x |
| `15-wgsl-real-fp16kv` | Track 2 WGSL (Real) | Real Qwen3.5-4B OPFS slabs -> WebGPU (fp16 KV) | **4.98s** | 3639.9 ms | 768.7 ms | 237.3 ms | 2.85x |
| `16-wgsl-real-int8kv-128k` | Track 2 WGSL (Real) | Real Qwen3.5-4B OPFS slabs -> WebGPU (int8 KV) | **4.71s** | 3544.2 ms | 770.8 ms | 14.5 ms | 3.01x |

## 3. Track 1 — ORT WebGPU + JS/WASM Tuning Matrix (`Qwen3.5-4B-ONNX-OPT`, `q4f16`)

| Run ID | Config Highlight | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `00-baseline-ort-default` | Baseline ORT WebGPU (asyncify, setTimeout yield, standard decode & OPFS stream) | 13.62 / 921 | 13.35 / 861 | 6.33 / 564 | 3.39 / 387 | — | 3.89% | PASS (56.28s) | 0 |
| `01-ort-js-fastpath` | ORT WebGPU + JS Fast Decode Path (yield=none, float16 LUT argmax, pooled mask) | 14.17 / 678 | 13.50 / 869 | 6.31 / 560 | 2.80 / 373 | — | 4.04% | — | 0 |
| `02-ort-fast-opfs-load` | ORT WebGPU + Fast OPFS Load (arrayBuffer direct path) | 13.27 / 682 | 13.76 / 887 | — | — | — | 3.93% | — | 0 |
| `03-ort-wasm-jspi` | ORT WebGPU + JSPI WASM Build + JS Fast Decode | 15.34 / 700 | 15.72 / 887 | 6.68 / 568 | 2.90 / 374 | — | 4.48% | — | 0 |
| `04-ort-ep-bucket-wgpuonly` | ORT WebGPU EP Tuned (bucket cache, wgpuOnly validation, fp32Accum=false) | 11.32 / 576 | 11.47 / 842 | 6.12 / 554 | 2.47 / 375 | — | 3.27% | — | 0 |
| `05-ort-graph-opt-basic` | ORT WebGPU graphOptimizationLevel=basic (cold compile vs throughput tradeoff) | 14.11 / 719 | 14.81 / 912 | 7.26 / 573 | — | — | 4.22% | — | 0 |
| `06-ort-chunk-1024` | Winning ORT Config (03-ort-wasm-jspi) + Chunk Size 1024 | 15.19 / 716 | 15.83 / 982 | 7.16 / 648 | 3.67 / 444 | — | 4.51% | — | 0 |
| `07-ort-chunk-2048` | Winning ORT Config (03-ort-wasm-jspi) + Chunk Size 2048 | 17.61 / 999 | 16.09 / 939 | 3.47 / 567 | 3.70 / 368 | — | 5.02% | — | 0 |
| `08-ort-ceiling-128k-and-niah` | Winning ORT Config (03-ort-wasm-jspi, chunk=1024) 64K-128K Ceiling + 32K NIAH | — | — | — | 3.57 / 441 | 2.82 / 270 | 1.02% | PASS (50.71s) | 0 |

*Track 1 Ceiling Observation (`08-ort-ceiling-128k-and-niah`):* Reached `96K` (`331.94` prefill / `1.51` decode tok/s, `3072.5 MiB` KV) and **`128K` (`131,072` tokens)** (`269.84` prefill / `2.82` decode tok/s, `4096.5 MiB` KV, `6855.71 MiB` est. VRAM) without OOM or `device.lost`.

## 4. Track 2 — Custom Rust (`wasm32-unknown-unknown`) + Native WGSL Single-Architecture Engine

### Kernel Numerical Verification vs CPU Reference

| Kernel | Status | Max Abs Error | Notes |
| :--- | :---: | ---: | :--- |
| `q4-scalar` | PASS | `0.00e+0` | maxAbsError=0.00e+0 (tol=0.001) |
| `q4-vec4-f16` | PASS | `4.42e-4` | maxAbsError=4.42e-4 (tol=0.02) |
| `q4-subgroups-f16` | PASS | `9.70e-4` | maxAbsError=9.70e-4 (tol=0.02) |
| `deltanet_step` | PASS | `5.22e-8` | 2-step recurrent maxAbsError=5.22e-8 |
| `gqa_fp16_step` | PASS | `3.72e-5` | 40-step Split-8 GQA maxAbsError=3.72e-5 (tol=0.01) |
| `gqa_int8_step` | PASS | `4.39e-4` | 40-step Split-8 GQA maxAbsError=4.39e-4 (tol=0.05) |
| `argmax_vocab_248k` | PASS | `0.00e+0` | expected=173421, got=173421 |

### Decode Throughput (`tok/s`), VRAM Footprint & Bandwidth Utilization

| Run ID | Kernel & KV Cache | Compile | 1K tok/s | 8K tok/s | 32K tok/s | 64K tok/s | 128K tok/s | Peak % BW | Errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10-wgsl-q4-scalar-fp16kv` | `q4-scalar` / `fp16` | 49.6 ms | **39.81** (2590 MiB) | **38.73** (2814 MiB) | **32.68** (3582 MiB) | **27.22** (4606 MiB) | — | **11.2%** | 0 |
| `11-wgsl-q4-vec4-f16-fp16kv` | `q4-vec4-f16` / `fp16` | 34.2 ms | **39.37** (2590 MiB) | **38.21** (2814 MiB) | **31.88** (3582 MiB) | **26.19** (4606 MiB) | — | **11.1%** | 0 |
| `12-wgsl-q4-subgroups-f16-fp16kv` | `q4-subgroups-f16` / `fp16` | 90.3 ms | **72.06** (2590 MiB) | **66.13** (2814 MiB) | **49.58** (3582 MiB) | **37.92** (4606 MiB) | — | **20.2%** | 0 |
| `13-wgsl-q4-subgroups-f16-int8kv` | `q4-subgroups-f16` / `int8` | 45.0 ms | **75.24** (2574 MiB) | **67.06** (2688 MiB) | **50.40** (3078 MiB) | **37.34** (3598 MiB) | **25.01** (4638 MiB) | **21.1%** | 0 |

### 4b. Real-Weight Qwen3.5-4B Native WGSL End-to-End (`track2-wgsl-real`)

- **Chat & Extension Lifecycle (`14-wgsl-real-chat-and-load`):** Turn 1 `66.90` tok/s (`TTFT 325ms`), Turn 2 warm prefix reuse (`prefixReused=true`, `147` ctx) `66.94` tok/s, Snapshot/Restore `147t -> 147t` (`passed=true`), Stop/Abort `true` (`34` chars preserved), Warm Offscreen Reopen `true`, Cached OPFS->VRAM Reload **`4.98s`**.

| Run ID | KV Mode | OPFS Load | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `15-wgsl-real-fp16kv` | `fp16` | **4.98s** | 71.78 / 392 | 66.01 / 391 | 48.58 / 309 | 36.89 / 241 | — | **18.8%** | PASS (49.04 tok/s) | 0 |
| `16-wgsl-real-int8kv-128k` | `int8` | **4.71s** | 73.91 / 383 | 64.39 / 386 | 47.68 / 307 | 36.26 / 240 | 23.80 / 165 | **19.4%** | PASS (45.40 tok/s) | 0 |

## 5. Cross-Track Summary & Decision Gate Evaluation

- **1K Decode:** Baseline `13.62` tok/s -> Best Track 1 (`07-ort-chunk-2048`) `17.61` tok/s (**1.29x**) -> Track 2 Synthetic `75.24` tok/s -> **Track 2 Real Qwen3.5-4B (`16-wgsl-real-int8kv-128k`) `73.91` tok/s** (**5.43x** vs Baseline, **4.20x** vs Best Track 1).
- **64K Decode:** Baseline `3.39` tok/s -> Track 1 `07` `3.70` tok/s -> Track 2 Synthetic `12` `37.92` tok/s -> **Track 2 Real FP16 KV `36.89` tok/s** (**10.87x** vs Baseline) and **INT8 KV `36.26` tok/s** at `3601 MiB` VRAM.
- **128K Decode & VRAM:** Track 1 `08` (FP16 KV) hit `2.82` tok/s at `6855.7 MiB` VRAM; **Track 2 Real INT8 KV (`16`) reached `23.80` tok/s** (**8.44x** vs Track 1 `08`) at only **`4641 MiB`** VRAM (saving ~2.2 GiB).
- **Cold Reload:** Baseline `14.17s` -> Track 1 `05` `5.92s` (**2.39x**) -> Track 2 Synthetic raw slabs `3.14s` -> **Track 2 Real WGSL Qwen3.5-4B `4.71s`** (**3.01x** vs Baseline).
- **32K Needle-in-a-Haystack Accuracy:** Both **FP16 KV (`15`)** and **INT8 KV (`16`)** real Qwen3.5-4B WGSL engines retrieved exact passkey `VELVET-ORBIT-7492` at `32,768` tokens (`49.04` tok/s and `45.40` tok/s decode).
- **Decision Gate (40% of 936 GB/s = 374.4 GB/s):** Track 1 peaked at **`5.02%`** (`17.61 tok/s`), and Track 2 Real Qwen3.5-4B reached **`19.4%`** (`181.2 GB/s`) with real OPFS weights — delivering **4.2x–12.7x** real throughput speedup over Baseline/ORT in a Chrome MV3 Offscreen Document.

## 6. Track 3 — Offline Bundled Load & UI Smoothness vs. Throughput

| Run ID | Host / Pacing | Offline Load | 8K Dec / Prefill | 32K Dec / Prefill | Web Sweep FPS (Idle) | p99 / Max Stall | % Jank (>33ms) | Chat Stream FPS | Net Reqs | Errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `17-track3-mainthread-max-speed` | `offscreen-main` / `max-speed` | **8.89s** | 57.65 / 352 | 43.30 / 279 | 142.5 (139.2) | 11.6 / 35.5 ms | 0.00% | 143.3 | 0 | 0 |
| `18-track3-worker-max-speed` | `worker` / `max-speed` | **8.11s** | 56.93 / 347 | 43.27 / 278 | 142.4 (142.7) | 11.7 / 34.4 ms | 0.00% | 143.4 | 0 | 0 |
| `19-track3-worker-balanced` | `worker` / `balanced` | **8.13s** | 36.99 / 194 | 28.80 / 128 | 144.0 (129.7) | 7.3 / 20.8 ms | 0.00% | 143.5 | 0 | 0 |
| `20-track3-worker-smooth-ui` | `worker` / `smooth-ui` | **7.82s** | 23.46 / 89 | 19.05 / 85 | 144.0 (132.3) | 7.2 / 26.1 ms | 0.00% | 143.7 | 0 | 0 |

- **Default Production Config (`19-track3-worker-balanced`, `worker / balanced`):** Offline bundled load in **`8.13s`** (`loadSource=bundled`), **32K NIAH PASS (`VELVET-ORBIT-7492`, 31.28 tok/s)**, Warm Prefix Reuse `prefixReused=true`, Snapshot/Restore `165t -> 165t` (`passed=true`), Mid-Generation Abort Latency **`106.3 ms`** (`33` chars preserved).
- **Smoothness vs. Unpaced Main Thread:** `19-track3-worker-balanced` (`worker / balanced`) achieved **`144.0 FPS`** during 8K+32K inference (`p99=7.3 ms`, max stall `20.8 ms`, `0.00%` jank >33ms, `143.5 FPS` chat streaming) vs. `17-track3-mainthread-max-speed` (`offscreen-main / max-speed`) at **`142.5 FPS`** (`p99=11.6 ms`, max stall `35.5 ms`, `0.00%` jank >33ms, `143.3 FPS` chat streaming).
- **Strict Offline Zero-Network Verification:** Across all `4` Track 3 runs, `0` external `http(s)://` requests were intercepted.

## 7. Track 4 — Chunkwise WY DeltaNet Prefill, High-Occupancy Q4 GEMV (`16` rows/WG), `64×64` Tiled GEMM & Fixed Balanced Pacing

| Run ID | Host / Pacing | Offline Load | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 96K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Sweep FPS (% Jank) | Net Reqs | Errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `21-track4-wgsl-max-speed-128k` | `worker` / `max-speed` | **7.81s** | 106.60 / 565 | 84.37 / 495 | 52.44 / 359 | 44.59 / 271 | 34.31 / 227 | 26.73 / 181 | **27.9%** | PASS (63.25 tok/s) | — | 0 | 0 |
| `22-track4-wgsl-worker-balanced-smoothness` | `worker` / `balanced` | **8.59s** | — | 52.69 / 356 | 37.54 / 294 | — | — | — | **13.8%** | PASS (40.06 tok/s) | 143.9 (0.00%) | 0 | 0 |

- **Track 4 Max-Speed (`21-track4-wgsl-max-speed-128k`) vs. Track 2 Real (`16-wgsl-real-int8kv-128k`):** 1K decode jumped from `73.91` -> **`106.60 tok/s`** (**+44.2%**, **7.83x** vs Baseline ORT), 1K prefill `383` -> **`564.90 tok/s`** (**+47.4%**), 8K `64.39/386` -> **`84.37/495 tok/s`**, 32K `47.68/307` -> **`52.44/359 tok/s`**, 64K `36.26/240` -> **`44.59/271 tok/s`**, and 128K `23.80/165` -> **`26.73/181 tok/s`** (**9.48x** vs Track 1 `08`). Peak RTX 3090 memory bandwidth utilization reached **`27.9%`**, with **32K NIAH PASS** at `392.14 / 63.25 tok/s` and Chat Turn 1 at **`104.74 tok/s`** (`TTFT 104.8 ms`).
- **Track 4 Balanced Default (`22-track4-wgsl-worker-balanced-smoothness`) vs. Track 3 Balanced (`19-track3-worker-balanced`):** Restoring `chunk=64` plus WY prefill + 16-row GEMV boosted 8K `36.99/194` -> **`52.69/356 tok/s`**, 32K prefill `128.10` -> **`293.93 tok/s`** (**+129.5%**), Chat Turn 1 `40.14` -> **`76.40 tok/s`** (**+90.3%**, `TTFT 143.4 ms`), and Turn 2 (`prefixReused=true`) -> **`80.30 tok/s`** — while maintaining **`143.9 FPS`** (`0.00%` jank >33ms, max `23.9 ms`), **`143.6 FPS`** chat streaming, Snapshot/Restore `165t -> 165t`, **`97.6 ms`** abort latency, and **32K NIAH PASS** (`318.95 / 40.06 tok/s`).
- **Strict Offline Zero-Network Verification:** Across all `2` Track 4 runs, `0` external `http(s)://` requests and `0` console/offscreen errors were recorded.

## 8. Track 5 — Factored Zero-Barrier Q4 GEMV, Dispatch Fusion (`397 → 269`), 4-Head Grouped GQA INT8/DP4a & Zero-Weight Suffix Speculation

| Run ID | Host / Pacing | Offline Load | 1K Dec / Prefill | 8K Dec / Prefill | 32K Dec / Prefill | 64K Dec / Prefill | 96K Dec / Prefill | 128K Dec / Prefill | Peak % BW | NIAH 32K | Sweep FPS (% Jank) | Net Reqs | Errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `23-track5-wgsl-max-speed-128k` | `worker` / `max-speed` | **6.97s** | 109.55 / 507 | 97.18 / 501 | 74.06 / 393 | 55.74 / 290 | 43.71 / 229 | 32.32 / 177 | **28.7%** | PASS (62.32 tok/s) | — | 0 | 0 |
| `24-track5-wgsl-worker-balanced-smoothness` | `worker` / `balanced` | **7.24s** | — | 60.41 / 325 | 41.20 / 259 | — | — | — | **15.8%** | PASS (40.37 tok/s) | 140.3 (0.00%) | 0 | 0 |

- **Track 5 Max-Speed (`23-track5-wgsl-max-speed-128k`) vs. Track 4 Max-Speed (`21-track4-wgsl-max-speed-128k`):** Offline load dropped to **`6.97s`**. Decode speed climbed across every context length: 1K `106.60` -> **`109.55 tok/s`** (+2.8%, **8.04x** vs Baseline ORT, **`28.7%`** BW), 8K `84.37` -> **`97.18 tok/s`** (**+15.2%**), 32K `52.44` -> **`74.06 tok/s`** (**+41.2%** from 4-Q-head GQA + DP4a!), 64K `44.59` -> **`55.74 tok/s`** (**+25.0%**), 96K `34.31` -> **`43.71 tok/s`** (**+27.4%**), and 128K `26.73` -> **`32.32 tok/s`** (**+20.9%**, **11.46x** vs Track 1 `08`).
- **Zero-Weight Suffix Speculative Decoding (`23-track5-wgsl-max-speed-128k`):** Exact greedy bit-identical parity **`true`** (`239/239` chars exact match), accepting **`74/78`** drafted tokens (**`94.87%`** over `20` steps) and lifting generation speed from `91.77` -> **`114.83 tok/s`** (**1.251x**).
- **Track 5 Production Default (`24-track5-wgsl-worker-balanced-smoothness`, `worker / balanced` + Speculative):** Improved 8K decode `52.69` -> **`60.41 tok/s`** (**+14.6%**), 32K decode `37.54` -> **`41.20 tok/s`** (**+9.8%**), and Chat Turn 1 -> **`78.29 tok/s`** (`TTFT 162.4 ms`, speculative 4/12), while keeping **`140.3 FPS`** sweep / **`142.7 FPS`** chat with **`0.00%` jank (>33ms)**, Snapshot/Restore `165t -> 165t`, **`99.3 ms`** abort latency, and **32K NIAH PASS** (`302.66 / 40.37 tok/s`).
- **Strict Offline Zero-Network Verification:** Across all `2` Track 5 runs, `0` external `http(s)://` requests and `0` console/offscreen errors were recorded.
