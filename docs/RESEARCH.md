# TinyGPT Frontier Research: Hyper-Optimizing In-Browser WebGPU Inference & Training a Specialized Browser-Agent Tiny Model

> **Scope & Grounding:** This research synthesis maps frontier inference systems (**Strata**, **magnitude.dev**, **exo**, **ktransformers**, **rapid-mlx**), WebGPU hardware proposals, speculative decoding, extreme quantization, and specialized small-model training directly onto TinyGPT's Chrome MV3 native Rust WASM + WGSL engine. All TinyGPT baseline numbers and architectural details are grounded directly in [REPORT.md](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/REPORT.md), [lib.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs), [wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts), [gpuPacing.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gpuPacing.ts), [onnxWeightLoader.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/onnxWeightLoader.ts), [wgslWebgpu.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslWebgpu.ts), [webgpuLimits.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/webgpuLimits.ts), and [engineWorker.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/engineWorker.ts).

---

## 1. Executive Summary & Local Baseline Diagnosis

TinyGPT has already cleared the hardest browser engineering hurdle: replacing ONNX Runtime Web's `13.62–17.61 tok/s` (`3.89%–5.02%` bandwidth utilization) generic graph executor with a custom `no_std` Rust WASM + native WGSL engine ([lib.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs)) running **real Qwen3.5-4B** weights in a Chrome MV3 Offscreen Worker at **`73.91 tok/s`** decode (1K, INT8 KV), **`23.80 tok/s`** at **128K context**, **`383–392 tok/s`** 1K prefill, **`4.71s`** OPFS cold reload, and **`144.0 FPS`** UI smoothness (`0.00%` jank >33 ms in `worker / balanced` mode, [REPORT.md](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/REPORT.md#L76-L100)).

Yet our best real-weight run (`16-wgsl-real-int8kv-128k`) utilizes only **`19.4%` (`181.2 GB/s`)** of the RTX 3090's `936 GB/s` theoretical memory bandwidth, and prefill (`383–392 tok/s`) is actually **2.5× slower** than Track 1's best ORT WebGPU prefill (`999 tok/s` in `07-ort-chunk-2048`). Direct inspection of [lib.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs) and [wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts) pinpoints five concrete structural bottlenecks:

| Baseline Bottleneck | Exact Root Cause in TinyGPT Codebase | Impact on Current Metrics | Frontier Remedy |
| :--- | :--- | :--- | :--- |
| **1. AoS Q4 Layout, 32-Thread Tail & `f32` Activations** | [WGSL_Q4_SUBGROUPS_F16](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L269-L390) uses `@workgroup_size(32)` looping `b += 32` over `80` blocks (16 threads run 3 blocks, 16 run 2 → warp tail idle), reads 20-byte strided AoS blocks (`5 u32`s), and loads `array<f32>` activations with 32 `f16()` casts/block. | Caps decode bandwidth at **19.4%** (`181.2 GB/s`) on RTX 3090. | SoA / Marlin-style `vec4<u32>` coalesced Q4 + native `vec2<f16>` / `f16` storage activations + 64/128-thread workgroups. |
| **2. Dispatch Fragmentation (397 Dispatches/Token)** | [buildStaticDecodePlan](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L712-L823) emits `12` dispatches/DeltaNet layer (`×24`), `13` dispatches/Full-Attn layer (`×8`), plus `embed`, `finalNorm`, `lmHead`, and 2-pass `argmax`. Tiny `wAb` (`64×2560`, 16 workgroups) and `wK`/`wV` (`1024×2560`) starve SMs. | High driver/barrier overhead; `normBuf` & `workBuf` (`f32`) reloaded repeatedly. | Fuse `QKV+Z+AB`, `QG+K+V`, `Gate+Up+SwiGLU`, and epilogue residual adds (`397 → ~170–190` dispatches). |
| **3. Token-Sequential DeltaNet Global State Prefill** | [WGSL_GDN_RECURRENT](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L1304-L1402) launches only `32` workgroups (`wid.x = 0..31`) and loops `t = 0..num_tokens` in global VRAM, reading & writing the `2 MiB/layer` state matrix (`dnStateBuf`) twice per token (`6.14 GiB` state traffic per 64-token chunk across 24 layers). | Caps Track 2 prefill at **`383–392 tok/s`** (2.5× below ORT `999 tok/s`). | **Chunkwise WY representation** (`fla-org/flash-linear-attention`) + register/shared-memory state tiling across `M`-token chunks. |
| **4. Scalar-`dot` `32×32` Prefill GEMM** | [WGSL_Q4_GEMM_F16](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L1108-L1248) tiles `32×32×32` with 64 threads and scalar `vec4<f16>` `dot()` rather than cooperative matrix hardware or 64×64 MLX-style tiles. | Prefill GEMMs are compute/SMEM-bound instead of Tensor-Core-bound. | `chromium_experimental_subgroup_matrix` (`subgroupMatrixMultiplyAccumulate`) + fused Q4 dequant-in-registers. |
| **5. Static Pacing Sleeps & Sync `mapAsync` Bubbles** | [decodeStep](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L1171-L1257) stalls every token on `staging4B.mapAsync(GPU_MAP_READ)`, and `balanced` mode ([gpuPacing.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gpuPacing.ts#L43-L59)) sleeps `2–6 ms` after every token + splits at ≥16K (`36.99 tok/s` vs `56.93 tok/s` in `worker/max-speed`). | Leaves `~35–50%` of decode wall-clock time idle in `balanced` mode; zero speculative batching. | Zero-weight speculative decoding (`batch=4–8` verification), `timestamp-query` closed-loop pacing, and async double-buffered readback. |

### Key Strategic Takeaways

1. **Today (Engine-Only, Zero Weight Changes):** Fixing DeltaNet chunkwise/cached prefill, fusing `QKV+Z+AB` / `Gate+Up+SwiGLU`, moving activations to `f16`, switching to coalesced SoA Q4 + `packed_4x8_integer_dot_product` (`dot4I8Packed`) for INT8 KV, and enabling **zero-weight Suffix/Prompt-Lookup Speculative Decoding** (`batch=4–8` verification rides *for free* inside our unused `80.6%` memory bandwidth headroom) can realistically push TinyGPT from **`73.9 → 180–280+ effective tok/s`** decode and **`386 → 1,500–2,500+ tok/s`** prefill in Chromium.
2. **Near-Term (Compression + Co-Designed Quantization):** Qwen3.5-4B currently consumes **`~2,507 MiB`** of weights across `slab0` (`1,062.5 MiB`), `slab1` (`1,062.5 MiB`), `slab2` (`378.9 MiB` for `248,320×2,560` tied `lm_head`), and `aux.bin` (`~3.4 MiB`). Because **only 8 of 32 layers** use Full GQA Attention (`layers 3, 7, 11, 15, 19, 23, 27, 31`), applying **SpinQuant/QuaRot** + mixed-precision **2–3 bit weights** on the 24 DeltaNet/MLP FFNs shrinks weights to **`~1.3–1.6 GiB`**, while **KIVI 2/4-bit KV + DuoAttention** cuts 128K KV cache from `~2.03 GiB` down to `<0.65 GiB`.
3. **Horizon (Purpose-Built Tiny Browser-Agent Model):** A `1.5B–2.5B` hybrid model trained/distilled specifically for browser DOM + screenshot-coordinate grounding (Magnitude/UI-TARS style), with a compact `32K–64K` tokenizer (cutting `slab2` from `379 MiB` to `<100 MiB`), **Minitron** structured pruning, **QAT matched to our WGSL block-32 Q4/2-bit/1.58-bit kernels**, **WASM XGrammar-2** constrained JSON/action decoding, and co-trained **MTP/EAGLE-3** draft heads, can deliver **300–500+ effective tok/s** inside a `<1.2 GiB` MV3 extension bundle.

---

## 2. TinyGPT Baseline Architecture Map (Local Grounding)

To ground every optimization directly in code we observed:

- **Model Geometry ([lib.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L10-L35), [onnxWeightLoader.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/onnxWeightLoader.ts#L5-L9)):**
  - `HIDDEN_SIZE = 2560`, `INTERMEDIATE_SIZE = 9216`, `NUM_LAYERS = 32`, `VOCAB_SIZE = 248320`.
  - **24 Gated DeltaNet Layers:** `DN_KEY_HEADS = 16`, `DN_KEY_DIM = 128`, `DN_VAL_HEADS = 32`, `DN_VAL_DIM = 128`, `DN_CONV_KERNEL = 4`, `DN_CONV_DIM = 8192`. Recurrent state: `24 × (8192×3 + 32×128×128) × 4 B = 50.25 MiB`.
  - **8 Full-Attention GQA Layers (`{3, 7, 11, 15, 19, 23, 27, 31}`):** `ATTN_HEADS = 16`, `ATTN_KV_HEADS = 4`, `ATTN_HEAD_DIM = 256`, partial NeoX RoPE (`rotary_dim = 64`), QK-RMSNorm (`256`), sigmoid output gate (`q_proj` is `8192×2560` interleaving Q and gate).
  - **KV Cache ([lib.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L91-L100)):** `fp16` uses `128 u32`s/token-head (`32 KiB/tok` across 8 layers; `2.00 GiB` at 64K); `int8` uses `65 u32`s/token-head (`64 × 4-i8 u32` + `1` packed `2×f16` K/V scale word = `16.25 KiB/tok` across 8 layers; `~2.03 GiB` at 128K).
- **Storage & Sharding ([wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L280-L340), [onnxWeightLoader.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/onnxWeightLoader.ts#L454-L528)):**
  - `slab0` (layers 0–15: `1,114,112,000 B` = `1,062.5 MiB`), `slab1` (layers 16–31: `1,062.5 MiB`), `slab2` (`lm_head`: `397,312,000 B` = `378.91 MiB`), `aux.bin` (`~3.4 MiB` FP32 norms, `inv_freq`, DeltaNet `A_neg_exp`, `dt_bias`, `gdn_norm`, `conv1d`). Total weights: **`~2,507 MiB` (`2.45 GiB`)**, sharded ≤128 MiB to stay under Dawn's `maxBufferSize` (`2.00 GiB` granted via [webgpuLimits.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/webgpuLimits.ts)) and bound Dawn staging memory.
- **Worker & Pacing ([engineWorker.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/engineWorker.ts), [gpuPacing.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gpuPacing.ts)):**
  - Runs inside a dedicated Web Worker in the MV3 Offscreen Document.
  - `balanced` pacing uses `prefillChunkShort=32`, `prefillChunkLong=16` (≥16K), `prefillBurstBudgetMs=8` / `prefillBreatherMs=4`, `decodeSplitsLong=2` (≥16K or ≥11 ms), `decodePostStepRatio=0.22` (`min 2 ms`, `max 6 ms`).

---

## 3. Reference Project Deep Dives & Transferable Patterns

### 3.1 Strata (Disambiguating 3 Distinct Projects)

Multiple prominent projects share the name **Strata** in modern LLM/agent systems; all three contribute orthogonal ideas to TinyGPT:

1. **`Niko1221/Strata` — Consumer Local MoE Inference Engine**
   - **Verified Links:** [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata), [MCP_SERVER.md](https://github.com/Niko1221/Strata/blob/main/docs/MCP_SERVER.md), [MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)
   - **Architecture:** Local-first runtime targeting consumer GPUs (NVIDIA RTX 20/30/40/50 and AMD Radeon 12 GB+ VRAM) to serve large MoE models (e.g., *Qwen3.8-Flash-Next*, *DeepSeek-V4.1-Flash*) via **expert-tiering** (hot experts pinned in VRAM, cold experts in system RAM/NVMe), **MTP (Multi-Token Prediction)** speculative ("guess-and-check") decoding, chunked prefill, local OpenAI/Anthropic-compatible HTTP APIs, a localhost browser UI with GPU telemetry, and a Python MCP server (`terminal/tools/strata_mcp.py`) for agentic control.
2. **Strata Paper (Stanford / NVIDIA / SJTU / CMU / CU Boulder, OSDI '26) — Hierarchical Context Caching**
   - **Verified Link:** [arXiv:2508.18572](https://arxiv.org/abs/2508.18572) (*Strata: Hierarchical Context Caching for Long Context Language Model Serving*)
   - **Architecture:** Implemented inside SGLang to solve long-context KV cache spillover across GPU HBM → CPU DRAM → SSD. Introduces **GPU-assisted I/O** (coalescing fragmented KV blocks on-GPU so host/disk DMA transfers saturate PCIe/NVMe bandwidth rather than issuing thousands of tiny uncoalesced copies) and a **cache-aware scheduler** that overlaps KV loading with ongoing compute to hide prefix-reload latency.
3. **`Klavis-AI/strata` (`Klavis-AI/klavis`) — Progressive MCP Tool Discovery**
   - **Verified Link:** Search-verified under `Klavis-AI/strata` / `Klavis-AI/klavis` (note: public GitHub repository was subsequently archived/deprecated as the team pivoted to RL environments/data, while hosted services remained active).
   - **Architecture:** Meta-MCP router solving **Tool Overload / Context Overload** in agents. Instead of dumping hundreds of tool schemas into the system prompt (bloating TTFT and degrading tool-selection accuracy), Strata exposes **progressive tool discovery**—routing intent → category → specific tool schema on demand.

**What TinyGPT Should Steal from Strata:**
- **From `Niko1221/Strata`:** MTP-style "guess-and-check" speculative decoding on consumer GPUs and exposing the local MV3 extension engine as a browser-native tool/agent endpoint with real-time GPU bandwidth telemetry.
- **From *Strata* (`arXiv:2508.18572`):** **GPU-assisted coalesced snapshot I/O.** Look at [snapshotState](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L847-L892): today TinyGPT runs `8 layers × 4 heads × 2 planes = 64` separate `copyBufferToBuffer` + `mapAsync` round-trips per snapshot! Packing active KV slices + DeltaNet state on-GPU via a tiny WGSL gather kernel into **one contiguous staging buffer** before a single `mapAsync` / OPFS write (or keeping an LRU ring of recent prefix states in OPFS) will eliminate 64× synchronization stalls and make multi-tab context switching near-instant.
- **From `Klavis-AI/strata`:** **Progressive tool/DOM schema disclosure** for our browser agent—never stuff an entire 50-tool MCP catalog or raw 40K-token HTML page into Qwen3.5-4B's prompt; progressively disclose only the active domain/page-type actions and distilled interactive subtree.

---

### 3.2 magnitude.dev (Vision-First Browser Agent & Local Inference Pairing)

- **Verified Repositories & Architecture:** `magnitudedev/browser-agent` (`magnitude-test` / `create-magnitude-app`) and `magnitudedev/magnitude` (local hardware-optimized inference engine); see [github.com/magnitudedev/browser-agent](https://github.com/magnitudedev/browser-agent) (verified via search).
- **Core Methodology:**
  - **Vision-First Coordinate Grounding vs. Brittle DOM Selectors:** Traditional Playwright/Selenium agents break on obfuscated Tailwind/CSS-in-JS classes, dynamic SPAs, `<canvas>`/WebGL surfaces, charts, drag-and-drop Kanban boards, and nested `<iframe>`s. Magnitude takes viewport screenshots and relies on visually grounded VLMs (e.g., Claude Sonnet, Qwen-2.5-VL, or lightweight local *Moondream 2B*) to emit exact `(x, y)` pixel coordinates for `agent.act()`, paired with Zod-typed `agent.extract()` and natural-language visual assertions (`.step(...).check(...)`).
  - **Planner vs. Executor Separation:** Separates high-level multi-step reasoning (*Planner*) from fine-grained coordinate/click/drag synthesis (*Executor*).
  - **The Latency & Token Cost Bottleneck:** Sending high-resolution screenshots + page context to cloud VLMs on every action + `.check()` loop incurs huge token bills, multi-second round-trip latency, and privacy exposure—driving the Magnitude team to build `magnitudedev/magnitude` for local GPU/RAM/SSD execution. Even with small local VLMs like Moondream 2B, pure-vision grounding struggles with misclick precision on dense UIs.

**What TinyGPT Should Steal from magnitude.dev:**
- **Hybrid Accessibility-Tree + Set-of-Mark / Crop Grounding inside the Extension:** Unlike an external Playwright process, a Chrome MV3 extension sits **inside the browser** with zero-latency access to the live DOM, `getBoundingClientRect()`, `chrome.debugger` / `Accessibility` tree, and `captureVisibleTab()`. TinyGPT should **not** burn vision tokens on full-page OCR when the DOM/a11y tree already exposes crisp interactive elements (`role`, `aria-label`, `bbox`). Instead:
  1. **Fast Path (90% of steps):** Distill interactive elements into compact `[id | role | name | (cx,cy)]` descriptors (~300–800 tokens instead of 2,000+ vision patch tokens) so Qwen3.5-4B executes directly in `<150 ms`!
  2. **Visual Fallback Path (Canvas/SVG/Unlabeled Icons):** Only crop the ambiguous viewport region or run a tiny local grounding head / distilled vision-coordinate projector when DOM accessibility info is missing.
- **Planner/Executor Split + `.check()` Verification Loop:** Use TinyGPT locally as the **sub-second Executor & `.check()` Verifier** (checking post-action DOM/viewport state diffs and extracting Zod-validated JSON via WASM grammar decoding), optionally escalating only genuinely stuck high-level replanning to a cloud frontier model.

---

### 3.3 exo (`exo-explore/exo` — P2P Heterogeneous Inference)

- **Verified Repository:** [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
- **Core Methodology:**
  - **Masterless P2P Clustering:** Automatic peer discovery exposing OpenAI/Claude/Ollama-compatible APIs without a centralized coordinator.
  - **Ring Memory-Weighted Partitioning:** Inventories free RAM/VRAM across heterogeneous peers (Macs, Linux/NVIDIA PCs, mobile/edge devices) and assigns contiguous transformer layer slices proportional to each device's memory capacity.
  - **Pipeline vs. Tensor Parallelism:** Pipeline parallelism pools memory across machines (sequential layer handoff); Tensor parallelism shards matmuls within each layer (`mlx.distributed` over Thunderbolt/fast LAN) to cut single-request decode latency.
  - **Multi-Backend Interop:** Combines **MLX** (Apple Silicon unified memory, 4-bit weights), **Tinygrad** (Linux/NVIDIA GPUs), and browser/edge **WebGPU** nodes with dynamic cross-engine tensor serialization.

**What TinyGPT Should Steal & Evaluate:**
- **Honest Latency Feasibility Check:** WebRTC DataChannels / LAN WebSockets have `0.5–5 ms` RTT per hop—**Tensor Parallelism** (2 AllReduces per layer × 32 layers = 64 syncs/token = `32–320 ms` network overhead/token) is **non-viable** over browser networking and would crush TinyGPT's `73.9 tok/s` (`13.5 ms/token`).
- **Where `exo` *Does* Transfer to TinyGPT:**
  1. **Browser-as-Client + Local Companion Offload (Pipeline / Speculative Offload):** Keep the fast `~1.5 GB` 4B (or 2B) **Draft/Executor model in-browser WebGPU** for instant `<50 ms` UI responses, and optionally pair via `localhost` WebSocket/WebTransport with an `exo` / `Strata` / `Rapid-MLX` local companion node hosting a 32B–120B Planner when the user has a desktop/Mac mini on LAN.
  2. **Multi-GPU / Multi-Tab Pipeline Partitioning (`slab0` vs `slab1`):** TinyGPT already splits layers 0–15 (`slab0`) and 16–31 (`slab1`) cleanly (`1,062.5 MiB` each) in [wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L635-L647)! That same clean layer-boundary abstraction supports **memory-weighted splitting** across a discrete GPU (`requestAdapter({ powerPreference: 'high-performance' })`) and a secondary/integrated adapter or WASM-SIMD CPU fallback on low-VRAM (4 GB) laptops—holding 4 Full-Attn + 12 DeltaNet layers on GPU and streaming/running cold layers only if VRAM is exhausted.

---

### 3.4 ktransformers (`kvcache-ai/ktransformers`)

- **Verified Repository:** [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) (MADSys Lab, Tsinghua University; also Marlin [arXiv:2408.11743](https://arxiv.org/abs/2408.11743))
- **Core Methodology:**
  - **Template-Based Kernel Injection:** Uses declarative YAML/Python rules to swap standard PyTorch `nn.Linear`/attention/expert modules at runtime with fused quantized operators without rewriting the whole model graph.
  - **Zero-Dequantization & Marlin FP16×INT4 Kernels:** Marlin (`IST-DASLab/marlin`) reorganizes 4-bit weights into an offline-permuted, `16×64` Tensor-Core-aligned layout with asynchronous global-to-shared (`cp.async`) double-buffering and vectorized `int4 -> fp16` bitwise register unpacking (via fast bit-manipulation rather than scalar shifts), achieving near-ideal `~4×` memory-bandwidth saturation.
  - **Heterogeneous CPU-GPU Execution & Unbroken Graph Capture:** Keeps dense attention/KV in VRAM while running sparse MoE experts on CPU AVX-512/AMX (`kt-kernel`). Crucially, to prevent host callbacks from breaking **CUDA Graphs**, `ktransformers` wraps CPU submissions inside `cudaLaunchHostFunc`, capturing the entire decode pass into a **single unbroken graph** (delivering up to **1.23×** extra decode speedup purely by eliminating kernel launch overhead).

**What TinyGPT Should Steal from ktransformers:**
1. **Offline-Permuted Marlin Weight Layout inside `repack_q4_rows`:** Look at [repack_q4_rows](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L2003-L2037) in Rust WASM—we *already* repack ONNX weights offline once before saving to OPFS! Instead of copying ONNX's AoS block-32 layout verbatim (where `u32[4]` scale/zero sits inline with nibbles, forcing 20-byte strided reads and 8 scalar bit-shifts per word), `repack_q4_rows` can interleave/transpose blocks into **SoA coalesced `vec4<u32>` strips** (16 bytes/thread = 128-bit load) with scales/zeros packed in a separate aligned header array.
2. **Single-Pass Unbroken CommandBuffer (`max-speed` / Pre-Recorded Bundle):** In WebGPU, `GPURenderBundle` exists for render, while for compute we have [buildStaticDecodePlan](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L712-L823). Currently `decodeStep()` loops through 397 JS calls (`pass.setPipeline`, `pass.setBindGroup`, `pass.dispatchWorkgroups`) on every single token, and in `balanced` / `long` mode splits across multiple `submit()` calls. Minimizing pipeline switches (via fused kernels) + `dispatchWorkgroupsIndirect` + WASM-side command encoding or persistent-kernel loops cuts JS/Dawn command-encoding overhead dramatically.
3. **Declarative Kernel Injection for Precision Tiering:** Adopt a lightweight JSON/TS layer-policy map (like `ktransformers`'s YAML injection rules) so TinyGPT can dispatch `Q2`/`Q3` kernels on insensitive MLP layers and `Q4` on Full-Attn layers without duplicating engine orchestration.

---

### 3.5 rapid-mlx (`raullenchai/Rapid-MLX`) & Apple MLX (`ml-explore/mlx`)

- **Verified Repositories:** `raullenchai/Rapid-MLX` ([rapidmlx.com](https://rapidmlx.com)), `ml-explore/mlx` (verified via search).
- **Core Methodology:**
  - **Pure Native Metal Kernels (No `llama.cpp` Wrapper):** `Rapid-MLX` achieves 2–4× speedups over generic runners by staying 100% native in MLX Metal kernels, mirroring TinyGPT's Track 2 philosophy (dropping ONNX Runtime Web for native Rust WASM + WGSL).
  - **MLX `qmv_fast` / `affine_quantized_matmul`:** Uses group-wise affine quantization (`w * scale + bias`, matching TinyGPT's `fma(q, scale, zero)`!), wide threadgroup tiling (`64×64×64` / `128×64×128` NAX configs), vectorized `ulong`/`uint4` loads, and **fused `RMSNorm` + `RoPE` + `SwiGLU`** kernels that keep intermediates in registers/threadgroup memory rather than round-tripping through global DRAM.
  - **Radix + Recurrent-State Prompt Caching (`~0.08s` TTFT):** Caches both KV prefixes (radix tree) and **DeltaNet/recurrent state snapshots** so repeated agent system prompts, tool schemas, and DOM headers skip prefill entirely.
  - **Live INT4/INT8 KV Quantization (`--kv-cache-quantization`), Continuous Batching, MTP (`--enable-mtp`), & 17 Agent Tool Parsers:** Built explicitly to serve local coding/browser agents with zero wasted TTFT and strict tool-call extraction.

**How UMA / Metal Kernel Tricks Translate to Chromium WebGPU:**
- **What transfers directly:**
  - **Fused Epilogues & `f16` Register Chains:** Just as Metal fuses `RMSNorm → QKV` or `Gate+Up → SwiGLU` in registers/threadgroup memory, WGSL can keep `norm_x` in `var<workgroup> sh_x: array<vec4<f16>, 640>` (`2560 × 2B = 5 KiB`—easily fits in WebGPU's `16–32 KiB` guaranteed `maxComputeWorkgroupStorageSize`!) or pass `f16` directly via packed `u32` storage buffers.
  - **Radix + DeltaNet State Prefix Caching:** TinyGPT *already* proved `snapshotState` / `restoreState` (`147t -> 147t`, `165t -> 165t`) in [REPORT.md](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/REPORT.md#L74-L99)! Turning that into an automatic **multi-slot prefix cache** (system prompt + browser-agent instructions + persistent site DOM skeleton) cuts Turn-2+ TTFT from `325 ms` down to **`<30 ms`**.
- **What does *not* transfer 1:1 (UMA vs. Discrete PCIe/Browser Sandbox):**
  - On Apple Silicon UMA, CPU and GPU share physical DRAM with zero-copy pointers. In Chromium WebGPU (especially on discrete GPUs like RTX 3090), `writeBuffer` and `mapAsync` cross IPC + PCIe/staging shared memory! Therefore, TinyGPT must avoid `Rapid-MLX`-style CPU-side token sampling/parsing round-trips by keeping **argmax + greedy/top-k + speculative token acceptance + DeltaNet state snapshots strictly GPU-resident**, touching CPU only for final 4-byte token IDs (or small accepted token batches).

---

## 4. Deep Technical Synthesis for TinyGPT

### 4.1 Track A: WebGPU Compute Kernels & Spec Frontier

#### A1. `chromium_experimental_subgroup_matrix` (Tensor Cores in WGSL)
- **Spec Status:** Experimental Dawn/Chromium feature (`chromium-experimental-subgroup-matrix`, modeled on `VK_KHR_cooperative_matrix` / Metal `simdgroup_matrix`) exposing `subgroup_matrix_left<f16, K, M>`, `subgroup_matrix_right<f16, N, K>`, `subgroup_matrix_result<f32, N, M>`, and `subgroupMatrixMultiplyAccumulate`.
- **TinyGPT Target:** Replace [WGSL_Q4_GEMM_F16](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L1108-L1248)'s scalar `32×32` `dot()` loop during **prefill (`M = 16..64`)** and **speculative verification (`M = 4..16`)**.
- **Design:** Tile `M×N×K = 32×64×32` (or `16×64×32` for small speculative batches `M = 4..8`), dequantize block-32 Q4 weights into `workgroup` `f16` tiles (or directly into `subgroup_matrix` operands where tile dimensions match, e.g. `16×16×16` on NVIDIA Ampere/Ada), and accumulate in `f32`.
- **Expected Gain:** **3–5×** speedup on prefill GEMMs (`wGate`, `wUp`, `wDown`, `wQkv`, `wOut`) and nearly free `M=4–8` speculative verification. Fallback path remains our `vec4<f16>` GEMM when `chromium-experimental-subgroup-matrix` is absent.

#### A2. `packed_4x8_integer_dot_product` (DP4a: `dot4I8Packed` / `dot4U8Packed`)
- **Spec Status:** Standardized WGSL language extension (`requires packed_4x8_integer_dot_product;`) shipping in modern Chrome/Dawn.
- **TinyGPT Target #1 — INT8 KV Cache Attention (`WGSL_GQA_INT8_SPLIT` & `WGSL_GQA_INT8_PREFILL`):**
  Look at [WGSL_GQA_INT8_SPLIT](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L679-L695): today, for **every historical token `t`**, every thread calls `unpack4_i8(kv_cache[k_src + tid])` (4 `extractBits` + 4 `i32->f32` casts + 1 `f32` multiply) and computes `dot(q_vec, ki)` in FP32, followed by a `workgroupBarrier()` on *every single token*.
  - **DP4a Upgrade:** Quantize `Q` once at thread entry into packed INT8 (`pack4xI8`) with a single per-head `q_scale` in registers, then compute `dot4I8Packed(q_i8_packed, kv_cache[k_src + tid])` in **1 hardware DP4a instruction** per 4 dims (eliminating 4 `extractBits` + 4 `f32` casts + FP32 FMAs), and unroll `t` by 4–8 tokens per barrier!
- **TinyGPT Target #2 — W4A8 Integer-Quantized Decode GEMV:** Quantize the input activation vector `x` (`2560` or `9216` elements) to INT8 once per block-32 (1 `u32` scale + 8 `u32`s of packed `i8`), pack Q4 weights as `i8` offsets in registers or `0..15` unsigned nibbles, and execute `dot4I8Packed` / `dot4U8Packed`—doubling ALU throughput on integrated GPUs and older architectures lacking fast `f16` dual-issue.

#### A3. Direct `f16` Storage Activations (`storage_16bit` / Packed `vec2<f16>`)
- **Current Bottleneck:** [wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L480-L500) allocates `hiddenBuf`, `normBuf`, `workBuf`, `attnOutBuf`, and `logitsBuf` as `f32`. Every Q4 block in [WGSL_Q4_SUBGROUPS_F16](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L341-L349) executes `32` `f16(x[kb + i])` conversions and reads `128 B` instead of `64 B`.
- **Fix:** Store `normBuf`, `workBuf`, and `attnOutBuf` as packed `vec2<f16>` (`array<u32>`, or `array<f16>` when `shader-f16` storage buffers validate cleanly in Tint/Dawn, keeping `hiddenBuf` residual stream in `f32` or `f16` with verified accuracy). A single `vec4<u32>` load fetches **8 `f16` activations in 1 instruction** with `unpack2x16float` / `bitcast<vec2<f16>>`—zero `f32->f16` casts and 50% lower activation VRAM bandwidth.

#### A4. Coalesced SoA Block-32 Q4 Layout & Warp-Balanced GEMV
- **Current Bottleneck in [WGSL_Q4_SUBGROUPS_F16](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L314-L360):**
  1. `workgroup_size(32)` with `for (b = tid; b < blocks; b += 32)`: for `K = 2560`, `blocks = 80`. Lanes `0..15` compute 3 blocks; lanes `16..31` compute 2 blocks and sit **33% idle** on the 3rd iteration.
  2. `row_stride = blocks * 5`: adjacent threads read `b*5` words apart (`20 B` stride), breaking 128-byte coalesced cache-line reads, and interleave metadata (`u32[4]`) inside the nibble stream.
- **Marlin/MLX-Inspired Upgrade in [repack_q4_rows](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L2003-L2037):**
  - Separate nibble data from `scale/zero` metadata (SoA) or pack groups of `4` blocks (`128` weights = `16 u32`s of nibbles + `4 u32`s of metadata) so threads load contiguous **`vec4<u32>`** (16 bytes/lane = 512 bytes/warp = 4 exact 128B cache lines!).
  - Use `@workgroup_size(64)` or `@workgroup_size(128)` (or `80`-aligned tile splitting / 2 warps per 4–8 rows) with vectorized bitmask dequantization (`(w >> 0) & 0x0F0F0F0Fu` style SIMD-within-a-register nibble extraction) and **no `workgroupBarrier()` when `sg_size == 32`** (single-subgroup reduction uses pure `subgroupAdd`!).

#### A5. Kernel Fusion Blueprint (`397 → ~175` Dispatches per Decode Token)
Directly mapped to [buildStaticDecodePlan](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L748-L815):

| Current Unfused Sequence | Fused Replacement | Dispatches Saved / Token | Memory Traffic Eliminated |
| :--- | :--- | :---: | :--- |
| `wQkv` + `wZ` + `wAb` (24 DeltaNet layers) | Single `QKVZAB` (`12288×2560`) GEMV/GEMM | **48** | Reads `normBuf` (`2560`) 1× instead of 3×; eliminates tiny `64×2560` under-occupancy kernel |
| `wQg` + `wK` + `wV` (8 Full-Attn layers) | Single `QGK_V` (`10240×2560`) GEMV/GEMM | **16** | Reads `normBuf` 1× instead of 3×; eliminates `1024×2560` under-occupancy |
| `wGate` + `wUp` + `swigluPipe` (32 layers) | Fused `GateUpSwiGLU` (or `GateUp` interleaved rows + epilogue SiLU×Up) | **64** | Eliminates `workBuf` write+read of `2 × 9216 × 4 B = 73.7 KB/layer` (`2.36 MB/tok`) |
| `rmsNorm` before projections (64× + final) | Inline `rmsNorm` into `workgroup` memory or fuse with previous `add_residual` | **65** | Eliminates `hiddenBuf → normBuf` round-trip (`65 × 10 KB` + 65 barriers) |
| `gqaSplit` + `gqaReduce` (when `seqPos ≤ 1024`) | Single-pass GQA when `num_splits == 1` | **8** | Eliminates `gqaSplitBuf` round-trip at short context |
| **Total** | | **~201 dispatches saved (~50% reduction!)** | **Significant driver/barrier & activation BW savings** |

#### A6. Prefill DeltaNet: Chunkwise WY Representation (`fla-org/flash-linear-attention`)
- **The Problem in [WGSL_GDN_RECURRENT](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L1304-L1402):** For `M=64` prefill tokens, `WGSL_GDN_RECURRENT` dispatches only `32` workgroups (`v_head = 0..31`) and executes `for (t = 0; t < M; t++)` where each iteration reads `dn_state[128×128]` (`64 KiB/head`) twice and writes it once. Across 24 layers × 64 tokens, that is **`~6.14 GiB` of global VRAM state R/W per 64-token chunk** while leaving >60% of RTX 3090's 82 SMs idle!
- **Two-Stage Upgrade:**
  1. **Immediate Win (Register-Cached Multi-Token Recurrence):** Each workgroup still handles 1 value head (`128` threads, `j = 0..127`), or splits `128` columns across `4` workgroups of `32` threads (`32 × 4 = 128` workgroups across 32 heads -> full SM occupancy!), and keeps all `128` state rows `s[0..127]` in **per-thread registers** (`var s_col: array<f32, 128>` or `4×32` split across 4 warps = `32` floats/thread!) across the entire `M=32–64` chunk. `dnStateBuf` is read **once** at chunk start and written **once** at chunk end — cutting DeltaNet prefill state bandwidth by **64×**!
  2. **Frontier Win (Chunkwise WY Representation, [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet), [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)):** Factor the intra-chunk Householder/delta transitions inside a chunk of $C = 32$ or $64$ tokens into compact $W Y^T$ low-rank matrices, turning intra-chunk token updates into parallel `C×C` and `C×128` Tensor-Core / `subgroupMatrix` GEMMs and propagating state $S_t$ only across chunk boundaries.

#### A7. Zero-Overhead Dispatch & Closed-Loop Pacing
- **`dispatchWorkgroupsIndirect`:** For dynamic speculative tree sizes (`k` accepted/verified tokens), dynamic KV page lists (Quest), or GPU-side stop-token detection (`EOS` check inside `argmax_pass2`), write `(wx, wy, wz)` into an indirect buffer on-GPU so the CPU doesn't stall between steps.
- **Timestamp-Query Closed-Loop Pacing:** Replace static `2–6 ms` post-step sleeps in [gpuPacing.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gpuPacing.ts#L43-L59) with WebGPU `timestamp-query` measurements + `requestAnimationFrame` vsync alignment so `balanced` mode only yields when GPU frame time approaches the display's `1/refreshRate` budget (`6.94 ms` at `144 Hz`!).

---

### 4.2 Track B: Zero-Weight & Low-Weight Speculative Decoding in the Browser

On RTX 3090, single-token decode (`batch = 1`) uses only **`19.4%`** of memory bandwidth and **`<2%`** of FP16 Tensor Core FLOPs. Verifying **`K = 4–8` candidate tokens in parallel** via a `batch = K` GEMM reads the **`2.45 GiB` weight slabs ONCE**, meaning **verification latency for `K = 4–8` is within `~1.05×–1.15×` of a single `batch = 1` decode step**! Any accepted tokens yield near-linear throughput multiplication.

#### Zero-Weight Methods (Immediate, Zero Bundle Size Increase)
1. **Prompt Lookup Decoding (`[ngram]`):**
   - **Mechanism:** Match the last `n=2..4` generated tokens against the user prompt / context buffer in Rust WASM (`<0.05 ms` on CPU!) and propose the next `K=4..6` tokens verbatim.
   - **Why It Crushes in TinyGPT:** Browser agents constantly extract fields, copy URLs/titles/snippets, rewrite DOM text, and summarize tabs—workloads with `40%–70%` verbatim n-gram overlap.
2. **SuffixDecoding ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975), [snowflakedb/ArcticInference](https://github.com/snowflakedb/ArcticInference)):**
   - **Mechanism:** Build a compact suffix automaton / suffix tree in Rust WASM (`<2 MiB` RAM) over **both** the current DOM/prompt **and** prior agent turns/tool-call templates (JSON keys, `{"action":"click","element_id":`, Playwright-like schemas). Dynamically adjusts speculation length $K$ using empirical suffix acceptance probability.
   - **Why It Crushes for Browser Agents:** Agent outputs follow repetitive structured action templates + DOM identifiers, yielding `2.0×–3.0×` speedups with **0 extra VRAM**.
3. **Lookahead Decoding ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057), `hao-ai-lab/LookaheadDecoding`):**
   - Uses parallel Jacobi-style n-gram branches verified simultaneously inside one causal mask; requires no draft model and works even on free-form generation where n-grams aren't in the prompt.

#### Low-Weight Methods (Next Step / Horizon)
- **EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840), [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)):** Fuses multi-layer target hidden states into a single lightweight draft transformer block (`~80–150 MB` in Q4) trained with "training-time test" multi-step rollout; achieves `3.0×–4.5×` acceptance lengths across chat and reasoning.
- **Medusa ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774), `FasterDecoding/Medusa`) & Native MTP (Multi-Token Prediction):** Adds `2–4` parallel prediction heads directly on top of the final hidden state (`Rapid-MLX --enable-mtp` and `Niko1221/Strata` both leverage native MTP on modern Qwen architectures). Note: because Qwen3.5-4B has a huge `248,320` vocab, naive Medusa heads (`3 × 248,320 × 2560` Q4 = `~1.1 GiB`!) would be too heavy unless pruned to a `32K` top-frequency vocab or sharing the tied `slab2` `lm_head` via a lightweight `[2560, 2560]` EAGLE adapter!

> **Critical Architectural Insight — Speculative Rollback in Hybrid Gated DeltaNet + GQA:**
> Unlike pure Transformers where rejecting draft tokens `t+3..t+5` simply decrements `seqPos` (leaving stale KV entries beyond `seqPos` harmlessly ignored), **Gated DeltaNet mutates `dnStateBuf` (`2 MiB/layer`) and `dnConvBuf` in-place!**
> To run `K = 4–8` speculative verification safely on Qwen3.5-4B without corrupting the 24 DeltaNet recurrent states:
> 1. **Double-Buffered Delta State Rollback (Recommended for `K ≤ 8`):** Before verifying `K` draft tokens in `WGSL_GDN_RECURRENT`, snapshot or read-from-`S_prev` and store intermediate per-token delta checkpoints `(S_prev, conv_prev)` only if needed, **or** compute the `K`-step chunk inside registers/shared memory, writing out **only the `accepted_count` state** (determined after the `batch=K` forward pass + GPU verification!). Since `K ≤ 8` is tiny, holding `S_prev` untouched in `dnStateBuf` until acceptance and committing only `accepted_count` steps (or keeping a tiny `K × 2 MiB` scratch or recomputing `accepted_count` steps from `S_prev` in `<0.15 ms`) guarantees **exact bit-identical** DeltaNet state with zero rollback corruption!

---

### 4.3 Track C: Extreme Quantization & Hybrid KV Compression

#### C1. Shrinking Weights from `2.45 GiB` → `~1.3–1.6 GiB`
In Qwen3.5-4B, weight bytes break down as:
- **24 DeltaNet layers:** `24 × 67.19 MiB = 1,612.5 MiB` (`64.3%` of weights; MLP `gate+up+down` alone is `24 × 42.19 MiB = 1,012.5 MiB`!).
- **8 Full-Attn layers:** `8 × 64.06 MiB = 512.5 MiB` (`20.4%`).
- **`lm_head` (`slab2`, 248,320 × 2560):** `378.9 MiB` (`15.1%`).

Because **DeltaNet/MLP FFN weights account for ~64% of total model size** and are substantially more tolerant to aggressive quantization than Full-Attention Q/K/V and `lm_head`, we can apply **mixed-precision quantization** without retraining:
1. **SpinQuant ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406), [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)) & QuaRot ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456), [spcl/QuaRot](https://github.com/spcl/QuaRot)):** Apply orthogonal Hadamard rotations ($H$) offline to eliminate activation and weight outliers (`W' = H W`, absorbing the inverse rotation into RMSNorm weights or doing a fast $O(D \log D)$ Walsh-Hadamard transform in WGSL shared memory on `2560` elements!). Outlier removal makes 2–3 bit weights and 4-bit KV caches dramatically cleaner.
2. **Mixed-Precision Tiering (`Q4` + `IQ3_XXS`/`Q3` + `IQ2_XS`/`LUT-GEMM`):**
   - Keep **8 Full-Attn layers + `lm_head` + `in_proj_ab`** at `Q4` (`~892 MiB`).
   - Compress **24 MLP `gate/up/down` (`~1.01 GiB` at Q4) and DeltaNet `qkv/z/out`** to **2.5–3 bit** (`~510–610 MiB` via block-32 3-bit / `IQ3_XXS` or **LUT-GEMM** [arXiv:2206.09557](https://arxiv.org/abs/2206.09557) / `IQ2_XS` importance-matrix quantization).
   - **Result:** Total weight footprint drops from **`2,507 MiB` → `~1,400–1,550 MiB` (`1.37–1.51 GiB`)**, cutting OPFS cold reload from `4.71s` to **`~2.8s`**, fitting comfortably inside **`maxBufferSize = 2.00 GiB`** even as a single slab, and boosting decode tok/s by **`~1.6×`** simply from reading fewer bytes per token!
3. **BitNet b1.58 ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764), [microsoft/BitNet](https://github.com/microsoft/BitNet)):** For the Horizon custom-trained model, ternary weights `{-1, 0, +1}` pack `5` trits per byte (`1.58 bits/wt`, or `2-bit` fast register decode) and replace floating-point FMA weight multiplies with conditional add/subtract (`select` / LUT-GEMM), shrinking a 4B model to **`<1.0 GiB`** and eliminating multiplier pressure on mobile/integrated WebGPU GPUs.

#### C2. Exploiting Qwen3.5's 8-Layer Full-Attn Footprint for Ultra-Compact KV Cache
Because Qwen3.5 only stores KV caches for **8 layers** (`4 KV heads × 256 dims`), our `INT8` KV cache (`~2.03 GiB` at 128K) is already 4× smaller than a pure 32-layer transformer. We can push it another **2×–4×** smaller and faster:
- **KIVI Asymmetric 2/4-Bit KV ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750), [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)):** Keys exhibit channel-wise outliers (quantize **per-channel** across a residual window of `32` tokens), while Values exhibit token-wise variation (quantize **per-token**). Dropping `WGSL_GQA_INT8_SPLIT` (`65 u32`s/tok-head) to **INT4 KV** (`33 u32`s/tok-head = `132 B`) halves 128K KV cache from `~2.03 GiB` to **`~1.03 GiB`** and doubles long-context GQA bandwidth!
- **DuoAttention ([arXiv:2410.10819](https://arxiv.org/abs/2410.10819), [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention)):** Across the `8 layers × 4 KV heads = 32` total KV heads in Qwen3.5-4B, identify which heads are *Retrieval Heads* (requiring full KV history) vs. *Streaming Heads* (needing only attention sinks + a 128–256 token sliding window). Converting even 50% of the 32 KV heads to streaming cuts 128K KV memory & attention scan bandwidth by **~2×** with near-zero NIAH loss.
- **Quest Query-Aware Page Sparsity ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) & SnapKV ([arXiv:2404.14469](https://arxiv.org/abs/2404.14469), [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)):** Store `(k_min, k_max)` metadata per 16-token or 32-token KV page (`256 × 2 × 2B = 1 KiB/page`) and let [WGSL_GQA_INT8_SPLIT](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L606-L767) skip loading pages whose upper-bound `Q · K_page` score falls below the Top-K threshold—turning 128K decode (`23.80 tok/s`) into near-constant **`~55–65 tok/s`**.

---

### 4.4 Track D (Horizon): Training & Fine-Tuning a Specialized Tiny Browser-Agent Model

Instead of squeezing a general-purpose `248K`-vocab `4B` chatbot into the browser, our Horizon goal is training/distilling a **`1.5B–2.5B` WebGPU-native Browser-Agent Foundation Model** co-designed with our WGSL kernels:

#### 1. Optimal Hybrid Architecture (Hardware-Co-Designed)
- **Linear:Full-Attention Mix:** Follow Qwen3.5's 3:1 ratio or **Gated DeltaNet-2** ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791), [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2), decoupling erase/write gates for sharper associative recall on long DOMs) combined with **Hymba** ([arXiv:2411.13676](https://arxiv.org/abs/2411.13676), [NVlabs/hymba](https://github.com/NVlabs/hymba)) learnable meta-tokens + cross-layer KV sharing, or **LFM2** double-gated convolutions + sparse GQA.
- **WebGPU-Aligned Dimensions & Pruned Vocabulary:**
  - Align all projection dims (`hidden=2048`, `intermediate=6144` or `7168`, `head_dim=128`) to exact multiples of `128` (eliminating `80/32` warp tail divergence in WGSL!).
  - Shrink `VOCAB_SIZE` from `248,320` (`378.9 MiB` in `slab2`) to **`32,768` or `49,152`** (English/code/HTML/JSON-focused BPE), reclaiming **`~300–330 MiB`** of VRAM and cutting `embed` + `lm_head` + 2-pass `argmax` cost by **5×–7.5×**!

#### 2. Compression & Kernel-Matched QAT
- **Minitron Structured Pruning + Distillation ([arXiv:2407.14679](https://arxiv.org/abs/2407.14679), [NVlabs/Minitron](https://github.com/NVlabs/Minitron)):** Prune width/depth/attention-heads from a strong `4B–9B` hybrid teacher and distill using both KL-divergence logits and intermediate hidden-state alignment.
- **Kernel-Exact QAT (Quantization-Aware Training):** Train with fake-quantization matching TinyGPT's *exact* block-32 asymmetric Q4 / 2-bit / BitNet b1.58 representation (`scale + zero` in `f16`) and SpinQuant/Hadamard rotation—eliminating post-training accuracy cliffs.

#### 3. Browser-Agent Datasets, Playwright GRPO RL & Zero-Overhead Constrained Decoding
- **Multimodal & Structural Grounding Corpus:** Synthesize trajectories combining **Agent-E** ([arXiv:2407.13032](https://arxiv.org/abs/2407.13032), `EmergenceAI/Agent-E`) distilled DOM / accessibility trees with **OS-Atlas** ([arXiv:2410.23218](https://arxiv.org/abs/2410.23218)), **SeeClick** ([arXiv:2401.10935](https://arxiv.org/abs/2401.10935)), **ShowUI** (`showlab/ShowUI`), **UI-TARS** ([arXiv:2501.12326](https://arxiv.org/abs/2501.12326), [bytedance/UI-TARS](https://github.com/bytedance/UI-TARS)), and **Magnitude-style** `(x, y)` coordinate + visual `.check()` trajectories.
- **Verifiable Playwright RL (WebRL + GRPO):** Following **WebRL** ([arXiv:2411.02337](https://arxiv.org/abs/2411.02337), `THUDM/WebRL`), run online self-evolving curriculum RL in headless Playwright sandboxes using **Group Relative Policy Optimization (GRPO)** with deterministic verifiable rewards (DOM mutation match, URL/network state assertion, Zod schema extraction accuracy, and minimal step count).
- **Zero-Overhead WASM Constrained Decoding (`XGrammar-2`):** Compile **XGrammar / XGrammar-2** ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100), [arXiv:2601.04426](https://arxiv.org/abs/2601.04426), [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)) into our Rust `wasm32` worker module. Precomputing token bitmasks in `<50 µs` on CPU while the GPU executes the forward pass guarantees **100% schema-valid browser tool calls** (`click`, `type`, `scroll`, Zod `extract`) with zero wasted tokens on syntax retries—and pairs with tag-triggered grammar switching for `<think>` → `<tool_call>` transitions!
- **Co-Trained EAGLE-3 / MTP Draft Heads:** Co-train 2–3 MTP or EAGLE-3 draft layers alongside the specialized model so structured browser-action generation achieves **3×–4×** speculative acceptance directly on WebGPU.

---

### 4.5 Silky-Smooth End-UX Latency Hiding (Zero Perceived Stall)

1. **Predictive Prefill on Hover / Focus / Tab Switch:** When the user hovers over the extension action button, focuses the omnibox/prompt input, or switches browser tabs, trigger background `prefillChunk` ([wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts#L927)) for the static system prompt + active tab's distilled accessibility skeleton. By the time the user finishes typing their command, TTFT is **`<30 ms`**.
2. **Structural DOM Delta Prefix Caching + GPU-Resident Ring Snapshots:** Instead of re-prefilling the whole page after each click (`agent.act()`), diff the accessibility tree in the content script, keep a prefix cache of the unchanging page header/navigation (inspired by `Rapid-MLX` radix + DeltaNet snapshots), and save/restore DeltaNet + KV checkpoints via **GPU-to-GPU buffer copies** (avoiding `snapshotState()`'s current CPU `mapAsync` round-trip).
3. **Streaming Speculative Action Preview:** As SuffixDecoding or EAGLE-3 streams the candidate JSON action (`{"action":"click","target_id":42,"bbox":[...]}`), immediately render a subtle highlight ring over element `#42` in the page overlay before final token completion—cutting perceived interaction latency in half.
4. **Zero-Copy `Transferable` Workers & Frame-Budget Pacing:** In [engineWorker.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/engineWorker.ts#L35-L37), use `postMessage(msg, [buffer])` `Transferable` ownership for any screenshot crops, DOM binary buffers, or state snapshots (`SharedArrayBuffer` when COOP/COEP applies), and replace fixed `2–6 ms` sleeps in [gpuPacing.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gpuPacing.ts) with adaptive `timestamp-query` GPU timing so `balanced` mode hits `144 FPS` without sacrificing 35% of decode speed (`36.99 → ~65+ tok/s`).

---

## 5. Prioritized Experiment & Horizon Roadmap

| Priority | Idea | Category | Est. Speedup / Quality Impact | VRAM / Bundle Impact | Implementation Effort | Prerequisites |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0** | **Register/SMEM-Cached Multi-Token `WGSL_GDN_RECURRENT`** (eliminate `2*M` global state passes) | Engine Kernels (Prefill) | **3×–6×** prefill (`386 → 1,200+ tok/s`); closes ORT prefill gap | Neutral (`0 MiB`) | Low–Med (1–2 days) | Modify [WGSL_GDN_RECURRENT](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs#L1304) |
| **P0** | **Fused `QKV+Z+AB`, `QG+K+V`, & `Gate+Up+SwiGLU`** (`397 → ~175` dispatches/tok) | Engine Kernels (Decode/Prefill) | **1.25×–1.45×** decode (`73.9 → ~95–105 tok/s`), lower JS overhead | Neutral (slight `workBuf` reduction) | Low–Med (2–3 days) | Update [wgslRealEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslRealEngine.ts) + [onnxWeightLoader.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/onnxWeightLoader.ts) |
| **P0** | **Coalesced SoA Block-32 Q4 + Packed `f16` Activations + 64/128-Thread GEMV** | Engine Kernels (Bandwidth) | **1.5×–2.0×** decode (`19.4% → ~35–42%` BW; `~130–150 tok/s`) | `-50%` activation buffers | Med (3–5 days) | Update `repack_q4_rows` + `WGSL_Q4_*` in [lib.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/lib.rs) |
| **P1** | **Prompt Lookup + SuffixDecoding (`batch=4–8` Speculative Verify)** | Speculative Decoding | **1.8×–2.8×** effective decode on DOM/agent/code (`200–350+ tok/s`) | `+2–4 MiB` CPU suffix tree; `+0` weight bundle | Med (4–6 days) | Batch GEMM for `M=4..8` + DeltaNet `K`-step safe commit |
| **P1** | **`packed_4x8_integer_dot_product` (`dot4I8Packed`) in `WGSL_GQA_INT8_*` + Unrolled `t`** | Engine Kernels (Long-Ctx) | **1.6×–2.2×** GQA attention at 32K–128K (`23.8 → ~40+ tok/s` at 128K) | Neutral | Low–Med (2–3 days) | Chrome `packed_4x8_integer_dot_product` check in [wgslWebgpu.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/wgslWebgpu.ts) |
| **P1** | **GPU-Coalesced Prefix Cache & DeltaNet/KV Ring Snapshots** (Strata / Rapid-MLX style) | End-UX / Caching | **~10×** Turn-2+ TTFT (`325 ms → <30 ms`); removes 64× `mapAsync` copies | Optional `+64–256 MiB` GPU ring or OPFS LRU | Med (3–4 days) | WGSL pack/unpack kernel + prefix hash key |
| **P1** | **Closed-Loop `timestamp-query` Adaptive GPU Pacing** | End-UX / Smoothness | **+40%–60%** `balanced` tok/s (`36.99 → ~55–65 tok/s`) at `144 FPS` | Neutral | Low (1–2 days) | `timestamp-query` feature + [gpuPacing.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gpuPacing.ts) |
| **P2** | **`chromium_experimental_subgroup_matrix` Prefill & Verify GEMM** | Engine Kernels (Tensor Core) | **2.5×–4×** prefill GEMM (`1,500–2,500+ tok/s`) | Neutral | Med–High (1 wk) | Feature detection + fallback to `vec4<f16>` GEMM |
| **P2** | **SpinQuant/QuaRot + Mixed-Precision 2.5–3 Bit DeltaNet/MLP FFNs** | Quantization (Weights) | **~1.5×–1.6×** decode bandwidth + faster load (`4.71s → ~2.8s`) | **`2.45 GiB → ~1.45 GiB`** (`-1.0 GiB`!) | Med–High (1–2 wks) | Offline Hadamard rotation + WGSL 2/3-bit or LUT-GEMM kernel |
| **P2** | **KIVI INT4/INT2 KV + DuoAttention + Quest Page Sparsity (8 Full-Attn Layers)** | Quantization (KV Cache) | **2×–3×** 128K decode speed (`~55–65 tok/s` at 128K) | **128K KV: `2.03 GiB → ~0.5–1.0 GiB`** | Med–High (1–2 wks) | Calibration profile for 32 KV heads + page min/max index |
| **P2** | **Chunkwise WY Representation for Gated DeltaNet Prefill** | Engine Kernels (Prefill) | **4×–8×** prefill on long chunks (`M=64–256`) | Small `W/Y` scratch buffer | High (1–2 wks) | `subgroupMatrix` or tiled GEMM + WY factor kernel |
| **P2** | **WASM XGrammar-2 Constrained Decoding + Hybrid DOM/Crop Agent Loop** | Agent Reliability / UX | **100%** valid JSON/tool syntax (`0` retries), `<150 ms` DOM action steps | `~1–2 MiB` WASM | Med (4–5 days) | Rust/WASM `xgrammar` integration in `crates/` |
| **P3** | **Predictive Keystroke/Hover Prefill + Transferable Worker Buffers** | End-UX Latency Hiding | Near-zero perceived TTFT on user submit | Neutral | Low–Med (2–3 days) | Incremental token-prefix invalidation + [engineWorker.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/engineWorker.ts) |
| **P3** | **Optional Local Companion Offload (`exo` / `Strata` / `Rapid-MLX` WebSocket)** | Hybrid / Multi-Device | Access to 30B–120B Planner while keeping 4B/2B Executor in-browser | `0` extra browser VRAM | Med (1 wk) | Local companion daemon + planner/executor router |
| **Horizon** | **Minitron-Pruned `1.5B–2.5B` Gated DeltaNet-2 / Hymba Model (`32K–64K` Vocab)** | Custom Model Architecture | **2×** raw decode + **5×** faster `lm_head` (`slab2`: `379 → <100 MiB`) | **`<1.1–1.3 GiB`** total bundle | High (4–8 wks) | GPU training cluster + distillation pipeline |
| **Horizon** | **WGSL-Aligned QAT (Q4/2-bit/BitNet b1.58) + Co-Trained EAGLE-3/MTP Heads** | Custom Model Training | **300–500+ effective tok/s** in browser at minimal accuracy loss | **`<0.9–1.2 GiB`** total bundle | High (4–8 wks) | Custom training loop + WGSL kernel parity tests |
| **Horizon** | **Playwright GRPO / WebRL + Magnitude/OS-Atlas/UI-TARS Grounding SFT** | Agent Specialization | Surpasses 4B–7B generic models on DOM + coordinate web tasks | `0` runtime overhead | High (4–8 wks) | Playwright RL harness + OS-Atlas/ShowUI/WebRL corpora |

---

## 6. Verified External References

- **Strata (Local MoE Engine):** [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · [MCP_SERVER.md](https://github.com/Niko1221/Strata/blob/main/docs/MCP_SERVER.md) · [MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)
- **Strata (Hierarchical Context Caching, OSDI '26):** [arXiv:2508.18572](https://arxiv.org/abs/2508.18572)
- **Strata (Klavis AI Progressive MCP):** `Klavis-AI/strata` / `Klavis-AI/klavis`
- **magnitude.dev:** [github.com/magnitudedev/browser-agent](https://github.com/magnitudedev/browser-agent) · `magnitudedev/magnitude`
- **exo:** [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
- **ktransformers:** [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) · Marlin: [arXiv:2408.11743](https://arxiv.org/abs/2408.11743)
- **Rapid-MLX & Apple MLX:** `raullenchai/Rapid-MLX` ([rapidmlx.com](https://rapidmlx.com)) · `ml-explore/mlx`
- **Speculative Decoding:** EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840), [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)) · SuffixDecoding ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975), [snowflakedb/ArcticInference](https://github.com/snowflakedb/ArcticInference)) · Lookahead Decoding ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057)) · Medusa ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774))
- **Quantization & KV Compression:** SpinQuant ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406), [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)) · QuaRot ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456), [spcl/QuaRot](https://github.com/spcl/QuaRot)) · KIVI ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750), [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)) · SnapKV ([arXiv:2404.14469](https://arxiv.org/abs/2404.14469), [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)) · Quest ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) · DuoAttention ([arXiv:2410.10819](https://arxiv.org/abs/2410.10819), [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention)) · BitNet b1.58 ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764), [microsoft/BitNet](https://github.com/microsoft/BitNet)) · LUT-GEMM ([arXiv:2206.09557](https://arxiv.org/abs/2206.09557))
- **Linear Attention, Pruning, Grammars & GUI/Web Agents:** Gated DeltaNet ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464), [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)) · Gated DeltaNet-2 ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791), [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2)) · Flash Linear Attention ([fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)) · Hymba ([arXiv:2411.13676](https://arxiv.org/abs/2411.13676), [NVlabs/hymba](https://github.com/NVlabs/hymba)) · LFM2 ([huggingface.co/LiquidAI](https://huggingface.co/LiquidAI)) · Minitron ([arXiv:2407.14679](https://arxiv.org/abs/2407.14679), [NVlabs/Minitron](https://github.com/NVlabs/Minitron)) · XGrammar ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100), [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)) · XGrammar-2 ([arXiv:2601.04426](https://arxiv.org/abs/2601.04426)) · OS-Atlas ([arXiv:2410.23218](https://arxiv.org/abs/2410.23218)) · SeeClick ([arXiv:2401.10935](https://arxiv.org/abs/2401.10935)) · UI-TARS ([arXiv:2501.12326](https://arxiv.org/abs/2501.12326), [bytedance/UI-TARS](https://github.com/bytedance/UI-TARS)) · WebRL ([arXiv:2411.02337](https://arxiv.org/abs/2411.02337)) · Agent-E ([arXiv:2407.13032](https://arxiv.org/abs/2407.13032))
