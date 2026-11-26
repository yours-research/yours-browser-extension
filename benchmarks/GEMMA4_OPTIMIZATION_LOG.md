# Gemma 4 E4B QAT — WebGPU Optimization & Head-to-Head Log

Model: `onnx-community/gemma-4-E4B-it-qat-mobile-ONNX` (42 layers: 35 sliding + 7 full attention, 18 KV-shared layers, 262,144 vocab with 256-dim Per-Layer Embeddings)  
Hardware: NVIDIA RTX 3090 (Ampere, 24 GiB VRAM, 936 GB/s theoretical bandwidth, Chromium/Dawn D3D12, `subgroups` min/max=32, `shader-f16`, `packed_4x8_integer_dot_product` DP4A, 144Hz desktop compositor)

---

## 1. Cumulative Progression Summary

| Stage | Run ID | Cold Load | VRAM @ 128K | 1K Prefill / Decode | 32K Prefill / Decode | 128K Prefill / Decode | Chat Turn 1 | Speculative Decode | 32K NIAH | Balanced 8K+32K FPS (Jank) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline ORT WebGPU** | [g01-gemma4-ort-baseline.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g01-gemma4-ort-baseline.json) | 10,930 ms | OOM > 32K | 649.7 / 23.76 | 291.4 / 3.53 | — | — | — | — | — |
| **WGSL FP16 KV (<=32K)** | [g02-gemma4-wgsl-fp16kv.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g02-gemma4-wgsl-fp16kv.json) | 5,426 ms | 2,702.3 MiB (@32K) | 498.4 / 28.23 | 259.8 / 23.22 | — | — (61.15 Golden B) | — | — | — |
| **WGSL INT8 KV 128K** | [g03-gemma4-wgsl-int8kv-128k.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g03-gemma4-wgsl-int8kv-128k.json) | 5,890 ms | 3,222.3 MiB | 506.1 / 28.27 | 264.2 / 23.45 | 109.6 / 15.55 | — (61.64 Golden B) | — | **PASS** (23.81 tok/s) | — |
| **Track G4/G5 (Subgroup+SplitK)** | [g04](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g04-gemma4-wgsl-max-speed-128k.json) / [g05](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g05-gemma4-wgsl-worker-balanced-smoothness.json) | 7,684 ms | 3,222.8 MiB | 466.1 / 59.04 | 267.0 / 47.97 | 117.6 / 30.56 | 59.95 tok/s (116.5 ms) | 0.742x (62.33 -> 46.27, 88.1% acc) | PASS (g03) | 76.66 fps (7.35% jank) |
| **Phase 1: Fusion + Spec Fix** | [g07-phase1-fusion-spec.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g07-phase1-fusion-spec.json) | 7,334 ms | 3,250.6 MiB | 497.4 / 60.71 | 276.7 / 49.24 | *(partial sweep)* | 57.18 tok/s (104.6 ms) | **1.483x** (68.51 -> **101.60**, 88.1% acc) | — | — |
| **Phase 2: `vec4` + DP4A + FA2** | [g08-phase2-bandwidth-flash.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g08-phase2-bandwidth-flash.json) | 7,279 ms | 3,258.7 MiB | 456.5 / 60.38 | **361.5** / 48.61 | *(partial sweep)* | 58.16 tok/s (106.6 ms) | **1.316x** (71.29 -> **93.85**, 88.1% acc) | **PASS** (**48.11 tok/s**, 91.0s) | Chat **142.9 fps** (0% jank) / Sweep 68.85 fps (10.93% jank) |
| **Phase 2B: Warp Attn + 6L Slice** | [g08b-phase2b-warp-attn-smooth.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g08b-phase2b-warp-attn-smooth.json) | 7,737 ms | 3,258.7 MiB | 455.6 / **71.97** | **360.7** / **59.36** | *(partial sweep)* | **72.68 tok/s** (110.1 ms) | **1.249x** (75.10 -> **93.79**, 88.1% acc) | **PASS** (**57.51 tok/s**, 91.4s) | Chat **144.0 fps** (0% jank) / Sweep **143.93 fps** (**0.00% jank**!) |
| **Phase 3A: Hot PLE + Q8 PLE + Pipe** | [g09-phase3-rd-experiments.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g09-phase3-rd-experiments.json) | **7,416 ms** | 3,346.8 MiB | **475.2** / **91.09** | **370.1** / **75.69** | *(see g10)* | **85.91 tok/s** (105.6 ms) | **1.067x** (91.16 -> **97.29**, 88.1% acc) | **PASS** (**71.89 tok/s**, 89.6s) | *(see g11)* |
| **Final Gemma 4 WGSL (Locked)** | [g10](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g10-gemma4-final-max-speed-128k.json) / [g11](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g11-gemma4-final-balanced-smoothness.json) | **7,210 ms** | **3,346.8 MiB** | **482.0** / **94.51** | **372.8** / **75.78** | **220.7** / **51.88** | **88.65 tok/s** (106.4 ms) | **1.029x** (96.66 -> **99.49**, 88.1% acc) | **PASS** (**73.25 tok/s**, 88.9s) | Chat **144.0 fps** (`0%` jank, `37.95 tok/s`) / Sweep **140.23 fps** (`0.01%` jank, **55.03/51.86 tok/s**) |
| **Qwen3.5-4B Ref (Track 5)** | [g06-gemma4-vs-qwen35-comparison.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g06-gemma4-vs-qwen35-comparison.json) | 6,969 ms | 4,640.7 MiB | 507.0 / 109.55 | 393.1 / 74.06 | 176.9 / 32.32 | 91.71 tok/s (118.8 ms) | 1.251x (91.77 -> 114.83, 94.87% acc) | **PASS** (62.32 tok/s) | 140.25 fps (0.00% jank) |

---

## 2. Phase-by-Phase Engineering & Empirical Notes

### Baseline Tracks G1–G5 (`g01`..`g05`)
- **Quality Baseline:** Golden A `"4"` (`id=236812`, `val=28.5` ORT / `28.52` WGSL), Golden B `"WebGPU is a modern, high-performance API for running graphics and compute shaders directly in the web browser."` (`id=6721`, `val=28.25` ORT / `28.256` WGSL).
- **Root Bottleneck Found in G4/G5:**
  1. Scalar `u32` 16-lane Q4 GEMV (`100`) only reached `119–123 GB/s` (`13%` of 936 GB/s).
  2. Batched speculative verification fell back to unbatched Q4 GEMM (`64x64` tiles) and loop-of-`T` argmax passes, turning `88.1%` draft acceptance into `0.742x` slowdown (`62.33 -> 46.27 tok/s`).
  3. 32K prefill `chunkSize=64` exceeded 144Hz frame budgets in Balanced mode (`7.35%` jank).

### Phase 1: Fusion + Small-Batch Speculative Fix (`g07`)
- **Changes ([bundle-gemma4.mjs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/scripts/bundle-gemma4.mjs) v4, [gemma4.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/gemma4.rs) `127..=133`, [gemma4WgslEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gemma4WgslEngine.ts)):**
  - 16-byte aligned all weights/scales/aux; fused `qkv_proj` on donor layers `0..23`; prebuilt `per_layer_projection_q8` (`[10752, 2560]`); sharded PLE into `256` shards (`2.71 MiB` each) + `ple_hot_32k.bin` (`86.625 MiB`).
  - Hoisted `sum(x)` in `100/104/107/126`, added `127` fused RMSNorm+Residual (`-84` barriers/token), `128` fused QKV+RoPE+KVWrite, Adaptive Split-K (`sSplits/gSplits = 1` below `128` tokens), and `129..=133` specialized small-batch (`T<=8`) GEMV/GeGLU/LM-Head/Argmax kernels.
- **Empirical Results (`g07`):**
  - Golden A (`"4"`, `236812`, `28.521`), Golden B (`6721`, `28.2537`, `63.92 tok/s` vs `55.84` in G4), `0` console errors.
  - Speculative decoding flipped from `0.742x` (`46.27 tok/s`) to **`1.483x`** (`68.51 -> 101.60 tok/s`, `88.1%` acceptance, `parityPassed=true`).
  - 1K/32K decode moved slightly (`59.04 -> 60.71` / `47.97 -> 49.24 tok/s`), confirming narrow scalar `u32` GEMV DRAM latency is the primary decode bottleneck to solve in Phase 2.

### Phase 2: 128-Bit `vec4<u32>` GEMV + DP4A + FlashAttention-2 Prefill + Balanced Pacing (`g08`)
- **Changes ([gemma4_opt.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/gemma4_opt.rs) `134..=142`, [gemma4WgslEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gemma4WgslEngine.ts)):**
  - Wired `134..=137` (`vec4<u32>` Q4 GEMV / GeGLU / wide-K double-buffered GEMV / Q2 LM head), `138` DP4A global Split-K decode, `139` DP4A Q8 PLE GEMV, and `140..=142` FlashAttention-2 tiled prefill + Split-K merge (`prefillSplits = 8` for `>= 1024`).
  - Added Gemma 4 balanced prefill chunk clamping (`48` @ 4K, `32` @ 16K, `16` @ 49K).
- **Empirical Results (`g08`):**
  - **Quality & Safety:** Golden A `"4"` (`id=236812, val=28.5254`), Golden B `top0 id=6721 (28.2535)`, Speculative `parityPassed=true` (`71.29 -> 93.85 tok/s`, `1.316x`, `88.1%` acc), **32K NIAH `passed=true`** (`VELVET-ORBIT-7492`, `ttft=91,046ms`, `359.91 pref / 48.11 dec`), `0` console errors.
  - **Big Wins:** 32K prefill jumped `+30.7%` (`276.65 -> 361.51 tok/s`, reaching `92%` of Qwen3.5's `393.1 tok/s`); 32K NIAH decode **2.02x faster** than `g03` (`23.81 -> 48.11 tok/s`); Balanced Chat hit **`142.93 fps` with `0.00%` jank** (`p99=7.2ms`, `abort=23.7ms`, Turn 2 `prefixReused=true`).
  - **Re-plan Triggered:** 1K decode stayed at `60.38 tok/s` (`122.68 GB/s`) instead of `+20%` (`85+ tok/s`), AND Balanced Sweep (`8K+32K`) had `10.93%` jank (`68.85 fps`, `p99=62.5ms`) because all 42 layers of each 32-token prefill chunk still submitted in one monolithic compute pass (vs Qwen's 4-layer intra-chunk slicing).

### Phase 2B: Zero-Barrier Warp Attention Decode (`148`/`149`) + LM Head Revert (`107`) + 6-Layer Prefill Slicing (`g08b`)
- **GPU Kernel Micro-Profile Diagnosis (before 2B):** `gqaDecSlide + Reduce` (`3.25 ms`), `gqaDecGlobalDp4a + Reduce` (`1.79 ms`), `q2LmHeadVec4` regression (`0.988 ms` vs `0.488 ms` on `107`), `pleProj` (`1.25 ms`), `q4GemvWideKVec4` (`2.61 ms` w/ 10 barriers).
- **Changes ([gemma4_opt.rs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/crates/tinygpt-wgsl-engine/src/gemma4_opt.rs) `148`/`149` + widened `136`, [gemma4WgslEngine.ts](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/src/offscreen/gemma4WgslEngine.ts)):**
  - Added `148` (`WGSL_G4_GQA_DECODE_SLIDING_WARP`: 8 warps/Q-head, zero barriers in token loop, no separate reduce pass) and `149` (`WGSL_G4_GQA_DECODE_GLOBAL_WARP_DP4A`: 1 warp/Q-head, zero barriers in entire kernel, coalesced 128B reads).
  - Widened `136` wide-K tile (`32 -> 64` `vec4<u32>`s), reverted `T=1` LM head to `107`, and added 6-layer intra-chunk prefill slicing + clamped balanced chunks (`32` @ 4K, `16` @ 24K).
- **Empirical Results (`g08b`):**
  - **1K..32K Decode Jumped +19% to +25%:** `1K`: `60.38 -> 71.97 tok/s` (`146.23 GB/s`), `8K`: `53.10 -> 66.09 tok/s` (`+24.5%`), `32K`: `48.61 -> 59.36 tok/s` (`+22.1%`), Chat Turn 1: `58.16 -> 72.68 tok/s` (`+25.0%`), 32K NIAH: **`passed=true` at `57.51 tok/s`** (**2.42x faster** than `g03`!).
  - **144Hz Balanced Smoothness Fixed Completely:** Balanced Chat = **`144.00 fps` (`0.00%` jank, `p99=7.1ms`, `abort=6.0ms`, Turn 2 `prefixReused=true`)**, Balanced 8K+32K Sweep = **`143.93 fps` with `0.00%` jank** (down from `10.93%` jank / `68.85 fps` in `g08`, beating Qwen3.5's `140.25 fps`!). Note balanced 32K prefill slowed (`238 -> 135 tok/s`) due to `chunk<=16` + 6-layer breathers — tunable in Phase 3 (`chunk<=32`, 14-layer slice) for faster balanced prefill while staying `0%` jank!

### Phase 3 R&D A/B Suite (`g09-phase3-rd-experiments.json`)
All five Phase 3 R&D tracks were benchmarked behind toggle flags in `Gemma4LoadOptions` / `Gemma4GenerateOptions`, keeping strict parity & 32K NIAH quality gates:
- **Experiment 3A (`enableGpuHotPle`, `enableQ8PleProj`, Split-K `148`, `enablePipelinedDecode`) — ADOPTED BY DEFAULT:**
  - `3A-Sync` (Hot PLE `143` + Q8 PLE Proj `139` `mode=2` + Split-K `148`, no pipelining): `1K` `77.29 tok/s`, `32K` `67.41 tok/s` (`+13.6%` over Phase 2B), Chat Turn 1 `81.44 tok/s`.
  - `3A-Pipe` (+ 2-in-flight GPU/CPU pipelined decode with `tokenIdsBuf` feedback & cold-token fallback): `1K` **`91.09 tok/s`** (`185.08 GB/s`, `+26.6%` vs `g08b`), `32K` **`75.69 tok/s`** (`188.24 GB/s`, `+27.5%` vs `g08b` — **beats Qwen3.5-4B `74.06 tok/s` at 32K!**), Chat Turn 1 `85.91 tok/s`, Speculative `88.1%` acc (`97.29 tok/s`, `parity=true`), Golden A/B PASS, **32K NIAH `passed=true` (`71.89 tok/s`, `365.74` pref)**.
- **Experiment 3B (`enableW4A8Dp4a` — `144` W4A8 Q4 GEMV + `145` W2A8 LM Head) — KEPT OPT-IN (`false` by default):**
  - Peak short-context speed hit **`110.88 tok/s`** at 1K (`225.29 GB/s` — beats Qwen3.5 `109.55 tok/s`!) and `98.95 tok/s` at 8K (`98.9 tok/s` Chat Turn 1, Golden A/B PASS).
  - Rejected as default because symmetric per-64 W4A8/W2A8 activation quantization cut suffix speculative acceptance (`88.1% -> 56.82%`, `79.71 tok/s`, `0.746x`) AND reproduced `[Device] is lost` during sustained 32K sweep / 32K NIAH prefill.
- **Experiment 3C (`kvMode='int4'` — `146` / `147` + `114`/`128` `kv_mode=2`) — KEPT OPT-IN (`int8` remains default):**
  - Cuts 128K VRAM by **`-488 MiB`** (`3346.81 -> 2858.81 MiB`, KV `1072 -> 584 MiB`) and **passes Golden A/B & 32K NIAH (`passed=true`)** at `92.73 tok/s` (1K) and `92.94 tok/s` (Chat Turn 1).
  - Kept opt-in because `int8` stays faster on 32K prefill (`370.1 vs 312.2 tok/s`), 32K decode (`75.69 vs 72.63 tok/s`), cold load (`7.4s vs 9.7s`), and preserves full `88.1%` speculative acceptance (vs `56.82%` on `int4`).
- **Experiment 3D (`draftSkipLayers = 6 / 12 / 18` self-speculation) — REJECTED (`0` default):**
  - `parity=true` confirmed correctness, but Gemma 4 PLE + late layers break draft alignment: `6L` skip = `27.21 tok/s` (`0.317x`, `27.18%` acc), `12L` skip = `21.86` (`0.254x`, `12.5%` acc), `18L` skip = `16.85` (`0.196x`, `0%` acc).
- **Experiment 3E (`chromium_experimental_subgroup_matrix` cooperative matrix audit) — UNAVAILABLE:**
  - Verified not exposed on Windows Chromium/Dawn D3D12 (RTX 3090).

### Final Locked 128K (`g10`), Final 144Hz Balanced (`g11`), and Head-to-Head (`g06`)
- **Final Max-Speed 128K + 32K NIAH ([g10-gemma4-final-max-speed-128k.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g10-gemma4-final-max-speed-128k.json)):**
  - Cold Load `7,210 ms`, VRAM `3,346.81 MiB` (`1,072 MiB` KV at 128K — `1.29 GiB` smaller than Qwen3.5's `4,640.7 MiB`).
  - Golden A `"4"` (`236812`, `28.529`), Golden B `6721` (`28.2405`, `90.56 tok/s`), Speculative `96.66 -> 99.49 tok/s` (`88.1%` acc, `parity=true`), **32K NIAH `passed=true`** (`368.44 pref / 73.25 dec`, `88.9s`), `0` console errors.
  - Full context sweep (`1K..128K`): `1K`: `481.97 / 94.51`, `8K`: `440.89 / 89.71`, `32K`: `372.81 / 75.78`, `64K`: `303.17 / 68.07`, `96K`: `256.35 / 59.58`, `128K`: **`220.69 / 51.88 tok/s`** (`202.14 GB/s`).
- **Final 144Hz Balanced Smoothness ([g11-gemma4-final-balanced-smoothness.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g11-gemma4-final-balanced-smoothness.json)):**
  - Balanced Chat: **`37.95 tok/s @ 144.00 fps` (`0.00%` jank, `p99=7.1ms`, `abort=30.1ms`, Turn 2 `prefixReused=true`)**.
  - Balanced 8K+32K Sweep: **`140.23 fps` (`0.01%` jank, `0%` freeze, `p99=13.9ms`)** — `8K`: **`275.75 / 55.03 tok/s`**, `32K`: **`185.16 / 51.86 tok/s`** (+46–51% faster balanced decode and +22–37% faster balanced prefill vs Phase 2B `g08b`!).
- **Final Gemma 4 WGSL vs Qwen3.5-4B Track 5 ([g06-gemma4-vs-qwen35-comparison.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g06-gemma4-vs-qwen35-comparison.json)):**
  - Cross-over at **32K+**: Gemma 4 beats Qwen3.5-4B decode at `32K` (`75.78 vs 74.06`, `1.02x`), `64K` (`68.07 vs 55.74`, **`1.22x`**), `96K` (`59.58 vs 43.71`, **`1.36x`**), and **both prefill (`220.69 vs 176.93`) and decode (`51.88 vs 32.32`, `1.61x`) at `128K`** while using `1.29 GiB` less VRAM!

### Offline Chrome Extension Head-to-Head Verification (`g12-extension-head-to-head.json`)
- **Built & Verified ([g12-extension-head-to-head.json](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/benchmarks/runs/g12-extension-head-to-head.json), [test-extension-head-to-head.mjs](file:///C:/Users/taylo/.t3/projects/tinygpt-browser/scripts/test-extension-head-to-head.mjs)):**
  - Bundled `dist/models/qwen35-4b/` (unchanged) + `dist/models/gemma4-e4b/` (`ple_shards/`, `ple_hot_32k.bin`, `ple_hot_map.bin`, slabs/aux/tokenizer/`chat_template.jinja`; `embed_tokens_q2.bin` & `ple_tokens_q2.bin` excluded).
  - Live Chrome `--load-extension=dist` (`0` external HTTPS requests, `0` console/page errors, switching via `<select data-testid="model-switcher">`):
    - **Qwen3.5-4B (`onnx-community/Qwen3.5-4B-ONNX-OPT`):** Cold load `6,690 ms` (`Slot0=192t`, `tier1-f16-subgroups`, `2,563 MiB` base est VRAM), `"2 + 2"` -> `"4"` (`ttft=439ms`, `92.6 tok/s`), Page-On `"QUANTUM-ANCHOR-8821"` (`reused=true`, `ttft=134ms`, `93.5 tok/s`), Followup Turn 2 (`reused=true`, `slot=live`, `ttft=128ms`, `119.7 tok/s`).
    - **Gemma 4 E4B QAT (`onnx-community/gemma-4-E4B-it-qat-mobile-ONNX`):** Cold load `6,793 ms` (`Slot0=211t`, `tier1-f16-subgroups`, `3,347 MiB` est VRAM), `"2 + 2"` -> `"4"` (`ttft=398ms`, `77.5 tok/s`), Page-On `"QUANTUM-ANCHOR-8821"` (`reused=true`, `ttft=152ms`, `47.5 tok/s`), Followup Turn 2 (`reused=true`, `slot=live`, `ttft=157ms`, `65.9 tok/s`).
