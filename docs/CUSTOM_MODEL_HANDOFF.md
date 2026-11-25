# Yours Browser Model Foundry — Architecture, Training & WebGPU Handoff Specification

> **Purpose:** Complete, self-contained technical handoff for building a dedicated model optimization, distillation, fine-tuning, quantization, and RL training pipeline that outputs custom checkpoints directly consumable by the [`yours-browser-extension`](https://github.com/yours-research/yours-browser-extension) WebGPU/WGSL browser engine.  
> **Grounded Sources:** [docs/RESEARCH.md](./RESEARCH.md) · [docs/DOWNLOAD_BENCHMARKS.md](./DOWNLOAD_BENCHMARKS.md) · [benchmarks/REPORT.md](../benchmarks/REPORT.md) · [crates/tinygpt-wgsl-engine/src/lib.rs](../crates/tinygpt-wgsl-engine/src/lib.rs) · [src/offscreen/wgslRealEngine.ts](../src/offscreen/wgslRealEngine.ts) · [src/offscreen/onnxWeightLoader.ts](../src/offscreen/onnxWeightLoader.ts)

---

## 1. Strategic Verdicts

### 1.1 Is `Qwen3.5-4B` the Best Base Model for This?

**Short Answer:** **Yes — the `Qwen3.5` hybrid family (Apache 2.0) is currently the single best base family for a browser-native agent, with a deliberate two-tier split between `Qwen3.5-4B` (Day-1 flagship & immediate drop-in) and `Qwen3.5-2B` (universal fast laptop/edge student).**

Why **Qwen3.5** beats pure-attention small models (`Qwen3-4B`, `Llama-3.2-3B`, `Gemma-3-4B`) for WebGPU extensions:
1. **3:1 Gated DeltaNet + Full GQA Hybrid Architecture:** Only **1 in 4 layers** stores a KV cache. At 128K context (`INT8 KV`), `Qwen3.5-4B` needs only **`~2.03 GiB` KV + `50.25 MiB` constant DeltaNet state** (`4,641 MiB` total VRAM vs. `6,855.7 MiB` and `2.82 tok/s` for generic ORT FP16; measured **`23.80 tok/s` at 128K** in Track 2 Run `16`, [REPORT.md](../benchmarks/REPORT.md#L76-L86)).
2. **Kernel Co-Design Already Paid For:** Our custom Rust WASM + native WGSL runtime ([lib.rs](../crates/tinygpt-wgsl-engine/src/lib.rs)) already implements Qwen3.5's exact non-standard primitives: depthwise causal Conv1D + SiLU (`8192×4`), `2:1` GQA-grouped Gated DeltaNet (`16` K-heads / `32` V-heads × `128`), Gated RMSNorm, partial NeoX RoPE (`rotary_dim=64` out of `256`), and sigmoid-gated Full Attention (`wQg = 8192`).
3. **Native 262K Context & Early-Fusion Vision:** Pre-trained natively across text, HTML/code, images, and UI screenshots under Apache 2.0, defaulting to non-thinking mode (`enable_thinking=False`) so tool/action JSON isn't blocked by runaway chain-of-thought tags.

#### Candidate Base Model Comparison

| Candidate | Architecture & Params | Est. Browser Bundle (Q4 / Mixed 2.5–3b + 49K Vocab) | Key Strengths | Weaknesses vs. Goal | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen3.5-4B`** (Current baseline) | 32L (24 Gated DeltaNet + 8 GQA), `H=2560`, `I=9216`, `V=248320` | `2.45 GiB` raw WGSL Q4 → **`~2.15 GiB`** (vocab pruned) → **`~1.25–1.45 GiB`** (mixed Q4/2.5b) | Zero engine geometry changes for Day-1 SFT/GRPO/QAT; strong reasoning & coding intelligence | `H=2560` (`80` Q4 blocks) causes `32`-lane warp tail divergence; `1.01 GiB` of FFN weights in 24 DN layers keeps raw size ~2.5 GiB | **Primary Day-1 Base** (zero-friction SFT, vocab surgery, QAT & RL) |
| **`Qwen3.5-2B`** | 24L (18 Gated DeltaNet + 6 GQA), `H=2048`, `I=6144`, `V=248320`, `dn_val_heads=16` | **`~1.05–1.25 GiB`** (vocab pruned Q4) → **`~0.75–0.90 GiB`** (mixed 3b) | Exact power-of-2 WebGPU dims (`2048=64×32`, `6144=192×32`, `16` DN heads); `~2×` faster raw decode; fits 4 GB iGPUs; ~5 GB VRAM LoRA | Lower zero-shot multi-step planning ceiling than 4B unless distilled from 9B/frontier teacher | **Top Speed / Edge Student** (Distill + RL from 9B/frontier → best Chrome Web Store default!) |
| **Minitron-Pruned `4B → 2.5B–3B`** | Pruned depth/width from `Qwen3.5-4B` or `9B` (`H=2048` aligned) | `~1.1–1.5 GiB` | Keeps richer teacher representations than training 2B from scratch ([arXiv:2407.14679](https://arxiv.org/abs/2407.14679)) | Requires pruning + recovery distillation before SFT/RL | Best intermediate step if 2B loses too much agent capability |
| **`Qwen3.5-9B` / `Qwen3.5-397B-A17B`** | Hybrid Gated DeltaNet | Too large for client WebGPU bundle (`>4.8 GiB`) | Massive visual/DOM/planning accuracy; Apache 2.0 | Exceeds browser bundle/VRAM budgets | **Teacher & Trajectory Generator ONLY** |
| **Pure Transformers** (`Qwen3-4B`, `Llama-3.2-3B`, `Gemma-3`) | 100% GQA/MHA (zero linear attn) | `1.8–2.5 GiB` weights, **4× KV bloat at 128K** | Mature tooling (Unsloth, vLLM, Marlin) | Fails tiny-browser long-DOM memory economics (`4×` slower/heavier at 32K–128K) | Reject for long-context DOM agents |
| **Greenfield Custom** (`Gated DeltaNet-2` + `Hymba` + `BitNet b1.58`) | Decoupled erase/write GDN-2 ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791)), `V=32K–49K`, ternary/2b QAT | **`<0.9–1.1 GiB`** | Ultimate theoretical `300–500+ tok/s` ceiling | Needs custom pre-training or heavy CPT ($10K+ compute) + new WGSL kernels | **Stage 3+ Horizon Architecture** (not Day 1) |

**Balanced Decision:**
1. **Start on `Qwen3.5-4B` immediately** for **Stage 0–2** because the WGSL engine, shard pipeline, and parity tests already match its exact dimensions. Removing unused multilingual vocab tokens (`248,320 → 49,152` or `32,768`), distilling browser tasks, and quantizing DeltaNet/MLP FFNs cuts it from **`2.47 GiB` → `~1.25–1.45 GiB`** while keeping 4B intelligence.
2. **In parallel or Stage 2B, distill `Qwen3.5-2B` (`H=2048, I=6144, L=24`)** from `Qwen3.5-9B` / frontier teachers on browser trajectories. Its dims divide evenly across WebGPU warp widths (`2048/32 = 64`, `6144/32 = 192` — zero tail waste!), halves weight reads for **~2× native decode speed**, and shrinks the entire extension + model bundle near **`~1.0 GiB`**, enabling even 4–8 GB ultrabook iGPUs (`Apple Silicon` / `Intel Arc` / `AMD 780M`) to run effortlessly.

### 1.2 Should This Be a Separate Project/Repo or a Separate Thread Here?

**Verdict: Create a SEPARATE REPOSITORY (e.g., `yours-research/yours-model-foundry`) + separate T3 Code workspace/thread, coupled ONLY through a strict binary export & verification contract.**

Why a separate repo is decisively cleaner:
1. **Completely Disjoint Tech Stacks & OS Environments:**
   - **`yours-browser-extension`** (this repo, Windows + Chrome + RTX 3090): TypeScript, React, Vite, Chrome MV3, Rust `wasm32-unknown-unknown`, WGSL shaders, Playwright E2E. Zero Python or CUDA runtime deps.
   - **Model Foundry** (new repo, Linux + multi-GPU CUDA/H100/A100/RTX nodes): Python, PyTorch, Hugging Face `transformers`, `fla-org/flash-linear-attention` (Triton kernels for fast Gated DeltaNet training), `TRL` / `GRPO`, `verl` / Playwright headless RL sandboxes, SpinQuant/Hadamard calibrators, SafeTensors Q4 repacker + Ed25519 manifest signer.
2. **Artifact Size & Git Hygiene:** Training datasets, LoRA checkpoints, activation calibration matrices, and RL rollouts would pollute a lean browser extension repo.
3. **Clean Interface Boundary:** The Foundry emits a versioned **`WgslBundle` directory** (`manifest.json` + signed `slab*.partNN.bin` + `aux_params.bin` + `tokenizer.json` + `config.json`) AND runs a **cross-repo PyTorch-vs-WGSL logit parity check** before handing an artifact over to `yours-browser-extension`.

*(Within the extension repo itself, keep one small config task for the future: making [lib.rs](../crates/tinygpt-wgsl-engine/src/lib.rs#L11-L29), [onnxWeightLoader.ts](../src/offscreen/onnxWeightLoader.ts#L5-L9), and [wgslRealEngine.ts](../src/offscreen/wgslRealEngine.ts) read `hidden_size`, `intermediate_size`, `num_layers`, `vocab_size`, `full_attn_interval`, and `dn_val_heads` dynamically from `manifest.json` / `config.json` instead of hardcoded Qwen3.5-4B constants so the same compiled extension loads `2B`, pruned `2.5B`, or `4B` models seamlessly.)*

---

## 2. Exact TinyGPT WebGPU / WGSL Engine Contract

Any model checkpoint produced by the Foundry **must** obey this hardware & binary contract (or ship alongside an explicitly versioned WGSL kernel update).

### 2.1 Tensor Geometry & Layer Ordering (`v1` Current `Qwen3.5-4B` vs. `v2` Config-Driven `Qwen3.5-2B`)

| Parameter | `Qwen3.5-4B` (`v1` Hardcoded) | `Qwen3.5-2B` (`v2` Recommended Config-Driven) | Code Location |
| :--- | :---: | :---: | :--- |
| `num_layers` | `32` | `24` | `lib.rs#L13`, `onnxWeightLoader.ts#L7` |
| `full_attn_layers` (`interval=4`) | `{3, 7, 11, 15, 19, 23, 27, 31}` (`8` layers) | `{3, 7, 11, 15, 19, 23}` (`6` layers) | `onnxWeightLoader.ts#L9` |
| `hidden_size` (`H`) | `2560` (`80` Q4 blocks) | `2048` (`64` Q4 blocks — exact power of 2!) | `lib.rs#L11`, `onnxWeightLoader.ts#L5` |
| `intermediate_size` (`I`) | `9216` (`288` Q4 blocks) | `6144` (`192` Q4 blocks) | `lib.rs#L12`, `onnxWeightLoader.ts#L6` |
| `vocab_size` (`V`) | `248320` (Target after surgery: `32768` or `49152`) | `248320` (Target after surgery: `32768` or `49152`) | `lib.rs#L15`, `onnxWeightLoader.ts#L8` |
| Full-Attn Q / KV heads / dim | `16` Q / `4` KV / `256` dim (`rotary_dim=64`) | Verify `config.json` at export (`8` Q / `2` KV / `256`) | `lib.rs#L18-L20` |
| DeltaNet K / V heads / dim | `16` K / `32` V / `128` dim (`conv_kernel=4`) | `16` V heads / `128` dim (`conv_dim = K*128 + K*128 + V*128`) | `lib.rs#L23-L28` |

#### Exact Binary Weight Slabs (`slab0`, `slab1`, `slab2`, `aux_params.bin`)

From [wgslRealEngine.ts](../src/offscreen/wgslRealEngine.ts#L629-L672) and [onnxWeightLoader.ts](../src/offscreen/onnxWeightLoader.ts#L87-L133):

1. **`slab0`** (Layers `0..N/2 - 1`; for 4B layers `0..15`, `1,114,112,000 B` = `1,062.5 MiB`) and **`slab1`** (Layers `N/2..N-1`; for 4B layers `16..31`, `1,062.5 MiB`):
   Weights are concatenated row-major per layer in exact order:
   - **Gated DeltaNet Layer:**
     1. `wQkv` `[Q_dim + K_dim + V_dim, H]` = `[8192, 2560]` (`Q=0..2047`, `K=2048..4095`, `V=4096..8191`)
     2. `wZ` `[V_dim, H]` = `[4096, 2560]` (output SiLU gate)
     3. `wAb` `[2 * DN_VAL_HEADS, H]` = `[64, 2560]` (low `0..31` = `b_proj` beta gate, `32..63` = `a_proj` decay gate)
     4. `wOut` `[H, V_dim]` = `[2560, 4096]`
     5. `wGate` `[I, H]` = `[9216, 2560]`
     6. `wUp` `[I, H]` = `[9216, 2560]`
     7. `wDown` `[H, I]` = `[2560, 9216]`
   - **Full-Attention GQA Layer:**
     1. `wQg` `[2 * ATTN_HEADS * 256, H]` = `[8192, 2560]` (**interleaved `[Q_head (256), gate_head (256)] × 16`**, verified in [wgslRealEngine.ts](../src/offscreen/wgslRealEngine.ts#L1483-L1504))
     2. `wK` `[ATTN_KV_HEADS * 256, H]` = `[1024, 2560]`
     3. `wV` `[ATTN_KV_HEADS * 256, H]` = `[1024, 2560]`
     4. `wO` `[H, ATTN_HEADS * 256]` = `[2560, 4096]`
     5. `wGate` `[I, H]` = `[9216, 2560]`
     6. `wUp` `[I, H]` = `[9216, 2560]`
     7. `wDown` `[H, I]` = `[2560, 9216]`
2. **`slab2`** (Tied Embeddings & LM Head `[V, H]`, e.g., `248320 × 2560` = `397,312,000 B` = `378.91 MiB`; **`49152 × 2560` drops to `78.64 MB`**!):
   - Note: Both `embedPipe` (`WGSL_EMBED_LOOKUP`) and `lmHead` (`WGSL_Q4_SUBGROUPS_F16`) read directly from `slab2`, enforcing **tied word embeddings & LM head** in Q4 (unless separated in a v2 manifest).
3. **`aux_params.bin`** (All small FP32 vectors concatenated linearly):
   - **CRITICAL OS RULE:** **NEVER name this file `aux.bin`!** `AUX` is a legacy Win32 DOS reserved device name (`CON`, `PRN`, `AUX`, `NUL`, `COM1..9`, `LPT1..9`) blocked by Chromium's `ExtensionURLLoader` on Windows. Always use `aux_params.bin`!
   - Layout order (`f32` little-endian): `invFreq[32]` (`rope_theta = 10,000,000`, partial `rotary_dim = 64`), `finalNorm[H]`, then for each layer `l = 0..NUM_LAYERS-1`:
     - `inputNorm[H]`, `postNorm[H]`
     - If Full-Attn: `qNorm[256]`, `kNorm[256]`
     - If DeltaNet: `A_neg[32]` (`-exp(A_log)`), `dt_bias[32]`, `gdn_norm[128]`, `convW[8192 * 4]` (causal `conv1d` kernel=4 flattened `[channel, 4]`)

#### Exact Block-32 Q4 Packing Format (`v1`)

From [`repack_q4_rows`](../crates/tinygpt-wgsl-engine/src/lib.rs#L2003-L2037) and [`dot_q4_block`](../crates/tinygpt-wgsl-engine/src/lib.rs#L297-L312):
- Every `K` must be divisible by **`32`** (ideally divisible by **`512` (`16` blocks)** so zero GPU lanes sit idle in multi-warp GEMV!).
- Each block of `32` weights = **`5 × u32` (`20` bytes)**:
  - `u32[0]`: weights `0..7` (4-bit unsigned `q ∈ [0, 15]`, weight `0` in bits `0..3`, weight `7` in bits `28..31`)
  - `u32[1]`: weights `8..15`
  - `u32[2]`: weights `16..23`
  - `u32[3]`: weights `24..31`
  - `u32[4]`: `(scale_f16 as u32) | ((zero_f16 as u32) << 16)` where **`zero_f16 = f32_to_f16(-f16_to_f32(scale_f16) * zp)`**. WGSL executes `fma(f16(q), scale_f16, zero_f16)`, i.e. `w_hat = q * scale + zero`.

### 2.2 Browser Distribution, WebGPU & Integrity Hard Constraints (From Verified Benchmarks)

1. **Zero `2.00 GiB` Single-Buffer Exceedance:** Chromium grants elevated `maxBufferSize = 2147483648 B` (`2.00 GiB`), so weights must remain split across `slab0` / `slab1` / `slab2` (`<= 1.2 GiB` each unless total weights shrink below `2.0 GiB`).
2. **`<= 128 MiB` File Shard Cap:** Chromium's `ExtensionURLLoader` crashes cleanly when fetching a single bundled asset `> ~1 GiB` (`slab0.part00..08.bin` capped at `134,217,728 B` = `128 MiB` prevents this and keeps Dawn staging shared memory bounded via per-shard `onSubmittedWorkDone()`).
3. **Never Gzip/Brotli Pre-Quantized Shards:** Verified on `slab0.part00.bin` in [DOWNLOAD_BENCHMARKS.md](./DOWNLOAD_BENCHMARKS.md#L87-L97): Shannon entropy is **`7.707 / 8.000 bits/byte`**. Compression only saves `3.5%–6.1%`, adds `7.7s–12.8s` decompression CPU, breaks broadband throughput, and destroys HTTP `Range` byte offsets.
4. **CDN Hosting & Signing (When Non-Bundled):** Hugging Face Xet CAS throttles Chromium's coalesced HTTP/2 to **`1.51–1.53 MiB/s`** (`~27.5 min` for `2.47 GiB`), whereas multi-socket HTTP/1.1 (`c=8–16`) hit **`21.74–22.83 MiB/s`** (`~111–116s`). Host raw pre-repacked WGSL shards on **Cloudflare R2 + custom domain (HTTP/3 QUIC, `$0` egress)** with an **Ed25519-signed `manifest.json`** containing **per-shard / per-16 MiB chunk SHA-256 hashes** (`1,857 MiB/s` verification cost in JS/Worker, caught 1-byte tampering immediately and repaired in `1.51s`).
5. **144 FPS UI Smoothness & Worker Budget:** Target model weights + KV + runtime state must leave comfortable headroom on consumer `8–24 GB` GPUs (ideally `<1.5 GiB` weights total so laptops with 8–16 GB unified memory can also run at 60–144 FPS).

---

## 3. Custom Model Pipeline Blueprint (From Day 1 to Horizon)

Skip `ONNX` completely in the new Foundry repo. Train/prune/quantize directly in **PyTorch + `fla` (Flash Linear Attention) + SafeTensors**, and export directly into `WGSL` shards (`slab0/1/2` + `aux_params.bin` + `manifest.json`) verified bit-by-bit against PyTorch reference logits!

```
  [Frontier Teachers: Qwen3.5-9B / 397B / Claude]
                         │
        (Playwright + DOM + Crop Trajectories)
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Stage 1: Vocab Surgery + Agent SFT/Distill   │  Qwen3.5-4B (~2.15 GiB) & Qwen3.5-2B (~1.05 GiB)
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Stage 2: SpinQuant Rotation + WGSL-Exact QAT │  Mixed 2.5–3b FFN + Q4 Attn (~1.25–1.45 GiB 4B / ~0.75 GiB 2B)
  │          + Minitron Prune + MTP / EAGLE-3    │  Co-trained multi-token heads for 3x browser decode
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Stage 3: WebRL / Playwright GRPO + XGrammar  │  Verifiable DOM/URL/visual rewards in headless browser
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Direct PyTorch → WGSL Shard Exporter + Eval  │  SafeTensors -> Q4/Q3 slabs (<=128 MiB) + Ed25519 sign
  │ + PyTorch-vs-WGSL Logit Parity Gate          │  Automated test inside `yours-browser-extension`
  └──────────────────────────────────────────────┘
```

### Stage 0 — Direct SafeTensors → WGSL Exporter & Numerical Parity Harness (Build First!)
- Write a pure Python/NumPy/SafeTensors (or Rust CLI) converter duplicating [`repack_q4_rows`](../crates/tinygpt-wgsl-engine/src/lib.rs#L2003-L2037) and [`computeAuxLayout`](../src/offscreen/onnxWeightLoader.ts#L87-L133), emitting `manifest.json` (`version: 1`, `<=128 MiB` shards, `aux_params.bin`, `tokenizer.json`, `config.json`) directly from Hugging Face `Qwen/Qwen3.5-4B` or `Qwen/Qwen3.5-2B` BF16/FP16 checkpoints.
- **Why this matters:** Eliminates the multi-step `PyTorch → ONNX → ONNX Runtime → WASM repack` pipeline (~`248 MB` smaller, zero ONNX exporter bugs with Gated DeltaNet ops, and total control over group size, rounding, SpinQuant rotations, and vocab size!).
- **Golden Parity Check:** Export hidden activations and top-10 logits for 5 fixed prompts from PyTorch (with simulated WGSL Q4 rounding) and assert KL/cosine/max-abs match against `yours-browser-extension` Chromium WebGPU output.

### Stage 1 — Fast, High-ROI Specialization (~$50–$200 GPU budget)
1. **Vocabulary Surgery (`248,320 → 32,768` or `49,152` tokens):**
   - Keep English, code, HTML/CSS/ARIA tokens, JSON syntax, digits, and high-frequency ASCII/Unicode byte fallbacks; prune unused multilingual script tokens and remap `embed_tokens` / tied `lm_head` + `tokenizer.json`.
   - **Immediate Payoff:** Shrinks `slab2` from **`378.91 MiB → ~50–79 MiB`** (**`~300 MiB` free VRAM & download savings**) AND shrinks `lm_head` + `argmax_pass1/2` per-token latency by **5×–7.5×**!
   - Follow with a brief `0.5B–1B` token Continued Pre-Training (CPT) recovery warmup on clean code/web/instruction mix so embedding boundaries heal cleanly.
2. **Browser-Agent & Coding SFT / Logit Distillation (`enable_thinking=False`):**
   - Generate trajectories using `Qwen3.5-9B` / frontier teachers interacting with **compressed Accessibility Trees + `(id, role, name, cx, cy)` bounding boxes** AND **optional local viewport crops** (`Magnitude`-style visual grounding for `<canvas>`, SVG charts, unlabeled icon buttons).
   - **Data Mix (Balanced to Prevent Catastrophic Forgetting):**
     - `40%` **Extension-Native Browser Agent Trajectories:** Strict JSON tool calls (`click`, `type`, `scroll`, `extract`, `wait`, `check`), DOM element resolution, multi-turn ARIA state diffs, error recovery.
     - `25%` **Structured Extraction & RAG over Long HTML/Docs (8K–64K):** Needle retrieval, Markdown table/Zod extraction, multi-document synthesis (protects our 32K/128K NIAH strength!).
     - `25%` **Coding, Debugging & Concise Reasoning:** Keeps instruction-following sharp.
     - `10%` **General Chat & Safety/Refusal Calibration.**
   - Use **all-linear LoRA** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` + DeltaNet `in_proj_qkv/z/ab`, `out_proj`, `conv1d`, `A_log`, `dt_bias`) or full BF16 fine-tuning (`Qwen3.5-2B` fits full BF16 SFT with gradient checkpointing on a single 48–80 GB GPU; `4B` needs 1–2× 80 GB H100/A100 or FSDP/LoRA on 24 GB).

### Stage 2 — Kernel-Aligned Compression & Speculation (~$300–$1.5K GPU budget)
1. **SpinQuant / QuaRot Orthogonal Rotations ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406), [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)):**
   - Apply Walsh-Hadamard orthogonal matrices offline to weights and absorb inverse rotations into RMSNorm weights (or run a fast shared-memory $O(D \log D)$ Hadamard kernel on `H=2048`/`2560`). Flattens extreme activation channels that otherwise blow up 4-bit KV and 2–3 bit weights.
2. **WGSL-Exact QAT (Block-32 Q4 + Mixed-Precision 2.5–3 Bit DeltaNet FFN):**
   - Run **Quantization-Aware Training (QAT)** with Straight-Through Estimator (STE) simulating our **exact WebGPU math**: symmetric/asymmetric block-32 uniform quantization with **`scale` & `zero` rounded to IEEE FP16 (`torch.float16`)**, plus **INT8 / INT4 KIVI simulated KV quantization** on the 6–8 Full-Attn layers!
   - Use **sensitive layer mixed precision:** keep Full-Attn layers, `lm_head`, and `in_proj_ab` at Q4 (`f16` block scales), while training the FFN `gate/up/down` matrices (`~64%` of model size!) in **2-bit or 3-bit (`IQ3`/`Q3` or 2-bit block-32 LUT)**. Drops `Qwen3.5-4B` total weights from **`2.45 GiB → ~1.25–1.45 GiB`** and `Qwen3.5-2B` to **`~0.75–0.90 GiB`**!
3. **Minitron Structured Width/Depth Pruning ([arXiv:2407.14679](https://arxiv.org/abs/2407.14679)):**
   - If `Qwen3.5-2B` lags on complex multi-tab planning while `4B` is slightly heavy for laptops, prune `Qwen3.5-4B` down to `hidden=2048`, `intermediate=6144` or `7168`, `layers=24–28` using activation-importance ranking + forward-KL logit & cosine hidden-state distillation.
4. **Co-Trained MTP (Multi-Token Prediction) / EAGLE-3 Draft Head ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)):**
   - Train 1 lightweight EAGLE-3 fusion block (or 2–3 shared-vocab MTP heads reusing our pruned `49K` `slab2` `lm_head`!) directly on our browser-agent trajectories (`~50–90 MB` Q4 overhead). Combined with zero-weight **SuffixDecoding** in Rust WASM, this yields **`2.5×–4.5×` effective decode tokens/s** at practically `1×` memory-bandwidth cost in `yours-browser-extension`.

### Stage 3 — Verifiable Online Browser RL (`WebRL` + `GRPO`) & Visual Crop Grounding
- **Playwright GRPO Environment (Inspired by `WebRL` [arXiv:2411.02337](https://arxiv.org/abs/2411.02337)):**
  - Spin up containerized synthetic + cloned web tasks (forms, e-commerce carts, multi-page navigation, dynamic canvas/table widgets, authentication flows).
  - Run **Group Relative Policy Optimization (GRPO)** over groups of `G = 8–16` live trajectories with **100% deterministic verifiable rewards**:
    - `+1.0` task goal verified via DOM state / URL / local backend DB / Zod extraction exact match;
    - `+0.1` valid schema & existing `target_id` / accurate `(x, y)` click hit;
    - `-0.1 × extra_steps` latency penalty + `-0.5` hallucinated selector / infinite loop penalty.
- **Native Early-Fusion Visual Crop Grounding (Inspired by `OS-Atlas` [arXiv:2410.23218](https://arxiv.org/abs/2410.23218), `UI-TARS` [arXiv:2501.12326](https://arxiv.org/abs/2501.12326), & `magnitude.dev`):**
  - Instead of converting the entire 4K viewport into 2,000+ vision tokens every step, fine-tune on **336×336 / 512×512 targeted element/region crops** paired with accessibility snippets for `<canvas>`, SVG charts, CAPTCHA-free sliders, and unlabeled icon buttons (using the native Qwen3.5 vision tower — and keep vision weights in an optional lazy-loaded `vision.bin` shard so text-only tabs don't pay VRAM for it!).

---

## 4. Lessons Mapped from Frontier Projects ([RESEARCH.md](./RESEARCH.md))

| Frontier Project | Core Insight | Actionable Requirement for Custom Model Pipeline |
| :--- | :--- | :--- |
| **Strata** (`Niko1221/Strata`, [arXiv:2508.18572](https://arxiv.org/abs/2508.18572), `Klavis`) | Native **MTP speculative decoding**, coalesced context state caching, and **progressive tool disclosure** (don't bloat prompt with 50 tool schemas) | Co-train **MTP heads** using the pruned tied `lm_head`; train the model on **2-stage tool routing** (category → active page tools) instead of giant static system prompts |
| **magnitude.dev** | Vision-first `(x, y)` coordinates + Planner/Executor split + post-action **`.check()` verification loop** | Include explicit `.check(expected_visual_or_dom_state)` verification turns in SFT/GRPO data so the tiny local model acts as a sub-second self-verifying **Executor** |
| **exo** (`exo-explore/exo`) | Dynamic memory-weighted partitioning across heterogeneous nodes (`slab0`/`slab1` clean layer boundaries) | Always keep strict homogeneous block boundaries (`layers = 24` or `32` cleanly divisible across 2 slabs) and preserve **dual-tier `2B` + `4B` compatibility** |
| **ktransformers** (`kvcache-ai/ktransformers`, Marlin) | Offline-permuted Q4 memory layout (`RePack` once!) + kernel-matched linear algebra (`16×` block multiples) | Enforce `H`, `I`, `V` divisible by **512** (or minimally **128**) so SoA/Marlin Q4 packing in Foundry hits 100% warp occupancy with zero masked tail lanes |
| **rapid-mlx** (`raullenchai/Rapid-MLX`) | Pure native kernels + **Radix + DeltaNet state prefix caching** (`~0.08s` TTFT) + 17 strict tool-call parsers | Structure system prompt + site accessibility headers as **immutable cacheable prefixes** at turn boundaries, and align training outputs with **XGrammar-2** strict JSON schemas |

---

## 5. Starter Foundry Repository Layout & Extension Graduation Gates

### Recommended New Repository Structure (`yours-model-foundry`)

```text
yours-model-foundry/
├── pyproject.toml                   # PyTorch, fla-org/flash-linear-attention, trl, transformers, safetensors, cryptography
├── configs/
│   ├── qwen35_4b_agent.yaml         # Day-1 target (32L, H=2560, V=49152)
│   └── qwen35_2b_edge.yaml          # Speed/laptop target (24L, H=2048, V=49152)
├── src/foundry/
│   ├── vocab_surgery.py             # Prune 248320 -> 49152/32768 + build updated tokenizer.json & tied weights
│   ├── spinquant_rotate.py          # Hadamard orthogonal rotation & RMSNorm absorption
│   ├── qat_wgsl.py                  # Exact block-32 Q4 (f16 scale/zero) & KIVI KV fake-quant modules
│   ├── prune_minitron.py            # Width/depth importance scoring & KL+hidden distillation
│   ├── train_sft_distill.py         # Multi-task DOM/a11y + crop grounding SFT/distillation
│   ├── train_grpo_playwright.py     # Headless Playwright verifiable RL loop (WebRL/GRPO)
│   ├── train_eagle_mtp.py           # Lightweight EAGLE-3 / MTP draft head training
│   └── export_wgsl_bundle.py        # SafeTensors -> slab0/1/2 (<=128 MiB shards) + aux_params.bin + Ed25519 manifest
├── evals/
│   ├── parity_reference.py          # Export PyTorch reference hidden states & top-10 logits for WebGPU parity
│   ├── bench_webarena_lite.py       # Playwright end-to-end task success, JSON validity, & step-efficiency eval
│   └── bench_niah_longctx.py        # 32K / 128K needle-in-a-haystack & multi-tab retrieval gate
└── scripts/
    ├── verify_manifest.py           # Check shard <=128 MiB, no reserved Win32 filenames (`aux.bin`), SHA-256 + Ed25519
    └── publish_r2.py                # Upload pre-repacked immutable shards to Cloudflare R2 + signed manifest
```

### Mandatory Graduation Gates Before Shipping Any Custom Model Into `yours-browser-extension`

1. **PyTorch-vs-WGSL Parity Gate:** `maxAbsError < 1e-3` on single-step logits (using simulated Q4 weights) across both DeltaNet and Full-Attn layers; `0` NaN/Inf across FP16 intermediates.
2. **Long-Context NIAH Gate:** **100% pass at 32K** (and **>= 95% pass at 128K** for 4B) — proves DeltaNet state + INT8/INT4 KV rotations haven't damaged long-range retrieval.
3. **Agent Execution & Grammar Gate:** `>= 99.5%` valid structured tool JSON even without constrained decoding (and `100%` with WASM XGrammar), plus higher task success / fewer steps than stock `Qwen3.5-4B` on held-out DOM + visual crop tasks.
4. **Bundle & Bandwidth Budget Gate:** Shards `<= 128 MiB`, `aux_params.bin` naming, zero compression on `.bin` shards, `<= 1.5 GiB` total weight target (`<= 1.0 GiB` for 2B), `100%` SHA-256 + Ed25519 signature pass.
5. **Browser Smoothness & Speed Gate (via `yours-browser-extension` Playwright E2E):** Preserves **`144 FPS` / `0%` jank (>33 ms)** in `worker / balanced` mode with `0` console/offscreen errors and clear decode/TTFT speedup over baseline.
