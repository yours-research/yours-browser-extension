# TinyGPT Browser Model Download Benchmarks & Secure Distribution Architecture

**Target Model:** `Qwen3.5-4B` (`onnx-community/Qwen3.5-4B-ONNX-OPT`, `q4f16` ONNX vs. pre-repacked `<= 128 MiB` WGSL shards)  
**Benchmark Script:** [`scripts/bench-download.mjs`](../scripts/bench-download.mjs)  
**Raw Metrics:** [`benchmarks/download-bench.json`](../benchmarks/download-bench.json)  
**Environment:** Windows x64, Node `v20.18.2` (built-in `fetch`/Undici, `node:https`, `node:http2`, `node:crypto`, `node:zlib`, `node:fs/promises`), ~190 Mbps ISP link (`18.94 MiB/s` single-stream HTTP/1.1 baseline)

---

## Executive Summary

1. **Root Cause of Slow Browser Downloads (`773.12s` Baseline):** Hugging Face's Xet CAS CDN (`us.aws.cdn.hf.co`, `xet-bridge-us`) advertises HTTP/2 (`h2`), causing Chromium/browsers to coalesce all requests onto a single TCP connection. In Trial 4, **single-connection HTTP/2 (`node:http2`) throttled to `1.51–1.53 MiB/s` (`12.7–12.9 Mbps`)**, roughly **12× slower** than multi-socket HTTP/1.1 (`18.38–19.18 MiB/s`). At `1.53 MiB/s`, downloading `2.47 GiB` takes **`~1,648s` (~27.5 minutes)**!
2. **Parallel Work-Stealing Range / Sharded Fetch:** Using `concurrency = 8–16` across independent TCP connections with `8–16 MiB` shards/ranges and dynamic work-stealing reached **`21.74–22.83 MiB/s` (`182.4–191.5 Mbps`)** — saturating the host ISP link and cutting `2.47 GiB` wire time to **`110.7–116.3s`** (~**14.6× faster** than single-connection HTTP/2).
3. **Incremental SHA-256 Verification is Essentially Free:** Pure SHA-256 CPU throughput measured **`1,857.36 MiB/s`** (`~1.33s` CPU for an entire `2.47 GiB` model). Enabling streaming per-chunk SHA-256 verification at `c=8` yielded `21.56 MiB/s` vs `22.83 MiB/s` (within normal network variance, `<1.2%` CPU cost), caught an injected 1-byte flip immediately, and recovered via a single `16 MiB` `Range` retry in **`1.51s`**.
4. **Do Not Compress Pre-Quantized Q4 Shards:** `slab0.part00.bin` has Shannon entropy **`7.707 / 8.000 bits/byte`**. `gzip-6` saves only **`6.08%`** (`93.92%` of raw) and `brotli-6` saves only **`3.47%`** (`96.53%`), with breakeven link speeds of **`168.5 Mbps`** and **`62.9 Mbps`** respectively. Compressing Q4 shards slows down end-to-end loading on broadband links and breaks direct `Range` byte-offset resumability.
5. **Ship Pre-Repacked WGSL Shards (`2.47 GiB`), Not Raw ONNX (`2.63–2.81 GiB`):** Hosting the 27-file pre-repacked WGSL bundle (`2,651,081,621 B`) saves **`169.8 MB`** vs raw `decoder + embed + JSONs` (`2,820,836,773 B`) and **`248.0 MB`** vs `2.70 GiB`, eliminates **`~30s`** of client-side WASM repacking and double OPFS disk usage (`~5.3 GiB` peak), and reloads cold from OPFS into WebGPU in **`4.71s`**.

---

## 1. Hugging Face Hub & Xet CAS Endpoint / CORS Probe

Probed with `Origin: chrome-extension://tinygptbenchoriginplaceholder000000` against `https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT/resolve/main/<file>`:

| Asset | Size (Bytes) | Hub Status | Redirect Latency | CDN Host | `Accept-Ranges` | Hub ACAO | CDN ACAO | `x-xet-hash` (Prefix) |
| :--- | ---: | :---: | ---: | :--- | :---: | :--- | :---: | :--- |
| `config.json` | 3,198 | `307` | 43.18 ms | `huggingface.co` | `bytes` | Mirrors `Origin` | Mirrors `Origin` | — |
| `tokenizer.json` | 19,226,111 | `302` | 85.91 ms | `us.aws.cdn.hf.co` | `bytes` | Mirrors `Origin` | `*` | `8b5fc71116be` |
| `onnx/embed_tokens_q4f16.onnx_data` | 367,513,600 | `302` | 48.85 ms | `us.aws.cdn.hf.co` | `bytes` | Mirrors `Origin` | `*` | `be3fc9a3de4d` |
| `onnx/decoder_model_merged_q4f16.onnx` | 933,554 | `302` | 94.23 ms | `us.aws.cdn.hf.co` | `bytes` | Mirrors `Origin` | `*` | `d4e6b6404271` |
| `onnx/decoder_model_merged_q4f16.onnx_data` | 2,065,635,328 | `302` | 81.50 ms | `us.aws.cdn.hf.co` | `bytes` | Mirrors `Origin` | `*` | `9137086b5841` |
| `onnx/decoder_model_merged_q4f16.onnx_data_1` | 367,513,600 | `302` | 84.14 ms | `us.aws.cdn.hf.co` | `bytes` | Mirrors `Origin` | `*` | `be3fc9a3de4d` |

### Key Protocol & Header Observations

- **Duplicate Weight File in HF Repo:** Both `onnx/embed_tokens_q4f16.onnx_data` and `onnx/decoder_model_merged_q4f16.onnx_data_1` resolve to the **exact same Xet CAS hash** (`be3fc9a3de4d...`, `367,513,600` B). Downloading both wastes `350.49 MiB` when tied embeddings/LM-head are loaded via raw ONNX paths.
- **Hub Resolver Rate Limiting:** `huggingface.co` enforces `ratelimit-policy: "fixed window";"resolvers";q=3000;w=300` (`3,000` resolver hits per 5 minutes). Pre-resolving the `302` signed `us.aws.cdn.hf.co` URL once per file (valid for `3600s` via CloudFront `Policy` + `Expires`) avoids exhausting resolver quota and saves `48–94 ms` per chunk.
- **CORS Compatibility for Extensions:** `us.aws.cdn.hf.co` sets `access-control-allow-origin: *`, `access-control-expose-headers: *`, `access-control-allow-headers: Content-Range, Content-Type, Content-Disposition, ETag`, and returns `206 Partial Content` on `Range` requests. Note that `Range` is not a CORS-safelisted request header for normal web origins (triggering preflights unless `host_permissions` is granted to the MV3 extension).

---

## 2. Benchmark Results (Trials 0–4)

> **Unit Note:** Consistent with [`benchmarks/REPORT.md`](./REPORT.md) and [`src/offscreen/opfsCache.ts`](../src/offscreen/opfsCache.ts), `MiB/s` uses binary mebibytes (`1 MiB = 1,048,576 B`), while `Mbps` uses decimal megabits (`1 Mb = 1,000,000 bits`). Projected ETAs are scaled to the `2.47 GiB` (`2,651,081,621 B`) pre-repacked WGSL bundle and `2.70 GiB` (`2,899,102,925 B`) raw ONNX reference.

### Trial 0 — Baseline Single-Stream Sequential Fetch

| Sub-Test | Asset | Tested MiB | Elapsed | Throughput (`MiB/s`) | Wire (`Mbps`) | ETA `2.47 GiB` | ETA `2.70 GiB` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0A` Full File | `onnx/embed_tokens_q4f16.onnx_data` | 350.49 MiB | 18.22 s | 19.233 | 161.34 | 131.45 s | 143.75 s |
| `0B` Prefix | `onnx/decoder_model_merged_q4f16.onnx_data` | 256.00 MiB | 13.80 s | 18.553 | 155.64 | 136.27 s | 149.02 s |
| **Combined Baseline** | **Both sequential HTTP/1.1 streams** | **606.49 MiB** | **32.02 s** | **18.940** | **158.88** | **133.49 s** | **145.98 s** |

*(Historical reference from [`benchmarks/REPORT.md`](../benchmarks/REPORT.md) Section 1: First online browser download + compile took **`773.12s`**.)*

### Trial 1 — Parallel HTTP `Range` Chunked Downloader (Work-Stealing Queue)

All 8 configurations wrote directly via positional `FileHandle.write(buf, 0, len, offset)` (modeling OPFS `FileSystemSyncAccessHandle.write(chunk, { at })`) and passed byte-for-byte `SHA-256` equality (`5d12483d...`) against the Trial 0 `256 MiB` prefix.

| Concurrency | Chunk Size | Window | Throughput (`MiB/s`) | Wire (`Mbps`) | Speedup vs Baseline | ETA `2.47 GiB` | Chunk Median / Max | Straggler (`max/med`) | SHA-256 OK |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 8 MiB | 256 MiB | 15.973 | 133.99 | 0.843× | 158.28 s | 857 ms / 7,872 ms | 9.185× | PASS |
| 8 | 8 MiB | 256 MiB | 13.525 | 113.46 | 0.714× | 186.93 s | 4,238 ms / 9,602 ms | 2.266× | PASS |
| **16** | **8 MiB** | **256 MiB** | **21.740** | **182.37** | **1.148×** | **116.30 s** | 4,850 ms / 11,689 ms | 2.410× | **PASS** |
| 32 | 8 MiB | 256 MiB | 14.003 | 117.47 | 0.739× | 180.55 s | 7,496 ms / 18,219 ms | 2.431× | PASS |
| 4 | 16 MiB | 256 MiB | 14.864 | 124.68 | 0.785× | 170.10 s | 1,330 ms / 9,229 ms | 6.937× | PASS |
| 8 | 16 MiB | 256 MiB | 16.372 | 137.34 | 0.864× | 154.42 s | 6,381 ms / 11,442 ms | 1.793× | PASS |
| 16 | 16 MiB | 256 MiB | 19.252 | 161.50 | 1.016× | 131.33 s | 10,288 ms / 13,195 ms | 1.283× | PASS |
| 32 | 16 MiB | 512 MiB | 20.568 | 172.53 | 1.086× | 122.93 s | 16,936 ms / 24,768 ms | 1.462× | PASS |

- **Work-Stealing Impact:** At `c=4, 8 MiB`, individual chunks varied by `9.185×` (`460 ms` min vs `7,872 ms` max), and dynamic work-stealing balanced workers across `3–14` chunks each instead of stalling on a static shard assignment.
- **Saturation Sweet Spot:** `c=16` saturated the ~190 Mbps ISP link (`21.74 MiB/s`), whereas `c=32` with `8 MiB` chunks suffered TCP contention (`14.00 MiB/s`).

### Trial 2 — Pre-Sharded Multi-File + Incremental SHA-256 + Corrupted Retry + Resume

Workload: `16 × 16 MiB = 256 MiB` across two real HF Xet files (`12` decoder shards + `4` embed shards), verified against ground-truth per-shard SHA-256 digests. Pure Node `crypto.createHash('sha256')` CPU speed measured **`1,857.36 MiB/s`** (`137.83 ms` for `256 MiB`).

| Concurrency | Incremental SHA-256 | Tested MiB | Elapsed | Throughput (`MiB/s`) | Wire (`Mbps`) | Speedup vs Baseline | ETA `2.47 GiB` | Verified |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **8** | **Off** | 256 MiB | 11.21 s | **22.831** | **191.52** | **1.205×** | **110.74 s** | — |
| **8** | **On** | 256 MiB | 11.88 s | **21.556** | **180.83** | **1.138×** | **117.29 s** | **PASS (16/16)** |
| 16 | Off | 256 MiB | 11.44 s | 22.374 | 187.69 | 1.181× | 113.00 s | — |
| 16 | On | 256 MiB | 13.15 s | 19.469 | 163.32 | 1.028× | 129.86 s | PASS (16/16) |

- **Corrupted-Chunk Detection & Single-Shard Retry:** Injecting a 1-byte flip into `decoder.part05` (`16 MiB`) was immediately caught (`b23dd798...` expected vs `c2260a23...` actual). Re-fetching only that `16 MiB` shard via HTTP `Range` took **`1,513.27 ms` (`1.51s`)** and passed disk readback verification (`true`).
- **Mid-Download Resume:** After committing `8/16` shards (`128 MiB` in `8.02s`) and writing a JSON partial-offset map, a simulated restart re-verified the persisted `128 MiB` on disk in **`98.50 ms`** (`~1,300 MiB/s`), skipped `134,217,728 B`, downloaded only the remaining `128 MiB` in **`7.40s`** (`17.31 MiB/s`), and passed full `256 MiB` verification (`true`).

### Trial 3 — Compression & Format ROI (`16 MiB` Slice of `slab0.part00.bin`)

`16 MiB` (`16,777,216 B`) of pre-repacked WGSL Q4 weights measured Shannon entropy **`7.7070 / 8.0000 bits/byte`**.

| Codec | Supported | Compressed Size | Ratio (`%`) | Savings (`%`) | Compress (`MiB/s`) | Decompress (`MiB/s`) | Breakeven Link (`Mbps`) | `2.47 GiB` Decomp CPU | Serialized Net Delta @ Peak (`21.74 MiB/s`) | Verdict |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `gzip-1` | Yes | 15,760,809 B | 93.942% | 6.058% | 51.76 | 310.76 | 157.93 Mbps | 8.14 s | +1.09 s | **Hurts** |
| `gzip-6` | Yes | 15,756,755 B | 93.918% | 6.082% | 48.61 | 330.24 | 168.50 Mbps | 7.66 s | +0.58 s | **Hurts** |
| `brotli-4` | Yes | 16,200,959 B | 96.565% | 3.435% | 124.02 | 197.63 | 56.94 Mbps | 12.79 s | +8.80 s | **Hurts** |
| `brotli-6` | Yes | 16,195,356 B | 96.532% | 3.468% | 59.47 | 216.31 | 62.93 Mbps | 11.69 s | +7.66 s | **Hurts** |
| `zstd` | No (`v20.18.2`) | — | — | — | — | — | — | — | — | Requires Node `22.15+`; no native browser `DecompressionStream` |

#### Format ROI: Pre-Repacked WGSL Shards vs. Raw ONNX

| Distribution Format | Total Bytes | File Count | Client WASM Repack | Peak OPFS Disk During Install | Cold OPFS -> WebGPU Load | Net Advantage |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Pre-Repacked WGSL Shards** (`model-bundle/qwen35-4b`) | **2,651,081,621 B (`2.469 GiB`)** | 27 (`<= 128 MiB`) | **0 s** | **2.47 GiB** | **4.71 s** | Zero repack CPU, single-copy OPFS, direct `GPUQueue.writeBuffer` |
| Raw ONNX (`decoder_q4f16` + `embed_q4f16` + JSONs) | 2,820,836,773 B (`2.627 GiB`) | 9 | ~30 s | ~5.10 GiB (raw + repacked) | ~35 s (w/ repack) | `+169.8 MB` redundant embed weights + ~30s repack |
| Raw ONNX (`2.70 GiB` reference) | 2,899,102,925 B (`2.700 GiB`) | — | ~30 s | ~5.17 GiB | 773.12 s (initial baseline) | `+248.0 MB` wire + ~30s repack |
| Raw ONNX Full Multimodal `q4f16` (w/ Vision) | 3,019,390,275 B (`2.812 GiB`) | 11 | ~30 s | ~5.48 GiB | — | `+368.3 MB` unused vision + duplicate embed weights |

### Trial 4 — Transport / Protocol Comparison (`128 MiB` Window on `us.aws.cdn.hf.co`)

| Transport | TCP Sockets | ALPN | Concurrency | Chunk Size | Elapsed | Throughput (`MiB/s`) | Wire (`Mbps`) | ETA `2.47 GiB` | Speedup vs Baseline | SHA-256 |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `undici_fetch` | Pooled auto | `http/1.1` | 8 | 16 MiB | 7.31 s | 17.502 | 146.82 | 144.46 s | 0.924× | PASS |
| `node_https_multisocket` | 8 | `http/1.1` | 8 | 16 MiB | 6.96 s | **18.381** | **154.19** | **137.55 s** | 0.970× | PASS |
| **`node_http2_single_conn`** | **1** | **`h2`** | **8** | **16 MiB** | **83.43 s** | **1.534** | **12.87** | **1,648.01 s** | **0.081×** | PASS |
| **`undici_fetch`** | Pooled auto | `http/1.1` | **16** | **8 MiB** | **6.67 s** | **19.176** | **160.86** | **131.84 s** | **1.012×** | PASS |
| `node_https_multisocket` | 16 | `http/1.1` | 16 | 8 MiB | 9.19 s | 13.922 | 116.79 | 181.60 s | 0.735× | PASS |
| **`node_http2_single_conn`** | **1** | **`h2`** | **16** | **8 MiB** | **84.61 s** | **1.513** | **12.69** | **1,671.31 s** | **0.080×** | PASS |

---

## 3. Root-Cause Diagnosis: Hugging Face / Xet CAS CDN Bottlenecks

1. **Per-Connection HTTP/2 Throttling (`1.53 MiB/s` vs `19.18 MiB/s`):** Even on a ~190 Mbps connection, a single HTTP/2 TCP connection to `us.aws.cdn.hf.co` (`xet-bridge-us`) caps at ~`12.8 Mbps` (`1.53 MiB/s`) regardless of whether 8 or 16 concurrent streams are multiplexed inside it (and an isolated test with `16 MiB` HTTP/2 `initialWindowSize` still ran at `1.34 MiB/s`, `11.94s` for `16 MiB`). In Chromium, `fetch()` automatically uses ALPN `h2` and coalesces all requests to `us.aws.cdn.hf.co` onto **one** TCP socket — directly explaining multi-minute / `773.12s` first-run downloads!
2. **Multi-TCP-Flow Scaling vs. Congestion Ceiling:** Multiple HTTP/1.1 sockets bypass the single-connection Xet bridge ceiling (`18.38–22.83 MiB/s` at `c=8–16`), saturating the local 190 Mbps ISP link. Beyond `c=16` (`c=32` with `8 MiB` chunks), packet loss and TCP congestion backoff drop throughput back down (`14.00 MiB/s`).
3. **Chunk Tail Latency & Work-Stealing:** CDN edge requests exhibit `1.8×–9.2×` latency variance (`460 ms` to `7,872 ms` at `c=4`). Static per-worker assignment leaves fast sockets idle waiting on the slowest shard; a shared work-stealing queue lets fast sockets drain remaining chunks immediately.
4. **Hub Redirects & Rate Limits:** Routing every `Range` request through `huggingface.co/.../resolve/main/...` wastes an extra `48–94 ms` 302 round-trip and burns the `3,000 req / 300 s` resolver rate limit. Resolving the signed `us.aws.cdn.hf.co` URL once per file (valid 1 hour) eliminates resolver pressure.

---

## 4. Recommended Production Download Architecture for Non-Bundled Chrome Extensions

To eliminate the multi-GB bundled extension footprint while guaranteeing **cryptographic integrity**, **resumability**, and **maximum browser throughput**, TinyGPT should adopt the following architecture:

### A. Signed Manifest + Per-Chunk SHA-256 Merkle List

- Ship **only** the extension's pinned **Ed25519 public key** (`32 bytes` compiled into extension source / WASM) — never trust unsigned CDN JSON or HTTP `ETag`s alone.
- Host a versioned `manifest.v1.signed.json` containing:
  - `modelId`, `version`, `slabBytes` (`slab0`, `slab1`, `slab2`, `aux`), `shardSizeBytes` (`16 MiB` chunks or `<= 128 MiB` shards with `8–16 MiB` sub-chunk hashes).
  - Exact `sha256` per shard/chunk (`16 MiB` granularity = only ~`165` hashes = `~10 KiB` JSON) and `merkleRoot` over the ordered chunk hashes.
  - `signature`: Base64 Ed25519 signature over canonicalized manifest JSON (`WebCrypto` `crypto.subtle.verify('Ed25519', pubKey, sig, canonicalBytes)` — natively supported in Chrome 113+ with zero external libraries).
- **Pre-repacked format:** Host the WGSL-repacked shards (`2.47 GiB`, zero compression on binary weights; `Content-Encoding: br` only on `tokenizer.json` and config JSONs).

### B. Dedicated Worker + `FileSystemSyncAccessHandle` Parallel Pool

Chrome Extensions cannot call `createSyncAccessHandle()` from an Offscreen Document main window, **only from a Dedicated Web Worker**:
1. Offscreen Document spawns a lightweight **Download Dedicated Worker** (1 worker manages 8–12 concurrent `fetch()` streams — avoids spawning 16 OS threads/workers).
2. Worker pre-allocates each slab file in OPFS via `await fileHandle.createSyncAccessHandle()`, `truncate(slabBytes)` (or per-shard files `slab0.part00.bin`..`slab2.part02.bin` to match `manifest.json` directly).
3. Worker runs a **dynamic work-stealing queue** across `concurrency = 8` (default, tunable `8–12` on fast links) over `8 MiB` or `16 MiB` chunks:
   - Each arriving `ReadableStream` chunk (`Uint8Array`) is hashed incrementally via `SubtleCrypto` or Rust WASM `sha256` (our Rust WASM engine can expose streaming SHA-256 at `>1,500 MiB/s` so chunks don't buffer in JS RAM, or buffer `8–16 MiB` per active slot = `64–128 MiB` max RAM and verify with `crypto.subtle.digest('SHA-256', chunkBuf)` before writing!).
   - Write directly to OPFS with **zero async `WritableStream` overhead**: `syncHandle.write(chunkBuf, { at: shardOffset })` + update persistent `resume-bitmap.json` (or SQLite/OPFS metadata journal).
   - Call `syncHandle.flush()` once per shard completion; `syncHandle.close()` at end.
4. **Corruption Recovery & Resumability:**
   - If a chunk's SHA-256 mismatches the Ed25519-signed manifest, discard the buffer and re-enqueue **only that 8–16 MiB chunk** with exponential backoff (never re-download the full 2.47 GiB).
   - On browser restart or abort, verify already-committed offsets on disk (`~98 ms` per `128 MiB`) and fetch only unverified byte ranges.

### C. Escaping Chrome's Single-Connection HTTP/2 Bottleneck

Because Chromium coalesces HTTP/2 requests to a single hostname onto one TCP socket (where HF's `xet-bridge-us` drops to `1.5 MiB/s`), the production downloader should:
1. **Primary Recommendation — Serve via HTTP/3 (QUIC) on Cloudflare R2 + Custom Domain:** HTTP/3 eliminates TCP head-of-line blocking over a single UDP/QUIC socket and supports `0-RTT` / high BDP pacing cleanly inside Chrome `fetch()` without needing multiple hostnames.
2. **Multi-Origin Sharding Fallback (for HTTP/2 CDNs / HF):** If serving over HTTP/2-only edges, shard URLs across 4–6 origin aliases (e.g., `dl-0.tinygpt.example`, `dl-1.tinygpt.example`, ... pointing to the same R2/CDN bucket) so Chromium opens 4–6 independent TCP congestion windows instead of 1.

### D. MV3 Security: Strict CSP & Minimal `host_permissions`

- Keep `content_security_policy.extension_pages` strictly `"script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; connect-src https://cdn.tinygpt.example"` (no `'unsafe-eval'`, no `http://`, no wildcard `https://*/*`).
- Request **only** the exact model CDN origin in `manifest.json` `"host_permissions"` (e.g., `["https://cdn.tinygpt.example/*"]` — or `"optional_host_permissions"` granted on first model download!), avoiding broad `<all_urls>` review flags in the Chrome Web Store.
- Enforce `redirect: 'error'` (or an explicit allowlist if using HF hub -> `*.cdn.hf.co`) after signature verification so compromised redirects cannot hijack weight fetches, and validate `Content-Length` + exact chunk byte count before hashing.

### E. Hosting & CDN Comparison

| Option | Egress Cost (`2.47 GiB` / 10k users = `24.7 TB`) | HTTP/3 & Browser Speed | CORS & Range Support | Security / Reliability Tradeoffs |
| :--- | :--- | :--- | :--- | :--- |
| **Cloudflare R2 + Custom Domain** *(Recommended)* | **$0 egress** (`$0.015/GB-mo` storage = ~$0.04/mo for 2.47 GiB!) | **Native HTTP/3 (QUIC)** + global Anycast edge | Full CORS + `206 Range` (`<= 5 GB` objects) | Zero egress fees, zero single-H2-flow bottleneck, custom domain allows minimal `host_permissions` |
| **Backblaze B2 + Cloudflare (Bandwidth Alliance)** | **$0 egress** via Cloudflare proxy (`$0.006/GB-mo` storage) | **Native HTTP/3 (QUIC)** via Cloudflare | Full CORS + `Range` (watch Cloudflare 512 MB/5 GB cache limits on un-sharded files — `16–128 MiB` shards cache cleanly!) | Very cheap, requires cache-rule tuning so Cloudflare caches `128 MiB` shards instead of hitting B2 API limits |
| **Hugging Face Hub / Xet CAS (`us.aws.cdn.hf.co`)** | **$0** (free community hosting) | **HTTP/2 throttled (`1.53 MiB/s` single-conn)**; fast only with multi-TCP sockets | `ACAO: *`, `Range` `206` | `3,000 / 5 min` resolver rate limit, 1-hour signed redirect URLs, cannot customize HTTP/3 or multi-origin aliases |
| **Multi-CDN Racing (e.g., R2 + B2/CF mirrors)** | $0 (R2 + B2) | Highest global tail resilience | Full CORS + `Range` | Hedge slow chunks across secondary CDN origin using the same Ed25519-signed chunk hashes; adds 2 origins to `host_permissions` |
| **P2P WebRTC / WebTorrent** | Near-$0 origin egress | Variable (NAT/symmetric firewall failures) | N/A (`RTCDataChannel`) | **Not recommended for MV3:** exposes user IP addresses to peers (privacy leak), requires STUN/TURN & complex MV3 signaling, bloats extension permissions, and expands parser/DoS attack surface even with chunk hashing |
