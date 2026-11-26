# Live Chrome WebGPU Benchmark: `EXP-003c` (`Pragma-1-4B` v81920) vs Stock `Qwen3.5-4B` (v248320)

Generated: 2026-11-26T19:28:58.000Z

## Architecture & VRAM Footprint

| Metric | Stock `Qwen3.5-4B` | `EXP-003c` (`Pragma-1-4B`) | Delta |
|---|---:|---:|---:|
| Vocab Size | 248,320 | 81,920 | -67.0% |
| `slab2` (`lm_head`) Size | 378.91 MiB | 125.00 MiB | -253.91 MiB (3.03x smaller) |
| Est. Runtime VRAM | 2567.36 MiB | 2313.47 MiB | -253.89 MiB |
| Cold Load Time | 7393.9 ms | 7301 ms | -92.9 ms |

## Live WebGPU Performance & Accuracy

| Workload | Stock TTFT / Decode | `EXP-003c` TTFT / Decode | Stock FPS (Jank) | `EXP-003c` FPS (Jank) | Stock / EXP Pass |
|---|---:|---:|---:|---:|:---:|
| Direct Chat (`enableThinking=false`) | 323.4ms / 80.55 tok/s | 77.1ms / 109.12 tok/s | 144.01 (0%) | 144.21 (0%) | true / true |
| Adaptive Thinking (`enableThinking=true`) | 133ms / 69.26 tok/s | 75.1ms / 94.12 tok/s | 144 (0%) | 144.11 (0%) | true / true |
| Multi-Tab Tool Chain (`list_tabs`+`read_tab`+`run_sandbox`) | 81.5ms / 77.52 tok/s | 177.4ms / 56.13 tok/s | 144 (0%) | 144.18 (0%) | true / true |
| **3-Workload Mean** | **179.3ms / 75.78 tok/s** | **109.9ms / 86.46 tok/s** | — | — | **+14.09% decode** |

Runtime errors during run: 0
