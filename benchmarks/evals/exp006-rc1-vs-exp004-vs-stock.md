# Head-to-Head Live WebGPU Benchmark: `EXP-006` RC1 (`Pragma-1-4B`) vs `EXP-004` vs Stock `Qwen3.5-4B`

Generated: 2026-11-27T08:12:38.580Z

## Footprint & Model Configuration

| Metric | Stock `Qwen3.5-4B` | `EXP-004` GSPO-RL | `EXP-006` RC1 (`Pragma-1-4B`) | `EXP-006` vs Stock |
|---|---:|---:|---:|---:|
| Vocab Size | 248,320 | 81,920 | 81,920 | -67.0% |
| `slab2` (`lm_head`) | 378.9 MiB | 125.0 MiB | 125.0 MiB | -253.91 MiB |
| Est. Runtime VRAM | 2562.2 MiB | 2313.47 MiB | 2313.47 MiB | -248.73 MiB |
| Cold Load Time | 8389.4 ms | 13140.3 ms | 6719.4 ms | -1670 ms |

## Live WebGPU Workloads (Direct / Thinking / Multi-Tab Tools)

| Workload | Stock TTFT / tok/s | `EXP-004` TTFT / tok/s | `EXP-006` RC1 TTFT / tok/s | Pass (Stock / 004 / 006) |
|---|---:|---:|---:|:---:|
| Direct Chat (`enableThinking=false`) | 77.2ms / 84.53 | 85ms / 69.44 | 74.7ms / 70.77 | true / true / true |
| Adaptive Thinking (`enableThinking=true`) | 86.6ms / 67.46 | 158.5ms / 57.17 | 145.2ms / 59.55 | true / true / true |
| Multi-Tab Tool Chain (`list_tabs`+`read_tab`+`run_sandbox`) | 82.3ms / 80.32 | 84ms / 59.25 | 85.4ms / 59.43 | true / true / true |
| **3-Workload Mean** | **82ms / 77.44 tok/s** | **109.2ms / 61.95 tok/s** | **101.8ms / 63.25 tok/s** | **-18.32% vs Stock (2.1% vs 004)** |

Runtime errors during run: 0
