# Head-to-Head Live WebGPU Benchmark: `EXP-004` vs `EXP-003c` vs Stock `Qwen3.5-4B`

Generated: 2026-11-26T23:28:40.081Z

## Footprint & Model Configuration

| Metric | Stock `Qwen3.5-4B` | `EXP-003c` Tool-QAD | `EXP-004` GSPO-RL | `EXP-004` vs Stock |
|---|---:|---:|---:|---:|
| Vocab Size | 248,320 | 81,920 | 81,920 | -67.0% |
| `slab2` (`lm_head`) | 378.9 MiB | 125.0 MiB | 125.0 MiB | -253.91 MiB |
| Est. Runtime VRAM | 2567.36 MiB | 2313.47 MiB | 2313.47 MiB | -253.89 MiB |
| Cold Load Time | 10106.2 ms | 8474.6 ms | 13625.2 ms | 3519 ms |

## Live WebGPU Workloads (Direct / Thinking / Multi-Tab Tools)

| Workload | Stock TTFT / tok/s | `EXP-003c` TTFT / tok/s | `EXP-004` TTFT / tok/s | Pass (Stock / 003c / 004) |
|---|---:|---:|---:|:---:|
| Direct Chat (`enableThinking=false`) | 246.5ms / 79.29 | 89.8ms / 109.2 | 94ms / 80.74 | true / true / true |
| Adaptive Thinking (`enableThinking=true`) | 81.9ms / 83.78 | 79.7ms / 109.99 | 84.7ms / 89.11 | true / true / true |
| Multi-Tab Tool Chain (`list_tabs`+`read_tab`+`run_sandbox`) | 85.9ms / 83.51 | 85.5ms / 79.78 | 86ms / 82.76 | true / true / true |
| **3-Workload Mean** | **138.1ms / 82.19 tok/s** | **85ms / 99.66 tok/s** | **88.2ms / 84.2 tok/s** | **2.45% vs Stock** |

Runtime errors during run: 0
