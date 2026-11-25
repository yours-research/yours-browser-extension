<img src="assets/icon.svg" width="64" height="64" alt="">

# TinyGPT Extension (v0.1 POC)

Chromium Manifest V3 extension running **Qwen3.5-4B** (`onnx-community/Qwen3.5-4B-ONNX-OPT`) entirely inside the browser via **WebGPU**, **OPFS**, and **`@assistant-ui/react`**.

## Architecture

1. **Background Service Worker** ([src/background/service-worker.ts](src/background/service-worker.ts)) — Spawns the persistent Offscreen Document on demand (`ENSURE_OFFSCREEN`) and opens the Side Panel on toolbar icon click.
2. **Offscreen Document** ([src/offscreen/main.ts](src/offscreen/main.ts)) — Hosts the WebGPU inference engine ([src/offscreen/inferenceEngine.ts](src/offscreen/inferenceEngine.ts)), elevated `maxBufferSize` / `maxStorageBufferBindingSize` wrapper ([src/offscreen/webgpuLimits.ts](src/offscreen/webgpuLimits.ts)), streaming OPFS weight cache ([src/offscreen/opfsCache.ts](src/offscreen/opfsCache.ts)), chunked prefill loop, and hybrid Gated DeltaNet + KV cache management.
3. **Side Panel UI** ([src/sidepanel/App.tsx](src/sidepanel/App.tsx)) — Minimal React + `@assistant-ui/react` (`useLocalRuntime` + cumulative `ChatModelAdapter`) connected over a long-lived `chrome.runtime.connect` port, with live Telemetry Bar (`tok/s`, `TTFT`, `Active Context Tokens`, `Est. VRAM / KV Usage`, `Model Load State`) and Lab drawer for Phase 0–2 benchmarks.

## Build & Verify

```bash
npm install
npm run typecheck
npm test
npm run build
```

## Load Unpacked in Chrome / Chromium

1. Run `npm run build` to generate `dist/`.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Click **Load unpacked** and select `dist/`.
4. Click the **TinyGPT** extension icon to open the Side Panel.

## POC Verification Checklist

- **Phase 0 Spike**: Open **Lab → Phase 0 Spike** and click **Probe WebGPU Limits** (`shader-f16`, `subgroups`, elevated `maxBufferSize`) and **Run 2.5 GiB OPFS Spike**, then restart the browser and click **Verify After Restart**.
- **Phase 1 Baseline**: Click **Load Model** to stream `onnx-community/Qwen3.5-4B-ONNX-OPT` (`q4f16`, text-only `embed_tokens` + `decoder_model_merged`) into OPFS and WebGPU; verify reload from OPFS on next browser restart and test streaming chat + Stop cancellation.
- **Phase 2 Context & Speed**: Open **Lab → Bench & NIAH** to run the **1K / 8K / 32K / 64K Sweep** (with chunked prefill) and **32K Needle-in-a-Haystack** test, and review the theoretical memory-bandwidth utilization Decision Gate metric.
