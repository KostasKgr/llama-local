# Gyro-S T3 benchmark — NVIDIA-SMI 615.83

Measured 2026-11-25 on NVIDIA-SMI 615.83, KMD 617.71, CUDA UMD 13.4. Comparison is with the [592.27 baseline](BASELINE_NVIDIA_SMI_592.27.md). Both use the Gyro-S Strata 0.1.59 profile, the same pinned fixture and P2 investigation, pi with high thinking, and a fresh engine. The new run used the already-started server. Start temperature was 45 °C. The benchmark bracket's free-GPU guard was bypassed for that loaded server; no server was stopped.

| Measure | NVIDIA-SMI 592.27 | NVIDIA-SMI 615.83 | Change |
| --- | ---: | ---: | ---: |
| T3 wall time, s | 140.5 | 140.3 | -0.1% |
| Max prompt tokens | 85,756 | 81,318 | -5.2% |
| API requests / tool calls | 25 / 33 | 19 / 32 | — |
| Initial full prefill, tok/s | 2,148.2 | 886.7 | -58.7% |
| Later prefill <32K, tok/s | 2,339.7 | 1,271.9 | -45.6% |
| Later prefill 32–64K, tok/s | 2,319.7 | 1,252.4 | -46.0% |
| Later prefill 64–96K, tok/s | 1,625.5 | 1,686.2 | +3.7% |
| Decode <32K, tok/s | 160.0 | 134.4 | -16.0% |
| Decode 32–64K, tok/s | 167.5 | 145.8 | -13.0% |
| Decode 64–96K, tok/s | 137.1 | 116.6 | -15.0% |
| Overall prefill, tok/s | 2,101.1 | 1,260.2 | -40.0% |
| Overall decode, tok/s | 141.9 | 124.0 | -12.6% |
| Prefill tokens / seconds | 77,610 / 36.94 | 76,067 / 60.36 | — |
| Generated tokens / decode seconds | 14,441 / 101.74 | 9,699 / 78.21 | — |
| Median busy GPU power, W | 167.5 | 140.5 | -16.1% |
| Median busy GPU temperature, °C | 70 | 69 | -1 °C |
| Median CPU package power, W | 31.6 | 70.0 | +121.5% |
| Minimum available Windows RAM, GiB | 4.18 | 0.42 | — |
| Resident experts in RAM, GiB | 12.99 | 6.07 | -53.3% |

Bucket request counts (later prefill / decode): 592.27: 5 / 6, 7 / 7, 12 / 12; 615.83: 7 / 8, 5 / 5, 6 / 6. Rates count uncached prompt and engine-generated tokens, weighted by their timing. Wall time is T3 run start to completion. Initial prompts were 8,838 and 9,197 tokens respectively. These are individual agentic runs with different generated histories; elapsed time and token counts are not fixed-work comparisons.

The new run passed the mechanical audit: all five files read fully, nonce read and included, correct model, no forbidden tools, fixture unchanged. Its answer has not been manually reviewed for correctness, as in the baseline. The engine placed much less of the expert set in host RAM, and its log records file reads during the new run. Together with much lower available RAM and different GPU and CPU power, this means the comparison does not isolate the effect of the driver.

Thread: [Gyro-S agentic NVIDIA-SMI 615.83](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:94a97714-b6cc-4487-ad6a-451d0a6e63d6). Raw evidence: `logs/gyros/20261125T152805.582530Z-t3-nvidia-smi-615.83/` (Git-ignored). Telemetry uses sampled values and can miss peaks.
