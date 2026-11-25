# Gyro-S T3 benchmark — NVIDIA-SMI 615.83

Measured 2026-11-25 on NVIDIA-SMI 615.83, KMD 617.71, CUDA UMD 13.4, after a reboot that also installed Windows build 26300.9974 (KB5132061) and WSL 3.0.5.0. Comparison is with the [592.27 baseline](BASELINE_NVIDIA_SMI_592.27.md). All three runs use the Gyro-S Strata 0.1.59 profile, the same pinned fixture and P2 investigation, and pi with high thinking. Since the baseline, the profile has added CPU vision (7c1c769), which used about 1.1 GiB more RAM during load.

- **Run 1** used the Gyro-S server that was already loaded. The benchmark bracket's free-GPU guard was bypassed for it; no server was stopped.
- **Run 2** ran after llama-swap was stopped and restarted, with more RAM free. The bracket loaded a fresh engine.

| Measure | 592.27 | 615.83 run 1 | 615.83 run 2 | Run 1 change | Run 2 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Engine | Fresh, 31.6 s load | Already loaded | Fresh, 51.1 s load | — | — |
| CPU vision | Off | On | On | — | — |
| Start GPU temperature, °C | 37 | 45 | 45 | — | — |
| PCIe host-to-device probe, GB/s | 54.3 | 50.3 | 54.3 | -7.4% | 0.0% |
| Available Windows RAM before load, GiB | 21.8 | 15.0 | 19.6 | — | — |
| Available RAM when the RAM copy was sized, GiB | 20.27 | 12.32 | 16.84 | — | — |
| Resident experts in RAM, GiB | 12.99 | 6.07 | 10.84 | -53.3% | -16.6% |
| Experts outside the GPU cache kept in RAM | All | 63% (3.6 GiB on file) | All | — | — |
| Borrowed GPU slots backed in RAM | 3,391 / 3,391 | 0 / 3,391 | 1,229 / 3,391 | — | — |
| Expert reads from the model file during the run | 0 | 260,683 (~274 GB) | 37,133 (~39 GB) | — | — |
| CPU prefill share | On | Off | On | — | — |
| Prompt chunks | 8,192 | 1,024 above 8,192 tokens | 8,192 | — | — |
| Minimum available Windows RAM, GiB | 4.18 | 0.42 | 0.46 | — | — |
| T3 wall time, s | 140.5 | 140.3 | 123.5 | -0.1% | -12.1% |
| Initial / max prompt tokens | 8,838 / 85,756 | 9,197 / 81,318 | 9,203 / 80,847 | — | — |
| API requests / tool calls | 25 / 33 | 19 / 32 | 24 / 33 | — | — |
| Prefill tokens / seconds | 77,610 / 36.94 | 76,067 / 60.36 | 74,619 / 40.58 | — | — |
| Generated tokens / decode seconds | 14,441 / 101.74 | 9,699 / 78.21 | 10,869 / 80.66 | — | — |
| Draft acceptance | 58.2% | 60.4% | 59.0% | — | — |
| Decode GPU-cache hit rate | 99.37% | 99.15% | 99.33% | — | — |
| Initial full prefill, tok/s | 2,148.2 | 886.7 | 1,298.0 | -58.7% | -39.6% |
| Later prefill <32K, tok/s | 2,339.7 | 1,271.9 | 1,839.3 | -45.6% | -21.4% |
| Later prefill 32–64K, tok/s | 2,319.7 | 1,252.4 | 2,156.8 | -46.0% | -7.0% |
| Later prefill 64–96K, tok/s | 1,625.5 | 1,686.2 | 1,742.5 | +3.7% | +7.2% |
| Decode <32K, tok/s | 160.0 | 134.4 | 154.9 | -16.0% | -3.2% |
| Decode 32–64K, tok/s | 167.5 | 145.8 | 157.3 | -13.0% | -6.1% |
| Decode 64–96K, tok/s | 137.1 | 116.6 | 127.7 | -15.0% | -6.9% |
| Overall prefill, tok/s | 2,101.1 | 1,260.2 | 1,838.7 | -40.0% | -12.5% |
| Overall decode, tok/s | 141.9 | 124.0 | 134.7 | -12.6% | -5.1% |
| Busy GPU power, median / p90, W | 167.5 / 172 | 140.5 / 148 | 142.1 / 145 | -16.1% | -15.2% |
| Busy SM clock, median, MHz | 2,257 | 2,036 | 1,942 | -9.8% | -14.0% |
| Busy GPU temperature, median, °C | 70 | 69 | 68 | -1 °C | -2 °C |
| CPU package power, median / max, W | 31.6 / 40.8 | 70.0 / 94.7 | 70.4 / 147.0 | +121.5% | +122.9% |
| CPU cores power, median, W | 24.4 | 61.8 | 62.4 | +152.8% | +155.3% |
| Server CPU time, s | 1,644 | 1,361 | 1,440 | — | — |

Run 2 is the closer driver comparison: overall prefill was 12.5% slower and decode 5.1% slower than 592.27. GPU power and CPU power still differ, so the driver's effect remains confounded.

## Memory

Strata sizes its page-locked RAM copy once at load: available Windows RAM at that moment, minus `ram_keep_gib` (6 GiB) and a 0.25 GiB margin. The full copy has two parts: the 9.64 GiB of experts the GPU cache does not hold, and 3.31 GiB backing the GPU slots that prompts borrow.

Run 1 started 12 minutes after the reboot, with 15.0 GiB available and 18.2 GiB committed. Applications' memory was still resident after the boot. The engine kept 6.07 GiB and left 3.6 GiB of the least-used experts on the model file. Its prefill plan turned off the CPU prefill share and read prompts above 8,192 tokens in 1,024-token chunks. It read about 274 GB of experts from the file, mostly through the Windows file cache: system disk reads had a median of 33 MiB/s and peaked at 1.2 GiB/s. The tests run after the reboot did not leave memory behind. Available RAM before and after each of the two Gyro-S smoke loads was within 0.2 GiB.

Run 1 pushed available RAM down to 0.42 GiB, and Windows paged application memory out. Afterwards, with no model loaded, 11.1 GiB was in use while 20.1 GiB was committed, and 20.8 GiB was available. Run 2 started with 19.6 GiB available and kept every expert the GPU does not hold. Its file reads were only for borrowed slots without a RAM backup, 7,274 of them during the first prompt. The 592.27 run had also started after memory was squeezed: another Gyro-S load three minutes earlier had brought available RAM down to 6.2 GiB. Available RAM before a load therefore depends on what ran before it.

## Power

Run 2 removed most of the file reads but not the power difference. In both 615.83 runs the busy GPU stayed at about 141–148 W, against 168–172 W in the baseline, with a 10–14% lower SM clock at a similar temperature. After run 2, `nvidia-smi` reported an enforced power limit of 148.6 W out of a 175 W maximum. Its counters showed about 144 s each of software power capping and software thermal slowdown since boot. The baseline did not record these counters. CPU package power was 2.2 times the baseline's, for similar server CPU time. On this laptop the CPU and GPU share one power and cooling budget. Omen Command Center was still in Performance mode, as checked by the user after both runs.

CPU vision does not explain the difference. Gyro-S sessions with CPU vision on driver 592.27 (Nov 20 and Nov 23; not T3 benchmarks) had a median CPU package power of 33–35 W while the GPU was busy. On driver 592.27, the [GGUF-in-place T3 attempts](GGUF_IN_PLACE_BASELINE.md) 2–5 showed a similar pattern: 59–64 W CPU package power and 142 W busy GPU power. That report attributed it to more CPU-side expert work. After run 2, with Gyro-S loaded but idle, CPU package power was 19.8 W at 5% CPU utilization, against about 15 W before the baseline load. No background process used noticeable CPU. Whether the driver or the Windows update caused the higher CPU draw is not established. Fan noise was not recorded.

## Notes

Bucket request counts (later prefill / decode) for <32K, 32–64K and 64–96K: 592.27: 5 / 6, 7 / 7, 12 / 12; run 1: 7 / 8, 5 / 5, 6 / 6; run 2: 8 / 9, 6 / 6, 9 / 9. Rates count uncached prompt and engine-generated tokens, weighted by their timing. Wall time is T3 run start to completion. File reads come from the engine log's cumulative counters over the run. These are individual agentic runs with different generated histories, so elapsed time and token counts are not fixed-work comparisons.

Both new runs passed the mechanical audit: all five files read fully, nonce read and included, correct model, no forbidden tools, fixture unchanged. As with the baseline, their answers have not been manually reviewed for correctness.

Threads: [Gyro-S agentic NVIDIA-SMI 615.83](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:94a97714-b6cc-4487-ad6a-451d0a6e63d6) (run 1) and [Gyro-S agentic NVIDIA-SMI 615.83 02](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:fe4fb90568b92f9a578786a06755efab0d98227d823f9ed6fadb0fc8742b4ef0) (run 2). Raw evidence is Git-ignored: `logs/gyros/20261125T152805.582530Z-t3-nvidia-smi-615.83/` and `logs/gyros/20261125T155346.280383Z-t3-nvidia-smi-615.83-02/`. Telemetry uses sampled values and can miss peaks.
