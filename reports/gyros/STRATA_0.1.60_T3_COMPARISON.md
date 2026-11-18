# Gyro-S T3 comparison on Strata engines 0.1.59 and 0.1.60

Measured on 2026-11-18 on the RTX 5090 Laptop GPU (24 GB) using Windows-native CUDA 13. **Retained engine 0.1.59.** Across three pairs, 0.1.60 cut uncached prefill throughput by a median 23.9%, increased decode throughput by 0.5%, and reduced total T3 time by 1.6%. The candidate fell short of the planned 10% wall-time improvement.

## Setup

A used the pinned [v0.2.13 engine 0.1.59](https://github.com/Niko1221/Strata/releases/tag/v0.2.13). B used the Windows CUDA 13 engine 0.1.60 from [v0.2.14](https://github.com/Niko1221/Strata/releases/tag/v0.2.14), build `09b603886925+cuda13.0`. [v0.2.15](https://github.com/Niko1221/Strata/releases/tag/v0.2.15) ships the same engine. Both arms used Control 0.1.59, the same Gyro-S TQ1_0 GGUF and `experts.bin` pack, expert ranking, DFlash2 drafter and settings, 128K context, and int8 KV. The candidate ran from an isolated checkout and Windows storage directory. Its signed-engine probe, all 117 verification tests, and the API and pi smoke checks passed.

The [P2 workload](GGUF_IN_PLACE_BASELINE.md#real-t3-agentic-baseline) ran in fresh T3 threads against the read-only fixture at `32ecd71726fee16f335484c1f95cbce126d461f6`. Each attempt used pi, thinking `high`, the explicit Gyro-S alias, a fresh nonce, and a fresh engine through llama-swap. Pair order was A/B, B/A, A/B. Total T3 time is wall time from T3 run start to completion after the one-token warm-up request. Prefill throughput counts newly evaluated tokens only. Decode throughput counts engine-generated tokens and excludes single-token probes.

## Results

| Run | Engine | Uncached prefill (tok/s) | Decode (tok/s) | Total T3 time (s) | Generated tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | 0.1.59 | 2,176 | 140.4 | 101.4 | 8,752 |
| B1 retry | 0.1.60 | 1,639 | 141.1 | 138.9 | 12,583 |
| B2 | 0.1.60 | 1,640 | 142.9 | 125.2 | 10,699 |
| A2 | 0.1.59 | 2,156 | 140.9 | 133.1 | 13,012 |
| A3 | 0.1.59 | 2,124 | 141.6 | 122.9 | 11,872 |
| B3 | 0.1.60 | 1,787 | 135.1 | 121.0 | 10,303 |

- **Prefill regressed in all three pairs.** The initial ~8.8K-token prompt took 4.04–4.12 s on A and 5.09–5.29 s on B. At 64–96K context, median later-turn prefill fell from 1,914 to 1,088 tok/s.
- **Decode throughput was close overall.** B1 generated 44% more tokens and finished 37.6 s later than A1. B2 and B3 generated fewer tokens and finished 7.9 s and 2.0 s sooner than their paired baselines.
- **Cold llama-swap requests stayed near 31.6 s.** The engine placed 12.15 GiB of experts in RAM in each candidate run and 12.47–12.94 GiB in the baseline runs. The benchmark measured the deployed engine behavior with automatic expert placement. It did not isolate the effect of placement from prefill kernel changes.

All six scored runs passed the automated read and nonce audit. Manual review found unsupported or overstated claims in both arms, so answer quality was graded mixed. The original B1 attempt ended in 0.21 s before inference because the isolated trial lacked the production Host allowlist. The retry used the same allowlist as A, and the failed attempt remains in the raw evidence.

After the final run, the original 0.1.59 server was restored and verified to answer through llama-swap. Production pins and defaults stayed unchanged. Raw evidence and per-run context buckets were retained in the ignored `logs/gyros/` and `logs/strata-ab/` directories, with the paired summary in `logs/strata-ab/comparison.json`. See also the [earlier experts.bin baseline](baseline.md).
