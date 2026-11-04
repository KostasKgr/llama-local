# Swift 1.5: no-unified-KV agentic baseline, 2026-11-04

This is a historical measurement and comparison baseline, not the current
launcher configuration. It is **one run**, not an average or a bound on normal
variation. The later planned repetitions were stopped at the user's request.

## Setup

- Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, SHA-256
  `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`.
- Runtime: Windows-native llama.cpp b12001 CUDA 13.4, commit `eb4e244be`.
- One server slot, unified KV **off**, total context 131072, main K/V Q8_0,
  MTP enabled with maximum depth 3, batch/microbatch 2048/512, vision projector
  loaded and RAM prompt cache limit 4096 MiB.
- Hardware: RTX 5090 Laptop GPU. User-reported setup: Omen Performance mode,
  lid open, built-in screen off and no external cooling pad.
- Workload: one T3/PI thread with xhigh thinking and the read-only investigation
  prompt in [AGENTIC_PROMPT.md](AGENTIC_PROMPT.md), against fixture commit
  `32ecd71726fee16f335484c1f95cbce126d461f6`. The model was warmed first;
  no other benchmark requests were intentionally sent during the run.
- T3 run: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:015797243f961dd93737a603ff39e4c36d2fb99b2bab64845b4ba4afec48a174>

## Throughput

The first prompt above 98304 tokens was the stop trigger and is excluded. The
39 completed requests counted here reached a largest prompt of 95730 tokens;
88537 prompt tokens were evaluated and 13038 tokens decoded. Rates are total
phase tokens divided by summed request phase durations. Context ranges refer to
full prompt length. Later-turn prefill excludes the initial full prefill and
other logged full reprocessing.

| Prompt context | Decode | Later-turn prefill |
| --- | ---: | ---: |
| <32K | 81.20 tok/s | 1323.10 tok/s |
| 32–64K | 69.31 tok/s | 916.67 tok/s |
| 64–96K | 44.50 tok/s | 577.55 tok/s |

The initial 9902-token full prefill was 1472.82 tok/s. Across all counted
requests, prefill was 818.18 tok/s and decode 46.09 tok/s. Overall rates depend
on the context and output mix. MTP acceptance was 0.6254; the fixture's Git state
was unchanged. The bounded run was intentionally interrupted, so this is not a
task-completion or answer-quality result.

## Thermals and memory

GPU temperature at the benchmark start, after warm-up, was 44°C. At one-second
sampling resolution, busy-sample median/max temperature was 72/75°C and
median/max power was 142.5/190.24 W. Whole-GPU sampled memory peaked at 20377
MiB. Software power-cap flags were active in about 98.9% of busy samples; no
software thermal slowdown, hardware thermal slowdown, hardware slowdown or
hardware power-brake flags were observed in the separate reason sampler.
The separate reason sampler began roughly 21 seconds after the thread; the
benchmark's GPU sampler covered the entire bracket. Fan RPM was unavailable.

The log records 79 checkpoint creations and 3 restores before the cutoff.
Each checkpoint was 149.626 MiB and up to two were retained in the active slot
(299.252 MiB). The one logged prompt-cache state was 450.848 MiB, including
associated state; it must not be added to the active-slot checkpoint figure as
an estimate of total system memory without measuring the allocations. Its
prompt-cache update took 117.53 ms. Three checkpoint restore log brackets had
a 26.70 ms median and 26.73 ms maximum; these intervals include the logged
checking step and are not isolated restore durations. Checkpoint-creation
execution time is not explicitly logged.

## Evidence and interpretation

Ignored run evidence: `logs/swift15/agentic-20261104T195630Z-no-pad-A1/`.
`summary-96k.json`, `requests-96k.json`, `server-excerpt.log`, `gpu.csv`,
`thermal.csv` and `checkpoint-summary.json` contain the measurements. The
study's [visual partial comparison](NO_PAD_COMPARISON.html) is retained in the
repository; detailed generated results remain in
`logs/swift15/no-pad-study-20261104/PARTIAL_REPORT.md`.

The no-pad A/B/C comparison has one completed run for each configuration plus
one interrupted partial B run. Different generated histories, prompt chunk
sizes, MTP acceptance and warm-up temperatures limit causal conclusions. The
baseline is useful as a reference for this exact configuration and setup; it
is not a universal speed guarantee for the model.
