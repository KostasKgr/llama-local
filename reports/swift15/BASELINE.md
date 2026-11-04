# Swift 1.5: agentic baseline, 2026-11-04

This is the comparison baseline for Swift's saved two-slot unified KV profile.
It is run C1 of the no-pad study and is **one run**, not an average or a bound
on normal variation. The previous single-slot reference is retained as
[NO_KV_UNIFIED_BASELINE.md](NO_KV_UNIFIED_BASELINE.md).

## Setup

- Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`.
- Runtime: Windows-native llama.cpp b12001 CUDA 13.4, commit `eb4e244be`.
- Two server slots sharing one unified 131072-token KV pool, main K/V Q8_0, MTP
  enabled with maximum depth 3, batch/microbatch 2048/512, two context
  checkpoints, vision projector loaded and RAM prompt cache limit 4096 MiB.
- Hardware: RTX 5090 Laptop GPU. User-reported setup: Omen Performance mode,
  lid open, built-in screen off and no external cooling pad.
- Workload: one T3/PI thread with xhigh thinking and the read-only investigation
  prompt in [AGENTIC_PROMPT.md](AGENTIC_PROMPT.md), against fixture commit
  `32ecd71726fee16f335484c1f95cbce126d461f6`. The second slot had no benchmark
  workload. The model was warmed first.
- T3 run: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:dd36915bd406615fed0a1a5108ceb63ea93aa7057a12a01ee6a3710077d97f05>

## Throughput

The first prompt above 98304 tokens was the stop trigger and is excluded. The
43 completed requests counted here reached a largest prompt of 96253 tokens;
89189 prompt tokens were evaluated and 10831 tokens decoded. Rates are total
phase tokens divided by summed request phase durations. Context ranges refer to
full prompt length. Later-turn prefill excludes the initial full prefill.

| Prompt context | Decode | Later-turn prefill |
| --- | ---: | ---: |
| <32K | 78.4 tok/s | 1225.1 tok/s |
| 32–64K | 65.4 tok/s | 849.2 tok/s |
| 64–96K | 47.4 tok/s | 563.1 tok/s |

The initial 9902-token full prefill was 1501.2 tok/s. Across all counted
requests, prefill was 803.95 tok/s and decode 49.90 tok/s. Overall rates depend
on the context and output mix. MTP acceptance was 0.6832; the fixture's Git state
was unchanged.

`summary-96k.json` reports 1305.92 tok/s for <32K prefill because the server did
not flag this run's first prefill as reprocessing, so `agentic_bench.py finish`
counted it as a later turn. Use the 1225.1 tok/s figure above, and exclude the
first request explicitly when comparing new runs.

## Thermals and memory

GPU temperature at the benchmark start, after warm-up, was 52°C. Busy-sample
median/max temperature was 70/77°C and median/max power was 142.1/189.01 W.
Whole-GPU sampled memory peaked at 20975 MiB. Software power-cap flags were
active in 99.7% of busy samples; no software or hardware thermal slowdown was
observed. Fan RPM was unavailable.

The log records 87 checkpoint creations and 8 restores before the cutoff. Each
checkpoint was 149.626 MiB and up to two were retained in the active slot
(299.252 MiB). Eight restore log brackets had a 24.00 ms median and 40.96 ms
maximum; these intervals include the logged checking step and are not isolated
restore durations. Checkpoint-creation execution time is not logged.

## Limits

- One run has no measured variance, and generated histories differ between runs.
- It started warmer than the single-slot reference (52°C versus 44°C).
- The bounded run was intentionally interrupted, so this is not a
  task-completion or answer-quality result.
- Only one long conversation was active. It does not measure two long
  conversations or a previously occupied long second slot.

## Evidence

Ignored run evidence: `logs/swift15/agentic-20261104T201547Z-no-pad-C1/`.
`summary-96k.json`, `requests-96k.json`, `server-excerpt.log`, `gpu.csv`,
`thermal.csv` and `checkpoint-summary.json` contain the measurements. The
study's [visual partial comparison](NO_PAD_COMPARISON.html) is retained in the
repository.
