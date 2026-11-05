# Swift 1.5: agentic baseline, 2026-11-04

This is the comparison baseline for Swift's saved two-slot unified KV profile on
b12110. It is run 1 of the [b12110 comparison](B12110_COMPARISON.md) and is **one
run**, not an average or a bound on normal variation. Run 2 is listed beside it as
a second data point. The previous reference is retained as
[B12001_BASELINE.md](B12001_BASELINE.md).

## Setup

- Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`.
- Runtime: Windows-native llama.cpp b12110 CUDA 13.4, commit `cd959bae9` (v0.9.0).
  Flash Attention over K/V rows was on, the default with unified KV.
- Two server slots sharing one unified 131072-token KV pool, main K/V Q8_0, MTP
  enabled with maximum depth 3, batch/microbatch 2048/512, two context
  checkpoints, vision projector loaded and RAM prompt cache limit 4096 MiB.
- Hardware: RTX 5090 Laptop GPU. Laptop setup assumed unchanged from the b12001
  baseline and not reconfirmed: Omen Performance mode, lid open, built-in screen
  off and no external cooling pad.
- Workload: one T3/PI thread with xhigh thinking and the read-only investigation
  prompt in [AGENTIC_PROMPT.md](AGENTIC_PROMPT.md), against fixture commit
  `32ecd71726fee16f335484c1f95cbce126d461f6`. The server was freshly started and
  warmed, and the second slot had no benchmark workload.
- T3 run: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:4eb5ee7bedd8c207e66c227676fa165124c33f2d3e43d1105df829678aea8984>

## Throughput

The thread finished on its own without reaching the 98304-token cap. Its 28
completed requests reached a largest prompt of 95534 tokens; 83678 prompt tokens
were evaluated and 16934 tokens decoded. Rates are total phase tokens divided by
summed request phase durations. Context ranges refer to full prompt length.
Later-turn prefill excludes the initial full prefill.

| Prompt context | Decode | Later-turn prefill | Run 2 decode | Run 2 later-turn prefill |
| --- | ---: | ---: | ---: | ---: |
| <32K | 81.3 tok/s | 1312.2 tok/s | 81.4 tok/s | 1292.2 tok/s |
| 32–64K | 68.6 tok/s | 879.0 tok/s | 61.7 tok/s | 873.9 tok/s |
| 64–96K | 47.0 tok/s | 594.6 tok/s | 46.2 tok/s | 602.3 tok/s |

The initial 9902-token full prefill was 1465.2 tok/s. Across all counted requests,
prefill was 872.72 tok/s and decode 47.94 tok/s. Overall rates depend on the
context and output mix. MTP acceptance was 0.6801; the fixture's Git state was
unchanged.

The <32K and 32–64K decode rows hold only 472 and 461 tokens. Run 2 stopped at
83035 tokens after 17 requests, with acceptance 0.6200, 1472.4 tok/s initial
prefill, and overall 957.37 tok/s prefill and 47.37 tok/s decode.

`summary-96k.json` counts the first prefill as a later turn because the server did
not flag it as reprocessing, so its <32K prefill differs. Use the figures above,
and exclude the first request explicitly when comparing new runs.

## Thermals and memory

GPU temperature at the benchmark start, after warm-up, was 51°C. Busy-sample
median/max temperature was 72/78°C and median/max power was 140.9/175.45 W.
Whole-GPU sampled memory peaked at 20979 MiB. Software power-cap flags were active
in 99.1% of busy samples, and no other throttle reason was observed. Fan RPM was
unavailable.

The log records 57 checkpoint creations and 6 restores. Most checkpoints were
149.646 MiB, and up to two were retained in the active slot. Unlike the b12001
figures, there is no separate restore-timing analysis for this run.

## Limits

- One run has no measured variance, and generated histories differ between runs.
  Run 2 gives one rough indication of spread, not a bound.
- The thread finished with a much larger share of long-context decode than the
  b12001 baseline: 16001 of 16934 decoded tokens were in 64–96K.
- This is not a task-completion or answer-quality result.
- Only one long conversation was active. It does not measure two long
  conversations or a previously occupied long second slot.

## Evidence

Ignored run evidence: `logs/swift15/agentic-20261104T221308Z-no-pad-b12110-1/`.
`summary-96k.json`, `requests-96k.json`, `server-excerpt.log` and `gpu.csv` contain
the measurements. Run 2 is in
`logs/swift15/agentic-20261104T222347Z-no-pad-b12110-2/`.
