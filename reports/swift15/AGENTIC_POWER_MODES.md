# Swift 1.5 agentic benchmark — laptop power modes and runtime, 2026-10-30

Real agentic runs of Swift 1.5 IQ3_S MTP on the RTX 5090 Laptop GPU, comparing
Omen Gaming Hub power modes and cooling, then llama.cpp b11393 against b12001
with CUDA 12.8 and 13.4. Swift is now pinned to b12001 CUDA 13.4.

## Setup

- Runtime b11393 CUDA 12.8 for the power-mode runs; b12001 CUDA 12.8 and 13.4
  for the runtime runs (same profile and alias, only `runtime` changed), 131072 context, q8_0/q8_0 KV,
  MTP draft max 3, batch 2048/512, served through llama-swap.
- Client: PI 1.4.0 launched as a T3 Code thread, model
  `llama-local/swift-1.5-qwen3.8-27b-iq3s-mtp`, thinking `xhigh`.
- Fixture: fresh clone of upstream `pingdotgg/t3code` at
  `32ecd71726fee16f335484c1f95cbce126d461f6` in `/projects/agentic-bench-t3code`.
- Task (read-only): trace server-side thread launch to the Claude driver through
  `ThreadLaunchService.ts`, `ProviderTurnStartService.ts`,
  `ProviderSessionManager.ts`, `ProviderRegistry.ts` and `Drivers/ClaudeDriver.ts`
  (about 60K tokens if read in full), then write a call chain and three risks.
  Runs differ in how much they read, so per-token speeds are compared rather than
  wall time. The fixture was unchanged after every run.
- Context cap: speeds count only requests up to the first one above 96K
  (98,304 tokens), via `agentic_bench.py finish --max-context 98304`. From
  Performance 2 on, the run is also interrupted at the cap, so every run covers
  the same context range, stays clear of PI compaction near 128K, and gets a
  comparable heat load. Earlier runs were cut at 96K in analysis only; their GPU
  rows cover the whole run.
- Measured with `experiments/agentic_bench.py` (server-log timings plus 1 s
  `nvidia-smi` samples; "busy" means GPU utilization ≥ 50%). Throttle rows are
  the driver's clock-event reasons; fan speed cannot be read without Windows
  admin rights, so it is the user's reading. CPU temperature is not recorded.
- Hardware: OMEN MAX 16 (16t-ah000). Runs to Performance + pad 1 had the lid
  closed; the last run had the lid open with the built-in screen turned off in
  Windows. The cooling pad blows upward into the bottom intake. Each run starts with
  the GPU at or below 50 °C. Evidence: ignored `logs/swift15/agentic-*/`.
- Balanced 1–2 and Performance 1 paused on PI permission prompts (`sed`, and a
  read outside the fixture); this affects wall time, and lets the GPU cool during
  the pause, but not server-side speeds. Performance 1 stalled about 5 minutes.
  From Performance 2 on, a deny-first read-only project policy
  (`experiments/agentic_bench_pi_policy.json`, installed by `agentic_bench.py
  start`) denies instead of prompting. Because it denies MCP, PI no longer
  sends the T3 MCP tool definitions, so the first request shrinks from about
  28.4K to 9.8K tokens; at the same context, Performance 2 holds more code.

## Overview (to 96K context)

| | Balanced 1 | Balanced 2 | Perf 1 | Perf 2 | Perf + pad 1 | Perf + pad + lid open |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Requests (to 96K) | 28 | 16 | 15 | 39 | 27 | 26 |
| Peak context, whole run | 96.6K | 115.5K | 122.1K | 98.9K | 92.9K | 95.6K |
| Decode tok/s, <32K context | 51.4 | 49.8 | 63.8 | 67.7 | 73.1 | 67.1 |
| Decode tok/s, 32–64K | 42.8 | 46.8 | 48.8 | 51.9 | 56.2 | 54.4 |
| Decode tok/s, 64–96K | 33.7 | 36.0 | 42.4 | 43.5 | 39.0 | 40.2 |
| Decode tok/s, overall | 34.5 | 37.6 | 45.5 | 45.8 | 40.2 | 42.4 |
| Prefill tok/s, overall | 534 | 572 | 727 | 645 | 692 | 699 |
| Decode / prefill time (s) | 355 / 172 | 78 / 167 | 83 / 128 | 198 / 140 | 331 / 124 | 277 / 129 |
| MTP draft acceptance | 0.65 | 0.73 | 0.68 | 0.71 | 0.61 | 0.64 |
| Checkpoint restores / full reprocesses | 4 / 1 | 3 / 1 | 3 / 1 | 8 / 1 | 2 / 1 | 6 / 1 |
| Busy GPU samples (1 s), whole run | 466 | 612 | 939 | 297 | 413 | 355 |
| SW power cap, share of busy | 100% | 100% | 99% | 97% | 99% | 99% |
| SW/HW thermal, HW slowdown, HW power brake | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% |
| GPU power median / max (W) | 113 / 131 | 112 / 142 | 144 / 192 | 146 / 190 | 146 / 185 | 146 / 188 |
| Memory clock at 14 GHz, share of busy | 49% | 42% | 91% | 85% | 89% | 83% |
| GPU temperature median / max (°C) | 64 / 67 | 66 / 67 | 67 / 77 | 68 / 73 | 68 / 72 | 68 / 72 |
| Start temperature (°C) | 44 | 49 | 47 | 50 | 45 | 45 |
| Fan RPM (user-reported peak) | ~3000–3400 | ~3400 | 4300–4700 | – | 4700 (~1.3 min in) | 3700 → 4700 (~2.3 min in) |

Whole-run figures before the cap (Balanced 2 to 115.5K, Performance 1 to 122.1K):
decode 32.6 and 41.3 tok/s overall, 31.7 and 36.0 tok/s at ≥96K. Performance 1
also had two extra full prefills after PI compacted the context near 128K.

## Prefill (to 96K context)

The first request has no cache to reuse, so it processes the whole start
(system prompt, tools and task). Later turns process only the new tool output
appended to the context, grouped by the context size at that request.

| | Balanced 1 | Balanced 2 | Perf 1 | Perf 2 | Perf + pad 1 | Perf + pad + lid open |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First request (full prefill) | 28,431 tok, 34.9 s, 816 tok/s | 28,427 tok, 32.3 s, 880 tok/s | 28,447 tok, 25.0 s, 1,140 tok/s | 9,841 tok, 8.4 s, 1,174 tok/s | 9,841 tok, 8.2 s, 1,195 tok/s | 9,841 tok, 8.2 s, 1,200 tok/s |
| Later turns, <32K | 400 tok/s | – | 450 tok/s | 995 tok/s | 1115 tok/s | 1034 tok/s |
| Later turns, 32–64K | 602 tok/s | 623 tok/s | 813 tok/s | 739 tok/s | 803 tok/s | 781 tok/s |
| Later turns, 64–96K | 374 tok/s | 402 tok/s | 552 tok/s | 400 tok/s | 403 tok/s | 484 tok/s |

Small appended chunks at long context run at a fraction of bulk prefill speed;
per-request overhead and attention over the long context dominate.

## Observations

- Performance vs Balanced (means of two runs each, to 96K): decode +27% overall
  (65.8 vs 50.6 tok/s under 32K, +30%; 50.4 vs 44.8 at 32–64K, +12%; 43.0 vs
  34.9 at 64–96K, +23%). The identical 28.4K-token first request ran 35% faster
  (1,140 vs 848 tok/s). Power rises from ~112 W to ~145 W median (~190 W peak),
  the memory clock stays at 14 GHz 85–91% of the time instead of under half, and
  the GPU peaks at 73–77 °C instead of 67 °C. Fans rise from ~3000–3400 to
  4300–4700 RPM.
- Cooling pad (Performance + pad 1, lid closed): no measurable change. Power
  (146 W median), the power-cap flag (99%) and memory clock match the no-pad
  Performance runs; GPU temperature peaks at 72 °C (vs 73–77 °C), within
  run-to-run variation, minute by minute within 1–2 °C; fan reached 4700 RPM
  early and stayed there. Decode by context range differs both ways (73.1 /
  56.2 / 39.0 vs 65.8 / 50.4 / 43.0 tok/s), which is run-to-run variation.
- Lid open (Performance + pad + lid open): also no measurable change. Power
  146 W median, power cap 99%, GPU temperature 68 / 72 °C, the same per-minute
  profile (71–72 °C early, 68 °C later); decode 67.1 / 54.4 / 40.2 tok/s by
  context range. The fan ramped more slowly (3700 RPM after ~1 min, 4700 after
  ~2.3 min, vs ~1.3 min with the lid closed) but reached the same 4700 RPM.
- No run shows thermal or hardware throttling (0% of busy samples); the only
  reason reported is the software power cap. The GPU is power-limited by the
  selected mode, not heat-limited. A laptop-controller power reduction caused by
  heat would also appear as a power cap, so a small thermal effect is not ruled out.
- Balanced holds the GPU at about 112 W of its 175 W limit, with the power-cap
  throttle reason active almost continuously, and drops the memory clock to
  9–11 GHz for over half the busy time. Decode is memory-bandwidth bound, so this
  directly limits speed. Thermals are not the limit (max 67 °C).
- Decode falls by about 35% from short context to 64–96K in both modes.
- `clocks.sm` reads about 180 MHz under load in WSL and is not trustworthy;
  power and memory clock are used instead.

## Runtime comparison (Performance, pad, lid open)

All in Omen Performance mode with the cooling pad (the first column had the lid
closed, which changed nothing measurable), to 96K context. b12001 is
`eb4e244be`; the CUDA 13.4 build bundles CUDA runtime 13.4 (13040) on driver
592.27, already validated for Bolt. Each b12001 runtime got a fresh server; its
first run's first request included model loading, so the identical 9,841-token
first request is compared on the second runs.

| | b11393 pad 1 | b11393 pad + lid open | b12001 CUDA 12.8 #1 | b12001 CUDA 12.8 #2 | b12001 CUDA 13.4 #1 | b12001 CUDA 13.4 #2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Requests (to 96K) | 27 | 26 | 42 | 44 | 38 | 31 |
| Decode tok/s, <32K context | 73.1 | 67.1 | 71.2 | 65.5 | 74.5 | 78.3 |
| Decode tok/s, 32–64K | 56.2 | 54.4 | 64.8 | 58.1 | 62.1 | 66.5 |
| Decode tok/s, 64–96K | 39.0 | 40.2 | 47.4 | 41.0 | 46.0 | 42.9 |
| Decode tok/s, overall | 40.2 | 42.4 | 49.0 | 47.7 | 47.2 | 45.5 |
| First request (9,841 tok) prefill tok/s | 1,195 | 1,200 | with model load | 1,353 | with model load | 1,351 |
| Later-turn prefill tok/s, <32K | 1115 | 1034 | 1181 | 1128 | 1280 | 1177 |
| Later-turn prefill tok/s, 32–64K | 803 | 781 | 823 | 678 | 930 | 782 |
| Later-turn prefill tok/s, 64–96K | 403 | 484 | 521 | 559 | 538 | 555 |
| MTP draft acceptance | 0.61 | 0.64 | 0.70 | 0.64 | 0.67 | 0.63 |
| Busy GPU samples (1 s) | 413 | 355 | 240 | 337 | 315 | 408 |
| SW power cap, share of busy | 99% | 99% | 98% | 99% | 99% | 100% |
| SW/HW thermal, HW slowdown, HW power brake | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% | 0% / 0% / 0% / 0% |
| GPU power median / max (W) | 146 / 185 | 146 / 188 | 149 / 192 | 149 / 190 | 143 / 190 | 142 / 185 |
| Memory clock at 14 GHz, share of busy | 89% | 83% | 95% | 98% | 97% | 97% |
| GPU temperature median / max (°C) | 68 / 72 | 68 / 72 | 67 / 71 | 69 / 75 | 67 / 72 | 67 / 72 |
| Start temperature (°C) | 45 | 45 | 45 | 49 | 50 | 50 |

- b12001 CUDA 13.4 is fastest: mean decode 76.4 / 64.3 / 44.5 tok/s at <32K /
  32–64K / 64–96K, against 70.1 / 55.3 / 39.6 on b11393 (+9% / +16% / +12%) and
  68.4 / 61.5 / 44.2 on b12001 CUDA 12.8, at about 142 W median instead of 146–149 W.
- Prefill: the identical first request runs at about 1,350 tok/s on both b12001
  builds against about 1,200 on b11393 (+13%); later-turn prefill is similar or
  higher on CUDA 13.4.
- Draft acceptance varies between 0.61 and 0.70 by run, independent of runtime;
  it moves decode speed, so single runs differ by up to ~10%.
- All runtimes remain power-capped (98–100% of busy samples) with no thermal or
  hardware throttling; GPU peaks at 71–75 °C.
- Validation: `smoke swift15` passed on both b12001 builds; `pi_smoke.py swift15
  --port 1236` passed on CUDA 13.4. Every run left the fixture unchanged.
