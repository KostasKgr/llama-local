# Gyro-S baseline

Current baseline, measured 2026-11-12. It supersedes the
[GGUF-in-place baseline](GGUF_IN_PLACE_BASELINE.md), whose machine, runtime
pins, profile defaults, fixture and agentic method are unchanged here. The only
change is the pack: `install gyros` now writes `experts.bin`
(`iq_pack.py --experts-bin`), so the engine reads experts from one file instead
of the GGUF in place. See [docs/strata.md](../../docs/strata.md),
"Loading and expert placement", for the mechanism and the API load comparison.

## Load

| | GGUF in place (previous) | experts.bin |
| --- | ---: | ---: |
| Engine load (owner readiness) | 69.8–80.6 s | 22.1–24.2 s |
| GPU-cache fill | 44–56 s, about 0.3 GB/s | 4.0–4.1 s, about 3.8 GB/s |
| Cold first request through llama-swap | about 82 s | 30.7–31.6 s |

## Real T3 agentic run

One P2 attempt, same fixture commit, prompt (apart from the nonce path),
permission policy, pi model `llama-local/qwen3.8-flash-next-gyro-s-strata`,
thinking `high`, full-access runtime and fresh engine as the previous baseline's attempts 2–5.
Rates use the baseline's method; recomputing attempts 2–5 with it reproduces
the previous baseline table exactly.

| Thread | Result | Wall s | Max prompt | API requests / tool calls | Prefill / decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| [Gyro-S agentic experts.bin 1](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:e9071d5eecddca876412942cddae782a747eef646deeb5bfde784a3ac3cae0ba) | mechanical pass; content not manually reviewed | 140.5 | 85,756 | 25 / 33 | 2,101.1 / 141.9 |

| Measure | GGUF in place: median (range), attempts 2–5 | experts.bin | Change vs median |
| --- | ---: | ---: | ---: |
| Initial full prefill, tok/s | 638.1 (626.3–649.2) | 2,148.2 | +237% |
| Later prefill <32K, tok/s | 1,020.0 (952.0–1,178.7) | 2,339.7 | +129% |
| Later prefill 32–64K, tok/s | 1,278.1 (1,200.0–1,432.2) | 2,319.7 | +81% |
| Later prefill 64–96K, tok/s | 979.7 (943.3–1,092.1) | 1,625.5 | +66% |
| Decode <32K, tok/s | 144.0 (139.4–147.4) | 160.0 | +11% |
| Decode 32–64K, tok/s | 146.9 (136.8–150.9) | 167.5 | +14% |
| Decode 64–96K, tok/s | 115.7 (114.3–120.2) | 137.1 | +18% |
| Overall prefill, tok/s | 1,025.3 (979.9–1,151.1) | 2,101.1 | +105% |
| Overall decode, tok/s | 123.2 (120.8–125.6) | 141.9 | +15% |

Bucket request counts (later prefill / decode): 5 / 6, 7 / 7 and 12 / 12. Every
experts.bin value lies outside the previous range. This is a single attempt:
generated history differs between attempts, so per-bucket values carry
attempt-to-attempt variation not measured here.

Mechanical audit: exact model, full successful reads of all five files, nonce
read and in a 683-word final answer, no write/edit/MCP calls, fixture unchanged.
One tool error was followed by recovery; 33 calls exceeded the soft 32-call budget
by one, as in three previous attempts. The answer's citations and risks were not
reviewed against the source, so content quality is not compared.

During the run, median CPU package power was 31.6 W against 44.7–55.7 W in
the previous attempts, and median busy GPU power 167.5 W against 141.7–142.8 W: less
CPU-side expert work and a busier GPU. Minimum available Windows RAM was
4.18 GiB (previously 3.15–3.95 GiB). Raw evidence:
`logs/gyros/20261112T192316.621691Z-t3-xbin-01/` (Git-ignored).

The synthetic API benchmark (`experiments/strata_benchmark.py run`) was not
repeated with `experts.bin`; its previous results and chart are in the
superseded report.
