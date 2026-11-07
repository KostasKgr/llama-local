# Swift: one slot, no unified KV, twenty checkpoints, 2026-11-07

## Decision and outcome

Both threads failing in the previous two-slot unified-KV test is unacceptable.
That configuration was rejected for concurrent long agent sessions; see
[T3_CP10_RAM16_MEMORY.md](T3_CP10_RAM16_MEMORY.md).

The requested replacement passed both bounded reliability checks: one T3 client,
then two concurrent T3 clients queueing through a single inference slot. Neither
repeat had a later full re-prefill, KV-space retry, idle-slot capacity purge or
context-size error. All clients were intentionally interrupted at the established
prompt cap. This is not an assertion of indefinite reliability or task quality.

## Saved configuration and validation

Swift now uses one 131072-token inference slot, unified KV disabled, twenty
context checkpoints, minimum checkpoint spacing 8192 tokens and 16384 MiB RAM
prompt cache. Runtime Windows b12110 CUDA 13.4, main q8_0 K/V, batch/ubatch
2048/512, MTP depth 3, model and vision projector remain unchanged.
The profile, current README and default-setting test expectations were updated.
`task agent:verify` passed all 89 tests; a sequential text/red/blue vision smoke
passed with the new settings on owned trial port 1260 before benchmarking.
The smoke log confirms kv_unified=false and max checkpoints=20.

Each benchmark used a fresh server through llama-swap, the existing trusted
read-only fixture at 32ecd71726fee16f335484c1f95cbce126d461f6, the unmodified
AGENTIC_PROMPT.md investigation prompt, PI and xhigh thinking. The deny-first
PI policy was installed before each run. Fixture Git state was unchanged.
The GPU cooled below 50 C before each fresh launch (39 C single, 49 C pair).
Post-warm-up start temperatures were 41 C and 51 C. Laptop power/cooling hardware
were not independently reconfirmed. These are one run each with varying model
outputs, not matched-work throughput measurements.

Both T3 clients in the second test ran concurrently, but the server processed
one request at a time. Conversation state was saved/restored through the RAM
prompt cache. There were zero slot samples with two active inference slots, as
expected. This test measures queueing and state switching, not parallel decode.

## Requests and reliability

| Measure before first over-cap prompt | Single client | Two queued clients |
| --- | ---: | ---: |
| Completed requests | 39 | 68 (34 per client) |
| Initial requests excluded from later-turn counts | 1 | 2 |
| Later requests | 38 | 66 |
| Later full re-prefills, started or completed | 0 | 0 |
| Context-size/client capacity errors | 0 | 0 |
| KV-space batch retries / capacity purges | 0 / 0 | 0 / 0 |
| Checkpoint restores | 3 | 12 |
| Checkpoint creates | 79 | 136 |
| Maximum retained checkpoints observed | 19 | 19 |
| Maximum included prompt | 93731 | 97314 |

The second client's initial prompt reused 9762 of 9766 tokens and evaluated only
four tokens. Initial requests are excluded by client identity/order, regardless
of whether their initial prefill was full or cached. The one forcing-full log
event in each run belongs to the first initial prompt, not a later turn.

The single-client cap was crossed by a 98791-token prompt. That request completed
before interruption, and a 99078-token prompt also began. They are excluded from
the capped request count, which is therefore 39 rather than the PI session's 40
completed requests. The two-client test first crossed the cap at 99407 tokens.
Before that, both clients completed 34 requests, reaching 97314 and 95988 prompt
tokens respectively (193302 combined history tokens). Neither client failed.
Their final aborted session entries are from deliberate benchmark interruption,
not capacity errors. Slot/server telemetry includes the short over-cap tail.

Twenty is the retention limit; nineteen was the largest retained count actually
logged. The spacing policy prunes checkpoints too close to earlier checkpoints.
The test changed slot count, unified KV and checkpoint retention together, so
it does not isolate their individual performance or memory effects.

## Memory

| Peak measure | Single client | Two queued clients |
| --- | ---: | ---: |
| Server resident RAM, Windows peak counter | 4.98 GiB (5101 MiB) | 17.95 GiB (18377 MiB) |
| Server resident RAM, periodic samples | 4.96 GiB (5080 MiB) | 17.88 GiB (18307 MiB) |
| Whole-GPU VRAM, sampled | 19.75 GiB (20229 MiB) | 19.76 GiB (20239 MiB) |
| RAM prompt-cache occupancy, server log | 0.44 GiB (450.887 MiB) | 6.96 GiB (7123.81 MiB) |
| Server private committed bytes | 25.30 GiB (25907 MiB) | 38.25 GiB (39169 MiB) |

The prior two-slot/ten-checkpoint pair peaked at 8.56 GiB server working set but
failed; the replacement pair peaked at 17.95 GiB and had no capacity failure
before the cap. The histories and output mix differ, so this is an observed
memory cost, not a controlled attribution to checkpoint count alone.

The 16 GiB limit applies to the separate RAM prompt cache, not total server RAM.
Minimum sampled Windows available physical RAM was 11510 MiB (11.24 GiB) for
the single client and 806 MiB (0.79 GiB) for the pair. The pair avoided client
failure but produced substantial host-memory pressure; the low available-memory
sample must not be hidden by reporting only the server's cache occupancy.
Active checkpoint/state buffers and runtime allocations also occupy memory.
Private committed bytes are not resident physical RAM and must not be added to
working set or GPU VRAM. Prompt-cache occupancy includes related state and is
not additive to active-checkpoint sizes. GPU numbers cover the whole device;
system-memory counters include other applications. The Windows server runs
outside the WSL scope's memory enforcement, so near-zero cgroup memory does not
measure its actual host RAM.

Host sampling was approximately every 3.2 seconds, GPU sampling every second,
and slot sampling every two seconds. Windows' process peak-working-set counter
captures peaks that periodic resident-memory samples can miss. Each process was
fresh, so its lifetime peak includes this test's model load and warm-up.
Post-cap/interrupt tail samples are included in memory peaks. Cooling thresholds
were consistent but post-warm-up temperatures differed, limiting performance
comparisons.

## Evidence, threads and restoration

Ignored evidence directories:

- logs/swift15/agentic-20261107T131950Z-cp20-single-slot-single/
- logs/swift15/agentic-20261107T132904Z-cp20-single-slot-two-clients/

Each contains captured settings/threads in meta.json and threads.json, host.csv,
gpu.csv, slots.jsonl, context-cap.json, server-excerpt.log, summary-96k.json,
requests-96k.json and memory-cache-analysis.json. Analysis includes initial
requests, incomplete requests, first cached-token counts, checkpoint counts,
capacity events and saved PI session paths/stop reasons. Helpers remain ignored
under logs/swift15/t3-memory-checkpoints-20261107/.

Single client, interrupted at cap:
<t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:2828c330bbf3fc20ba74fd0d71dd5db3803d21ba946ab3c561e68de93ee03a61>

Concurrent client 1, interrupted at cap:
<t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:7fc1d66470ffd13ad1cdeb9c322063e764b783e416bcddde9ac0d010d55ff67d>

Concurrent client 2, interrupted at cap:
<t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:1826a3ffa9db4dfcbaaf76d96df365756734be24b35b07739c072723ff9f8ceb>

All clients were confirmed interrupted before cleanup. Telemetry was finalized
and llama-swap restarted cleanly with the requested saved profile to release
benchmark conversation states. Production health/chat were checked afterward.
No weights, runtimes, downloads or raw logs were added to Git.

Interactive resident/available-memory timeline: ignored
`logs/swift15/t3-memory-checkpoints-20261107/cp20-memory-comparison.html`.


## Throughput for the capped completed-request sets

Rates below use the same 39 and 68 completed requests counted above. Prefill
counts only evaluated tokens, excluding reused cached tokens, and includes each
initial prompt's evaluated tokens. Phase rates divide token totals by phase
active time. With one inference slot there is no cross-slot phase overlap.

| Measure | Single client | Two queued clients, aggregate |
| --- | ---: | ---: |
| Evaluated prompt tokens | 87321 | 169995 |
| Generated tokens | 11815 | 18609 |
| Prefill active time | 106.02 s | 214.97 s |
| Decode active time | 249.30 s | 392.04 s |
| Prefill phase throughput | 823.61 tok/s | 790.78 tok/s |
| Decode phase throughput | 47.39 tok/s | 47.47 tok/s |
| First-start to last-finish window | 363.91 s | 764.30 s |
| Output tokens per whole-window second | 32.47 tok/s | 24.35 tok/s |
| Time outside prefill/decode phases | 8.59 s | 157.29 s |
| MTP draft acceptance | 0.6452 | 0.6391 |

The whole-window output rate includes prefill and inter-request gaps, including
cache switching, tools and client scheduling. The gap duration is not an
isolated measurement of RAM-cache switching cost. It excludes model load,
initial T3 setup before the first request and the final over-cap/interrupt tail;
therefore it is not whole-T3-run end-to-end throughput. Two-client figures are
combined server throughput, not a per-client rate or two-way parallel decode.

Decode phase rates by full prompt context:

| Prompt context | Single client | Two queued clients, aggregate |
| --- | ---: | ---: |
| <32K | 83.18 tok/s | 81.13 tok/s |
| 32–64K | 64.48 tok/s | 67.20 tok/s |
| 64–96K | 46.24 tok/s | 46.07 tok/s |

Decode active-phase throughput was effectively unchanged in these observations.
The aggregate whole-window output rate was lower for the queued pair. Generated
histories and context mixes differ and there was only one run per setup, so these
numbers do not isolate the effect of concurrency or establish a variance bound.
Evidence: each run's summary-96k.json and memory-cache-analysis.json combined_timing.


## Comparison with the saved b12110 baseline

Reference: BASELINE.md run 1, 2026-11-04. Both used one active T3 client,
Windows b12110, Swift IQ3_S MTP, 128K context, q8_0 K/V and batch/ubatch
2048/512. The baseline configured two slots with unified KV, two checkpoints
and 4 GiB RAM cache; the repeat configured one slot, no unified KV, twenty
checkpoints and 16 GiB RAM cache.

| Measure | Saved baseline | Latest single client | Change |
| --- | ---: | ---: | ---: |
| Decode active-phase tok/s | 47.94 | 47.39 | -1.1% |
| Prefill active-phase tok/s | 872.72 | 823.61 | -5.6% |
| Whole-request-window output tok/s | 37.19 | 32.47 | -12.7% |
| Sampled whole-GPU memory peak | 20979 MiB | 20229 MiB | -750 MiB |
| Later full re-prefills | 0 | 0 | unchanged |
| Checkpoint restores | 6 | 3 | workload dependent |
| MTP acceptance | 0.6801 | 0.6452 | -3.49 percentage points |

| Prompt context | Baseline decode tok/s | Current decode tok/s | Baseline later prefill tok/s | Current later prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| <32K | 81.30 | 83.18 | 1312.20 | 1267.62 |
| 32–64K | 68.60 | 64.48 | 879.00 | 842.73 |
| 64–96K | 47.00 | 46.24 | 594.60 | 573.49 |

Initial full prefill was approximately unchanged: baseline 1465.2 tok/s,
current 1463.96 tok/s. Later-prefill context rows exclude the first request
explicitly in both runs. Baseline host RAM was not sampled, so the current
4.98 GiB peak server working set has no equivalent baseline measurement.

These are observations, not a controlled performance regression result. Initial
prompts differed (9902 versus 9766 tokens), generated histories differed, MTP
acceptance differed and each setup has only one comparison run. The baseline
finished naturally with 28 requests and 16934 output tokens; the capped current
set has 39 requests and 11815 output tokens. Current evaluated-prefill tokens
per generated token were 7.39 versus baseline 4.94, so whole-window output
throughput is particularly sensitive to this changed prefill/output mix.
The baseline's low/mid-context decode buckets had only 472/461 output tokens.
Baseline run 2's decode rates were 81.4 / 61.7 / 46.2 tok/s, illustrating the
variation already present in the older configuration.

Baseline request-window rates were recomputed with agentic_timing from its
immutable server-excerpt.log: 455.302345 s, 28 completed requests, 83678 evaluated
prompt tokens and 16934 output tokens. Current rates use the same method and
completed-request boundary convention. Neither rate includes T3 setup or the
final interrupt tail.
