# Swift: single versus two parallel T3 threads, 2026-11-07

The single-thread test reached the 96K prompt cap without a later full prefill.
The parallel test exhausted the shared KV pool before reaching the cap: one idle
slot was purged, its next request began a full prefill, and both active requests
ultimately failed with `Context size has been exceeded.`

**Acceptance decision:** both client threads failing is unacceptable. This
two-slot unified configuration is rejected for concurrent long agent sessions.
Checkpoint and RAM-cache headroom did not make that failure behavior acceptable.
The requested replacement is one slot, unified KV disabled, twenty checkpoints
and the existing 16 GiB RAM cache, to be tested with one and two T3 clients.

## Setup

Both tests used a fresh Windows b12110 CUDA 13.4 server, Swift 1.5 IQ3_S MTP,
two slots sharing a unified 131072-token KV pool, ten context checkpoints per
slot, default 8192-token minimum checkpoint spacing, 16384 MiB RAM prompt-cache
limit, q8_0 K/V, batch/ubatch 2048/512, MTP depth 3 and the vision projector.
Saved profile settings were unchanged during these tests.

Workload: the existing trusted read-only fixture at commit
32ecd71726fee16f335484c1f95cbce126d461f6, the unmodified investigation prompt in
AGENTIC_PROMPT.md, PI and xhigh thinking. Both parallel threads shared that
read-only fixture. The deny-first PI policy was installed before each test.
Fixture Git state remained unchanged in both tests.

The GPU cooled below 50 C before each fresh launch (38 C before single, 49 C
before parallel). After model load/warm-up, bracket temperatures were 41 C and
54 C. This is not a matched thermal or fixed-work throughput comparison. Laptop
power mode/cooling hardware were not independently reconfirmed. Generation had
normal sampling without a fixed seed, so histories and output lengths differ.

The single run used a 98304 prompt-token cap. The pair was to stop when either
conversation first crossed that cap; capacity failures ended it sooner. The
single run was interrupted after crossing the cap. Neither investigation is a
completed-task/answer-quality result.

## Requests and cache behavior

| Measure | Single thread | Two parallel threads |
| --- | ---: | ---: |
| Requests started before cap/failure | 40 | 32 |
| Requests completed before cap/failure | 40 | 30 |
| Initial full prefills (excluded below) | 1 | 2 |
| Later requests started | 39 | 30 |
| Later full-prefill attempts | 0 | 1 |
| Later full prefills completed | 0 | 0 |
| Checkpoint restores | 6 | 14 |
| Maximum retained checkpoints in an active slot | 10 | 10 |
| Idle-slot purges for KV capacity | 0 | 1 |
| KV-space batch retries | 0 | 63 |
| Context-capacity errors | 0 | 2 |

The single thread's largest included prompt was 93616 tokens. Its next prompt
was 99076 tokens, crossing the cap. A subsequent 99371-token prompt started
before interruption; over-cap tasks are excluded from the counted requests.
Memory telemetry includes that short over-cap tail.

In the pair, thread 1 used slot 1: 12 requests started, 11 completed, final prompt
57592 tokens. Thread 2 used slot 0: 20 requests started, 19 completed, final prompt
77071 tokens. Their largest completed prompts were 49557 and 75644 respectively.
Their final requested histories totaled 134663 tokens, beyond the 131072-token
shared pool; generated tokens also consume the pool. Slots are not independent
128K reservations. Slot samples observed both processing in 143 of 165 samples.

The capacity failure sequence is visible in the server log:

1. Task 1110 on slot 1 restored a checkpoint and began extending its conversation.
2. The server could not find free KV space, retried smaller batches, then purged
   idle slot 0 containing 75970 tokens.
3. Slot 0's next task, 1119, started from zero cached tokens for a 77071-token
   prompt. This full-prefill attempt was not accompanied by the usual
   `forcing full prompt re-processing` message.
4. With both slots active, retrying down to batch size 1 could not resolve the
   capacity shortage. Tasks 1119 and 1110 both received context-size errors.

The full-prefill attempt did not finish. Reporting one completed full prefill,
or reporting zero merely because the usual message was absent, would both be
incorrect. The original bracket summary counts only completed requests and is
therefore supplemented by memory-cache-analysis.json, which includes unfinished
requests, first cached-token counts, purge events and server errors.

## Memory

GPU sampling used one-second intervals; Windows/host memory used approximately
3.2-second intervals. Slot sampling used two-second intervals. The Windows peak
working-set counter also captures brief process peaks between periodic samples.
The processes were fresh, so their lifetime peaks belong to these tests, including
model loading and warm-up. Whole-GPU and system-memory figures include other
applications and must not be treated as model-only allocations.

| Peak measure | Single thread | Two parallel threads |
| --- | ---: | ---: |
| Server working set, sampled | 3.67 GiB (3758 MiB) | 8.56 GiB (8765 MiB) |
| Server working set, Windows peak counter | 3.68 GiB (3768 MiB) | 8.56 GiB (8765 MiB) |
| Server private committed bytes | 24.59 GiB (25180 MiB) | 29.50 GiB (30212 MiB) |
| Whole-GPU dedicated memory, sampled | 20.34 GiB (20830 MiB) | 20.34 GiB (20832 MiB) |
| RAM prompt-cache occupancy, log maximum | 0.44 GiB (450.887 MiB) | 4.14 GiB (4236.18 MiB) |
| Whole-system Windows commit | 45.99 GiB (47093 MiB) | 49.01 GiB (50191 MiB) |

Private committed bytes are not resident physical RAM and must not be added to
working set or VRAM. Prompt-cache occupancy can include associated state and
checkpoints, so it is not additive to active-checkpoint sizes. The 16 GiB setting
is a cache limit, not a reservation; log occupancy peaked at only 4.14 GiB in the
parallel run. RAM cache and checkpoints did not increase active KV capacity.

The Windows server is outside the WSL cgroup's host-memory enforcement, so the
near-zero WSL scope memory is not a measure of server RAM. GPU maximum sampled
temperature was 75 C in the single run and 79 C in the pair; busy median power was
140.9 W and 162.7 W. These differences further limit throughput comparisons.

## Evidence and restoration

- Single: ignored logs/swift15/agentic-20261107T121658Z-cp10-ram16-single/
- Parallel: ignored logs/swift15/agentic-20261107T122542Z-cp10-ram16-parallel/
- Analysis/sampling helpers: ignored logs/swift15/t3-memory-checkpoints-20261107/

Each run contains meta.json (settings and T3 links), host.csv, gpu.csv,
slots.jsonl, monitor-state.json, server-excerpt.log, summary-96k.json,
requests-96k.json and memory-cache-analysis.json. The single run also contains
context-cap.json. Session paths and terminal stop/error reasons are recorded in
the analysis JSON. Compare unfinished requests using the custom analysis, not
only the completed-request timing summaries.

T3 single (interrupted at cap):
<t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:f0e204560e042ed4785e4021cbee54811d23a737dd8c7d0210a1ce88860666d3>

T3 parallel thread 1 (failed with context-size error):
<t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:f1e96f4b9a2c848060ecc6466b6f6e725b50c03f50361dff98e881b19d421c8f>

T3 parallel thread 2 (failed with context-size error):
<t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:bd82624496307254005126c5e6412cad23d8a2aad8fbb2c08e99d5f47860b15c>

After both threads were confirmed failed, telemetry was finalized and llama-swap
was restarted cleanly. The saved 2-slot, 10-checkpoint, 16-GiB-cache profile
returned `OK` through its production alias. No implementation/profile edits were
made for these tests; only ignored analysis scripts and this report were added.
One run of each setup is evidence for these observations, not an estimate of
failure probability or performance variance.

Interactive memory timeline: ignored
`logs/swift15/t3-memory-checkpoints-20261107/memory-comparison.html`.
