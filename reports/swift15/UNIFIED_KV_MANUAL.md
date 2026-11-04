# Unified 128K pool: one T3 thread and small helper requests

Validated 2026-11-04 with the pinned Swift Windows b12001 CUDA 13.4 runtime.

T3 run: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:edbdda8c621c34ee2a07d137a52e3b53de0a9d6b2275a71af9f055000c8905de>

## Configuration and workload

Two slots shared a single 131072-token unified KV pool. Both main and MTP
contexts logged kv_unified=true and n_ctx_seq=131072. Main KV remained Q8_0;
MTP KV remained F16. Runtime pins, MTP depth, context, batching, vision and RAM
cache settings were unchanged. Swift's saved profile now enables parallel=2
and kv_unified=true. Other profiles retain their previous configuration.

One real T3 thread used PI, xhigh reasoning and the existing repository
investigation prompt. The usual deny-first permission policy was installed in
the trusted benchmark fixture. Four short streaming chat completions ran while
idle, then four during the investigation, and one after the main context exceeded
64K. Requests used automatic slot assignment, temperature zero, thinking off,
a 128-token output cap and an exact-text check. Each returned 32 output tokens.
These are deliberately small helper requests, not another long investigation.

The T3 thread completed normally in about 425 seconds; the context cap did not
fire. Its 54 completed inference requests included 83,160 evaluated prompt tokens
and 15,225 decoded tokens (including reasoning and tool-call output). The largest
prompt was 91,622 tokens. Main-request time-weighted prefill was 788.08 tok/s and
decode was 49.57 tok/s. These rates include contention from helper requests;
there is no matched main-thread run without helper traffic.

## Helper responsiveness

| Conditions | Requests | Median total time | Median first-content latency | Total time range |
| --- | ---: | ---: | ---: | ---: |
| Idle model | 4 | 0.67 s | 0.30 s | 0.62–0.85 s |
| During main investigation | 4 | 2.66 s | 1.57 s | 1.75–3.60 s |
| Main context above 64K | 1 | 4.46 s | 3.63 s | single sample |

All nine exact-text checks passed. All five concurrent requests had the main
slot processing before submission; the late request observed 66,118 main tokens.
Sampling observed both slots processing concurrently. The main conversation
continued on its slot and completed after the helpers. This demonstrates a
conversation above 64K beside a small helper in the shared 128K pool; it does not
establish a full 128K main conversation plus independently reserved helper space.

Cache states differed: idle requests reported no cached prompt tokens, while
concurrent requests reported 56 cached tokens out of 60. Requests arrived at
different main-context lengths and phases. The latency differences describe this
run, not a controlled estimate of the cost of unified KV. First-content latency
includes HTTP/server scheduling and prefill, not solely model prefill.

## Stability and limits

No out-of-memory or failed-message strings appeared in the captured server log.
The bracket summary recorded 12 checkpoint restores and one initial full prompt
reprocessing. The benchmark fixture's Git state was unchanged. Sampled whole-GPU
peak was 20,985 MiB (20.49 GiB). Busy-sample median temperature was 70 C, maximum
74 C; median power was 141.9 W. Non-idle throttle-reason flags were active in
99% of busy samples, so comparisons with past runs require care.

The pool's total capacity remains 128K across active conversations. A near-full
main conversation leaves little space for a helper. No per-slot reservation or
helper-specific context enforcement was added. Long helper prefills may cause
more interference than the tiny prompts exercised here.

## Evidence and validation

Ignored evidence directory: logs/swift15/agentic-20261104T193846Z-t3-unified-manual/

- thread.json: T3 launch/provider/model/options and link
- server-excerpt.log, gpu.csv, slots.jsonl: runtime and concurrency evidence
- manual_requests.py and manual-baseline/during/late.json: reproducible request
  harness, exact answers, usage, slot observations and client timings
- behavior-summary.json, analyze_manual.py: separated main/helper analysis
- concurrent-summary.json: phase-union summary including all 63 completed requests
- summary.json: original benchmark bracket, fixture check and GPU telemetry
- timing.html: helper-latency chart

The unified two-slot text/vision smoke passed before the T3 test. All 78 checks
in task agent:verify passed after the launcher/profile changes. The service was
restarted from the saved profile without temporary overrides and its two 128K
slot limits, model alias, text response and health were checked.
