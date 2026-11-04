# Two concurrent T3 threads, 2026-11-04

This is a bounded throughput run through T3's thread-launch flow and PI, using
the read-only repository investigation in [AGENTIC_PROMPT.md](AGENTIC_PROMPT.md).
Both threads were launched together (launch requests 77 ms apart), with the
existing Swift alias and xhigh reasoning. This is a different workload from the
three-file PI CLI concurrency smoke.

- Thread 1: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:81908ac82259d510193c0d61e939c01b6fe6259bd5a0a8e43d6f342023c0294c>
- Thread 2: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:88b5f44e540ff00f262694a5e519562427f8b328104de05dfbae640427a62adb>

## Setup and permissions

Pinned Windows b12001 CUDA 13.4 runtime, Swift GSQ-RCO IQ3_S MTP, main-model
Q8_0 KV and the existing MTP cache settings. Temporary SWIFT_PARALLEL=2 and
SWIFT_CONTEXT=131072 gave two 65,536-token slots. The Swift entry in PI's
models.json temporarily advertised 65,536 context tokens. Runtime pins, saved
profiles, batch sizes, MTP depth and other measured defaults were unchanged.

agentic_bench.py start --fixture installed the existing deny-first PI permission
policy in the trusted /projects/agentic-bench-t3code fixture. It denies writes,
edits, MCP and paths outside the fixture, allows selected read-only shell
commands and denies unanswered permission prompts after one second. The
fixture's Git state was unchanged after the run.

The intended cap was 49,152 prompt tokens. One request jumped from 36,792 to
54,366 before both threads were interrupted. The capped test did not complete
the requested investigation and is not a quality/completion result. A later
4,793-token prompt is consistent with PI compacting the larger conversation.
Two unfinished requests were excluded from timing counts.

## Measured throughput

13 completed requests contained 79,333 evaluated prompt tokens and 2,702 decoded
tokens. Their first-start to last-finish window was 103.94094 seconds.

| Measure | Prefill | Decode |
| --- | ---: | ---: |
| Combined phase-active throughput | 1,072.57 tok/s | 31.00 tok/s |
| Union of phase intervals across slots | 73.965 s | 87.170 s |
| Throughput over the whole completed-request window | 763.25 tok/s | 26.00 tok/s |

Combined phase-active throughput divides tokens from both slots by the union
of that phase's elapsed intervals. Overlap is counted once. Phase intervals are
reconstructed backwards from the final timing log timestamp using the server's
prefill and decode timers, so boundaries are approximate at log/timer precision.
These durations include scheduler delays and contention. Prefill and decode
overlapped for 59.125 seconds. The two phase rates must not be added together;
their denominators differ.

The completed-request window excludes thread setup before the first inference
and the tail of the two unfinished requests. Consequently these whole-window
rates are not whole-T3-run throughput. Canceled-request work can still contend
with included requests. Input retrieved from cache is not counted as new prefill.

Sampling observed both slots processing concurrently. Sampled whole-GPU peak:
20,843 MiB (20.35 GiB). Busy-sample median power: 163.7 W, maximum 183.85 W;
median GPU temperature 69 C, maximum 76 C. A non-idle GPU throttle-reason bit was
active in 98% of busy samples. These laptop operating conditions limit comparison
with earlier reports. No matched one-slot T3 baseline was run.

## Evidence and reproduction

Ignored evidence directory:
logs/swift15/agentic-20261104T192418Z-t3-two-slots/

- threads.json: T3 launch metadata and links
- server-excerpt.log: immutable server-log excerpt
- concurrent-requests.json: completed requests and reconstructed phase intervals
- concurrent-summary.json: combined timing calculation
- summary.json: original bracket summary and GPU telemetry (its summed-time rates
  are per-sequence aggregates, not combined concurrent throughput)
- gpu.csv, slots.jsonl, context-cap.json: sampling and cap evidence
- timing.html: phase timeline

Recalculate the concurrent metrics with:

```bash
python3 experiments/agentic_timing.py logs/swift15/agentic-20261104T192418Z-t3-two-slots
```

Both T3 runs were confirmed interrupted before server teardown. The original PI
configuration was restored byte for byte, and llama-swap was restarted without
benchmark overrides. Swift's original single 131,072-token slot, health and text
response were checked after restoration. task agent:verify passed all 77 tests.
