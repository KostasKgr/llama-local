# Swift 1.5: llama.cpp b12110 against the b12001 baseline, 2026-11-04

The current baseline drawn from these runs is [BASELINE.md](BASELINE.md).

Swift moved from Windows b12001 (`eb4e244be`) to Windows b12110 CUDA 13.4
(`cd959bae9`, the v0.9.0 commit). The saved two-slot unified profile, Q8_0 KV, MTP
depth 3, batching, checkpoints and RAM cache are unchanged. All measurements use
one active request or thread. b12001 was not rerun: the agentic comparison is
against the single run in [B12001_BASELINE.md](B12001_BASELINE.md).

## Result

No failures, and speed is about the same or slightly better. Later-turn prefill
rose 3–6%; decode moved by -2% to +4%. Differences of this size are within the
runbook's run-to-run variation of up to about 10%, so this shows no meaningful
regression, not a proven speedup.

## Agentic runs

Same prompt, fixture, xhigh thinking and procedure as the baseline. Each run used
a freshly started server. Both threads finished on their own, without reaching the
98304-token cap. The laptop setup was assumed unchanged from the baseline (Omen
Performance mode, lid open, screen off, no pad) and was not reconfirmed.

| | Baseline b12001 | b12110 run 1 | b12110 run 2 | Mean vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Decode <32K (tok/s) | 78.4 | 81.3 | 81.4 | +3.8% |
| Decode 32–64K | 65.4 | 68.6 | 61.7 | -0.4% |
| Decode 64–96K | 47.4 | 47.0 | 46.2 | -1.7% |
| Later prefill <32K (tok/s) | 1225.1 | 1312.2 | 1292.2 | +6.3% |
| Later prefill 32–64K | 849.2 | 879.0 | 873.9 | +3.2% |
| Later prefill 64–96K | 563.1 | 594.6 | 602.3 | +6.3% |
| First 9902-token prefill | 1501.2 | 1465.2 | 1472.4 | -2.2% |
| MTP acceptance | 0.6832 | 0.6801 | 0.6200 | |
| Completed requests | 43 | 28 | 17 | |
| Largest prompt | 96253 | 95534 | 83035 | |
| Prefill / decode tokens | 89189 / 10831 | 83678 / 16934 | 71587 / 13274 | |
| Overall prefill / decode (tok/s) | 803.95 / 49.90 | 872.72 / 47.94 | 957.37 / 47.37 | |
| Start temperature (°C) | 52 | 51 | 51 | |
| Busy temperature median / max (°C) | 70 / 77 | 72 / 78 | 72 / 76 | |
| Busy power median / max (W) | 142.1 / 189.0 | 140.9 / 175.5 | 142.6 / 184.4 | |
| Throttle-reason share of busy samples | 99.4% | 99.1% | 99.7% | |
| Sampled peak GPU memory (MiB) | 20975 | 20979 | 20979 | |
| Checkpoint restores | 8 | 6 | 2 | |

Later prefill excludes each run's first request. Overall rates depend on the
context and output mix and are not comparable here: the b12110 runs decoded more
of their tokens at long context. The <32K and 32–64K decode buckets hold only
274–580 tokens per b12110 run, so they are weak evidence. Run 2 has ten requests
in 64–96K. No aborts, `seq_rm` failures or out-of-memory strings were logged, and
the fixture was unchanged after both runs.

## Flash Attention over K/V rows

b12110 enables `flash_attn_ext_rows` for unified KV, confirmed in the server log
for both the main and MTP contexts. Every step then uses the MMA kernel instead of
the vector kernel for single-token steps. `LLAMA_KV_ROWS_DISABLE` turns it off.

| Test (b12110, one sequence) | Rows on | Rows off | Rows on vs off |
| --- | ---: | ---: | ---: |
| No MTP, batched-bench 64K decode | 26.4 tok/s | 32.0 tok/s | -18% |
| No MTP, batched-bench 96K decode | 21.8 | 28.6 | -24% |
| No MTP, batched-bench 64K prefill | 1145 | 1105 | +4% |
| No MTP, batched-bench 96K prefill | 925 | 908 | +2% |
| MTP, filler prompt 8K decode (6 reps, median) | 94.3 | 93.1 | +1% |
| MTP, filler prompt 32K decode (6 reps, median) | 79.8 | 80.4 | -1% |
| MTP, filler prompt 32K prefill | 1367 | 1344 | +2% |
| MTP, real code 64K decode | 57.0 / 57.5 | 58.3 | -1 to -2% |
| MTP, real code 96K decode | 50.0 / 50.1 | 52.0 | -4% |
| MTP, real code 64K prefill | 1056 / 1059 | 1024 | +3% |
| MTP, real code 96K prefill | 868 / 869 | 848 | +2% |

The penalty is large only for single-token steps. MTP verifies several tokens per
step, which already used MMA, so production sees about 1–4% lower decode at 64K+
and 2–3% higher prefill. The real-code test used fixture TypeScript at temperature
0 with 256 output tokens, on-off-on order, so it has one rows-off sample. Draft
counts and output samples matched between rows on and off.

Long agentic sessions are decode-heavy, so rows off may be slightly faster for one
thread above 64K. The difference is small, needs a code change to apply through
llama-swap, and would give up the upstream multi-sequence improvement that the
two-slot profile can use. Rows stay on, the upstream default.

## Limits

- Two concurrent or occupied slots were deliberately not tested on b12110.
- There was no matched fixed-prompt b12001 run on the current defaults, so fixed
  prompts compare rows on and off, not builds.
- The baseline is one run, and generated histories differ between runs.
- The runs are not task-completion or answer-quality results.

## Validation and evidence

`task agent:verify`, install/verify, `smoke swift15` (text, red/blue images),
`pi_smoke.py swift15 --port 1257` (all checks) and Swift→Bolt→Swift through
llama-swap passed.

Ignored evidence: `logs/swift15/rows-ab-20261104T212228Z/` (fixed, batched and
long-context results plus their scripts),
`logs/swift15/agentic-20261104T221308Z-no-pad-b12110-1/` and
`logs/swift15/agentic-20261104T222347Z-no-pad-b12110-2/`. The directory ending in
`-aborted-stale-server` was discarded before any thread ran, because that server
still held run 1's context.

- Run 1: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:4eb5ee7bedd8c207e66c227676fa165124c33f2d3e43d1105df829678aea8984>
- Run 2: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:3a94b0308d11104ef1cc65eb868065f75aa23d14e2fa73dbf6bf92af75e0e6ee>
