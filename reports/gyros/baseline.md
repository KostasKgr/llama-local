# Gyro-S baseline

Measured on 2026-11-11 UTC. Four budgeted investigations completed through real
T3 → pi → llama-swap → the managed Windows Strata server. Successful read-tool
results cover all five required source files, and each final answer includes its
fresh nonce. Source review found inaccurate citations and false-positive risks;
review generated findings against the code before acting on them.

## Build and machine

The repository base is `d4e5d405377694f4228499baaf0165f9cdcbac77` plus the working-tree
implementation. Each run records source-file SHA-256s, the profile, artifact pins,
prompt hash and native configuration. Existing Bolt/Swift profile files,
`manifests/runtimes.json` and `manifests/llama-swap.json` match that base byte for
byte. No fresh performance comparison against those models is claimed.

- Windows 11 Home `10.0.26300`; Ubuntu 24.04 WSL; Core Ultra 9 275HX;
  31.42 GiB usable Windows RAM; RTX 5090 Laptop GPU, 24,463 MiB reported VRAM,
  SM 12.0, NVIDIA driver 592.27. AC online; Windows Balanced power scheme.
  No power settings changed. Vendor thermal mode and background desktop activity
  were not controlled.
- Official stable [Strata v0.2.13](https://github.com/Niko1221/Strata/releases/tag/v0.2.13):
  actual Windows Control **0.1.59**, bundled CPython **3.12.15**, and Windows engine
  **0.1.59**, built with CUDA **13.0**. The Linux-only Control patch is not a Windows
  runtime choice. The engine component signature, file hashes, driver floor,
  SM coverage, version, geometry and required interfaces passed verification.
- llama-swap **v267** and pi **1.5.0**. Gyro uses a per-model local pi endpoint;
  the shared provider transport and all unrelated model entries match the
  pre-registration backup. Pi's default model remains Swift.
- Full `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`, 58,487,015,808 bytes, including the
  PLE table. It stays in `C:\AI\llama-local\strata`. Runtime, packs, user state and
  logs use that managed root; the existing `C:\AI\strata` installation is excluded.

Complete source URLs, immutable model revisions, sizes and preparation pins are
in [strata.json](../../manifests/strata.json) and
[models.json](../../manifests/models.json). Key verified SHA-256s:

| Artifact | SHA-256 |
| --- | --- |
| `strata-control-0.1.59-windows-x86_64.zip` | `c81f6800094942a5e750c7d93f348062bc16484f946ea83b0c0d2f8c91c44e87` |
| `strata-engine-0.1.59-windows-x86_64-cuda-cuda13.tar.gz` | `3f071a7ccbbf3dcdefed8bff777896d8dd0b580417a078154a7d2c2402ec4158` |
| `gyros/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` | `c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87` |
| `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` | `4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd` |

The stock DFlash2 sources produced both expected packs; the selected pack is the
stock Q4 selector variant, not a tuned `strata1` pack. Native preparation used the
pinned `iq_pack.py --compat-bf16`: 48 Hadamard-folded expert layers were recognized;
364 dense tensors were rounded, with reported maximum absolute error 0.0144.
Expert and PLE bytes remain in the original GGUF. Prepared-output inventories are
local provenance and are checked by `verify`.

## Settings and procedure

The measured [profile](../../profiles/gyros.json) has context **131072**, one slot,
int8 KV in VRAM, resident experts, automatic expert-cache/prefill sizing,
**6 GiB RAM headroom**, direct SSD PLE reads, stock DFlash2 window **6** and
confidence **0.35**. A **1536 MiB VRAM reserve** replaced the engine's default
reserve after the initial smoke warned that only about 600 MiB remained.
Only the new Gyro profile changed. All scored runs use the final reserve.

Typical placement was 14.35 GiB of GPU experts (14,681 slots), about 12.9 GiB of
page-locked RAM experts including lendable slots, and an effective prefill chunk
of 8192. Some starts reported a small uncovered lendable portion because RAM
availability varied. The engine logs preserve the effective placement and file
fallback. The full GGUF remains required for expert/PLE access.

The [operating guide](../../docs/strata.md#check-and-measure) contains repeatable
commands. Fresh-process trials use unique scopes and an owned proxy; production
T3 trials use a fresh engine and top-level thread per attempt. GPU memory must be
below the occupied threshold, required ports must be free, and the GPU cools below
50 °C before loading. No unrelated server is stopped. These are process-cold
loads with an uncontrolled, partly warm OS file cache.

## Fixed API baseline

Three repetitions, nine cases each, temperature 0, reasoning off and a 512-token
cap. Fresh requests contain repeated records followed by text, code or JSON
instructions. Actual token counts replace the approximate size targets below.
Warm-up and API validation are excluded from request-rate calculations.

Public first-request load latency was **82.04–82.16 s**; the Windows owner's
internal readiness timer was **71.36–71.92 s**. The public timer also includes
launcher verification and the one-token warm-up request.

Medians across three repetitions; prefill counts newly read tokens after cache
reuse. Decode uses engine-generated tokens and native decode time. A one-token
probe has no scored decode rate. See the [chart](throughput.html).

| Actual prompt tokens | Output instruction | Prefill tok/s | Decode tok/s | Request wall s |
| ---: | --- | ---: | ---: | ---: |
| 4,276 | text | 1,821.4 | 112.1 | 6.97 |
| 4,280 | code | 2,152.8 | 136.5 | 4.01 |
| 4,277 | json | 2,166.6 | 187.8 | 4.67 |
| 17,357 | text | 2,434.0 | 106.1 | 12.00 |
| 17,361 | code | 2,474.3 | 136.7 | 9.54 |
| 17,358 | json | 2,476.1 | 201.3 | 9.76 |
| 35,789 | text | 2,570.1 | 113.2 | 18.56 |
| 35,793 | code | 2,567.9 | 141.1 | 16.66 |
| 35,790 | json | 2,570.4 | 199.7 | 16.69 |

The nine-case weighted rates per repetition were **2846 / 2488 / 2421 tok/s**
for prefill and **146.5 / 139.5 / 130.3 tok/s** for decode. Repeated source and
copy-heavy output favor prompt lookup; these figures do not estimate general
agentic speed. The output cap also limits free-form responses; code/JSON workload
outputs were retained, not treated as correctness-certified artifacts.

## Real T3 agentic baseline

Fixture: `/projects/agentic-bench-t3code` at
`32ecd71726fee16f335484c1f95cbce126d461f6`, clean before and after each attempt.
The task traces Claude thread launch and first-prompt delivery, requiring full
reads of ThreadLaunchService, ProviderTurnStartService, ProviderSessionManager,
ProviderRegistry and ClaudeDriver, a numbered call chain, reuse/error handling,
three risks, and a fresh nonce read through the actual `read` tool.

Every thread explicitly selects pi / `llama-local/qwen3.8-flash-next-gyro-s-strata`,
thinking `high`, default interaction mode, full-access runtime, and the fixture's
root workspace. Native pi events confirm the actual model and thinking level.
The fixture permission policy denies writes, edits, network/MCP work and external
paths. Its previous policy/exclude content is restored after each bracket.

P1 used the original investigation prompt. It exceeded the **98304-token**
benchmark cap and was interrupted after an observed 108562-token request; the
sampling interval allowed overshoot. All five full reads succeeded, but no final
answer was produced. It is retained as a failed attempt. P2 adds a soft research
budget: 32 tool calls, contiguous required reads, and limited extra reads. Its
four repetitions use the same prompt apart from the nonce-file path. Their
prompt/configuration hashes are retained. Do not compare P1 and P2 as identical
workloads.

Wall time below comes from T3 run start/completion timestamps and excludes model
loading. Rate pairs are weighted new-token prefill / engine decode, including
reasoning and tool-call generation. Warm-up is excluded; API output counts remain
separate in raw request records. P2 public cold-load requests took about 82.1 s;
internal owner readiness was 69.8–72.3 s.

| Thread | Prompt | Result | Wall s | Max prompt | API requests / tool calls | Prefill / decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [Gyro-S agentic baseline 1](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:df63765376bb4fd88b3f139a43b02a81bb1ad6a7440f42a0cc7980c7478e8db5) | P1 | interrupted; no answer | 161.5 | 108,562 | 44 / 54 | excluded: failed attempt |
| [Gyro-S agentic baseline 2](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:33cbc1fa123200d185ce0244b6ffbdebbce09c165700283237111a25ffd165b2) | P2 | mechanical pass; content mixed | 173.4 | 84,757 | 30 / 34 | 979.8 / 120.8 |
| [Gyro-S agentic baseline 3](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:f42af35668d66ecbfe26ca89e7cce8cb6047cd4a7149eec5baee118a0bca2306) | P2 | mechanical pass; content mixed | 172.7 | 82,772 | 22 / 32 | 1,020.2 / 122.3 |
| [Gyro-S agentic baseline 4](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:64c898ddd9a78f8f7d07ed9fd75d1ab5db6e2bc24f3c821d7792190a07144a02) | P2 | mechanical pass; content mixed | 153.4 | 84,790 | 14 / 34 | 1,151.1 / 125.6 |
| [Gyro-S agentic baseline 5](t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:1ffed9c1e7d93436361ace583413fb707d533b487a459dc684305b065f317210) | P2 | mechanical pass; content mixed | 190.5 | 87,561 | 24 / 34 | 1,030.4 / 124.0 |

### Throughput by conversation length

The following rates are derived from the completed P2 requests above. Each
context bucket uses the full prompt length: `<32K` means fewer than 32768 tokens,
and the other boundaries are 65536 and 98304 tokens. Later-turn prefill excludes
the initial full prefill and cached tokens. Within each attempt and bucket,
rates divide the sum of newly evaluated or engine-generated tokens by the sum of
the corresponding native phase times; decode excludes single-token probes. The
median is taken across the four attempt rates, not across individual requests.
There were no later full prefills in these attempts.

| Measure | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 | Median | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial full prefill, tok/s | 626.27 | 636.94 | 649.24 | 639.31 | 638.13 | 626.27–649.24 |
| Later prefill <32K, tok/s | 952.03 | 989.90 | 1,178.71 | 1,050.17 | 1,020.03 | 952.03–1,178.71 |
| Later prefill 32–64K, tok/s | 1,199.96 | 1,261.81 | 1,432.20 | 1,294.41 | 1,278.11 | 1,199.96–1,432.20 |
| Later prefill 64–96K, tok/s | 961.32 | 998.06 | 1,092.13 | 943.27 | 979.69 | 943.27–1,092.13 |
| Decode <32K, tok/s | 143.47 | 147.35 | 144.59 | 139.40 | 144.03 | 139.40–147.35 |
| Decode 32–64K, tok/s | 136.81 | 143.78 | 149.95 | 150.92 | 146.87 | 136.81–150.92 |
| Decode 64–96K, tok/s | 115.08 | 114.26 | 116.38 | 120.23 | 115.73 | 114.26–120.23 |
| Overall prefill, tok/s | 979.85 | 1,020.17 | 1,151.11 | 1,030.42 | 1,025.30 | 979.85–1,151.11 |
| Overall decode, tok/s | 120.78 | 122.34 | 125.63 | 124.00 | 123.17 | 120.78–125.63 |

Initial prompts contained 8,836 / 8,841 / 8,841 / 8,838 tokens in attempts 2–5.
The `<32K`, `32–64K` and `64–96K` buckets contained respectively 9 / 9 / 12,
7 / 7 / 8, 5 / 4 / 5, and 4 / 6 / 14 completed requests. Results depend on
the generated history, output mix, engine timing and tokenizer; they are not a
matched comparison with the [Swift baseline](../swift15/T3_CP20_SINGLE_SLOT_MEMORY.md).

The four P2 mechanical audits require the exact model, successful matched
read-tool results covering every line of the five files, a successful nonce read,
that nonce in a substantive final answer, and no write/edit/MCP calls. Final
answers have 645 / 618 / 654 / 638 words. All fixtures remained clean and all measurement
collectors stopped. Two tool errors occurred in each completed attempt, followed
by recovery. The soft 32-call budget was exceeded by two calls in runs 2, 4 and 5;
read-only `sed` also appeared despite the prompt restriction.

Manual source review is **mixed**, distinct from those mechanical passes:

- The main launch → preparation → outbox → provider session → adapter → prompt
  queue path is recognizable, but many file/line references are inaccurate.
  For example, `ProviderSessionManager.open` is at line 2120, not run 3's 2244.
- Runs 2 and 4 incorrectly infer serialized preparation jobs from
  `Scope.make("sequential")`. That option controls
  [finalizer strategy](https://github.com/Effect-TS/effect/blob/main/packages/effect/src/Scope.ts);
  the preparations are forked independently. Actual source locations are
  ThreadLaunchService lines 218 and 803.
- Run 3 misreads a comment about failed/stopped starts as evidence that successful
  turns omit terminal events. Its handoff risk also confuses no-op persistence
  during budget checks with later actual delivery bookkeeping. Neither claim is
  a demonstrated bug. Live-cwd/instance reuse concerns remain conditional risks
  without a reproduced caller path.
- Run 5's idle-recheck risk overlooks the busy-turn/generation guards and the
  recursive recheck on the existing fiber. Its scope-close timeout claim also
  omits the detached join that continues waiting for cleanup. These are not
  demonstrated orphan-process or idle-release defects. Some exact citations are
  wrong: `providerThreadLoadKey` starts at line 304, not its cited 407.

These runs establish functional local tool use and answer delivery. They also
show that generated bug findings and precise citations require independent review.
No broad quality, token-parity or full-131072-token recall claim follows.

## Memory and available power

Windows process metrics sum the captured API/engine tree. Working set measures
resident pages; private bytes are committed virtual memory and can include GPU
backing, so private bytes are not physical RAM use. Fixed API trials sampled
16.83 GiB maximum aggregate working set, 40.19 GiB private commit, 58.11 GiB total
Windows commit, and 3.20 GiB minimum Windows available RAM. Reported GPU allocation
peaked at **22669 MiB (22.14 GiB)**.

Agentic values use each exact T3 run time window. GPU power uses samples with
utilization at least 10%; CPU values are RAPL package observations over that run.
Power columns show median / sampled maximum in watts.

| Attempt | Peak server working set GiB | Min Windows available GiB | GPU active W | CPU package W |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.62 | 4.13 | 154.8 / 191.6 | 47.7 / 102.1 |
| 2 | 16.53 | 3.95 | 141.7 / 184.9 | 59.4 / 90.8 |
| 3 | 16.53 | 3.15 | 142.9 / 192.3 | 58.9 / 108.4 |
| 4 | 16.49 | 3.95 | 142.3 / 191.1 | 61.9 / 95.0 |
| 5 | 16.52 | 3.73 | 142.3 / 189.4 | 64.2 / 91.1 |

In attempts 2–5, whole-GPU allocation peaked at 22672 MiB; busy-sample median /
maximum GPU temperatures were 65 / 71, 63 / 70, 65 / 70 and 65 / 71 °C.

GPU power limit was unavailable from the driver. Core RAPL, GPU clocks,
temperature, utilization, available RAM and total commit are retained in CSV.
Sampling at about one second can miss peaks. Raw idle-boundary GPU readings
included impossible values around 20 kW; these have zero utilization and remain
in the raw evidence as counter anomalies. The active-sample filter excludes them;
most also fall after their run windows. These counters do not measure
whole-machine power or energy. Background applications, OS cache state, thermal
behavior and sampling noise limit comparison and portability.

## Verification and evidence

`task agent:verify` passed **112 tests** on the finished code before the final
T3 repetition. Checks include archive/integrity/path validation, backend overrides,
exclusive proxy draining, pi transport preservation, ownership/PID reuse,
Windows nested-job cleanup after normal and abrupt owner exit, immutable runtime
files, telemetry and evidence-audit behavior.

The final audit found that samplers from attempts 1, 2 and 4 had survived `finish`:
their identities were captured before exec, with an empty command line. Cleanup
matched their exact recorded PID/start time and expected command before signaling
them. Startup now waits for the exact command line, and an incomplete identity
fails cleanup explicitly. Regression checks include a real sampler start/stop.
Earlier frozen summaries remain valid for their recorded brackets; their raw
`metrics.jsonl` files extend past finish. Overlapping read-only metrics polling
adds uncontrolled overhead to attempts 2–4. The final repetition follows the fix.
Attempt 5 confirms automatic sampler exit, restored fixture policy/exclude,
removed nonce, clean fixture Git state, and the selected model's actual tool use.

Sequential live checks passed for existing Bolt and Swift, Gyro with DFlash2 and
without a drafter, and Bolt → Gyro → Swift → Gyro → Bolt, including Swift images.
Gyro API checks verified arithmetic/extraction, thinking, streaming and a
structured tool-result round trip; pi passed discovery, text, thinking and an
actual read-tool round trip through the public proxy. A captured production
coordinator supervisor was killed deliberately: its owned Windows tree exited,
ports were released and GPU memory returned to baseline in **8.0 s**. No unrelated
process was terminated.

Raw evidence remains local and Git-ignored under `logs/gyros/` and the managed
Windows run directories. It is not included in the repository:

- API: `20261111T200420.714212Z-trial`, `20261111T200752.363909Z-trial`,
  `20261111T201138.535629Z-trial` (`baseline.json`, responses, configuration,
  owner record, engine/server logs and telemetry).
- T3: `20261111T202232.869022Z-t3-01`, `20261111T203115.012771Z-t3-02`,
  `20261111T220109.096902Z-t3-03`, `20261111T221052.274328Z-t3-04`,
  `20261111T222612.077657Z-t3-05`
  (`meta.json`, `prompt.txt`, `thread-launch.json`, `thread-result.json`, native
  `pi-session.jsonl` plus SHA-256, `metrics.jsonl`, `summary.json`,
  `review-window.json`, logs and telemetry).
- `agent-verify.log`, `verify.log`, `install.log`, `production-api.log`,
  `production-failure-cleanup.json`, `hardware.json`, `collector-race-cleanup.json`,
  `final-cleanup-audit.json`; pi smoke
  `pi-smoke-20261111T200319.713329Z`; swap evidence
  `logs/swap/smoke-20261111T195609.451634Z`.

The managed three-profile coordinator is left available on `127.0.0.1:1235` with
Gyro loaded and healthy. Benchmark scopes, collectors and fixture policy changes
are cleaned up. Installation/verification, opt-in selection and rollback commands
are in the [Strata guide](../../docs/strata.md); the default coordinator selection
remains Bolt/Swift when started without `--profiles`.
