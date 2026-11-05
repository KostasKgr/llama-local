# Bolt 3.5: WSL b11808 to Windows b12110, 2026-11-05

Bolt moved from WSL b11808 CUDA 13.4 (`4912ac24f`) to Windows-native b12110 CUDA
13.4 (`cd959bae9`), the runtime Swift pins. Only `runtime` and `platform` changed;
every launch flag is otherwise identical. This report is the WSL baseline and the
new Windows baseline.

## Result

- **Prefill is 15–24% faster and decode is 10–14% slower.** The prefill and overall
  decode ranges do not overlap. For a typical run's token mix these nearly cancel,
  with server time about 3–8% longer on Windows. Decode-heavy work loses more.
- **The CPU runs much cooler on average.** Median package power fell from about 39 W
  to 28 W, against 17 W idle. The user reported silent fans on Windows, where the
  fans had risen quickly on WSL. Peaks did not change.
- **Resident RAM is unchanged, at about 5.7 GiB.** The 3.5 GiB "extra" memory is the
  model's embedding tables, not a leak. It stays on Windows, but memory now returns
  to the system when the server stops, instead of staying in the WSL VM.
- **The 9 GiB memory limit no longer applies to Bolt.** The scope's cgroup held 1–2
  MiB during Windows runs. Swift has always been in the same position.
- **Windows and the build change are not separated.** Both changed together, so
  nothing here attributes a difference to one or the other.

## Setup

- Model: `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` with the BF16 DFlash drafter, maximum
  depth 7. One slot, 65536 context, f16 KV, batch/microbatch 2048/512, two
  checkpoints, 2048 MiB RAM prompt cache, sampling per the README.
- Hardware: RTX 5090 Laptop GPU, Intel Core Ultra 9 275HX, 32 GB RAM, on AC power.
  Laptop setup assumed unchanged from the Swift baselines and not reconfirmed.
- Workload: one T3/PI thread through production llama-swap, using the prompt and
  fixture commit in [AGENTIC_PROMPT.md](../swift15/AGENTIC_PROMPT.md). Each attempt
  had a freshly started server, loaded with `agentic_bench.py start --profile
  bolt35`, and was summarized with `finish RUN_DIR --max-context 57344`.
- Two differences from the Swift procedure. `thinking` is `high`, the only enabled
  level Bolt's binary control has. The cap is 57344 because PI compacts at 65536 −
  8144 = 57392 tokens.
- All WSL attempts ran first, then all Windows attempts, so they are not interleaved.

## Attempts

T3 reports `completed` even when a run did nothing, so every attempt was audited
against the PI session: real tool calls, all five required files read, fixture
unchanged. Each side had nine attempts.

| | WSL b11808 | Windows b12110 |
| --- | ---: | ---: |
| Valid | 6 | 8 |
| First turn wrote a tool call as text and stopped | 3 | 0 |
| Agent ran but skipped a required file | 0 | 1 |

The first-turn failure ends the run after one request. The model emits text such as
`$ bash{command: "ls ..."}` with no tool-call tokens, so the server has nothing to
parse and PI sees a final answer. Bolt samples at temperature 1.0, and the failure
is intermittent. 3 of 9 against 0 of 9 is suggestive only (Fisher exact p = 0.21).
Throughput below uses valid runs only.

## Throughput

Median of per-run rates, with the range across runs, in tok/s. Later prefill excludes
full prefills. "Long responses" are requests with at least 500 decoded tokens, which
removes the effect of the mix of short tool calls.

| | WSL (6 runs) | Windows (8 runs) | Change |
| --- | ---: | ---: | ---: |
| Decode <32K | 151.7 (146.9–165.6) | 136.3 (121.2–157.9) | -10% |
| Decode 32–64K | 121.5 (114.7–131.9) | 108.6 (91.1–125.8) | -11% |
| Decode overall | 145.7 (143.1–152.3) | 125.4 (107.2–139.5) | -14% |
| Decode, long responses | 147.7 (142.5–153.2) | 128.7 (104.2–139.4) | -13% |
| Later prefill <32K | 4817 (4664–5233) | 5817 (5079–6118) | +21% |
| Later prefill 32–64K | 3992 (3855–4084) | 4608 (4436–4723) | +15% |
| First 10081-token prefill | 4807 (4430–5064) | 5940 (5740–6447) | +24% |
| Prefill overall | 4654 (4517–4783) | 5387 (5180–5599) | +16% |
| Draft acceptance | 0.301 (0.289–0.330) | 0.309 (0.274–0.335) | |
| Server time per run (s) | 40.5 (35.6–44.0) | 43.7 (36.7–124.8) | +8% |

The 124.8 s run compacted four times over 24 requests. A fixed mix of 3300 decoded
and 80000 prefilled tokens works out to 39.8 s on WSL and 41.2 s on Windows (+3%).

Windows decode varies much more between runs. Within Windows runs it follows draft
acceptance (r = 0.91, eight runs), so that spread is mostly workload. Acceptance does
not explain the gap between platforms: at acceptance of about 0.30–0.31, WSL decoded
143–146 tok/s and Windows 120–128. Decode also rose with CPU clock (r = 0.84), but
clock and acceptance are not separable here, so the cause of the decode loss is open.

## GPU

| | WSL | Windows |
| --- | ---: | ---: |
| Busy median power (W) | 123.2 (117.9–130.3) | 138.9 (113.7–152.1) |
| Busy max power (W) | 176.0–197.8 | 172.8–209.0 |
| Busy median temperature (°C) | 59 (58–60) | 63 (61–66) |
| Busy max temperature (°C) | 62–66 | 67–72 |
| Start temperature (°C) | 45–52 | 47–54 |
| Sampled peak GPU memory (MiB) | 8107–8356 | 8139–8460 |
| Share of busy samples with a throttle reason | about 0.9 | about 0.9 |

The GPU works harder and runs about 4°C warmer on Windows, so GPU load does not
explain the quieter fans. Lower fan speed may itself contribute to that 4°C.

## CPU and fans

Measured from Windows counters that need no admin rights. RAPL package power is the
heat the CPU produces. Die temperature and fan RPM need admin rights and were not
measured. CPU telemetry was added partway through, so WSL has two runs and Windows
has eight. A 30-second idle reading with nothing loaded gave 17.3 W package, 8.4 W
cores and 3.9% utilization.

| Per run | WSL (2 runs) | Windows (8 runs) |
| --- | ---: | ---: |
| Median package power (W) | 38.7–39.3 | 25.4–31.3 |
| Peak package power (W) | 43.9–75.8 | 45.8–72.7 |
| Median cores power (W) | 29.4–30.9 | 17.9–21.8 |
| Median utilization | 15.6–15.9% | 10.7–14.2% |
| Server CPU (mean cores) | 0.77–0.81 | 0.69–0.79 |
| Whole WSL VM CPU (mean cores) | 1.07–1.12 | 0.17–0.22 |

Sustained power above idle roughly halves, from about 22 W to about 11 W. Short peaks
are similar on both, so the difference is sustained load, not spikes. No thermal-zone
throttling was recorded on either side, and the sampler itself used 0.01–0.03 cores.

**Fans (user observation, not measured):** the fans stayed silent through the Windows
runs. During the WSL runs they rose "really high really fast". This matches the lower
sustained CPU power, while GPU power went up, but it is a correlation.

The server uses the same CPU on both platforms, under one core. Its 24 threads are
not spinning. Windows counters attribute the extra WSL package power to neither the
server nor other WSL processes, which points to the cost of the WSL GPU path. That is
an inference, not a measurement.

## RAM

| MiB | WSL | Windows |
| --- | ---: | ---: |
| `CUDA_Host` model buffer | 3536 | 3536 |
| Server resident peak | 5883–6060 (cgroup) | 5631–5831 (working set) |
| Server resident after load | not separated | 4347–4348 |
| Server private bytes, peak | not applicable | 12612–13648 |
| Prompt cache peak | 1037–1192 | 1008–2021 |
| `vmmemWSL` working set, peak | 10622–11447 | 3571–5100 |
| `vmmemWSL` at the next run's start | 6096–10758 | 3507–4675 |
| Windows available RAM at run start | 7977–12237 | 14563–16661 |

The longest Windows run reached a 6662 MiB working set and 2021 MiB of the 2048 MiB
prompt cache. The other seven stayed below 5831 MiB and are the range above.

The main practical gain is at idle. On WSL, `vmmemWSL` grew by about 7 GiB and mostly
stayed there after the server stopped, leaving 4–8 GiB less available between runs.
On Windows the memory is released at exit. Only the first WSL attempt had a clean
idle reference, so changes from idle are not reported across WSL runs.

Windows private bytes look alarming but are commit charge, not RAM in use. They run
about 6.6 GiB above the working set, close to the 7 GiB of VRAM the server allocates,
which suggests WDDM charges commit for GPU allocations. Commit rose by a similar 11
GiB on WSL. That explanation is an inference.

### The 3.5 GiB

The October investigation of a Bolt memory stall found a 3536 MiB `CUDA_Host` buffer
despite full GPU offload, with no explanation. Reading the
tensor table from the GGUF header accounts for it exactly:

| Tensor | Type | Size (MiB) |
| --- | --- | ---: |
| `per_layer_token_embd.weight` [10752, 262144] | Q8_0 | 2856.00 |
| `token_embd.weight` [2560, 262144] | Q8_0 | 680.00 |
| Total | | **3536.00** |

These match the 2856 + 680 MiB `CPU_Mapped` split seen under mmap. Bolt uses the
`gemma4` architecture, whose per-layer embeddings are large lookup tables. llama.cpp
keeps input embedding tables in host memory, so "full offload" does not move them.
The GPU buffer also adds up: 2290.5 MiB of layers, 52.5 MiB projection and a 680 MiB
tied output head give the logged 3023 MiB. `token_embd` therefore exists in both
places. This is expected behavior and is identical on both platforms.

## Workload limits

The five required files total about 60K tokens, so the task does not fit Bolt's 64K
window. Every valid run compacted at least once, usually right after the fifth read,
and `keepRecentTokens: 0` then discards the file contents. Several final answers say
the content is no longer available, or use placeholder line numbers. Compaction, not
the 57344 cap, ended the growth of most runs. This is a speed measurement of the same
workload on both sides, not a task-completion or answer-quality result.

Other limits:

- Platform and build changed together.
- WSL has two runs of CPU telemetry, and its peak is one sample.
- The 32–64K decode rows hold little data: 3943 tokens across all WSL runs.
- `vmmemWSL` CPU time is not readable without admin rights.

## Problems found and fixed

- **Bolt→Swift failed with HTTP 500 on every switch.** llama-swap starts the next model
  when the previous scope exits, while the Windows server can still hold the port. The
  models used different ports until now. Under llama-swap, `start` now waits for the
  previous server to exit, which also keeps Swift from fitting itself around VRAM still
  in use. Manual starts still refuse an occupied port at once.
- `smoke.py` and `bolt_benchmark.py` could finish before the Windows server exited, so a
  following run found the port in use.
- `pi_smoke.py` passed on an open stdin, which `pi --print` reads forever. Agent shells
  hit this, and the request never left PI.
- `swap_smoke.py` did not wait for the port after a Windows server released it, and did
  not use the real Windows backend port.

## Validation

`task agent:verify`, install/verify, `smoke bolt35` with speculation off and on run
back to back, `bolt_benchmark.py --repeats 1` (thinking, conversational mode, streaming,
tool round trip, exact answers), `pi_smoke.py bolt35` (discovery, text, thinking, tool),
`smoke swift15` and `swap_smoke.py` (Bolt/Swift/Bolt on port 1257, Swift red/blue images)
passed. A separate production check through llama-swap passed five of five switches, at
6–9 s for Bolt and 16–17 s for Swift, with Swift at full 66/66 layer offload.

## Evidence

Ignored run directories are under `logs/bolt35/`, named `agentic-20261105T*-wsl-b11808-*`
and `agentic-20261105T*-win-b12110-*`. Discarded attempts end in `-aborted-no-tool-call`
or `-excluded-missed-file`. `summary-56k.json`, `requests-56k.json`, `host.csv`,
`gpu.csv` and `server-excerpt.log` hold the measurements. Threads are titled "Bolt
agentic bench" in the `agentic-bench-t3code` T3 project.
