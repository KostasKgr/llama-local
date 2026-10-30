# Swift 1.5 — Windows-native llama.cpp vs WSL, 2026-10-30

Same llama.cpp b12001 (`eb4e244be`) CUDA 13.4 build and identical server flags,
run as the Linux binary in WSL2 and as the Windows `llama-server.exe`, on the
RTX 5090 Laptop GPU (driver 592.27). Laptop setup for every run: Omen Performance
mode, cooling pad, lid open, built-in screen off. Swift was then moved to the
Windows build (profile `"platform": "windows"`, runtime `win-b12001-cuda13.4`).
Evidence: ignored `logs/swift15/winbench-20261030T124*/` and
`logs/swift15/agentic-20261030T1*-win-*/`.

## Fixed-prompt A/B

`experiments/swift_fixed_compare.py` (then a scratch copy): temperature 0, seed
42, filler prompts, 128 output tokens, prompt cache off, after a warm-up. Three
reps each, servers started directly on 1235 with llama-swap stopped. Draft counts
were identical in every rep on both (95 of 95 accepted at 8K/32K), so output was
the same.

| Prompt | WSL decode | Windows decode | WSL prefill | Windows prefill |
| --- | ---: | ---: | ---: | ---: |
| 8K | 93.5 tok/s | 101.7 (+9%) | 1,397 tok/s | 1,550 (+11%) |
| 32K | 80.5 | 86.5 (+7%) | 1,263 | 1,367 (+8%) |

Rep-to-rep spread was 1–3%. Power was the same (about 168 W median, memory at
14 GHz); Windows peaked at 77 °C against 72 °C. Both passed the text and red/blue
image checks. The 32-token row is omitted: timing noise exceeds any difference.

## Agentic runs

Procedure as in `reports/swift15/AGENTIC_PROMPT.md` and the README runbook (PI
1.4.0 via T3 Code, thinking xhigh, fixture `pingdotgg/t3code@32ecd717`), counted
up to the first request above 96K. WSL figures are the two b12001 CUDA 13.4 runs
from `AGENTIC_POWER_MODES.md`. Windows runs 1–2 used a temporary wrapper behind
llama-swap; the integrated run used the final `platform: windows` support (backend
port 1257). Model warmed before every Windows run.

| | WSL 1 | WSL 2 | Windows 1 | Windows 2 | Windows integrated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Requests (to 96K) | 38 | 31 | 36 | 40 | 30 |
| Decode tok/s, <32K | 74.5 | 78.3 | 79.6 | 78.4 | 76.5 |
| Decode tok/s, 32–64K | 62.1 | 66.5 | 62.0 | 67.1 | 64.0 |
| Decode tok/s, 64–96K | 46.0 | 42.9 | 47.3 | 46.2 | 51.0 |
| First request (9,841 tok) prefill tok/s | with model load | 1,351 | 1,432 | 1,485 | 1,473 |
| MTP draft acceptance | 0.67 | 0.63 | 0.65 | 0.65 | 0.71 |
| Busy GPU samples (1 s) | 315 | 408 | 401 | 376 | 222 |
| SW power cap, share of busy | 99% | 100% | 99% | 99% | 99% |
| Thermal / HW slowdown / HW power brake | 0% | 0% | 0% | 0% | 0% |
| GPU power median / max (W) | 143 / 190 | 142 / 185 | 144 / 189 | 144 / 189 | 150 / 176 |
| GPU temperature median / max (°C) | 67 / 72 | 67 / 72 | 69 / 75 | 69 / 77 | 72 / 77 |
| Start temperature (°C) | 50 | 50 | 44 | 49 | 49 |

- Mean of Windows 1–2 against WSL 1–2: decode +3% / 0% / +5% by context range,
  identical first request +8% prefill, same power, 3–5 °C warmer peak.
- The integrated run (final implementation, port 1257) matches Windows 1–2:
  76.5 / 64.0 / 51.0 tok/s and 1,473 tok/s first-request prefill. Its 64–96K
  figure is helped by the highest acceptance of the day (0.71).
- The agentic decode gain is smaller than the fixed-prompt one and within
  run-to-run variation (acceptance 0.63–0.67); prefill matches the A/B.
- Fan (user-reported, Windows 1 only): 2,300 RPM about a minute in, 3,400 by about
  2 minutes, 4,700 after about 4 minutes (near 86K context), then alternating
  4,300/4,700. The WSL pad runs reached 4,700 after 1.3–2.3 minutes. One run; no
  readings for the WSL b12001 runs.
- Every run left the fixture unchanged.

## Integration findings

- WSL interop: SIGTERM or SIGKILL to the WSL side of a Windows exe terminates
  the Windows process, so the profile's systemd scope still owns the server. The
  exit is asynchronous (VRAM is freed immediately; the process lingers for a
  moment), so stop and llama-swap cleanup wait for it.
- Mirrored networking: a port released by a Linux listener is unbindable for
  Windows for about a minute, and vice versa. A shared backend port made
  Bolt↔Swift switches fail or stall; a dedicated Windows backend port (1257) and
  bind waits on both sides fixed it.
- User systemd services (llama-swap) do not inherit the Windows PATH, so
  `powershell.exe` is called by absolute path when not found on PATH.
- Validated: install/verify on `C:\AI\llama-local`, `smoke swift15`,
  `pi_smoke.py swift15 --port 1257`, Swift→Bolt→Swift through llama-swap (Swift
  11–14 s, Bolt 9 s loads), `swap stop`, and SIGKILL of the llama-swap service
  with Swift loaded (cleanup removed the Windows server, no stale state).

## Build facts

Both release builds report CUDA ARCHS 750–1210, `USE_GRAPHS = 1`,
`BLACKWELL_NATIVE_FP4 = 1`, FA quants including q8_0/q8_0, and contain sm_120a
code objects; no forced MMQ/cuBLAS. `--backend-sampling` (experimental) exists
and is off; MTP draft sampling already runs on the backend.
