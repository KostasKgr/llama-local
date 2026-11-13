# Strata Gyro-S

The `gyros` profile serves text and tools with Strata on Windows. Its runtime,
full GGUF, prepared pack and drafter live in separate managed Windows storage.
It does not use an existing Strata application installation.

Runtime and preparation pins live in `manifests/strata.json`; model sources live
in `manifests/models.json`. Defaults and the client alias live in
`profiles/gyros.json`. Installing this profile leaves the llama.cpp profiles and
their runtime files intact.

## Install and verify

Run from the repository root:

```bash
python3 llama.py install gyros
python3 llama.py verify gyros
python3 llama.py start gyros --dry-run
```

Installation requires a Windows NVIDIA GPU compatible with the pinned engine,
WSL interoperability, and enough free space for the complete GGUF and preparation.
The first installation checks an 80 GiB allowance, counting a resumed partial
model download. It verifies downloads before promotion and prepares the native
model pack and stock DFlash2 drafter with the bundled Windows interpreter.
No compiler, system Python installation, driver change or CUDA toolkit is needed.

The pack must include `experts.bin`: install runs the bundled
`tools/iq_pack.py` with **`--experts-bin`** (and `--compat-bf16`), which copies
every expert out of the GGUF into one 24 GiB file. It takes about 5 minutes and
needs that much extra space. An installation made before 2026-11-12 adds it on the
next `install gyros`; start refuses a pack without it. Keep passing this flag in
any manual or future preparation. Without it the engine reads experts from the
GGUF in place and loads about 3.5 times slower (see "Loading and expert placement").

`LLAMA_STRATA_WINDOWS_ROOT` in the ignored `.env` selects a dedicated local Windows
directory. Its default is `C:\AI\llama-local\strata`; it is independent of
`LLAMA_WINDOWS_ROOT`. Keep it on a fast local SSD: the full GGUF contains the PLE
table, which the engine reads during inference. Do not remove the GGUF after
preparation.

`verify` checks complete source artifacts, installed runtime files against their
pinned archives, gguf-py source identities, and prepared outputs. Generated native
pack hashes are local provenance; the drafter additionally has upstream expected
output hashes. Starting a profile never downloads or prepares missing files.
Install and `verify` do all hashing. Start checks only that files exist with
their recorded sizes: prepared outputs against `installation.json`, the GGUF
against its pin, and engine files against the engine manifest. Hashing at start
cost about 1 s in the launcher and 0.6 s in the Windows owner, or about 21 s more
with `experts.bin`. The engine signature is checked at install. Run `verify gyros`
after any doubt about the managed storage.

Installation is resumable. A corrupted installed runtime or prepared output is
refused instead of being accepted as a new baseline. For changed preparation pins,
use a fresh managed storage directory and keep the prior directory for rollback.

## Serve

```bash
python3 llama.py start gyros
# From another terminal:
python3 llama.py status gyros
python3 llama.py stop gyros
```

The foreground launcher owns a Windows Job Object containing the API server and
its engine descendants. The Windows owner stops after launcher heartbeat loss;
closing its job also ends descendants after a crash. Stop waits for the captured
owner to exit. It never stops processes by executable name.

The WSL scope limits apply to the supervisor, not Windows RAM or VRAM. Strata's
resident-expert mode keeps experts outside the GPU cache in RAM when they fit;
the engine may use its documented file-backed fallback when memory is short.
Inspect the engine log for the effective cache, RAM placement and fallback.
The profile's VRAM reserve leaves room for runtime workspace and display use;
the automatic expert cache uses the remaining capacity.

Start supports `--context`, `--port`, `--spec dflash2|none` and `--draft-max`.
Equivalent `GYROS_*` environment variables follow the normal CLI-over-environment
precedence. llama.cpp cache, batch and loading flags are refused for Strata.
Without a drafter, the native IQ path still uses its minimum verification buffer;
prompt-lookup drafting is disabled as well.

## Terminal monitor

With Gyro loaded through llama-swap, open PowerShell and run:

```powershell
cd C:\AI\llama-local\strata\runtimes\control-0.1.59
.\python\python.exe -B strata.py monitor --url http://127.0.0.1:1257
```

Press `q` to quit. Port 1257 is the default Windows backend port; the public
llama-swap port 1235 does not expose Strata's telemetry endpoint. For a directly
started Gyro server, use port 1235 by default instead. Match any port overrides
and replace the directory above if `LLAMA_STRATA_WINDOWS_ROOT` was customized.
Run the monitor on Windows: the Windows backend may not be reachable from WSL.

## Loading and expert placement

Gyro-S has 48 layers of 512 experts. The engine places them by the ranking in
the Control bundle's `data/expert-profile.bin`, which lists every (layer, expert)
pair most-used first. It ships with Strata and is not built from local use.

- **VRAM:** `--expert-cache auto` sizes the GPU cache from free VRAM after the
  profile's reserve and the drafter (14,681 slots, 14.35 GiB here), then copies
  the top-ranked pairs into it in rank order.
- **RAM:** `--resident-experts` continues down the same ranking, skipping pairs
  already on the GPU, into page-locked RAM. Its budget is available Windows
  commit at that point minus `ram_keep_gib`, so less free RAM at start means a
  smaller RAM copy (5.4–13 GiB observed).
- **Files:** every other expert is read from the GGUF when a token routes to it.
- **While serving:** an adaptive tier swaps experts between VRAM and RAM
  according to actual routing. Nothing is kept across restarts unless the
  engine's `--expert-profile-save` is used, which the profile does not set.

Where the experts come from decides the load time. With `experts.bin`, the
engine closes its file mapping and fills the GPU cache and the RAM copy with
unbuffered parallel reads (engine log: "the GPU cache's fill and the RAM copy
read unbuffered"). With the GGUF in place, that path is unavailable on Windows
("read through the mapping (experts.bin only)", `begin_startup_unbuffered` in
`src/core/expert_source.cpp`): the fill copies one expert at a time through the
mapping, with under one read in flight.

Measured 2026-11-12, alternating sequential starts, GPU free, a 28-token and a
22,180-token prompt with reasoning off:

| Pack | Load | GPU-cache fill | RAM copy | 22K prefill | Decode (short / after 22K) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GGUF in place | 79.3 / 80.6 s | 50.6 / 51.5 s, 0.3 GB/s | 14 / 15 s | 950 / 953 tok/s | 75 / 78, 88 / 89 tok/s |
| `experts.bin` | 22.4 / 22.1 s | 4.1 / 4.0 s, 3.8 GB/s | 5 / 5 s | 2,795 / 2,807 tok/s | 102 / 86, 124 / 105 tok/s |

A cold first request through llama-swap (launcher checks, engine load, answer)
took 30.6 s, against about 82 s before. Of that, the launcher took 1.9 s before the
engine scope (1 s of it a connect timeout probing the free backend port), the
owner's checks 0.6 s, the engine load 21.4 s, and about 5 s passed between engine
readiness and the answer. Hashing at start had added about 1.6 s (now at install).
Both first requests right after a fresh `swap start` were slower (41.9 s, and
once an empty reply after 130 s with no engine started), not yet explained. Both placed
14,681 experts in VRAM and 12.99 GiB in RAM. Minimum available RAM
and peak commit during load were unchanged; CPU package energy during load fell
from about 1.8 to 0.55 kJ. Prefill and decode were not re-measured beyond these
two prompts here. The [Gyro-S baseline](../reports/gyros/baseline.md) compares a
real T3 agentic attempt with `experts.bin` by context range against the superseded
[GGUF-in-place baseline](../reports/gyros/GGUF_IN_PLACE_BASELINE.md).

The fill's look-ahead hint (`STRATA_FILL_AHEAD`, default 256) is unrelated: it
depends on read-ahead, which the 0.1.59 engine disables on Windows
(`read_ahead_enabled()` returns false in `src/platform/memory.cpp`), so the engine
logs `0 pairs ahead` whatever value is set. Upstream issue #4064 reported the
single-threaded fill on Linux; its fix (#4065) was not merged.

`--pack gyros-NAME` (or `GYROS_PACK`) starts from a trial pack folder beside the
installed one under the managed `packs/` directory. Install and verify do not
inventory trial packs; copy the installed pack's files to make one.

## llama-swap and pi

The default coordinator selection remains Bolt and Swift. Enable Gyro explicitly
after installing all selected profiles, with no other server using the GPU or ports:

```bash
python3 llama.py swap config --profiles bolt35 swift15 gyros
python3 llama.py swap start --profiles bolt35 swift15 gyros
python3 llama.py swap status
```

All selected models share one exclusive group. A switch completes backend cleanup
before loading another model. Selecting Gyro gives the proxy a longer readiness
deadline for its larger model. Manual production-profile start/stop refuses while
the coordinator owns the machine; use `swap stop` first.

Register only Gyro's transport against the local public proxy, preserving the
shared provider endpoint used by other pi models:

```bash
python3 llama.py pi add gyros --dry-run --model-base-url http://127.0.0.1:1235/v1
python3 llama.py pi add gyros --model-base-url http://127.0.0.1:1235/v1
pi --list-models llama-local
```

Registration preserves unrelated models and settings and creates a backup before
an actual edit. `--update` is required for conflicting managed values. It does not
select a default model. Strata advertises its supported reasoning-effort levels;
the profile remains text-only.

## Check and measure

Run `task agent:verify` after changes. When installed, its Windows lifecycle tests
use CPU-only API fixtures and nested child jobs; they never load model weights.
Live GPU checks must run sequentially:

```bash
python3 llama.py smoke bolt35
python3 llama.py smoke swift15
python3 llama.py smoke gyros
python3 experiments/swap_smoke.py --profiles bolt35 swift15 gyros
```

Against an already loaded Gyro backend, check pi inference through the public proxy
and health through the private backend separately:

```bash
python3 experiments/pi_smoke.py gyros \
  --base-url http://127.0.0.1:1235/v1 --health-url http://127.0.0.1:1257
```

With the coordinator stopped and GPU free, the fixed baseline owns its trial proxy
and unique model scopes, including cleanup on failure:

```bash
python3 experiments/strata_benchmark.py run --repeats 3
```

For a real T3/pi investigation, start the three-profile coordinator and bracket a
new thread on the benchmark fixture:

```bash
python3 experiments/strata_benchmark.py start --label t3-01 \
  --fixture /projects/agentic-bench-t3code
# Use the printed run directory's prompt.txt in a new T3/pi thread on that fixture.
python3 experiments/strata_benchmark.py finish RUN_DIRECTORY \
  --thread-id THREAD_ID --pi-session PI_SESSION_FILE
python3 experiments/strata_benchmark.py report logs/gyros
```

Select the registered Gyro alias explicitly, thinking `high`, and the fixture's
root workspace. Use a fresh engine and thread for each attempt. Stop only that
thread if it exceeds twenty minutes or 96K context. Preserve failed attempts.
The session audit requires successful read-tool results covering the five required
files and the random nonce in the final answer. Review the cited call chain and
risks against the source; thread completion alone is insufficient.
The prompt limits extra research so the required full-file reads fit the context
budget. Keep its recorded hash when comparing repeated attempts.

Logs and telemetry stay under ignored `logs/gyros/` and the managed Windows run
directory. `disk.csv` samples system-wide disk reads, read latency, read queue
length and hard page faults each second. At readiness the owner writes `load.json`:
the engine's own phase durations (dense weights, GPU-cache fill, RAM complement),
available RAM at start, time not covered by those phases, and disk reads during
the load. The Windows owner contains its GPU/power collectors in its job. Timing
summaries exclude cached input from prefill and keep engine-generated token counts
separate from API output counts. Samples can miss peaks, and power counters do not
measure whole-machine energy. Fresh-process loading does not imply a cold OS cache.

## Roll back

```bash
python3 llama.py swap stop
python3 llama.py stop gyros
python3 llama.py swap start
```

This returns to the default Bolt/Swift selection without changing either profile.
Restore the backup printed by pi registration if no concurrent config edits have
occurred; otherwise remove only the Gyro model entry. Keep managed artifacts for a
later retry. Before deleting a managed directory, confirm its owner has exited;
never remove an unrelated Strata installation or shared llama.cpp runtime.
