# Local llama.cpp inference

One WSL/Linux repository for Swift 1.5 IQ3_S MTP and Nemotron 3.5 Bolt E4B DFlash.
Both profiles use a pinned Windows-native llama.cpp runtime with CUDA 13.4.
No source build, CUDA toolkit installation, or driver changes are required.

The optional [Strata Gyro-S backend](docs/strata.md) has its own pinned Windows
runtime, full GGUF and managed storage. It supports text and tools, with explicit
opt-in to the exclusive llama-swap group. The [Gyro-S baseline](reports/gyros/baseline.md)
compares load and real T3 tool-use speed with the superseded
[GGUF-in-place baseline](reports/gyros/GGUF_IN_PLACE_BASELINE.md). Its pack must include
`experts.bin` (`iq_pack.py --experts-bin`, written by `install gyros`): loading
fell from about 80 s to 22 s with it, and a cold request through llama-swap from
about 82 s to 31 s (2026-11-12, [details](docs/strata.md)).

Requirements: Python 3.12+, curl, working NVIDIA GPU passthrough, systemd user
manager and cgroup v2. Tested on an RTX 5090 Laptop GPU with 24 GB VRAM.

```bash
cd /projects/inference/llama-local
python3 llama.py install swift15
python3 llama.py verify swift15
python3 llama.py start swift15 --dry-run
python3 llama.py start swift15
# From another terminal:
python3 llama.py status
python3 llama.py stop swift15
```

Only one profile can use the default port at a time. Start refuses an occupied
port or active profile scope. Servers bind `127.0.0.1:1235`; API base is
`http://127.0.0.1:1235/v1`. Existing model aliases are preserved:
`swift-1.5-qwen3.8-27b-iq3s-mtp` and `nemotron-3.5-bolt-e4b-q4km-dflash`.
Swift uses one 131072-token slot with unified KV disabled. Concurrent client
requests queue for that slot; cached conversation states may be swapped through
the RAM prompt cache. Other profiles also use one slot.
Select the alias for the running profile in your client. OpenCode and PI
registration are available through the explicit commands below.

| Profile | Runtime | Context | KV | Batch/microbatch | Speculation |
| --- | --- | ---: | --- | --- | --- |
| swift15 | b12110 CUDA 13.4, Windows-native | 131072 | q8_0/q8_0 | 2048/512 | embedded MTP, max 3 |
| bolt35 | b12110 CUDA 13.4, Windows-native | 65536 | f16/f16 | 2048/512 | separate DFlash, max 7 |

The llama.cpp profiles use full layer offload, Flash Attention, Direct I/O loading,
two checkpoints and 2 GiB RAM prompt cache, except Swift's twenty checkpoints and 10 GiB
(`"cache_ram": 10240` MiB in `profiles/swift15.json`). Swift uses the runtime's
default minimum checkpoint spacing of 8192 tokens; this is a minimum, not an
exact periodic checkpoint schedule. OpenCode thread naming can displace
the main prompt and cause cache thrashing when the RAM cache is too small.
Swift's cache limit was raised after a 2302 MiB prompt state exceeded the old
2048 MiB limit, and is now 10240 MiB. A larger cache
cannot reuse a changed prefix without compatible state/checkpoints and does not
guarantee retention of every long prompt.
Scopes throttle host memory above MemoryHigh=8G and use a hard MemoryMax=9G
limit with MemorySwapMax=0. If memory cannot be reclaimed at the
9 GiB limit, the scope's OOM killer terminates the offending process. These
limits apply to host RAM, not VRAM, and only reach WSL processes: a Windows-native
server (Swift, Bolt) runs outside the scope's cgroup, so neither limit applies to it.
Runtime upgrades require independent model validation.

Swift15 also loads `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` for image input.
The 927,606,912-byte (885 MiB) projector comes from
[UkisAI's standard Swift GGUF repository](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF),
with an immutable revision and SHA-256 in `manifests/models.json`.
Run `python3 llama.py install swift15` to add it to an existing installation,
then `python3 llama.py verify swift15`. The launcher passes `--mmproj` and
refuses to start if the projector is missing. The runtime pin, 128K context,
KV cache settings and MTP depth remain unchanged. Image encoding needs additional
GPU memory beyond the projector weights; large-image capacity is workload dependent.
The API accepts OpenAI-style `image_url` content, including base64 data URLs.
`python3 llama.py smoke swift15` checks text and red/blue image recognition,
using locally generated PNGs, and cleans up its own trial scope.
Swift moved from b11393 CUDA 12.8 to b12001 CUDA 13.4 (commit `eb4e244be`) on
2026-10-30; context, KV, batch and MTP settings are unchanged. Both b12001 CUDA
12.8 and 13.4 passed `smoke swift15` (text, red/blue images), and CUDA 13.4 passed
`experiments/pi_smoke.py swift15 --port 1236` against the llama-swap backend
(discovery, streaming, thinking off/on, read-tool round trip, images). In capped
agentic PI runs (two per runtime, Omen Performance mode), CUDA 13.4 decoded
76 / 64 / 45 tok/s at <32K / 32–64K / 64–96K context versus 70 / 55 / 40 on
b11393, prefilled the same 9.8K-token first request 13% faster, and drew about
4 W less. See `reports/swift15/AGENTIC_POWER_MODES.md`. The WSL b11393, b12001
CUDA 12.8 and b12001 CUDA 13.4 runtimes stay installed.

Since 2026-10-30 Swift runs the same b12001 CUDA 13.4 build natively on Windows
(`"platform": "windows"`, runtime `win-b12001-cuda13.4`); flags are unchanged. In
fixed-prompt A/B runs it decoded 7–9% and prefilled 8–11% faster than WSL; in two
capped agentic runs decode was +3% / 0% / +5% by context range and the identical
first request prefilled 8% faster, at the same power. See "Windows-native profiles"
and `reports/swift15/WINDOWS_NATIVE.md`. The WSL model copy was removed. To roll
back to WSL, set `runtime` to `b12001-cuda13.4`, remove `platform`, run `python3
llama.py install swift15` (downloads the model to `models/swift15/` again), and
restart the server or llama-swap.

On 2026-11-04 Swift moved to Windows b12110 CUDA 13.4 (commit `cd959bae9`, llama.cpp
v0.9.0; runtime `win-b12110-cuda13.4`); flags are unchanged. Against the single b12001
run in `reports/swift15/B12001_BASELINE.md`, two capped agentic runs decoded 81 / 65 / 47 tok/s
by context range (+4% / 0% / -2%) and later-turn prefill rose 3–6%, all within normal
run-to-run variation. Two concurrent slots were not tested on this build. The new
K/V-rows attention path, on by default with unified KV, costs about 1–4% decode above
64K with MTP and gains 2–3% prefill; `LLAMA_KV_ROWS_DISABLE` turns it off. See
`reports/swift15/B12110_COMPARISON.md`. To roll back, stop llama-swap or the server
first, set `runtime` to `win-b12001-cuda13.4`, and restart; that runtime stays installed.

Validated on 2026-10-19 with the then-pinned b11393 runtime, 128K context and MTP
enabled: text chat, red/blue API image recognition, tool-call formatting and
a green image attached through `opencode run --standalone` all passed.
The small-image smoke used about 20 GiB of GPU memory; it does not establish
capacity for large images or a fully populated 128K prompt.

Bolt uses the official NVIDIA `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` target and
`Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf` drafter, pinned to immutable revisions in
`manifests/models.json`. Together they occupy 6,917,258,304 bytes on disk; the
model's “4GB” name is not a full-context VRAM guarantee. It shares Swift's Windows
b12110 CUDA 13.4 runtime, with no CUDA 12 Bolt variant. No driver/toolkit change
is performed. CUDA driver compatibility must be checked on other machines.
Full offload still leaves 3536 MiB of embedding tables in host RAM
(`per_layer_token_embd` and `token_embd`), which llama.cpp keeps out of VRAM; peak
resident memory is about 5.7 GiB with a typical prompt cache.
Sources: [target card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB),
[drafter card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash),
[runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b11808).

```bash
python3 llama.py install bolt35
python3 llama.py verify bolt35
python3 llama.py start bolt35 --dry-run
./start_bolt35.sh
# From another terminal:
python3 llama.py stop bolt35
python3 llama.py opencode add bolt35
# Select llama.cpp/nemotron-3.5-bolt-e4b-q4km-dflash in OpenCode.
```

Bolt exposes text input/output, a 65536-token context and 8192-token OpenCode
output limit, with no inferred reasoning-effort presets. Sampling follows the
card's temperature 1.0 and top-p 0.95; top-k/min-p filtering is disabled and
repetition penalty is 1.0. The runtime's embedded template handles
`chat_template_kwargs.enable_thinking` and `conversational_mode`; no custom
chat template is installed. Context includes prompt and generated tokens.
Batch sizing and f16 KV are provisional settings, not tuned performance defaults.
Overrides use `BOLT35_*`; `--spec none` omits the drafter for baseline decoding.

On 2026-11-05 Bolt moved from WSL b11808 to Windows b12110 CUDA 13.4 (commit
`cd959bae9`; runtime `win-b12110-cuda13.4`); flags are unchanged. Against six WSL
runs, eight agentic runs prefilled 15–24% faster and decoded 10–14% slower, for
about 3–8% more server time per run. Median CPU package power fell from about 39 W
to 28 W, and the user reported silent fans where WSL had driven them high. Resident
RAM is unchanged, but it now returns to the system at exit instead of staying in the
WSL VM. Platform and build changed together, so these are not separated. See
`reports/bolt35/BASELINE.md`. To roll back, stop llama-swap or the server, set
`runtime` to `b11808-cuda13.4`, remove `platform`, run `python3 llama.py install
bolt35` (downloads the model to `models/bolt35/` again), and restart. The WSL runtime
stays installed; the WSL model copy was removed.

Validated on 2026-10-25 with the then-pinned b11808 commit `4912ac24f`, bundled CUDA runtime
13040 and RTX 5090 Laptop GPU on driver 592.27: artifact verification, 64K
configured-context smoke with speculation off/on, thinking on/off, conversational
mode, streaming, structured tool-call round trips and the registered OpenCode
alias all passed. OpenCode returned exactly `hello world` in both modes.
CUDA 13.4 is the sole retained Bolt runtime; no CUDA 12 Bolt runtime was needed.

Three repetitions of short text workloads at temperature 0, with prompt reuse
disabled and a 512-token output cap, gave median generation speeds of 99.4 →
113.1 tokens/s for chat, 98.8 → 195.6 for code and 97.7 → 165.2 for JSON-schema
output (ordinary decoding → DFlash). DFlash accepted 3323 of 9563 proposed tokens
across the validation, workloads and client run. GPU memory samples were 5453 MiB
without speculation and 7325 MiB with DFlash. Greedy free-form outputs differed
between modes, including wording and response lengths; these measurements do
not establish token-identical decoding or a fixed-work end-to-end speedup.
Evidence is in ignored `logs/bolt35/comparison-20261025-132156-none/` and
`logs/bolt35/comparison-20261025-132252-draft-dflash/`. A Qwen smoke check also
passed after the shard-preflight change. These checks establish short-request
operation at the configured context capacity, not full-64K recall or quality.
An additional comparison confirmed exact arithmetic, capital-city and extraction
answers in both modes, and verified reasoning content appears only with thinking
enabled (`comparison-20261025-132430-none/` and
`comparison-20261025-132455-draft-dflash/` under `logs/bolt35/`). Swift was restored
after testing; its production scope, health, original alias and text chat passed.

When the GPU and port are free, run the smoke checks sequentially:

```bash
BOLT35_SPEC_TYPE=none python3 llama.py smoke bolt35
python3 llama.py smoke bolt35
python3 experiments/bolt_benchmark.py --repeats 3 --opencode
```

The comparison validates thinking on/off, conversational mode, streaming, a
structured tool-call round trip, and optionally the registered OpenCode alias.
It also checks exact arithmetic/extraction answers and verifies reasoning
content follows the thinking control. It compares three text workloads with speculation off/on, records API timing,
GPU memory samples and speculation metrics under ignored `logs/bolt35/`, and
stops only its uniquely named trial scopes. GPU samples are not continuous peak
measurements. It refuses to run when more than 2 GiB GPU memory is already used.
These synthetic workloads do not establish coding quality or long-context recall.

Start overrides: `--context`, `--batch`, `--ubatch`, `--draft-max`, `--spec`,
`--cache-k`, `--cache-v`, `--load-mode`, `--port`. CLI overrides take precedence
over the legacy `SWIFT_*` / `BOLT35_*` environment variables, which take precedence
over profiles.
`SWIFT_SPEC_TYPE` and `BOLT35_SPEC_TYPE` are supported.
`start_swift15_iq3s_128k.sh` remains a compatibility entrypoint.

## OpenCode registration

```bash
python3 llama.py opencode add bolt35 --dry-run
python3 llama.py opencode add bolt35
python3 llama.py opencode add swift15
opencode models
```

The command supports OpenCode v2 and adds the profile alias under
`providers["llama.cpp"].models` in
`~/.config/opencode/opencode.jsonc`. It does not start a server or select a default
model. All profiles may be listed; start the matching profile before using it.
Registration requires neither GPU access nor installed weights.

Swift advertises text and image input. To update an existing text-only entry, run
`python3 llama.py opencode add swift15 --update`. Include `--output-tokens N`
to preserve a custom output limit (otherwise Swift registration uses the 16000 default).
Restart an already running Swift server after installing the projector so image
input is available.

`--config PATH` targets another config. `--context`, `--port` and
`--output-tokens` match launch overrides and customize advertised limits; they
do not change the server. Output defaults to 16000 tokens for Swift and 8192 for other profiles.
Context and port also honor the existing profile environment overrides. Display names describe the
profile's default context; the numeric `limit.context` reflects overrides.

The provider's `baseURL` is read from `LLAMA_OPENCODE_BASE_URL` in the
git-ignored `.env` file at the repository root (template in the committed
`.env.sample`); when unset, registration targets the local server at
`http://127.0.0.1:<port>/v1`. The URL value is never committed.

JSONC comments, trailing commas and unrelated text are preserved. Identical
registration is a no-op. Missing fields are added, existing custom fields and
variants are retained, and conflicting managed fields require `--update`.
`--update` replaces only those conflicting fields. Changing the provider's
baseURL affects every model under `llama.cpp`, so use it only when all registered
models should use that endpoint. Use a separate config for independent endpoints.
The v2 entry uses `package`, provider `settings`, model `capabilities` and limits.
Swift retains its reasoning settings and v2 variant array. Bolt has text and tool-call capabilities.

Every actual edit prints a diff, saves a uniquely named backup beside the original
config, and replaces it atomically while preserving permissions. Invalid JSONC,
duplicate keys and detected concurrent edits cause refusal. `--dry-run` writes
nothing, including backups and directories. Only the selected config is edited;
there is no chezmoi integration. To undo an edit, restore the printed backup.

## PI registration

```bash
python3 llama.py pi add bolt35 --dry-run
python3 llama.py pi add bolt35
pi --list-models llama-local
# Start Bolt before requesting inference:
./start_bolt35.sh
# From another terminal:
pi --provider llama-local --model nemotron-3.5-bolt-e4b-q4km-dflash \
  --thinking off --no-session --no-tools --print \
  'Reply with exactly these two words and nothing else: hello world'
```

`pi add PROFILE` supports every profile and registers the existing API alias
under `providers["llama-local"].models` in `~/.pi/agent/models.json`, or
`$PI_CODING_AGENT_DIR/models.json` when that environment variable is set.
The separate provider does not use PI's built-in llama.cpp router or its model
management. Registration requires neither weights nor GPU access and does not
start a server, change default selections, or edit settings, authentication,
extensions or permission policies. All profiles can be listed together; start
the matching profile before requesting inference.

The provider's `baseUrl` is read from `LLAMA_PI_BASE_URL` in the git-ignored
`.env` file at the repository root (template in the committed `.env.sample`);
when unset, registration targets the local server at
`http://127.0.0.1:<port>/v1`. The URL value is never committed. A configured
URL applies to every model under the shared `llama-local` provider, so
register each profile against the endpoint that serves it.

`--config PATH`, `--dry-run`, `--update`, `--context`, `--port`, and
`--output-tokens` follow the OpenCode registration semantics above. Output defaults
to 16000 tokens for Swift and 8192 for other profiles; context and port honor the profile's environment overrides, with
CLI arguments taking precedence. These are advertised limits, so launch the
server with matching overrides. Swift advertises text and image input; other
profiles advertise text input. The OpenAI chat-completions adapter uses
`max_tokens` and a system role. Swift matches OpenCode's explicit effort presets:
PI's off/low/medium/xhigh send `reasoning_effort` none/low/medium/xhigh respectively.
Update an older binary-only registration with `python3 llama.py pi add swift15 --update`.
To default Swift to xhigh, set `modelThinkingLevels["llama-local/swift-1.5-qwen3.8-27b-iq3s-mtp"]`
to `"xhigh"` in PI's `settings.json`; registration does not edit settings.
Bolt uses `chat_template_kwargs.enable_thinking`:
`--thinking off` disables it and `--thinking high` enables it. Intermediate effort
levels, xhigh and max are not advertised for those binary-only profiles. Sampling is
left to the server.

PI uses strict JSON, with a model array merged by `id`. Unrelated providers,
models, custom fields and existing provider credentials are preserved; a new
provider gets a dummy local API key. Identical registration is a no-op. Conflicting
managed fields require `--update`; changing the shared provider's `baseUrl`
affects all models inheriting it. Invalid JSON, duplicate keys/model IDs, and
detected concurrent edits are refused. Other JSON text is preserved, and the
selected provider's model array is pretty printed with two-space indentation
when changed. Actual edits
print a diff and create a unique backup beside the config before atomic
replacement, preserving permissions. Restore the printed backup to undo an edit.
Dry runs write nothing, including backups and directories.

Test PI against an already running matching server:

```bash
python3 experiments/pi_smoke.py bolt35
# Match server overrides if applicable:
python3 experiments/pi_smoke.py bolt35 --port 1236 --context 32768
```

Behind llama-swap, point the smoke at the loaded backend (`--port 1236` for WSL
profiles, `--port 1257` for Windows profiles such as `swift15` and `bolt35`); the
coordinator's own `/health` returns plain `OK`, not the server's JSON. The port covers
the smoke's direct checks; with `LLAMA_PI_BASE_URL` set, PI's own requests use that URL.

The smoke runner never starts or stops a server. It uses a temporary working
directory and PI agent directory, registers through the actual CLI, verifies
discovery, and checks streamed text, thinking off/on, and a successful read-tool
round trip with a random file token. Swift also checks locally generated red/blue
image attachments. It disables extensions, MCP, skills, project context, themes,
startup network operations and session saving. Results and PI output go under
ignored `logs/<profile>/pi-smoke-*/`; each PI invocation has a 180-second timeout
by default (`--timeout N` overrides it). Run live checks sequentially. These
checks validate the client transport and built-in read tool, not the user's
permission extensions or interactive UI.

Validated on 2026-10-26 with PI 1.4.0 and Bolt's pinned b11808 CUDA 13.4 runtime
at the default 65536 context with DFlash enabled: the existing API smoke, PI model
discovery, streaming, exact text output, thinking off/on and read-tool round trip
passed. PI discovered generated entries for all five profiles; live PI inference
for the other four profiles remains untested. The isolated PI check used a
1024-token output allowance and short prompts, not a full-context workload.
Evidence: ignored `logs/bolt35/smoke-20261026-112817/` and
`logs/bolt35/pi-smoke-20261026T093141.550370Z/`. Bolt was also registered in the
real PI model config, preserving the existing providers; an actual-config PI
request returned exactly `hello world`. Its evidence is in ignored
`logs/bolt35/pi-registered-13ffbab20f884693ba7213b204fea2af/`.
Only uniquely owned Bolt trial scopes were started and stopped. No runtime pins
or profile defaults changed.

## On-demand Bolt and Swift with llama-swap

By default, the optional coordinator exposes the existing Bolt and Swift aliases at
`http://127.0.0.1:1235/v1`. Both run natively on Windows behind private port 1257;
port 1236 remains for WSL backends. Only one model runs at a time: switching stops
the current backend before starting the next.
Other profiles remain available for manual launching and are not listed by swap.
Strata can be selected explicitly as described in [its operating guide](docs/strata.md).
The coordinator does not load models at startup or unload them on an idle timer.

```bash
python3 llama.py swap install
python3 llama.py swap config
# After stopping your existing server yourself:
python3 llama.py swap start
python3 llama.py swap status
python3 llama.py swap stop
```

`install` downloads llama-swap v267 for Linux x86_64 from its
[release](https://github.com/mostlygeek/llama-swap/releases/tag/v267), pinned by
size and SHA-256 in `manifests/llama-swap.json`. Archives and the installed binary
stay in ignored `downloads/` and `runtimes/`. It does not install model artifacts;
install and verify `bolt35` and `swift15` separately using the existing commands.

`config` prints the resolved YAML using JSON syntax (valid YAML), without writing
anything or starting a server. `config` and `start` accept `--port`,
`--backend-port` and `--windows-backend-port`; the public port must differ from
both. For example:

```bash
python3 llama.py swap config --port 1235 --backend-port 1236 --windows-backend-port 1257
```

Each entry uses `llama.py start PROFILE --port 1236` (or `1257`) as `cmd` and
`llama.py stop PROFILE` as `cmdStop`. Existing runtime pins, model parameters,
Swift's projector and MTP, Bolt's DFlash, scope memory limits and server logs are
preserved. Profile environment overrides still apply, except the backend port and
scope are explicitly selected. The same Linux user must have access to the user
systemd manager and the installed artifacts.

`start` launches a transient user service, `llama-local-swap.service`, and prints
the coordinator log path. Startup is asynchronous; use `status`, `/health` and
the log to check readiness. The first inference request loads its requested model.
`/v1/models` lists both aliases even before they are loaded. Existing OpenCode and
PI entries for those aliases need no change when using the default public port.
Other previously registered models are not served by this endpoint.

The built-in dashboard is at `http://127.0.0.1:1235/ui/`, with model load/unload
controls, logs, a chat playground and performance statistics. Under WSL, the
coordinator adds `/usr/lib/wsl/lib` to its subprocess PATH when that directory
exists, allowing GPU monitoring to find `nvidia-smi`. Existing PATH entries keep
their precedence. Launcher changes take effect on the next coordinator start.

The existing `bolt35.scope` and `swift15.scope` names are reused. Startup refuses
occupied public/backend ports, either active profile scope, another coordinator
or pending cleanup. During coordinator ownership, ordinary manual start/stop of
these two production profiles refuses; use `swap stop` before manual operation.
Startup never stops a pre-existing server. Other-model launchers are not a global
GPU scheduler; do not start competing GPU workloads while using swap.

The service's `ExecStopPost` stops its reserved profile scopes after shutdown,
including coordinator failure. A per-instance ownership marker prevents cleanup
from an older instance stopping a newer one, and a lock prevents duplicate
coordinators. If cleanup fails, the marker remains and subsequent startup refuses;
`swap stop` retries cleanup. Use `journalctl --user -u llama-local-swap.service`
for service-level failures. The transient service is not enabled at login and does
not automatically restart. Roll back by running `swap stop` and using the existing
manual launch commands.

Generated config, coordinator logs and ownership state live under ignored
`logs/swap/`. Model server logs remain under `logs/bolt35/` and `logs/swift15/`.
Keep the active ownership marker and lock file intact while the coordinator runs.
Model switches reload weights and discard backend in-memory prompt/KV caches;
clients must allow for cold-load latency. Loading uses a 300-second readiness
timeout and unloading a 40-second graceful timeout. The pinned proxy scheduler
waits for active requests before swapping and may batch requests for the current
model to reduce switching. No additional request concurrency cap is imposed;
the model profiles retain their single-slot configuration.

After `swap install`, the normal unit suite also exercises the actual pinned
proxy with temporary CPU-only HTTP fixtures: two-model discovery, no preload,
one shared backend port, Bolt/Swift/Bolt routing, active-request draining,
tool/thinking/image payload forwarding and custom stop-command cleanup. This
also checks a temporary user service with fake backends: killing the supervisor
runs `ExecStopPost`, clears its ownership marker and leaves no fake backend.
It does not establish real-model correctness or real model-scope cleanup on the GPU.

When the GPU and both ports are free, run sequentially:

```bash
python3 llama.py smoke bolt35
python3 llama.py smoke swift15
python3 experiments/swap_smoke.py
```

The swap smoke refuses an active coordinator or more than 2 GiB of GPU use. It
uses unique trial scopes, checks both aliases before loading, Bolt/Swift/Bolt
switches, one active backend and scope memory limits, Swift red/blue image input,
and Bolt's existing thinking, streaming and tool-round-trip checks through the
proxy. It records request durations including loading (not isolated load times)
and cleans up only its trial scopes. Results live in ignored `logs/swap/smoke-*/`.
It does not exercise the production coordinator service or full-context quality.

Validated on 2026-10-26 with the pinned v267 proxy and unchanged profile runtimes:
both sequential per-profile smokes passed; the swap smoke passed discovery,
Bolt/Swift/Bolt switching, scope limits, Swift image input and Bolt API checks.
The three switch requests took 9.96, 7.88 and 5.74 seconds respectively, including
loading and inference; these are single observations, not a latency benchmark.
A separate production-service check passed the same model sequence, manual
start/stop reservation and duplicate-start refusal, normal shutdown, and forced
supervisor termination with real model-scope cleanup. All test-owned servers were
stopped afterward. Evidence is in ignored
`logs/swap/smoke-20261026T141323.517207Z/`,
`logs/swap/service-check-6925c44d52a5457abd3d554dd5f82cc7/`,
`logs/bolt35/smoke-20261026-161424/` and
`logs/swift15/smoke-20261026-161430/`. These checks do not establish full-context
quality, long-running scheduling fairness or recovery from a systemd manager crash.

## Artifacts

`manifests/` contains pinned URLs, hashes and known sizes. Interrupted downloads
resume from `.part`; validation happens before promotion to the final filename.
Existing files are verified on install. `verify` checks weights and downloaded
runtime archives without installing or starting a server. Launch checks artifact
presence; run verification after supplying/replacing files yourself.

Weights, versioned runtimes, downloads and logs are ignored by Git. Installation
diagnostics live under each runtime as `installation.json`.

Every server launch saves combined stdout/stderr to
`logs/<profile>/<UTC-timestamp>/server.log` and prints the path at startup.
Output remains visible in the terminal (or in the caller's redirected output),
including existing performance statistics. Logs are kept until manually deleted;
there is no automatic rotation, pruning, or separate launch metadata file.

## Adding a model

Add each model or runtime variant as a separate profile. Keep existing profiles'
runtime pins and measured defaults unchanged. Start from an existing profile as an example.
The steps below use `newmodel` as the new profile name; replace it throughout.

1. **Check compatibility and pin artifacts.** Confirm that the chosen runtime
   supports the architecture, tensor types, activation transforms, chat template
   and intended speculation mode. A GGUF loading successfully does not prove
   correct inference. Add `newmodel` to `manifests/models.json`, listing every
   required file with a path relative to `models/`, an immutable download URL,
   SHA-256 and byte size. Pin Hugging Face URLs to a commit revision. Include all
   shards and any separate draft model; preserve shard filenames and adjacency.
   Reuse a validated runtime or add a separate key in `manifests/runtimes.json`
   with release-specific URLs, hashes and sizes. Include compatible CUDA libraries
   when the binary archive omits them. The current installer expects tar archives
   with one top-level folder and exactly one CUDA 12 or CUDA 13 runtime library;
   other layouts or backends require explicit installer changes.

2. **Create `profiles/newmodel.json`.** Set the runtime key, model path relative
   to `models/`, unique API alias, scope equal to `newmodel`, port, context,
   batch/microbatch sizes, KV types, load mode and `fit_off`. Start with settings
   supported by the model and GPU; label unmeasured settings as provisional.
   Use `spec: "none"` unless speculation has been explicitly chosen. `draft_max`
   must remain positive even with speculation disabled. Embedded MTP uses
   `draft-mtp`; a separate DFlash model uses `draft-dflash` plus `draft`.
   Optional `draft_min` and `draft_p_min` apply only when speculation is enabled.
   If adding `sampling`, include `temperature`, `top_k` and `repeat_penalty`;
   optional supported fields are `top_p`, `min_p`, `presence_penalty` and
   `repeat_last_n`.

3. **Register the profile in code.** Add a `core.PROFILES` entry in `lib/core.py`,
   such as `'newmodel': ('NEWMODEL', 'none')`. The tuple defines its environment
   prefix and allowed speculation mode; `none` is always allowed. CLI choices
   come from this registry, so a JSON file alone is insufficient. Check
   `command()` and `preflight()` for model-specific assumptions. Split target
   models require every adjacent shard from their own profile manifest, even
   with speculation disabled; DFlash also requires its separate draft file.
   An optional executable `start_newmodel.sh`
   can follow `start_bolt35.sh` and forward arguments to `llama.py start newmodel`.

4. **Describe OpenCode capabilities.** Populate the profile's `opencode` object
   using an existing profile as the schema. Set a display name, `output_tokens`
   within the configured context, `tool_call`, and input/output `modalities`.
   Advertise only verified capabilities. Use `variants: {}` in the profile to
   generate `variants: []` in OpenCode and suppress inferred effort presets.
   Add `options` or named variants only after verifying the model/runtime accepts
   those settings. Vision support requires a compatible `mmproj` path in the
   profile and a pinned projector artifact in the model manifest; advertising an
   image modality alone does not load a projector.

5. **Validate and install.** Add tests for generated arguments, speculation
   enable/disable behavior and OpenCode registration preserving existing models.
   Run the commands in [Validation](#validation), including syntax checks for
   any new launcher. Then install, verify and inspect the resolved configuration:

   ```bash
   python3 llama.py install newmodel
   python3 llama.py verify newmodel
   python3 llama.py start newmodel --dry-run
   python3 llama.py opencode add newmodel --dry-run
   ```

6. **Run live checks and document the result.** When the GPU and port are free,
   run `python3 llama.py smoke newmodel`. It starts a unique trial scope and stops
   its own server. Adapt the smoke prompt or output allowance if the model needs
   different reasoning behavior. Check correct output and tool-call formatting;
   for speculation, confirm initialization and accepted draft tokens in the log.
   Register with `python3 llama.py opencode add newmodel`, start the profile, and
   test the actual client with the alias from the profile:

   ```bash
   opencode run --standalone --model llama.cpp/model-alias \
     'Reply with exactly these two words and nothing else: hello world'
   ```

   Stop your server with `python3 llama.py stop newmodel` when finished. Run live
   checks sequentially and never stop an unrelated server. Add the model filename,
   runtime source, launch/stop commands, alias, limits and any compatibility notes
   to this README. Speed measurements require a separate benchmark; a short smoke
   response does not establish performance or coding quality.

Before committing, inspect the diff and stage only intended code, profiles,
manifests, tests and documentation. Weights, runtimes, downloads, logs, legacy
archives and `HANDOFF.md` must remain outside Git.

## Probes and historical results

Stop any existing server before fresh-process probes:

```bash
python3 llama.py smoke swift15
python3 experiments/swift_prefill_benchmark.py --repeats 1
```

Swift's `experiments/swift_benchmark.py` probes an already running server.
`experiments/agentic_bench.py` brackets a real agentic PI run (launched separately,
for example as a T3 Code thread) with 1 s GPU sampling and a server-log summary;
it never starts or stops a server. `start` first sends a 1-token request for the
profile alias, so llama-swap loads the model before measurement begins (skip with
`--no-warm`). `start --fixture DIR` installs the read-only PI
policy `experiments/agentic_bench_pi_policy.json` into the trusted fixture,
`finish RUN_DIR [--max-context N]` summarizes decode/prefill by context range,
MTP acceptance, power, throttle reasons and temperature, and `cool` waits for the
GPU to cool. The prompt and procedure are in `reports/swift15/AGENTIC_PROMPT.md`.
`start` also runs `experiments/host_memory.py`, which samples host RAM and CPU load
into `host.csv`, from before the model loads until `finish`. Compare Windows
figures across platforms: a Windows server's scope cgroup reads near zero.
The prefill and DFlash workloads remain distinct. They share HTTP, GPU telemetry
and owned-scope cleanup. Fresh-process probes use unique trial scopes and always
stop their own servers, including on failure. Production servers use fixed
profile scopes. Results live under `logs/<profile>/`; historical local logs are
under `logs/<profile>/historical/`. Qwen experiment resumption uses new results
only, preventing accidental reuse of historical measurements.

### Swift benchmark runbook

Two methods; record which one a result came from.

- **Fixed-prompt A/B** (`experiments/swift_fixed_compare.py`): temperature 0,
  seed 42, filler prompts of 32/8K/32K tokens, exactly 128 output tokens, prompt
  cache off. Output, and so MTP acceptance, is identical between runtimes, so it
  isolates runtime/OS differences in a few minutes. Absolute tok/s overstate real
  sessions (repetitive text drafts too well).
- **Agentic** (`experiments/agentic_bench.py`): a real PI session with normal
  sampling and no fixed seed. Realistic, but acceptance varies 0.61–0.70 between
  runs and single runs differ by up to ~10%, so run each setup at least twice.

Keep the laptop setup fixed across compared runs (currently Omen Performance mode,
cooling pad, lid open, built-in screen off) and note it with the results.

Fixed-prompt A/B against the server on 1235, with llama-swap stopped (`python3
llama.py swap stop`) and the profile started directly (`python3 llama.py start
swift15`, stopped with `python3 llama.py stop swift15`):

```bash
python3 experiments/agentic_bench.py cool --below 50
python3 experiments/swift_fixed_compare.py LABEL logs/swift15/fixed-$(date -u +%Y%m%dT%H%M%SZ) 3
```

Agentic run, through llama-swap so PI keeps its model name:

1. `python3 llama.py swap start` (if not running), then
   `python3 experiments/agentic_bench.py cool --below 50`.
2. `python3 experiments/agentic_bench.py start --label LABEL --fixture
   /projects/agentic-bench-t3code`. It loads the model with a 1-token request,
   then records the server log offset and starts GPU sampling; it prints RUN_DIR.
3. Start a new T3 Code thread with the prompt from
   `reports/swift15/AGENTIC_PROMPT.md` as its first message: project
   `/projects/agentic-bench-t3code`, provider/instance `pi`, model
   `llama-local/swift-1.5-qwen3.8-27b-iq3s-mtp`, option `thinking` = `xhigh`,
   runtime mode full access. From an agent, `t3_thread_launch` with
   `modelSelection` `{"provider": "pi", "instanceId": "pi", "model":
   "llama-local/swift-1.5-qwen3.8-27b-iq3s-mtp", "options": [{"id": "thinking",
   "value": "xhigh"}]}`.
4. Watch the newest `logs/swift15/2*/server.log` (the `server_log` in
   `RUN_DIR/meta.json`) and interrupt the thread at the first request with
   `task.n_tokens` above 98304:

   ```bash
   tail -n0 -F "$LOG" | awk '/task\.n_tokens = [0-9]+/ { match($0,/task\.n_tokens = [0-9]+/);
     n=substr($0,RSTART+16,RLENGTH-16)+0; if (n>98304) { print "CAP " n; fflush() } }'
   ```

5. `python3 experiments/agentic_bench.py finish RUN_DIR --max-context 98304`, then
   compare decode by context range, first-request prefill, acceptance, busy
   samples, power-cap share, thermal reasons, power and temperature with earlier
   runs in `reports/swift15/`. Fan speed and CPU temperature cannot be read
   without Windows admin rights; record user-reported fan RPM with times. RAPL
   package power in the summary's `cpu` block is the available proxy for CPU heat.

For Bolt use `--profile bolt35`, `thinking` = `high` and `--max-context 57344`, as PI
compacts its 64K window at 57392 tokens. T3 reports `completed` even when Bolt's first
turn wrote a tool call as text and stopped, so check the PI session for tool calls and
all five file reads before counting a run. See `reports/bolt35/BASELINE.md`.

### Concurrent agent slot test

`experiments/agentic_slots.py` launches `--threads N` isolated PI CLI sessions
at a barrier against an existing idle backend. Each agent reads three generated
Python modules, traces a call chain and reports validation risks. The check
requires successful reads, correct per-agent evidence markers and a successful
final answer; it does not grade the prose review. Only the read tool is enabled.
This is a short concurrency/tool-use test, not a long-context quality benchmark.
PI configuration, fixtures, events, per-agent elapsed time/token usage, GPU
samples and the summary remain under ignored `logs/<profile>/agentic-slots-*`.
Sampled peaks can miss brief memory/concurrency spikes. Whole-GPU memory includes
other applications. Concurrent agents share GPU throughput.

The launcher accepts `--parallel N`; `--context` is **total** context, divided among slots when unified KV is off. Swift defaults to one slot with unified KV disabled;
other profiles also default to one slot. With `--kv-unified`, context is shared
instead of divided. `--no-kv-unified` restores the divided layout. Environment
overrides use `SWIFT_KV_UNIFIED=1` or `0` and `SWIFT_PARALLEL=N`.
For two 64K slots, when the GPU/port are free:

```bash
# First terminal (temporary settings; no profile changes):
LLAMA_SCOPE=swift15-trial-slots python3 llama.py start swift15 --port 1238 --parallel 2 --context 131072 --no-kv-unified
# Second terminal:
python3 experiments/agentic_slots.py swift15 --port 1238 --threads 2
# Stop only this trial, after the test:
LLAMA_SCOPE=swift15-trial-slots python3 llama.py stop swift15
```

Use the direct backend port exposing `/slots`, not llama-swap's public port.
The test refuses a busy backend or fewer slots than threads. `--allow-queue`
explicitly permits a queueing test and labels it accordingly. `--thinking xhigh`
enables reasoning; the default is off to keep this first check short. Each agent
has a `--timeout` (default 300 seconds), after which only its own process group
is terminated. The script never starts or stops the model server. Stop an active
server only with its owner's authorization and restore its original setup after
an authorized temporary test.

Validated on 2026-11-04 with the pinned Windows b12001 CUDA 13.4 runtime:
the default one-slot text/vision smoke passed, followed sequentially by two
65,536-token MTP slots running two PI agents with thinking off and xhigh.
Both agents passed in each run; sampling observed two slots processing together.
The xhigh run completed in 18.91 seconds with a sampled whole-GPU peak of
20,841 MiB. Each agent made three successful reads. These short fixtures do not
establish full-64K capacity under real agent histories. Evidence is under
`logs/swift15/agentic-slots-20261104T191814.547967Z/`. The original one-slot
128K Swift configuration was restored after testing.

The subsequent [two-thread T3 benchmark](reports/swift15/T3_TWO_SLOTS.md) uses
real T3 launches and the larger repository-investigation prompt. For concurrent
runs, calculate combined rates from the captured excerpt with
`python3 experiments/agentic_timing.py RUN_DIR`. It divides tokens by the union
of phase intervals across slots, counting overlapping elapsed time once. The
original `agentic_bench.py finish` summed-time rates remain per-sequence
aggregates and must not be presented as combined concurrent throughput.

The [unified-KV test with one T3 thread and small helper requests](reports/swift15/UNIFIED_KV_MANUAL.md)
completed successfully, including a helper while the main conversation exceeded
64K. That historical test used two slots sharing 128K total with unified KV enabled.
A helper consumed part of that same pool; the main conversation had no separate
128K reservation. The test used tiny helper prompts and does not establish the
impact of a second long prefill or fully populated concurrent contexts.

The historical [agentic baseline](reports/swift15/BASELINE.md) records throughput,
thermals and checkpoint measurements for the earlier two-slot unified profile.
The [long-session concurrency test](reports/swift15/T3_CP10_RAM16_MEMORY.md)
ended with context-size errors in both threads. Both threads failing is
unacceptable; Swift now uses one slot, unified KV disabled and twenty checkpoints.
The [one-slot repeat](reports/swift15/T3_CP20_SINGLE_SLOT_MEMORY.md) handled two
queued T3 clients to the prompt cap without later full re-prefills or context
errors. Peak server RAM was 17.95 GiB and Windows available RAM briefly fell to
806 MiB; the 16 GiB prompt-cache limit does not cap total server memory.
The historical [b12001 baseline](reports/swift15/B12001_BASELINE.md) is retained for
comparison. The historical [no-unified-KV single-slot baseline](reports/swift15/NO_KV_UNIFIED_BASELINE.md)
records the same measurements for the earlier A1 configuration.

### Windows-native profiles

A profile with `"platform": "windows"` runs a Windows `llama-server.exe` from WSL
through interop, still inside the profile's systemd scope with the same flags,
server log and smoke tests. llama-swap keeps running in WSL. Requirements: WSL2
with `networkingMode=mirrored` (Windows `127.0.0.1` reachable from WSL) and the
default `/mnt/c` automount.

- **Artifacts** live on NTFS under `C:\AI\llama-local` (override with
  `LLAMA_WINDOWS_ROOT=D:\path` in `.env`): `runtimes\`, `models\`, `downloads\`.
  `python3 llama.py install swift15` downloads and verifies the model and the two
  pinned release zips (llama.cpp and its bundled CUDA runtime DLLs) there, extracts
  the zips and records `installation.json`; `verify swift15` re-checks the hashes.
  Swift's and Bolt's models exist only there; their WSL copies under `models/` were
  removed.
- **Start/stop:** `llama.py start` converts model paths to Windows paths and runs
  the exe from its runtime directory. Stopping the scope (SIGTERM or SIGKILL)
  terminates the Windows process through interop; `llama.py stop` and llama-swap
  cleanup also wait until no `llama-server.exe` from this runtime directory is left,
  so the next model never loads into still-held VRAM. llama-swap moves on as soon as
  the scope exits, so under it `start` also waits for any server from this install's
  runtimes to exit first. Other Windows processes are never matched.
- **Ports:** under mirrored networking, a port released by one side stays
  unbindable for the other for about a minute. Windows backends therefore get their
  own llama-swap port (1257), and starts wait up to 150 s for the port to become
  bindable on the side that needs it (Windows for Windows profiles, Linux for
  llama-swap and WSL profiles). A direct `llama.py start swift15` right after
  llama-swap released 1235 waits out that delay.
- **Validated 2026-10-30:** install/verify, `smoke swift15`, `pi_smoke.py swift15
  --port 1257`, Swift→Bolt→Swift switching through llama-swap (11–14 s Swift, 9 s
  Bolt loads, VRAM released at each switch), `swap stop`, and SIGKILL of the
  llama-swap service with Swift loaded (cleanup removed the Windows server).

Reports under `reports/` are historical snapshots. Their original commands,
paths and statements about running servers describe the original experiments;
use this README for current operation. Token arrays are model/tokenizer specific;
do not reuse one model's token IDs to benchmark another model. Synthetic speed
probes are not coding-quality evaluations.

## Migration and rollback

Both original histories are retained in Git. Existing assets were moved without
redownloading. The old `/projects/inference/llama-swift15` and `llama-qwen36`
paths now link here, preserving launcher entrypoints. The original repositories
remain locally under ignored `legacy/` for rollback; `HANDOFF.md` is excluded
from this repository and was not used. Historical documentation may link to that
excluded file. This repository has no remote configured: publishing/renaming the
GitHub repository is a separate action.

To roll back, stop all new production/trial scopes, remove only the compatibility
symlinks, restore each legacy repository to its old directory, and return its
models, runtime, downloads and historical logs from this repository. Keep the
consolidated Git history until rollback has been verified.

## Validation

```bash
task agent:verify
```

This runs, from `taskfile.yml`:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q llama.py lib experiments tests
bash -n start_swift15_iq3s_128k.sh start_bolt35.sh
```

Before committing, `git diff --check` optionally flags whitespace errors and
conflict markers in unstaged changes.

Live smoke checks verify each profile's default context loading, health, model
alias, chat and scope limits, plus image recognition for profiles with `mmproj`.
Performance experiments are explicit commands and do not run during unit tests.
