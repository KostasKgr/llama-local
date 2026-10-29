# Local llama.cpp inference

One WSL/Linux repository for Swift 1.5 IQ3_S MTP, Qwen3.6 IQ3_S DFlash,
Ternary Bonsai 2 9B Q2_0 (without speculation), and Nemotron 3.5 Bolt E4B DFlash.
Each profile pins its own llama.cpp prebuilt runtime (CUDA 12.8 for existing
profiles, CUDA 13.4 for Bolt). No source
build, CUDA toolkit installation, or driver changes are required.
An optional Bonsai MTP profile uses the PrismML fork and an embedded prediction head.

Requirements: Python 3.12+, curl, working NVIDIA GPU passthrough, systemd user
manager and cgroup v2. Tested on an RTX 5090 Laptop GPU with 24 GB VRAM.

```bash
cd /projects/inference/llama-local
python3 llama.py install swift15
python3 llama.py install qwen36
python3 llama.py verify swift15
python3 llama.py start swift15 --dry-run
python3 llama.py start swift15
# From another terminal:
python3 llama.py status
python3 llama.py stop swift15
python3 llama.py start qwen36
# Bonsai shortcut (128K context, upstream runtime):
python3 llama.py install bonsai2
./start_bonsai2.sh
# Bonsai MTP (128K context, Prism runtime):
python3 llama.py install bonsai2mtp
./start_bonsai2_mtp.sh
```

Only one profile can use the default port at a time. Start refuses an occupied
port or active profile scope. Servers bind `127.0.0.1:1235`; API base is
`http://127.0.0.1:1235/v1`. Existing model aliases are preserved:
`swift-1.5-qwen3.8-27b-iq3s-mtp` and `qwen3.6-35b-a3b-iq3s-dflash`.
Select the alias for the running profile in your client. OpenCode and PI
registration are available through the explicit commands below.

| Profile | Runtime | Context | KV | Batch/microbatch | Speculation |
| --- | --- | ---: | --- | --- | --- |
| swift15 | b11393 | 131072 | q8_0/q8_0 | 2048/512 | embedded MTP, max 3 |
| qwen36 | b11433 | 131072 | f16/f16 | 4096/2048 | separate DFlash, max 8 |
| bonsai2 | b11669 | 131072 | f16/f16 | 2048/512 | none |
| bonsai2mtp | prism-b10777-e3de715 | 131072 | f16/f16 | 2048/512 | embedded MTP, max 2 |
| bolt35 | b11808 CUDA 13.4 | 65536 | f16/f16 | 2048/512 | separate DFlash, max 7 |

All profiles use full layer offload, Flash Attention, one slot, Direct I/O loading,
two checkpoints and 2 GiB RAM prompt cache. OpenCode thread naming can displace
the main prompt and cause cache thrashing when the RAM cache is too small.
Scopes throttle host memory above MemoryHigh=8G and use a hard MemoryMax=9G
limit with MemorySwapMax=0. If memory cannot be reclaimed at the
9 GiB limit, the scope's OOM killer terminates the offending process. These
limits apply to host RAM, not VRAM.
Qwen uses `--fit off`. Runtime upgrades require independent model validation.

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
Validated on 2026-10-19 with the pinned b11393 runtime, 128K context and MTP
enabled: text chat, red/blue API image recognition, tool-call formatting and
a green image attached through `opencode run --standalone` all passed.
The small-image smoke used about 20 GiB of GPU memory; it does not establish
capacity for large images or a fully populated 128K prompt.

Bonsai uses upstream b11669 (commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`),
which supports the `prism.hadamard` activation transforms required by Bonsai 2.
The older pinned runtimes recognize Q2_0 but lack these transforms. Only the
text model `Ternary-Bonsai-2-9B-Q2_0.gguf` is installed, without a vision projector
or draft model. Alias: `ternary-bonsai-2-9b-q2_0`. Thinking uses the model card's
temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0, repetition
penalty 1.0 and a 64-token penalty window. OpenCode advertises 131072 context and
8192 output tokens, with `variants: []`; no reasoning-effort presets are inferred.
These are initial settings for evaluation, not measured performance defaults.
Bonsai uses `--fit off` and the `BONSAI_*` environment prefix. Stop it with
`python3 llama.py stop bonsai2`.

The separate `bonsai2mtp` profile pins PrismML's `prism-b10777-e3de715`
(commit `e3de715222b2667c54a267441159a5729f11155e`), the release pinned by
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo).
Install downloads `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` from
[the model repository](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf),
with the exact revision, size and SHA-256 in `manifests/models.json`.
The single GGUF includes the language model and MTP head; no separate draft model
or vision projector is needed. The Prism binary archive is paired with the pinned
upstream CUDA 12.8 library archive because Prism's Linux archive omits CUDA libraries.
It uses the same context, f16 KV cache and sampling as `bonsai2`, with draft depth 2,
minimum draft length 0 and minimum draft probability 0, following the model card.
OpenCode alias: `ternary-bonsai-2-9b-pq2_0-mtp` (128K context, 8192 output tokens).
Use `./start_bonsai2_mtp.sh`, then select that alias in OpenCode; stop with
`python3 llama.py stop bonsai2mtp`. Overrides use `BONSAI_MTP_*`; `--spec none`
disables MTP for comparisons. The upstream `bonsai2` profile remains available.

Bolt uses the official NVIDIA `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` target and
`Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf` drafter, pinned to immutable revisions in
`manifests/models.json`. Together they occupy 6,917,258,304 bytes on disk; the
model's “4GB” name is not a full-context VRAM guarantee. It uses a separate
upstream b11808 CUDA 13.4 runtime and matching libraries, with no CUDA 12 Bolt
variant. Existing profiles retain their runtime pins. No driver/toolkit change
is performed. CUDA driver compatibility must be checked on other machines.
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

Validated on 2026-10-25 with b11808 commit `4912ac24f`, bundled CUDA runtime
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
over the legacy `SWIFT_*` / `QWEN_*` environment variables, which take precedence
over profiles.
`SWIFT_SPEC_TYPE` and `QWEN_SPEC_TYPE` are supported.
`start_swift15_iq3s_128k.sh` and `start_qwen36.sh` remain compatibility entrypoints.

## OpenCode registration

```bash
python3 llama.py opencode add qwen36 --dry-run
python3 llama.py opencode add qwen36
python3 llama.py opencode add swift15
python3 llama.py opencode add bonsai2
python3 llama.py opencode add bonsai2mtp
opencode models
```

The command supports OpenCode v2 and adds the profile alias under
`providers["llama.cpp"].models` in
`~/.config/opencode/opencode.jsonc`. It does not start a server or select a default
model. All profiles may be listed; start the matching profile before using it.
Registration requires neither GPU access nor installed weights.

Swift advertises text and image input. To update an existing text-only entry, run
`python3 llama.py opencode add swift15 --update`. Include `--output-tokens N`
to preserve a custom output limit (otherwise registration uses the 8192 default).
Restart an already running Swift server after installing the projector so image
input is available.

`--config PATH` targets another config. `--context`, `--port` and
`--output-tokens` match launch overrides and customize advertised limits; they
do not change the server. Output defaults to 8192 tokens for all profiles.
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
Swift retains its reasoning settings and v2 variant array. Qwen and Bonsai have text
and tool-call capabilities, without unvalidated reasoning-effort settings.

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
to 8192 tokens; context and port honor the profile's environment overrides, with
CLI arguments taking precedence. These are advertised limits, so launch the
server with matching overrides. Swift advertises text and image input; other
profiles advertise text input. The OpenAI chat-completions adapter uses
`max_tokens` and a system role. Swift matches OpenCode's explicit effort presets:
PI's off/low/medium/xhigh send `reasoning_effort` none/low/medium/xhigh respectively.
Update an older binary-only registration with `python3 llama.py pi add swift15 --update`.
To default Swift to xhigh, set `modelThinkingLevels["llama-local/swift-1.5-qwen3.8-27b-iq3s-mtp"]`
to `"xhigh"` in PI's `settings.json`; registration does not edit settings.
Other profiles use `chat_template_kwargs.enable_thinking`:
`--thinking off` disables it and `--thinking high` enables it. Intermediate effort
levels, xhigh and max are not advertised for those binary-only profiles. Sampling is
left to the server.

PI uses strict JSON, with a model array merged by `id`. Unrelated providers,
models, custom fields and existing provider credentials are preserved; a new
provider gets a dummy local API key. Identical registration is a no-op. Conflicting
managed fields require `--update`; changing the shared provider's `baseUrl`
affects all models inheriting it. Invalid JSON, duplicate keys/model IDs, and
detected concurrent edits are refused. Other JSON text is preserved, although
the selected provider's model array is reformatted when changed. Actual edits
print a diff and create a unique backup beside the config before atomic
replacement, preserving permissions. Restore the printed backup to undo an edit.
Dry runs write nothing, including backups and directories.

Test PI against an already running matching server:

```bash
python3 experiments/pi_smoke.py bolt35
# Match server overrides if applicable:
python3 experiments/pi_smoke.py bolt35 --port 1236 --context 32768
```

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

The optional coordinator exposes exactly the existing Bolt and Swift aliases at
`http://127.0.0.1:1235/v1`. Both backends share private port 1236. Only one model
runs at a time: switching stops the current backend before starting the next.
Other profiles remain available for manual launching and are not listed by swap.
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
anything or starting a server. `config` and `start` accept `--port` and
`--backend-port`; they must differ. For example:

```bash
python3 llama.py swap config --port 1235 --backend-port 1236
```

Each entry uses `llama.py start PROFILE --port 1236` as `cmd` and
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
diagnostics live under each runtime as `installation.json`. Qwen's two shards
must remain adjacent with their original filenames. The first is passed to `-m`.

Every server launch saves combined stdout/stderr to
`logs/<profile>/<UTC-timestamp>/server.log` and prints the path at startup.
Output remains visible in the terminal (or in the caller's redirected output),
including existing performance statistics. Logs are kept until manually deleted;
there is no automatic rotation, pruning, or separate launch metadata file.

## Adding a model

Add each model or runtime variant as a separate profile. Keep existing profiles'
runtime pins and measured defaults unchanged. Start from `profiles/bonsai2.json`
for ordinary decoding or `profiles/bonsai2mtp.json` for an embedded MTP head.
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
   can follow `start_bonsai2.sh` and forward arguments to `llama.py start newmodel`.

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
python3 llama.py smoke qwen36
python3 llama.py smoke bonsai2
python3 llama.py smoke bonsai2mtp
python3 experiments/swift_prefill_benchmark.py --repeats 1
python3 experiments/qwen_benchmark.py --label screening --smoke
python3 experiments/qwen_run_experiment.py
```

Swift's `experiments/swift_benchmark.py` probes an already running server.
The prefill and DFlash workloads remain distinct. They share HTTP, GPU telemetry
and owned-scope cleanup. Fresh-process probes use unique trial scopes and always
stop their own servers, including on failure. Production servers use fixed
profile scopes. Results live under `logs/<profile>/`; historical local logs are
under `logs/<profile>/historical/`. Qwen experiment resumption uses new results
only, preventing accidental reuse of historical measurements.

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
python3 -m unittest discover -s tests -v
python3 -m compileall -q llama.py lib experiments tests
bash -n start_swift15_iq3s_128k.sh start_qwen36.sh start_bonsai2.sh start_bonsai2_mtp.sh start_bolt35.sh
git diff --check
```

Live smoke checks verify each profile's default context loading, health, model
alias, chat and scope limits, plus image recognition for profiles with `mmproj`.
Performance experiments are explicit commands and do not run during unit tests.
