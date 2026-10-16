# Local llama.cpp inference

One WSL/Linux repository for Swift 1.5 IQ3_S MTP, Qwen3.6 IQ3_S DFlash,
LFM2.5 24B A2B NVFP4, and Ternary Bonsai 2 9B Q2_0 (the latter two without speculation).
Each profile pins its own tested llama.cpp CUDA 12.8 prebuilt runtime. No source
build, CUDA toolkit installation, or driver changes are required.
An optional Bonsai MTP profile uses the PrismML fork and an embedded prediction head.

Requirements: Python 3.12+, curl, working NVIDIA GPU passthrough, systemd user
manager and cgroup v2. Tested on an RTX 5090 Laptop GPU with 24 GB VRAM.

```bash
cd /projects/inference/llama-local
python3 llama.py install swift15
python3 llama.py install qwen36
python3 llama.py install lfm25
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
Select the alias for the running profile in your client. OpenCode registration
is available through the explicit command below.

| Profile | Runtime | Context | KV | Batch/microbatch | Speculation |
| --- | --- | ---: | --- | --- | --- |
| swift15 | b11393 | 131072 | q8_0/q8_0 | 2048/512 | embedded MTP, max 3 |
| qwen36 | b11433 | 131072 | f16/f16 | 4096/2048 | separate DFlash, max 8 |
| lfm25 | b11433 | 131072 | f16/f16 | 4096/1024 | none; no embedded MTP |
| bonsai2 | b11669 | 131072 | f16/f16 | 2048/512 | none |
| bonsai2mtp | prism-b10777-e3de715 | 131072 | f16/f16 | 2048/512 | embedded MTP, max 2 |

All profiles use full layer offload, Flash Attention, one slot, Direct I/O loading,
two checkpoints and 2 GiB RAM prompt cache. OpenCode thread naming can displace
the main prompt and cause cache thrashing when the RAM cache is too small.
Scopes use MemoryHigh=6G,
MemoryMax=8G and MemorySwapMax=0; these limits apply to host RAM, not VRAM.
Qwen and LFM use `--fit off`. Runtime upgrades require independent model validation.
LFM uses alias `lfm2.5-24b-a2b-nvfp4` and a pinned WhiskyAKM GGUF conversion.
Its native context limit and launch default are 131072. The model always reasons;
no reasoning-effort variants are advertised. The profile explicitly sets the
model page's temperature 0.1, top-k 50 and repetition penalty 1.05.
NVFP4 activation precision follows runtime/model defaults; no forced W4A4 override.

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

Start overrides: `--context`, `--batch`, `--ubatch`, `--draft-max`, `--spec`,
`--cache-k`, `--cache-v`, `--load-mode`, `--port`. CLI overrides take precedence
over the legacy `SWIFT_*` / `QWEN_*` environment variables, which take precedence
over profiles. LFM uses the `LFM_*` prefix, including `LFM_SPEC_TYPE=none`.
`SWIFT_SPEC_TYPE` and `QWEN_SPEC_TYPE` are supported.
`start_swift15_iq3s_128k.sh` and `start_qwen36.sh` remain compatibility entrypoints.

## OpenCode registration

```bash
python3 llama.py opencode add qwen36 --dry-run
python3 llama.py opencode add qwen36
python3 llama.py opencode add swift15
python3 llama.py opencode add lfm25
python3 llama.py opencode add bonsai2
python3 llama.py opencode add bonsai2mtp
opencode models
```

The command supports OpenCode v2 and adds the profile alias under
`providers["llama.cpp"].models` in
`~/.config/opencode/opencode.jsonc`. It does not start a server or select a default
model. All profiles may be listed; start the matching profile before using it.
Registration requires neither GPU access nor installed weights.

`--config PATH` targets another config. `--context`, `--port` and
`--output-tokens` match launch overrides and customize advertised limits; they
do not change the server. Output defaults to 8192 tokens for Swift/Qwen and
15360 (15K) for LFM and 8192 for Bonsai. Context and port also
honor the existing profile environment overrides. Display names describe the
profile's default context; the numeric `limit.context` reflects overrides.

JSONC comments, trailing commas and unrelated text are preserved. Identical
registration is a no-op. Missing fields are added, existing custom fields and
variants are retained, and conflicting managed fields require `--update`.
`--update` replaces only those conflicting fields. Changing the provider's
baseURL affects every model under `llama.cpp`, so use it only when all registered
models should use that endpoint. Use a separate config for independent endpoints.
The v2 entry uses `package`, provider `settings`, model `capabilities` and limits.
Swift retains its reasoning settings and v2 variant array. Qwen and LFM have text
and tool-call capabilities, without unvalidated reasoning-effort settings.

Every actual edit prints a diff, saves a uniquely named backup beside the original
config, and replaces it atomically while preserving permissions. Invalid JSONC,
duplicate keys and detected concurrent edits cause refusal. `--dry-run` writes
nothing, including backups and directories. Only the selected config is edited;
there is no chezmoi integration. To undo an edit, restore the printed backup.

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

## Probes and historical results

Stop any existing server before fresh-process probes:

```bash
python3 llama.py smoke swift15
python3 llama.py smoke qwen36
python3 llama.py smoke lfm25
python3 llama.py smoke bonsai2
python3 llama.py smoke bonsai2mtp
python3 experiments/lfm_benchmark.py
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
bash -n start_swift15_iq3s_128k.sh start_qwen36.sh start_bonsai2.sh start_bonsai2_mtp.sh
git diff --check
```

Live smoke checks verify each profile's default context loading, health, model
alias, chat and scope limits.
Performance experiments are explicit commands and do not run during unit tests.

LFM's benchmark runs sequential fresh processes with a fixed 128K (131072)
server context and 36K (36864) / 96K (98304) input prompts, three repeats each.
Each synthetic Python prompt is followed by 512 generated tokens with EOS ignored,
using the profile's recommended sampling settings and seed 42.
It tokenizes through LFM itself and checks complete, uncached prompt evaluation.
Results include prefill/decode rates, GPU telemetry and resolved commands under
`logs/lfm25/benchmark-*/`. Override `--prompt-sizes`, `--repeats` or
`--output-tokens` explicitly. These are speed probes, not coding-quality tests.
Run another context manually with `python3 llama.py start lfm25 --context 36864`.
