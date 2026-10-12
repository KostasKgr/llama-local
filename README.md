# Local llama.cpp inference

One WSL/Linux repository for Swift 1.5 IQ3_S MTP and Qwen3.6 IQ3_S DFlash.
Each profile pins its own tested llama.cpp CUDA 12.8 prebuilt runtime. No source
build, CUDA toolkit installation, or driver changes are required.

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

Both retain full layer offload, Flash Attention, one slot, Direct I/O loading,
two checkpoints and 512 MiB RAM prompt cache. Scopes use MemoryHigh=6G,
MemoryMax=8G and MemorySwapMax=0; these limits apply to host RAM, not VRAM.
Qwen retains `--fit off`. Runtime upgrades require independent model validation.

Start overrides: `--context`, `--batch`, `--ubatch`, `--draft-max`, `--spec`,
`--cache-k`, `--cache-v`, `--load-mode`, `--port`. CLI overrides take precedence
over the legacy `SWIFT_*` / `QWEN_*` environment variables, which take precedence
over profiles. `SWIFT_SPEC_TYPE` and `QWEN_SPEC_TYPE` are supported.
`start_swift15_iq3s_128k.sh` and `start_qwen36.sh` remain compatibility entrypoints.

## OpenCode registration

```bash
python3 llama.py opencode add qwen36 --dry-run
python3 llama.py opencode add qwen36
python3 llama.py opencode add swift15
opencode models llama.cpp
```

The command adds the profile alias under `provider["llama.cpp"].models` in
`~/.config/opencode/opencode.jsonc`. It does not start a server or select a default
model. Both models may be listed; start the matching profile before using it.
Registration requires neither GPU access nor installed weights.

`--config PATH` targets another config. `--context`, `--port` and
`--output-tokens` match launch overrides and customize advertised limits; they
do not change the server. Output defaults to 8192 tokens. Context and port also
honor the existing profile environment overrides. Display names describe the
profile's default context; the numeric `limit.context` reflects overrides.

JSONC comments, trailing commas and unrelated text are preserved. Identical
registration is a no-op. Missing fields are added, existing custom fields and
variants are retained, and conflicting managed fields require `--update`.
`--update` replaces only those conflicting fields. Changing the provider's
baseURL affects every model under `llama.cpp`, so use it only when all registered
models should use that endpoint. Use a separate config for independent endpoints.
Swift retains its current reasoning options/variants. Qwen has text, reasoning,
temperature and tool-call metadata but no unvalidated reasoning-effort options.

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

## Probes and historical results

Stop any existing server before fresh-process probes:

```bash
python3 llama.py smoke swift15
python3 llama.py smoke qwen36
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
bash -n start_swift15_iq3s_128k.sh start_qwen36.sh
git diff --check
```

Live smoke checks verify 128K loading, health, model alias, chat and scope limits.
Performance experiments are explicit commands and do not run during unit tests.
