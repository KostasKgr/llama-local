# Qwen3.6-35B-A3B IQ3_S + DFlash

Isolated upstream llama.cpp b11433 CUDA 12.8 experiment for the RTX 5090
Laptop GPU under Linux/WSL2. Uses prebuilt binaries only. Existing Swift and
NInfer projects are independent and are not changed by this setup.

## Installation

```bash
cd /projects/inference/llama-qwen36
python3 download.py
python3 install_runtime.py
```

Downloads are pinned to publisher revisions and all five files are checked
against SHA256. `ARTIFACTS.json` preserves URLs, hashes, and sizes;
`RUNTIME.json` records the binary version, detected GPU, and packaged CUDA
runtime version. The runtime installer extracts both archives to the same
directory. No compilation, CUDA toolkit installation, or driver changes.

Main files:

- `models/qwen36-35b-a3b/IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00001-of-00002.gguf`
- `models/qwen36-35b-a3b/IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00002-of-00002.gguf`
- `models/qwen36-35b-a3b/dflash-Qwen3.6-35B-A3B-Q8_0.gguf`

The first shard is passed as `-m`; llama.cpp finds the second beside it.

Sources: [release](https://github.com/ggml-org/llama.cpp/releases/tag/b11433),
[main model](https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF),
[DFlash draft](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF).

## Start and stop

```bash
./start_qwen36.sh
# Stop from a second terminal, or press Ctrl+C in the launch terminal:
systemctl --user stop qwen36.scope
```

The measured default is **131072 context, microbatch 2048, DFlash max 8,
batch 4096, F16 KV**. The server binds `127.0.0.1:1235`, uses one slot,
Flash Attention, and full layer offload for both models. Configuration overrides:
`QWEN_CONTEXT`, `QWEN_UBATCH`, `QWEN_DRAFT_MAX`, `QWEN_SPEC_TYPE` (`none` or
`draft-dflash`), `QWEN_CACHE_K`, `QWEN_CACHE_V`, `QWEN_LOAD_MODE`.
The explicit `--fit off` prevents automatic fitting from silently changing
the tested setup. Initial KV format is f16/f16, including the draft.

The launcher uses systemd user scope `qwen36.scope` with MemoryHigh=6 GiB,
MemoryMax=8 GiB, and MemorySwapMax=0. Direct I/O avoids filling host page
cache during model loading where supported. Context checkpoints are capped
at two per slot; RAM prompt cache is capped at 512 MiB. These limits contain
host-memory pressure to this server; they do not cap GPU VRAM. The fixed
scope name prevents two launches through this launcher at once.

```bash
systemctl --user show qwen36.scope \
  -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax -p MemorySwapMax
```

## Measurements

```bash
python3 benchmark.py --label baseline-32k --spec none --smoke
python3 benchmark.py --label dflash-32k-ub1024-d15
python3 benchmark.py --label dflash-32k-ub512-d15 --ubatch 512
# Run the full sequential experiment, resuming saved results:
python3 run_experiment.py
```

Each trial starts a fresh server, measures loading, samples whole-device
VRAM from startup through generation approximately every 200 ms plus query
overhead, and stops only its dedicated scope afterward. Port conflicts
cause refusal before starting. Trials preserve server logs, full responses,
metrics, GPU traces, scope limits, and result JSON under `logs/<label>/`.

The tuning workload is an identical frozen 8192-token synthetic Python
prompt, greedy sampling, no cached prompt, and 2048 generated tokens with
EOS ignored to ensure adequate measurement length. It measures speed and
speculative acceptance, not response quality. Prefill and decode timings
are reported separately; acceptance is accepted/proposed tokens from server
metrics. Peak VRAM is sampled whole-device usage, not allocator telemetry.

Tune microbatch 512/1024/2048/4096 first, then draft maximum 8/12/15/20 at
the selected microbatch. Record runtime clamping of draft size explicitly.
Only after choosing the best stable 32K configuration, test 131072 context
with f16 KV. Report any allocation failure before considering quantized KV.

See [REPORT.md](REPORT.md) for measured outcomes and residency qualifications.
