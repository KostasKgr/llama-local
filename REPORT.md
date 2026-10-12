# Qwen3.6-35B-A3B IQ3_S + DFlash report — 2026-10-06

Upstream prebuilt llama.cpp b11433 CUDA 12.8 successfully runs the requested
IQ3_S split model and Q8_0 DFlash draft at **131072 context with F16 KV**.
The selected launcher defaults are **batch 4096, microbatch 2048, draft max 8,
one slot, Flash Attention enabled**. The final server is running on
`http://127.0.0.1:1235`, alias `qwen3.6-35b-a3b-iq3s-dflash`.

The 128K allocation survived an actual **122880-token prompt plus 2048 generated
tokens**, without truncation, OOM, or server failure. Peak whole-device memory
was **22902 MiB**, with minimum sampled free VRAM **1236 MiB**. This fits but
leaves about 1.2 GiB of headroom; no quantized-KV fallback was needed or tested.

## Requested tuning matrix

All rows below use the identical 8192-token cold prompt and 2048 generated
output tokens. Throughput comes from llama.cpp server timings; acceptance
is accepted/proposed draft tokens. All trials completed and remained healthy.

| Context | Microbatch | DFlash max | Prefill tok/s | Decode tok/s | Acceptance | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 512 | 15 | 1833.9 | 139.0 | 21.5% | 17249 |
| 32K | 1024 | 15 | 2300.5 | 148.5 | 24.1% | 18417 |
| 32K | 2048 | 15 | 2416.4 | 157.7 | 25.2% | 20757 |
| 32K | 4096 | 15 | 1441.3 | 139.8 | 23.6% | 23684 |
| 32K | 2048 | 8 | 2420.2 | 242.3 | 77.8% | 20214 |
| 32K | 2048 | 12 | 2293.1 | 147.6 | 27.8% | 20466 |
| 32K | 2048 | 20 → 15 | 2425.8 | 155.8 | 25.2% | 20968 |
| 128K | 2048 | 8 | 2276.1 | 241.7 | 77.8% | 22902 |

The 2048/15 result supplies both the microbatch-sweep row and the draft-15
comparison; it was reused rather than measured twice. Requested draft max 20
was accepted by the CLI but explicitly **clamped to 15** because this draft
is trained with `dflash.block_size=16`. It is not an independent 20-token test.

Microbatch 2048 had the highest measured prefill with no decode regression
against the initial 1024/15 trial. Microbatch 4096 nearly filled VRAM and was
slower on both measures. At the chosen microbatch, draft max 8 had the highest
measured decode throughput. Selection uses one trial per configuration; it
is the fastest observed setting on this workload, not a statistically
established optimum across production prompts.

**Output anomaly:** 2048/15 and requested 2048/20 each produced 500 Unicode
replacement characters near the end of the output. Their timings are
preserved, but server health alone does not establish clean generation.
The selected 2048/8 setting had zero replacement characters in its 32K,
128K, 28K-prompt, and 120K-prompt responses. The final ordinary chat request
also produced a coherent Python function and ended normally at EOS. No
quality assessment or root-cause diagnosis of the longer-draft anomaly is
claimed. Forced output length, differing verification batch shapes, and
greedy output divergence limit conclusions about acceptance versus draft size.

## Longer prompts

| Context | Prompt tokens | Generated tokens | Prefill tok/s | Decode tok/s | Acceptance | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 28672 | 2048 | 2593.1 | 139.6 | 36.4% | 20214 |
| 128K | 122880 | 2048 | 1979.8 | 145.1 | 49.7% | 22902 |

Both use microbatch 2048, draft max 8, and F16 main/draft KV. The 28K prompt
leaves room for the full output inside the 32K context. The 120K prompt plus
output exercises 124928 tokens inside the 131072-token context; there was
zero prompt-cache reuse and no context shifting or prompt truncation.

## Exact artifacts and machine

- llama.cpp **b11433**, commit **50569eb87**, version 0.6.0-dev, GNU 13.3.0 Linux x86_64.
- Release archives: `llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz` and `cudart-llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz`.
- Server SHA256: `a2bd95e075b90541548148538e02436d521da7306e56f2a305df89c4d5e52a7d`.
- Main repo: `AesSedai/Qwen3.6-35B-A3B-GGUF`, revision `85b84632d37b84b23c7fb7ee25e47c32318c585f`.
- Main files: `IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00001-of-00002.gguf` and `IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00002-of-00002.gguf`.
- The first shard is passed to `-m`; both retain their original numbered names.
- Draft repo: `ggml-org/Qwen3.6-35B-A3B-GGUF`, revision `baec3ebee244827cda0f4557eafa8b28f7545fa6`.
- Draft file: `dflash-Qwen3.6-35B-A3B-Q8_0.gguf` (421060800 bytes).
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, **24463 MiB** reported total VRAM.
- NVIDIA driver: **592.27**; nvidia-smi reports CUDA compatibility **13.1**.
- Packaged CUDA runtime: **12.8** (`cudaRuntimeGetVersion=12080`). The runtime version was queried directly from packaged libcudart; llama.cpp startup reports the CUDA backend and native architecture 1200, without a separate runtime-version string.
- Live process maps confirm libcudart, libcublas, libcublasLt, and libggml-cuda are loaded from this project’s private runtime directory.
- **42/42 main layers and 7/7 draft layers offloaded to CUDA0**. Main and draft KV buffers are on CUDA0; Flash Attention is enabled for both.
- Main and draft KV: **f16/f16 throughout**. No alternate quant, MTP, NInfer, compilation, toolkit installation, or driver change.

`ARTIFACTS.json` contains every URL, SHA256, and exact size; all five downloads
passed full verification. `RUNTIME.json` contains packaged runtime diagnostics.

**Residency qualification:** upstream always keeps its input layer on CPU
([b11433 source](https://github.com/ggml-org/llama.cpp/blob/b11433/src/llama-model.cpp#L1605)).
The loader reports a **397.85 MiB CUDA_Host model buffer** for host/input
tensors despite full transformer/output-layer offload. Thus all model layers
used for transformer computation are GPU-offloaded, but claiming every byte
of the model is GPU-resident would be inaccurate. This is upstream behavior,
also present in the existing Swift setup; no transformer layers were
deliberately offloaded to CPU. Offload logs and nvidia-smi do not independently
prove WSL/Windows driver-level physical residency of every allocation.

## Baseline and memory allocations

At 32768 context without DFlash, readiness took **5.03 seconds**, sampled
VRAM after loading was **14761 MiB**, and the short chat answered `4`.
All 42/42 layers were offloaded, Flash Attention was enabled, and the main
F16 KV buffer was 640 MiB. No meaningful baseline throughput benchmark was
performed. DFlash at the initial 1024/15 setting reached readiness in about
5.02 seconds and used 18271 MiB after loading. Readiness includes model loading,
context allocation, warmup, and up to one second of health-poll granularity.

Final unquantized 128K, 2048/8 logged CUDA allocations:

| Allocation | MiB |
| --- | ---: |
| Main model | 12550.85 |
| Main F16 KV | 2560.00 |
| Main recurrent state | 565.31 |
| Main compute, after prompt graph reservation | 1056.08 |
| DFlash model | 391.12 |
| DFlash full-attention F16 KV | 512.00 |
| DFlash sliding-window F16 KV | 120.00 |
| DFlash final reserved compute | 3896.95 |

These logged buffers are not an exhaustive CUDA allocator total and some
reservation messages replace earlier sizes. Use the sampled peak of 22902 MiB
and full logs for total-device accounting. Both 128K trials loaded in about
5.02 seconds, using 22664 MiB immediately after loading. The 120K-prompt trial
peaked at **3590148096 bytes (3.34 GiB) of scope host memory**.

Host protection matches the existing 27B launcher: MemoryHigh=6 GiB,
MemoryMax=8 GiB, MemorySwapMax=0; checkpoint limit **2**; RAM prompt-cache
limit **512 MiB**; Direct I/O loading. The logs confirm creation of checkpoint
1 of 2 and 2 of 2. These cgroup limits do not constrain GPU VRAM.

## Timing and acceptance evidence

All requests below generated 2048 tokens. Timings are the server’s separate
prompt and generation measurements, not an end-to-end combined rate.

| Trial | Prompt tokens | Prefill seconds | Decode seconds | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: |
| dflash-32k-ub512-d15 | 8192 | 4.467 | 14.723 | 1559 / 7255 |
| dflash-32k-ub1024-d15 | 8192 | 3.561 | 13.786 | 1603 / 6647 |
| dflash-32k-ub2048-d15 | 8192 | 3.390 | 12.978 | 1618 / 6433 |
| dflash-32k-ub4096-d15 | 8192 | 5.684 | 14.640 | 1595 / 6766 |
| dflash-32k-ub2048-d8 | 8192 | 3.385 | 8.447 | 1763 / 2266 |
| dflash-32k-ub2048-d12 | 8192 | 3.573 | 13.866 | 1573 / 5660 |
| dflash-32k-ub2048-d20 | 8192 | 3.377 | 13.135 | 1618 / 6433 |
| dflash-128k-ub2048-d8-f16 | 8192 | 3.599 | 8.470 | 1763 / 2266 |
| dflash-32k-28kprompt-ub2048-d8 | 28672 | 11.057 | 14.661 | 1523 / 4186 |
| dflash-128k-120kprompt-ub2048-d8-f16 | 122880 | 62.067 | 14.107 | 1635 / 3292 |

## Comparison with Swift and limits

The existing [Swift cold-prefill report](../llama-swift15/PREFILL_REPORT.md)
records about **832–855 prompt tok/s** on a 50000-token synthetic code prompt.
This Qwen run records **1979.8 tok/s at 122880 tokens** and about **2276–2420
tok/s at 8192 tokens** for the selected setup. Those are higher observed
prefill figures, but context lengths, exact prompt framing, KV formats,
architectures, and runtime versions differ; this is not a matched speedup
estimate. No new Swift or NInfer trials were run.

The existing [Swift handoff](../llama-swift15/HANDOFF.md) records about
**61–70 decode tok/s** on repetitive prompts with only 128 output tokens.
Qwen’s selected configuration measures **241.7 at 8K** and **145.1 at 120K**,
each with 2048 output tokens. This is promising throughput evidence, not
proof of a production-quality replacement. Acceptance is strongly dependent
on prompt and generated text: the same draft-8 setting ranges from 36.4%
at a 28K prompt to 77.8% at 8K. Greedy runs can diverge with different
verification batch shapes even when the input and sampling seed are fixed.

Tuning runs were single trials, not thermally normalized; maximum recorded
GPU temperature was 64°C. Whole-device telemetry includes desktop/background
use and is sampled roughly every 200 ms plus query overhead. No OOM or
host-memory-limit failure occurred. Synthetic forced-length generation does
not measure model quality or long-context recall.

## Reproduce and inspect

```bash
cd /projects/inference/llama-qwen36
./start_qwen36.sh
# Stop the running server before benchmarking:
systemctl --user stop qwen36.scope
python3 run_experiment.py
```

The experiment driver resumes existing labeled results; use a new project
log directory or new benchmark labels for fresh measurements. Initial benchmark
parameters remain explicit even though launcher defaults now use the winner.

Artifacts: `logs/<trial>/server.log`, `response.json`, `metrics.txt`, `gpu.csv`,
`result.json`, and frozen prompts. `logs/selection.json` records the selection
rule. `logs/final-chat.json` holds the ordinary EOS-terminated chat; it
produced 206 output tokens of clean Python, and is a smoke check, not a
long-output decode benchmark. `logs/final-state.json` records health, cgroup
limits, loaded CUDA libraries, and final GPU usage.

Validation: Python compilation, Bash syntax, artifact SHA256 verification,
actual baseline/DFlash chat, all requested tuning trials, and 128K short/
long-prompt runs passed. The existing Swift and NInfer repositories remain
clean and unmodified. Final server health is `ok`; scope limits are active.
