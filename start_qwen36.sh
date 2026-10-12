#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$ROOT/llama.cpp-runtime/llama-server"
MODEL="$ROOT/models/qwen36-35b-a3b/IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00001-of-00002.gguf"
DRAFT="$ROOT/models/qwen36-35b-a3b/dflash-Qwen3.6-35B-A3B-Q8_0.gguf"
CONTEXT="${QWEN_CONTEXT:-131072}"
BATCH="${QWEN_BATCH:-4096}"
UBATCH="${QWEN_UBATCH:-2048}"
DRAFT_MAX="${QWEN_DRAFT_MAX:-8}"
SPEC_TYPE="${QWEN_SPEC_TYPE:-draft-dflash}"
LOAD_MODE="${QWEN_LOAD_MODE:-dio}"
CACHE_K="${QWEN_CACHE_K:-f16}"
CACHE_V="${QWEN_CACHE_V:-f16}"
[[ -x "$SERVER" ]] || { echo "Missing executable: $SERVER" >&2; exit 1; }
[[ -f "$ROOT/ARTIFACTS.json" ]] || { echo "Run download.py to verify artifacts first" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Missing verified model: $MODEL" >&2; exit 1; }
args=(-m "$MODEL" --load-mode "$LOAD_MODE" --fit off -ngl 999 -c "$CONTEXT" -fa on
      -b "$BATCH" -ub "$UBATCH" -np 1 --ctx-checkpoints 2 --cache-ram 512
      --cache-type-k "$CACHE_K" --cache-type-v "$CACHE_V"
      --spec-type "$SPEC_TYPE" --jinja --host 127.0.0.1 --port 1235
      --alias qwen3.6-35b-a3b-iq3s-dflash --metrics --perf --log-verbosity 4)
case "$SPEC_TYPE" in
    draft-dflash)
        [[ -f "$DRAFT" ]] || { echo "Missing verified DFlash draft: $DRAFT" >&2; exit 1; }
        args+=(-md "$DRAFT" -ngld 999 --spec-draft-n-max "$DRAFT_MAX") ;;
    none) ;;
    *) echo "Unsupported speculation type: $SPEC_TYPE" >&2; exit 1 ;;
esac
export LD_LIBRARY_PATH="$ROOT/llama.cpp-runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'Context=%s batch=%s ubatch=%s spec=%s draft-max=%s KV=%s/%s load=%s\n' \
    "$CONTEXT" "$BATCH" "$UBATCH" "$SPEC_TYPE" "$DRAFT_MAX" "$CACHE_K" "$CACHE_V" "$LOAD_MODE"
# A dedicated scope contains host OOM to this server, including file-cache charges.
# These cgroup limits do not limit GPU memory.
exec systemd-run --user --scope --unit=qwen36 --collect \
    -p MemoryHigh=6G -p MemoryMax=8G -p MemorySwapMax=0 \
    "$SERVER" "${args[@]}"
