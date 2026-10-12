# Consolidation validation — 2026-10-12

Implemented under `/projects/inference/llama-local`. Both original Git histories
are preserved. Model weights and versioned runtimes were moved locally, not
redownloaded. Old repository paths are compatibility symlinks; original checkouts
are retained under ignored `legacy/`. HANDOFF.md was not imported or consulted.

Validation:

- Six unit tests passed: pinned configuration, invalid overrides, owned-scope
  validation, failed download promotion, port conflict refusal, manifest coverage.
- Python compilation, Bash syntax and Git whitespace checks passed.
- Both profiles loaded at 131072 context, advertised their preserved aliases,
  answered a chat request, and used the expected 6G/8G/no-swap scope limits.
- All pinned model and runtime archive sizes/hashes verified successfully.
- Both installers successfully reused the existing archives, extracted the runtimes,
  and recorded matching server hashes and bundled CUDA runtime version 12080.
- Qwen: cold 8192-token prompt, 128 generated tokens, F16 KV, DFlash max 8,
  batch/microbatch 4096/2048, 128K context. Fully evaluated prompt, no cache reuse,
  no truncation, and healthy server afterward. About 2187 prompt tokens/s and
  84 decode tokens/s in this short screening run.
- Swift: cold 50000-token prefill, one output token, MTP disabled, q8_0 KV,
  batch/microbatch 2048/512, 128K context. Fully evaluated prompt, no cache reuse,
  no truncation, and healthy server afterward. About 918 prompt tokens/s.
  Production MTP remains enabled and was exercised by the API smoke check.
- Fresh-process probes used unique owned scopes and stopped their servers.

These are migration checks, not a new tuning study or performance parity claim.
The short Qwen decode run is not comparable to the historical 2048-token run.
Swift's prefill check deliberately used MTP disabled; it does not measure
production MTP decode performance. Existing tested production defaults are
unchanged. Historical reports retain their original methodological limitations.

Local evidence: `logs/qwen36/migration-check/`,
`logs/swift15/prefill-20261012-094441/`, and both profiles' `smoke-*` directories.
The first adapted Swift prefill attempt failed from a missing GPU helper binding;
its failed result is retained, the binding was corrected, and the rerun passed.
