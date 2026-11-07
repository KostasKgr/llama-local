# Swift resident slot, no checkpoints: A trial, 2026-11-07

One T3/PI xhigh trial using the unchanged read-only investigation fixture at
32ecd71726fee16f335484c1f95cbce126d461f6. Windows b12110 CUDA 13.4,
131072 context, unified KV, q8_0 K/V, batch 2048/512, MTP depth 3 and vision
projector unchanged. Temporary settings: parallel 1, cache-prompt enabled,
cache-ram 0, no-cache-idle-slots, ctx-checkpoints 0. No profile changes.
GPU cooled below 50 C before launch; measured start after warm-up 41 C.
Laptop power/cooling configuration was not independently reconfirmed.

## Result

Resident prefix reuse worked on many turns, but did not eliminate full prefill.
44 completed requests before the first prompt above 98304 tokens reached
97868 prompt tokens. Five later full prefills occurred at 48314, 55371, 61553,
67482 and 71484 tokens, costing 279.7 seconds combined. Initial prefill was
9766 tokens in 6.42 seconds and is excluded from that count.

Total evaluated prompt tokens: 365913; prefill 347.7 s; decode 7015 tokens in
121.8 s. Decode rates by <32K / 32–64K / 64–96K prompt context:
69.08 / 61.58 / 48.43 tok/s. MTP acceptance 0.7055.
GPU busy median/max temperature 75/79 C, power 165.4/187.52 W;
sampled whole-GPU memory peak 20230 MiB. Telemetry includes the short over-cap
tail and must not be presented as precisely capped phase measurements.

The first over-cap request was 98577 tokens. Detection was sampled: subsequent requests through a
99943-token request began before interruption. All over-cap requests are excluded from the capped
request summary. The fixture Git state was unchanged. The thread was interrupted
at the cap, so this is not an investigation completion or answer-quality result.

## Divergence evidence

Before task 1177 the slot released with 41613 cached tokens. The next request
matched through 41548 tokens (65 tokens before the cached endpoint). The server
selected that same slot by prefix similarity, then logged forcing full prompt
re-processing due to lack of cache data for SWA/hybrid/recurrent memory.
This supports a rollback-state limitation rather than idle-slot eviction.
Exact changed text has not been established: full rendered prompt token arrays
were not captured. PI's full session preserves the structured conversation but
not the original raw generated tool-call serialization. Parsing/serialization
is a hypothesis, not a proven cause. The runtime supports --log-prompts-dir for
a future diagnostic run; it was not enabled during this trial.

The summary reports six full_reprocesses because it includes the initial prefill.
There were five later full re-prefills. No checkpoints were restored and prompt
RAM cache peak was zero. One trial does not establish relative performance against
the historical checkpoint baseline with different generated histories.

## Evidence and restoration

Ignored evidence: logs/swift15/agentic-20261107T100347Z-resident-a-no-checkpoints/
(meta.json, requests-96k.json, summary-96k.json, server-excerpt.log, gpu.csv,
host.csv). Ignored launcher: logs/swift15/trial-a-20261107/launch.py.
An initial benchmark-bracket attempt hit HTTP 503 while the model was loading;
its host sampler was cleaned up, and no T3 trial was launched from that attempt.
The successful bracket began only after health was ready.

T3 trial: <t3-thread://v2/af93199b-2123-4d3f-8b8d-2f8f1197fd11/mcp:launch:2e7cc39961f86fbe48c899444bdc8c8894dc0374a4f9f31745d20ea0cee93681>

The owned trial scope was stopped and llama-swap restarted with the saved
production profile. No implementation or saved profile settings were changed.

Restoration verified: llama-swap health OK, Swift active, original alias returned
`OK` on b12110. The mirrored-network port release delayed swap startup.

## Follow-up: endpoint replay and generated-token capture

Diagnostic evidence: ignored `logs/swift15/prefix-diagnostic-20261107/`.
The production server's `/apply-template` and `/tokenize` endpoints reproduced
both original request lengths exactly: 41524 before task 1146's response and
48314 after the recorded tool result. The original next request's LCP of 41548
lands at the `=path` token inside `<parameter=path>`.

A fresh owned A server on port 1257 then replayed that 41524-token prompt, using
the existing installed PI 1.5 adapter's `onPayload` callback to reconstruct the
request without issuing a PI network request. A short seed-42 chat generation
with xhigh thinking, 512-token limit and the same tools produced a `read` call.
Repeating it with `logprobs: true` captured the actual generated token IDs,
including the tool-call representation. Its parsed tool call and reasoning
matched the preceding seed-42 chat probe. No fixture tools were executed.

The generated tool call and its rendered history had identical decoded text
and parameter order (`path`, then `offset`). Nevertheless, at absolute token
41552, generation used token 28 (`=`) followed by token 2267 (`path`), whereas
re-tokenizing the rendered history used token 79114 (`=path`). The prefix up to
that point was identical. This is a noncanonical token segmentation in generated
tool-call text; neither changed dates nor argument reordering is needed.

Sending the parsed assistant message and the recorded tool result as the next
chat request reproduced the full-prefill failure: 48304 prompt tokens evaluated,
cache_n 0, prompt_n 48304, prompt_ms 39618.419. The server selected the existing
slot by LCP and then logged forcing full re-processing for missing compatible
hybrid/recurrent state. See `comparison.json`, `logprobs-response.json`,
`chat-raw-output.txt`, `chat-replayed.txt`, `chat-replayed-tokens.json`,
`next-response.json` and `server.log` in the diagnostic directory.

An auxiliary raw `/completion` probe generated parameters in a different order,
but that probe used a different generation path. It does not establish argument
reordering as the chat failure's cause; the chat-token evidence above supersedes
that initial hypothesis. The original trial's raw generated tokens remain
unavailable, so this is a direct reproduction of the mechanism at the same
field, not proof of the exact original generated token sequence or all five
original failures. Whether grammar constraints specifically caused the split
requires further investigation; no runtime fix was attempted.

`--log-prompts-dir` captured rendered input text, not raw generated tokens.
The chat logprobs supplied the missing generated-token evidence. No full agentic
rerun was needed and no saved runtime/profile defaults were changed.

After the diagnostic, the trial scope was stopped, llama-swap was restarted,
and Swift returned `OK` through its production alias.


## All-case review and why generation selects the split

All 47 completed request payloads available in the saved session (including
three beyond the cap) were reconstructed with PI's installed adapter, then
rendered/tokenized against the running pinned server. All 47 token counts
matched their original logged prompt lengths. The capped set contains one
initial prefill, five later full-prefill failures and 38 later requests without
full prefill. Every successful later request matched to the cached endpoint;
its rendered continuation began with an EOS delimiter or newline. Every failed
later request instead mismatched at token 79114 (`=path`) inside the preceding
assistant's `read` call.

| Next prompt tokens | Cached endpoint | First mismatch position | Rollback tokens | Full prefill seconds |
| ---: | ---: | ---: | ---: | ---: |
| 48314 | 41613 | 41548 | 65 | 39.71 |
| 55371 | 48625 | 48576 | 49 | 48.23 |
| 61553 | 55461 | 55398 | 63 | 56.21 |
| 67482 | 61644 | 61579 | 65 | 64.91 |
| 71484 | 67570 | 67505 | 65 | 70.62 |

The initial 9766-token request is a different case: it followed the unrelated
warm-up prompt and necessarily required a fresh prefill. It is not a later-turn
cache regression. Raw original generated token arrays were not recorded, so the
five identical mismatch locations support the reproduced token-boundary mechanism
but do not individually prove the original split sequences. The all-case audit
is saved in ignored `all-cases.json`, `all-cases.csv`, `all-payloads/` and the
per-request token windows under the diagnostic directory.

The live replay's `logprobs-response.json` gives the following pre-sampling
model probabilities at the relevant steps:

- After `<parameter`, standalone token 28 (`=`) has probability 0.8394223 and is
  the highest-probability token. It is a valid partial spelling of the required
  parameter opener. There is no longest-token/canonical-segmentation rule in
  sampling. The alternate single token 79114 (`=path`) is not forbidden merely
  because it is merged, but it was not the model's preferred next token.
- After the sampled `=`, raw top token `offset` has probability 0.8544880.
  Raw probability for the selected `path` token is 0.00000094945. These are
  **pre-sampling** probabilities, not the grammar-filtered sampling probability.
  Post-sampling probabilities are unsupported with speculation in this build.
- The Qwen coder-style parser used for this template generates a lazy tool-call
  grammar. Required arguments are emitted before the optional-argument loop.
  `read` has required `path` and optional `offset`/`limit`, so at the first
  parameter it must admit `path` and reject `offset`/`limit`.
- The common sampler checks a sampled token against the grammar; if invalid,
  it reapplies the grammar to the logits and runs the sampling chain again.
  Grammar checks decoded token pieces as text, allowing both `=` + `path` and
  `=path`. It enforces valid syntax/schema, not canonical BPE segmentation.
- Rendering the parsed call and re-tokenizing it deterministically chooses
  `=path` as one token. The active state still represents the two generated
  tokens. A hybrid/recurrent model without a usable rollback checkpoint cannot
  simply remove that suffix, so the server resets n_past to zero.

This explains the replay without attributing it to dates, idle-slot eviction,
PI parameter reordering or a blanket MTP bug. MTP can propose tokens, but the
pinned verification loop samples/verifies with the target common sampler and
accepts through that sampler. There was no MTP-off control, so this review does
not establish whether speculation changes the frequency of the behavior.
The model's training-level reason for preferring this spelling cannot be read
from a single inference; the logits establish the runtime preference.

Pinned source references (commit cd959bae99ff4bed22bef2604f45241c09bbb8c8):

- [Template parser selection](https://github.com/ggml-org/llama.cpp/blob/cd959bae9/common/chat.cpp#L1410)
- [Qwen tool grammar: required before optional arguments](https://github.com/ggml-org/llama.cpp/blob/cd959bae9/common/parsers/qwen3-coder.cpp#L109)
- [Sampler grammar validation and resampling](https://github.com/ggml-org/llama.cpp/blob/cd959bae9/common/sampling.cpp#L610)
- [Grammar checks decoded token pieces](https://github.com/ggml-org/llama.cpp/blob/cd959bae9/src/llama-grammar.cpp#L1435)
- [Raw versus post-sampling probability collection](https://github.com/ggml-org/llama.cpp/blob/cd959bae9/tools/server/server-context.cpp#L2266)
- [Speculation probability limitation](https://github.com/ggml-org/llama.cpp/blob/cd959bae9/tools/server/server-context.cpp#L5043)

This review used only rendering/tokenization endpoints and source analysis. No
new inference requests, profile changes or server restarts were needed.
