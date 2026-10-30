# Swift 1.5 agentic benchmark prompt

Used by [AGENTIC_POWER_MODES.md](AGENTIC_POWER_MODES.md). Send it as the first
message of a new T3 Code thread:

- Project: `/projects/agentic-bench-t3code`, a fresh clone of upstream
  `pingdotgg/t3code` at `32ecd71726fee16f335484c1f95cbce126d461f6`.
- Provider: PI, model `llama-local/swift-1.5-qwen3.8-27b-iq3s-mtp`, thinking `xhigh`.
- Runtime mode: full access. Read-only behavior is enforced by the deny-first
  PI permission policy `experiments/agentic_bench_pi_policy.json`, which
  `agentic_bench.py start --fixture` installs into
  `<fixture>/.pi/extensions/pi-permission-system/config.json` before every run
  (git-excluded). It allows only read-only shell commands, denies writes, edits,
  MCP and paths outside the fixture without prompting, and auto-denies any
  remaining prompt after one second. PI loads project policy only for trusted
  projects, so the fixture is listed in `~/.pi/agent/trust.json`; `start`
  refuses an untrusted fixture.

Bracket each run with `python3 experiments/agentic_bench.py start --label LABEL
--fixture /projects/agentic-bench-t3code` and `... finish RUN_DIR`. Run
`... cool --below 50` before starting. Interrupt the thread once the server log
shows a request above 96K context (`task.n_tokens` > 98304), then also run
`... finish RUN_DIR --max-context 98304` for the comparable summary.

## Prompt

```text
This is a read-only code investigation. Do not create, modify, or delete any file. Do not install packages, use the network, or run commands that write to disk. Use only the read tool and read-only shell commands (rg, grep, find, ls, wc). Never use sed; to view part of a file, use the read tool with offset and limit.

Question: in this repository, trace what happens on the server when a client starts a new thread whose provider is Claude, from the moment the launch request reaches the server until the Claude provider process receives its first prompt.

Requirements:
1. Find the entry point, then follow the chain through apps/server/src/orchestration-v2 (ThreadLaunchService.ts, ProviderTurnStartService.ts, ProviderSessionManager.ts) and apps/server/src/provider (ProviderRegistry.ts, Drivers/ClaudeDriver.ts). Read each of these five files in full, not just search hits; use multiple reads for long files.
2. Give a numbered call chain with file:line for every step.
3. Explain when an existing provider session is reused and when a new one is created, and where errors during start are handled.
4. List 3 risks or bugs you noticed, each with file:line evidence.

Keep the final answer under 900 words.
```

## Revisions

- Balanced runs 1–2 allowed `sed -n` in the shell command list. PI's permission
  policy prompted for `sed` and the user approved it for the session, adding a
  pause to wall time. From Performance run 1 on, `sed` is removed and forbidden;
  the rest of the prompt is unchanged. Swift still used `sed` in Performance 1,
  so `sed -n` is now allowed by the policy (in-place `-i` stays denied).
