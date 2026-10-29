# Local inference

Read README.md before changing tooling. Check existing changes first.
Keep profiles' runtime pins and measured defaults unchanged unless explicitly
requested. Do not add model weights, runtimes, downloads, logs, legacy archives,
or HANDOFF.md to Git. Historical reports are evidence, not current instructions.
After code changes, run `task agent:verify`. For launch/lifecycle changes also
run sequential live smoke checks when the GPU and port are free.
Never interrupt an unrelated server; benchmark cleanup must own its scope.
When adding top-level Python paths or `start_*.sh` launchers, update
`taskfile.yml` so `task agent:verify` checks them.
