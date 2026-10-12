#!/usr/bin/env python3
"""Run the requested experiment sequentially; resume completed labeled trials."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def trial(label, *options):
    path = ROOT / 'logs' / label / 'result.json'
    if not path.exists():
        subprocess.run([sys.executable, str(ROOT / 'benchmark.py'), '--label', label, *map(str, options)],
                       check=False)
    if not path.exists():
        raise RuntimeError(f'Trial {label} did not save a result; inspect before resuming')
    result = json.loads(path.read_text())
    print('TRIAL', label, 'stable=' + str(result['stable']), flush=True)
    return result


def main():
    baseline = trial('baseline-32k', '--spec', 'none', '--smoke')
    if not baseline['stable']:
        raise SystemExit('Baseline failed; stopping before DFlash')
    smoke = trial('dflash-smoke-32k', '--smoke')
    if not smoke['stable']:
        raise SystemExit('DFlash smoke failed; stopping before tuning')
    initial = trial('dflash-32k-ub1024-d15')
    if not initial['stable']:
        raise SystemExit('DFlash benchmark failed; stopping before tuning')
    microbatches = [initial]
    for ub in (512, 2048, 4096):
        result = trial(f'dflash-32k-ub{ub}-d15', '--ubatch', ub)
        if not result['stable']:
            print('Stopping microbatch sweep at failed configuration', ub, flush=True)
            break
        microbatches.append(result)
    # Require decode to retain at least 90% of the initial stable configuration.
    reference_decode = initial['timings']['predicted_per_second']
    eligible = [r for r in microbatches if r['timings']['predicted_per_second'] >= 0.9 * reference_decode]
    best_prefill = max(eligible, key=lambda r: r['timings']['prompt_per_second'])
    ub = best_prefill['ubatch']
    print('SELECTED_UBATCH', ub, json.dumps(best_prefill['timings']), flush=True)
    drafts = [best_prefill]
    for draft in (8, 12, 20):
        result = trial(f'dflash-32k-ub{ub}-d{draft}', '--ubatch', ub, '--draft-max', draft)
        if result['stable']:
            drafts.append(result)
    # This pinned draft is trained with block_size=16: requested 20 is clamped to 15.
    eligible_drafts = [r for r in drafts if r['draft_max'] <= 15]
    best_decode = max(eligible_drafts, key=lambda r: r['timings']['predicted_per_second'])
    draft = best_decode['draft_max']
    selection = {'ubatch': ub, 'draft_max': draft,
        'prefill_trial': best_prefill['label'], 'decode_trial': best_decode['label'],
        'selection_rule': 'Highest prefill with decode >=90% of ub1024; then highest decode, excluding clamped duplicate',
        'single_run_selection': True}
    (ROOT / 'logs/selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('SELECTED', json.dumps(selection), flush=True)
    near32 = trial(f'dflash-32k-28kprompt-ub{ub}-d{draft}', '--ubatch', ub, '--draft-max', draft,
                   '--prompt-size', 28672)
    large = trial(f'dflash-128k-ub{ub}-d{draft}-f16', '--context', 131072, '--ubatch', ub,
                  '--draft-max', draft)
    if not large['stable']:
        raise SystemExit('Unquantized 128K failed; report exact allocation output before considering quantized KV')
    trial(f'dflash-128k-120kprompt-ub{ub}-d{draft}-f16', '--context', 131072, '--ubatch', ub,
          '--draft-max', draft, '--prompt-size', 122880)
    print('EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
