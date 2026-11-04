#!/usr/bin/env python3
"""Aggregate concurrent llama-server timings using unions of phase intervals.

Reads a captured server-excerpt.log. Durations come from server request timers;
phase boundaries are reconstructed backwards from the final timing log timestamp.
They include scheduling/contention time. A phase union counts overlapping slots
once, unlike summed per-request durations. Cached input tokens are not prefill.
"""
import argparse
import json
from pathlib import Path
import re


def union_seconds(intervals):
    total = 0.0
    end = float('-inf')
    for lo, hi in sorted(intervals):
        total += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return total


def parse(text):
    tasks = {}
    for line in text.splitlines():
        match = re.match(r'(\d+)\.(\d+)\.(\d+)\.(\d+) .*?id\s+(\d+) \| task (\d+) \| (.*)', line)
        if not match:
            continue
        minute, second, milli, micro, slot, task, detail = match.groups()
        stamp = int(minute) * 60 + int(second) + int(milli) / 1000 + int(micro) / 1000000
        row = tasks.setdefault(int(task), {'task': int(task), 'slot': int(slot)})
        context = re.search(r'new prompt.*task.n_tokens = (\d+)', detail)
        if context:
            row['context'] = int(context[1])
        for phase, pattern in [('prefill', r'^prompt eval time'), ('decode', r'^\s*eval time')]:
            timing = re.match(pattern + r' =\s*([\d.]+) ms /\s*(\d+) tokens', detail)
            if timing:
                row[phase + '_s'] = float(timing[1]) / 1000
                row[phase + '_tokens'] = int(timing[2])
                if phase == 'decode':
                    row['end'] = stamp
    rows = [r for r in tasks.values() if all(k in r for k in ('prefill_s', 'decode_s', 'end'))]
    for r in rows:
        r['decode_start'] = r['end'] - r['decode_s']
        r['start'] = r['decode_start'] - r['prefill_s']
    return sorted(rows, key=lambda r: r['start'])


def summarize(rows):
    if not rows:
        raise ValueError('No completed request timings found')
    prefill = [(r['start'], r['decode_start']) for r in rows]
    decode = [(r['decode_start'], r['end']) for r in rows]
    window = max(r['end'] for r in rows) - min(r['start'] for r in rows)
    result = {'requests': len(rows), 'window_s': window,
              'active_union_s': union_seconds(prefill + decode),
              'max_context': max((r.get('context', 0) for r in rows), default=0)}
    for phase, intervals in [('prefill', prefill), ('decode', decode)]:
        tokens = sum(r[phase + '_tokens'] for r in rows)
        elapsed = union_seconds(intervals)
        result[phase] = {'tokens': tokens, 'phase_union_s': elapsed,
                         'combined_tok_s': tokens / elapsed if elapsed else None,
                         'whole_window_tok_s': tokens / window if window else None}
    # Both-phase overlap means some decode occurs while another slot prefills.
    result['prefill_decode_overlap_s'] = (result['prefill']['phase_union_s']
                                          + result['decode']['phase_union_s'] - result['active_union_s'])
    slots = sorted({r['slot'] for r in rows})
    result['per_slot'] = {}
    for slot in slots:
        subset = [r for r in rows if r['slot'] == slot]
        result['per_slot'][str(slot)] = {'requests': len(subset), **{
            phase + '_tok_s': (sum(r[phase + '_tokens'] for r in subset)
                              / sum(r[phase + '_s'] for r in subset))
            for phase in ('prefill', 'decode') if sum(r[phase + '_s'] for r in subset)}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    rows = parse((args.run_dir / 'server-excerpt.log').read_text())
    summary = summarize(rows)
    (args.run_dir / 'concurrent-requests.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.run_dir / 'concurrent-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
