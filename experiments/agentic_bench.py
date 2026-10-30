"""Bracket an externally driven agentic run (for example a T3/PI thread) with GPU
sampling and a server-log summary. It never starts or stops a model server.

  python3 experiments/agentic_bench.py start --label balanced --fixture /projects/agentic-bench-t3code
  python3 experiments/agentic_bench.py finish logs/swift15/agentic-...
  python3 experiments/agentic_bench.py cool --below 50
"""
import argparse
import json
import os
import re
import signal
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core

PI_POLICY = Path(__file__).with_name('agentic_bench_pi_policy.json')
PI_AGENT = Path.home() / '.pi' / 'agent'
GPU_FIELDS = ('timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,'
              'memory.used,clocks_event_reasons.active')
BUCKETS = [(0, 32768), (32768, 65536), (65536, 98304), (98304, 1 << 30)]


def newest_log(profile):
    logs = sorted((core.ROOT / 'logs' / profile).glob('2*/server.log'))
    return logs[-1] if logs else None


def git_state(path):
    run = lambda *a: subprocess.run(['git', '-C', str(path), *a], capture_output=True, text=True, check=True).stdout
    return {'head': run('rev-parse', 'HEAD').strip(), 'status': run('status', '--porcelain', '--untracked-files=all')}


def gpu_temp():
    out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def install_pi_policy(fixture):
    """Place the read-only PI permission policy in the fixture before every run.

    Project and global pattern maps shallow-merge, so every global key is
    overridden here too; a global allow added later cannot leak into the bench.
    Project policy only loads for trusted projects, so refuse an untrusted fixture.
    """
    fixture = Path(fixture).resolve()
    trust = json.loads((PI_AGENT / 'trust.json').read_text())
    if trust.get(str(fixture)) is not True:
        sys.exit(f'{fixture} is not trusted by PI; its permission policy would be ignored')
    policy = json.loads(PI_POLICY.read_text())
    glob = json.loads((PI_AGENT / 'extensions' / 'pi-permission-system' / 'config.json').read_text())['permission']
    for surface, rules in glob.items():
        mine = policy['permission'].get(surface)
        if isinstance(rules, dict) and isinstance(mine, dict):
            # Prepend unknown global keys as deny; later (specific) policy keys still win.
            policy['permission'][surface] = {**{k: 'deny' for k in rules if k not in mine}, **mine}
    target = fixture / '.pi' / 'extensions' / 'pi-permission-system' / 'config.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(policy, indent=2) + '\n')
    exclude = fixture / '.git' / 'info' / 'exclude'
    if '.pi/' not in exclude.read_text().split():
        with exclude.open('a') as f:
            f.write('.pi/\n')
    return str(target)


def start(args):
    policy = install_pi_policy(args.fixture) if args.fixture else None
    log = newest_log(args.profile)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = core.ROOT / 'logs' / args.profile / f'agentic-{stamp}-{args.label}'
    out.mkdir(parents=True)
    with (out / 'gpu.csv').open('w') as f:
        sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=' + GPU_FIELDS, '--format=csv,nounits',
                                    '-lms', str(args.interval_ms)], stdout=f, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    meta = {'label': args.label, 'profile': args.profile, 'started': stamp, 'start_temp_c': gpu_temp(),
            'server_log': str(log) if log else None, 'log_offset': log.stat().st_size if log else 0,
            'sampler_pid': sampler.pid, 'fixture': args.fixture, 'pi_policy': policy,
            'fixture_before': git_state(args.fixture) if args.fixture else None}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(out)


def parse_server(text):
    tasks = {}
    get = lambda t: tasks.setdefault(t, {'task': int(t)})
    for t, n in re.findall(r'task (\d+) \| new prompt.*?task\.n_tokens = (\d+)', text):
        get(t)['context'] = int(n)
    for t, ms, n in re.findall(r'task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens', text):
        get(t).update(prefill_ms=float(ms), prefill_tokens=int(n))
    for t, ms, n in re.findall(r'task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens', text):
        get(t).update(decode_ms=float(ms), decode_tokens=int(n))
    for t, a, acc, gen in re.findall(r'task (\d+) \| draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated', text):
        get(t).update(acceptance=float(a), draft_accepted=int(acc), draft_generated=int(gen))
    for t in re.findall(r'task (\d+) \| restored context checkpoint', text):
        get(t)['checkpoint_restored'] = True
    for t in re.findall(r'task (\d+) \| forcing full prompt re-processing', text):
        get(t)['full_reprocess'] = True
    return [r for r in tasks.values() if 'decode_ms' in r]


def rate(rows, ms, n):
    total = sum(r.get(ms, 0) for r in rows)
    return round(sum(r.get(n, 0) for r in rows) / total * 1000, 2) if total else None


def summarize_gpu(path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        f = [x.strip() for x in line.split(',')]
        try:
            rows.append({'power': float(f[1]), 'sm': int(f[2]), 'mem': int(f[3]), 'temp': int(f[4]),
                         'util': int(f[5]), 'vram': int(f[6]), 'reasons': int(f[7], 16)})
        except (ValueError, IndexError):
            pass
    busy = [r for r in rows if r['util'] >= 50]
    if not busy:
        return {'samples': len(rows), 'busy_samples': 0}
    med = lambda k: round(st.median(r[k] for r in busy), 1)
    # Idle (0x1) is not a throttle; any other active reason while busy is.
    throttled = sum(1 for r in busy if r['reasons'] & ~0x1)
    return {'samples': len(rows), 'busy_samples': len(busy), 'power_w_median': med('power'),
            'power_w_max': max(r['power'] for r in busy), 'sm_mhz_median': med('sm'), 'mem_mhz_median': med('mem'),
            'temp_c_median': med('temp'), 'temp_c_max': max(r['temp'] for r in busy),
            'vram_mib_max': max(r['vram'] for r in rows), 'throttled_fraction': round(throttled / len(busy), 3)}


def finish(args):
    out = Path(args.run_dir).resolve()
    meta = json.loads((out / 'meta.json').read_text())
    excerpt = out / 'server-excerpt.log'
    if excerpt.exists():
        # Re-analysis of a finished run: the sampler is gone and its PID may be reused,
        # and the live log now continues past this run.
        text = excerpt.read_text()
        log = Path(json.loads((out / 'summary.json').read_text())['server_log'])
    else:
        try:
            os.killpg(meta['sampler_pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        log = newest_log(meta['profile'])
        # A server restart during the run produces a new log; read that one from the start.
        offset = meta['log_offset'] if log and str(log) == meta['server_log'] else 0
        with log.open('rb') as f:
            f.seek(offset)
            text = f.read().decode(errors='replace')
        excerpt.write_text(text)
    rows = parse_server(text)
    if args.max_context:
        # Stop at the first request beyond the cap: later requests (including any
        # after a client compaction shrinks the context again) are excluded.
        rows.sort(key=lambda r: r['task'])
        over = next((i for i, r in enumerate(rows) if r.get('context', 0) > args.max_context), len(rows))
        rows = rows[:over]
    summary = {'label': meta['label'], 'started': meta['started'], 'max_context_cap': args.max_context,
               'finished': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
               'start_temp_c': meta['start_temp_c'], 'server_log': str(log), 'requests': len(rows),
               'prefill_tokens': sum(r.get('prefill_tokens', 0) for r in rows),
               'decode_tokens': sum(r['decode_tokens'] for r in rows),
               'prefill_s': round(sum(r.get('prefill_ms', 0) for r in rows) / 1000, 1),
               'decode_s': round(sum(r['decode_ms'] for r in rows) / 1000, 1),
               'prefill_tok_s': rate(rows, 'prefill_ms', 'prefill_tokens'),
               'decode_tok_s': rate(rows, 'decode_ms', 'decode_tokens'),
               'max_context': max((r.get('context', 0) for r in rows), default=0),
               'checkpoint_restores': sum(1 for r in rows if r.get('checkpoint_restored')),
               'full_reprocesses': sum(1 for r in rows if r.get('full_reprocess'))}
    acc, gen = sum(r.get('draft_accepted', 0) for r in rows), sum(r.get('draft_generated', 0) for r in rows)
    summary['draft_acceptance'] = round(acc / gen, 4) if gen else None
    summary['decode_tok_s_by_context'] = {
        f'{lo // 1024}-{hi // 1024 if hi < 1 << 30 else "max"}K': rate([r for r in rows if lo <= r.get('context', 0) < hi],
                                                                        'decode_ms', 'decode_tokens')
        for lo, hi in BUCKETS}
    # Incremental prefill only: full reprocesses are reported separately.
    incremental = [r for r in rows if not r.get('full_reprocess')]
    summary['prefill_tok_s_by_context'] = {
        f'{lo // 1024}-{hi // 1024 if hi < 1 << 30 else "max"}K': rate([r for r in incremental if lo <= r.get('context', 0) < hi],
                                                                        'prefill_ms', 'prefill_tokens')
        for lo, hi in BUCKETS}
    summary['full_prefills'] = [{'tokens': r['prefill_tokens'], 'seconds': round(r['prefill_ms'] / 1000, 1),
                                 'tok_s': rate([r], 'prefill_ms', 'prefill_tokens')}
                                for r in rows if r.get('full_reprocess') and r.get('prefill_ms')]
    summary['gpu'] = summarize_gpu(out / 'gpu.csv')
    if meta['fixture']:
        after = git_state(meta['fixture'])
        summary['fixture_unchanged'] = after == meta['fixture_before']
    suffix = f'-{args.max_context // 1024}k' if args.max_context else ''
    (out / f'requests{suffix}.json').write_text(json.dumps(rows, indent=2))
    (out / f'summary{suffix}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cool(args):
    deadline = time.monotonic() + args.timeout
    while (temp := gpu_temp()) >= args.below:
        if time.monotonic() > deadline:
            sys.exit(f'GPU still at {temp} C after {args.timeout}s')
        time.sleep(10)
    print(f'GPU at {temp} C')


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest='cmd', required=True)
s = sub.add_parser('start')
s.add_argument('--label', required=True)
s.add_argument('--profile', default='swift15', choices=sorted(core.PROFILES))
s.add_argument('--fixture', help='git checkout that must remain unchanged')
s.add_argument('--interval-ms', type=int, default=1000)
s.set_defaults(func=start)
f = sub.add_parser('finish')
f.add_argument('run_dir')
f.add_argument('--max-context', type=int, help='only count requests up to the first one above this context')
f.set_defaults(func=finish)
c = sub.add_parser('cool')
c.add_argument('--below', type=int, default=50)
c.add_argument('--timeout', type=int, default=900)
c.set_defaults(func=cool)
args = parser.parse_args()
args.func(args)
