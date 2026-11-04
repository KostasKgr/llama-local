#!/usr/bin/env python3
"""Start N isolated PI agents together against an existing llama-server backend.

Never starts/stops a server or changes client configuration. Use the backend
port (not llama-swap), whose /slots endpoint must be available and idle.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, pi, probes


def fixture(root):
    """A small code investigation with an independently checkable tool round trip."""
    names = ['api.py', 'service.py', 'store.py']
    markers = [uuid.uuid4().hex[:12] for _ in names]
    sources = [
        'from service import charge\ndef checkout(user, amount):\n    return charge(user, int(amount))\n',
        'from store import balances\ndef charge(user, amount):\n    balances[user] -= amount\n    return balances[user]\n',
        'balances = {"alice": 100, "bob": 50}\n',
    ]
    for name, marker, source in zip(names, markers, sources):
        (root / name).write_text(f'# Evidence marker: {marker}\n' + source)
    return names, ':'.join(markers)


def check_events(events, names, expected, alias):
    messages = [e['message'] for e in events if e.get('type') == 'message_end'
                and e.get('message', {}).get('role') == 'assistant']
    if not messages or messages[-1].get('stopReason') in ('error', 'aborted', 'length'):
        raise ValueError('Agent did not finish successfully')
    if any(m.get('model') != alias or m.get('provider') != pi.PROVIDER for m in messages):
        raise ValueError('Agent used a different model/provider')
    starts = [e for e in events if e.get('type') == 'tool_execution_start']
    if any(e.get('toolName') != 'read' for e in starts):
        raise ValueError('Unexpected tool')
    successful = {e.get('toolCallId') for e in events
                  if e.get('type') == 'tool_execution_end' and not e.get('isError')}
    reads = {Path(e.get('args', {}).get('path', '')).name for e in starts
             if e.get('toolCallId') in successful}
    if not set(names) <= reads:
        raise ValueError(f'Missing successful reads: {set(names) - reads}')
    answer = ''.join(b.get('text', '') for b in messages[-1].get('content', []) if b.get('type') == 'text')
    if answer.strip().splitlines()[-1:] != ['EVIDENCE=' + expected]:
        raise ValueError('Final evidence markers do not match fixture')
    return {'answer': answer, 'assistant_turns': len(messages), 'tool_calls': len(starts),
            'output_tokens': sum(m.get('usage', {}).get('output', 0) for m in messages),
            'input_tokens': sum(m.get('usage', {}).get('input', 0) for m in messages)}


def run(args, cancelled=None):
    cancelled = cancelled if cancelled is not None else threading.Event()
    p = core.load(args.profile, {'port': args.port})
    base = f'http://127.0.0.1:{p["port"]}'
    slots = probes.request(base, '/slots', timeout=10)
    if not isinstance(slots, list) or not slots:
        raise ValueError('Use a direct backend port with /slots enabled')
    if any(s['is_processing'] for s in slots):
        raise ValueError('Backend busy; refusing to mix workloads')
    if len(slots) < args.threads and not args.allow_queue:
        raise ValueError(f'{args.threads} threads require {args.threads} slots; backend has {len(slots)}. '
                         'Use --allow-queue only for a queueing test.')
    models = probes.request(base, '/v1/models', timeout=10)
    if not any(m['id'] == p['alias'] for m in models['data']):
        raise ValueError('Backend model does not match profile')
    context = min(s['n_ctx'] for s in slots)
    if args.output_tokens > context:
        raise ValueError('Output limit exceeds slot context')
    p['context'] = context
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = core.ROOT / 'logs' / args.profile / f'agentic-slots-{stamp}'
    out.mkdir(parents=True)
    print(out, flush=True)
    record = {'profile': args.profile, 'threads': args.threads, 'base': base,
              'slots_before': slots, 'context_per_slot': context, 'thinking': args.thinking,
              'queueing_test': args.threads > len(slots), 'passed': False}
    (out / 'meta.json').write_text(json.dumps(record, indent=2) + '\n')
    gate = threading.Barrier(args.threads)
    done = threading.Event()
    samples = []

    def monitor():
        with (out / 'samples.jsonl').open('w') as f:
            while not done.is_set():
                sample = {'monotonic': time.monotonic()}
                try:
                    sample['gpu'] = probes.gpu()
                    observed = probes.request(base, '/slots', timeout=3)
                    sample['processing_slots'] = sum(bool(s['is_processing']) for s in observed)
                except Exception as exc:
                    sample['error'] = str(exc)
                samples.append(sample)
                f.write(json.dumps(sample) + '\n')
                f.flush()
                done.wait(0.5)

    def worker(index):
        root = out / f'thread-{index:03d}'
        work = root / 'work'
        work.mkdir(parents=True)
        agent = root / 'agent'
        agent.mkdir()
        names, expected = fixture(work)
        (agent / 'models.json').write_text(pi.merge('{}', p, args.output_tokens, base_url=base + '/v1'))
        prompt = ('Read api.py, service.py and store.py in full using the read tool. '
                  'Trace checkout through charge to balances. Explain the result of checkout("alice", "-5") '
                  'and identify two validation risks with file:line evidence. Keep your answer under 250 words. '
                  'On the final line write EVIDENCE= followed by the three Evidence marker values joined by colons, '
                  'in api.py, service.py, store.py order. Do not guess them.')
        cmd = ['pi', '--offline', '--no-approve', '--no-session', '--no-mcp', '--no-extensions',
               '--no-skills', '--no-prompt-templates', '--no-context-files', '--no-themes',
               '--provider', pi.PROVIDER, '--model', p['alias'], '--thinking', args.thinking,
               '--tools', 'read', '--mode', 'json', '--print', '--', prompt]
        env = dict(os.environ, PI_CODING_AGENT_DIR=str(agent), PI_OFFLINE='1', PI_TELEMETRY='0')
        result = {'thread': index, 'passed': False}
        process = None
        try:
            gate.wait(timeout=30)
            result['started_monotonic'] = time.monotonic()
            with (root / 'events.jsonl').open('w') as stdout, (root / 'stderr.log').open('w') as stderr:
                process = subprocess.Popen(cmd, cwd=work, env=env, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                deadline = time.monotonic() + args.timeout
                while process.poll() is None:
                    if cancelled.is_set():
                        raise RuntimeError('Benchmark cancelled')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Agent deadline exceeded')
                    try:
                        process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        pass
                result['returncode'] = process.returncode
            if process.returncode:
                raise ValueError(f'PI exited {process.returncode}')
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()
                      if line.startswith('{')]
            result.update(check_events(events, names, expected, p['alias']))
            result['passed'] = True
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if 'started_monotonic' in result:
                result['elapsed_s'] = round(time.monotonic() - result['started_monotonic'], 3)
            (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            record['agents'] = list(pool.map(worker, range(1, args.threads + 1)))
        record['wall_s'] = round(time.monotonic() - started, 3)
        record['slots_after'] = probes.request(base, '/slots', timeout=10)
        record['health_after'] = probes.request(base, '/health', timeout=10)
        record['passed'] = (all(a['passed'] for a in record['agents'])
                            and record['health_after'].get('status') == 'ok')
    finally:
        done.set()
        monitor_thread.join(timeout=15)
        record['sampled_peak_gpu_mib'] = max((s['gpu'][0] for s in samples if 'gpu' in s), default=None)
        record['sampled_peak_processing_slots'] = max((s['processing_slots'] for s in samples
                                                       if 'processing_slots' in s), default=0)
        (out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('slots_before', 'slots_after', 'agents')}, indent=2))
    return record['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=core.PROFILES)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--port', type=int, required=True, help='Direct backend port exposing /slots')
    parser.add_argument('--timeout', type=int, default=300, help='Seconds per agent')
    parser.add_argument('--output-tokens', type=int, default=2048)
    parser.add_argument('--thinking', choices=('off', 'low', 'medium', 'high', 'xhigh'), default='off')
    parser.add_argument('--allow-queue', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= 128 or args.timeout <= 0 or args.output_tokens <= 0:
        parser.error('threads must be 1..128; timeout and output tokens must be positive')
    cancelled = threading.Event()
    def interrupt(signum, frame):
        cancelled.set()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    return 0 if run(args, cancelled) else 1


if __name__ == '__main__':
    sys.exit(main())
