#!/usr/bin/env python3
"""Repeatable Strata API baseline and evidence brackets for real T3/pi threads."""
import argparse
import base64
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, probes, strata, swap

REQUIRED = ('apps/server/src/orchestration-v2/ThreadLaunchService.ts',
            'apps/server/src/orchestration-v2/ProviderTurnStartService.ts',
            'apps/server/src/orchestration-v2/ProviderSessionManager.ts',
            'apps/server/src/provider/ProviderRegistry.ts',
            'apps/server/src/provider/Drivers/ClaudeDriver.ts')
FIXTURE_COMMIT = '32ecd71726fee16f335484c1f95cbce126d461f6'


def provenance():
    files = [core.ROOT / 'llama.py', core.ROOT / 'taskfile.yml']
    for pattern in ('lib/*.py', 'experiments/*.py', 'profiles/*.json', 'manifests/*.json'):
        files.extend(core.ROOT.glob(pattern))
    return {'head': subprocess.check_output(['git', '-C', str(core.ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {str(path.relative_to(core.ROOT)): strata.digest(path) for path in sorted(files)},
            'pi_version': subprocess.check_output(['pi', '--version'], text=True).strip(),
            'power_scheme': core.powershell('powercfg /getactivescheme'),
            'runtime': {'strata': strata.pins(), 'models': core.manifest('models')['gyros'],
                        'llama_swap': swap.artifact()}}


def output_dir(label):
    path = core.ROOT / 'logs/gyros' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ-') + label)
    path.mkdir(parents=True)
    return path


def get(base, route):
    return probes.request(base, route, timeout=10)


def wait_public(base, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urlopen(base + '/health', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('Coordinator public endpoint did not become ready')
        time.sleep(0.25)


def chat(base, alias, messages, **options):
    return probes.request(base, '/v1/chat/completions', {'model': alias, 'messages': messages,
         'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': 512, **options}, timeout=1200)


def cool():
    deadline = time.monotonic() + 900
    while True:
        gpu = probes.gpu()
        if gpu[0] > 2048:
            raise ValueError('GPU occupied; leave its owner running')
        if gpu[2] < 50:
            return gpu[2]
        if time.monotonic() > deadline:
            raise TimeoutError('GPU did not cool below 50 C')
        time.sleep(5)


def validate_api(base, alias):
    results = {}
    for label, prompt, answer in [('math', 'What is 17 multiplied by 23? Reply with only the integer.', '391'),
                                  ('extract', 'Extract only the value: code=XQ19; status=ready', 'XQ19')]:
        result = chat(base, alias, [{'role': 'user', 'content': prompt}])
        if result['choices'][0]['message']['content'].strip() != answer:
            raise ValueError('Incorrect ' + label + ': ' + json.dumps(result))
        results[label] = result
    result = chat(base, alias, [{'role': 'user', 'content': 'What is 17 multiplied by 23? Give the integer.'}],
                  reasoning_effort='high', max_tokens=1024)
    if ('391' not in result['choices'][0]['message']['content']
            or not result['choices'][0]['message'].get('reasoning_content')):
        raise ValueError('Thinking response failed: ' + json.dumps(result))
    results['thinking'] = result
    tools = [{'type': 'function', 'function': {'name': 'read_nonce', 'description': 'Read the secret nonce.',
               'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
    messages = [{'role': 'user', 'content': 'Use read_nonce to get the nonce. Do not guess it.'}]
    result = chat(base, alias, messages, tools=tools)
    assistant = result['choices'][0]['message']
    calls = assistant.get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'read_nonce':
        raise ValueError('No structured tool call: ' + json.dumps(result))
    json.loads(calls[0]['function']['arguments'])
    nonce = 'STRATA_' + uuid.uuid4().hex
    result = chat(base, alias, messages + [assistant, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                  'content': nonce}, {'role': 'user', 'content': 'Reply with only the nonce.'}], tools=tools)
    if result['choices'][0]['message']['content'].strip() != nonce:
        raise ValueError('Tool result did not reach final answer: ' + json.dumps(result))
    results['tool_round_trip'] = result
    payload = {'model': alias, 'messages': [{'role': 'user', 'content': 'Reply with exactly: hello world'}],
               'temperature': 0, 'max_tokens': 64, 'reasoning_effort': 'none', 'stream': True}
    request = Request(base + '/v1/chat/completions', json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    chunks, done = [], False
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line[6:].strip() == b'[DONE]':
                done = True
                break
            chunks.append(json.loads(line[6:]))
    text = ''.join(c.get('delta', {}).get('content') or '' for chunk in chunks for c in chunk.get('choices', []))
    if not done or text.strip().lower() != 'hello world':
        raise ValueError('Streaming output failed: ' + text)
    results['streaming'] = {'chunks': chunks, 'done': done, 'text': text}
    return results


@contextmanager
def trial(public=1235, backend=1257):
    if swap.managed() or swap.service_state() not in ('inactive', 'failed'):
        raise ValueError('Coordinator active; stop its owner before a trial')
    if probes.gpu()[0] > 2048:
        raise ValueError('GPU occupied; no process was stopped')
    swap.ports(public, backend, backend)
    for port in (public, backend):
        if strata.port_open(port):
            raise ValueError(f'Port {port} occupied; no process was stopped')
    core.wait_port(public)
    core.wait_port(backend, core.windows_port_free)
    p = core.load('gyros', {'port': backend})
    p['scope'] = 'gyros-trial-' + uuid.uuid4().hex
    core.preflight(p)
    out = output_dir('trial')
    cfg = json.loads(swap.config(public, backend, backend, ['gyros']))
    cfg['models'][p['alias']]['env'] = ['LLAMA_SWAP_MANAGED=1', 'LLAMA_SCOPE=' + p['scope']]
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    process = None
    try:
        with (out / 'proxy.log').open('wb') as log:
            process = subprocess.Popen([str(swap.binary()), '-config', str(out / 'config.json'),
                 '-listen', f'127.0.0.1:{public}'], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise ValueError('Trial proxy exited; inspect ' + str(out))
                try:
                    with urlopen(f'http://127.0.0.1:{public}/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Trial proxy not ready')
                time.sleep(0.1)
            yield {'profile': p, 'out': out, 'base': f'http://127.0.0.1:{public}',
                   'backend': f'http://127.0.0.1:{backend}'}
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=50)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        strata.stop(p)
        if strata.running(p) or core.scope_active(p):
            raise RuntimeError('Trial cleanup did not finish')


def summarize_requests(records):
    rows = []
    for record in records:
        read = record.get('prompt_read')
        if read is None and record.get('prompt_tokens') is not None:
            read = max(0, record['prompt_tokens'] - (record.get('reused') or 0))
        rows.append({**record, 'prefill_tokens': read})
    result = {'requests': len(rows), 'rows': rows}
    for phase, tokens, milliseconds in [('prefill', 'prefill_tokens', 'prompt_ms'),
                                       ('decode', 'engine_generated', 'decode_ms')]:
        valid = [r for r in rows if r.get(milliseconds) and r.get(tokens) is not None
                 and (phase != 'decode' or r[tokens] >= 2)]
        total_ms = sum(r[milliseconds] for r in valid)
        result[phase + '_tok_s'] = sum(r[tokens] for r in valid) * 1000 / total_ms if total_ms else None
        result[phase + '_s'] = total_ms / 1000
        result[phase + '_tokens'] = sum(r[tokens] for r in valid)
        result[phase + '_timed_requests'] = len(valid)
    return result


def linux_identity(pid):
    try:
        base = Path('/proc') / str(pid)
        return {'pid': pid, 'start': base.joinpath('stat').read_text().rpartition(') ')[2].split()[19],
                'cmd': base.joinpath('cmdline').read_bytes().hex()}
    except FileNotFoundError:
        return None


def stop_collector(expected):
    if not expected or not expected.get('cmd'):
        raise ValueError('Incomplete owned collector identity; refusing to signal')
    if linux_identity(expected['pid']) != expected:
        return
    os.kill(expected['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while linux_identity(expected['pid']) == expected:
        if time.monotonic() > deadline:
            raise TimeoutError('Owned collector did not exit')
        time.sleep(0.1)


def start_collector(out, backend):
    command = [sys.executable, str(Path(__file__).resolve()), '_collect', str(out),
               '--health-url', backend]
    command_hex = (b'\0'.join(os.fsencode(arg) for arg in command) + b'\0').hex()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while True:
            current = linux_identity(process.pid)
            if current and current['cmd'] == command_hex:
                return current
            if process.poll() is not None:
                raise RuntimeError('Owned collector exited before its identity was captured')
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned collector did not finish exec')
            time.sleep(0.01)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise


def collect(out, backend):
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    with (out / 'metrics.jsonl').open('a') as stream:
        while not stopped:
            try:
                metrics = get(backend, '/metrics?requests=all')
                telemetry = get(backend, '/v1/telemetry')
                stream.write(json.dumps({'at': time.time(), 'instance': telemetry['instance']['id'],
                                         'metrics': metrics}) + '\n')
                stream.flush()
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.5)


def restore_policy(snapshot):
    for item in snapshot['files']:
        path = Path(item['path'])
        current = path.read_bytes() if path.exists() else None
        original = base64.b64decode(item['before']) if item['before'] is not None else None
        if current == original:
            continue
        if current is not None and hashlib.sha256(current).hexdigest() != item.get('installed_sha256'):
            raise ValueError('Fixture policy changed concurrently; retained backup in run metadata: ' + str(path))
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)
    if snapshot.get('nonce_file'):
        Path(snapshot['nonce_file']).unlink(missing_ok=True)


def bracket_start(args):
    fixture = Path(args.fixture).resolve()
    paths = [fixture / '.pi/extensions/pi-permission-system/config.json', fixture / '.git/info/exclude']
    snapshot = {'files': [{'path': str(path), 'before': base64.b64encode(path.read_bytes()).decode()
                          if path.exists() else None} for path in paths]}
    out = output_dir(args.label)
    try:
        _bracket_start(args, out, snapshot)
    except BaseException:
        restore_policy(snapshot)
        raise


def _bracket_start(args, out, policy_snapshot):
    if not swap.managed():
        raise ValueError('Start an owned coordinator with Gyro selected before the T3 bracket')
    wait_public(args.base_url)
    if 'gyros' not in json.loads(swap.state_path().read_text()).get('profiles', []):
        raise ValueError('The owned coordinator must select Gyro')
    start_temp = cool()
    fixture = Path(args.fixture).resolve()
    from experiments.agentic_bench import install_pi_policy, git_state
    initial = git_state(fixture)
    if initial['head'] != FIXTURE_COMMIT or initial['status'].strip():
        raise ValueError('Benchmark fixture must be clean at its pinned commit')
    policy = install_pi_policy(fixture)
    for item in policy_snapshot['files']:
        path = Path(item['path'])
        item['installed_sha256'] = strata.digest(path) if path.exists() else None
    nonce = 'T3_STRATA_' + uuid.uuid4().hex
    nonce_file = fixture / '.pi' / ('strata-' + uuid.uuid4().hex + '.txt')
    nonce_file.write_text(nonce + '\n')
    policy_snapshot['nonce_file'] = str(nonce_file)
    started = time.time()
    begin = time.monotonic()
    p = core.load('gyros')
    chat(args.base_url, p['alias'], [{'role': 'user', 'content': 'Hi'}], max_tokens=1)
    warm_s = time.monotonic() - begin
    before = get(args.health_url, '/metrics?requests=all')
    record, run = strata.owner(p)
    if record is None:
        raise ValueError('Coordinator did not launch the managed Gyro owner')
    prompt_doc = (core.ROOT / 'reports/swift15/AGENTIC_PROMPT.md').read_text()
    prompt = prompt_doc.split('```text\n', 1)[1].split('```', 1)[0].rstrip()
    prompt += ('\nResearch budget: use at most 32 tool calls. Read the five required files completely, '
               'once each, in contiguous chunks of at most 500 lines. For all additional files combined, '
               'use at most six targeted reads of at most 150 lines each; use short rg results to locate them. '
               'Do not read tests. Keep search output short. Once the five complete reads and the entry/dispatch '
               'and Claude prompt-delivery links are established, stop researching and write the answer. '
               'If a risk is uncertain, label it as a risk rather than investigating unrelated paths.\n')
    prompt += ('\n5. Read ' + str(nonce_file.relative_to(fixture)) + ' with the read tool and include its exact contents '
               'as the final line of your answer. Do not delegate. Stay within the read-only task.\n')
    (out / 'prompt.txt').write_text(prompt)
    meta = {'started': started, 'measured_at': time.time(), 'profile': p, 'fixture': str(fixture),
            'fixture_before': git_state(fixture), 'nonce': nonce, 'nonce_file': str(nonce_file), 'policy': policy,
            'base_url': args.base_url, 'health_url': args.health_url, 'windows_run': str(run),
            'cold_request_s': warm_s, 'before_ids': [r['id'] for r in before.get('requests', [])],
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'owner_token': record['token'],
            'policy_snapshot': policy_snapshot, 'start_temp_c': start_temp, 'provenance': provenance()}
    meta['collector'] = start_collector(out, args.health_url)
    strata.atomic(out / 'meta.json', meta)
    print(out)


def audit_session(path, fixture, nonce):
    calls, results, finals, actual_models = {}, {}, [], set()
    for line in Path(path).read_text().splitlines():
        event = json.loads(line)
        message = event.get('message', {})
        if message.get('role') == 'assistant':
            actual_models.add((message.get('provider'), message.get('model')))
            for block in message.get('content', []):
                if block.get('type') == 'toolCall':
                    calls[block['id']] = block
            if message.get('stopReason') == 'stop':
                finals.append(''.join(b.get('text', '') for b in message.get('content', []) if b.get('type') == 'text'))
        if message.get('role') == 'toolResult':
            results[message['toolCallId']] = message
    coverage = {name: [] for name in REQUIRED}
    successful_reads = 0
    nonce_read = False
    for call_id, call in calls.items():
        result = results.get(call_id)
        if call.get('name') != 'read' or result is None or result.get('isError'):
            continue
        successful_reads += 1
        text = '\n'.join(b.get('text', '') for b in result.get('content', []) if b.get('type') == 'text')
        nonce_read = nonce_read or nonce in text
        args = call.get('arguments', {})
        raw_path = args.get('path', args.get('file_path', ''))
        file = Path(raw_path)
        if not file.is_absolute():
            file = Path(fixture) / file
        for name in REQUIRED:
            if file.resolve() != (Path(fixture) / name).resolve():
                continue
            offset = int(args.get('offset') or 1)
            end = offset + len(text.splitlines()) - 1
            shown = re.search(r'Showing lines (\d+)[-–](\d+)', text)
            if shown:
                offset, end = map(int, shown.groups())
            limit = args.get('limit')
            if limit:
                end = min(end, offset + int(limit) - 1)
            source_lines = file.read_text().splitlines()
            end = min(end, len(source_lines))
            while end >= offset and '\n'.join(source_lines[offset - 1:end]) not in text:
                end -= 1
            coverage[name].append((offset, end))
    complete = {}
    for name, spans in coverage.items():
        end = 0
        for lo, hi in sorted(spans):
            if lo > end + 1:
                break
            end = max(end, hi)
        complete[name] = end >= len((Path(fixture) / name).read_text().splitlines())
    final = finals[-1] if finals else ''
    expected = ('llama-local', core.load('gyros')['alias'])
    forbidden = [c.get('name') for c in calls.values() if c.get('name') not in ('read', 'bash', 'grep', 'find', 'ls')]
    passed = (actual_models == {expected} and successful_reads >= 6 and all(complete.values()) and not forbidden
              and nonce_read and nonce in final and len(final.split()) >= 100)
    return {'passed': passed, 'actual_models': sorted(actual_models), 'successful_reads': successful_reads,
            'full_file_reads': complete, 'nonce_read': nonce_read, 'nonce_in_final': nonce in final, 'final_answer': final,
            'forbidden_tools': forbidden,
            'session_sha256': strata.digest(path)}


def finish(args):
    out = Path(args.run_dir).resolve()
    meta = json.loads((out / 'meta.json').read_text())
    stop_collector(meta['collector'])
    records = {}
    for line in (out / 'metrics.jsonl').read_text().splitlines():
        snapshot = json.loads(line)
        for record in snapshot['metrics'].get('requests', []):
            if record['id'] not in meta['before_ids']:
                records[(snapshot['instance'], record['id'])] = record
    summary = summarize_requests(list(records.values()))
    summary.update(meta=meta, finished=time.time(), thread_id=args.thread_id)
    from experiments.agentic_bench import git_state
    summary['fixture_unchanged'] = git_state(meta['fixture']) == meta['fixture_before']
    if args.pi_session:
        shutil.copyfile(args.pi_session, out / 'pi-session.jsonl')
        summary['agentic_audit'] = audit_session(out / 'pi-session.jsonl', meta['fixture'], meta['nonce'])
    for name in ('host.csv', 'gpu.csv', 'power.csv', 'owner.json', 'config.json', 'engine.log', 'server.log'):
        path = Path(meta['windows_run']) / name
        if path.exists():
            shutil.copyfile(path, out / name)
    from experiments import host_memory
    summary['telemetry'] = {name: host_memory.summarize_strata(out / name, meta['measured_at'], summary['finished'])
                            for name in ('host.csv', 'gpu.csv', 'power.csv') if (out / name).exists()}
    if (out / 'gpu.csv').exists():
        summary['telemetry']['gpu_busy'] = host_memory.summarize_strata(
            out / 'gpu.csv', meta['measured_at'], summary['finished'], busy_only=True)
    owner_record = json.loads((out / 'owner.json').read_text())
    summary['process_load_s'] = owner_record.get('load_s')
    strata.atomic(out / 'summary.json', summary)
    restore_policy(meta['policy_snapshot'])
    print(json.dumps({k: v for k, v in summary.items() if k not in ('meta', 'rows')}, indent=2))


def run(args):
    results = []
    for repeat in range(args.repeats):
        start_temp = cool()
        with trial(args.port, args.backend_port) as t:
            started = time.monotonic()
            chat(t['base'], t['profile']['alias'], [{'role': 'user', 'content': 'Hi'}], max_tokens=1)
            cold_s = time.monotonic() - started
            _, windows_run = strata.owner(t['profile'])
            validation = validate_api(t['base'], t['profile']['alias'])
            before = {r['id'] for r in get(t['backend'], '/metrics?requests=all')['requests']}
            samples = []
            for length in (4096, 16384, 32768):
                for kind, instruction in [('text', 'Write a detailed explanation of the document.'),
                                           ('code', 'Write a Python function that indexes these records, with comments.'),
                                           ('json', 'Return a JSON array summarizing these records.')]:
                    body = f'Run {repeat}, case {length}, {kind}.\n'
                    body += ''.join(f'Record {i}: the blue station stores seven boxes beside the river.\n'
                                    for i in range(length // 16))
                    messages = [{'role': 'user', 'content': body + '\n' + instruction}]
                    begin = time.monotonic()
                    response = chat(t['base'], t['profile']['alias'], messages)
                    samples.append({'target_prompt_tokens': length, 'kind': kind,
                                    'wall_s': time.monotonic() - begin,
                                    'prompt_sha256': hashlib.sha256(body.encode()).hexdigest(), 'response': response})
                    # Retain completed responses even if a later request or cleanup fails.
                    strata.atomic(t['out'] / 'partial.json', {'repeat': repeat + 1,
                                  'cold_request_s': cold_s, 'validation': validation, 'samples': samples})
            requests = [r for r in get(t['backend'], '/metrics?requests=all')['requests'] if r['id'] not in before]
            record = {'repeat': repeat + 1, 'cold_request_s': cold_s, 'validation': validation,
                      'start_temp_c': start_temp, 'provenance': provenance(),
                      'samples': samples, 'timings': summarize_requests(requests),
                      'profile': t['profile'], 'windows_run': str(windows_run)}
            strata.atomic(t['out'] / 'baseline.json', record)
            results.append(str(t['out']))
        print('Completed owned trial:', results[-1], flush=True)
        from experiments import host_memory
        for name in ('host.csv', 'gpu.csv', 'power.csv', 'owner.json', 'config.json', 'engine.log', 'server.log'):
            if (windows_run / name).is_file():
                shutil.copyfile(windows_run / name, t['out'] / name)
        record['telemetry'] = {name: host_memory.summarize_strata(t['out'] / name)
                               for name in ('host.csv', 'gpu.csv', 'power.csv') if (t['out'] / name).exists()}
        strata.atomic(t['out'] / 'baseline.json', record)
    print(json.dumps({'runs': results}, indent=2))


def report(args):
    rows = []
    for path in Path(args.run_dir).glob('**/summary.json'):
        row = json.loads(path.read_text())
        rows.append({'path': str(path), 'prefill_tok_s': row.get('prefill_tok_s'),
                     'decode_tok_s': row.get('decode_tok_s'), 'requests': row.get('requests'),
                     'agentic_passed': row.get('agentic_audit', {}).get('passed', False)})
    print(json.dumps(rows, indent=2))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    api = sub.add_parser('run')
    api.add_argument('--repeats', type=int, default=3)
    api.add_argument('--port', type=int, default=1235)
    api.add_argument('--backend-port', type=int, default=1257)
    start = sub.add_parser('start')
    start.add_argument('--label', required=True)
    start.add_argument('--fixture', default='/projects/agentic-bench-t3code')
    start.add_argument('--base-url', default='http://127.0.0.1:1235')
    start.add_argument('--health-url', default='http://127.0.0.1:1257')
    end = sub.add_parser('finish')
    end.add_argument('run_dir')
    end.add_argument('--thread-id', required=True)
    end.add_argument('--pi-session')
    summary = sub.add_parser('report')
    summary.add_argument('run_dir')
    collector = sub.add_parser('_collect')
    collector.add_argument('run_dir')
    collector.add_argument('--health-url', required=True)
    args = parser.parse_args()
    if args.action == 'run':
        if args.repeats <= 0:
            parser.error('repeats must be positive')
        run(args)
    elif args.action == 'start':
        bracket_start(args)
    elif args.action == 'finish':
        finish(args)
    elif args.action == '_collect':
        collect(Path(args.run_dir), args.health_url)
    else:
        report(args)


if __name__ == '__main__':
    main()
