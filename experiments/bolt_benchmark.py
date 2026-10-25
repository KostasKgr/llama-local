#!/usr/bin/env python3
"""Sequential Bolt API validation and DFlash comparison in owned trial scopes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, probes


def interrupted(signum, frame):
    raise KeyboardInterrupt


def validate(base, alias):
    def chat(messages, **kwargs):
        return probes.request(base, '/v1/chat/completions', {
            'model': alias, 'messages': messages, 'temperature': 0, 'max_tokens': 512,
            'chat_template_kwargs': {'enable_thinking': False}, **kwargs})

    results = {}
    results['known_answers'] = []
    for prompt, expected in [
        ('What is 7 multiplied by 8? Reply with only the integer.', '56'),
        ('What is the capital of France? Reply with only the city name.', 'Paris'),
        ('Extract the code from "code=XQ19; status=ready". Reply with only the code.', 'XQ19')]:
        response = chat([{'role': 'user', 'content': prompt}])
        assert response['choices'][0]['message']['content'].strip() == expected, response
        results['known_answers'].append({'expected': expected, 'response': response})
    for thinking in (False, True):
        response = chat([{'role': 'user', 'content': 'What is 17 multiplied by 23?'}],
                        chat_template_kwargs={'enable_thinking': thinking})
        assert '391' in response['choices'][0]['message']['content'], response
        assert bool(response['choices'][0]['message'].get('reasoning_content')) == thinking, response
        results[f'thinking_{thinking}'] = response
    response = chat([{'role': 'user', 'content': 'Say hello in one short sentence.'}],
                    chat_template_kwargs={'enable_thinking': False, 'conversational_mode': True})
    assert response['choices'][0]['message']['content'], response
    results['conversational'] = response
    tools = [{'type': 'function', 'function': {
        'name': 'lookup_temperature', 'description': 'Return current temperature in a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    messages = [{'role': 'user', 'content': 'Use lookup_temperature to get the current temperature in Athens.'}]
    response = chat(messages, tools=tools)
    assistant = response['choices'][0]['message']
    calls = assistant.get('tool_calls', [])
    assert len(calls) == 1 and calls[0]['function']['name'] == 'lookup_temperature', response
    assert 'Athens' in json.loads(calls[0]['function']['arguments'])['city'], response
    results['tool_call'] = response
    messages += [assistant, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                             'content': json.dumps({'temperature_c': 19, 'city': 'Athens'})}]
    response = chat(messages, tools=tools)
    assert '19' in response['choices'][0]['message']['content'], response
    results['tool_round_trip'] = response
    payload = {'model': alias, 'messages': [{'role': 'user', 'content': 'Count from 1 to 5.'}],
               'max_tokens': 128, 'temperature': 0, 'stream': True,
               'chat_template_kwargs': {'enable_thinking': False}}
    request = Request(base + '/v1/chat/completions', json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    chunks, done, first_content_s = [], False, None
    started = time.monotonic()
    with urlopen(request, timeout=180) as stream:
        for line in stream:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            chunk = json.loads(data)
            if first_content_s is None and any(c.get('delta', {}).get('content') for c in chunk['choices']):
                first_content_s = time.monotonic() - started
            chunks.append(chunk)
    content = ''.join(c['choices'][0].get('delta', {}).get('content') or '' for c in chunks if c['choices'])
    assert done and all(str(n) in content for n in range(1, 6)), content
    results['streaming'] = {'content': content, 'done': done, 'first_content_s': first_content_s}
    return results


def run(spec, repeats, client):
    p = core.load('bolt35', {'spec': spec})
    p['scope'] = 'bolt35-trial-' + uuid.uuid4().hex
    core.preflight(p)
    # Do not compete with another GPU process, even if it uses a different port.
    if probes.gpu()[0] > 2048:
        raise RuntimeError('GPU is occupied; leave its owner running and retry when free')
    out = core.ROOT / 'logs/bolt35' / (time.strftime('comparison-%Y%m%d-%H%M%S-') + spec)
    out.mkdir(parents=True, exist_ok=False)
    record = {**core.snapshot('bolt35'), 'profile': p, 'command': core.command(p),
              'stable': False, 'samples': []}
    base = f"http://127.0.0.1:{p['port']}"
    env = dict(os.environ, LLAMA_SCOPE=p['scope'], BOLT35_SPEC_TYPE=spec)
    process = None
    try:
        with (out / 'server.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(core.ROOT / 'llama.py'), 'start', 'bolt35',
                                        '--spec', spec], env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Server exited; inspect {out / "server.log"}')
            try:
                if probes.request(base, '/health', timeout=2)['status'] == 'ok':
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Startup timeout')
            time.sleep(1)
        record['validation'] = validate(base, p['alias'])
        workloads = {
            'chat': 'Explain five practical ways to reduce household energy use, with a paragraph for each.',
            'code': 'Write a Python LRU cache class with get and put, capacity validation and a short usage example.',
            'tool': 'Write a JSON schema for a weather lookup tool accepting city, country and units, and explain its fields.'}
        for name, prompt in workloads.items():
            for repeat in range(repeats):
                started = time.monotonic()
                response = probes.request(base, '/v1/chat/completions', {
                    'model': p['alias'], 'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0, 'seed': 42, 'max_tokens': 512, 'cache_prompt': False,
                    'chat_template_kwargs': {'enable_thinking': False}})
                record['samples'].append({'workload': name, 'repeat': repeat,
                                          'elapsed_s': time.monotonic() - started, 'response': response,
                                          'gpu_sample': probes.gpu()})
        if client:
            result = subprocess.run(['opencode', 'run', '--standalone', '--model', 'llama.cpp/' + p['alias'],
                                     'Reply with exactly these two words and nothing else: hello world'],
                                    capture_output=True, text=True, timeout=180)
            record['opencode'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
            assert result.returncode == 0 and 'hello world' in result.stdout.lower(), record['opencode']
        with urlopen(base + '/metrics', timeout=10) as metrics:
            record['metrics'] = metrics.read().decode()
        record['stable'] = True
    finally:
        if process is not None:
            probes.stop(p['scope'])
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(out, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', choices=['none', 'draft-dflash', 'both'], default='both')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--opencode', action='store_true', help='Also exercise an already registered OpenCode alias')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    signal.signal(signal.SIGTERM, interrupted)
    modes = ['none', 'draft-dflash'] if args.spec == 'both' else [args.spec]
    for spec in modes:
        run(spec, args.repeats, args.opencode)


if __name__ == '__main__':
    main()
