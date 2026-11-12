#!/usr/bin/env python3
"""Sequential real-model swap smoke; owns only unique trial scopes."""
import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, probes, swap
from experiments.bolt_benchmark import validate
from experiments.pi_smoke import solid_png


def interrupted(signum, frame):
    raise KeyboardInterrupt


def run(port=1235, backend_port=1236, windows_backend_port=1257, profiles=None):
    swap.ports(port, backend_port, windows_backend_port)
    if swap.managed() or swap.service_state() not in ('inactive', 'failed'):
        raise ValueError('llama-swap is active; leave its owner running')
    if probes.gpu()[0] > 2048:
        raise ValueError('GPU is occupied; leave its owner running and retry when free')
    if not os.access(swap.binary(), os.X_OK):
        raise ValueError('Run python3 llama.py swap install first')
    names = swap.selected(profiles)
    swap.preflight(port, backend_port, windows_backend_port, profiles)
    # A Windows server may have just released the public port (see core.wait_port).
    core.wait_port(port)
    token = uuid.uuid4().hex
    scopes = {name: name + '-trial-' + token for name in names}
    config = json.loads(swap.config(port, backend_port, windows_backend_port, profiles))
    aliases = {name: core.load(name)['alias'] for name in scopes}
    for name, alias in aliases.items():
        config['models'][alias]['env'] = ['LLAMA_SWAP_MANAGED=1', 'LLAMA_SCOPE=' + scopes[name]]
    out = core.ROOT / 'logs' / 'swap' / datetime.now(timezone.utc).strftime('smoke-%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(json.dumps(config, indent=2) + '\n')
    record = {'stable': False, 'scopes': scopes, 'switches': [], 'port': port,
              'backend_port': backend_port, 'windows_backend_port': windows_backend_port}
    process = None
    base = f'http://127.0.0.1:{port}'
    try:
        with (out / 'coordinator.log').open('w') as log:
            process = subprocess.Popen([str(swap.binary()), '-config', str(out / 'config.yaml'),
                                        '-listen', f'127.0.0.1:{port}'], stdout=log,
                                       stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError('llama-swap exited; inspect coordinator.log')
                try:
                    with urlopen(base + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('llama-swap did not become ready')
                time.sleep(0.1)
            listed = probes.request(base, '/v1/models')
            assert {m['id'] for m in listed['data']} == set(aliases.values()), listed
            assert not any(core.scope_active({'scope': s}) for s in scopes.values())
            sequence = (('bolt35', 'gyros', 'swift15', 'gyros', 'bolt35')
                        if set(names) == {'bolt35', 'swift15', 'gyros'} else names + names[:1])
            for name in sequence:
                started = time.monotonic()
                response = probes.request(base, '/v1/chat/completions', {
                    'model': aliases[name], 'messages': [{'role': 'user', 'content':
                    'What is 2 + 2? Answer with the number only.'}], 'temperature': 0,
                    **({'reasoning_effort': 'none'} if name == 'gyros' else {}),
                    'max_tokens': 64, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=1200)
                assert response['choices'][0]['message']['content'].strip() == '4', response
                active = [n for n, scope in scopes.items() if core.scope_active({'scope': scope})]
                assert active == [name], active
                limits = subprocess.check_output(['systemctl', '--user', 'show', scopes[name] + '.scope',
                    '-p', 'MemoryHigh', '-p', 'MemoryMax', '-p', 'MemorySwapMax'], text=True)
                assert 'MemoryHigh=8589934592' in limits and 'MemoryMax=9663676416' in limits
                assert 'MemorySwapMax=0' in limits
                record['switches'].append({'profile': name, 'request_seconds': time.monotonic() - started,
                                           'response': response, 'scope_limits': limits})
                if name == 'swift15':
                    record['vision'] = []
                    for color, rgb in [('red', b'\xff\x00\x00'), ('blue', b'\x00\x00\xff')]:
                        image = base64.b64encode(solid_png(rgb)).decode()
                        response = probes.request(base, '/v1/chat/completions', {
                            'model': aliases[name], 'messages': [{'role': 'user', 'content': [
                                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}},
                                {'type': 'text', 'text': 'What color is this image? Answer with one color word only.'}]}],
                            'max_tokens': 32, 'temperature': 0,
                            'chat_template_kwargs': {'enable_thinking': False}})
                        assert color in response['choices'][0]['message']['content'].lower(), response
                        record['vision'].append({'expected': color, 'response': response})
            if 'bolt35' in aliases:
                record['bolt_api'] = validate(base, aliases['bolt35'])
            record['checks_passed'] = True
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        try:
            errors = []
            for name, scope in scopes.items():
                try:
                    p = core.load(name)
                    p['scope'] = scope
                    if name == 'gyros':
                        from lib import strata
                        strata.stop(p)
                    else:
                        probes.stop(scope)
                        core.wait_windows_exit(p)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    errors.append(str(exc))
            record['cleanup_errors'] = errors
            if errors:
                raise RuntimeError('Trial scope cleanup failed: ' + '; '.join(errors))
            assert not any(core.scope_active({'scope': s}) for s in scopes.values())
            record['stable'] = record.get('checks_passed', False)
        finally:
            (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=1235)
    parser.add_argument('--backend-port', type=int, default=1236)
    parser.add_argument('--windows-backend-port', type=int, default=1257)
    parser.add_argument('--profiles', nargs='+', choices=core.PROFILES)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    run(args.port, args.backend_port, args.windows_backend_port, args.profiles)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
