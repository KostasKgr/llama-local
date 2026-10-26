"""Optional transport test against the pinned binary; never loads real models."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch
from urllib.request import urlopen

from lib import core, probes, swap


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@unittest.skipUnless(swap.binary().is_file(), 'Run llama.py swap install for real proxy checks')
class SwapIntegrationTests(unittest.TestCase):
    def test_transient_service_abrupt_exit_runs_cleanup(self):
        manager = subprocess.run(['systemctl', '--user', 'show-environment'],
                                 capture_output=True, timeout=10)
        if manager.returncode:
            self.skipTest('User systemd manager unavailable')
        executable = swap.binary()
        repository = core.ROOT
        fixture = Path(__file__).parent / 'fixtures' / 'swap_backend.py'
        public, backend = free_port(), free_port()
        while backend == public:
            backend = free_port()
        config = json.loads(swap.config(public, backend))
        unit = 'llama-local-swap-test-' + uuid.uuid4().hex + '.service'
        with tempfile.TemporaryDirectory(prefix='swap-service-') as temp:
            root = Path(temp)
            (root / 'manifests').mkdir()
            (root / 'manifests' / 'llama-swap.json').write_text(json.dumps(swap.artifact()))
            target = root / 'runtimes' / executable.parent.name / 'llama-swap'
            target.parent.mkdir(parents=True)
            target.symlink_to(executable)
            for name, model in config['models'].items():
                cmd = [sys.executable, str(fixture)]
                model['cmd'] = shlex.join(cmd + ['start', name, str(backend), temp])
                model['cmdStop'] = shlex.join(cmd + ['stop', name, str(backend), temp])
                model['env'] = []
            # Exercise the real service wrapper with no real model scopes.
            (root / 'llama.py').write_text(
                'import sys\nfrom pathlib import Path\n'
                f'sys.path.insert(0, {str(repository)!r})\n'
                'from lib import core, swap\n'
                f'core.ROOT = Path({temp!r})\n'
                'swap.PROFILES = ()\nswap.preflight = lambda *args: None\n'
                'action = sys.argv[2]\n'
                'opts = dict(zip(sys.argv[3::2], sys.argv[4::2]))\n'
                'if action == "_cleanup": swap.cleanup(opts["--token"])\n'
                'else: swap.serve(opts["--token"], opts["--run-dir"], '
                'int(opts["--port"]), int(opts["--backend-port"]))\n')
            with patch.object(core, 'ROOT', root), patch.object(swap, 'UNIT', unit), \
                    patch.object(swap, 'preflight'), patch.object(swap, 'config', return_value=json.dumps(config)):
                try:
                    swap.start(public, backend)
                    deadline = time.monotonic() + 15
                    while True:
                        try:
                            with urlopen(f'http://127.0.0.1:{public}/health', timeout=1) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        if time.monotonic() > deadline:
                            logs = list(root.glob('logs/swap/*/coordinator.log'))
                            self.fail('Service did not start: ' + '\n'.join(p.read_text() for p in logs))
                        time.sleep(0.05)
                    self.assertTrue(swap.state_path().exists())
                    if os.access('/usr/lib/wsl/lib/nvidia-smi', os.X_OK):
                        logs = list(root.glob('logs/swap/*/coordinator.log'))
                        self.assertIn('using nvidia-smi for GPU monitoring',
                                      '\n'.join(p.read_text() for p in logs))
                    alias = next(iter(config['models']))
                    response = probes.request(f'http://127.0.0.1:{public}', '/v1/chat/completions', {
                        'model': alias, 'messages': [{'role': 'user', 'content': 'hello'}]}, timeout=20)
                    self.assertEqual(response['model'], alias)
                    subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=main',
                                    '--signal=KILL', unit], check=True, timeout=10)
                    deadline = time.monotonic() + 20
                    while swap.state_path().exists() or list(root.glob('*.pid')):
                        if time.monotonic() > deadline:
                            self.fail('Service failed to clean marker/backend after abrupt exit')
                        time.sleep(0.05)
                finally:
                    subprocess.run(['systemctl', '--user', 'stop', unit],
                                   capture_output=True, timeout=60)
                    for pidfile in root.glob('*.pid'):
                        subprocess.run([sys.executable, str(fixture), 'stop', pidfile.stem,
                                        str(backend), temp], check=True, timeout=15)

    def test_shared_port_sequential_switch_and_active_request_drain(self):
        public, backend = free_port(), free_port()
        while backend == public:
            backend = free_port()
        with tempfile.TemporaryDirectory(prefix='swap integration ') as temp:
            root = Path(temp)
            config = json.loads(swap.config(public, backend))
            names = list(config['models'])
            fixture = Path(__file__).parent / 'fixtures' / 'swap_backend.py'
            for name, model in config['models'].items():
                cmd = [sys.executable, str(fixture)]
                model['cmd'] = shlex.join(cmd + ['start', name, str(backend), temp])
                model['cmdStop'] = shlex.join(cmd + ['stop', name, str(backend), temp])
                model['env'] = []
            config['healthCheckTimeout'] = 15
            config['unloadTimeout'] = 15
            config_path = root / 'config.yaml'
            config_path.write_text(json.dumps(config))
            subprocess.run([str(swap.binary()), '-config', str(config_path), '-validate'],
                           check=True, capture_output=True, timeout=15)
            base = f'http://127.0.0.1:{public}'
            with (root / 'proxy.log').open('w+') as log:
                process = subprocess.Popen([str(swap.binary()), '-config', str(config_path),
                                            '-listen', f'127.0.0.1:{public}'],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    def healthy():
                        with urlopen(base + '/health', timeout=1) as response:
                            return response.status == 200
                    self.wait(healthy, process, log)
                    listed = probes.request(base, '/v1/models', timeout=5)
                    self.assertEqual({m['id'] for m in listed['data']}, set(names))
                    self.assertEqual(list(root.glob('*.pid')), [])

                    def request(name, content='hello'):
                        return probes.request(base, '/v1/chat/completions', {
                            'model': name, 'messages': [{'role': 'user', 'content': content}],
                            'chat_template_kwargs': {'enable_thinking': False},
                            'tools': [{'type': 'function', 'function': {'name': 'read',
                                       'parameters': {'type': 'object'}}}]}, timeout=30)

                    first = request(names[0])
                    self.assertEqual(first['model'], names[0])
                    self.assertEqual(first['received']['chat_template_kwargs'], {'enable_thinking': False})
                    self.assertEqual(first['received']['tools'][0]['function']['name'], 'read')
                    self.assertEqual(len(list(root.glob('*.pid'))), 1)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        slow = pool.submit(request, names[0], 'slow')
                        events = root / 'events.jsonl'
                        self.wait(lambda: 'slow-start' in events.read_text(), process, log)
                        switched = pool.submit(request, names[1])
                        self.assertEqual(slow.result(timeout=30)['model'], names[0])
                        self.assertEqual(switched.result(timeout=30)['model'], names[1])
                    image = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
                             {'type': 'text', 'text': 'image'}]
                    self.assertEqual(request(names[1], image)['received']['messages'][0]['content'], image)
                    self.assertEqual(request(names[0])['model'], names[0])
                    self.assertEqual(len(list(root.glob('*.pid'))), 1)
                    entries = [json.loads(line) for line in events.read_text().splitlines()]
                    transitions = [(e['model'], e['event']) for e in entries
                                   if e['event'] in ('start', 'stop')]
                    self.assertEqual(transitions, [(names[0], 'start'), (names[0], 'stop'),
                                                   (names[1], 'start'), (names[1], 'stop'),
                                                   (names[0], 'start')])
                    slow_end = next(i for i, e in enumerate(entries) if e['event'] == 'slow-end')
                    first_stop = next(i for i, e in enumerate(entries) if e['event'] == 'stop')
                    self.assertLess(slow_end, first_stop)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    remaining = list(root.glob('*.pid'))
                    # Only fixture-owned PIDs, even if the proxy crashed.
                    for pidfile in remaining:
                        subprocess.run([sys.executable, str(fixture), 'stop', pidfile.stem,
                                        str(backend), temp], check=True, timeout=15)
                self.assertEqual(remaining, [], 'Proxy shutdown left a backend running')
                self.assertEqual(list(root.glob('*.pid')), [])

    def wait(self, predicate, process, log):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                self.fail('Proxy exited: ' + log.read())
            try:
                if predicate():
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        log.seek(0)
        self.fail('Timed out: ' + log.read())
