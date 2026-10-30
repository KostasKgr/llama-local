import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lib import core, swap


class SwapTests(unittest.TestCase):
    def test_config_uses_launchers_aliases_and_one_backend_port(self):
        before = {name: core.command(core.load(name)) for name in swap.PROFILES}
        config = json.loads(swap.config())
        self.assertEqual(set(config['models']), {core.load(n)['alias'] for n in swap.PROFILES})
        for name in swap.PROFILES:
            entry = config['models'][core.load(name)['alias']]
            self.assertEqual(shlex.split(entry['cmd'])[-4:], ['start', name, '--port', '1236'])
            self.assertEqual(shlex.split(entry['cmdStop'])[-2:], ['stop', name])
            self.assertEqual(entry['proxy'], 'http://127.0.0.1:1236')
            self.assertIn('LLAMA_SCOPE=' + name, entry['env'])
            self.assertEqual(core.command(core.load(name)), before[name])
        self.assertEqual(config['groups']['local']['members'], list(config['models']))
        self.assertTrue(config['groups']['local']['swap'])
        self.assertEqual(config['globalTTL'], 0)
        self.assertNotIn('hooks', config)

    def test_windows_profiles_get_their_own_backend_port(self):
        config = json.loads(swap.config(1235, 1236, 1257))
        for name in swap.PROFILES:
            p = core.load(name)
            port = '1257' if p['platform'] == 'windows' else '1236'
            entry = config['models'][p['alias']]
            self.assertEqual(shlex.split(entry['cmd'])[-2:], ['--port', port])
            self.assertEqual(entry['proxy'], 'http://127.0.0.1:' + port)
        self.assertEqual(core.load('swift15')['platform'], 'windows')
        with self.assertRaises(ValueError):
            swap.config(1235, 1236, 1235)
        with patch('lib.swap.socket.socket') as sock, patch.object(core, 'preflight') as preflight:
            sock.return_value.__enter__.return_value.connect_ex.return_value = 1
            swap.preflight(1235, 1236, 1257)
            checked = [c.args[0] for c in sock.return_value.__enter__.return_value.connect_ex.call_args_list]
            self.assertEqual(checked, [('127.0.0.1', 1235), ('127.0.0.1', 1236), ('127.0.0.1', 1257)])
            self.assertEqual({c.args[0]['scope']: c.args[0]['port'] for c in preflight.call_args_list},
                             {'bolt35': 1236, 'swift15': 1257})

    def test_config_is_read_only_and_quotes_paths(self):
        with tempfile.TemporaryDirectory(prefix='swap path ') as temp:
            with patch.object(core, 'ROOT', Path(temp)), patch.object(core, 'load', side_effect=[
                {'alias': 'bolt'}, {'alias': 'swift'}]):
                config = json.loads(swap.config(18001, 18002))
                self.assertEqual(list(Path(temp).iterdir()), [])
                self.assertEqual(shlex.split(config['models']['bolt']['cmd'])[1],
                                 str(Path(temp) / 'llama.py'))
                self.assertEqual(config['models']['swift']['proxy'], 'http://127.0.0.1:18002')

    def test_invalid_ports(self):
        for public, backend in [(1235, 1235), (0, 1236), (1235, 65536)]:
            with self.assertRaises(ValueError):
                swap.config(public, backend)

    def test_preflight_refuses_occupied_port_before_inspecting_profiles(self):
        with patch('lib.swap.socket.socket') as sock, patch.object(core, 'preflight') as preflight:
            sock.return_value.__enter__.return_value.connect_ex.return_value = 0
            with self.assertRaisesRegex(ValueError, 'occupied'):
                swap.preflight(1235, 1236)
            preflight.assert_not_called()

    def test_lock_reserves_production_scopes_without_writing_on_read(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)):
            self.assertFalse(swap.managed())
            self.assertEqual(list(Path(temp).iterdir()), [])
            swap.directory().mkdir(parents=True)
            with (swap.directory() / 'instance.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(swap.managed())
                with patch.dict(os.environ, {'LLAMA_SWAP_MANAGED': ''}):
                    with self.assertRaisesRegex(ValueError, 'reserved'):
                        swap.guard({'scope': 'swift15'})
                    swap.guard({'scope': 'swift15-trial-test'})
                with patch.dict(os.environ, {'LLAMA_SWAP_MANAGED': '1'}):
                    swap.guard({'scope': 'swift15'})
            self.assertFalse(swap.managed())

    def test_start_refuses_active_instance_without_launching(self):
        with patch.object(swap, 'managed', return_value=True), patch('lib.swap.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError, 'already active'):
                swap.start()
            run.assert_not_called()

    def test_cleanup_requires_matching_preflight_marker(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)), \
                patch.object(swap, 'unit_state', return_value='deactivating'), \
                patch.object(core, 'load', side_effect=lambda name: {'scope': name}), \
                patch('lib.swap.subprocess.run') as run:
            swap.cleanup('mine')
            run.assert_not_called()
            swap.directory().mkdir(parents=True)
            swap.state_path().write_text(json.dumps({'token': 'mine'}))
            swap.cleanup('another-instance')
            run.assert_not_called()
            swap.cleanup('mine')
            self.assertEqual([c.args[0] for c in run.call_args_list], [
                ['systemctl', '--user', 'stop', 'bolt35.scope'],
                ['systemctl', '--user', 'stop', 'swift15.scope']])
            self.assertFalse(swap.state_path().exists())

    def test_cleanup_waits_for_windows_servers_to_exit(self):
        profiles = {'bolt35': {'scope': 'bolt35', 'platform': 'linux'},
                    'swift15': {'scope': 'swift15', 'platform': 'windows'}}
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)), \
                patch.object(swap, 'unit_state', return_value='active'), \
                patch.object(core, 'load', side_effect=profiles.get), \
                patch.object(core, 'windows_server_running', side_effect=[True, False]) as running, \
                patch('lib.core.time.sleep'), patch('lib.swap.subprocess.run'):
            swap.directory().mkdir(parents=True)
            swap.state_path().write_text(json.dumps({'token': 'mine'}))
            swap.cleanup('mine')
            self.assertEqual([c.args[0]['scope'] for c in running.call_args_list], ['swift15', 'swift15'])
            self.assertFalse(swap.state_path().exists())

    def test_failed_cleanup_retains_ownership_for_retry(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)), \
                patch.object(swap, 'unit_state', return_value='active'), \
                patch.object(core, 'load', return_value={}), \
                patch('lib.swap.subprocess.run', side_effect=subprocess.TimeoutExpired('stop', 40)):
            swap.directory().mkdir(parents=True)
            swap.state_path().write_text(json.dumps({'token': 'mine'}))
            with self.assertRaisesRegex(ValueError, 'cleanup failed'):
                swap.cleanup('mine')
            self.assertTrue(swap.state_path().exists())

    def test_service_has_scope_cleanup_after_abrupt_termination(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)), \
                patch.object(swap, 'managed', return_value=False), \
                patch.object(swap, 'service_state', return_value='inactive'), \
                patch.object(swap, 'binary', return_value=Path('/bin/true')), \
                patch.object(swap, 'preflight'), patch.object(swap, 'config', return_value='{}'), \
                patch.dict(os.environ, {'SWIFT_CONTEXT': '65536', 'BOLT35_SPEC_TYPE': 'none'}), \
                patch('lib.swap.subprocess.run') as run:
            swap.start()
            args = run.call_args.args[0]
            cleanup = next(a for a in args if a.startswith('ExecStopPost='))
            cleanup_args = shlex.split(cleanup.split('=', 1)[1])
            self.assertIn('_cleanup', cleanup_args)
            self.assertEqual(cleanup_args[-1], args[args.index('--token') + 1])
            self.assertIn('KillMode=control-group', args)
            self.assertIn('--setenv=SWIFT_CONTEXT=65536', args)
            self.assertIn('--setenv=BOLT35_SPEC_TYPE=none', args)
            configs = list(Path(temp).glob('logs/swap/*/config.yaml'))
            self.assertEqual(len(configs), 1)

    def test_serve_failure_before_preflight_never_claims_scopes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)), \
                patch.object(swap, 'preflight', side_effect=ValueError('busy')), \
                patch.object(swap, 'cleanup') as cleanup, patch.dict(os.environ):
            with self.assertRaisesRegex(ValueError, 'busy'):
                swap.serve('mine', temp, 1235, 1236)
            self.assertFalse(swap.state_path().exists())
            cleanup.assert_not_called()

    def test_serve_failure_after_claim_cleans_scopes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'ROOT', Path(temp)), \
                patch.object(swap, 'preflight'), \
                patch.object(swap, 'binary', return_value=Path('/bin/false')), \
                patch.object(swap, 'unit_state', return_value='inactive'), patch.object(core, 'wait_port'), \
                patch.object(core, 'load', return_value={}), patch.dict(os.environ):
            with self.assertRaises(subprocess.CalledProcessError):
                swap.serve('mine', temp, 1235, 1236)
            self.assertFalse(swap.state_path().exists())
            self.assertFalse(swap.managed())

    def test_live_smoke_refuses_busy_gpu_without_launching_or_stopping(self):
        from experiments import swap_smoke
        with patch.object(swap, 'managed', return_value=False), \
                patch.object(swap, 'service_state', return_value='inactive'), \
                patch('experiments.swap_smoke.probes.gpu', return_value=[8192]), \
                patch('experiments.swap_smoke.subprocess.Popen') as launch, \
                patch('experiments.swap_smoke.probes.stop') as stop:
            with self.assertRaisesRegex(ValueError, 'GPU is occupied'):
                swap_smoke.run()
            launch.assert_not_called()
            stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
