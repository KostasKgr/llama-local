import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from lib import core


class CoreTests(unittest.TestCase):
    def test_start_captures_both_streams_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            p = core.load('swift15')
            command = ['bash', '-c', 'echo output; echo error >&2; exit 7']
            def execute(file, args, env):
                # Exercise the actual logging pipeline without starting a GPU server.
                result = subprocess.run(args[args.index('bash'):], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(result.stdout, 'output\nerror\n')
            with patch.object(core, 'ROOT', Path(temp)), patch.object(core, 'preflight'), \
                    patch.object(core, 'command', return_value=command), \
                    patch('lib.core.os.execvpe', side_effect=execute):
                core.start(p)
                core.start(p)
            logs = list((Path(temp) / 'logs' / 'swift15').glob('*/server.log'))
            self.assertEqual(len(logs), 2)
            for log in logs:
                self.assertEqual(log.read_text(), 'output\nerror\n')

    def test_preserved_profiles(self):
        for name, runtime, spec, batch, ubatch, kv in [
            ('swift15', 'b11393-cuda12.8', 'draft-mtp', 2048, 512, 'q8_0'),
            ('qwen36', 'b11433-cuda12.8', 'draft-dflash', 4096, 2048, 'f16')]:
            p = core.load(name)
            self.assertEqual((p['runtime'], p['spec'], p['batch'], p['ubatch'], p['cache_k']),
                             (runtime, spec, batch, ubatch, kv))
            args = core.command(p)
            self.assertIn('131072', args)
            self.assertEqual('-md' in args, name == 'qwen36')
            self.assertEqual('--fit' in args, name == 'qwen36')

    def test_owned_scope_validation(self):
        with patch.dict('os.environ', {'LLAMA_SCOPE': 'unrelated'}):
            with self.assertRaises(ValueError):
                core.load('swift15')
        with patch.dict('os.environ', {'LLAMA_SCOPE': 'swift15-trial-abc123'}):
            self.assertEqual(core.load('swift15')['scope'], 'swift15-trial-abc123')

    def test_invalid_overrides(self):
        for override in [{'ubatch': 8192}, {'port': 0}, {'context': -1}, {'spec': 'draft-dflash'}]:
            with self.assertRaises(ValueError):
                core.load('swift15', override)

    def test_failed_download_never_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            def download(*args, **kwargs):
                (base / 'model.part').write_bytes(b'bad')
            with patch('lib.core.subprocess.run', side_effect=download):
                with self.assertRaises(ValueError):
                    core.fetch(base, {'file': 'model', 'url': 'https://example.invalid/model', 'sha256': '0' * 64})
            self.assertFalse((base / 'model').exists())
            self.assertTrue((base / 'model.part').exists())

    def test_port_conflict_before_scope_checks(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            p = core.load('swift15', {'port': sock.getsockname()[1]})
            with patch('lib.core.os.access', return_value=True), patch.object(Path, 'is_file', return_value=True), patch('lib.core.scope_active') as active:
                with self.assertRaisesRegex(ValueError, 'occupied'):
                    core.preflight(p)
                active.assert_not_called()

    def test_manifests_include_all_model_files(self):
        for name in ('swift15', 'qwen36'):
            p = core.load(name)
            files = {a['file'] for a in core.manifest('models')[name]}
            self.assertIn(p['model'], files)
            if 'draft' in p:
                self.assertIn(p['draft'], files)


if __name__ == '__main__':
    unittest.main()
