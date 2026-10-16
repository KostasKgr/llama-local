import json
import io
from pathlib import Path
import socket
import subprocess
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from lib import core


class CoreTests(unittest.TestCase):
    def test_install_archive_with_root_directory_entry(self):
        class Cuda:
            def cudaRuntimeGetVersion(self, version):
                version._obj.value = 12080
                return 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'runtime.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                directory = tarfile.TarInfo('.')
                directory.type = tarfile.DIRTYPE
                tar.addfile(directory)
                server = tarfile.TarInfo('runtime/llama-server')
                server.size = 6
                tar.addfile(server, io.BytesIO(b'server'))
            p = {'runtime': 'prism-test'}
            with patch.object(core, 'ROOT', root), patch.object(core, 'load', return_value=p), \
                    patch.object(core, 'manifest', side_effect=[{'test': []}, {'prism-test': [{}]}]), \
                    patch.object(core, 'fetch', return_value=archive), \
                    patch('lib.core.subprocess.check_output', return_value='test'), \
                    patch('lib.core.ctypes.CDLL', return_value=Cuda()):
                core.install('test')
            self.assertEqual((root / 'runtimes/prism-test/llama-server').read_bytes(), b'server')

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
        for name in core.PROFILES:
            p = core.load(name)
            files = {a['file'] for a in core.manifest('models')[name]}
            self.assertIn(p['model'], files)
            if 'draft' in p:
                self.assertIn(p['draft'], files)

    def test_lfm_without_draft_and_independent_overrides(self):
        with patch.dict('os.environ', {'LFM_CONTEXT': '36864', 'QWEN_CONTEXT': '1024'}):
            p = core.load('lfm25')
            self.assertEqual(p['context'], 36864)
            self.assertEqual(core.load('lfm25', {'context': 98304})['context'], 98304)
        args = core.command(p)
        self.assertEqual(args[args.index('--spec-type') + 1], 'none')
        self.assertNotIn('-md', args)
        self.assertNotIn('--spec-draft-n-max', args)
        self.assertEqual(args[args.index('--temp') + 1], '0.1')
        self.assertEqual(args[args.index('--top-k') + 1], '50')
        self.assertEqual(args[args.index('--repeat-penalty') + 1], '1.05')
        for spec in ('draft-mtp', 'draft-dflash'):
            with self.assertRaises(ValueError):
                core.load('lfm25', {'spec': spec})
        with patch.dict('os.environ', {'LLAMA_SCOPE': 'lfm25-trial-abc123'}):
            self.assertEqual(core.load('lfm25')['scope'], 'lfm25-trial-abc123')

    def test_bonsai_sampling_and_no_speculation(self):
        p = core.load('bonsai2')
        args = core.command(p)
        for flag, value in {'--spec-type': 'none', '--temp': '1.0', '--top-p': '0.95',
                            '--top-k': '20', '--min-p': '0.0', '--presence-penalty': '1.0',
                            '--repeat-penalty': '1.0', '--repeat-last-n': '64',
                            '-c': '131072', '--cache-type-k': 'f16', '--cache-type-v': 'f16'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn('-md', args)
        self.assertNotIn('--spec-draft-n-max', args)
        for spec in ('draft-mtp', 'draft-dflash'):
            with self.assertRaises(ValueError):
                core.load('bonsai2', {'spec': spec})
        with patch.dict('os.environ', {'BONSAI_CONTEXT': '16384', 'LFM_CONTEXT': '65536'}):
            self.assertEqual(core.load('bonsai2')['context'], 16384)

    def test_bonsai_mtp_embedded_head_and_disable(self):
        p = core.load('bonsai2mtp')
        args = core.command(p)
        self.assertIn('prism-b10777-e3de715-cuda12.8', args[0])
        for flag, value in {'--spec-type': 'draft-mtp', '--spec-draft-n-max': '2',
                            '--spec-draft-n-min': '0', '--spec-draft-p-min': '0.0',
                            '-c': '131072'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn('-md', args)
        args = core.command(core.load('bonsai2mtp', {'spec': 'none'}))
        for flag in ('--spec-draft-n-max', '--spec-draft-n-min', '--spec-draft-p-min'):
            self.assertNotIn(flag, args)
        with self.assertRaises(ValueError):
            core.load('bonsai2mtp', {'spec': 'draft-dflash'})
        with patch.dict('os.environ', {'BONSAI_MTP_CONTEXT': '32768', 'BONSAI_CONTEXT': '16384'}):
            self.assertEqual(core.load('bonsai2mtp')['context'], 32768)


if __name__ == '__main__':
    unittest.main()
