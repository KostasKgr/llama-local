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
    def check_install_archive(self, major):
        class Cuda:
            def cudaRuntimeGetVersion(self, version):
                version._obj.value = major * 1000 + 80
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
                library = tarfile.TarInfo(f'runtime/libcudart.so.{major}')
                library.size = 4
                tar.addfile(library, io.BytesIO(b'cuda'))
            p = {'runtime': 'prism-test'}
            with patch.object(core, 'ROOT', root), patch.object(core, 'load', return_value=p), \
                    patch.object(core, 'manifest', side_effect=[{'test': []}, {'prism-test': [{}]}]), \
                    patch.object(core, 'fetch', return_value=archive), \
                    patch('lib.core.subprocess.check_output', return_value='test'), \
                    patch('lib.core.ctypes.CDLL', return_value=Cuda()) as cdll:
                core.install('test')
            cdll.assert_called_once_with(str(root / f'runtimes/prism-test/libcudart.so.{major}'))
            self.assertEqual((root / 'runtimes/prism-test/llama-server').read_bytes(), b'server')
            info = json.loads((root / 'runtimes/prism-test/installation.json').read_text())
            self.assertEqual(info['cuda_runtime_version'], major * 1000 + 80)

    def test_install_archive_with_root_directory_entry(self):
        self.check_install_archive(12)

    def test_install_cuda13(self):
        self.check_install_archive(13)

    def test_bolt_dflash_and_disable(self):
        p = core.load('bolt35')
        args = core.command(p)
        self.assertIn('b11808-cuda13.4', args[0])
        for flag, value in {'-c': '65536', '--spec-type': 'draft-dflash',
                            '--spec-draft-n-max': '7', '-ngld': '999',
                            '--temp': '1.0', '--top-p': '0.95', '--top-k': '0'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(args[args.index('-md') + 1], str(core.ROOT / 'models' / p['draft']))
        args = core.command(core.load('bolt35', {'spec': 'none'}))
        for flag in ('-md', '-ngld', '--spec-draft-n-max'):
            self.assertNotIn(flag, args)
        with patch.dict('os.environ', {'BOLT35_CONTEXT': '32768', 'QWEN_CONTEXT': '16384'}):
            self.assertEqual(core.load('bolt35')['context'], 32768)

    def test_bolt_preflight_independent_artifacts(self):
        p = core.load('bolt35')
        available = {core.ROOT / 'models' / p[k] for k in ('model', 'draft')}
        with patch('lib.swap.guard'), patch('lib.core.os.access', return_value=True), \
                patch.object(Path, 'is_file', autospec=True, side_effect=lambda path: path in available), \
                patch('lib.core.socket.socket') as sock, patch.object(core, 'scope_active', return_value=False):
            sock.return_value.__enter__.return_value.connect_ex.return_value = 1
            core.preflight(p)
            for key in ('model', 'draft'):
                path = core.ROOT / 'models' / p[key]
                available.remove(path)
                with self.assertRaisesRegex(ValueError, 'artifact missing'):
                    core.preflight(p)
                available.add(path)
            available.remove(core.ROOT / 'models' / p['draft'])
            core.preflight(core.load('bolt35', {'spec': 'none'}))

    def test_qwen_shards_required_without_speculation(self):
        p = core.load('qwen36', {'spec': 'none'})
        with patch('lib.core.os.access', return_value=True), \
                patch.object(Path, 'is_file', autospec=True,
                             side_effect=lambda path: '-00002-of-' not in path.name):
            with self.assertRaisesRegex(ValueError, 'artifact missing'):
                core.preflight(p)

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
            ('swift15', 'b12001-cuda13.4', 'draft-mtp', 2048, 512, 'q8_0'),
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

    def test_projector_arguments_and_missing_artifact(self):
        p = core.load('swift15')
        args = core.command(p)
        projector = core.ROOT / 'models' / p['mmproj']
        self.assertEqual(args[args.index('--mmproj') + 1], str(projector))
        for name in ('qwen36', 'bonsai2', 'bonsai2mtp'):
            self.assertNotIn('--mmproj', core.command(core.load(name)))
        with patch('lib.swap.guard'), patch('lib.core.os.access', return_value=True), \
                patch.object(Path, 'is_file', autospec=True,
                             side_effect=lambda path: path != projector), \
                patch('lib.core.socket.socket') as sock, patch.object(core, 'scope_active') as active:
            with self.assertRaisesRegex(ValueError, 'artifact missing'):
                core.preflight(p)
            sock.assert_not_called()
            active.assert_not_called()

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
            with patch('lib.swap.guard'), patch('lib.core.os.access', return_value=True), \
                    patch.object(Path, 'is_file', return_value=True), patch('lib.core.scope_active') as active:
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
            if p.get('mmproj'):
                self.assertIn(p['mmproj'], files)

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
        with patch.dict('os.environ', {'BONSAI_CONTEXT': '16384', 'QWEN_CONTEXT': '65536'}):
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
