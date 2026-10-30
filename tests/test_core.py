import json
import io
from pathlib import Path
import socket
import subprocess
import tempfile
import tarfile
import zipfile
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
            p = core.load('bolt35')
            command = ['bash', '-c', 'echo output; echo error >&2; exit 7']
            def execute(file, args, env):
                # Exercise the actual logging pipeline without starting a GPU server.
                result = subprocess.run(args[args.index('bash'):], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(result.stdout, 'output\nerror\n')
            with patch.object(core, 'ROOT', Path(temp)), patch.object(core, 'preflight'), \
                    patch.object(core, 'command', return_value=command), patch.object(core, 'wait_port'), \
                    patch('lib.core.os.execvpe', side_effect=execute):
                core.start(p)
                core.start(p)
            logs = list((Path(temp) / 'logs' / 'bolt35').glob('*/server.log'))
            self.assertEqual(len(logs), 2)
            for log in logs:
                self.assertEqual(log.read_text(), 'output\nerror\n')

    def test_preserved_profiles(self):
        for name, runtime, spec, batch, ubatch, kv in [
            ('swift15', 'win-b12001-cuda13.4', 'draft-mtp', 2048, 512, 'q8_0'),
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
        projector = core.artifact_root(p) / 'models' / p['mmproj']
        self.assertEqual(args[args.index('--mmproj') + 1], core.model_arg(p, p['mmproj']))
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

    def test_windows_profile_paths(self):
        p = core.load('swift15')
        self.assertEqual(p['platform'], 'windows')
        with patch.object(core, 'load_env_file', return_value={'LLAMA_WINDOWS_ROOT': 'D:\\Inference\\'}):
            args = core.command(p)
            self.assertEqual(args[0], '/mnt/d/Inference/runtimes/win-b12001-cuda13.4/llama-server.exe')
            self.assertEqual(args[args.index('-m') + 1],
                             'D:\\Inference\\models\\swift15\\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf')
            self.assertEqual(core.artifact_root(p), Path('/mnt/d/Inference'))
        for value in ('relative\\path', '/mnt/c/AI', 'C:'):
            with patch.object(core, 'load_env_file', return_value={'LLAMA_WINDOWS_ROOT': value}):
                with self.assertRaisesRegex(ValueError, 'LLAMA_WINDOWS_ROOT'):
                    core.windows_root()
        bolt = core.load('bolt35')
        self.assertEqual(bolt['platform'], 'linux')
        self.assertEqual(core.command(bolt)[0], str(core.ROOT / 'runtimes' / bolt['runtime'] / 'llama-server'))
        with patch('lib.core.json.loads', return_value={**core.json.loads(
                (core.ROOT / 'profiles' / 'bolt35.json').read_text()), 'platform': 'macos'}):
            with self.assertRaisesRegex(ValueError, 'platform'):
                core.load('bolt35')

    def check_windows_install(self, members):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'runtime.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name, data in members.items():
                    z.writestr(name, data)
            p = {'runtime': 'win-test', 'platform': 'windows', 'scope': 'swift15'}
            with patch.object(core, 'artifact_root', return_value=root), patch.object(core, 'load', return_value=p), \
                    patch.object(core, 'manifest', side_effect=[{'swift15': []}, {'win-test': [{}]}]), \
                    patch.object(core, 'fetch', return_value=archive), \
                    patch('lib.core.subprocess.check_output', return_value='test') as run:
                core.install('swift15')
            run.assert_called_with([str(root / 'runtimes/win-test/llama-server.exe'), '--list-devices'],
                                   cwd=root / 'runtimes/win-test', stderr=subprocess.STDOUT, text=True)
            return json.loads((root / 'runtimes/win-test/installation.json').read_text())

    def test_windows_install_flat_zip(self):
        info = self.check_windows_install({'llama-server.exe': b'server', 'cudart64_13.dll': b'cuda'})
        self.assertEqual((info['platform'], info['cuda_runtime_major']), ('windows', 13))
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.check_windows_install({'llama-server.exe': b'server'})
        for bad in ('../escape.dll', '/abs.dll', 'C:/drive.dll'):
            with self.assertRaisesRegex(ValueError, 'archive structure'):
                self.check_windows_install({'llama-server.exe': b'server', bad: b'x'})

    def test_windows_start_waits_for_port_and_runs_from_runtime(self):
        p = core.load('swift15')
        with patch.object(core, 'preflight'), patch.object(core, 'windows_port_free', side_effect=[False, True]), \
                patch('lib.core.time.sleep') as sleep, patch('lib.core.os.chdir') as chdir, \
                patch('lib.core.os.execvpe') as execute, tempfile.TemporaryDirectory() as temp, \
                patch.object(core, 'ROOT', Path(temp)):
            core.start(p)
        sleep.assert_called_once()
        chdir.assert_called_once_with(core.server_path(p).parent)
        self.assertNotIn(p['runtime'], execute.call_args.args[2].get('LD_LIBRARY_PATH', ''))
        with patch.object(core, 'windows_port_free', return_value=False), patch('lib.core.time.sleep'), \
                patch('lib.core.time.monotonic', side_effect=[0, 0, 1000]):
            with self.assertRaisesRegex(ValueError, 'cannot bind'):
                core.wait_port(1257, core.windows_port_free)

    def test_windows_stop_waits_for_server_exit(self):
        p = core.load('swift15')
        with patch('lib.swap.guard'), patch('lib.core.subprocess.run') as run, \
                patch.object(core, 'windows_server_running', side_effect=[True, False]) as running, \
                patch('lib.core.time.sleep'):
            core.stop(p)
        run.assert_called_once()
        self.assertEqual(running.call_count, 2)
        with patch('lib.swap.guard'), patch('lib.core.subprocess.run'), \
                patch.object(core, 'windows_server_running') as running:
            core.stop(core.load('bolt35'))
        running.assert_not_called()

    def test_linux_port_wait(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            port = sock.getsockname()[1]
            self.assertFalse(core.linux_port_free(port))
            with patch('lib.core.time.sleep'), patch('lib.core.time.monotonic', side_effect=[0, 1000]):
                with self.assertRaisesRegex(ValueError, 'Linux cannot bind'):
                    core.wait_port(port)
        self.assertTrue(core.linux_port_free(port))

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
