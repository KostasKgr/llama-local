import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from lib import core, pi, strata, swap
from lib import strata_windows


class StrataTests(unittest.TestCase):
    def test_backend_overrides_and_no_llama_flags(self):
        p = core.load('gyros')
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)):
            cfg = strata.config(p, Path(temp) / 'runs' / 'test')
        self.assertEqual(p['context'], 131072)
        self.assertEqual(cfg['drafting'], 'dflash2')
        self.assertNotIn('--cache-type-k', cfg['args'])
        self.assertIn('--resident-experts', cfg['args'])
        self.assertNotIn('--mtp', cfg['args'])
        self.assertEqual(cfg['user_args'], ['--vram-reserve-mib', '1536', '--spec', '6', '--draft-conf', '0.35'])
        for override in ({'batch': 4096}, {'cache_k': 'f16'}, {'parallel': 2},
                         {'context': 262144}, {'spec': 'draft-mtp'}, {'draft_max': 1}, {'draft_max': 8}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                core.load('gyros', override)
        with patch.dict(os.environ, {'GYROS_CONTEXT': '65536', 'GYROS_SPEC_TYPE': 'none'}):
            self.assertEqual(core.load('gyros')['context'], 65536)
            self.assertEqual(core.load('gyros')['spec'], 'none')
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)):
            cfg = strata.config(core.load('gyros', {'spec': 'none'}), Path(temp))
        self.assertNotIn('--draft', cfg['args'])
        self.assertEqual(cfg['args'][cfg['args'].index('--spec') + 1], '2')
        self.assertEqual(cfg['args'][cfg['args'].index('--lookup-max-t') + 1], '0')

    def test_trial_pack_override_stays_in_managed_packs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)):
            cfg = strata.config(core.load('gyros'), Path(temp))
            self.assertTrue(cfg['args'][cfg['args'].index('--pack') + 1].endswith('packs\\gyros'))
            cfg = strata.config(core.load('gyros', {'pack': 'gyros-xbin'}), Path(temp))
            self.assertTrue(cfg['args'][cfg['args'].index('--pack') + 1].endswith('packs\\gyros-xbin'))
            self.assertTrue(cfg['tokenizer'].endswith('packs\\gyros-xbin\\tokenizer'))
        for value in ('../gyros', 'other', 'gyros/../x', 'GYROS', 'gyros-'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.load('gyros', {'pack': value})
        with self.assertRaises(ValueError):
            core.load('swift15', {'pack': 'gyros'})

    def test_start_refuses_a_pack_without_experts_bin(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)):
            p = core.load('gyros')
            for path in (strata.python(), strata.engine(), strata.engine().parent / 'engine-component.json',
                         Path(temp) / 'installation.json', strata.pack(p) / 'native_experts.txt', strata.pack(p) / 'tokenizer/vocab.json',
                         Path(temp) / 'models' / p['model']):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('x')
            with self.assertRaisesRegex(ValueError, 'experts.bin'):
                strata.preflight(p)

    def test_start_checks_generated_sizes_without_hashing(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)):
            p = core.load('gyros', {'spec': 'none'})
            for path in (strata.python(), strata.engine(), strata.engine().parent / 'engine-component.json',
                         strata.pack(p) / 'native_experts.txt', strata.pack(p) / 'tokenizer/vocab.json',
                         Path(temp) / 'models' / p['model']):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('x')
            (strata.pack(p) / 'experts.bin').write_bytes(b'1234')
            installation = {'manifest_sha256': 'm', 'model_manifest': core.manifest('models')['gyros'],
                            'generated': {'packs/gyros/experts.bin': {'size': 4, 'sha256': 'never'},
                                          'packs/gyros/native_experts.txt': {'size': 1, 'sha256': 'never'}}}
            (Path(temp) / 'installation.json').write_text(json.dumps(installation))
            hashed = []
            with patch.object(strata, 'digest', lambda path: hashed.append(Path(path).name) or 'm'), \
                    patch.object(strata, 'state') as state:
                state.return_value.exists.return_value = False
                with self.assertRaisesRegex(ValueError, 'Full Gyro GGUF size mismatch'):
                    strata.preflight(p)
                self.assertEqual(hashed, ['strata.json'])
                installation['generated']['packs/gyros/experts.bin']['size'] = 5
                (Path(temp) / 'installation.json').write_text(json.dumps(installation))
                with self.assertRaisesRegex(ValueError, 'Size mismatch; run verify gyros: packs/gyros/experts.bin'):
                    strata.preflight(p)

    def test_load_summary_reports_engine_phases_and_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'engine.log').write_text(
                'strata generate: memory guard: limit 31.4 GiB (source: meminfo), current 14.7 GiB, reclaimable cache '
                'credited 0.0 GiB; available 16.7 GiB (MemAvailable 16.7 GiB), floor 2560 MiB\n'
                'strata generate: 496 native projection matrices, 2573.44 MiB of weights, in 4.4 s\n'
                'strata generate: pre-filled 14681 of 14681 slots from the profile in 56.4 s (273 MB/s); slot 0 verified\n'
                'FileExpertSource: mapped pinned cache complement ready: resident 5.38 GiB, pinned 5.38 GiB in 6 s; '
                'page-locked and mapped\n')
            (run / 'disk.csv').write_text('time,read_mib_s,reads_s,read_ms,read_queue,page_reads_s\n'
                                          '99,900,1,1,1,1\n100,100,10,0.5,1,5\n101,300,30,1.5,2,9\n')
            summary = strata_windows.load_summary(run, 100, 80)
        self.assertEqual(summary['phases']['gpu_fill'], {'slots': 14681, 's': 56.4, 'mb_s': 273})
        self.assertEqual(summary['phases']['available_ram_gib'], 16.7)
        self.assertEqual(summary['unaccounted_s'], 13.2)
        self.assertEqual(summary['disk'], {'samples': 2, 'read_gib': 0.39, 'read_mib_s_mean': 200.0,
                                           'read_mib_s_max': 300.0, 'read_ms_max': 1.5})
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(strata_windows.load_summary(Path(temp), 0, 5),
                             {'load_s': 5, 'phases': {}, 'unaccounted_s': 5})

    def test_storage_does_not_discover_other_installations(self):
        for value in (r'C:\AI\strata', r'c:\AI\STRATA\data', r'\\host\share\data',
                      r'C:\AI\..\strata', 'C:\\', "C:\\a\\b'c"):
            with patch.object(core, 'load_env_file', return_value={'LLAMA_STRATA_WINDOWS_ROOT': value}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    strata.windows_root()
        with patch.object(core, 'load_env_file', return_value={'LLAMA_WINDOWS_ROOT': r'D:\unrelated'}):
            self.assertEqual(strata.windows_root(), strata.DEFAULT_ROOT)

    def test_unsafe_archives_never_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('../escape', '/absolute', 'C:/drive', 'a\\b', 'x/CON.txt', 'x/ends.'):
                archive = root / 'bad.zip'
                with zipfile.ZipFile(archive, 'w') as out:
                    out.writestr(name, 'bad')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    strata.extract(archive, root / 'runtime')
                self.assertFalse((root / 'runtime').exists())
            archive = root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as out:
                member = tarfile.TarInfo('link')
                member.type = tarfile.SYMTYPE
                member.linkname = '/outside'
                out.addfile(member)
            with self.assertRaises(ValueError):
                strata.extract(archive, root / 'runtime')

    def test_extraction_is_verified_against_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'test.zip'
            with zipfile.ZipFile(archive, 'w') as out:
                out.writestr('python/python.exe', 'runtime')
            strata.extract(archive, root / 'runtime')
            strata.verify_extraction(archive, root / 'runtime')
            (root / 'runtime/python/python.exe').write_text('changed')
            with self.assertRaises(ValueError):
                strata.verify_extraction(archive, root / 'runtime')
            with self.assertRaises(ValueError):
                strata.extract(archive, root / 'runtime')

    def test_identity_requires_creation_time_and_command(self):
        expected = {'pid': 12, 'created': 1, 'exe': 'python.exe', 'cmd': ['owned']}
        with patch.object(strata_windows, 'identity', return_value={**expected, 'created': 2}):
            self.assertFalse(strata_windows.same_process(expected))
        with patch.object(strata_windows, 'identity', return_value=expected):
            self.assertTrue(strata_windows.same_process(expected))

    def test_generated_inventory_cannot_escape_owned_preparation(self):
        for name in ('../outside', '/absolute', 'dflash2/../outside', r'dflash2\..\outside',
                     'runtimes/file', 'packs/other/file', 'C:/outside', 'dflash2//file'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                strata.generated_path(name)
        self.assertEqual(strata.generated_path('packs/gyros/index.txt'), strata.root() / 'packs/gyros/index.txt')

    def test_stop_absent_owner_does_not_signal(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)), \
                patch('lib.strata.subprocess.run') as run:
            strata.stop(core.load('gyros'))
            run.assert_not_called()

    def test_install_refuses_a_trial_or_pending_owner_before_touching_artifacts(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)), \
                patch.object(core, 'fetch') as fetch:
            pending = Path(temp) / 'state/gyros-trial-owned.json'
            pending.parent.mkdir()
            pending.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Stop the owned Gyro server'):
                strata.install()
            fetch.assert_not_called()

    def test_managed_trial_checks_other_backends_without_inheriting_its_scope(self):
        with patch.dict(os.environ, {'LLAMA_SCOPE': 'gyros-trial-owned', 'LLAMA_SWAP_MANAGED': '1'}), \
                patch.object(strata, 'preflight'), patch.object(core, 'wait_port'), \
                patch.object(core, 'wait_windows_exit') as wait, \
                patch('lib.strata.os.execvpe', side_effect=RuntimeError('exec reached')):
            p = core.load('gyros')
            with self.assertRaisesRegex(RuntimeError, 'exec reached'):
                strata.start(p)
        self.assertEqual([call.args[0]['scope'] for call in wait.call_args_list], ['bolt35', 'swift15'])

    def test_wrong_ownership_token_refused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(strata, 'root', return_value=Path(temp)):
            p = core.load('gyros')
            strata.state(p).parent.mkdir()
            strata.state(p).write_text(json.dumps({'token': '../../unrelated'}))
            with self.assertRaises(ValueError):
                strata.stop(p)

    def test_optional_swap_preserves_default_and_uses_one_exclusive_group(self):
        default = json.loads(swap.config())
        extended = json.loads(swap.config(1235, 1236, 1257, ['bolt35', 'swift15', 'gyros']))
        self.assertEqual(len(default['models']), 2)
        self.assertEqual(default['healthCheckTimeout'], 300)
        self.assertEqual(len(extended['models']), 3)
        self.assertEqual(extended['healthCheckTimeout'], 960)
        self.assertTrue(extended['groups']['local']['exclusive'])
        self.assertEqual({m['proxy'] for m in extended['models'].values()}, {'http://127.0.0.1:1257'})
        for names in ([], ['gyros', 'gyros'], ['unknown']):
            with self.assertRaises(ValueError):
                swap.config(profiles=names)

    def test_model_endpoint_override_preserves_shared_provider(self):
        source = {'providers': {'llama-local': {'baseUrl': 'https://shared.example/v1',
                   'api': 'custom', 'apiKey': '$KEEP', 'models': [{'id': 'custom', 'foo': 2}]}}}
        output = json.loads(pi.merge(json.dumps(source), core.load('gyros'),
                                    model_base_url='http://127.0.0.1:1235/v1'))
        provider = output['providers']['llama-local']
        self.assertEqual(provider['baseUrl'], source['providers']['llama-local']['baseUrl'])
        self.assertEqual(provider['api'], 'custom')
        self.assertEqual(provider['apiKey'], '$KEEP')
        model = provider['models'][-1]
        self.assertEqual(model['baseUrl'], 'http://127.0.0.1:1235/v1')
        self.assertEqual(model['api'], 'openai-completions')
        self.assertEqual(model['thinkingLevelMap']['high'], 'high')
        self.assertEqual(model['thinkingLevelMap']['xhigh'], None)
        self.assertEqual(pi.merge(json.dumps(output), core.load('gyros'),
                                  model_base_url=model['baseUrl']), json.dumps(output))


if __name__ == '__main__':
    unittest.main()
