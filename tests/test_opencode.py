import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from lib import core, jsonc, opencode


class OpenCodeTests(unittest.TestCase):
    def setUp(self):
        self.profile = core.load('bolt35')

    def test_comments_trailing_commas_and_unrelated_bytes(self):
        source = '''{
  // url is not a comment
  "other": {"url": "http://example.test/*literal*/",},
  "providers": {"llama.cpp": {
    "package": "@opencode/ai/providers/openai-compatible",
    "settings": {"baseURL": "http://127.0.0.1:1235/v1"},
    "models": {"existing": {"limit": {"context": 100,},}, /* keep */},
  },},
}
'''
        result = opencode.merge(source, self.profile)
        self.assertIn('"other": {"url": "http://example.test/*literal*/",}', result)
        self.assertIn('/* keep */', result)
        self.assertIn('"existing": {"limit": {"context": 100,},}', result)
        self.assertEqual(opencode.merge(result, self.profile), result)
        self.assertIn(self.profile['alias'], jsonc.parse(result).value['providers']['llama.cpp']['models'])

    def test_missing_provider_and_existing_provider_name(self):
        result = opencode.merge('{}', self.profile)
        self.assertEqual(jsonc.parse(result).value['providers']['llama.cpp']['name'], 'llama.cpp (WSL)')
        source = '{"providers":{"llama.cpp":{"name":"Custom"}}}'
        result = opencode.merge(source, self.profile)
        self.assertEqual(jsonc.parse(result).value['providers']['llama.cpp']['name'], 'Custom')

    def test_conflicts_update_preserves_custom_fields(self):
        source = opencode.merge('{}', self.profile)
        path = ['providers', 'llama.cpp', 'models', self.profile['alias']]
        source = jsonc.set_value(source, path + ['limit', 'context'], 32000)
        source = jsonc.set_value(source, path + ['custom'], {'temperature': 0.2})
        with self.assertRaisesRegex(ValueError, '--update'):
            opencode.merge(source, self.profile)
        updated = opencode.merge(source, self.profile, update=True)
        model = jsonc.parse(updated).value['providers']['llama.cpp']['models'][self.profile['alias']]
        self.assertEqual(model['limit']['context'], 65536)
        self.assertEqual(model['custom'], {'temperature': 0.2})


    def test_swift_default_output_limit(self):
        self.assertEqual(opencode.model_config(core.load('swift15'))['limit']['output'], 16000)


    def test_bolt_registration_preserves_existing_models(self):
        source = '{}'
        for name in ('swift15',):
            source = opencode.merge(source, core.load(name))
        before = jsonc.parse(source).value['providers']['llama.cpp']['models']
        p = core.load('bolt35')
        result = opencode.merge(source, p)
        models = jsonc.parse(result).value['providers']['llama.cpp']['models']
        for alias, model in before.items():
            self.assertEqual(models[alias], model)
        self.assertEqual(models[p['alias']]['limit'], {'context': 65536, 'output': 8192})
        self.assertEqual(models[p['alias']]['capabilities'],
                         {'tools': True, 'input': ['text'], 'output': ['text']})
        self.assertEqual(models[p['alias']]['variants'], [])
        self.assertEqual(opencode.merge(result, p), result)

    def test_swift_options_and_override_precedence(self):
        p = core.load('swift15', {'context': 65536})
        result = opencode.merge('{}', p, output_tokens=4096)
        model = jsonc.parse(result).value['providers']['llama.cpp']['models'][p['alias']]
        self.assertEqual(model['settings'], {'reasoningEffort': 'xhigh'})
        self.assertEqual({v['id'] for v in model['variants']}, {'none', 'low', 'medium', 'xhigh'})
        self.assertEqual(model['limit'], {'context': 65536, 'output': 4096})
        self.assertEqual(model['capabilities']['input'], ['text', 'image'])
        with patch.dict(os.environ, {'BOLT35_CONTEXT': '32768', 'BOLT35_PORT': '5555'}):
            p = core.load('bolt35', {'port': 1235})
            result = opencode.merge('{}', p)
            provider = jsonc.parse(result).value['providers']['llama.cpp']
            self.assertEqual(provider['settings']['baseURL'], 'http://127.0.0.1:1235/v1')
            self.assertEqual(provider['models'][p['alias']]['limit']['context'], 32768)

    def test_swift_vision_updates_existing_text_registration(self):
        p = core.load('swift15')
        source = opencode.merge('{}', self.profile)
        source = opencode.merge(source, p)
        path = ['providers', 'llama.cpp', 'models', p['alias']]
        source = jsonc.set_value(source, path + ['capabilities', 'input'], ['text'])
        before = jsonc.parse(source).value['providers']['llama.cpp']['models']
        with self.assertRaisesRegex(ValueError, 'capabilities.input'):
            opencode.merge(source, p)
        updated = opencode.merge(source, p, update=True)
        models = jsonc.parse(updated).value['providers']['llama.cpp']['models']
        self.assertEqual(models[self.profile['alias']], before[self.profile['alias']])
        expected = before[p['alias']]
        expected['capabilities']['input'] = ['text', 'image']
        self.assertEqual(models[p['alias']], expected)
        self.assertEqual(opencode.merge(updated, p), updated)

    def test_shared_endpoint_conflict(self):
        source = opencode.merge('{}', self.profile)
        p = core.load('bolt35', {'port': 4321})
        with self.assertRaisesRegex(ValueError, 'baseURL'):
            opencode.merge(source, p)
        updated = opencode.merge(source, p, update=True)
        self.assertEqual(jsonc.parse(updated).value['providers']['llama.cpp']['settings']['baseURL'], 'http://127.0.0.1:4321/v1')

    def test_invalid_jsonc_and_duplicate_keys(self):
        for source in ['{"a":1,"a":2}', '{/*', '{"a":NaN}', '{"a":}', '[]', '{} {}']:
            with self.assertRaises(ValueError):
                jsonc.parse(source)

    def test_escaped_strings_and_arrays(self):
        source = '{"a": ["escaped\\\" // text", {"x": true,},], /* block */}'
        result = jsonc.set_value(source, ['new'], 1)
        self.assertEqual(jsonc.parse(result).value['a'], jsonc.parse(source).value['a'])

    def test_set_value_pretty_prints_container_replacements(self):
        source = '{\n  "a": {"x": [1, 2, 3]},\n  "b": 1\n}\n'
        big = [{'id': f'm{i}', 'settings': {'temperature': i / 10}} for i in range(10)]
        result = jsonc.set_value(source, ['a', 'x'], big)
        self.assertLess(max(len(line) for line in result.splitlines()), 100)
        self.assertEqual(jsonc.parse(result).value['a']['x'], big)
        self.assertIn('\n  "b": 1\n', result)
        # Scalars stay compact and unrelated bytes are preserved.
        self.assertEqual(jsonc.set_value(source, ['b'], 2),
                         '{\n  "a": {"x": [1, 2, 3]},\n  "b": 2\n}\n')

    def test_dry_run_backup_atomic_and_noop(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch('lib.core.load_env_file', return_value={}):
            path = Path(temp) / 'opencode.jsonc'
            source = b'{"custom":true}\n'
            path.write_bytes(source)
            path.chmod(0o640)
            opencode.add('bolt35', path, dry_run=True)
            self.assertEqual(path.read_bytes(), source)
            self.assertEqual(list(path.parent.glob('*.backup-*')), [])
            opencode.add('bolt35', path)
            backups = list(path.parent.glob('*.backup-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), source)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            updated = path.read_bytes()
            opencode.add('bolt35', path)
            self.assertEqual(path.read_bytes(), updated)
            self.assertEqual(list(path.parent.glob('*.backup-*')), backups)

    def test_concurrent_change_and_failed_replace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.jsonc'
            path.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                opencode.write_config(path, b'{ }', '{"a":1}')
            with patch('lib.opencode.os.replace', side_effect=OSError('simulated failure')):
                with self.assertRaises(OSError):
                    opencode.write_config(path, b'{}', '{"a":1}')
            self.assertEqual(path.read_bytes(), b'{}')
            self.assertEqual(list(path.parent.glob('.config*')), [])

    def test_missing_config_and_output_limits(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch('lib.core.load_env_file', return_value={}):
            path = Path(temp) / 'nested/config.jsonc'
            opencode.add('bolt35', path, dry_run=True)
            self.assertFalse(path.parent.exists())
            opencode.add('bolt35', path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        for output in [0, -1, 131073]:
            with self.assertRaises(ValueError):
                opencode.model_config(self.profile, output)


class OpenCodeEnvBaseUrlTests(unittest.TestCase):
    def test_resolve_base_url_with_opencode_key(self):
        profile = core.load('bolt35')
        self.assertEqual(core.resolve_base_url(profile, opencode.ENV_KEY, {}),
                         'http://127.0.0.1:1235/v1')
        self.assertEqual(core.resolve_base_url(profile, opencode.ENV_KEY,
                                               {'LLAMA_OPENCODE_BASE_URL': 'https://a.example/v1/'}),
                         'https://a.example/v1')
        for bad in ('relative/v1', 'a.example/v1', 'ftp://a.example/v1'):
            with self.subTest(bad), self.assertRaises(ValueError):
                core.resolve_base_url(profile, opencode.ENV_KEY, {'LLAMA_OPENCODE_BASE_URL': bad})

    def test_merge_with_configured_base_url(self):
        profile = core.load('bolt35')
        base = core.resolve_base_url(profile, opencode.ENV_KEY,
                                     {'LLAMA_OPENCODE_BASE_URL': 'https://models.example/v1'})
        result = opencode.merge('{}', profile, base_url=base)
        provider = jsonc.parse(result).value['providers']['llama.cpp']
        self.assertEqual(provider['settings']['baseURL'], 'https://models.example/v1')
        # The same configured URL is a no-op, not a conflict.
        self.assertEqual(opencode.merge(result, profile, base_url=base), result)
        with self.assertRaisesRegex(ValueError, 'baseURL'):
            opencode.merge(result, profile, base_url='https://other.example/v1')
        updated = opencode.merge(result, profile, base_url='https://other.example/v1', update=True)
        self.assertEqual(jsonc.parse(updated).value['providers']['llama.cpp']['settings']['baseURL'],
                         'https://other.example/v1')

    def test_add_reads_env_file(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            path = Path(temp) / 'opencode.jsonc'
            with patch('lib.core.load_env_file',
                       return_value={'LLAMA_OPENCODE_BASE_URL': 'https://env.example/v1'}):
                opencode.add('bolt35', path)
            config = jsonc.parse(path.read_text()).value
            self.assertEqual(config['providers']['llama.cpp']['settings']['baseURL'],
                             'https://env.example/v1')
            # Without the configured URL, the local default conflicts with it.
            with patch('lib.core.load_env_file', return_value={}):
                with self.assertRaisesRegex(ValueError, 'baseURL'):
                    opencode.add('bolt35', path)
