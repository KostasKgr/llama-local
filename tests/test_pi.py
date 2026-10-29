import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from lib import core, pi
from experiments.pi_smoke import check_case


class PiRegistrationTests(unittest.TestCase):
    def test_all_profiles_preserve_existing_providers_and_models(self):
        source = '{"providers":{"other":{"models":[{"id":"custom"}]}},"custom":true}\n'
        previous = {}
        for name in core.PROFILES:
            profile = core.load(name)
            result = pi.merge(source, profile)
            parsed = json.loads(result)
            self.assertEqual(parsed['providers']['other'], {'models': [{'id': 'custom'}]})
            self.assertTrue(parsed['custom'])
            models = {m['id']: m for m in parsed['providers'][pi.PROVIDER]['models']}
            for alias, model in previous.items():
                self.assertEqual(models[alias], model)
            model = models[profile['alias']]
            self.assertEqual(model['contextWindow'], profile['context'])
            self.assertEqual(model['maxTokens'], 8192)
            self.assertEqual(model['input'], profile['opencode']['modalities']['input'])
            self.assertEqual(pi.merge(result, profile), result)
            previous = copy.deepcopy(models)
            source = result

    def test_models_array_pretty_printed(self):
        result = pi.merge('{}\n', core.load('bolt35'))
        result = pi.merge(result, core.load('swift15'))
        models_line = next(line for line in result.splitlines() if '"models":' in line)
        self.assertTrue(models_line.rstrip().endswith('['))
        self.assertLess(max(len(line) for line in result.splitlines()), 100)
        self.assertEqual(len(json.loads(result)['providers'][pi.PROVIDER]['models']), 2)
        # A compact existing array is reprinted pretty when a model is added;
        # only the untouched preamble of the compact input keeps its line.
        updated = pi.merge(json.dumps(json.loads(result)), core.load('qwen36'))
        lines = updated.splitlines()
        self.assertTrue(lines[0].rstrip().endswith('"models": ['))
        self.assertLess(max(len(line) for line in lines[1:]), 100)
        self.assertEqual(len(json.loads(updated)['providers'][pi.PROVIDER]['models']), 3)

    def test_conflicts_and_update_preserve_custom_metadata(self):
        profile = core.load('bolt35')
        config = json.loads(pi.merge('{}', profile))
        provider = config['providers'][pi.PROVIDER]
        provider['apiKey'] = '$CUSTOM_KEY'
        provider['headers'] = {'X-Custom': 'keep'}
        model = provider['models'][0]
        model['maxTokens'] = 512
        model['custom'] = {'keep': True}
        model['compat']['requiresToolResultName'] = True
        source = json.dumps(config)
        with self.assertRaisesRegex(ValueError, '--update'):
            pi.merge(source, profile)
        result = json.loads(pi.merge(source, profile, update=True))['providers'][pi.PROVIDER]
        self.assertEqual(result['apiKey'], '$CUSTOM_KEY')
        self.assertEqual(result['headers'], provider['headers'])
        self.assertEqual(result['models'][0]['custom'], {'keep': True})
        self.assertTrue(result['models'][0]['compat']['requiresToolResultName'])
        self.assertEqual(result['models'][0]['maxTokens'], 8192)

    def test_shared_endpoint_conflict_and_overrides(self):
        with patch.dict(os.environ, {'BOLT35_CONTEXT': '32768', 'BOLT35_PORT': '5555'}):
            profile = core.load('bolt35', {'port': 4321})
        result = pi.merge('{}', profile, 4096)
        provider = json.loads(result)['providers'][pi.PROVIDER]
        self.assertEqual(provider['baseUrl'], 'http://127.0.0.1:4321/v1')
        self.assertEqual(provider['models'][0]['contextWindow'], 32768)
        self.assertEqual(provider['models'][0]['maxTokens'], 4096)
        with self.assertRaisesRegex(ValueError, 'baseUrl'):
            pi.merge(result, core.load('bolt35'))

    def test_binary_thinking_without_opencode_effort_presets(self):
        model = pi.model_config(core.load('bolt35'))
        self.assertEqual(model['compat']['chatTemplateKwargs'],
                         {'enable_thinking': {'$var': 'thinking.enabled'}})
        self.assertFalse(model['compat']['supportsReasoningEffort'])
        self.assertIsNone(model['thinkingLevelMap']['xhigh'])
        self.assertNotIn('settings', model)

    def test_swift_efforts_match_opencode_and_update_binary_registration(self):
        profile = core.load('swift15')
        model = pi.model_config(profile)
        self.assertEqual(model['thinkingLevelMap'], {
            'off': 'none', 'minimal': None, 'low': 'low', 'medium': 'medium',
            'high': None, 'xhigh': 'xhigh', 'max': None,
        })
        self.assertTrue(model['compat']['supportsReasoningEffort'])
        self.assertEqual(model['compat']['thinkingFormat'], 'openai')
        old = copy.deepcopy(model)
        old['compat'].update(supportsReasoningEffort=False, thinkingFormat='chat-template',
                             chatTemplateKwargs={'enable_thinking': {'$var': 'thinking.enabled'}})
        old['thinkingLevelMap'].update(low=None, medium=None, high='high', xhigh=None)
        source = json.dumps({'providers': {pi.PROVIDER: {'models': [old]}}})
        with self.assertRaisesRegex(ValueError, '--update'):
            pi.merge(source, profile)
        updated = json.loads(pi.merge(source, profile, update=True))
        actual = updated['providers'][pi.PROVIDER]['models'][0]
        self.assertEqual(actual['thinkingLevelMap'], model['thinkingLevelMap'])
        self.assertEqual(actual['compat']['thinkingFormat'], 'openai')

    def test_model_transport_overrides_cannot_silently_redirect_registration(self):
        profile = core.load('bolt35')
        config = json.loads(pi.merge('{}', profile))
        model = config['providers'][pi.PROVIDER]['models'][0]
        model.update(api='openai-responses', baseUrl='http://other.invalid/v1')
        with self.assertRaisesRegex(ValueError, '--update'):
            pi.merge(json.dumps(config), profile)
        result = json.loads(pi.merge(json.dumps(config), profile, update=True))
        updated = result['providers'][pi.PROVIDER]['models'][0]
        self.assertEqual(updated['api'], 'openai-completions')
        self.assertEqual(updated['baseUrl'], 'http://127.0.0.1:1235/v1')

    def test_invalid_configs_and_duplicate_model_ids(self):
        profile = core.load('bolt35')
        bad = ['[]', '{"providers":[]}', '{"providers":{"llama-local":[]}}',
               '{"a":1,"a":2}', '{"a":NaN}', '{/* comment */}', '{"a":1,}',
               '{"providers":{"llama-local":{"models":{}}}}',
               '{"providers":{"llama-local":{"models":[{}]}}}',
               '{"providers":{"llama-local":{"models":[{"id":"x"},{"id":"x"}]}}}']
        for source in bad:
            with self.subTest(source=source), self.assertRaises(ValueError):
                pi.merge(source, profile)
        for output in (0, -1, 65537, True):
            with self.assertRaises(ValueError):
                pi.model_config(profile, output)

    def test_dry_run_backup_permissions_and_noop(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch('lib.core.load_env_file', return_value={}):
            path = Path(temp) / 'nested/models.json'
            pi.add('bolt35', path, dry_run=True)
            self.assertFalse(path.parent.exists())
            pi.add('bolt35', path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(path.parent.glob('*.backup-*')), [])
            original = path.read_bytes()
            path.chmod(0o640)
            pi.add('swift15', path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            backups = list(path.parent.glob('*.backup-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            after = path.read_bytes()
            pi.add('swift15', path)
            self.assertEqual(path.read_bytes(), after)
            self.assertEqual(list(path.parent.glob('*.backup-*')), backups)

    def test_concurrent_change_and_failed_replace(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch('lib.core.load_env_file', return_value={}):
            path = Path(temp) / 'models.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                pi.write_config(path, b'{ }', '{}')
            with patch('lib.opencode.os.replace', side_effect=OSError('simulated failure')):
                with self.assertRaises(OSError):
                    pi.add('bolt35', path)
            self.assertEqual(path.read_bytes(), b'{}')
            self.assertEqual(list(path.parent.glob('.models*')), [])

    def test_cli_honors_isolated_agent_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PI_CODING_AGENT_DIR=temp)
            command = [sys.executable, str(core.ROOT / 'llama.py'), 'pi', 'add', 'bolt35']
            result = subprocess.run(command + ['--dry-run'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temp).iterdir()), [])
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((Path(temp) / 'models.json').read_text())
            self.assertEqual(config['providers'][pi.PROVIDER]['models'][0]['id'], core.load('bolt35')['alias'])


class PiEnvBaseUrlTests(unittest.TestCase):
    def test_env_file_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '.env'
            self.assertEqual(core.load_env_file(path), {})
            path.write_text(
                '# comment\n'
                '\n'
                'LLAMA_PI_BASE_URL = https://first.example/v1\n'
                'OTHER=value\n'
                'not a pair\n'
                'LLAMA_PI_BASE_URL=https://second.example/v1/\n')
            self.assertEqual(core.load_env_file(path),
                             {'LLAMA_PI_BASE_URL': 'https://second.example/v1/', 'OTHER': 'value'})

    def test_resolve_base_url_fallback_and_validation(self):
        profile = core.load('bolt35')
        self.assertEqual(core.resolve_base_url(profile, pi.ENV_KEY, {}), 'http://127.0.0.1:1235/v1')
        self.assertEqual(core.resolve_base_url(profile, pi.ENV_KEY, {'LLAMA_PI_BASE_URL': '  '}),
                         'http://127.0.0.1:1235/v1')
        self.assertEqual(core.resolve_base_url(profile, pi.ENV_KEY,
                                               {'LLAMA_PI_BASE_URL': 'https://a.example/v1/'}),
                         'https://a.example/v1')
        for bad in ('relative/v1', 'a.example/v1', 'ftp://a.example/v1'):
            with self.subTest(bad), self.assertRaises(ValueError):
                core.resolve_base_url(profile, pi.ENV_KEY, {'LLAMA_PI_BASE_URL': bad})

    def test_merge_with_configured_base_url(self):
        profile = core.load('bolt35')
        base = core.resolve_base_url(profile, pi.ENV_KEY,
                                     {'LLAMA_PI_BASE_URL': 'https://models.example/v1'})
        result = pi.merge('{}', profile, base_url=base)
        provider = json.loads(result)['providers'][pi.PROVIDER]
        self.assertEqual(provider['baseUrl'], 'https://models.example/v1')
        # The same configured URL is a no-op, not a conflict.
        self.assertEqual(pi.merge(result, profile, base_url=base), result)
        with self.assertRaisesRegex(ValueError, 'baseUrl'):
            pi.merge(result, profile, base_url='https://other.example/v1')
        updated = json.loads(pi.merge(result, profile, base_url='https://other.example/v1', update=True))
        self.assertEqual(updated['providers'][pi.PROVIDER]['baseUrl'], 'https://other.example/v1')

    def test_add_reads_env_file(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            path = Path(temp) / 'models.json'
            with patch('lib.core.load_env_file',
                       return_value={'LLAMA_PI_BASE_URL': 'https://env.example/v1'}):
                pi.add('bolt35', path)
            config = json.loads(path.read_text())
            self.assertEqual(config['providers'][pi.PROVIDER]['baseUrl'], 'https://env.example/v1')
            # Without the configured URL, the local default conflicts with it.
            with patch('lib.core.load_env_file', return_value={}):
                with self.assertRaisesRegex(ValueError, 'baseUrl'):
                    pi.add('bolt35', path)


class PiSmokeAssertionsTests(unittest.TestCase):
    def events(self, answer='hello world', stop='stop'):
        return [
            {'type': 'message_update', 'assistantMessageEvent': {'type': 'text_delta', 'delta': answer}},
            {'type': 'message_end', 'message': {'role': 'assistant', 'stopReason': stop,
                                              'content': [{'type': 'text', 'text': answer}]}},
        ]

    def test_rejects_success_exit_with_wrong_or_incomplete_output(self):
        check_case(self.events(), 'hello world')
        for events in (self.events('wrong'), self.events(stop='length'), [], self.events()[1:]):
            with self.assertRaises(ValueError):
                check_case(events, 'hello world')

    def test_requires_successful_actual_fixture_tool_call(self):
        events = self.events()
        with self.assertRaises(ValueError):
            check_case(events, 'hello world', tool=True)
        events.insert(0, {'type': 'tool_execution_start', 'toolName': 'read',
                          'toolCallId': '1', 'args': {'path': 'pi-fixture.txt'}})
        events.insert(1, {'type': 'tool_execution_end', 'toolName': 'read', 'toolCallId': '1', 'isError': False})
        check_case(events, 'hello world', tool=True)
        events[1]['isError'] = True
        with self.assertRaises(ValueError):
            check_case(events, 'hello world', tool=True)

    def test_thinking_content_must_follow_requested_mode(self):
        events = self.events()
        events[-1]['message']['content'].insert(0, {'type': 'thinking', 'thinking': 'Working it out'})
        check_case(events, 'hello world', thinking=True)
        with self.assertRaises(ValueError):
            check_case(events, 'hello world')


if __name__ == '__main__':
    unittest.main()
