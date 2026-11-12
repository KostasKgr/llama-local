import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from experiments import host_memory
from experiments import strata_benchmark as bench
from lib import core


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_public_readiness_waits_for_plain_text_proxy_health(self):
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch.object(bench, 'urlopen', side_effect=[URLError('starting'), response]) as open_url, \
                patch.object(bench.time, 'sleep'):
            bench.wait_public('http://127.0.0.1:1235')
        self.assertEqual(open_url.call_count, 2)

    def test_rates_exclude_cached_tokens_and_single_token_decode(self):
        summary = bench.summarize_requests([
            {'prompt_tokens': 110, 'reused': 100, 'prompt_ms': 100, 'engine_generated': 20, 'decode_ms': 200},
            {'prompt_tokens': 1000, 'prompt_read': 5, 'reused': 0, 'prompt_ms': 50,
             'engine_generated': 1, 'decode_ms': 1},
            {'prompt_tokens': 12, 'reused': 2, 'prompt_ms': None, 'engine_generated': None, 'decode_ms': None}])
        self.assertEqual(summary['prefill_tokens'], 15)
        self.assertEqual(summary['prefill_tok_s'], 100)
        self.assertEqual(summary['decode_tokens'], 20)
        self.assertEqual(summary['decode_tok_s'], 100)
        self.assertEqual(summary['decode_timed_requests'], 1)

    def test_recycled_sampler_pid_is_never_signalled(self):
        expected = {'pid': 123, 'start': 'before', 'cmd': 'owned'}
        with patch.object(bench, 'linux_identity', return_value={**expected, 'start': 'after'}), \
                patch('experiments.strata_benchmark.os.kill') as kill:
            bench.stop_collector(expected)
            kill.assert_not_called()

    def test_collector_waits_for_exec_before_recording_identity(self):
        process = MagicMock(pid=123)
        process.poll.return_value = None
        out = Path('/tmp/owned-benchmark')
        command = [bench.sys.executable, str(Path(bench.__file__).resolve()), '_collect', str(out),
                   '--health-url', 'http://127.0.0.1:1']
        expected = {'pid': 123, 'start': 'captured',
                    'cmd': (b'\0'.join(os.fsencode(arg) for arg in command) + b'\0').hex()}
        with patch.object(bench.subprocess, 'Popen', return_value=process), \
                patch.object(bench, 'linux_identity', side_effect=[{**expected, 'cmd': ''}, expected]), \
                patch.object(bench.time, 'sleep') as sleep:
            self.assertEqual(bench.start_collector(out, 'http://127.0.0.1:1'), expected)
            sleep.assert_called_once()
        process.terminate.assert_not_called()

    def test_incomplete_collector_identity_refuses_cleanup(self):
        with patch.object(bench.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                bench.stop_collector({'pid': 123, 'start': 'captured', 'cmd': ''})
            kill.assert_not_called()

    def test_real_collector_stops_with_captured_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            expected = bench.start_collector(Path(temp), 'http://127.0.0.1:1')
            try:
                self.assertTrue(expected['cmd'])
                self.assertEqual(bench.linux_identity(expected['pid']), expected)
            finally:
                bench.stop_collector(expected)
                os.waitpid(expected['pid'], 0)
            self.assertIsNone(bench.linux_identity(expected['pid']))

    def test_busy_gpu_trial_never_launches_or_stops_any_process(self):
        with patch.object(bench.swap, 'managed', return_value=False), \
                patch.object(bench.swap, 'service_state', return_value='inactive'), \
                patch.object(bench.probes, 'gpu', return_value=[3000]), \
                patch('experiments.strata_benchmark.subprocess.Popen') as spawn, \
                patch.object(bench.strata, 'stop') as stop:
            with self.assertRaisesRegex(ValueError, 'GPU occupied'), bench.trial():
                self.fail('Busy trial entered')
            spawn.assert_not_called()
            stop.assert_not_called()

    def session(self, root):
        messages = []
        model = core.load('gyros')['alias']
        for index, name in enumerate([*bench.REQUIRED, '.pi/nonce.txt']):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = 'NONCE\n' if name == '.pi/nonce.txt' else 'first line\nsecond line\nthird line\n'
            path.write_text(content)
            call = {'type': 'toolCall', 'id': str(index), 'name': 'read', 'arguments': {'path': name}}
            messages.append({'type': 'message', 'message': {'role': 'assistant', 'provider': 'llama-local',
                              'model': model, 'stopReason': 'toolUse', 'content': [call]}})
            messages.append({'type': 'message', 'message': {'role': 'toolResult', 'toolCallId': str(index),
                              'toolName': 'read', 'isError': False, 'content': [{'type': 'text', 'text': content}]}})
        messages.append({'type': 'message', 'message': {'role': 'assistant', 'provider': 'llama-local',
                          'model': model, 'stopReason': 'stop',
                          'content': [{'type': 'text', 'text': ('Source evidence and call chain. ' * 25) + 'NONCE'}]}})
        return messages

    def test_agentic_audit_requires_actual_results_complete_reads_and_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            messages = self.session(root)
            path = root / 'session.jsonl'
            def audit(rows):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                return bench.audit_session(path, root, 'NONCE')
            self.assertTrue(audit(messages)['passed'])
            self.assertFalse(audit([m for m in messages if m['message']['role'] != 'toolResult'])['passed'])
            messages[1]['message']['content'][0]['text'] = 'pretend output\n' * 20
            self.assertFalse(audit(messages)['passed'])
            messages = self.session(root)
            messages[0]['message']['model'] = 'different-model'
            self.assertFalse(audit(messages)['passed'])
            messages = self.session(root)
            messages[1]['message']['isError'] = True
            self.assertFalse(audit(messages)['passed'])

    def test_fixture_policy_restores_only_unchanged_owned_content(self):
        import base64
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.json'
            path.write_text('installed')
            snapshot = {'files': [{'path': str(path), 'before': base64.b64encode(b'original').decode(),
                                  'installed_sha256': hashlib.sha256(b'installed').hexdigest()}]}
            bench.restore_policy(snapshot)
            self.assertEqual(path.read_text(), 'original')
            path.write_text('user edit')
            with self.assertRaises(ValueError):
                bench.restore_policy(snapshot)
            self.assertEqual(path.read_text(), 'user edit')

    def test_owner_telemetry_is_windowed_and_missing_values_stay_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'gpu.csv'
            path.write_text('time,power_w,memory_mib\n1,20,100\n2,N/A,200\n3,80,300\n4,40,150\n')
            summary = host_memory.summarize_strata(path, 2, 3)
            self.assertEqual(summary['samples'], 2)
            self.assertEqual(summary['power_w']['median'], 80)
            self.assertEqual(summary['memory_mib']['max'], 300)


if __name__ == '__main__':
    unittest.main()
