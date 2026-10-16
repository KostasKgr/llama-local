import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments import lfm_benchmark
from lib import core


class LfmBenchmarkTests(unittest.TestCase):
    def test_gpu_conflict_does_not_launch_or_stop_anything(self):
        with patch.object(core, 'preflight'), \
                patch.object(lfm_benchmark.subprocess, 'check_output', return_value='1234\n'), \
                patch.object(lfm_benchmark.subprocess, 'Popen') as launch, \
                patch.object(lfm_benchmark.probes, 'stop') as stop:
            with self.assertRaisesRegex(RuntimeError, 'GPU compute process'):
                lfm_benchmark.trial(36864, 1, None, Path('/unused'))
            launch.assert_not_called()
            stop.assert_not_called()

    def test_startup_failure_stops_only_owned_scope_and_records_failure(self):
        process = Mock()
        process.poll.return_value = 1
        args = argparse.Namespace(output_tokens=512)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(core, 'preflight'), \
                patch.object(lfm_benchmark.subprocess, 'check_output', return_value=''), \
                patch.object(lfm_benchmark.subprocess, 'Popen', return_value=process), \
                patch.object(lfm_benchmark.subprocess, 'run', return_value=Mock(stdout='')), \
                patch.object(lfm_benchmark.threading, 'Thread'), \
                patch.object(lfm_benchmark.probes, 'stop') as stop:
            with self.assertRaisesRegex(RuntimeError, 'Server exited during startup'):
                lfm_benchmark.trial(36864, 1, args, Path(temp))
            record = json.loads((Path(temp) / 'prompt36864-r1' / 'result.json').read_text())
            self.assertFalse(record['stable'])
            self.assertTrue(record['scope'].startswith('lfm25-trial-'))
            stop.assert_called_once_with(record['scope'])
            process.wait.assert_called_once_with(timeout=30)
