"""Opt-in by installed bundled Python; CPU fixtures never load model weights."""
import json
import hashlib
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import uuid

from lib import strata


@unittest.skipUnless(strata.python().is_file(), 'Strata Windows bundle not installed')
class WindowsOwnershipTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        artifact = strata.pins()['packages'][0]
        strata.verify_extraction(strata.root() / 'downloads' / artifact['file'], strata.control())

    def test_native_hash_matches_source_and_refuses_outside_storage(self):
        strata.stage_bridge()
        path = strata.control() / 'BUNDLE.json'
        actual = json.loads(strata.helper('hash', '--file', strata.win(path)).stdout)['sha256']
        self.assertEqual(actual, hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(subprocess.CalledProcessError):
            strata.helper('hash', '--file', r'C:\Windows\win.ini')

    def wait(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.1)
        self.fail('Owned Windows fixture timed out')

    def scenario(self, abrupt, telemetry=False):
        strata.stage_bridge()
        token = uuid.uuid4().hex
        run = strata.root() / 'runs' / token
        run.mkdir(parents=True)
        process = None
        port = int(strata.helper('port').stdout)
        fixture = run / 'fixture.py'
        shutil.copyfile(Path(__file__).parent / 'fixtures/strata_child.py', fixture)
        strata.atomic(run / 'launch.json', {'run_dir': strata.win(run), 'token': token,
            'scope': 'gyros-trial-' + token, 'port': port, 'alias': 'fixture', 'context': 131072,
            'cwd': strata.win(strata.control()), 'telemetry': telemetry, 'verify_engine': False,
            'command': [strata.win(strata.python()), '-B', strata.win(fixture), strata.win(run), str(port)]})
        strata.atomic(run / 'heartbeat.json', {'token': token, 'sequence': 1})
        try:
            with (run / 'test-owner.log').open('wb') as log:
                process = subprocess.Popen([str(strata.python()), '-B', '-s', strata.win(strata.bridge()),
                    'serve', '--root', strata.windows_root(), '--run-dir', strata.win(run)],
                    cwd=strata.control(), env=strata.isolated_env(), stdout=log, stderr=subprocess.STDOUT)
                def ready():
                    if process.poll() is not None:
                        self.fail((run / 'test-owner.log').read_text() + '\n' + (run / 'server.log').read_text())
                    return json.loads((run / 'owner.json').read_text())['state'] == 'ready'
                self.wait(ready)
                owner = json.loads((run / 'owner.json').read_text())
                self.assertGreater(len(owner['children']), 1)
                strata.atomic(run / 'stop.json', {'token': 'wrong-token'})
                time.sleep(0.3)
                self.assertIsNone(process.poll())
                if abrupt:
                    process.kill()  # captured interop process, never a process-name match
                else:
                    strata.atomic(run / 'stop.json', {'token': token})
                process.wait(timeout=40)
                for ident in owner['children']:
                    def gone():
                        result = strata.helper('identity', '--pid', str(ident['pid']))
                        actual = json.loads(result.stdout)
                        return actual is None or actual['created'] != ident['created']
                    self.wait(gone)
                if telemetry:
                    from experiments.host_memory import summarize_strata
                    host = summarize_strata(run / 'host.csv')
                    self.assertGreater(host['samples'], 1)
                    self.assertGreater(host['win_commit_mib']['max'], 0)
                    self.assertTrue((run / 'gpu.csv').is_file())
                    self.assertTrue((run / 'power.csv').is_file())
                    # These counters are optional across Windows hosts, but a script error is never a sample.
                    self.assertNotIn('op_Division', (run / 'power.csv').read_text())
                    disk = (run / 'disk.csv').read_text().splitlines()
                    self.assertEqual(disk[0], 'time,read_mib_s,reads_s,read_ms,read_queue,page_reads_s')
                    self.assertGreater(len(disk), 1)
                    self.assertEqual(len(disk[1].split(',')), 6)
                    load = json.loads((run / 'load.json').read_text())
                    self.assertGreater(load['load_s'], 0)
        finally:
            strata.atomic(run / 'stop.json', {'token': token})
            if process is not None and process.poll() is None:
                process.wait(timeout=40)
            evidence = strata.core.ROOT / 'logs/gyros' / ('lifecycle-' + token)
            shutil.copytree(run, evidence)
            shutil.rmtree(run)

    def test_normal_stop_closes_nested_engine_job(self):
        self.scenario(False)

    def test_abrupt_owner_loss_closes_nested_engine_job(self):
        self.scenario(True)

    def test_telemetry_processes_share_owner_lifetime(self):
        self.scenario(False, telemetry=True)


if __name__ == '__main__':
    unittest.main()
