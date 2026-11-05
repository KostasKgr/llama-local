import csv
from pathlib import Path
import tempfile
import unittest

from experiments import host_memory


def row(time, **values):
    return {'time': time, **{f: values.get(f, '') for f in host_memory.FIELDS[1:]}}


class HostMemoryTests(unittest.TestCase):
    def write(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'host.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, host_memory.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def rows(self):
        return [row('20261105T100000Z', win_available_mib=20000, win_commit_mib=10000, vmmem_ws_mib=4000),
                row('20261105T100010Z', win_available_mib=15000, win_commit_mib=15000, vmmem_ws_mib=4000,
                    server_private_mib=5000, cgroup_current_mib=1),
                row('20261105T100020Z', win_available_mib=14000, win_commit_mib=16500, vmmem_ws_mib=4100,
                    server_private_mib=6500, cgroup_current_mib=1)]

    def test_summary_measures_changes_from_the_idle_row(self):
        summary = host_memory.summarize(self.write(self.rows()), '20261105T100010Z')
        self.assertEqual(summary['samples'], 3)
        self.assertEqual(summary['interval_s'], 10.0)
        self.assertEqual(summary['peak']['win_available_mib'], 14000)
        self.assertEqual(summary['peak']['server_private_mib'], 6500)
        self.assertEqual(summary['change_from_idle_mib'],
                         {'commit_peak': 6500, 'available_drop': 6000, 'vmmem_ws_peak': 100,
                          'commit_after_load': 5000, 'available_drop_after_load': 5000})
        self.assertEqual(summary['after_load']['server_private_mib'], 5000)
        self.assertNotIn('server_private_mib', summary['idle'])

    def test_no_idle_reference_when_a_server_was_already_running(self):
        summary = host_memory.summarize(self.write(self.rows()), '20261105T100010Z', idle_valid=False)
        self.assertNotIn('idle', summary)
        self.assertNotIn('change_from_idle_mib', summary)
        self.assertEqual(summary['peak']['win_commit_mib'], 16500)

    def test_empty_sample_file(self):
        self.assertEqual(host_memory.summarize(self.write([]), '20261105T100010Z'), {'samples': 0})

    def test_cpu_summary_uses_the_run_and_reports_cores_from_counters(self):
        rows = [row('20261105T100000Z', cpu_pkg_w=17.0, cpu_util_pct=4.0),
                row('20261105T100010Z', cpu_pkg_w=60.0, cpu_util_pct=40.0, cpu_cores_w=45.0, cgroup_cpu_s=100.0,
                    wsl_cpu_s=500.0, sampler_cpu_s=3.0, tz_passive_pct=100.0, tz_throttle=0.0),
                row('20261105T100020Z', cpu_pkg_w=80.0, cpu_util_pct=60.0, cpu_cores_w=65.0, cgroup_cpu_s=180.0,
                    wsl_cpu_s=600.0, sampler_cpu_s=3.5, tz_passive_pct=90.0, tz_throttle=1.0)]
        summary = host_memory.summarize_cpu(self.write(rows), '20261105T100010Z')
        self.assertEqual(summary['samples'], 2)
        self.assertEqual((summary['cpu_pkg_w_median'], summary['cpu_pkg_w_max']), (70.0, 80.0))
        self.assertEqual(summary['server_cores'], 8.0)  # Falls back to the cgroup on WSL.
        self.assertEqual((summary['wsl_cores'], summary['sampler_cores']), (10.0, 0.05))
        self.assertEqual((summary['tz_passive_pct_min'], summary['tz_throttle_max']), (90.0, 1.0))

    def test_windows_server_cpu_wins_over_the_scope_stub(self):
        rows = [row('20261105T100010Z', server_cpu_s=10.0, cgroup_cpu_s=1.0),
                row('20261105T100020Z', server_cpu_s=50.0, cgroup_cpu_s=1.1)]
        self.assertEqual(host_memory.summarize_cpu(self.write(rows), '20261105T100010Z')['server_cores'], 4.0)

    def test_top_cpu_ranks_mean_cores_since_the_start(self):
        top = host_memory.top_cpu({'a': 100.0, 'b': 10.0}, {'a': 110.0, 'b': 50.0, 'new': 5.0}, 10, count=2)
        self.assertEqual(top, {'b': 4.0, 'a': 1.0})

    def test_prompt_cache_peak(self):
        text = ('srv update:  - cache state: 0 prompts, 0.000 MiB (limits: 2048.000 MiB)\n'
                'srv update:  - cache state: 2 prompts, 832.492 MiB (limits: 2048.000 MiB)\n'
                'srv update:  - cache state: 1 prompt, 173.461 MiB (limits: 2048.000 MiB)\n')
        self.assertEqual(host_memory.prompt_cache_peak(text), 832.492)
        self.assertEqual(host_memory.prompt_cache_peak('no cache lines'), 0.0)


if __name__ == '__main__':
    unittest.main()
