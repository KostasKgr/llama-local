#!/usr/bin/env python3
"""Sample host RAM and CPU load for one model server, on both sides of WSL.

A WSL server's memory is charged to its scope's cgroup. A Windows server's is not:
the scope only holds the interop stub, so its counters read near zero and
MemoryHigh/MemoryMax do not apply. Commit charge and available memory are system-wide
Windows figures that include the WSL VM, so they are the ones to compare across
platforms. They also include every other application, so use changes, not levels.

CPU die temperature needs Windows admin rights. RAPL package power is readable without
them and is the heat the CPU produces, so it stands in; the ACPI thermal zone temperature
never moved under load, so only its throttle state is kept. cpu_perf_pct is clock speed
as a share of base, which Windows' frequency counter does not show. Server CPU time comes from the cgroup on WSL and the process on Windows. wsl_cpu_s
covers the whole VM, including a WSL server, so do not add the two together.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core

MEMORY_FIELDS = ('win_available_mib', 'win_commit_mib', 'gpu_shared_mib', 'vmmem_ws_mib', 'vmmem_private_mib',
                 'server_ws_mib', 'server_private_mib', 'server_peak_ws_mib')
CPU_FIELDS = ('cpu_util_pct', 'cpu_perf_pct', 'cpu_pkg_w', 'cpu_cores_w', 'tz_passive_pct', 'tz_throttle',
              'server_cpu_s', 'sampler_cpu_s')
WINDOWS_FIELDS = MEMORY_FIELDS + CPU_FIELDS
CGROUP_FIELDS = ('cgroup_current_mib', 'cgroup_peak_mib', 'cgroup_anon_mib', 'cgroup_file_mib', 'cgroup_shmem_mib',
                 'cgroup_cpu_s')
FIELDS = ('time',) + WINDOWS_FIELDS + CGROUP_FIELDS + ('wsl_available_mib', 'wsl_cpu_s')
PEAK_FIELDS = MEMORY_FIELDS + CGROUP_FIELDS[:-1] + ('wsl_available_mib',)
# Lower is worse for these; every other field peaks at its maximum.
LOWEST = ('win_available_mib', 'wsl_available_mib')
COUNTERS = ('\\GPU Adapter Memory(*)\\Shared Usage', '\\Processor Information(_Total)\\% Processor Utility',
            '\\Processor Information(_Total)\\% Processor Performance',
            '\\Energy Meter(rapl_package0_pkg)\\Power', '\\Energy Meter(rapl_package0_pp0)\\Power',
            '\\Thermal Zone Information(*)\\% Passive Limit', '\\Thermal Zone Information(*)\\Throttle Reasons')


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def windows_script(interval):
    """One long-lived PowerShell: starting one per sample would cost more than sampling."""
    runtimes = core.windows_root() + '\\runtimes\\'
    return ("$ErrorActionPreference='SilentlyContinue';"
            "function mib($b) { if ($null -eq $b) { '' } else { [math]::Round($b / 1MB) } };"
            "function num($b, $d) { if ($null -eq $b) { '' } else { [math]::Round($b / $d, 1) } };"
            "function pick($p) { $m = @($c | Where-Object { $_.Path -match $p });"
            " if ($m.Count) { ($m | Measure-Object CookedValue -Sum).Sum } };"
            "$paths = @(" + ','.join(f"'{counter}'" for counter in COUNTERS) + ");"
            "while ($true) {"
            "$o = Get-CimInstance Win32_OperatingSystem;"
            "$c = (Get-Counter $paths).CounterSamples;"
            "$v = Get-Process vmmemWSL | Select-Object -First 1;"
            "$s = Get-Process llama-server | Where-Object { $_.Path -and "
            f"$_.Path.StartsWith('{runtimes}', [StringComparison]::OrdinalIgnoreCase) }} | Select-Object -First 1;"
            "@([math]::Round($o.FreePhysicalMemory / 1KB),"
            " [math]::Round(($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1KB),"
            " (mib (pick 'shared usage$')), (mib $v.WorkingSet64), (mib $v.PrivateMemorySize64),"
            " (mib $s.WorkingSet64), (mib $s.PrivateMemorySize64), (mib $s.PeakWorkingSet64),"
            " (num (pick 'processor utility$') 1), (num (pick 'processor performance$') 1),"
            " (num (pick 'pkg\\)\\\\power$') 1000), (num (pick 'pp0\\)\\\\power$') 1000),"
            " (num (pick 'passive limit$') 1), (num (pick 'throttle reasons$') 1),"
            " (num $s.CPU 1), (num (Get-Process -Id $PID).CPU 1)) -join ',';"
            f"Start-Sleep -Seconds {int(interval)}"
            "}")


def cgroup(scope):
    """The scope exists only while its server does, so resolve it on every sample."""
    blank = [''] * len(CGROUP_FIELDS)
    path = subprocess.run(['systemctl', '--user', 'show', scope + '.scope', '-p', 'ControlGroup', '--value'],
                          capture_output=True, text=True, timeout=10).stdout.strip()
    if not path:
        return blank
    base = Path('/sys/fs/cgroup' + path)
    try:
        stat = dict(line.split() for line in (base / 'memory.stat').read_text().splitlines())
        raw = [(base / 'memory.current').read_text(), (base / 'memory.peak').read_text(),
               stat['anon'], stat['file'], stat['shmem']]
        cpu = dict(line.split() for line in (base / 'cpu.stat').read_text().splitlines())
        return [round(int(value) / 1048576) for value in raw] + [round(int(cpu['usage_usec']) / 1000000, 1)]
    except (OSError, KeyError, ValueError):
        return blank


def wsl_available():
    match = re.search(r'MemAvailable:\s+(\d+) kB', Path('/proc/meminfo').read_text())
    return round(int(match[1]) / 1024)


def wsl_cpu_seconds():
    """Busy time across the VM: every /proc/stat column except idle and iowait."""
    columns = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return round((sum(columns) - columns[3] - columns[4]) / os.sysconf('SC_CLK_TCK'), 1)


def sample(out, scope, interval):
    process = subprocess.Popen([core.powershell_exe(), '-NoProfile', '-NonInteractive', '-Command',
                                windows_script(interval)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for line in process.stdout:
            values = line.strip().split(',')
            if len(values) == len(WINDOWS_FIELDS):
                writer.writerow([timestamp()] + values + cgroup(scope) + [wsl_available(), wsl_cpu_seconds()])
                f.flush()


def wait_first_row(path, timeout=30):
    deadline = time.monotonic() + timeout
    while not path.is_file() or len(path.read_text().splitlines()) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError('Host memory sampler produced no data')
        time.sleep(0.2)


def read(path):
    with path.open(newline='') as f:
        return [{k: v if k == 'time' else (float(v) if v else None) for k, v in row.items()}
                for row in csv.DictReader(f)]


def seconds(row):
    return datetime.strptime(row['time'], '%Y%m%dT%H%M%SZ').timestamp()


def peaks(rows):
    result = {}
    for field in PEAK_FIELDS:
        values = [r[field] for r in rows if r[field] is not None]
        if values:
            result[field] = min(values) if field in LOWEST else max(values)
    return result


def summarize(path, loaded, idle_valid=True):
    """loaded: the timestamp after warm-up. The first row is an idle reference only when
    no server was running as sampling began."""
    rows = read(path)
    if not rows:
        return {'samples': 0}
    during = [r for r in rows if r['time'] >= loaded]
    result = {'samples': len(rows), 'interval_s': None, 'peak': peaks(rows)}
    if len(rows) > 1:
        result['interval_s'] = round((seconds(rows[-1]) - seconds(rows[0])) / (len(rows) - 1), 1)
    if during:
        result['after_load'] = {k: v for k, v in during[0].items() if v is not None}
        result['end'] = {k: v for k, v in rows[-1].items() if v is not None}
    if idle_valid:
        idle = rows[0]
        result['idle'] = {k: v for k, v in idle.items() if v is not None}
        peak = result['peak']
        result['change_from_idle_mib'] = {'commit_peak': peak['win_commit_mib'] - idle['win_commit_mib'],
                                          'available_drop': idle['win_available_mib'] - peak['win_available_mib'],
                                          'vmmem_ws_peak': peak['vmmem_ws_mib'] - idle['vmmem_ws_mib']}
        if during:
            result['change_from_idle_mib'].update({
                'commit_after_load': during[0]['win_commit_mib'] - idle['win_commit_mib'],
                'available_drop_after_load': idle['win_available_mib'] - during[0]['win_available_mib']})
    return result


def cores_used(rows, field):
    """Mean cores from a cumulative CPU-seconds counter."""
    used = [r for r in rows if r[field] is not None]
    elapsed = seconds(used[-1]) - seconds(used[0]) if len(used) > 1 else 0
    return round((used[-1][field] - used[0][field]) / elapsed, 2) if elapsed else None


def summarize_cpu(path, loaded):
    """Load between warm-up and the end of the run."""
    rows = read(path)
    during = [r for r in rows if r['time'] >= loaded]
    if not during or 'cpu_pkg_w' not in rows[0]:
        return {}
    result = {'samples': len(during),
              'server_cores': cores_used(during, 'server_cpu_s') or cores_used(during, 'cgroup_cpu_s'),
              'wsl_cores': cores_used(during, 'wsl_cpu_s'), 'sampler_cores': cores_used(during, 'sampler_cpu_s')}
    for field in ('cpu_util_pct', 'cpu_pkg_w', 'cpu_cores_w', 'cpu_perf_pct'):
        values = [r[field] for r in during if r[field] is not None]
        if values:
            result[field + '_median'] = round(st.median(values), 1)
            result[field + '_max'] = max(values)
    limits = [r['tz_passive_pct'] for r in during if r['tz_passive_pct'] is not None]
    reasons = [r['tz_throttle'] for r in during if r['tz_throttle'] is not None]
    result.update({'tz_passive_pct_min': min(limits, default=None), 'tz_throttle_max': max(reasons, default=None)})
    return result


def process_cpu():
    """Cumulative CPU seconds per Windows process name, for attributing load over a run."""
    raw = core.powershell("Get-Process | Where-Object { $_.CPU } | Group-Object ProcessName | ForEach-Object "
                          "{ [pscustomobject]@{n=$_.Name; s=($_.Group | Measure-Object CPU -Sum).Sum} } "
                          "| ConvertTo-Json -Compress")
    return {row['n']: row['s'] for row in json.loads(raw)} if raw else {}


def top_cpu(before, after, elapsed, count=8):
    """Mean cores per process over the run; processes that exited are left out."""
    used = {name: (seconds_used - before.get(name, 0)) / elapsed for name, seconds_used in after.items()}
    return {name: round(cores, 2) for name, cores in sorted(used.items(), key=lambda x: -x[1])[:count]}


def prompt_cache_peak(text):
    """Largest RAM prompt cache the server reported, in MiB."""
    return max((float(size) for size in re.findall(r'cache state: \d+ prompts?, ([\d.]+) MiB', text)), default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--interval', type=int, default=2, help='seconds between samples, plus query time')
    args = parser.parse_args()
    sample(args.out, args.scope, args.interval)


if __name__ == '__main__':
    main()
