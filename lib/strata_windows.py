"""Windows owner for one Strata API/engine tree; invoked with bundled Python."""
import argparse
import ctypes
from ctypes import wintypes
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import socket
import secrets
import sys
import threading
import time
from urllib.request import urlopen


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    deadline = time.monotonic() + 3
    while True:
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.03)


def setup(root):
    root = Path(root).resolve()
    control = root / 'runtimes' / 'control-0.1.59'
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'PYTHON', 'HF_', 'TRANSFORMERS_')):
            os.environ.pop(key)
    for key, suffix in {'APPDATA': 'appdata', 'LOCALAPPDATA': 'localappdata',
                        'USERPROFILE': 'home', 'HOME': 'home',
                        'STRATA_INSTALL_DIR': 'install', 'STRATA_PROFILE_DIR': 'profiles',
                        'STRATA_RUNTIME_DIR': 'runtime-state', 'STRATA_COMPONENTS_DIR': 'components'}.items():
        os.environ[key] = str(root / 'state' / suffix)
    os.environ.update(STRATA_NO_ACQUIRE='1', STRATA_OFFLINE='1', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', PIP_NO_INDEX='1', PYTHONNOUSERSITE='1',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                      STRATA_GGUF_PY=str(root / 'tools' / 'gguf-py'))
    sys.path.insert(0, str(control))
    sys.dont_write_bytecode = True
    os.chdir(control)
    return control


def identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return {'pid': p.pid, 'created': p.create_time(), 'exe': p.exe(), 'cmd': p.cmdline()}
    except psutil.NoSuchProcess:
        return None


def same_process(expected):
    actual = identity(expected['pid'])
    return actual is not None and actual == expected


def check_engine(root, full=True):
    from strata_control import engine_component as ec, signing
    folder = root / 'runtimes' / 'engine-0.1.59-cuda13'
    component = ec.read_manifest(folder)
    if component.version != '0.1.59' or component.platform != 'windows-x86_64' or component.backend != 'cuda':
        raise ValueError('Unexpected engine component identity')
    if full:
        signing.install_component_verifier(signing.load_trusted_keys(environ={}))
        if ec.verify_signature(component, environ={'STRATA_REQUIRE_SIGNED_COMPONENTS': '1'}) != 'verified':
            raise ValueError('Engine signature was not verified')
        ec.verify(component, full=True)
    else:
        # Serving checks sizes only; install's probe verified the signature and hashed every file (~1.2 s).
        for rel, _, size in component.files:
            if (folder / rel).stat().st_size != size:
                raise ValueError('Engine file size mismatch; run verify gyros: ' + rel)
    smi = str(Path(os.environ['SystemRoot']) / 'System32/nvidia-smi.exe')
    gpu = subprocess.check_output([smi, '--query-gpu=driver_version,compute_cap',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    driver, capability = [field.strip() for field in gpu.splitlines()[0].split(',')]
    version = lambda value: tuple(int(part) for part in value.split('.'))
    if version(driver) < version(component.min_driver['nvidia']):
        raise ValueError('NVIDIA driver is below the pinned engine floor')
    present = 'sm_' + capability.replace('.', '')
    if not ec.arch_covers(component.archs, [present]):
        raise ValueError('Engine does not cover this CUDA architecture: ' + present)
    outputs = {}
    for flag in ('--version', '--geometry', '--help'):
        result = subprocess.run([str(component.exe), flag], cwd=folder, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=60, check=True)
        outputs[flag] = result.stdout + result.stderr
    if 'flash_next' not in outputs['--geometry']:
        raise ValueError('Engine is not flash_next')
    for flag in ('--draft-fallback', '--mtp-draft-vocab', '--resident-experts', '--ram-keep-gib'):
        if flag not in outputs['--help']:
            raise ValueError('Engine interface missing ' + flag)
    return {'component': component.as_dict() if hasattr(component, 'as_dict') else read(folder / 'engine-component.json'),
            'probes': outputs, 'python': sys.version, 'gpu': gpu}


def commit_mib():
    class Performance(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
            ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
             'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
             (n, wintypes.DWORD) for n in ('HandleCount', 'ProcessCount', 'ThreadCount')]
    info = Performance()
    info.cb = ctypes.sizeof(info)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetPerformanceInfo.argtypes = [ctypes.POINTER(Performance), wintypes.DWORD]
    if not api.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return None
    return info.CommitTotal * info.PageSize / 1048576


def job():
    """Contain this owner before it has children; the handle is never inherited."""
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]
    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                    ('flags', wintypes.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                    ('active_limit', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]
    k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    handle = k.CreateJobObjectW(None, None)
    info = Extended()
    info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not handle or not k.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info))
            or not k.AssignProcessToJobObject(handle, k.GetCurrentProcess())):
        raise ctypes.WinError(ctypes.get_last_error())
    return k, handle


def start_telemetry(run_dir, api_pid):
    """All samplers inherit the owner's job; no sampler can outlive the owner."""
    import psutil
    done = threading.Event()
    children, streams, readers = [], [], []
    commands = [
        ('gpu.csv', [str(Path(os.environ['SystemRoot']) / 'System32/nvidia-smi.exe'),
                     '--query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits', '-lms', '1000']),
        ('power.csv', [str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'),
                       '-NoProfile', '-NonInteractive', '-Command',
                       "$ErrorActionPreference='SilentlyContinue';'time,cpu_pkg_w,cpu_cores_w';"
                       "Get-Counter '\\Energy Meter(rapl_package0_pkg)\\Power',"
                       "'\\Energy Meter(rapl_package0_pp0)\\Power' -SampleInterval 1 -Continuous | ForEach-Object {"
                       "$c=$_.CounterSamples;"
                       "$p=($c|Where-Object {$_.Path -match 'pkg\\)\\\\power$'}).CookedValue;"
                       "$q=($c|Where-Object {$_.Path -match 'pp0\\)\\\\power$'}).CookedValue;"
                       "@(([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000),"
                       "$(if($null -ne $p){$p/1000}),$(if($null -ne $q){$q/1000})) -join ','}"]),
        # System-wide: the GPU-cache fill reads the mapped GGUF through page faults, not process file I/O.
        ('disk.csv', [str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'),
                      '-NoProfile', '-NonInteractive', '-Command',
                      "$ErrorActionPreference='SilentlyContinue';"
                      "'time,read_mib_s,reads_s,read_ms,read_queue,page_reads_s';"
                      "$n='\\PhysicalDisk(_Total)\\';"
                      "Get-Counter ($n+'Disk Read Bytes/sec'),($n+'Disk Reads/sec'),($n+'Avg. Disk sec/Read'),"
                      "($n+'Avg. Disk Read Queue Length'),'\\Memory\\Page Reads/sec' -SampleInterval 1 -Continuous"
                      " | ForEach-Object {$c=$_.CounterSamples;"
                      "$v={param($m) ($c|Where-Object {$_.Path -like ('*'+$m)}).CookedValue};"
                      "$b=& $v 'disk read bytes/sec';$l=& $v 'avg. disk sec/read';"
                      "@(([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000),"
                      "$(if($null -ne $b){$b/1048576}),(& $v 'disk reads/sec'),$(if($null -ne $l){$l*1000}),"
                      "(& $v 'avg. disk read queue length'),(& $v 'memory\\page reads/sec')) -join ','}"])]
    for name, command in commands:
        stream = (run_dir / name).open('wb')
        streams.append(stream)
        try:
            if name == 'gpu.csv':
                stream.write(b'time,power_w,sm_mhz,memory_mhz,temperature_c,utilization_pct,memory_mib\n')
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                def copy_gpu(source=child.stdout, target=stream):
                    for line in source:
                        target.write(str(time.time()).encode() + b',' + line)
                        target.flush()
                reader = threading.Thread(target=copy_gpu, daemon=True)
                reader.start()
                readers.append(reader)
                children.append(child)
            else:
                children.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        except OSError as exc:
            stream.write(str(exc).encode())
    def sample():
        with (run_dir / 'host.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time', 'win_available_mib', 'win_commit_mib', 'server_ws_mib',
                             'server_private_mib', 'server_peak_ws_mib', 'server_cpu_s'])
            while not done.is_set():
                servers = []
                try:
                    api = psutil.Process(api_pid()) if api_pid() else None
                    if api:
                        servers = [api, *api.children(recursive=True)]
                except psutil.NoSuchProcess:
                    pass
                ws, private, peak, cpu = 0, 0, 0, 0
                for server in servers:
                    try:
                        memory = server.memory_info()
                        ws += memory.rss
                        private += memory.private
                        peak += memory.peak_wset
                        times = server.cpu_times()
                        cpu += times.user + times.system
                    except psutil.NoSuchProcess:
                        pass
                virtual = psutil.virtual_memory()
                writer.writerow([time.time(), virtual.available / 1048576, commit_mib(),
                                 ws / 1048576, private / 1048576, peak / 1048576, cpu])
                stream.flush()
                done.wait(1)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    def finish():
        time.sleep(2)  # retain post-exit observations
        done.set()
        thread.join(timeout=3)
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        for reader in readers:
            reader.join(timeout=2)
        for stream in streams:
            stream.close()
    return finish


# Startup phases the engine reports with their own durations; the load summary keeps the rest as unaccounted time.
PHASES = {
    'available_ram_gib': (r'memory guard: .*?; available ([\d.]+) GiB', None),
    'dense_weights': (r'native projection matrices, [\d.]+ MiB of weights, in ([\d.]+) s', 's'),
    'gpu_fill': (r'pre-filled (\d+) of \d+ slots from the profile in ([\d.]+) s \((\d+) MB/s\)', 'slots s mb_s'),
    'ram_complement': (r'pinned cache complement ready: resident ([\d.]+) GiB, .*? in ([\d.]+) s', 'gib s'),
}


def load_summary(run_dir, started_at, load_s):
    """Engine phase timings and disk reads between owner start and readiness, from existing logs."""
    try:
        log = (run_dir / 'engine.log').read_text(encoding='utf-8', errors='replace')
    except OSError:
        log = ''
    phases, accounted = {}, 0.0
    for name, (pattern, fields) in PHASES.items():
        match = re.search(pattern, log)
        if not match:
            continue
        values = [float(v) for v in match.groups()]
        phases[name] = values[0] if fields in (None, 's') else dict(zip(fields.split(), values))
        if fields:
            accounted += values[0] if fields == 's' else phases[name]['s']
    summary = {'load_s': round(load_s, 2), 'phases': phases, 'unaccounted_s': round(load_s - accounted, 2)}
    try:
        with (run_dir / 'disk.csv').open(encoding='utf-8') as stream:
            rows = [r for r in csv.DictReader(stream)
                    if r.get('time') and started_at <= float(r['time']) and r.get('read_mib_s')]
    except (OSError, ValueError):
        rows = []
    if rows:
        rates = [float(r['read_mib_s']) for r in rows]
        latency = [float(r['read_ms']) for r in rows if r.get('read_ms')]
        summary['disk'] = {'samples': len(rows), 'read_gib': round(sum(rates) / 1024, 2),
                           'read_mib_s_mean': round(sum(rates) / len(rates), 1), 'read_mib_s_max': round(max(rates), 1),
                           'read_ms_max': round(max(latency), 3) if latency else None}
    return summary


def serve(root, run_dir):
    import psutil
    run_dir = Path(run_dir)
    launch = read(run_dir / 'launch.json')
    if Path(launch['run_dir']).resolve() != run_dir.resolve() or run_dir.parent.resolve() != (root / 'runs').resolve():
        raise ValueError('Run directory outside managed storage')
    k, handle = job()
    if launch.get('verify_engine', True):
        check_engine(root, full=False)
    record = {'token': launch['token'], 'owner': identity(os.getpid()), 'state': 'starting',
              'scope': launch['scope'], 'port': launch['port'], 'started_at': time.time()}
    write(run_dir / 'owner.json', record)
    process = None
    finish_telemetry = start_telemetry(run_dir, lambda: process.pid if process is not None else None) \
        if launch.get('telemetry', True) else lambda: None
    started = time.monotonic()
    deadline = started + launch.get('startup_timeout', 900)
    last_heartbeat, last_seen = None, started
    try:
        with (run_dir / 'server.log').open('wb') as log:
            child_env = dict(os.environ)
            attestation = None
            if launch.get('verify_engine', True):
                from serve import launch_identity
                key = secrets.token_bytes(32)
                child_env['STRATA_OWNER_ATTEST_KEY'] = key.hex()
                args = launch['command'][launch['command'].index('serve.server') + 1:]
                attestation = launch_identity.attestation(key, (run_dir / 'config.json').read_bytes(), args)
            process = subprocess.Popen(launch['command'], cwd=launch['cwd'], stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT, env=child_env)
            record['api'] = identity(process.pid)
            write(run_dir / 'owner.json', record)
            while process.poll() is None:
                try:
                    beat = read(run_dir / 'heartbeat.json')
                    if beat['token'] == launch['token'] and beat['sequence'] != last_heartbeat:
                        last_heartbeat, last_seen = beat['sequence'], time.monotonic()
                except (OSError, ValueError, KeyError):
                    pass
                try:
                    if read(run_dir / 'stop.json')['token'] == launch['token']:
                        break
                except FileNotFoundError:
                    pass
                if time.monotonic() - last_seen > 10:
                    record['reason'] = 'owner heartbeat lost'
                    break
                if record['state'] == 'starting':
                    if time.monotonic() > deadline:
                        raise TimeoutError('Strata startup deadline exceeded')
                    ready_path = run_dir / 'ready.json'
                    if ready_path.exists():
                        ready = read(ready_path)
                        base = 'http://127.0.0.1:' + str(launch['port'])
                        with urlopen(base + '/health', timeout=3) as response:
                            health = json.load(response)
                        with urlopen(base + '/v1/models', timeout=3) as response:
                            models = json.load(response)
                        with urlopen(base + '/v1/telemetry', timeout=3) as response:
                            telemetry = json.load(response)
                        if (ready['pid'] != process.pid or ready['bound_port'] != launch['port']
                                or ready['model_id'] != launch['alias'] or ready['model_state'] != 'ready'
                                or health.get('service') != 'strata' or not health.get('loaded')
                                or health.get('max_context') != launch['context']
                                or not any(m['id'] == launch['alias'] for m in models['data'])
                                or (attestation is not None and not hmac.compare_digest(
                                    telemetry['instance'].get('launch_attestation', ''), attestation))
                                or telemetry['instance']['id'] != ready['instance_id']):
                            raise ValueError('Strata readiness/identity mismatch')
                        record.update(state='ready', ready=ready, load_s=time.monotonic() - started,
                                      children=[identity(p.pid) for p in psutil.Process().children(recursive=True)])
                        write(run_dir / 'owner.json', record)
                        write(run_dir / 'load.json', load_summary(run_dir, record['started_at'], record['load_s']))
                        print(json.dumps({'event': 'ready', **record}), flush=True)
                time.sleep(0.25)
            if process.poll() is not None:
                raise RuntimeError('API exited unexpectedly: ' + str(process.returncode))
    finally:
        record['state'] = 'stopping'
        write(run_dir / 'owner.json', record)
        if process is not None and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                record.update(state='forced', stopped_at=time.time())
                write(run_dir / 'owner.json', record)
                k.TerminateJobObject(handle, 1)
                raise RuntimeError('Owned job did not stop')
        record.update(state='stopped', stopped_at=time.time())
        write(run_dir / 'owner.json', record)
        finish_telemetry()
        # Leaving main closes our sole job handle even after an unhandled exception.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('serve', 'probe', 'identity', 'check', 'port', 'hash'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--file')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    setup(root)
    if args.action == 'serve':
        serve(root, args.run_dir)
    elif args.action == 'probe':
        print(json.dumps(check_engine(root)), flush=True)
    elif args.action == 'identity':
        print(json.dumps(identity(args.pid)), flush=True)
    elif args.action == 'port':
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            print(listener.getsockname()[1], flush=True)
    elif args.action == 'hash':
        path = Path(args.file).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Hash target outside managed storage')
        with path.open('rb') as stream:
            print(json.dumps({'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}), flush=True)
    else:
        owner = read(Path(args.run_dir) / 'owner.json')
        identities = [owner['owner'], *owner.get('children', [])]
        if owner.get('api'):
            identities.append(owner['api'])
        print(json.dumps({'running': any(same_process(p) for p in identities if p), 'owner': owner}), flush=True)


if __name__ == '__main__':
    main()
