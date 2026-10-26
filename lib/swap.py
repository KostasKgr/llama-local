"""Optional, two-model llama-swap coordinator using the existing launchers."""
import fcntl
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import tarfile
import uuid

from . import core

PROFILES = ('bolt35', 'swift15')
UNIT = 'llama-local-swap.service'


def directory():
    return core.ROOT / 'logs' / 'swap'


def state_path():
    return directory() / 'active.json'


def managed():
    """Read-only check, including startup and cleanup windows."""
    if state_path().exists():
        return True
    try:
        with (directory() / 'instance.lock').open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        pass
    return False


def guard(p):
    if (p['scope'] in PROFILES and os.environ.get('LLAMA_SWAP_MANAGED') != '1'
            and managed()):
        raise ValueError('Profile is reserved by llama-swap; use llama.py swap stop first')


def ports(port, backend_port):
    if not 1 <= port <= 65535 or not 1 <= backend_port <= 65535 or port == backend_port:
        raise ValueError('Public and backend ports must be distinct and within 1..65535')


def config(port=1235, backend_port=1236):
    ports(port, backend_port)
    models = {}
    for name in PROFILES:
        p = core.load(name)
        launcher = [sys.executable, str(core.ROOT / 'llama.py')]
        models[p['alias']] = {
            'cmd': shlex.join(launcher + ['start', name, '--port', str(backend_port)]),
            'cmdStop': shlex.join(launcher + ['stop', name]),
            'env': ['LLAMA_SWAP_MANAGED=1', 'LLAMA_SCOPE=' + name],
            'proxy': f'http://127.0.0.1:{backend_port}',
            'checkEndpoint': '/health',
        }
    # JSON is valid YAML, so generation needs no extra YAML dependency.
    return json.dumps({
        'healthCheckTimeout': 300,
        'unloadTimeout': 40,
        'globalTTL': 0,
        'sendLoadingState': False,
        'models': models,
        'groups': {'local': {'members': list(models), 'swap': True, 'exclusive': True}},
    }, indent=2) + '\n'


def artifact():
    return json.loads((core.ROOT / 'manifests' / 'llama-swap.json').read_text())


def binary():
    return core.ROOT / 'runtimes' / ('llama-swap-' + artifact()['version']) / 'llama-swap'


def install():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise ValueError('The pinned llama-swap archive supports Linux x86_64 only')
    archive = core.fetch(core.ROOT / 'downloads', artifact())
    target = binary()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name in ('llama-swap', './llama-swap')]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError('Expected one regular llama-swap binary in archive')
        with tar.extractfile(members[0]) as source:
            data = source.read()
    temporary = target.with_suffix('.part')
    temporary.write_bytes(data)
    temporary.chmod(0o755)
    temporary.replace(target)
    print('Installed:', target)


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode and 'not found' not in result.stderr and 'could not be found' not in result.stderr:
        raise ValueError('Cannot query user systemd manager: ' + result.stderr.strip())
    return result.stdout.strip() or 'inactive'


def service_state():
    return unit_state(UNIT)


def preflight(port, backend_port):
    ports(port, backend_port)
    for number in (port, backend_port):
        with socket.socket() as sock:
            sock.settimeout(2)
            if sock.connect_ex(('127.0.0.1', number)) == 0:
                raise ValueError(f'Port {number} occupied; stop its owner first')
    # Fixed scopes and backend port intentionally serialize these profiles.
    for name in PROFILES:
        p = core.load(name, {'port': backend_port})
        p['scope'] = name
        core.preflight(p)


def start(port=1235, backend_port=1236):
    if managed() or service_state() not in ('inactive', 'failed'):
        raise ValueError('llama-swap already active or cleanup pending')
    if not os.access(binary(), os.X_OK):
        raise ValueError('llama-swap missing; run llama.py swap install first')
    preflight(port, backend_port)
    token = uuid.uuid4().hex
    run_dir = directory() / token
    run_dir.mkdir(parents=True)
    (run_dir / 'config.yaml').write_text(config(port, backend_port))
    launcher = [sys.executable, str(core.ROOT / 'llama.py'), 'swap']
    cleanup_cmd = shlex.join(launcher + ['_cleanup', '--token', token])
    # Transient services inherit the manager environment, not the caller's.
    profile_env = []
    for name in PROFILES:
        prefix = core.PROFILES[name][0]
        for key in core.FIELDS:
            variable = prefix + '_' + ('SPEC_TYPE' if key == 'spec' else key.upper())
            if variable in os.environ:
                profile_env.append('--setenv=' + variable + '=' + os.environ[variable])
    subprocess.run([
        'systemd-run', '--user', '--unit=' + UNIT, '--collect', '--service-type=exec',
        '-p', 'TimeoutStopSec=120', '-p', 'KillMode=control-group',
        '-p', 'ExecStopPost=' + cleanup_cmd,
        '-p', 'StandardOutput=append:' + str(run_dir / 'coordinator.log'),
        '-p', 'StandardError=inherit',
    ] + profile_env + launcher + ['_serve', '--token', token, '--run-dir', str(run_dir),
                    '--port', str(port), '--backend-port', str(backend_port)], check=True)
    print('llama-swap starting:', f'http://127.0.0.1:{port}/v1')
    print('Coordinator log:', run_dir / 'coordinator.log')


def serve(token, run_dir, port, backend_port):
    """Private service entrypoint; ExecStopPost also handles abrupt termination."""
    directory().mkdir(parents=True, exist_ok=True)
    with (directory() / 'instance.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another llama-swap coordinator owns the lock') from None
        if state_path().exists():
            raise ValueError('Previous llama-swap cleanup is pending')
        os.environ['LLAMA_SWAP_MANAGED'] = '1'
        preflight(port, backend_port)
        state = {'token': token, 'port': port, 'backend_port': backend_port,
                 'run_dir': str(run_dir)}
        temporary = Path(run_dir) / 'active.json'
        temporary.write_text(json.dumps(state) + '\n')
        temporary.replace(state_path())
        env = dict(os.environ)
        # WSL exposes nvidia-smi outside the user systemd manager's default PATH.
        wsl_tools = '/usr/lib/wsl/lib'
        search_path = env.get('PATH', os.defpath).split(os.pathsep)
        if Path(wsl_tools).is_dir() and wsl_tools not in search_path:
            env['PATH'] = os.pathsep.join(search_path + [wsl_tools])
        try:
            subprocess.run([str(binary()), '-config', str(Path(run_dir) / 'config.yaml'),
                            '-listen', f'127.0.0.1:{port}'], check=True, env=env)
        finally:
            cleanup(token)


def cleanup(token):
    """Only an instance that passed preflight may stop the reserved scopes."""
    try:
        state = json.loads(state_path().read_text())
    except FileNotFoundError:
        return
    if state['token'] != token:
        return
    errors = []
    for name in PROFILES:
        try:
            # is-active alone misses a scope still deactivating during a swap.
            if unit_state(name + '.scope') not in ('inactive', 'failed'):
                subprocess.run(['systemctl', '--user', 'stop', name + '.scope'], check=True, timeout=40)
        except (ValueError, subprocess.SubprocessError) as exc:
            errors.append(name + ': ' + str(exc))
    if errors:
        raise ValueError('Owned scope cleanup failed: ' + '; '.join(errors))
    state_path().unlink(missing_ok=True)


def stop():
    token = json.loads(state_path().read_text())['token'] if state_path().exists() else None
    if service_state() != 'inactive':
        subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True, timeout=130)
    # Retry a previous failed ExecStopPost only after the service has stopped.
    if token is not None:
        cleanup(token)
    print('llama-swap stopped')


def status():
    print('llama-swap:', service_state())
    if state_path().exists():
        print(state_path().read_text().strip())
    for name in PROFILES:
        print(name, 'active' if core.scope_active({'scope': name}) else 'inactive')
