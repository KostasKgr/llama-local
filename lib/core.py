"""Pinned local llama.cpp installation and managed server lifecycle."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
PROFILES = {'swift15': ('SWIFT', 'draft-mtp'),
            'qwen36': ('QWEN', 'draft-dflash'),
            'lfm25': ('LFM', 'none'),
            'bonsai2': ('BONSAI', 'none'),
            'bonsai2mtp': ('BONSAI_MTP', 'draft-mtp')}
FIELDS = {'context': int, 'batch': int, 'ubatch': int, 'draft_max': int, 'port': int,
          'cache_k': str, 'cache_v': str, 'load_mode': str, 'spec': str}


def load(name, overrides=None):
    if name not in PROFILES:
        raise ValueError('Unknown profile')
    p = json.loads((ROOT / 'profiles' / f'{name}.json').read_text())
    scope = os.environ.get('LLAMA_SCOPE', name)
    if scope != name and not (scope.startswith(name + '-trial-') and all(c.isalnum() or c == '-' for c in scope)):
        raise ValueError('Invalid owned scope name')
    p['scope'] = scope
    prefix, supported_spec = PROFILES[name]
    for key, cast in FIELDS.items():
        env_key = 'SPEC_TYPE' if key == 'spec' else key.upper()
        if f'{prefix}_{env_key}' in os.environ:
            p[key] = cast(os.environ[f'{prefix}_{env_key}'])
    p.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if any(p[k] <= 0 for k in ('context', 'batch', 'ubatch', 'draft_max')):
        raise ValueError('Context, batch, microbatch and draft maximum must be positive')
    if p['ubatch'] > p['batch'] or not 1 <= p['port'] <= 65535:
        raise ValueError('Invalid microbatch or port')
    if p['spec'] not in ('none', supported_spec):
        raise ValueError('Unsupported speculation for this profile')
    if p['load_mode'] not in ('auto', 'dio', 'mmap', 'mlock'):
        raise ValueError('Unsupported load mode')
    if any(p[k] not in ('f16', 'f32', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl') for k in ('cache_k', 'cache_v')):
        raise ValueError('Unsupported KV format')
    return p


def command(p):
    server = ROOT / 'runtimes' / p['runtime'] / 'llama-server'
    # OpenCode thread naming can displace the main prompt and cause cache thrashing;
    # reserve 2 GiB of RAM cache for both profiles to retain displaced prompt states.
    args = [str(server), '-m', str(ROOT / 'models' / p['model']), '--load-mode', p['load_mode'],
            '-ngl', '999', '-fa', 'on', '-c', str(p['context']), '-np', '1',
            '-b', str(p['batch']), '-ub', str(p['ubatch']), '--ctx-checkpoints', '2', '--cache-ram', '2048',
            '--cache-type-k', p['cache_k'], '--cache-type-v', p['cache_v'], '--spec-type', p['spec'],
            '--host', '127.0.0.1', '--port', str(p['port']), '--jinja', '--alias', p['alias'],
            '--metrics', '--perf', '--log-verbosity', '4']
    if p['fit_off']:
        args += ['--fit', 'off']
    if 'sampling' in p:
        sampling = p['sampling']
        args += ['--temp', str(sampling['temperature']), '--top-k', str(sampling['top_k']),
                 '--repeat-penalty', str(sampling['repeat_penalty'])]
        for key, flag in [('top_p', '--top-p'), ('min_p', '--min-p'),
                          ('presence_penalty', '--presence-penalty'), ('repeat_last_n', '--repeat-last-n')]:
            if key in sampling:
                args += [flag, str(sampling[key])]
    if p['spec'] != 'none':
        args += ['--spec-draft-n-max', str(p['draft_max'])]
        for key, flag in [('draft_min', '--spec-draft-n-min'), ('draft_p_min', '--spec-draft-p-min')]:
            if key in p:
                args += [flag, str(p[key])]
        if p['spec'] == 'draft-dflash':
            args += ['-md', str(ROOT / 'models' / p['draft']), '-ngld', '999']
    return args


def scope_active(p):
    return subprocess.run(['systemctl', '--user', 'is-active', p['scope'] + '.scope'],
                          capture_output=True, timeout=10).returncode == 0


def preflight(p):
    args = command(p)
    if not os.access(args[0], os.X_OK):
        raise ValueError('Runtime missing; run install first')
    files = [p['model']]
    if p['spec'] == 'draft-dflash':
        files += [p['draft']]
        files += [a['file'] for a in manifest('models')['qwen36'] if '-of-' in a['file']]
    if any(not (ROOT / 'models' / f).is_file() for f in files):
        raise ValueError('Model artifact missing; run install first')
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(('127.0.0.1', p['port'])) == 0:
            raise ValueError(f"Port {p['port']} occupied; stop its owner first")
    if scope_active(p):
        raise ValueError('Profile scope already active')


def start(p):
    preflight(p)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(ROOT / 'runtimes' / p['runtime']) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    profile = p['scope'].split('-trial-', 1)[0]
    log_dir = ROOT / 'logs' / profile / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    log_dir.mkdir(parents=True, exist_ok=False)
    args = ['systemd-run', '--user', '--scope', '--unit=' + p['scope'], '--collect',
            '-p', 'MemoryHigh=6G', '-p', 'MemoryMax=8G', '-p', 'MemorySwapMax=0',
            'bash', '-c', 'set -o pipefail; log=$1; shift; "$@" 2>&1 | tee -- "$log"',
            'llama-log', str(log_dir / 'server.log')] + command(p)
    print('Server log:', log_dir / 'server.log', flush=True)
    print(json.dumps(p, indent=2), flush=True)
    os.execvpe(args[0], args, env)


def stop(p):
    subprocess.run(['systemctl', '--user', 'stop', p['scope'] + '.scope'], check=True, timeout=30)


def manifest(kind):
    return json.loads((ROOT / 'manifests' / f'{kind}.json').read_text())


def verify(path, artifact):
    if 'size' in artifact and path.stat().st_size != artifact['size']:
        raise ValueError(f'Size mismatch: {path}')
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != artifact['sha256']:
        raise ValueError(f'SHA256 mismatch: {path}')


def fetch(base, artifact):
    path = base / artifact['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_name(path.name + '.part')
        subprocess.run(['curl', '-fL', '--retry', '3', '-C', '-', '-o', str(part), artifact['url']], check=True)
        verify(part, artifact)
        part.replace(path)
    else:
        verify(path, artifact)
    return path


def install(name):
    p = load(name)
    for a in manifest('models')[name]:
        fetch(ROOT / 'models', a)
    archives = [fetch(ROOT / 'downloads', a) for a in manifest('runtimes')[p['runtime']]]
    runtime = ROOT / 'runtimes' / p['runtime']
    runtime.mkdir(parents=True, exist_ok=True)
    for path in archives:
        with tarfile.open(path) as archive:
            # Prism archives include a harmless root-directory entry before the single top-level folder.
            members = [m for m in archive.getmembers() if m.name not in ('.', './')]
            tops = {m.name.split('/')[0] for m in members}
            if len(tops) != 1 or any(m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
                raise ValueError('Unexpected archive structure')
            for m in members:
                m.name = m.name.partition('/')[2]
            archive.extractall(runtime, members=[m for m in members if m.name], filter='data')
    env = dict(os.environ, LD_LIBRARY_PATH=str(runtime))
    info = {'profile': name, 'runtime': p['runtime'], 'server_sha256': hashlib.sha256((runtime / 'llama-server').read_bytes()).hexdigest()}
    for key, flag in [('version', '--version'), ('devices', '--list-devices')]:
        info[key] = subprocess.check_output([str(runtime / 'llama-server'), flag], env=env, stderr=subprocess.STDOUT, text=True)
    cuda = ctypes.CDLL(str(runtime / 'libcudart.so.12'))
    version = ctypes.c_int()
    if cuda.cudaRuntimeGetVersion(ctypes.byref(version)) != 0:
        raise ValueError('Unable to query bundled CUDA runtime')
    info['cuda_runtime_version'] = version.value
    (runtime / 'installation.json').write_text(json.dumps(info, indent=2) + '\n')


def snapshot(name):
    p = load(name)
    server = ROOT / 'runtimes' / p['runtime'] / 'llama-server'
    return {'profile': p, 'command': command(p), 'model_artifacts': manifest('models')[name],
            'runtime_artifacts': manifest('runtimes')[p['runtime']],
            'server_sha256': hashlib.sha256(server.read_bytes()).hexdigest()}
