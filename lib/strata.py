"""Pinned Strata acquisition and a foreground WSL supervisor for its Windows owner."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import ntpath
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen
import uuid
import zipfile

from . import core

DEFAULT_ROOT = r'C:\AI\llama-local\strata'
OVERRIDES = {'context', 'port', 'spec', 'draft_max', 'parallel'}


def validate(p):
    scope = p.get('scope', '')
    if scope != 'gyros' and not re.fullmatch(r'gyros-trial-[A-Za-z0-9-]+', scope):
        raise ValueError('Invalid owned Strata scope')
    invalid = set(p) & (set(core.FIELDS) - OVERRIDES)
    if invalid:
        raise ValueError('Unsupported Strata overrides: ' + ', '.join(sorted(invalid)))
    if (type(p['context']) is not int or not 1024 <= p['context'] <= 131072
            or type(p['port']) is not int or not 1 <= p['port'] <= 65535
            or type(p['parallel']) is not int or p['parallel'] != 1):
        raise ValueError('Strata requires 1024..131072 context, one slot and a valid port')
    if p['spec'] not in ('none', 'dflash2') or type(p['draft_max']) is not int or not 2 <= p['draft_max'] <= 6:
        raise ValueError('Strata supports none/dflash2 and draft maximum 2..6')


def windows_root():
    value = core.load_env_file().get('LLAMA_STRATA_WINDOWS_ROOT', '').strip() or DEFAULT_ROOT
    path = PureWindowsPath(value)
    if (not path.is_absolute() or not path.drive or path.drive.startswith('\\')
            or '..' in path.parts or any(c in value for c in "'\"\r\n\x00")):
        raise ValueError('LLAMA_STRATA_WINDOWS_ROOT must be an absolute local Windows path')
    value = ntpath.normpath(value)
    legacy = r'c:\ai\strata'
    if value.lower() == legacy or value.lower().startswith(legacy + '\\'):
        raise ValueError('Use separate managed storage; the existing Strata installation is excluded')
    if len(path.parts) < 3:
        raise ValueError('Strata storage must be a dedicated directory below a drive root')
    return value


def root():
    return core.wsl_path(windows_root())


def win(path):
    relative = Path(path).relative_to(root())
    return windows_root() + ('\\' + str(relative).replace('/', '\\') if relative.parts else '')


def pins():
    return core.manifest('strata')


def control():
    return root() / 'runtimes' / ('control-' + pins()['control_version'])


def engine():
    return root() / 'runtimes' / ('engine-' + pins()['engine_version'] + '-cuda13') / 'strata.exe'


def python():
    return control() / 'python' / 'python.exe'


def digest(path):
    path = Path(path)
    if path.stat().st_size >= 256 * 1024 ** 2 and path.resolve().is_relative_to(root().resolve()):
        stage_bridge()
        return json.loads(helper('hash', '--file', win(path), timeout=300).stdout)['sha256']
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_file(path, artifact):
    if 'size' in artifact and path.stat().st_size != artifact['size']:
        raise ValueError('Size mismatch: ' + str(path))
    if digest(path) != artifact['sha256']:
        raise ValueError('SHA256 mismatch: ' + str(path))


def generated_path(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
            or path.as_posix() != name
            or not (name.startswith('packs/gyros/') or name.startswith('dflash2/'))):
        raise ValueError('Unsafe generated inventory path')
    return root() / name


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    deadline = time.monotonic() + 3
    while True:
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.03)


def bridge():
    source = core.ROOT / 'lib' / 'strata_windows.py'
    return root() / 'tools' / ('owner-' + digest(source) + '.py')


def stage_bridge():
    target = bridge()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.part')
        shutil.copyfile(core.ROOT / 'lib' / 'strata_windows.py', temporary)
        temporary.replace(target)
    if digest(target) != digest(core.ROOT / 'lib' / 'strata_windows.py'):
        raise ValueError('Windows owner source integrity mismatch')
    return target


def helper(action, *args, timeout=60, owner_bridge=None):
    return subprocess.run([str(python()), '-B', '-s', win(owner_bridge or bridge()), action, '--root', windows_root(), *args],
                          cwd=control(), env=isolated_env(), text=True, encoding='utf-8', errors='replace',
                          capture_output=True, timeout=timeout, check=True)


def archive_files(path):
    """Yield regular archive members after validating all names and Windows aliases."""
    archive = zipfile.ZipFile(path) if str(path).endswith('.zip') else tarfile.open(path)
    with archive:
        entries = archive.infolist() if isinstance(archive, zipfile.ZipFile) else archive.getmembers()
        seen = set()
        for entry in entries:
            name = entry.filename if isinstance(archive, zipfile.ZipFile) else entry.name
            name = name.removeprefix('./').rstrip('/')
            if not name:
                continue
            parts = PurePosixPath(name).parts
            if (name.startswith(('/', '\\')) or '\\' in name or ':' in name or '..' in parts
                    or any(part.endswith((' ', '.')) for part in parts)
                    or any(part.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                           *('COM' + str(i) for i in range(1, 10)), *('LPT' + str(i) for i in range(1, 10))}
                           for part in parts)):
                raise ValueError('Unsafe archive path: ' + name)
            if name.lower() in seen:
                raise ValueError('Duplicate archive member: ' + name)
            seen.add(name.lower())
            directory = entry.is_dir() if isinstance(archive, zipfile.ZipFile) else entry.isdir()
            if isinstance(archive, zipfile.ZipFile):
                mode = entry.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise ValueError('Archive symlink refused')
            elif not directory and not entry.isfile():
                raise ValueError('Archive link or special file refused')
        for entry in entries:
            name = (entry.filename if isinstance(archive, zipfile.ZipFile) else entry.name).removeprefix('./').rstrip('/')
            directory = entry.is_dir() if isinstance(archive, zipfile.ZipFile) else entry.isdir()
            if name and not directory:
                with (archive.open(entry) if isinstance(archive, zipfile.ZipFile) else archive.extractfile(entry)) as stream:
                    yield name, stream


def extract(archive, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + '.stage-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name, stream in archive_files(archive):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as out:
                shutil.copyfileobj(stream, out)
        if target.exists():
            raise ValueError('Refusing to overwrite an installed runtime: ' + str(target))
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_extraction(archive, target):
    for name, stream in archive_files(archive):
        path = target / name
        if path.is_symlink() or not path.is_file() or digest(path) != hashlib.file_digest(stream, 'sha256').hexdigest():
            raise ValueError('Installed runtime differs from pinned archive: ' + str(path))


def fetch_gguf_py():
    spec = pins()['gguf_py']
    base = root() / 'tools' / 'gguf-py'
    for entry in [*spec['files'], spec['license']]:
        path = base / entry['file']
        if path.exists():
            data = path.read_bytes()
        else:
            url = ('https://raw.githubusercontent.com/ggml-org/llama.cpp/' + spec['commit']
                   + '/gguf-py/' + entry['file'])
            with urlopen(url, timeout=60) as response:
                data = response.read()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != entry['size'] or blob != entry['git_blob']:
            raise ValueError('gguf-py Git blob mismatch: ' + entry['file'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(path)


def run_tool(script, *args):
    env = isolated_env()
    return subprocess.run([str(python()), '-B', '-s', win(control() / 'tools' / script), *map(str, args)],
                          cwd=control(), env=env, check=True)


def isolated_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('STRATA_', 'PYTHON', 'HF_', 'TRANSFORMERS_'))}
    for key, suffix in {'APPDATA': 'appdata', 'LOCALAPPDATA': 'localappdata', 'USERPROFILE': 'home',
                        'HOME': 'home', 'STRATA_INSTALL_DIR': 'install', 'STRATA_PROFILE_DIR': 'profiles',
                        'STRATA_RUNTIME_DIR': 'runtime-state', 'STRATA_COMPONENTS_DIR': 'components'}.items():
        env[key] = win(root() / 'state' / suffix)
    env.update(STRATA_GGUF_PY=win(root() / 'tools' / 'gguf-py'), STRATA_NO_ACQUIRE='1',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PIP_NO_INDEX='1')
    # WSL only forwards arbitrary variables to Windows when WSLENV names them.
    keys = [k for k in env if k.startswith(('STRATA_', 'PYTHON', 'HF_', 'TRANSFORMERS_', 'PIP_'))
            or k in ('APPDATA', 'LOCALAPPDATA', 'USERPROFILE', 'HOME')]
    env['WSLENV'] = ':'.join(keys)
    return env


def install():
    base = root()
    if (any((base / 'state').glob('gyros*.json')) or running(core.load('gyros'))
            or core.scope_active({'scope': 'gyros'})):
        raise ValueError('Stop the owned Gyro server before installation')
    ancestor = base
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not (base / 'installation.json').exists() and shutil.disk_usage(ancestor).free < 80 * 1024 ** 3:
        # A resumed model download has already consumed part of this allowance.
        partial = sum(p.stat().st_size for p in base.glob('models/**/*.part')) if base.exists() else 0
        if shutil.disk_usage(ancestor).free + partial < 80 * 1024 ** 3:
            raise ValueError('Initial Strata installation needs 80 GiB free (including its partial download)')
    (base / 'state').mkdir(parents=True, exist_ok=True)
    with (base / 'state' / 'install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for artifact in pins()['packages']:
            archive = core.fetch(base / 'downloads', artifact)
            target = control() if artifact['file'].endswith('.zip') else engine().parent
            if target.exists():
                verify_extraction(archive, target)
            else:
                print('Extracting', artifact['file'], flush=True)
                extract(archive, target)
        stage_bridge()
        bundle = json.loads((control() / 'BUNDLE.json').read_text())
        if (bundle['component'] != 'strata-control' or bundle['version'] != pins()['control_version']
                or bundle['target'] != 'windows-x86_64' or bundle['python']['version'] != '3.12.15'
                or (control() / 'COMMIT').read_text().strip() != pins()['source_commit']):
            raise ValueError('Unexpected Control bundle provenance')
        manifest = engine().parent / 'engine-component.json'
        verify_file(manifest, {'sha256': pins()['engine_manifest_sha256']})
        probe = helper('probe', timeout=180)
        (base / 'state' / 'runtime-probe.json').write_text(probe.stdout)
        for artifact in core.manifest('models')['gyros']:
            core.fetch(base / 'models', artifact, verifier=verify_file)
        fetch_gguf_py()
        previous = base / 'installation.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            if (old['manifest_sha256'] != digest(core.ROOT / 'manifests/strata.json')
                    or old['model_manifest'] != core.manifest('models')['gyros']):
                raise ValueError('Preparation inputs changed; choose a fresh managed storage directory')
            for name, artifact in old['generated'].items():
                verify_file(generated_path(name), artifact)
        pack = base / 'packs' / 'gyros'
        if not pack.exists():
            staging = pack.with_name('gyros.stage-' + uuid.uuid4().hex)
            staging.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_tool('iq_pack.py', '--gguf', win(base / 'models' / core.load('gyros')['model']),
                         '--out', win(staging), '--compat-bf16')
                for needed in ('native_experts.txt', 'index.txt', 'dense.bin', 'tokenizer/vocab.json'):
                    if not (staging / needed).is_file():
                        raise ValueError('Incomplete Gyro preparation: ' + needed)
                staging.replace(pack)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        drafter = base / 'dflash2'
        drafter.mkdir(exist_ok=True)
        stock, selected = pins()['generated_drafters']
        if not (drafter / stock['file']).exists():
            run_tool('dflash2_pack.py', win(base / 'models/gyros/drafter/model.safetensors'),
                     win(base / 'models/gyros/drafter/config.json'), win(drafter / (stock['file'] + '.part')), '--q4_0')
            verify_file(drafter / (stock['file'] + '.part'), stock)
            (drafter / (stock['file'] + '.part')).replace(drafter / stock['file'])
        verify_file(drafter / stock['file'], stock)
        if not (drafter / selected['file']).exists():
            run_tool('dflash2_pack.py', '--convert-selector', win(drafter / stock['file']),
                     win(drafter / (selected['file'] + '.part')))
            verify_file(drafter / (selected['file'] + '.part'), selected)
            (drafter / (selected['file'] + '.part')).replace(drafter / selected['file'])
        verify_file(drafter / selected['file'], selected)
        shutil.copyfile(control() / 'data' / 'draft_vocab.bin', drafter / 'draft_vocab.bin')
        generated = {str(p.relative_to(base)): {'size': p.stat().st_size, 'sha256': digest(p)}
                     for folder in (pack, drafter) for p in folder.rglob('*') if p.is_file()}
        atomic(base / 'installation.json', {'manifest_sha256': digest(core.ROOT / 'manifests/strata.json'),
               'model_manifest': core.manifest('models')['gyros'], 'generated': generated,
               'probe': json.loads(probe.stdout), 'installed_at': datetime.now(timezone.utc).isoformat()})
        print('Strata installed:', windows_root())


def verify():
    base = root()
    info = json.loads((base / 'installation.json').read_text())
    if info['manifest_sha256'] != digest(core.ROOT / 'manifests/strata.json'):
        raise ValueError('Strata recipe changed; run install')
    if info['model_manifest'] != core.manifest('models')['gyros']:
        raise ValueError('Strata model pins changed; run install')
    for artifact in core.manifest('models')['gyros']:
        verify_file(base / 'models' / artifact['file'], artifact)
    for artifact in pins()['packages']:
        archive = base / 'downloads' / artifact['file']
        verify_file(archive, artifact)
        verify_extraction(archive, control() if artifact['file'].endswith('.zip') else engine().parent)
    verify_file(engine().parent / 'engine-component.json', {'sha256': pins()['engine_manifest_sha256']})
    for name, artifact in info['generated'].items():
        verify_file(generated_path(name), artifact)
    for artifact in pins()['generated_drafters']:
        verify_file(base / 'dflash2' / artifact['file'], artifact)
    verify_gguf_py()
    print('All pinned Strata artifacts and generated outputs verified')


def verify_gguf_py():
    spec = pins()['gguf_py']
    for entry in [*spec['files'], spec['license']]:
        data = (root() / 'tools' / 'gguf-py' / entry['file']).read_bytes()
        if (len(data) != entry['size'] or hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                != entry['git_blob']):
            raise ValueError('gguf-py integrity mismatch: ' + entry['file'])


def config(p, run_dir):
    base = root()
    model = win(base / 'models' / p['model'])
    args = ['--pack', win(base / 'packs/gyros'), '--native', model, '--ple-gguf', model,
            '--expert-profile', win(control() / 'data/expert-profile.bin'), '--expert-cache', 'auto',
            '--prefill', 'auto', '--max-context', str(p['context']), '--kv', 'int8',
            '--resident-experts', '--ram-keep-gib', str(p['ram_keep_gib']), '--ple-io', 'direct',
            '--vram-reserve-mib', str(p['vram_reserve_mib'])]
    if p['spec'] == 'dflash2':
        args += ['--draft', 'dflash2', '--draft-model', win(base / 'dflash2' / pins()['generated_drafters'][1]['file']),
                 '--draft-fallback', 'none', '--mtp-draft-vocab', win(base / 'dflash2/draft_vocab.bin'),
                 '--spec', str(p['draft_max']), '--draft-conf', str(p['draft_conf'])]
    else:
        # Native IQ packs require a >=2-row verify buffer even without a drafter.
        args += ['--spec', '2', '--suffix-draft', '0', '--lookup-max-t', '0']
    return {'exe': win(engine()), 'args': args, 'cwd': win(control()),
            'lib_dirs': [win(engine().parent)], 'tokenizer': win(base / 'packs/gyros/tokenizer'),
            'model_name': p['alias'], 'host': '127.0.0.1', 'port': p['port'],
            'log': win(run_dir / 'engine.log'), 'parallel': 1, 'engine_ready_s': 900,
            'drafting': 'dflash2' if p['spec'] == 'dflash2' else 'off',
            'user_args': ['--vram-reserve-mib', str(p['vram_reserve_mib'])] +
                         (['--spec', str(p['draft_max']), '--draft-conf', str(p['draft_conf'])]
                          if p['spec'] != 'none' else []),
            'open_browser': False, 'idle_unload_s': 0}


def command(p):
    return [str(python()), '-B', '-s', win(bridge()), 'serve', '--root', windows_root(),
            '--run-dir', win(root() / 'runs' / '<run-token>')]


def state(p):
    return root() / 'state' / (p['scope'] + '.json')


def owner(p):
    try:
        record = json.loads(state(p).read_text())
        run = root() / 'runs' / record['token']
        if len(record['token']) != 32 or not all(c in '0123456789abcdef' for c in record['token']):
            raise ValueError('Invalid Strata ownership token')
        if record.get('scope', p['scope']) != p['scope']:
            raise ValueError('Strata ownership scope mismatch')
        return record, run
    except FileNotFoundError:
        return None, None


def running(p):
    record, run = owner(p)
    if record is None:
        return False
    if not (run / 'owner.json').exists():
        return True  # startup reservation: never start over an incomplete owner
    name = record.get('bridge', bridge().name)
    if not re.fullmatch(r'owner-[0-9a-f]{64}\.py', name):
        raise ValueError('Invalid owned Windows bridge')
    source = root() / 'tools' / name
    result = helper('check', '--run-dir', win(run), owner_bridge=source)
    return json.loads(result.stdout)['running']


def port_open(port):
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(('127.0.0.1', port)) == 0


def preflight(p):
    for path in (python(), engine(), root() / 'installation.json', root() / 'packs/gyros/native_experts.txt',
                 root() / 'packs/gyros/tokenizer/vocab.json', root() / 'models' / p['model']):
        if not path.is_file():
            raise ValueError('Strata artifact missing; run llama.py install gyros: ' + str(path))
    if p['spec'] == 'dflash2':
        for path in (root() / 'dflash2' / pins()['generated_drafters'][1]['file'], root() / 'dflash2/draft_vocab.bin'):
            if not path.is_file():
                raise ValueError('Strata drafter missing; run install gyros')
    verify_file(engine().parent / 'engine-component.json', {'sha256': pins()['engine_manifest_sha256']})
    installation = json.loads((root() / 'installation.json').read_text())
    if installation['manifest_sha256'] != digest(core.ROOT / 'manifests/strata.json'):
        raise ValueError('Strata recipe changed; run install gyros')
    if installation['model_manifest'] != core.manifest('models')['gyros']:
        raise ValueError('Strata model pins changed; run install gyros')
    for name, artifact in installation['generated'].items():
        verify_file(generated_path(name), artifact)
    if (root() / 'models' / p['model']).stat().st_size != core.manifest('models')['gyros'][0]['size']:
        raise ValueError('Full Gyro GGUF size mismatch; run verify gyros')
    if state(p).exists() or core.scope_active(p):
        raise ValueError('Strata scope active or cleanup pending; stop its owner first')
    # Linux cannot bind a recently released Windows port under mirrored networking;
    # only the Windows bind matters for this backend. A real Linux listener still refuses.
    if port_open(p['port']) or not core.windows_port_free(p['port']):
        raise ValueError('Port occupied on Windows or Linux; leave its owner running')


def start(p):
    from . import swap
    swap.guard(p)
    if os.environ.get('LLAMA_STRATA_IN_SCOPE') != '1':
        if os.environ.get('LLAMA_SWAP_MANAGED') == '1':
            for name in ('bolt35', 'swift15'):
                core.wait_windows_exit(core.load(name, scope=name))
            core.wait_port(p['port'], core.windows_port_free)
        preflight(p)
        env = dict(os.environ, LLAMA_STRATA_IN_SCOPE='1')
        argv = ['systemd-run', '--user', '--scope', '--unit=' + p['scope'], '--collect',
                '-p', 'MemoryHigh=8G', '-p', 'MemoryMax=9G', '-p', 'MemorySwapMax=0',
                sys.executable, str(core.ROOT / 'llama.py'), '_strata-serve',
                '--profile-json', json.dumps(p)]
        os.execvpe(argv[0], argv, env)
    supervise(p)


def supervise(p):
    stage_bridge()
    token = uuid.uuid4().hex
    run = root() / 'runs' / token
    run.mkdir(parents=True)
    out = core.ROOT / 'logs' / 'gyros' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    (out / 'run.json').write_text(json.dumps({'token': token, 'windows_run': str(run), 'profile': p}, indent=2))
    atomic(run / 'config.json', config(p, run))
    server = [win(python()), '-B', '-s', '-m', 'serve.server', '--engine', 'strata', '--config', win(run / 'config.json'),
              '--host', '127.0.0.1', '--port', str(p['port']), '--ready-file', win(run / 'ready.json'),
              '--owner-fd', '0', '--shutdown-timeout', '20', '--offline']
    atomic(run / 'launch.json', {'token': token, 'scope': p['scope'], 'port': p['port'], 'alias': p['alias'],
           'context': p['context'], 'cwd': win(control()), 'run_dir': win(run), 'command': server})
    record = {'token': token, 'scope': p['scope'], 'port': p['port'], 'log_dir': str(out),
              'bridge': bridge().name}
    state(p).parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL reserves a scope even when another launcher races systemd activation.
    with state(p).open('x') as stream:
        json.dump(record, stream)
    atomic(run / 'heartbeat.json', {'token': token, 'sequence': 0})
    process = None
    stopping = False
    def stop_requested(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, stop_requested) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        with (out / 'owner.log').open('wb') as log:
            process = subprocess.Popen([str(python()), '-B', '-s', win(bridge()), 'serve', '--root', windows_root(),
                                        '--run-dir', win(run)], cwd=control(), env=isolated_env(),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            print('Server log:', out / 'server.log', flush=True)
            print('Windows run:', run, flush=True)
            sequence = 0
            while process.poll() is None and not stopping:
                sequence += 1
                atomic(run / 'heartbeat.json', {'token': token, 'sequence': sequence})
                time.sleep(1)
            if not stopping and process.returncode:
                raise RuntimeError('Strata Windows owner failed; inspect ' + str(out / 'owner.log'))
    finally:
        atomic(run / 'stop.json', {'token': token})
        if process is not None:
            process.wait(timeout=45)
        for name in ('server.log', 'engine.log', 'owner.json', 'config.json', 'gpu.csv', 'host.csv', 'power.csv'):
            if (run / name).is_file():
                shutil.copyfile(run / name, out / name)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if (process is not None and process.poll() is not None and not (run / 'owner.json').exists()) or not running(p):
            if state(p).exists() and json.loads(state(p).read_text())['token'] == token:
                state(p).unlink()


def stop(p):
    record, run = owner(p)
    if record is None:
        return
    atomic(run / 'stop.json', {'token': record['token']})
    deadline = time.monotonic() + 45
    while running(p):
        if time.monotonic() > deadline:
            raise TimeoutError('Owned Windows Strata job still running; ownership retained')
        time.sleep(0.5)
    # Job termination and listener teardown are asynchronous on Windows.
    while not core.windows_port_free(record['port']):
        if time.monotonic() > deadline:
            raise TimeoutError('Windows listener still occupied after owned job exit; no other process was stopped')
        time.sleep(0.5)
    if core.scope_active(p):
        if state(p).exists() and json.loads(state(p).read_text())['token'] != record['token']:
            return
        subprocess.run(['systemctl', '--user', 'stop', p['scope'] + '.scope'], check=True, timeout=15)
    if state(p).exists() and json.loads(state(p).read_text())['token'] == record['token']:
        state(p).unlink()


def snapshot(p):
    return {'profile': p, 'command': command(p), 'runtime_artifacts': pins()['packages'],
            'model_artifacts': core.manifest('models')['gyros'], 'server_sha256': digest(engine())}
