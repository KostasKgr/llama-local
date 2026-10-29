"""Register local model profiles in OpenCode, preserving unrelated JSONC text."""
import difflib
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import tempfile
from . import core, jsonc

PROVIDER = 'llama.cpp'
ENV_KEY = 'LLAMA_OPENCODE_BASE_URL'


@contextmanager
def directory_lock(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        os.close(fd)



def model_config(profile, output_tokens=None):
    metadata = profile['opencode'].copy()
    output = output_tokens if output_tokens is not None else metadata.pop('output_tokens')
    metadata.pop('output_tokens', None)
    if not isinstance(output, int) or output <= 0 or output > profile['context']:
        raise ValueError('Output tokens must be positive and fit within context')
    model = {'name': metadata['name'],
             'capabilities': {'tools': metadata['tool_call'], **metadata['modalities']},
             'limit': {'context': profile['context'], 'output': output}}
    if 'options' in metadata:
        model['settings'] = metadata['options']
    if 'variants' in metadata:
        model['variants'] = [{'id': name, 'settings': settings}
                             for name, settings in metadata['variants'].items()]
    return model


def merge(source, profile, output_tokens=None, update=False, base_url=None):
    base = f"http://127.0.0.1:{profile['port']}/v1" if base_url is None else base_url
    desired = {
        'package': '@opencode/ai/providers/openai-compatible',
        'settings': {'baseURL': base},
        'models': {profile['alias']: model_config(profile, output_tokens)},
    }
    existing = jsonc.parse(source).value
    conflicts = []
    changes = []

    def visit(current, wanted, path):
        for key, value in wanted.items():
            target = path + [key]
            if key not in current:
                changes.append((target, value))
            elif isinstance(value, dict):
                if not isinstance(current[key], dict):
                    raise ValueError(f"Expected object at {'.'.join(target)}")
                visit(current[key], value, target)
            elif current[key] != value:
                conflicts.append('.'.join(target))
                changes.append((target, value))

    visit(existing, {'providers': {PROVIDER: desired}}, [])
    if conflicts and not update:
        raise ValueError('Conflicting settings: ' + ', '.join(conflicts) + '; use --update to replace those fields')
    # Display name is a default for a new provider; preserve an existing name.
    providers = existing.get('providers', {})
    if PROVIDER not in providers:
        desired['name'] = 'llama.cpp (WSL)'
    for path, value in changes:
        source = jsonc.set_value(source, path, value)
    return source


def write_config(path, original, updated):
    """Back up and atomically replace, refusing concurrent edits."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Directory lock avoids leaving a lock file and coordinates our own writers.
    with directory_lock(path.parent) as lock:
        exists = path.exists()
        current = path.read_bytes() if exists else None
        if current != original:
            raise ValueError('Config changed while preparing the edit; rerun the command')
        mode = stat.S_IMODE(path.stat().st_mode) if exists else 0o600
        fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
        backup = None
        try:
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(updated.encode('utf-8'))
                stream.flush()
                os.fsync(stream.fileno())
            if exists:
                fd, backup_name = tempfile.mkstemp(prefix=path.name + '.backup-', dir=path.parent)
                backup = Path(backup_name)
                with os.fdopen(fd, 'wb') as stream:
                    os.fchmod(stream.fileno(), mode)
                    stream.write(original)
                    stream.flush()
                    os.fsync(stream.fileno())
            if (path.read_bytes() if path.exists() else None) != original:
                raise ValueError('Config changed before replacement; rerun the command')
            os.replace(temp, path)
            os.fsync(lock)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return backup


def add(name, config, dry_run=False, update=False, context=None, port=None, output_tokens=None):
    profile = core.load(name, {'context': context, 'port': port})
    base = core.resolve_base_url(profile, ENV_KEY)
    path = Path(config).expanduser().resolve()
    original = path.read_bytes() if path.exists() else None
    source = original.decode('utf-8') if original is not None else '{}\n'
    updated = merge(source, profile, output_tokens, update, base)
    if updated == source:
        print(f"Already registered: {PROVIDER}/{profile['alias']}")
        return
    diff = ''.join(difflib.unified_diff(source.splitlines(keepends=True), updated.splitlines(keepends=True),
                                       fromfile=str(path), tofile=str(path)))
    print(diff, end='')
    if dry_run:
        return
    backup = write_config(path, original, updated)
    print(f"Registered: {PROVIDER}/{profile['alias']}")
    if backup:
        print(f'Backup: {backup}')
