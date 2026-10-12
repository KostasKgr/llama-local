#!/usr/bin/env python3
"""Verify and extract the two upstream CUDA 12.8 release archives; never compile."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'llama.cpp-runtime'
ARCHIVES = {
    'llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz':
        '4ee66f423e5e4152b87ba7f46dd30a680b4e1d4f5205bbb2cbf0d9daf1662e5e',
    'cudart-llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz':
        'e7b5eef48c4d30d78e1130f80b9f7d0113c1b9d039adac5b6397406a92b1235b',
}


def main():
    RUNTIME.mkdir(exist_ok=True)
    for filename, expected in ARCHIVES.items():
        path = ROOT / 'downloads' / filename
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError(f'SHA256 mismatch: {filename}')
        with tarfile.open(path) as archive:
            members = archive.getnames()
            top = {name.split('/')[0] for name in members}
            if len(top) != 1 or any(name.startswith('/') or '..' in Path(name).parts for name in members):
                raise RuntimeError('Unexpected archive structure')
            print(filename, 'top-level:', next(iter(top)))
        subprocess.run(['tar', 'xf', str(path), '-C', str(RUNTIME), '--strip-components=1'], check=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(RUNTIME))
    version = subprocess.check_output([str(RUNTIME / 'llama-server'), '--version'], env=env, text=True,
                                      stderr=subprocess.STDOUT)
    devices = subprocess.check_output([str(RUNTIME / 'llama-server'), '--list-devices'], env=env, text=True,
                                      stderr=subprocess.STDOUT)
    help_text = subprocess.check_output([str(RUNTIME / 'llama-server'), '--help'], env=env, text=True,
                                       stderr=subprocess.STDOUT)
    (ROOT / 'downloads/server-help.txt').write_text(help_text)
    runtime = ctypes.CDLL(str(RUNTIME / 'libcudart.so.12'))
    runtime_version = ctypes.c_int()
    error = runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
    if error:
        raise RuntimeError(f'cudaRuntimeGetVersion failed: {error}')
    info = {'version': version, 'devices': devices, 'cuda_runtime_version': runtime_version.value,
            'llama_server_sha256': hashlib.sha256((RUNTIME / 'llama-server').read_bytes()).hexdigest()}
    (ROOT / 'RUNTIME.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
