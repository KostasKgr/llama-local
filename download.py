#!/usr/bin/env python3
"""Download pinned upstream artifacts, streaming to disk and verifying SHA256."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / 'downloads'
MODELS = ROOT / 'models/qwen36-35b-a3b'
RELEASE = 'https://github.com/ggml-org/llama.cpp/releases/download/b11433/'
ARTIFACTS = [
    (RELEASE, 'llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz', DOWNLOADS, '4ee66f423e5e4152b87ba7f46dd30a680b4e1d4f5205bbb2cbf0d9daf1662e5e'),
    (RELEASE, 'cudart-llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz', DOWNLOADS, 'e7b5eef48c4d30d78e1130f80b9f7d0113c1b9d039adac5b6397406a92b1235b'),
    ('https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF/resolve/85b84632d37b84b23c7fb7ee25e47c32318c585f/', 'IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00001-of-00002.gguf', MODELS, '48d598e74fe1e12ce128413bfca14c73860989c63c8496b5eea0b2d0450d52dc'),
    ('https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF/resolve/85b84632d37b84b23c7fb7ee25e47c32318c585f/', 'IQ3_S/Qwen3.6-35B-A3B-IQ3_S-00002-of-00002.gguf', MODELS, 'c19ef1b1ae83061f950393ef16675574c9c2b9fc97bf9de2e4251813f1df5efa'),
    ('https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/baec3ebee244827cda0f4557eafa8b28f7545fa6/', 'dflash-Qwen3.6-35B-A3B-Q8_0.gguf', MODELS, '339030c9dad4f410c7ef2d6cbd6e38251a84fabb59c44db992538c2451a0efee'),
]
def fetch(artifact):
    base, name, directory, expected = artifact
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + '.part')
        print('Downloading ' + name, flush=True)
        subprocess.run(['curl', '-fL', '--retry', '5', '--retry-delay', '2', '--silent', '--show-error', '-C', '-', '-o', str(partial), base + name], check=True)
        partial.rename(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f'SHA256 mismatch: {name}: {actual} != {expected}')
    print('Verified ' + name + ': ' + actual, flush=True)
    return {'file': str(path.relative_to(ROOT)), 'url': base + name, 'sha256': actual, 'size': path.stat().st_size}
if __name__ == '__main__':
    DOWNLOADS.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(fetch, ARTIFACTS))
    (ROOT / 'ARTIFACTS.json').write_text(json.dumps(manifest, indent=2) + '\n')
