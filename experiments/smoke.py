"""Fresh-process API smoke check; always cleans up its owned server."""
import base64
import json
import os
import uuid
from pathlib import Path
import subprocess
import signal
import struct
import sys
import time
import zlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, probes

def interrupted(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupted)

name = sys.argv[1]
os.environ['LLAMA_SCOPE'] = name + '-trial-' + uuid.uuid4().hex
p = core.load(name)
core.preflight(p)
out = core.ROOT / 'logs' / name / time.strftime('smoke-%Y%m%d-%H%M%S')
out.mkdir(parents=True, exist_ok=False)
record = {**core.snapshot(name), 'stable': False}
base = f"http://127.0.0.1:{p['port']}"
process = None
try:
    with (out / 'server.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(core.ROOT / 'llama.py'), 'start', name], stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 300
    while True:
        if process.poll() is not None:
            raise RuntimeError('Server exited during startup')
        try:
            if probes.request(base, '/health', timeout=2)['status'] == 'ok':
                break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Startup timeout')
        time.sleep(1)
    record['models'] = probes.request(base, '/v1/models')
    assert any(m['id'] == p['alias'] for m in record['models']['data'])
    record['response'] = probes.request(base, '/v1/chat/completions', {
        'model': p['alias'], 'messages': [{'role': 'user', 'content': 'What is 2 + 2? Answer with the number only.'}],
        'max_tokens': 32, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False}})
    assert '4' in record['response']['choices'][0]['message']['content']
    if p.get('mmproj'):
        # Generate two solid PNGs locally: no external image dependency or download.
        def chunk(kind, data):
            return (struct.pack('>I', len(data)) + kind + data +
                    struct.pack('>I', zlib.crc32(kind + data)))

        record['vision'] = []
        for color, rgb in [('red', b'\xff\x00\x00'), ('blue', b'\x00\x00\xff')]:
            png = (b'\x89PNG\r\n\x1a\n' +
                   chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0)) +
                   chunk(b'IDAT', zlib.compress((b'\x00' + rgb * 128) * 128)) +
                   chunk(b'IEND', b''))
            response = probes.request(base, '/v1/chat/completions', {
                'model': p['alias'], 'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {
                        'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
                    {'type': 'text', 'text': 'What color is this image? Answer with one color word only.'}]}],
                'max_tokens': 32, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False}})
            record['vision'].append({'expected': color, 'response': response})
            assert color in response['choices'][0]['message']['content'].lower()
    record['scope'] = subprocess.check_output(['systemctl', '--user', 'show', p['scope'] + '.scope',
        '-p', 'MemoryHigh', '-p', 'MemoryMax', '-p', 'MemorySwapMax'], text=True)
    assert 'MemoryHigh=8589934592' in record['scope'] and 'MemoryMax=9663676416' in record['scope'] and 'MemorySwapMax=0' in record['scope']
    record['gpu'] = probes.gpu()
    record['stable'] = True
finally:
    if process is not None:
        probes.stop(p['scope'])
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
print(out)
