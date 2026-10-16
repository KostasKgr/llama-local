"""Fresh-process API smoke check; always cleans up its owned server."""
import json
import os
import uuid
from pathlib import Path
import subprocess
import signal
import sys
import time
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
    record['scope'] = subprocess.check_output(['systemctl', '--user', 'show', p['scope'] + '.scope',
        '-p', 'MemoryHigh', '-p', 'MemoryMax', '-p', 'MemorySwapMax'], text=True)
    assert 'MemoryHigh=6442450944' in record['scope'] and 'MemoryMax=8589934592' in record['scope'] and 'MemorySwapMax=0' in record['scope']
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
