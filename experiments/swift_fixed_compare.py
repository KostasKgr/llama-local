"""Fixed-prompt A/B driver for Swift: wait for health, run the smoke checks (text,
red/blue image) which also warm the server, then N reps of swift_benchmark.py
with 0.5 s GPU sampling. Targets whatever server is on 127.0.0.1:1235; it never
starts or stops one.

  python3 experiments/swift_fixed_compare.py LABEL OUTDIR [REPS]
"""
import base64, json, struct, subprocess, sys, time, zlib
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lib import probes

label, out, reps = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3
out.mkdir(parents=True, exist_ok=True)
base, alias = 'http://127.0.0.1:1235', 'swift-1.5-qwen3.8-27b-iq3s-mtp'
t0 = time.monotonic()
while True:
    try:
        if probes.request(base, '/health', timeout=2)['status'] == 'ok':
            break
    except Exception:
        pass
    if time.monotonic() - t0 > 600:
        sys.exit('startup timeout')
    time.sleep(1)
rec = {'label': label, 'ready_after_s': round(time.monotonic() - t0, 1)}
props = probes.request(base, '/props')
rec['build'] = props.get('build_info')
chat = lambda content: probes.request(base, '/v1/chat/completions', {
    'model': alias, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 32, 'temperature': 0,
    'chat_template_kwargs': {'enable_thinking': False}})['choices'][0]['message']['content']
rec['text'] = chat('What is 2 + 2? Answer with the number only.')
assert '4' in rec['text'], rec['text']
def png(rgb):
    c = lambda k, d: struct.pack('>I', len(d)) + k + d + struct.pack('>I', zlib.crc32(k + d))
    return (b'\x89PNG\r\n\x1a\n' + c(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0)) +
            c(b'IDAT', zlib.compress((b'\x00' + rgb * 128) * 128)) + c(b'IEND', b''))
for color, rgb in [('red', b'\xff\x00\x00'), ('blue', b'\x00\x00\xff')]:
    ans = chat([{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png(rgb)).decode()}},
                {'type': 'text', 'text': 'What color is this image? Answer with one color word only.'}])
    rec[color] = ans
    assert color in ans.lower(), ans
print(json.dumps(rec), flush=True)
(out / f'{label}-checks.json').write_text(json.dumps(rec, indent=2))
for i in range(1, reps + 1):
    with (out / f'{label}-{i}-gpu.csv').open('w') as f:
        smi = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,'
                                'utilization.gpu,clocks_event_reasons.active', '--format=csv,nounits', '-lms', '500'], stdout=f)
    try:
        subprocess.run([sys.executable, str(REPO / 'experiments/swift_benchmark.py'), '--label', f'{label}-{i}'], check=True)
    finally:
        smi.terminate(); smi.wait()
    src = REPO / 'logs/swift15' / f'benchmark-{label}-{i}.json'
    (out / src.name).write_text(src.read_text())
