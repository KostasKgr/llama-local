#!/usr/bin/env python3
"""Sequential cold-cache trials with isolated host-memory limits and GPU telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from lib import core, probes
ROOT = REPO / 'logs' / 'qwen36'
ROOT.mkdir(parents=True, exist_ok=True)
import uuid
SCOPE = 'qwen36-trial-' + uuid.uuid4().hex
os.environ['LLAMA_SCOPE'] = SCOPE
BASE = 'http://127.0.0.1:1235'


def request(route, payload=None, timeout=1800):
    return probes.request(BASE, route, payload, timeout)


gpu = probes.gpu


OWNED = False
def stop():
    if OWNED:
        probes.stop(SCOPE)


def main():
    global OWNED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--ubatch', type=int, default=1024)
    parser.add_argument('--draft-max', type=int, default=15)
    parser.add_argument('--spec', choices=['none', 'draft-dflash'], default='draft-dflash')
    parser.add_argument('--prompt-size', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=2048)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cache-k', default='f16')
    parser.add_argument('--cache-v', default='f16')
    parser.add_argument('--prompt-file', type=Path)
    args = parser.parse_args()
    core.preflight(core.load('qwen36'))
    if args.prompt_size + args.output_tokens + args.draft_max + 64 > args.context:
        parser.error('prompt and output must fit context without shifting')
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(('127.0.0.1', 1235)) == 0:
            raise RuntimeError('Port 1235 occupied; refusing to interrupt another server')
    scope = subprocess.run(['systemctl', '--user', 'is-active', SCOPE + '.scope'], capture_output=True, timeout=10)
    if scope.returncode == 0:
        raise RuntimeError('qwen36.scope is already active')
    out = ROOT / args.label
    out.mkdir(parents=True, exist_ok=False)
    record = vars(args).copy()
    record['prompt_file'] = str(args.prompt_file) if args.prompt_file else None
    record.update(stable=False, batch=4096, checkpoints=2, cache_ram_mib=512)
    (out / 'environment.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    (out / 'launcher.sh').write_text((REPO / 'start_qwen36.sh').read_text())
    env = dict(os.environ, QWEN_CONTEXT=str(args.context), QWEN_UBATCH=str(args.ubatch),
               QWEN_DRAFT_MAX=str(args.draft_max), QWEN_SPEC_TYPE=args.spec,
               QWEN_CACHE_K=args.cache_k, QWEN_CACHE_V=args.cache_v,
               QWEN_BATCH='4096', QWEN_LOAD_MODE='dio')
    record['resolved'] = core.snapshot('qwen36')
    # Resolve per-trial environment overrides for the recorded launch command.
    record['resolved']['profile'] = core.load('qwen36', {'context': args.context, 'ubatch': args.ubatch, 'draft_max': args.draft_max, 'spec': args.spec, 'cache_k': args.cache_k, 'cache_v': args.cache_v, 'batch': 4096, 'load_mode': 'dio'})
    record['resolved']['command'] = core.command(record['resolved']['profile'])
    done = threading.Event()
    samples = []
    process = None
    monitor_thread = None

    def monitor():
        with (out / 'gpu.csv').open('w') as f:
            f.write('monotonic,used_mib,free_mib,temp_c,power_w\n')
            while not done.is_set():
                try:
                    values = gpu()
                    samples.append(values)
                    f.write(','.join(map(str, [time.monotonic()] + values)) + '\n')
                    f.flush()
                except Exception as exc:
                    record['monitor_error'] = repr(exc)
                done.wait(0.2)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        start = time.monotonic()
        with (out / 'server.log').open('w') as f:
            process = subprocess.Popen([str(REPO / 'start_qwen36.sh')], env=env,
                                       stdout=f, stderr=subprocess.STDOUT)
            OWNED = True
        deadline = start + 300
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'server exited during startup: {process.returncode}')
            try:
                if request('/health', timeout=2).get('status') == 'ok':
                    break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('startup exceeded 300 seconds')
            time.sleep(1)
        record['load_seconds'] = time.monotonic() - start
        record['gpu_after_load'] = gpu()
        record['scope_limits'] = subprocess.check_output(['systemctl', '--user', 'show', SCOPE + '.scope',
            '-p', 'MemoryHigh', '-p', 'MemoryMax', '-p', 'MemorySwapMax', '-p', 'ControlGroup'], text=True)
        print('LOADED', json.dumps(record), flush=True)
        if args.smoke:
            payload = {'model': 'qwen3.6-35b-a3b-iq3s-dflash', 'messages': [
                {'role': 'user', 'content': 'What is 2 + 2? Answer with the number only.'}],
                'max_tokens': 32, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False}}
            response = request('/v1/chat/completions', payload)
            (out / 'response.json').write_text(json.dumps(response, indent=2))
            record['smoke_response'] = response
        else:
            frozen = ROOT / 'prompt-8192.json'
            if args.prompt_file:
                tokens = json.loads(args.prompt_file.read_text())[:args.prompt_size]
            elif frozen.exists() and args.prompt_size <= 8192:
                tokens = json.loads(frozen.read_text())[:args.prompt_size]
            else:
                source = 'Review these Python service modules for correctness. Write detailed findings and corrected implementations.\n'
                source += ''.join(
                    f'\n# services/worker_{i}.py\ndef process_{i}(records, limit={i % 97 + 1}):\n'
                    f'    """Validate shard {i} and return accepted records."""\n'
                    f'    accepted = []\n    for index, record in enumerate(records):\n'
                    f'        if index >= limit:\n            break\n'
                    f'        value = record.get("value", {i % 31})\n'
                    f'        if isinstance(value, int) and value % {i % 11 + 2} == 0:\n'
                    f'            accepted.append(("shard_{i}", value + {i}))\n'
                    f'    return accepted\n' for i in range(2200))
                (out / 'prompt-source.txt').write_text(source)
                tokens = request('/tokenize', {'content': source, 'add_special': True})['tokens'][:args.prompt_size]
                if args.prompt_size == 8192:
                    frozen.write_text(json.dumps(tokens))
            if len(tokens) != args.prompt_size or not all(isinstance(t, int) for t in tokens):
                raise ValueError('unexpected prompt token array')
            serialized = json.dumps(tokens)
            record['prompt_sha256'] = hashlib.sha256(serialized.encode()).hexdigest()
            (out / 'prompt-tokens.json').write_text(serialized)
            response = request('/completion', {'prompt': tokens, 'n_predict': args.output_tokens,
                'temperature': 0, 'seed': 42, 'cache_prompt': False, 'ignore_eos': True,
                'stream': False})
            (out / 'response.json').write_text(json.dumps(response, indent=2))
            record['timings'] = response['timings']
            record['replacement_characters'] = response.get('content', '').count('\ufffd')
            record['effective_draft_max'] = min(args.draft_max, 15) if args.spec == 'draft-dflash' else 0
            if response['timings']['prompt_n'] != args.prompt_size:
                raise RuntimeError('prompt was not fully evaluated')
            if response['timings'].get('cache_n', 0) != 0 or response.get('truncated', False):
                raise RuntimeError('cached or truncated prompt')
            if response['timings']['predicted_n'] < args.output_tokens:
                raise RuntimeError('insufficient generated tokens')
        with urlopen(BASE + '/metrics', timeout=10) as r:
            metrics = r.read().decode()
        (out / 'metrics.txt').write_text(metrics)
        for metric, key in [('num_draft_tokens', 'proposed'), ('num_accepted_tokens', 'accepted'),
                            ('num_drafts', 'verification_steps')]:
            match = re.search(r'^llamacpp:spec_decode_' + metric + r'_total\s+([\d.e+\-]+)', metrics, re.M)
            if match:
                record[key] = float(match[1])
        if record.get('proposed', 0):
            record['acceptance_percent'] = 100 * record['accepted'] / record['proposed']
        if request('/health', timeout=5)['status'] != 'ok':
            raise RuntimeError('server unhealthy after generation')
        record['stable'] = True
    except KeyboardInterrupt:
        record['interrupted'] = True
        raise
    except Exception as exc:
        record['error'] = repr(exc)
    finally:
        record['scope_memory'] = subprocess.run(['systemctl', '--user', 'show', SCOPE + '.scope',
            '-p', 'MemoryPeak', '-p', 'MemoryCurrent', '-p', 'Result'], capture_output=True, text=True).stdout
        stop()
        if process:
            try:
                record['launcher_exit'] = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                record['launcher_exit'] = process.wait()
        done.set()
        if monitor_thread:
            monitor_thread.join(timeout=15)
        if samples:
            record['peak_vram_mib'] = max(x[0] for x in samples)
            record['min_free_vram_mib'] = min(x[1] for x in samples)
            record['peak_temperature_c'] = max(x[2] for x in samples)
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print('RESULT', json.dumps(record), flush=True)
    if not record['stable']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
