#!/usr/bin/env python3
"""Sequential fresh-process LFM context benchmarks; synthetic speed, not quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, probes


def source_text():
    return 'Review these service modules for correctness and propose fixes.\n' + ''.join(
        f'\n# services/worker_{i}.py\ndef process_{i}(records, limit={i % 97 + 1}):\n'
        f'    """Validate shard {i} and return accepted records."""\n'
        '    accepted = []\n    for index, record in enumerate(records):\n'
        '        if index >= limit:\n            break\n'
        f'        value = record.get("value", {i % 31})\n'
        f'        if isinstance(value, int) and value % {i % 11 + 2} == 0:\n'
        f'            accepted.append(("shard_{i}", value + {i}))\n'
        '    return accepted\n' for i in range(2500))


def trial(prompt_tokens, repeat, args, out):
    context = 131072
    scope = 'lfm25-trial-' + uuid.uuid4().hex
    overrides = {'context': context}
    p = core.load('lfm25', overrides)
    p['scope'] = scope
    core.preflight(p)
    # Do not contend with a server using a different port or an unrelated GPU job.
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if processes.strip():
        raise RuntimeError('GPU compute process present; refusing to contend with another workload')
    directory = out / f'prompt{prompt_tokens}-r{repeat}'
    directory.mkdir()
    record = {'context': context, 'repeat': repeat, 'stable': False,
              'prompt_tokens': prompt_tokens, 'output_tokens': args.output_tokens,
              'scope': scope, 'profile': p, 'command': core.command(p),
              'artifacts': core.manifest('models')['lfm25']}
    base = f"http://127.0.0.1:{p['port']}"
    process = None
    done = threading.Event()
    samples = []

    def monitor():
        with (directory / 'gpu.csv').open('w') as f:
            f.write('monotonic,used_mib,free_mib,temp_c,power_w\n')
            while not done.is_set():
                try:
                    values = probes.gpu()
                    samples.append(values)
                    f.write(','.join(map(str, [time.monotonic()] + values)) + '\n')
                    f.flush()
                except Exception as exc:
                    record['monitor_error'] = repr(exc)
                done.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    try:
        thread.start()
        start = time.monotonic()
        env = dict(os.environ, LLAMA_SCOPE=scope)
        with (directory / 'launcher.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(core.ROOT / 'llama.py'),
                'start', 'lfm25', '--context', str(context)], env=env,
                stdout=log, stderr=subprocess.STDOUT)
        while True:
            if process.poll() is not None:
                raise RuntimeError('Server exited during startup; inspect launcher.log')
            try:
                if probes.request(base, '/health', timeout=2).get('status') == 'ok':
                    break
            except Exception:
                pass
            if time.monotonic() - start > 300:
                raise TimeoutError('Startup exceeded 300 seconds')
            time.sleep(1)
        record['load_seconds'] = time.monotonic() - start
        models = probes.request(base, '/v1/models')
        if not any(m['id'] == p['alias'] for m in models['data']):
            raise RuntimeError('Unexpected model alias')
        record['gpu_after_load'] = probes.gpu()
        source = source_text()
        (directory / 'prompt-source.txt').write_text(source)
        # Always tokenize through this model, never reuse another model's IDs.
        tokens = probes.request(base, '/tokenize', {'content': source, 'add_special': True})['tokens']
        tokens = tokens[:record['prompt_tokens']]
        if len(tokens) != record['prompt_tokens']:
            raise ValueError('Source text does not contain enough tokens')
        serialized = json.dumps(tokens)
        (directory / 'prompt-tokens.json').write_text(serialized)
        record['prompt_sha256'] = hashlib.sha256(serialized.encode()).hexdigest()
        print(f"RUN context={context} repeat={repeat} prompt={len(tokens)}", flush=True)
        start = time.monotonic()
        response = probes.request(base, '/completion', {
            'prompt': tokens, 'n_predict': args.output_tokens, **p['sampling'],
            'seed': 42, 'cache_prompt': False, 'ignore_eos': True, 'stream': False})
        record['wall_seconds'] = time.monotonic() - start
        (directory / 'response.json').write_text(json.dumps(response, indent=2) + '\n')
        timing = record['timings'] = response['timings']
        if timing['prompt_n'] != len(tokens) or timing.get('cache_n', 0) != 0:
            raise RuntimeError('Prompt was not fully evaluated without cache reuse')
        if response.get('truncated') or timing['predicted_n'] != args.output_tokens:
            raise RuntimeError('Truncated prompt or incomplete generation')
        if probes.request(base, '/health', timeout=5).get('status') != 'ok':
            raise RuntimeError('Server unhealthy after generation')
        record['stable'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['scope_memory'] = subprocess.run(['systemctl', '--user', 'show', scope + '.scope',
            '-p', 'MemoryPeak', '-p', 'MemoryHigh', '-p', 'MemoryMax', '-p', 'MemorySwapMax'],
            capture_output=True, text=True).stdout
        if process is not None:
            # This randomly named scope belongs exclusively to this trial.
            probes.stop(scope)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        done.set()
        thread.join(timeout=15)
        if samples:
            record['peak_vram_mib'] = max(v[0] for v in samples)
            record['min_free_vram_mib'] = min(v[1] for v in samples)
            record['max_temperature_c'] = max(v[2] for v in samples)
        (directory / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print('RESULT', json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-sizes', nargs='+', type=int, default=[36864, 98304])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.repeats < 1 or args.output_tokens < 1:
        parser.error('Positive repeats/output required')
    if any(n < 1 or n + args.output_tokens >= 131072 for n in args.prompt_sizes):
        parser.error('Prompt plus output must fit the fixed 131072 context')
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    out = core.ROOT / 'logs' / 'lfm25' / time.strftime('benchmark-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'workload': 'synthetic varied Python service modules; not a quality evaluation',
                'arguments': vars(args), 'environment': subprocess.check_output(['nvidia-smi'], text=True),
                'runtime': core.snapshot('lfm25'),
                'activation_precision_env': os.environ.get('GGML_CUDA_MMQ_PREC', 'auto')}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Artifacts:', out, flush=True)
    records = []
    for prompt_tokens in args.prompt_sizes:
        for repeat in range(1, args.repeats + 1):
            records.append(trial(prompt_tokens, repeat, args, out))
    summary = []
    for prompt_tokens in dict.fromkeys(args.prompt_sizes):
        group = [r for r in records if r['prompt_tokens'] == prompt_tokens]
        summary.append({'context': 131072, 'prompt_tokens': prompt_tokens,
            'output_tokens': args.output_tokens, 'runs': len(group),
            'median_prefill_tps': statistics.median(r['timings']['prompt_per_second'] for r in group),
            'median_decode_tps': statistics.median(r['timings']['predicted_per_second'] for r in group),
            'median_prefill_seconds': statistics.median(r['timings']['prompt_ms'] / 1000 for r in group),
            'peak_vram_mib': max(r['peak_vram_mib'] for r in group),
            'min_free_vram_mib': min(r['min_free_vram_mib'] for r in group)})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
