#!/usr/bin/env python3
"""Exercise PI registration and inference against an existing server; never stop it."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import uuid
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import core, pi, probes


def assistant_message(events):
    messages = [e['message'] for e in events if e.get('type') == 'message_end'
                and e.get('message', {}).get('role') == 'assistant']
    if not messages or messages[-1].get('stopReason') in ('error', 'aborted', 'length'):
        raise ValueError('PI did not finish a successful assistant message')
    return messages[-1]


def text_content(message):
    return ''.join(b.get('text', '') for b in message.get('content', []) if b.get('type') == 'text').strip()


def check_case(events, expected, thinking=False, tool=False):
    message = assistant_message(events)
    if text_content(message).lower() != expected.lower():
        raise ValueError(f'Unexpected PI answer: {text_content(message)!r}; expected {expected!r}')
    has_thinking = any(b.get('type') == 'thinking' and b.get('thinking')
                       for e in events if e.get('type') == 'message_end'
                       and e.get('message', {}).get('role') == 'assistant'
                       for b in e['message'].get('content', []))
    if bool(has_thinking) != thinking:
        raise ValueError('PI thinking content does not match the requested mode')
    if not any(e.get('type') == 'message_update'
               and e.get('assistantMessageEvent', {}).get('type') == 'text_delta' for e in events):
        raise ValueError('PI emitted no streamed text delta')
    starts = [e for e in events if e.get('type') == 'tool_execution_start']
    ends = [e for e in events if e.get('type') == 'tool_execution_end']
    if tool:
        reads = [e for e in starts if e.get('toolName') == 'read'
                 and Path(e.get('args', {}).get('path', '')).name == 'pi-fixture.txt']
        if not reads or any(e.get('toolName') != 'read' for e in starts):
            raise ValueError('PI did not invoke the fixture read tool')
        if any(not any(end.get('toolCallId') == read.get('toolCallId')
                       and not end.get('isError') for end in ends) for read in reads):
            raise ValueError('PI fixture read did not complete successfully')
    elif starts:
        raise ValueError('Unexpected PI tool execution')


def solid_png(rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\x00' + rgb * 128) * 128)) + chunk(b'IEND', b''))


def run(name, port=None, context=None, timeout=180):
    profile = core.load(name, {'port': port, 'context': context})
    base = f'http://127.0.0.1:{profile["port"]}'
    out = core.ROOT / 'logs' / name / datetime.now(timezone.utc).strftime('pi-smoke-%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True, exist_ok=False)
    record = {'profile': name, 'alias': profile['alias'], 'base': base,
              'passed': False, 'checks': {}}
    try:
        record['health_before'] = probes.request(base, '/health', timeout=10)
        if record['health_before'].get('status') != 'ok':
            raise ValueError('Existing server is not healthy')
        record['models'] = probes.request(base, '/v1/models', timeout=10)
        if not any(m['id'] == profile['alias'] for m in record['models']['data']):
            raise ValueError('Existing server does not expose the selected profile alias')
        with tempfile.TemporaryDirectory(prefix='llama-pi-') as temp:
            root = Path(temp)
            agent = root / 'agent'
            work = root / 'work'
            work.mkdir()
            env = dict(os.environ, PI_CODING_AGENT_DIR=str(agent), PI_OFFLINE='1', PI_TELEMETRY='0')

            def execute(label, command):
                result = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                        text=True, timeout=timeout)
                (out / f'{label}.stdout').write_text(result.stdout)
                (out / f'{label}.stderr').write_text(result.stderr)
                if result.returncode:
                    raise ValueError(f'{label} failed (exit {result.returncode}); inspect {out}')
                return result.stdout

            record['pi_version'] = execute('version', ['pi', '--version']).strip()
            execute('register', [sys.executable, str(core.ROOT / 'llama.py'), 'pi', 'add', name,
                                '--config', str(agent / 'models.json'), '--port', str(profile['port']),
                                '--context', str(profile['context']), '--output-tokens', str(min(1024, profile['context']))])
            common = ['pi', '--offline', '--no-approve', '--no-session', '--no-mcp', '--no-extensions',
                      '--no-skills', '--no-prompt-templates', '--no-context-files', '--no-themes']
            listing = execute('discovery', common + ['--list-models', pi.PROVIDER])
            if profile['alias'] not in listing or pi.PROVIDER not in listing:
                raise ValueError('PI did not discover the registered alias')
            record['checks']['discovery'] = True

            def case(label, prompt, expected, thinking=False, tool=False, attachment=None):
                command = common + ['--provider', pi.PROVIDER, '--model', profile['alias'],
                                    '--thinking', 'high' if thinking else 'off', '--mode', 'json', '--print']
                command += ['--tools', 'read'] if tool else ['--no-tools']
                if attachment:
                    command.append('@' + str(attachment))
                output = execute(label, command + [prompt])
                events = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
                message = assistant_message(events)
                if message.get('provider') != pi.PROVIDER or message.get('model') != profile['alias']:
                    raise ValueError('PI used a different provider or model')
                check_case(events, expected, thinking, tool)
                record['checks'][label] = True

            case('text', 'Reply with exactly these two words and nothing else: hello world', 'hello world')
            if pi.model_config(profile)['reasoning']:
                case('thinking', 'What is 17 multiplied by 23? Reply with only the integer.', '391', thinking=True)
            token = 'PI_' + uuid.uuid4().hex
            (work / 'pi-fixture.txt').write_text(token + '\n')
            case('tool', 'Use the read tool to read pi-fixture.txt. Reply with exactly its contents '
                 'and nothing else. Do not guess the contents.', token, tool=True)
            if profile.get('mmproj'):
                for color, rgb in [('red', b'\xff\x00\x00'), ('blue', b'\x00\x00\xff')]:
                    image = work / f'{color}.png'
                    image.write_bytes(solid_png(rgb))
                    case('vision-' + color, 'What color is this image? Answer with one color word only.',
                         color, attachment=image)
            record['registered_model'] = json.loads((agent / 'models.json').read_text())
        record['health_after'] = probes.request(base, '/health', timeout=10)
        if record['health_after'].get('status') != 'ok':
            raise ValueError('Server is not healthy after PI checks')
        record['passed'] = True
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(out, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=core.PROFILES)
    parser.add_argument('--port', type=int)
    parser.add_argument('--context', type=int)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    run(args.profile, args.port, args.context, args.timeout)


if __name__ == '__main__':
    main()
