"""Small HTTP process used to exercise real llama-swap without a GPU."""
import json
import os
from pathlib import Path
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

action, name, port, root = sys.argv[1:]
root = Path(root)
pidfile = root / (name + '.pid')


def event(kind):
    with (root / 'events.jsonl').open('a') as out:
        out.write(json.dumps({'model': name, 'event': kind}) + '\n')


if action == 'stop':
    if pidfile.exists():
        os.kill(int(pidfile.read_text()), signal.SIGTERM)
        deadline = time.monotonic() + 10
        while pidfile.exists():
            if time.monotonic() > deadline:
                sys.exit('Backend did not stop')
            time.sleep(0.02)
    sys.exit(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            self.reply({'status': 'ok'})
        else:
            self.reply({'data': [{'id': name}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        event('request')
        if payload.get('messages', [{}])[0].get('content') == 'slow':
            event('slow-start')
            time.sleep(1)
            event('slow-end')
        self.reply({'model': name, 'choices': [{'message': {'role': 'assistant', 'content': name}}],
                    'received': payload})


def terminated(signum, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, terminated)
server = ThreadingHTTPServer(('127.0.0.1', int(port)), Handler)
pidfile.write_text(str(os.getpid()))
event('start')
try:
    server.serve_forever(poll_interval=0.05)
finally:
    server.server_close()
    event('stop')
    pidfile.unlink(missing_ok=True)
