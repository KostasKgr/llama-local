"""CPU-only Windows API fixture with a nested-job grandchild."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.getcwd())
from serve import winjob

run = Path(sys.argv[1])
instance = run.name
child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(120)'])
if not winjob.contain(child):
    raise RuntimeError('Nested job containment failed')
(run / 'grandchild.json').write_text(json.dumps({'pid': child.pid}))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = {'/health': {'service': 'strata', 'loaded': True, 'max_context': 131072},
                '/v1/models': {'data': [{'id': 'fixture'}]},
                '/v1/telemetry': {'instance': {'id': instance}}}[self.path]
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(('127.0.0.1', int(sys.argv[2])), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
(run / 'ready.json').write_text(json.dumps({'pid': os.getpid(), 'bound_port': int(sys.argv[2]),
     'model_id': 'fixture', 'model_state': 'ready', 'instance_id': instance}))
sys.stdin.buffer.read()
server.shutdown()
