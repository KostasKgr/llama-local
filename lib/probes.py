"""Shared benchmark HTTP, telemetry and owned-scope cleanup."""
import json
import subprocess
from urllib.request import Request, urlopen


def request(base, route, payload=None, timeout=1800):
    req = Request(base + route, None if payload is None else json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,temperature.gpu,power.draw',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [float(x.strip()) for x in raw.splitlines()[0].split(',')]


def stop(scope):
    subprocess.run(['systemctl', '--user', 'stop', scope + '.scope'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
