#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAVE-SIM inline-plot HTTP server manager.

Serves output/_agent_runs/plots over http://127.0.0.1:8811 so the agent can
embed result / grid images directly in the conversation via markdown:

    ![title](http://127.0.0.1:8811/<file>.png)

Modes:
  python plot_server.py serve            # foreground server (use for tests / background jobs)
  python plot_server.py start            # detached server (self-healing; reused if port busy)
  python plot_server.py status           # JSON status
  python plot_server.py stop             # stop the detached server

Every mode except `serve` prints a single JSON object on stdout.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PLOTS_DIR = Path('/mnt/d/rave-sim-main/rave-sim-main/output/_agent_runs/plots')
PID_FILE = PLOTS_DIR / '.plot_server.pid'
LOG_FILE = PLOTS_DIR / '.plot_server.log'
HOST = '127.0.0.1'
PORT = 8811
BASE_URL = 'http://{0}:{1}'.format(HOST, PORT)


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # keep the log file clean

    def do_GET(self):
        if self.path.split('?')[0] == '/ping':
            body = json.dumps({'ok': True, 'url': BASE_URL}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def port_up(host=HOST, port=PORT, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def list_pngs():
    if not PLOTS_DIR.is_dir():
        return []
    pngs = sorted(PLOTS_DIR.glob('*.png'), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in pngs]


def status():
    pid = None
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
        except ValueError:
            pid = None
    up = port_up()
    if up:
        return {'ok': True, 'up': True, 'port': PORT, 'url': BASE_URL, 'pid': pid, 'pngs': list_pngs()}
    return {'ok': False, 'up': False, 'port': PORT, 'url': BASE_URL, 'pid': pid, 'pngs': []}


def start():
    if port_up():
        s = status()
        s['note'] = 'server already running; reused'
        return s
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, 'a')
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), 'serve'],
        cwd=str(PLOTS_DIR),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        PID_FILE.write_text(str(proc.pid))
    except OSError:
        pass
    deadline = time.time() + 5
    while time.time() < deadline:
        if port_up():
            s = status()
            s['note'] = 'started'
            return s
        time.sleep(0.1)
    return {'ok': False, 'up': False, 'error': 'server failed to come up', 'log': str(LOG_FILE)}


def stop():
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            try:
                os.kill(pid, 15)  # SIGTERM
            except ProcessLookupError:
                pass
            time.sleep(0.3)
        except ValueError:
            pass
        try:
            PID_FILE.unlink()
        except OSError:
            pass
    return {'ok': not port_up(), 'up': port_up(), 'port': PORT}


def serve():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(PLOTS_DIR)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description='RAVE-SIM inline-plot HTTP server manager')
    ap.add_argument('mode', choices=['serve', 'start', 'status', 'stop'])
    args = ap.parse_args()
    if args.mode == 'serve':
        serve()
    elif args.mode == 'start':
        print(json.dumps(start()))
    elif args.mode == 'status':
        print(json.dumps(status()))
    elif args.mode == 'stop':
        print(json.dumps(stop()))
    sys.exit(0)


if __name__ == '__main__':
    main()
