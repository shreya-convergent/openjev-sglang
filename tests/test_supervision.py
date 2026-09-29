"""Exercise fatal watchdog behavior in subprocesses, never inside pytest itself."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("child_code", [0, 7])
def test_child_exit_terminates_supervisor(child_code):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import subprocess, sys, time
from openjev.launch import watch_process
child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.exit({child_code})'])
watch_process(child, 'test child', poll_interval=0.01)
time.sleep(10)
""",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 1
    assert f"status {child_code}" in result.stderr


def test_normal_shutdown_disarms_watchdog():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import subprocess, sys, time
from openjev.launch import watch_process
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
stopping = watch_process(child, 'test child', poll_interval=0.01)
stopping.set()
child.terminate()
child.wait(timeout=3)
time.sleep(0.1)
""",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_snapshot_sleep_and_wake_call_order():
    import http.server
    import threading

    from openjev.launch import sleep_backend, wake_backend

    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        sleep_backend(url, url, b'{"warm": 1}')
        wake_backend(url, url, b'{"warm": 1}')
    finally:
        server.shutdown()
    assert [path for path, _ in calls] == [
        "/v1/systemone",
        "/release_memory_occupation",
        "/resume_memory_occupation",
        "/flush_cache",
        "/v1/systemone",
    ]
    assert calls[0][1] == b'{"warm": 1}'
