"""Every protocol observes the HTTP-body environment setting in a fresh server process."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


_PROBE = r'''
import http.client
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import app_for
from tests.test_server_openai_compat import FakeApp, serve_fake


@contextmanager
def serving(backend, directory):
    if backend == "cuda":
        with http_server(app_for(directory)) as port:
            yield port
    else:
        app = FakeApp()
        app.prepare = lambda chat, is_chat: SimpleNamespace(prompt=[1, 2, 3])
        server = serve_fake(app)
        try:
            yield server.server_port
        finally:
            server.shutdown()
            server.server_close()


messages = [{"role": "user", "content": "Hi"}]
routes = {
    "/v1/chat/completions": {"messages": messages, "max_tokens": 8},
    "/v1/responses": {"input": "Hi", "max_output_tokens": 8, "store": False},
    "/v1/messages": {"model": "test", "messages": messages, "max_tokens": 8},
    "/v1/messages/count_tokens": {"model": "test", "messages": messages},
}
with tempfile.TemporaryDirectory() as directory, serving(sys.argv[1], Path(directory)) as port:
    for route, body in routes.items():
        for size in (4096, 4097):
            # JSON whitespace grows wire bytes without changing the model's prompt or translated inner body.
            raw = json.dumps(body).encode()
            raw += b" " * (size - len(raw))
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.request("POST", route, raw, {"Content-Type": "application/json"})
                response = connection.getresponse()
                payload = json.loads(response.read())
                expected = 200 if size == 4096 else 400
                assert response.status == expected, (route, size, response.status, payload)
                if size == 4097:
                    assert "request body exceeds" in payload["error"]["message"], (route, payload)
                    assert response.getheader("Connection") == "close", route
            finally:
                connection.close()
print("all four protocol routes enforce the configured body boundary")
'''


@pytest.mark.parametrize("backend", ["cuda", "mac"])
def test_configured_body_limit_reaches_all_protocol_routes(backend):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "TF_REQUEST_BODY_LIMIT_BYTES": "4096", "CUDA_VISIBLE_DEVICES": ""}
    environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root),
                                               environment.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", _PROBE, backend], cwd=root, env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all four protocol routes enforce" in result.stdout
