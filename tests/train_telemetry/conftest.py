from __future__ import annotations

import gzip
import logging
import sys
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TextIO

import pytest
from logfire.testing import capfire as capfire


@dataclass
class Receiver:
    url: str = ''
    requests: list[tuple[str, bytes]] = field(default_factory=lambda: list[tuple[str, bytes]]())
    received: threading.Event = field(default_factory=threading.Event)


@pytest.fixture
def otlp_receiver() -> Iterator[Receiver]:
    receiver = Receiver()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self) -> None:
            self._respond()

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers['Content-Length']))
            if self.headers.get('Content-Encoding') == 'gzip':
                body = gzip.decompress(body)
            receiver.requests.append((self.path, body))
            receiver.received.set()
            self._respond()

        def _respond(self) -> None:
            body = (
                b'{"project_name":"test","project_url":"http://localhost/test"}' if self.path == '/v1/info' else b'{}'
            )
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    receiver.url = f'http://127.0.0.1:{server.server_port}'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield receiver
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture(autouse=True)
def restored_globals(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)
    monkeypatch.delenv('LOGFIRE_BASE_URL', raising=False)
    before = (
        sys.stdout,
        sys.stderr,
        logging.Logger.callHandlers,
        logging.StreamHandler[TextIO].emit,
        warnings.showwarning,
        threading.excepthook,
    )
    yield
    assert before == (
        sys.stdout,
        sys.stderr,
        logging.Logger.callHandlers,
        logging.StreamHandler[TextIO].emit,
        warnings.showwarning,
        threading.excepthook,
    )
