"""Python stream tee with progress-line normalization and recursion guards."""

from __future__ import annotations

import io
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import IO, BinaryIO, Literal, TextIO

import logfire


class CaptureGuard(threading.local):
    """Per-thread suppression shared by logging, warnings and console capture."""

    depth: int = 0

    @contextmanager
    def suppress(self) -> Generator[None]:
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1


class ConsoleTee(io.TextIOBase):
    """Mirror text unchanged while emitting completed lines to Logfire.

    `buffer`, `fileno` and `isatty` delegate to the wrapped stream. Binary and
    native file-descriptor writes bypass capture. Closing the tee never closes
    the application's stream; the run owns and closes the raw log file.
    """

    def __init__(
        self,
        original: TextIO,
        raw: IO[str] | None,
        raw_lock: threading.RLock,
        instance: logfire.Logfire,
        guard: CaptureGuard,
        *,
        stream: Literal['stdout', 'stderr'],
        to_logfire: bool = True,
    ) -> None:
        self._original = original
        self._raw = raw
        self._lock = raw_lock
        self._logfire = instance
        self._guard = guard
        self._stream = stream
        self._to_logfire = to_logfire
        self._pending = ''

    @property
    # Typeshed models TextIOBase readonly properties as instance attributes.
    def encoding(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self._original.encoding

    @property
    def errors(self) -> str | None:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self._original.errors

    @property
    def buffer(self) -> BinaryIO:
        return self._original.buffer

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        lines: list[str] = []
        with self._lock:
            self._original.write(text)
            if self._raw is not None:
                self._raw.write(text)
            if self._to_logfire and not self._guard.depth:
                pieces = (self._pending + text).split('\n')
                self._pending = pieces.pop()
                # Retain a trailing CR to distinguish CRLF from a new progress state.
                if '\r' in self._pending.rstrip('\r'):
                    self._pending = self._pending.rstrip('\r').rsplit('\r', 1)[-1] + (
                        '\r' if self._pending.endswith('\r') else ''
                    )
                lines = [line.rstrip('\r').rsplit('\r', 1)[-1] for line in pieces]
        if lines:
            with self._guard.suppress():
                for line in lines:
                    self._logfire.log(
                        'info',
                        '{line}',
                        {
                            'line': line,
                            'train.log.source': 'console',
                            'train.console.stream': self._stream,
                        },
                        tags=['console'],
                    )
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._original.flush()
            if self._raw is not None and not self._raw.closed:
                self._raw.flush()

    def close(self) -> None:
        self.flush()
