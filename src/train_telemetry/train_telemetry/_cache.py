"""Bounded reuse of snapshots across metric readers."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar('T')


class SnapshotCache(Generic[T]):
    """Share a one second snapshot; concurrent readers never wait on a collector."""

    def __init__(self, collect: Callable[[], T], empty: T) -> None:
        self._collect = collect
        self._value = empty
        self._at = float('-inf')
        self._lock = Lock()

    def get(self) -> T:
        if time.monotonic() - self._at >= 1 and self._lock.acquire(blocking=False):
            try:
                if time.monotonic() - self._at >= 1:
                    self._value = self._collect()
                    self._at = time.monotonic()
            finally:
                self._lock.release()
        return self._value
