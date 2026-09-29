from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import train_telemetry._cache as cache_module
from train_telemetry._cache import SnapshotCache


def test_readers_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader refreshed during another reader's preemption is not sampled twice."""
    paused, resume = threading.Event(), threading.Event()
    base_lock = threading.Lock()
    coordinate = False
    first: threading.Thread | None = None
    now = 0.0
    calls = 0

    class CoordinatedLock:
        def acquire(self, *, blocking: bool) -> bool:
            if coordinate and threading.current_thread() is first:
                paused.set()
                assert resume.wait(timeout=5)
            return base_lock.acquire(blocking=blocking)

        def release(self) -> None:
            base_lock.release()

    monkeypatch.setattr(cache_module, 'Lock', CoordinatedLock)
    monkeypatch.setattr(cache_module, 'time', SimpleNamespace(monotonic=lambda: now))

    def collect() -> int:
        nonlocal calls
        calls += 1
        return calls

    cache = SnapshotCache(collect, 0)
    assert cache.get() == 1
    coordinate = True
    now = 2.0
    results: list[int] = []
    first = threading.Thread(target=lambda: results.append(cache.get()))
    first.start()
    try:
        assert paused.wait(timeout=5)
        assert cache.get() == 2
    finally:
        resume.set()
        first.join(timeout=5)
    assert not first.is_alive()
    assert results == [2]
    assert calls == 2
