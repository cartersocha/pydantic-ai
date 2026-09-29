"""Host, process tree and filesystem metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import logfire
import psutil
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation

from ._cache import SnapshotCache


class ProcessTree:
    """Retain psutil process instances so CPU deltas survive between samples."""

    def __init__(self) -> None:
        self._parent = psutil.Process()
        self._processes = {self._parent.pid: self._parent}
        self._parent.cpu_percent()
        self._cache = SnapshotCache(self._collect, dict[str, list[Observation]]())

    def pids(self) -> set[int]:
        try:
            return {self._parent.pid, *(child.pid for child in self._parent.children(recursive=True))}
        except psutil.Error:
            return {self._parent.pid}

    def _collect(self) -> dict[str, list[Observation]]:
        live = self.pids()
        rss, cpu = 0, 0.0
        for pid in live:
            try:
                process = self._processes.get(pid)
                if process is None:
                    process = psutil.Process(pid)
                    process.cpu_percent()
                    self._processes[pid] = process
                with process.oneshot():
                    rss += process.memory_info().rss
                    cpu += process.cpu_percent()
            except (psutil.Error, OSError):
                continue
        self._processes = {pid: process for pid, process in self._processes.items() if pid in live}
        return {
            'train.process.tree.memory.rss': [Observation(rss)],
            'train.process.tree.cpu.utilization': [Observation(cpu / 100 / (psutil.cpu_count() or 1))],
            'train.process.tree.children': [Observation(len(live) - 1)],
        }

    def callback(self, name: str) -> Callable[[CallbackOptions], Iterable[Observation]]:
        return lambda _: self._cache.get().get(name, [])


def install(instance: logfire.Logfire, directory: Path, tree: ProcessTree) -> Callable[[], None]:
    """Install standard metrics and custom cached process tree/filesystem gauges."""
    instrumentor = SystemMetricsInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        raise RuntimeError('System metrics are already instrumented; start `TrainingRun` before instrumenting Logfire')
    instance.instrument_system_metrics(
        {
            'process.cpu.utilization': None,
            'process.memory.usage': None,
            'process.thread.count': None,
            'system.cpu.simple_utilization': None,
            'system.memory.usage': ['used', 'available', 'total'],
            'system.memory.utilization': ['available'],
            'system.disk.io': ['read', 'write'],
            'system.network.io': ['transmit', 'receive'],
        },
        base=None,
    )
    for name, unit in (
        ('train.process.tree.memory.rss', 'By'),
        ('train.process.tree.cpu.utilization', '1'),
        ('train.process.tree.children', '{process}'),
    ):
        instance.metric_gauge_callback(name, [tree.callback(name)], unit=unit)
    try:
        mounts = [Path(part.mountpoint) for part in psutil.disk_partitions(all=True)]
        mount = max(
            (path for path in mounts if path == directory or path in directory.parents),
            key=lambda path: len(path.parts),
            default=Path(directory.anchor),
        )
    except (psutil.Error, OSError):
        mount = Path(directory.anchor)

    def filesystem() -> dict[str, list[Observation]]:
        try:
            disk = psutil.disk_usage(str(directory))
        except (psutil.Error, OSError):
            return {}
        attributes = {'system.filesystem.mountpoint': str(mount)}
        return {
            'system.filesystem.utilization': [Observation(disk.percent / 100, attributes)],
            'system.filesystem.usage': [
                Observation(disk.used, {**attributes, 'system.filesystem.state': 'used'}),
                Observation(disk.free, {**attributes, 'system.filesystem.state': 'free'}),
            ],
        }

    cache = SnapshotCache(filesystem, dict[str, list[Observation]]())
    for name, unit in (('system.filesystem.utilization', '1'), ('system.filesystem.usage', 'By')):
        # Bind the name at registration, not when the callback is invoked later.
        def callback(options: CallbackOptions, name: str = name) -> Iterable[Observation]:
            return cache.get().get(name, [])

        instance.metric_gauge_callback(name, [callback], unit=unit)
    return instrumentor.uninstrument
