"""Optional NVIDIA and torch allocator collectors."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, TypeVar, cast

import logfire
from opentelemetry.metrics import CallbackOptions, Observation

from ._cache import SnapshotCache
from ._names import GPU_METRICS
from ._system_metrics import ProcessTree
from ._torch import get_torch


class Memory(Protocol):
    used: int
    total: int


class Utilization(Protocol):
    gpu: int
    memory: int


class GpuProcess(Protocol):
    pid: int
    usedGpuMemory: int | None


class Nvml(Protocol):
    NVMLError: type[Exception]
    NVMLError_NotSupported: type[Exception]
    NVML_TEMPERATURE_GPU: int
    NVML_CLOCK_SM: int
    NVML_CLOCK_MEM: int
    NVML_CLOCK_GRAPHICS: int
    NVML_MEMORY_ERROR_TYPE_CORRECTED: int
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED: int
    NVML_AGGREGATE_ECC: int

    def nvmlInit(self) -> None: ...
    def nvmlShutdown(self) -> None: ...
    def nvmlDeviceGetCount(self) -> int: ...
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...
    def nvmlDeviceGetUUID(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetName(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> Memory: ...
    def nvmlDeviceGetUtilizationRates(self, handle: object) -> Utilization: ...
    def nvmlDeviceGetEncoderUtilization(self, handle: object) -> tuple[int, int]: ...
    def nvmlDeviceGetDecoderUtilization(self, handle: object) -> tuple[int, int]: ...
    def nvmlDeviceGetTotalEccErrors(self, handle: object, error_type: int, counter_type: int) -> int: ...
    def nvmlDeviceGetTemperature(self, handle: object, sensor: int) -> int: ...
    def nvmlDeviceGetPowerUsage(self, handle: object) -> int: ...
    def nvmlDeviceGetEnforcedPowerLimit(self, handle: object) -> int: ...
    def nvmlDeviceGetClockInfo(self, handle: object, clock: int) -> int: ...
    def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> Sequence[GpuProcess]: ...
    def nvmlSystemGetDriverVersion(self) -> str | bytes: ...
    def nvmlSystemGetCudaDriverVersion(self) -> int: ...


T = TypeVar('T')
_Add = Callable[[str, int | float | None, Mapping[str, str | int]], None]


def _text(value: str | bytes) -> str:
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value


class GpuCollector:
    """Own the NVML session, with one cached snapshot for all GPU callbacks."""

    def __init__(self, tree: ProcessTree) -> None:
        self._tree = tree
        self._nvml: Nvml | None = None
        self._devices: list[tuple[int, object, dict[str, str | int]]] = []
        self._disabled: set[tuple[int, str]] = set()
        self.inventory: list[dict[str, object]] = []
        self.status = 'disabled'
        self._cache = SnapshotCache(self._collect, dict[str, list[Observation]]())

    def initialize(self) -> None:
        try:
            # nvidia-ml-py has no complete type stubs; this protocol describes
            # the vendor API used by the collector and the test backend.
            nvml = cast(Nvml, importlib.import_module('pynvml'))
        except ImportError:
            self.status = 'nvidia-ml-py not installed'
            return
        try:
            nvml.nvmlInit()
        except nvml.NVMLError as exc:
            self.status = f'NVML unavailable: {exc}'
            return
        self._nvml = nvml
        try:
            count = nvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                attrs: dict[str, str | int] = {
                    'hw.id': _text(nvml.nvmlDeviceGetUUID(handle)),
                    'hw.name': _text(nvml.nvmlDeviceGetName(handle)),
                    'hw.vendor': 'nvidia',
                    'gpu.index': index,
                }
                self._devices.append((index, handle, attrs))
                memory = self._read(index, 'memory', lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
                self.inventory.append(
                    {
                        'index': index,
                        'uuid': attrs['hw.id'],
                        'name': attrs['hw.name'],
                        'memory_total': memory.total if memory is not None else None,
                        'driver_version': self._read(index, 'driver', lambda: _text(nvml.nvmlSystemGetDriverVersion())),
                        'cuda_driver_version': self._read(index, 'cuda_driver', nvml.nvmlSystemGetCudaDriverVersion),
                    }
                )
            self.status = 'ok' if count else 'no NVIDIA devices'
        except nvml.NVMLError as exc:
            self.status = f'NVML discovery failed: {exc}'

    def _read(self, index: int, key: str, call: Callable[[], T]) -> T | None:
        nvml = self._nvml
        if nvml is None or (index, key) in self._disabled:
            return None
        try:
            return call()
        except nvml.NVMLError_NotSupported:
            self._disabled.add((index, key))
        except (nvml.NVMLError, OSError):
            pass
        return None

    def _collect(self) -> dict[str, list[Observation]]:
        result: dict[str, list[Observation]] = {}

        def add(name: str, value: int | float | None, attrs: Mapping[str, str | int]) -> None:
            if value is not None:
                result.setdefault(name, []).append(Observation(value, attrs))

        self._collect_nvml(add)
        self._collect_torch(add)
        return result

    def _collect_nvml(self, add: _Add) -> None:
        nvml = self._nvml
        if nvml is not None:
            pids = self._tree.pids()
            for index, handle, attrs in self._devices:
                rates = self._read(index, 'utilization', lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
                if rates is not None:
                    add('hw.gpu.utilization', rates.gpu / 100, {**attrs, 'hw.gpu.task': 'general'})
                    add('train.gpu.memory_controller.utilization', rates.memory / 100, attrs)
                for task, utilization_call in (
                    ('encoder', nvml.nvmlDeviceGetEncoderUtilization),
                    ('decoder', nvml.nvmlDeviceGetDecoderUtilization),
                ):
                    utilization = self._read(index, task, lambda: utilization_call(handle))
                    if utilization is not None:
                        add('hw.gpu.utilization', utilization[0] / 100, {**attrs, 'hw.gpu.task': task})
                memory = self._read(index, 'memory', lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
                if memory is not None:
                    add('hw.gpu.memory.usage', memory.used, attrs)
                    add('hw.gpu.memory.limit', memory.total, attrs)
                    if memory.total:
                        add('hw.gpu.memory.utilization', memory.used / memory.total, attrs)
                for error, kind in (
                    ('corrected', nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED),
                    ('uncorrected', nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED),
                ):
                    value = self._read(
                        index,
                        f'ecc_{error}',
                        lambda: nvml.nvmlDeviceGetTotalEccErrors(handle, kind, nvml.NVML_AGGREGATE_ECC),
                    )
                    add('hw.errors', value, {**attrs, 'hw.type': 'gpu', 'error.type': error})
                add(
                    'train.gpu.temperature',
                    self._read(
                        index, 'temperature', lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    ),
                    attrs,
                )
                for suffix, power_call in (
                    ('usage', nvml.nvmlDeviceGetPowerUsage),
                    ('limit', nvml.nvmlDeviceGetEnforcedPowerLimit),
                ):
                    power = self._read(index, f'power_{suffix}', lambda: power_call(handle))
                    add(f'train.gpu.power.{suffix}', power / 1000 if power is not None else None, attrs)
                for clock, kind in (
                    ('sm', nvml.NVML_CLOCK_SM),
                    ('memory', nvml.NVML_CLOCK_MEM),
                    ('graphics', nvml.NVML_CLOCK_GRAPHICS),
                ):
                    value = self._read(index, f'clock_{clock}', lambda: nvml.nvmlDeviceGetClockInfo(handle, kind))
                    add('train.gpu.clock.frequency', value, {**attrs, 'clock': clock})
                processes = self._read(index, 'processes', lambda: nvml.nvmlDeviceGetComputeRunningProcesses(handle))
                if processes is not None:
                    add(
                        'train.gpu.process.memory.usage',
                        sum(
                            process.usedGpuMemory
                            for process in processes
                            if process.pid in pids and process.usedGpuMemory is not None
                        ),
                        attrs,
                    )

    def _collect_torch(self, add: _Add) -> None:
        torch = get_torch()
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    for index in range(torch.cuda.device_count()):
                        torch_attrs: dict[str, str | int] = {'gpu.index': index, 'train.torch.backend': 'cuda'}
                        add('train.torch.memory.allocated', torch.cuda.memory_allocated(index), torch_attrs)
                        add('train.torch.memory.reserved', torch.cuda.memory_reserved(index), torch_attrs)
                        add('train.torch.memory.peak_allocated', torch.cuda.max_memory_allocated(index), torch_attrs)
                if torch.backends.mps.is_available():
                    mps_attrs = {'train.torch.backend': 'mps'}
                    add('train.torch.memory.allocated', torch.mps.current_allocated_memory(), mps_attrs)
                    add('train.torch.memory.reserved', torch.mps.driver_allocated_memory(), mps_attrs)
            except (RuntimeError, OSError):
                # Backend shutdown or a disappearing device must not fail training.
                pass

    def snapshot(self) -> dict[str, list[Observation]]:
        return self._cache.get()

    def install(self, instance: logfire.Logfire) -> None:
        for name, unit, kind in GPU_METRICS:

            def callback(options: CallbackOptions, name: str = name) -> Iterable[Observation]:
                return self.snapshot().get(name, [])

            if kind == 'updown':
                instance.metric_up_down_counter_callback(name, [callback], unit=unit)
            elif kind == 'counter':
                instance.metric_counter_callback(name, callbacks=[callback], unit=unit)
            else:
                instance.metric_gauge_callback(name, [callback], unit=unit)

    def close(self) -> None:
        if self._nvml is not None:
            nvml, self._nvml = self._nvml, None
            try:
                nvml.nvmlShutdown()
            except nvml.NVMLError:
                pass
