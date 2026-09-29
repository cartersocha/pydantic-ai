from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import logfire
import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire
from train_telemetry._gpu import GpuCollector
from train_telemetry._system_metrics import ProcessTree

from train_telemetry import TrainingRun, _cache


@dataclass
class Memory:
    used: int = 512
    total: int = 1024


@dataclass
class Utilization:
    gpu: int = 50
    memory: int = 25


@dataclass
class Process:
    pid: int
    usedGpuMemory: int | None


class NvmlError(Exception):
    pass


class NotSupported(NvmlError):
    pass


class FakeNvml:
    NVMLError = NvmlError
    NVMLError_NotSupported = NotSupported
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_CLOCK_GRAPHICS = 3
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_AGGREGATE_ECC = 0

    def __init__(self) -> None:
        self.closed = False
        self.memory_calls = 0
        self.ecc_calls: list[tuple[object, int]] = []

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        self.closed = True

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        return index

    def nvmlDeviceGetUUID(self, handle: object) -> str:
        return f'gpu-{handle}'

    def nvmlDeviceGetName(self, handle: object) -> bytes:
        return b'Test GPU'

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> Memory:
        self.memory_calls += 1
        return Memory()

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> Utilization:
        return Utilization()

    def nvmlDeviceGetEncoderUtilization(self, handle: object) -> tuple[int, int]:
        return 20, 1

    def nvmlDeviceGetDecoderUtilization(self, handle: object) -> tuple[int, int]:
        return 30, 1

    def nvmlDeviceGetTotalEccErrors(self, handle: object, error_type: int, counter_type: int) -> int:
        self.ecc_calls.append((handle, error_type))
        if handle == 1:
            raise NotSupported()
        return error_type + 1

    def nvmlDeviceGetTemperature(self, handle: object, sensor: int) -> int:
        return 65

    def nvmlDeviceGetPowerUsage(self, handle: object) -> int:
        return 100_000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle: object) -> int:
        return 200_000

    def nvmlDeviceGetClockInfo(self, handle: object, clock: int) -> int:
        return clock * 100

    def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> list[Process]:
        return [
            Process(pid=os.getpid(), usedGpuMemory=128),
            Process(999999, 256),
            Process(os.getpid(), None),
            Process(os.getpid(), None),
        ]

    def nvmlSystemGetDriverVersion(self) -> str:
        return 'test-driver'

    def nvmlSystemGetCudaDriverVersion(self) -> int:
        return 13000


def test_all_gpu_metrics(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    nvml = FakeNvml()
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    collector = GpuCollector(ProcessTree())
    collector.initialize()
    collector.install(logfire.DEFAULT_LOGFIRE_INSTANCE)
    assert collector.status == 'ok'
    data = collector.snapshot()
    assert {
        name: [sample.value for sample in samples]
        for name, samples in data.items()
        if not name.startswith('train.torch')
    } == snapshot(
        {
            'hw.gpu.utilization': [0.5, 0.2, 0.3, 0.5, 0.2, 0.3],
            'train.gpu.memory_controller.utilization': [0.25, 0.25],
            'hw.gpu.memory.usage': [512, 512],
            'hw.gpu.memory.limit': [1024, 1024],
            'hw.gpu.memory.utilization': [0.5, 0.5],
            'hw.errors': [1, 2],
            'train.gpu.temperature': [65, 65],
            'train.gpu.power.usage': [100.0, 100.0],
            'train.gpu.power.limit': [200.0, 200.0],
            'train.gpu.clock.frequency': [100, 200, 300, 100, 200, 300],
            'train.gpu.process.memory.usage': [128, 128],
        }
    )
    assert dict(data['hw.gpu.utilization'][0].attributes or {}) == snapshot(
        {'hw.id': 'gpu-0', 'hw.name': 'Test GPU', 'hw.vendor': 'nvidia', 'gpu.index': 0, 'hw.gpu.task': 'general'}
    )
    count = nvml.memory_calls
    capfire.metrics_reader.get_metrics_data()
    assert nvml.memory_calls == count
    monkeypatch.setattr(_cache, 'time', SimpleNamespace(monotonic=lambda: 10**12))
    collector.snapshot()
    assert nvml.ecc_calls.count((1, 0)) == 1
    assert nvml.ecc_calls.count((1, 1)) == 1
    collector.close()
    collector.close()
    assert nvml.closed


def test_no_nvml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'pynvml', None)
    with TrainingRun(run_dir=tmp_path) as run:
        pass
    assert 'nvidia-ml-py not installed' in (run.dir / 'spans.jsonl').read_text()


def test_failed_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenNvml(FakeNvml):
        def nvmlInit(self) -> None:
            raise NvmlError('driver missing')

    monkeypatch.setitem(sys.modules, 'pynvml', BrokenNvml())
    collector = GpuCollector(ProcessTree())
    collector.initialize()
    assert collector.status == 'NVML unavailable: driver missing'
    collector.close()


def test_torch_allocators(monkeypatch: pytest.MonkeyPatch) -> None:
    def allocated(device: int) -> int:
        return 10

    def reserved(device: int) -> int:
        return 20

    def peak(device: int) -> int:
        return 30

    initialized = False
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: initialized,
            device_count=lambda: 1,
            memory_allocated=allocated,
            memory_reserved=reserved,
            max_memory_allocated=peak,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(current_allocated_memory=lambda: 40, driver_allocated_memory=lambda: 50),
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    collector = GpuCollector(ProcessTree())
    assert 'train.torch.memory.peak_allocated' not in collector.snapshot()
    initialized = True
    monkeypatch.setattr(_cache, 'time', SimpleNamespace(monotonic=lambda: 10**12))
    data = collector.snapshot()
    assert {key: [observation.value for observation in observations] for key, observations in data.items()} == snapshot(
        {
            'train.torch.memory.allocated': [10, 40],
            'train.torch.memory.reserved': [20, 50],
            'train.torch.memory.peak_allocated': [30],
        }
    )
