from __future__ import annotations

import faulthandler
import io
import json
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, TextIO

import logfire
import psutil
import pytest
from logfire.testing import CaptureLogfire
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import train_telemetry._metadata as metadata_module
import train_telemetry._system_metrics as system_module
import train_telemetry._torch as torch_module
from train_telemetry import RunConfig, TrainingRun
from train_telemetry.__main__ import cli, main
from train_telemetry._export import JsonlLogExporter, JsonlSpanExporter
from train_telemetry._gpu import GpuCollector
from train_telemetry._system_metrics import ProcessTree

from .test_gpu import FakeNvml, Memory, NvmlError, Process, Utilization


def test_config_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='empty'):
        RunConfig(run_name=' ')
    config = RunConfig(run_name='original', run_dir=tmp_path, gpu=False)
    run = TrainingRun(config, name='new', interval_s=3)
    assert run.config.run_name == 'new'
    assert run.config.interval_s == 3
    assert config.run_name == 'original'
    with run:
        pass


def test_configure_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> logfire.Logfire:
        raise ValueError('configuration failed')

    with monkeypatch.context() as patch:
        patch.setattr(logfire, 'configure', fail)
        with pytest.raises(ValueError, match='configuration failed'):
            with TrainingRun(run_dir=tmp_path, gpu=False):
                pass
    with TrainingRun(run_dir=tmp_path, gpu=False):
        pass


def test_exporter_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open
    original_shutdown = BatchSpanProcessor.shutdown
    shutdown: list[bool] = []

    def open_file(
        self: Path,
        mode: Literal['r', 'a', 'w'] = 'r',
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> TextIO:
        if self.name == 'metrics.jsonl':
            raise OSError('metrics file unavailable')
        return original_open(self, mode, buffering, encoding, errors, newline)

    def close_processor(self: BatchSpanProcessor) -> None:
        shutdown.append(True)
        original_shutdown(self)

    run = TrainingRun(run_dir=tmp_path, gpu=False)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', open_file)
        patch.setattr(BatchSpanProcessor, 'shutdown', close_processor)
        with pytest.raises(OSError, match='metrics file unavailable'):
            with run:
                pass
    assert shutdown == [True]
    assert json.loads((run.dir / 'run.json').read_text())['status'] == 'failed'
    with TrainingRun(run_dir=tmp_path, gpu=False):
        pass


def test_startup_directory_failure(tmp_path: Path) -> None:
    parent = tmp_path / 'file'
    parent.write_text('not a directory')
    with pytest.raises(OSError):
        with TrainingRun(run_dir=parent):
            pass
    with TrainingRun(run_dir=tmp_path, gpu=False):
        pass


def test_cli_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / 'script.py'
    script.write_text('pass')
    monkeypatch.setattr(sys, 'argv', ['train-telemetry', '--run-dir', str(tmp_path / 'runs'), str(script)])
    with pytest.raises(SystemExit) as exc:
        cli()
    assert exc.value.code == 0
    blocked = tmp_path / 'blocked'
    blocked.write_text('file')
    assert main(['--run-dir', str(blocked), str(script)]) == 1
    with TrainingRun(run_dir=tmp_path, gpu=False):
        assert main(['--run-dir', str(tmp_path), str(script)]) == 1


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_signal_context(tmp_path: Path, sig: signal.Signals) -> None:
    previous = signal.getsignal(sig)
    run = TrainingRun(run_dir=tmp_path, gpu=False)
    with pytest.raises((SystemExit, KeyboardInterrupt)):
        with run:
            signal.raise_signal(sig)
    assert signal.getsignal(sig) == previous
    assert run.status == ('terminated' if sig == signal.SIGTERM else 'interrupted')


def test_background_context(tmp_path: Path) -> None:
    failures: list[BaseException] = []

    def collect() -> None:
        try:
            with TrainingRun(run_dir=tmp_path, gpu=False):
                pass
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=collect)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not failures


def test_faulthandler_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enabled: list[object] = []
    disabled: list[bool] = []
    monkeypatch.setattr(faulthandler, 'is_enabled', lambda: False)

    def enable(*, file: object) -> None:
        enabled.append(file)

    monkeypatch.setattr(faulthandler, 'enable', enable)
    monkeypatch.setattr(faulthandler, 'disable', lambda: disabled.append(True))
    with TrainingRun(run_dir=tmp_path, gpu=False):
        assert enabled
    assert disabled == [True]


def test_console_off_failure(tmp_path: Path) -> None:
    run = TrainingRun(run_dir=tmp_path, gpu=False, console='off')
    with pytest.raises(ValueError, match='failure'):
        with run:
            raise ValueError('failure')
    assert run.status == 'failed'


def test_file_export_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DiskFull(io.StringIO):
        def write(self, text: str) -> int:
            raise OSError('disk full')

    def open_file(self: Path, *args: object, **kwargs: object) -> io.StringIO:
        return DiskFull()

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', open_file)
        exporter = JsonlSpanExporter(tmp_path / 'spans.jsonl')
        assert exporter.export([]).name == 'FAILURE'
        exporter.shutdown()
    exporter = JsonlLogExporter(tmp_path / 'missing' / 'logs.jsonl')
    assert exporter.export([]).name == 'FAILURE'
    assert exporter.force_flush()
    exporter.shutdown()


def test_metadata_without_optional_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError('not available')

    monkeypatch.setattr(metadata_module.getpass, 'getuser', unavailable)

    def no_git(*args: object) -> None:
        return None

    def no_torch(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(metadata_module, '_git', no_git)
    monkeypatch.setattr(metadata_module, 'get_torch', no_torch)
    metadata, _ = metadata_module.collect(RunConfig(run_dir=tmp_path), 'test', tmp_path, started_at='now')
    assert metadata['host'] == {'hostname': metadata_module.socket.gethostname(), 'user': None}
    assert 'torch' not in metadata
    assert metadata['git'] is None


def test_gpu_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingNvml(FakeNvml):
        def nvmlDeviceGetMemoryInfo(self, handle: object) -> Memory:
            raise NvmlError('device lost')

        def nvmlDeviceGetUtilizationRates(self, handle: object) -> Utilization:
            raise NvmlError('device lost')

        def nvmlDeviceGetEncoderUtilization(self, handle: object) -> tuple[int, int]:
            raise NvmlError('device lost')

        def nvmlDeviceGetPowerUsage(self, handle: object) -> int:
            raise NvmlError('device lost')

        def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> list[Process]:
            raise NvmlError('device lost')

        def nvmlShutdown(self) -> None:
            raise NvmlError('already lost')

    monkeypatch.setitem(sys.modules, 'pynvml', MissingNvml())
    collector = GpuCollector(ProcessTree())
    collector.initialize()
    samples = collector.snapshot()
    assert 'hw.gpu.memory.usage' not in samples
    assert 'train.gpu.process.memory.usage' not in samples
    collector.close()


def test_gpu_discovery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingNvml(FakeNvml):
        def nvmlDeviceGetCount(self) -> int:
            raise NvmlError('discovery failed')

    monkeypatch.setitem(sys.modules, 'pynvml', MissingNvml())
    collector = GpuCollector(ProcessTree())
    collector.initialize()
    assert collector.status == 'NVML discovery failed: discovery failed'
    collector.close()


def test_torch_optional_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, 'torch', raising=False)

    def unavailable(name: str) -> None:
        raise ImportError('not installed')

    monkeypatch.setattr(torch_module, 'importlib', SimpleNamespace(import_module=unavailable))
    assert torch_module.get_torch(import_if_available=True) is None
    assert GpuCollector(ProcessTree()).snapshot() == {}


def test_existing_system_instrumentation(capfire: CaptureLogfire, tmp_path: Path) -> None:
    instrumentor = SystemMetricsInstrumentor()
    instrumentor.instrument()
    try:
        with pytest.raises(RuntimeError, match='already instrumented'):
            system_module.install(logfire.DEFAULT_LOGFIRE_INSTANCE, tmp_path, ProcessTree())
    finally:
        instrumentor.uninstrument()


def test_span_creation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: logfire.Logfire, template: str, **kwargs: object) -> logfire.LogfireSpan:
        raise ValueError('span creation failed')

    with monkeypatch.context() as patch:
        patch.setattr(logfire.Logfire, 'span', fail)
        with pytest.raises(ValueError, match='span creation failed'):
            with TrainingRun(run_dir=tmp_path, gpu=False):
                pass
    with TrainingRun(run_dir=tmp_path, gpu=False):
        pass


@pytest.mark.parametrize('cpu', ['processor: 0\nmodel name: Test CPU', 'no model field', None])
def test_linux_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu: str | None) -> None:
    monkeypatch.setattr(metadata_module, 'sys', SimpleNamespace(platform='linux', executable=sys.executable))
    monkeypatch.setattr(metadata_module.platform, 'processor', lambda: 'Fallback CPU')

    def no_torch(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(metadata_module, 'get_torch', no_torch)
    original = Path.read_text

    def cpuinfo(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if str(self) == '/proc/cpuinfo':
            if cpu is None:
                raise OSError('unavailable')
            return cpu
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, 'read_text', cpuinfo)
    metadata_module.collect(RunConfig(capture_git_diff=True), 'test', tmp_path, started_at='now')
    expected = 'Test CPU' if cpu and 'model name' in cpu else 'Fallback CPU'
    assert json.loads((tmp_path / 'metadata.json').read_text())['hardware']['cpu_model'] == expected


def test_missing_git_and_sysctl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError('command not installed')

    monkeypatch.setattr(metadata_module.subprocess, 'run', unavailable)
    metadata, _ = metadata_module.collect(RunConfig(), 'test', tmp_path, started_at='now')
    assert metadata['git'] is None
    assert metadata['hardware'] is not None


def test_sysctl_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata_module, 'sys', SimpleNamespace(platform='darwin', executable=sys.executable))
    monkeypatch.setattr(metadata_module.platform, 'processor', lambda: 'Fallback CPU')

    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout='')

    monkeypatch.setattr(metadata_module.subprocess, 'run', failed)
    metadata_module.collect(RunConfig(), 'test', tmp_path, started_at='now')
    assert json.loads((tmp_path / 'metadata.json').read_text())['hardware']['cpu_model'] == 'Fallback CPU'


def test_initialized_cuda_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Cuda:
        def is_initialized(self) -> bool:
            return True

        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 1

        def get_device_properties(self, device: int) -> SimpleNamespace:
            return SimpleNamespace(uuid='gpu-uuid', major=9, minor=0)

    runtime = SimpleNamespace(
        __version__='test',
        version=SimpleNamespace(cuda='13'),
        cuda=Cuda(),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False), cudnn=SimpleNamespace(version=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, 'torch', runtime)
    devices: list[dict[str, object]] = [{'uuid': 'gpu-uuid'}, {'uuid': 'not-visible'}, {}]
    metadata_module.collect(RunConfig(), 'test', tmp_path, started_at='now', gpus=devices)
    assert devices[0]['compute_capability'] == [9, 0]
    assert 'compute_capability' not in devices[1]


def test_clean_git_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def clean_git(cwd: Path, *args: str) -> str:
        return 'commit' if args == ('rev-parse', 'HEAD') else ''

    monkeypatch.setattr(metadata_module, '_git', clean_git)
    metadata_module.collect(RunConfig(capture_git_diff=True), 'test', tmp_path, started_at='now')
    assert not (tmp_path / 'diff.patch').exists()


def test_zero_gpu_memory_and_allocator_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    class ZeroMemory(FakeNvml):
        def nvmlDeviceGetMemoryInfo(self, handle: object) -> Memory:
            return Memory(0, 0)

    monkeypatch.setitem(sys.modules, 'pynvml', ZeroMemory())
    runtime = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, 'torch', runtime)
    collector = GpuCollector(ProcessTree())
    collector.initialize()
    assert 'hw.gpu.memory.utilization' not in collector.snapshot()
    collector.close()

    def unavailable() -> bool:
        raise RuntimeError('backend shut down')

    runtime.cuda.is_initialized = unavailable
    assert GpuCollector(ProcessTree()).snapshot() == {}


def test_system_query_failure(capfire: CaptureLogfire, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = ProcessTree()

    def inaccessible(self: psutil.Process, recursive: bool = False) -> list[psutil.Process]:
        raise psutil.AccessDenied()

    with monkeypatch.context() as patch:
        patch.setattr(psutil.Process, 'children', inaccessible)
        assert len(process.pids()) == 1
        patch.setattr(psutil.Process, 'oneshot', inaccessible)
        assert list(process.callback('train.process.tree.memory.rss')(system_module.CallbackOptions()))[0].value == 0

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError('filesystem unavailable')

    with monkeypatch.context() as patch:
        patch.setattr(psutil, 'disk_partitions', unavailable)
        uninstall = system_module.install(logfire.DEFAULT_LOGFIRE_INSTANCE, tmp_path, ProcessTree())
        try:
            patch.setattr(psutil, 'disk_usage', unavailable)
            data = capfire.metrics_reader.get_metrics_data()
            assert data is not None
            assert all(
                metric.name != 'system.filesystem.usage'
                for resource in data.resource_metrics
                for scope in resource.scope_metrics
                for metric in scope.metrics
            )
        finally:
            uninstall()
