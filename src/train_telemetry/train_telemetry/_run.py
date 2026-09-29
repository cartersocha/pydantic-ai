"""Run lifecycle and process state restoration."""

from __future__ import annotations

import faulthandler
import json
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Literal

import logfire
from typing_extensions import TypedDict, Unpack

from ._config import RunConfig
from ._console import CaptureGuard, ConsoleTee
from ._export import configure_logfire
from ._gpu import GpuCollector
from ._logging import install as install_logging
from ._metadata import collect
from ._system_metrics import ProcessTree, install as install_system_metrics

Status = Literal['running', 'ok', 'failed', 'interrupted', 'terminated']
_ACTIVE_LOCK = threading.Lock()


class RunOptions(TypedDict, total=False):
    """Keyword overrides accepted by `TrainingRun`."""

    run_name: str
    run_dir: Path
    service_name: str
    environment: str
    interval_s: float
    files: bool
    logfire: Literal['if-token-present', False]
    gpu: Literal['auto', False]
    console: Literal['tee', 'off']
    console_to_logfire: bool
    log_level: int
    capture_user: bool
    capture_git_diff: bool
    env_allowlist: tuple[str, ...]
    scrubbing: logfire.ScrubbingOptions | None


class _Terminated(SystemExit):
    def __init__(self) -> None:
        super().__init__(143)


def _signal_handler(signum: int, frame: object) -> None:
    if signum == signal.SIGTERM:
        raise _Terminated()
    raise KeyboardInterrupt


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingRun:
    """Collect telemetry for the lifetime of a synchronous context.

    Use `TrainingRun(name='baseline')` or pass a frozen `RunConfig`. Keyword
    overrides and `name` take precedence over that config. Entry configures
    process-wide Logfire; exit flushes and shuts down its providers. Only one run
    can be active, and each context instance can be entered once. Python stream,
    logging, warning, thread and signal hooks are restored on exit.
    """

    def __init__(
        self, config: RunConfig | None = None, *, name: str | None = None, **options: Unpack[RunOptions]
    ) -> None:
        self.config = replace(config, **options) if config is not None else RunConfig(**options)
        if name is not None:
            self.config = replace(self.config, run_name=name)
        self.id = uuid.uuid4().hex[:12]
        safe_name = re.sub(r'[^\w.-]+', '-', self.config.run_name).strip('.-') or 'training'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        self.dir = self.config.run_dir.resolve() / f'{safe_name}-{stamp}-{self.id[:8]}'
        self.status: Status = 'running'
        self._remote = self.config.send_to_logfire
        self._console_file: IO[str] | None = None
        self.exit_code = 0
        self.program: str | None = None
        self.module: str | None = None
        self.args: list[str] = []
        self._stack = ExitStack()
        self._entered = False
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._duration = 0.0
        self._started = 0.0
        self._logfire: logfire.Logfire | None = None
        self._span: logfire.LogfireSpan | None = None

    def _write_state(self) -> None:
        if self.config.files:
            state = {
                'id': self.id,
                'name': self.config.run_name,
                'started_at': self._started_at,
                'ended_at': self._ended_at,
                'duration_s': self._duration,
                'status': self.status,
                'exit_code': self.exit_code,
                'destinations': {'files': self.config.files, 'logfire': self._remote},
            }
            temporary = self.dir / 'run.json.tmp'
            temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
            temporary.replace(self.dir / 'run.json')

    def __enter__(self) -> TrainingRun:
        if self._entered:
            raise RuntimeError('A `TrainingRun` instance can only be entered once')
        if not _ACTIVE_LOCK.acquire(blocking=False):
            raise RuntimeError('Only one `TrainingRun` can be active per process')
        self._entered = True
        self._stack.callback(_ACTIVE_LOCK.release)
        try:
            if self.config.files:
                self.dir.mkdir(parents=True, exist_ok=False)
            self._started_at = _utc_now()
            self._started = time.monotonic()
            self._write_state()
            self._logfire = configure_logfire(self.config, self.id, self.dir)
            self._span = self._logfire.span('training run {run_name}', run_name=self.config.run_name)
            self._span.set_attribute('train.run.id', self.id)
            self._span.__enter__()
            self._install_signals()
            guard = CaptureGuard()
            install_logging(self._logfire, self._stack, guard, self.config.log_level)
            if self.config.console == 'tee':
                raw = (
                    self._stack.enter_context((self.dir / 'console.log').open('a', encoding='utf-8'))
                    if self.config.files
                    else None
                )
                self._console_file = raw
                raw_lock = threading.RLock()
                stdout, stderr = sys.stdout, sys.stderr

                def restore_streams() -> None:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    sys.stdout, sys.stderr = stdout, stderr

                self._stack.callback(restore_streams)
                sys.stdout = ConsoleTee(
                    stdout,
                    raw,
                    raw_lock,
                    self._logfire,
                    guard,
                    stream='stdout',
                    to_logfire=self.config.console_to_logfire,
                )
                sys.stderr = ConsoleTee(
                    stderr,
                    raw,
                    raw_lock,
                    self._logfire,
                    guard,
                    stream='stderr',
                    to_logfire=self.config.console_to_logfire,
                )
            if self.config.files:
                fault_file = self._stack.enter_context((self.dir / 'faulthandler.log').open('a', encoding='utf-8'))
                if not faulthandler.is_enabled():
                    faulthandler.enable(file=fault_file)
                    self._stack.callback(faulthandler.disable)
            tree = ProcessTree()
            self._stack.callback(
                install_system_metrics(self._logfire, self.dir if self.config.files else Path.cwd(), tree)
            )
            gpu = GpuCollector(tree)
            self._stack.callback(gpu.close)
            if self.config.gpu:
                gpu.initialize()
            metadata, package_count = collect(
                self.config,
                self.id,
                self.dir,
                started_at=self._started_at,
                program=self.program,
                module=self.module,
                args=self.args,
                gpus=gpu.inventory,
            )
            if self.config.gpu:
                gpu.install(self._logfire)
            metadata.pop('packages')
            self._logfire.info(
                'run metadata',
                metadata=metadata,
                package_count=package_count,
                gpu_status=gpu.status,
            )
            return self
        except BaseException as exc:
            if self._span is not None:
                self.__exit__(*sys.exc_info())
            else:
                self._finish(exc)
                try:
                    if self._logfire is not None:
                        self._logfire.shutdown()
                    if self._started_at is not None:
                        self._write_state()
                finally:
                    self._stack.close()
            raise

    def _install_signals(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous = signal.signal(sig, _signal_handler)
                self._stack.callback(signal.signal, sig, previous)

    def _finish(self, exc: BaseException | None) -> None:
        if isinstance(exc, _Terminated):
            self.status, self.exit_code = 'terminated', 143
        elif isinstance(exc, KeyboardInterrupt):
            self.status, self.exit_code = 'interrupted', 130
        elif isinstance(exc, SystemExit):
            self.exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            self.status = 'ok' if self.exit_code == 0 else 'failed'
        elif exc is not None:
            self.status, self.exit_code = 'failed', 1
        else:
            self.status, self.exit_code = 'ok', 0
        self._ended_at = _utc_now()
        self._duration = time.monotonic() - self._started

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> Literal[False]:
        self._finish(exc)
        instance, span = self._logfire, self._span
        assert instance is not None and span is not None
        try:
            if (
                exc is not None
                and not isinstance(exc, (SystemExit, KeyboardInterrupt))
                and self._console_file is not None
            ):
                traceback.print_exception(exc_type, exc, tb, file=self._console_file)
            span.set_attributes(
                {
                    'train.run.status': self.status,
                    'train.run.exit_code': self.exit_code,
                    'train.run.duration_s': self._duration,
                }
            )
            if self.exit_code:
                span.set_level('error')
            instance.log(
                'error' if self.exit_code else 'info',
                'run finished {status}',
                {
                    'status': self.status,
                    'exit_code': self.exit_code,
                    'duration_s': self._duration,
                    'run_dir': str(self.dir) if self.config.files else None,
                },
            )
            span.__exit__(exc_type, exc, tb)
            self._write_state()
        finally:
            try:
                instance.shutdown()
            finally:
                self._stack.close()
        return False
