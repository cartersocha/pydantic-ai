"""Configuration and environment defaults."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import logfire

ENV_ALLOWLIST = (
    'SLURM_',
    'CUDA_VISIBLE_DEVICES',
    'WORLD_SIZE',
    'RANK',
    'LOCAL_RANK',
    'LOCAL_WORLD_SIZE',
    'MASTER_ADDR',
    'OMP_NUM_THREADS',
    'PYTORCH_CUDA_ALLOC_CONF',
)


def _env(name: str, default: str) -> str:
    return os.environ.get(f'TRAIN_TELEMETRY_{name}', default)


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """Options for one run. Explicit values override `TRAIN_TELEMETRY_*` defaults.

    `files=False` requires an enabled Logfire destination and `LOGFIRE_TOKEN`.
    `scrubbing` customizes Logfire scrubbing; raw console files are never scrubbed.
    Environment allowlist entries ending in `_` match prefixes; other entries match
    exact names. Only one `TrainingRun` may be active in a process.
    """

    run_name: str = field(default_factory=lambda: _env('RUN_NAME', 'training'))
    run_dir: Path = field(default_factory=lambda: Path(_env('RUN_DIR', './runs')))
    service_name: str = field(default_factory=lambda: _env('SERVICE_NAME', 'training'))
    environment: str = field(default_factory=lambda: _env('ENVIRONMENT', 'local'))
    interval_s: float = field(default_factory=lambda: float(_env('INTERVAL', '15')))
    files: bool = True
    logfire: Literal['if-token-present', False] = 'if-token-present'
    gpu: Literal['auto', False] = 'auto'
    console: Literal['tee', 'off'] = 'tee'
    console_to_logfire: bool = True
    log_level: int = logging.INFO
    capture_user: bool = True
    capture_git_diff: bool = False
    env_allowlist: tuple[str, ...] = field(
        default_factory=lambda: tuple(_env('ENV_ALLOWLIST', ','.join(ENV_ALLOWLIST)).split(','))
    )
    scrubbing: logfire.ScrubbingOptions | None = None

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError('`run_name` must not be empty')
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError('`interval_s` must be finite and positive')
        if not self.files and not self.send_to_logfire:
            raise ValueError('`files=False` requires `LOGFIRE_TOKEN` and an enabled Logfire destination')

    @property
    def send_to_logfire(self) -> bool:
        """Whether this run exports to Logfire; credentials files are not consulted."""
        return self.logfire is not False and bool(os.environ.get('LOGFIRE_TOKEN'))
