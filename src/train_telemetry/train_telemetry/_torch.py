"""Optional torch interface, without a dependency or CUDA initialization."""

from __future__ import annotations

import importlib
import sys
from typing import Protocol, cast


class Cuda(Protocol):
    def is_initialized(self) -> bool: ...
    def is_available(self) -> bool: ...
    def device_count(self) -> int: ...
    def memory_allocated(self, device: int) -> int: ...
    def memory_reserved(self, device: int) -> int: ...
    def max_memory_allocated(self, device: int) -> int: ...
    def get_device_capability(self, device: int) -> tuple[int, int]: ...


class Mps(Protocol):
    def current_allocated_memory(self) -> int: ...
    def driver_allocated_memory(self) -> int: ...


class MpsBackend(Protocol):
    def is_available(self) -> bool: ...


class Cudnn(Protocol):
    def version(self) -> int | None: ...


class Backends(Protocol):
    mps: MpsBackend
    cudnn: Cudnn


class Version(Protocol):
    cuda: str | None


class Torch(Protocol):
    __version__: str
    cuda: Cuda
    mps: Mps
    backends: Backends
    version: Version


def get_torch(*, import_if_available: bool = False) -> Torch | None:
    module = sys.modules.get('torch')
    if module is None and import_if_available:
        try:
            module = importlib.import_module('torch')
        except ImportError:
            return None
    # torch is optional and loaded dynamically; the protocol describes only the
    # stable public interfaces used here, avoiding a runtime torch dependency.
    return cast(Torch, module) if module is not None else None
