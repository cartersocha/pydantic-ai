"""Run provenance, installed packages and allowlisted environment variables."""

from __future__ import annotations

import getpass
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psutil

from ._config import RunConfig
from ._torch import get_torch


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _cpu_model() -> str:
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=2, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if sys.platform.startswith('linux'):
        try:
            for line in Path('/proc/cpuinfo').read_text(encoding='utf-8').splitlines():
                if line.startswith(('model name', 'Hardware')):
                    return line.partition(':')[2].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def sanitize_remote(remote: str) -> str:
    """Remove URL credentials, query strings and fragments, including scp syntax."""
    if '://' not in remote:
        return remote.split('@', 1)[-1]
    parsed = urlsplit(remote)
    return urlunsplit((parsed.scheme, parsed.netloc.rsplit('@', 1)[-1], parsed.path, '', ''))


def collect(
    config: RunConfig,
    run_id: str,
    directory: Path,
    *,
    started_at: str,
    program: str | None = None,
    module: str | None = None,
    args: list[str] | None = None,
    gpus: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], int]:
    """Collect once at entry. Package names are kept in files, never in the log."""
    cwd = Path.cwd()
    disk = psutil.disk_usage(str(directory if config.files else cwd))
    host: dict[str, object] = {'hostname': socket.gethostname()}
    if config.capture_user:
        try:
            host['user'] = getpass.getuser()
        except (OSError, KeyError):
            host['user'] = None
    packages = sorted(
        {
            f'{distribution.metadata["Name"]}=={distribution.version}'
            for distribution in importlib.metadata.distributions()
        }
    )
    commit = _git(cwd, 'rev-parse', 'HEAD')
    remote = _git(cwd, 'config', '--get', 'remote.origin.url')
    git = (
        None
        if commit is None
        else {
            'commit': commit,
            'branch': _git(cwd, 'branch', '--show-current'),
            'remote': sanitize_remote(remote) if remote else None,
            'dirty': bool(_git(cwd, 'status', '--porcelain')),
        }
    )
    hardware: dict[str, object] = {
        'cpu_model': _cpu_model(),
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'disk_total': disk.total,
        'disk_used': disk.used,
        'gpus': gpus or [],
    }
    metadata: dict[str, object] = {
        'run': {
            'id': run_id,
            'name': config.run_name,
            'started_at': started_at,
            'program': program,
            'module': module,
            'args': args or [],
            'cwd': str(cwd),
            'executable': sys.executable,
        },
        'python': {'version': platform.python_version(), 'implementation': platform.python_implementation()},
        'os': platform.platform(),
        'host': host,
        'hardware': hardware,
        'git': git,
        'env': {
            key: value
            for key, value in os.environ.items()
            if any(
                key.startswith(entry) if entry.endswith('_') else key == entry
                for entry in config.env_allowlist
                if entry
            )
        },
        'packages': packages,
    }
    torch = get_torch(import_if_available=True)
    if torch is not None:
        if torch.cuda.is_initialized():
            capabilities = {}
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                capabilities[device.uuid] = [device.major, device.minor]
            for gpu in gpus or []:
                gpu_uuid = gpu.get('uuid')
                if isinstance(gpu_uuid, str) and gpu_uuid in capabilities:
                    gpu['compute_capability'] = capabilities[gpu_uuid]
        metadata['torch'] = {
            'version': torch.__version__,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'device_count': torch.cuda.device_count(),
        }
    if config.files:
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        (directory / 'requirements.txt').write_text('\n'.join(packages) + '\n', encoding='utf-8')
        if config.capture_git_diff and commit:
            diff = _git(cwd, 'diff', 'HEAD', '--no-ext-diff')
            if diff:
                (directory / 'diff.patch').write_text(diff + '\n', encoding='utf-8')
    return metadata, len(packages)
