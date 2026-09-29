from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from .utils import spans

WORKLOAD = Path(__file__).parents[2] / 'src/train_telemetry/examples/train_mnist.py'
pytestmark = pytest.mark.train_acceptance


def _environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('COVERAGE_') and key not in ('LOGFIRE_TOKEN', 'LOGFIRE_BASE_URL')
    }


@pytest.mark.skipif(
    importlib.util.find_spec('torch') is None or importlib.util.find_spec('torchvision') is None,
    reason='Install train-telemetry[example]',
)
@pytest.mark.parametrize(('crash', 'code', 'status'), [(False, 0, 'ok'), (True, 1, 'failed')])
def test_mnist(tmp_path: Path, crash: bool, code: int, status: str) -> None:
    env = _environment()
    env['OMP_NUM_THREADS'] = '2'
    command = [
        sys.executable,
        '-m',
        'train_telemetry',
        '--run-dir',
        str(tmp_path),
        str(WORKLOAD),
        '--data',
        'fake',
        '--epochs',
        '1',
        '--limit-batches',
        '2',
        '--batch-size',
        '4',
        '--device',
        'cpu',
    ]
    if crash:
        command += ['--crash-at-step', '1']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == code, result.stderr
    directory = next(tmp_path.iterdir())
    state = json.loads((directory / 'run.json').read_text())
    assert state['status'] == status
    for name in (
        'metadata.json',
        'requirements.txt',
        'console.log',
        'spans.jsonl',
        'metrics.jsonl',
        'faulthandler.log',
    ):
        assert (directory / name).is_file()
    records = spans(directory)
    run_span = next(span for span in records if span['name'] == 'training run {run_name}')
    assert run_span['attributes']['train.run.status'] == status
    assert any(span['attributes'].get('logfire.logger_name') == 'train_mnist' for span in records)
    raw = (directory / 'console.log').read_text()
    if crash:
        assert 'Traceback (most recent call last)' in raw
        assert run_span['events'][0]['name'] == 'exception'
    else:
        assert 'epoch 1 batch 0/' in raw
        warnings = [span for span in records if span['attributes'].get('train.log.source') == 'warnings']
        assert len(warnings) == 1


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process signals')
@pytest.mark.parametrize(
    ('sig', 'code', 'status'), [(signal.SIGTERM, 143, 'terminated'), (signal.SIGINT, 130, 'interrupted')]
)
def test_signal(tmp_path: Path, sig: signal.Signals, code: int, status: str) -> None:
    script = tmp_path / 'wait.py'
    script.write_text("import time\nprint('ready', flush=True)\ntime.sleep(120)\n")
    runs = tmp_path / 'runs'
    child = subprocess.Popen(
        [sys.executable, '-m', 'train_telemetry', '--run-dir', str(runs), str(script)],
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == 'ready'
        child.send_signal(sig)
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == code, (stdout, stderr)
        directory = next(runs.iterdir())
        state = json.loads((directory / 'run.json').read_text())
        assert state['status'] == status
        assert state['ended_at'] is not None
        assert (directory / 'metrics.jsonl').stat().st_size > 0
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_unreachable_logfire(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    script = tmp_path / 'offline.py'
    script.write_text("print('training completed')\n")
    env = _environment()
    env.update(LOGFIRE_TOKEN='test-token', LOGFIRE_BASE_URL=f'http://127.0.0.1:{port}', OTEL_EXPORTER_OTLP_TIMEOUT='1')
    result = subprocess.run(
        [sys.executable, '-m', 'train_telemetry', '--run-dir', str(tmp_path / 'runs'), str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    directory = next((tmp_path / 'runs').iterdir())
    assert json.loads((directory / 'run.json').read_text())['status'] == 'ok'
    assert 'training completed' in (directory / 'console.log').read_text()
    assert 'Traceback' not in (directory / 'console.log').read_text()
    assert (directory / 'metrics.jsonl').stat().st_size > 0


@pytest.mark.skipif(
    importlib.util.find_spec('torch') is None or importlib.util.find_spec('torchvision') is None,
    reason='Install train-telemetry[example]',
)
def test_dataloader_workers(tmp_path: Path) -> None:
    env = _environment()
    env['OMP_NUM_THREADS'] = '2'
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'train_telemetry',
            '--run-dir',
            str(tmp_path),
            '--interval',
            '0.2',
            str(WORKLOAD),
            '--data',
            'fake',
            '--epochs',
            '1',
            '--limit-batches',
            '2',
            '--batch-size',
            '4',
            '--device',
            'cpu',
            '--num-workers',
            '2',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    directory = next(tmp_path.iterdir())
    children = [
        point['asInt']
        for line in (directory / 'metrics.jsonl').read_text().splitlines()
        for resource in json.loads(line)['resourceMetrics']
        for scope in resource['scopeMetrics']
        for metric in scope['metrics']
        if metric['name'] == 'train.process.tree.children'
        for point in metric['gauge']['dataPoints']
    ]
    assert max(int(value) for value in children) >= 2
