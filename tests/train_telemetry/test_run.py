from __future__ import annotations

import json
import sys
from pathlib import Path

import logfire
import pytest
from google.protobuf.json_format import Parse
from inline_snapshot import snapshot
from opentelemetry.metrics import Observation
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

import train_telemetry._run as run_module
from train_telemetry import RunConfig, TrainingRun
from train_telemetry.__main__ import main

from .conftest import Receiver


def test_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('TRAIN_TELEMETRY_INTERVAL', '3')
    assert RunConfig().interval_s == 3
    assert RunConfig(interval_s=2).interval_s == 2
    for interval in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='positive'):
            RunConfig(interval_s=interval)
    with pytest.raises(ValueError, match='LOGFIRE_TOKEN'):
        RunConfig(files=False)
    with TrainingRun(name='a/b', run_dir=tmp_path, gpu=False) as run:
        with pytest.raises(RuntimeError, match='Only one'):
            with TrainingRun(run_dir=tmp_path):
                pass
    assert run.dir.parent == tmp_path
    with pytest.raises(RuntimeError, match='once'):
        with run:
            pass


@pytest.mark.parametrize(
    ('exception', 'status', 'code'),
    [
        (None, 'ok', 0),
        (ValueError('broken'), 'failed', 1),
        (SystemExit(7), 'failed', 7),
        (SystemExit(), 'ok', 0),
        (SystemExit('bad'), 'failed', 1),
        (KeyboardInterrupt(), 'interrupted', 130),
    ],
)
def test_lifecycle(tmp_path: Path, exception: BaseException | None, status: str, code: int) -> None:
    run = TrainingRun(name='example', run_dir=tmp_path, gpu=False)
    try:
        with run:
            if exception:
                raise exception
    except BaseException as exc:
        assert exc is exception
    state = json.loads((run.dir / 'run.json').read_text())
    assert (state['status'], state['exit_code']) == (status, code)
    assert state['ended_at'] is not None
    spans = [Parse(line, ExportTraceServiceRequest()) for line in (run.dir / 'spans.jsonl').read_text().splitlines()]
    assert spans
    assert all(request.resource_spans[0].resource.attributes for request in spans)


def test_dual_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, otlp_receiver: Receiver) -> None:
    monkeypatch.setenv('LOGFIRE_TOKEN', 'test-token')
    monkeypatch.setenv('LOGFIRE_BASE_URL', otlp_receiver.url)
    with TrainingRun(run_dir=tmp_path, interval_s=0.2, gpu=False) as run:
        logfire.metric_gauge_callback('test.metric', [lambda _: [Observation(1)]])
        logfire.info('test record')
        logfire.force_flush()
    routes = {path for path, _ in otlp_receiver.requests}
    assert {'/v1/traces', '/v1/metrics'} <= routes
    for path, body in otlp_receiver.requests:
        if path == '/v1/traces':
            request = ExportTraceServiceRequest.FromString(body)
            resources = [entry.resource for entry in request.resource_spans]
        elif path == '/v1/metrics':
            metrics = ExportMetricsServiceRequest.FromString(body)
            resources = [entry.resource for entry in metrics.resource_metrics]
        else:
            continue
        for resource in resources:
            assert any(
                attr.key == 'service.instance.id' and attr.value.string_value == run.id for attr in resource.attributes
            )
    assert Parse(
        (run.dir / 'metrics.jsonl').read_text().splitlines()[0], ExportMetricsServiceRequest()
    ).resource_metrics


def test_script_args(tmp_path: Path) -> None:
    script = tmp_path / 'script.py'
    script.write_text("import sys\nassert sys.argv[1:] == ['--run-name', 'target-value']\n")
    assert main(['--run-dir', str(tmp_path / 'runs'), str(script), '--run-name', 'target-value']) == snapshot(0)


def test_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_metadata(*args: object, **kwargs: object) -> tuple[dict[str, object], int]:
        raise OSError('metadata unavailable')

    with monkeypatch.context() as patch:
        patch.setattr(run_module, 'collect', fail_metadata)
        run = TrainingRun(run_dir=tmp_path, gpu=False)
        with pytest.raises(OSError, match='metadata unavailable'):
            with run:
                pass
    assert json.loads((run.dir / 'run.json').read_text())['status'] == 'failed'
    assert 'metadata unavailable' in (run.dir / 'console.log').read_text()
    with TrainingRun(run_dir=tmp_path, gpu=False):
        pass


@pytest.mark.parametrize(
    ('source', 'code'),
    [
        ("raise SystemExit('message')", 1),
        ('raise SystemExit(9)', 9),
        ('raise KeyboardInterrupt', 130),
        ('raise ValueError("broken")', 1),
    ],
)
def test_cli_exit_codes(tmp_path: Path, source: str, code: int) -> None:
    target = tmp_path / 'target.py'
    target.write_text(source)
    assert main(['--run-dir', str(tmp_path / 'runs'), str(target)]) == code


def test_cli_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'training_module.py').write_text(
        "import sys\nassert sys.argv[1:] == ['--target-flag']\nprint('module target')\n"
    )
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])
    assert main(['--run-dir', str(tmp_path / 'runs'), '-m', 'training_module', '--target-flag']) == 0


def test_cli_bad_config() -> None:
    with pytest.raises(SystemExit) as exc:
        main(['--interval', '0', 'missing.py'])
    assert exc.value.code == 2
