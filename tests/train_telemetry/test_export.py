from __future__ import annotations

import os
from pathlib import Path

import pytest
from google.protobuf.json_format import Parse
from opentelemetry._logs import SeverityNumber, get_logger
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.sdk.trace.export import SpanExportResult
from train_telemetry._export import JsonlSpanExporter

from train_telemetry import TrainingRun

from .conftest import Receiver


def test_native_logs(tmp_path: Path) -> None:
    with TrainingRun(run_dir=tmp_path, gpu=False) as run:
        get_logger('third_party').emit(
            body='native otel log', severity_number=SeverityNumber.INFO, attributes={'password': 'supersecret'}
        )
    text = (run.dir / 'logs.jsonl').read_text()
    assert 'supersecret' not in text
    request = Parse(text.splitlines()[0], ExportLogsServiceRequest())
    assert request.resource_logs[0].scope_logs[0].log_records[0].body.string_value == 'native otel log'
    assert 'Scrubbed' in text


def test_exporter_closed(tmp_path: Path) -> None:
    exporter = JsonlSpanExporter(tmp_path / 'spans.jsonl')
    assert exporter.export([]) == SpanExportResult.SUCCESS
    exporter.shutdown()
    assert exporter.export([]) == SpanExportResult.FAILURE
    assert exporter.force_flush()
    exporter.shutdown()


def test_remote_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, otlp_receiver: Receiver) -> None:
    monkeypatch.setenv('LOGFIRE_TOKEN', 'test-token')
    monkeypatch.setenv('LOGFIRE_BASE_URL', otlp_receiver.url)
    with TrainingRun(run_dir=tmp_path / 'not-created', files=False, gpu=False) as run:
        print('remote only')
    assert not run.dir.parent.exists()
    assert any(route == '/v1/traces' for route, _ in otlp_receiver.requests)


def test_interval_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OTEL_METRIC_EXPORT_INTERVAL', '12345')
    with TrainingRun(run_dir=tmp_path, interval_s=0.2, gpu=False):
        assert os.environ['OTEL_METRIC_EXPORT_INTERVAL'] == '12345'
    assert os.environ['OTEL_METRIC_EXPORT_INTERVAL'] == '12345'
