"""OTLP JSON Lines exporters and Logfire configuration."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import IO

import logfire
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message
from opentelemetry.exporter.otlp.proto.common._internal._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common._internal.metrics_encoder import encode_metrics
from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import encode_spans
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExporter,
    MetricExportResult,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from ._config import RunConfig

# Match Logfire's delta counters/histograms and cumulative up/down counters.
TEMPORALITY = {
    Counter: AggregationTemporality.DELTA,
    Histogram: AggregationTemporality.DELTA,
    ObservableCounter: AggregationTemporality.DELTA,
    UpDownCounter: AggregationTemporality.CUMULATIVE,
    ObservableUpDownCounter: AggregationTemporality.CUMULATIVE,
    ObservableGauge: AggregationTemporality.CUMULATIVE,
}


class _JsonlWriter:
    def __init__(self, path: Path, *, lazy: bool = False) -> None:
        self.path = path
        self._lock = Lock()
        self._closed = False
        self._file: IO[str] | None = None if lazy else path.open('a', encoding='utf-8')

    def write(self, message: Message) -> bool:
        line = MessageToJson(message, indent=None)
        with self._lock:
            if self._closed:
                return False
            try:
                if self._file is None:
                    self._file = self.path.open('a', encoding='utf-8')
                self._file.write(line + '\n')
                self._file.flush()
            except OSError:
                return False
        return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._file is not None:
                self._file.close()


class JsonlSpanExporter(SpanExporter):
    """Write OTLP trace requests, one JSON object per line.

    Long running spans may be exported repeatedly. Consumers must retain the last
    version of each span ID (within its trace ID).
    """

    def __init__(self, path: Path) -> None:
        self._writer = _JsonlWriter(path)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS if self._writer.write(encode_spans(spans)) else SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._writer.close()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class JsonlMetricExporter(MetricExporter):
    """Write OTLP metric requests using Logfire's aggregation temporality."""

    def __init__(self, path: Path) -> None:
        super().__init__(preferred_temporality=TEMPORALITY)
        self._writer = _JsonlWriter(path)

    def export(self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: object) -> MetricExportResult:
        return (
            MetricExportResult.SUCCESS
            if self._writer.write(encode_metrics(metrics_data))
            else MetricExportResult.FAILURE
        )

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        self._writer.close()

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True


class JsonlLogExporter(LogRecordExporter):
    """Create `logs.jsonl` only when a library emits native OTel logs."""

    def __init__(self, path: Path) -> None:
        self._writer = _JsonlWriter(path, lazy=True)

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        return (
            LogRecordExportResult.SUCCESS if self._writer.write(encode_logs(batch)) else LogRecordExportResult.FAILURE
        )

    def shutdown(self) -> None:
        self._writer.close()

    def force_flush(self, timeout_millis: int = 10_000) -> bool:
        return True


def configure_logfire(config: RunConfig, run_id: str, directory: Path) -> logfire.Logfire:
    """Configure both destinations before any instruments are registered."""
    interval = str(config.interval_s * 1000)
    previous_interval = os.environ.get('OTEL_METRIC_EXPORT_INTERVAL')
    os.environ['OTEL_METRIC_EXPORT_INTERVAL'] = interval
    processors: list[BatchSpanProcessor] = []
    readers: list[PeriodicExportingMetricReader] = []
    log_processors: list[BatchLogRecordProcessor] = []
    try:
        if config.files:
            processors.append(BatchSpanProcessor(JsonlSpanExporter(directory / 'spans.jsonl')))
            readers.append(
                PeriodicExportingMetricReader(
                    JsonlMetricExporter(directory / 'metrics.jsonl'), export_interval_millis=config.interval_s * 1000
                )
            )
            log_processors.append(BatchLogRecordProcessor(JsonlLogExporter(directory / 'logs.jsonl')))
        return logfire.configure(
            send_to_logfire=config.send_to_logfire,
            token=os.environ.get('LOGFIRE_TOKEN') if config.send_to_logfire else None,
            service_name=config.service_name,
            environment=config.environment,
            resource_attributes={'service.instance.id': run_id, 'train.run.name': config.run_name},
            console=False,
            additional_span_processors=processors,
            metrics=logfire.MetricsOptions(additional_readers=readers),
            advanced=logfire.AdvancedOptions(
                base_url=os.environ.get('LOGFIRE_BASE_URL'), log_record_processors=log_processors
            ),
            inspect_arguments=False,
            scrubbing=config.scrubbing,
        )
    except BaseException:
        for processor in processors:
            processor.shutdown()
        for reader in readers:
            # OpenTelemetry leaves its shutdown kwargs untyped.
            reader.shutdown()  # pyright: ignore[reportUnknownMemberType]
        for processor in log_processors:
            processor.shutdown()
        raise
    finally:
        if previous_interval is None:
            os.environ.pop('OTEL_METRIC_EXPORT_INTERVAL', None)
        else:
            os.environ['OTEL_METRIC_EXPORT_INTERVAL'] = previous_interval
