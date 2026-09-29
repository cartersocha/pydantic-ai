from __future__ import annotations

import io
import logging
import threading
import warnings
from pathlib import Path
from typing import Literal

import logfire
import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire

from train_telemetry import TrainingRun
from train_telemetry._console import CaptureGuard, ConsoleTee

from .utils import spans


def test_tee(capfire: CaptureLogfire) -> None:
    original, raw = io.StringIO(), io.StringIO()
    tee = ConsoleTee(
        original, raw, threading.RLock(), logfire.DEFAULT_LOGFIRE_INSTANCE, CaptureGuard(), stream='stdout'
    )
    assert tee.write('old\r') == 4
    tee.write('new {braces}\r\npartial')
    tee.flush()
    assert raw.getvalue() == 'old\rnew {braces}\r\npartial'
    assert original.getvalue() == raw.getvalue()
    assert [span['attributes']['logfire.msg'] for span in capfire.exporter.exported_spans_as_dict()] == snapshot(
        ['new {braces}']
    )
    assert not tee.isatty()
    with pytest.raises(io.UnsupportedOperation):
        tee.fileno()
    tee.close()
    assert not original.closed


def test_logging_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Restore the application's logging configuration after basicConfig(force=True).
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        with TrainingRun(run_dir=tmp_path, gpu=False) as run:
            logging.basicConfig(force=True, level=logging.INFO)
            logging.getLogger('train.example').info('structured {braces}')
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn('one warning', UserWarning)
            print('plain console {braces}')
            print('password=supersecret')
        text = (run.dir / 'spans.jsonl').read_text()
        # Inspect final OTLP records, excluding pending-span re-exports.

        records = spans(run.dir)
        messages = [span['attributes'].get('logfire.msg') for span in records]
        assert messages.count('structured {braces}') == 1
        assert messages.count('UserWarning: one warning') == 1
        assert messages.count('plain console {braces}') == 1
        assert 'supersecret' not in text
        raw = (run.dir / 'console.log').read_text()
        assert 'supersecret' in raw
        assert 'structured {braces}' in raw
    finally:
        root.handlers, root.level = handlers, level


def test_thread_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[threading.ExceptHookArgs] = []
    previous = threading.excepthook
    monkeypatch.setattr(threading, 'excepthook', seen.append)
    with TrainingRun(run_dir=tmp_path, gpu=False) as run:

        def crash() -> None:
            raise ValueError('thread broke')

        thread = threading.Thread(target=crash)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    monkeypatch.setattr(threading, 'excepthook', previous)
    assert len(seen) == 1
    assert 'thread broke' in (run.dir / 'spans.jsonl').read_text()


def test_stream_delegation(capfire: CaptureLogfire, tmp_path: Path) -> None:
    with (tmp_path / 'original').open('w') as original:
        tee = ConsoleTee(
            original, None, threading.RLock(), logfire.DEFAULT_LOGFIRE_INSTANCE, CaptureGuard(), stream='stderr'
        )
        assert tee.encoding == original.encoding
        assert tee.errors == original.errors
        assert tee.buffer is original.buffer
        assert tee.fileno() == original.fileno()
        assert tee.writable()
        tee.write('progress\rnew\rfinal')
        tee.write('\r\n')
        tee.close()
    assert capfire.exporter.exported_spans_as_dict()[0]['attributes']['logfire.msg'] == 'final'


@pytest.mark.parametrize(('console', 'to_logfire'), [('off', True), ('tee', False)])
def test_console_modes(tmp_path: Path, console: Literal['off', 'tee'], to_logfire: bool) -> None:

    with TrainingRun(run_dir=tmp_path, console=console, console_to_logfire=to_logfire, gpu=False) as run:
        print('a console line')
    assert not any(record['attributes'].get('train.log.source') == 'console' for record in spans(run.dir))
    if console == 'tee':
        assert 'a console line' in (run.dir / 'console.log').read_text()


def test_reconfigure_warning(tmp_path: Path) -> None:
    with TrainingRun(run_dir=tmp_path, gpu=False) as run:
        with pytest.warns(UserWarning, match='replaces'):
            logfire.configure(send_to_logfire=False, console=False, inspect_arguments=False)
    assert run.status == 'ok'


def test_logging_levels(tmp_path: Path) -> None:
    logger = logging.getLogger('training.levels')
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with TrainingRun(run_dir=tmp_path, gpu=False) as run:
            logger.debug('below capture level')
            logger.info('captured info')
            logging.getLogger('opentelemetry.test').warning('excluded telemetry')
        messages = [record['attributes'].get('logfire.msg') for record in spans(run.dir)]
        assert 'captured info' in messages
        assert 'below capture level' not in messages
        assert 'excluded telemetry' not in messages
    finally:
        logger.setLevel(previous)
