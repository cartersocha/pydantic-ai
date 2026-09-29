"""Restorable stdlib logging, warning and thread exception hooks."""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps
from typing import ParamSpec, TextIO, TypeVar

import logfire

from ._console import CaptureGuard

P = ParamSpec('P')
R = TypeVar('R')


def _warn_configure(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        warnings.warn(
            'Calling `logfire.configure()` during a training run replaces its telemetry destinations',
            UserWarning,
            stacklevel=2,
        )
        return function(*args, **kwargs)

    return wrapped


def install(instance: logfire.Logfire, stack: ExitStack, guard: CaptureGuard, level: int) -> None:
    """Capture records without installing a root handler or changing logger levels."""
    original_configure = logfire.configure
    original_handlers = logging.Logger.callHandlers
    original_emit = logging.StreamHandler[TextIO].emit
    original_warning = warnings.showwarning
    original_thread = threading.excepthook
    handler = logfire.LogfireLoggingHandler(level=level, logfire_instance=instance)
    excluded = ('logfire', 'opentelemetry', 'urllib3', 'requests')

    def call_handlers(self: logging.Logger, record: logging.LogRecord) -> None:
        if record.levelno >= level and not record.name.startswith(excluded) and not guard.depth:
            with guard.suppress():
                handler.handle(record)
        # Logfire already handles the record; StreamHandler's guard below prevents
        # its formatted text from being captured a second time by the console tee.
        original_handlers(self, record)

    def emit(self: logging.StreamHandler[TextIO], record: logging.LogRecord) -> None:
        with guard.suppress():
            original_emit(self, record)

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        with guard.suppress():
            instance.log(
                'warn',
                '{category}: {message}',
                {
                    'category': category.__name__,
                    'message': str(message),
                    'train.log.source': 'warnings',
                    'code.filepath': filename,
                    'code.lineno': lineno,
                },
            )
            original_warning(message, category, filename, lineno, file, line)

    def thread_exception(args: threading.ExceptHookArgs) -> None:
        with guard.suppress():
            instance.log(
                'error',
                'uncaught thread exception {thread_name}',
                {
                    'thread_name': args.thread.name if args.thread is not None else None,
                    'train.log.source': 'threading',
                },
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback) if args.exc_value is not None else False,
            )
            original_thread(args)

    def restore() -> None:
        logfire.configure = original_configure
        logging.Logger.callHandlers = original_handlers
        logging.StreamHandler.emit = original_emit
        warnings.showwarning = original_warning
        threading.excepthook = original_thread
        handler.close()

    stack.callback(restore)
    logfire.configure = _warn_configure(original_configure)
    logging.Logger.callHandlers = call_handlers
    logging.StreamHandler.emit = emit
    warnings.showwarning = showwarning
    threading.excepthook = thread_exception
