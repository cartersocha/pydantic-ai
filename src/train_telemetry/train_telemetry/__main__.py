"""Run a script or module unchanged under telemetry collection."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import traceback
from pathlib import Path

from ._config import RunConfig
from ._run import TrainingRun


def main(argv: list[str] | None = None) -> int:
    """Run the target in-process and return its exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--interval', type=float)
    parser.add_argument('--no-files', action='store_true')
    parser.add_argument('--no-logfire', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--console', choices=['tee', 'off'], default='tee')
    parser.add_argument('--no-console-to-logfire', action='store_true')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--no-user', action='store_true')
    parser.add_argument('--git-diff', action='store_true')
    parser.add_argument('-m', dest='module', action='store_true', help='Interpret target as a module name')
    parser.add_argument('target')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    try:
        config = RunConfig(
            run_name=parsed.run_name or os.environ.get('TRAIN_TELEMETRY_RUN_NAME') or Path(parsed.target).stem,
            run_dir=parsed.run_dir
            if parsed.run_dir is not None
            else Path(os.environ.get('TRAIN_TELEMETRY_RUN_DIR', './runs')),
            interval_s=parsed.interval
            if parsed.interval is not None
            else float(os.environ.get('TRAIN_TELEMETRY_INTERVAL', '15')),
            files=not parsed.no_files,
            logfire=False if parsed.no_logfire else 'if-token-present',
            gpu=False if parsed.no_gpu else 'auto',
            console=parsed.console,
            console_to_logfire=not parsed.no_console_to_logfire,
            log_level={'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}[parsed.log_level],
            capture_user=not parsed.no_user,
            capture_git_diff=parsed.git_diff,
        )
    except ValueError as exc:
        parser.error(str(exc))
    run = TrainingRun(config)
    run.program = None if parsed.module else str(Path(parsed.target).resolve())
    run.module = parsed.target if parsed.module else None
    run.args = parsed.args
    old_argv, old_path = sys.argv, sys.path[:]
    try:
        with run:
            sys.argv = [parsed.target, *parsed.args]
            try:
                if parsed.module:
                    runpy.run_module(parsed.target, run_name='__main__', alter_sys=True)
                else:
                    sys.path.insert(0, str(Path(parsed.target).resolve().parent))
                    runpy.run_path(parsed.target, run_name='__main__')
            except SystemExit as exc:
                if exc.code is not None and not isinstance(exc.code, int):
                    print(exc.code, file=sys.stderr)
                raise
            except KeyboardInterrupt:
                raise
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        if run.status == 'running':
            raise
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    print(f'Training run {run.id}: {run.status}' + (f' — {run.dir}' if config.files else ''))
    return run.exit_code


def cli() -> None:
    """Console entry point."""
    raise SystemExit(main())


if __name__ == '__main__':
    cli()
