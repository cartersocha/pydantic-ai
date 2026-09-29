"""Compare median epoch time of bare and instrumented MNIST runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

EPOCH_TIME = re.compile(r'Epoch 1 done in ([0-9.]+)s')


def main() -> None:
    """Alternate bare and collected runs and save the measured overhead."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, default=300)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batches <= 0 or args.repetitions <= 0:
        parser.error('Batch count and repetitions must be positive')
    workload = Path(__file__).with_name('train_mnist.py')
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('COVERAGE_') and key not in ('LOGFIRE_TOKEN', 'LOGFIRE_BASE_URL')
    }
    measurements: dict[str, list[float]] = {'bare': [], 'instrumented': []}
    with tempfile.TemporaryDirectory(prefix='train-telemetry-benchmark-') as directory:
        for repetition in range(args.repetitions):
            for mode in ('bare', 'instrumented') if repetition % 2 == 0 else ('instrumented', 'bare'):
                command = [sys.executable]
                if mode == 'instrumented':
                    command += ['-m', 'train_telemetry', '--run-dir', directory]
                command += [
                    str(workload),
                    '--data',
                    'fake',
                    '--epochs',
                    '1',
                    '--limit-batches',
                    str(args.batches),
                    '--device',
                    args.device,
                ]
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                match = EPOCH_TIME.search(result.stderr + result.stdout)
                if match is None:
                    raise RuntimeError('Workload did not report epoch time')
                seconds = float(match[1])
                measurements[mode].append(seconds)
                print(f'{mode} repetition {repetition + 1}: {seconds}s', flush=True)
    bare, collected = (statistics.median(measurements[mode]) for mode in ('bare', 'instrumented'))
    report = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'device': args.device,
        'batches': args.batches,
        'repetitions': args.repetitions,
        'epoch_seconds': measurements,
        'median_bare_s': bare,
        'median_instrumented_s': collected,
        'overhead_percent': (collected / bare - 1) * 100,
        'timing_resolution_s': 0.1,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
