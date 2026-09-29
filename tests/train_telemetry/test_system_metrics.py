from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import psutil
from opentelemetry.metrics import CallbackOptions

from train_telemetry import TrainingRun
from train_telemetry._system_metrics import ProcessTree

SYSTEM_METRICS = {
    'process.cpu.utilization',
    'process.memory.usage',
    'process.thread.count',
    'system.cpu.simple_utilization',
    'system.memory.usage',
    'system.memory.utilization',
    'system.disk.io',
    'system.network.io',
    'train.process.tree.memory.rss',
    'train.process.tree.cpu.utilization',
    'train.process.tree.children',
    'system.filesystem.utilization',
    'system.filesystem.usage',
}


def test_metric_names(tmp_path: Path) -> None:
    with TrainingRun(run_dir=tmp_path, gpu=False) as run:
        pass
    names = {
        metric['name']
        for line in (run.dir / 'metrics.jsonl').read_text().splitlines()
        for resource in json.loads(line)['resourceMetrics']
        for scope in resource['scopeMetrics']
        for metric in scope['metrics']
    }
    assert SYSTEM_METRICS <= names


def test_process_tree() -> None:
    # Process-tree collection depends on an actual OS child boundary.
    child = subprocess.Popen(
        [sys.executable, '-c', 'import sys; print("ready", flush=True); sys.stdin.read()'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == b'ready\n'
        tree = ProcessTree()
        children = list(tree.callback('train.process.tree.children')(CallbackOptions()))
        rss = list(tree.callback('train.process.tree.memory.rss')(CallbackOptions()))
        assert children[0].value >= 1
        assert child.pid in tree.pids()
        assert rss[0].value > psutil.Process().memory_info().rss
    finally:
        child.communicate(timeout=5)
