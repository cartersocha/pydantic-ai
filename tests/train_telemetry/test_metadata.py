from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from train_telemetry import TrainingRun
from train_telemetry._metadata import sanitize_remote


def test_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    for command in (
        ['init'],
        ['config', 'user.name', 'Test'],
        ['config', 'user.email', 'test@example.invalid'],
        ['remote', 'add', 'origin', 'https://user:secret@example.invalid/repo?token=secret'],
    ):
        subprocess.run(['git', *command], cwd=repo, check=True, capture_output=True)
    (repo / 'tracked').write_text('before\n')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-m', 'initial'], cwd=repo, check=True, capture_output=True)
    (repo / 'tracked').write_text('after\n')
    monkeypatch.chdir(repo)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES_SECRET', 'never')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'never')
    with TrainingRun(run_dir=tmp_path / 'runs', capture_git_diff=True, capture_user=False, gpu=False) as run:
        pass
    text = (run.dir / 'metadata.json').read_text()
    metadata = json.loads(text)
    assert metadata['git']['remote'] == 'https://example.invalid/repo'
    assert metadata['git']['dirty'] is True
    assert 'user' not in metadata['host']
    assert metadata['env'] == snapshot({'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '0'})
    assert 'never' not in text
    assert '-before\n+after' in (run.dir / 'diff.patch').read_text()
    assert (run.dir / 'requirements.txt').read_text().splitlines() == metadata['packages']


def test_no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with TrainingRun(run_dir=tmp_path / 'runs', gpu=False) as run:
        pass
    assert json.loads((run.dir / 'metadata.json').read_text())['git'] is None
    assert not (run.dir / 'diff.patch').exists()


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        ('git@example.invalid:org/repo.git', 'example.invalid:org/repo.git'),
        ('ssh://git@example.invalid:22/org/repo.git', 'ssh://example.invalid:22/org/repo.git'),
        ('https://user:pass@example.invalid/repo?token=x#secret', 'https://example.invalid/repo'),
    ],
)
def test_sanitize(url: str, expected: str) -> None:
    assert sanitize_remote(url) == expected
