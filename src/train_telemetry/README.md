# Training telemetry

Collect run metadata, system metrics and Python logs without modifying your training script.
Data is written locally and, when `LOGFIRE_TOKEN` is set, sent to Logfire too.
No token means files only, even if this machine has saved Logfire credentials.

## Install and run

From this workspace:

```bash
uv sync --all-packages
uv run python -m train_telemetry path/to/train.py --epochs 1
uv run python -m train_telemetry --run-name cnn-baseline --interval 15 -m mypkg.train --lr 0.1
```

Runner options go **before** the script/module. Everything after the target is passed
through unchanged. Scripts execute in-process with `runpy`, with their directory on
`sys.path`; modules behave like `python -m`. The runner restores `sys.argv` and `sys.path`.

The example workload needs torch and torchvision:

```bash
uv sync --all-packages --extra example
uv run python -m train_telemetry src/train_telemetry/examples/train_mnist.py --data fake --epochs 1 --limit-batches 50
```

For NVIDIA collection, install the `nvidia` extra (`nvidia-ml-py`, which provides
`pynvml`). Avoid the deprecated distribution named `pynvml`:

```bash
uv sync --all-packages --extra example --extra nvidia
```

The package can also be installed independently with `pip install ./src/train_telemetry`
(or `pip install './src/train_telemetry[nvidia]'`). It does not depend on Pydantic AI.

## Programmatic use

```python
from pathlib import Path
from train_telemetry import RunConfig, TrainingRun

with TrainingRun(name='cnn-baseline', run_dir=Path('runs')) as run:
    train()
print(run.id, run.dir, run.status)
```

`TrainingRun(config, name=..., **options)` accepts the fields of `RunConfig` as keyword
overrides. `name` aliases `run_name` and wins if both are given. `RunConfig` is frozen
and accepts only keyword arguments. Explicit values override environment defaults.
One run may be active per process, and each context instance is single use.

Entry configures process-wide Logfire; exit flushes and shuts down the configured
providers. Start the context before configuring system metrics yourself. Background
threads share the run's `service.instance.id` even when they are outside its trace.

## Destinations and files

Use `LOGFIRE_TOKEN` and `LOGFIRE_BASE_URL` for a local Logfire instance. The URL is the
instance's base URL; Logfire sends traces, metrics, logs and its startup info request
under `/v1/`. A token without a base URL uses Logfire's configured default endpoint.
`--no-logfire` forces offline mode. `--no-files` requires a token and enables Logfire only.

Runs are stored under `runs/{name}-{UTC timestamp}-{id[:8]}/`. Names are sanitized for
filesystem safety; the original name stays in metadata.

| File | Contents |
| --- | --- |
| `run.json` | Identity, timestamps, status, duration, exit code and destinations; updated atomically on exit |
| `metadata.json` | Runtime, hardware, git, allowlisted environment and installed packages |
| `requirements.txt` | Sorted installed distributions as `name==version` |
| `diff.patch` | Tracked changes against HEAD, only with `--git-diff` |
| `console.log` | Raw stdout/stderr text, interleaved; absent with `--console off` |
| `spans.jsonl` | OTLP trace requests, including Logfire logs and exception events |
| `metrics.jsonl` | OTLP metric requests, exported periodically and at shutdown |
| `logs.jsonl` | Created only when a library emits native OTel log records |
| `faulthandler.log` | Hard-crash diagnostics when this run enables faulthandler |

Each JSONL line parses into its corresponding `Export*ServiceRequest` protobuf.
**Keep the last record for each `(traceId, spanId)`** when reading spans: Logfire can
export an open span more than once. Counter and histogram temporality matches
Logfire (delta); up/down counters use cumulative temporality.

Logfire scrubbing applies to both remote export and OTLP files. A line containing
sensitive names such as `password` may be replaced entirely. **`console.log` is raw
and unscrubbed.** `capture_git_diff=True` also writes a raw patch. Choose capture
settings appropriate for the files you intend to retain or share. Programmatic
`scrubbing=logfire.ScrubbingOptions(...)` customizes scrubbing.

## Configuration

| `RunConfig` field | Default | CLI / environment |
| --- | --- | --- |
| `run_name` | Script/module stem in CLI; `training` programmatically | `--run-name`, `TRAIN_TELEMETRY_RUN_NAME` |
| `run_dir` | `Path('./runs')` | `--run-dir`, `TRAIN_TELEMETRY_RUN_DIR` |
| `service_name` | `training` | `TRAIN_TELEMETRY_SERVICE_NAME` |
| `environment` | `local` | `TRAIN_TELEMETRY_ENVIRONMENT` |
| `interval_s` | 15 | `--interval`, `TRAIN_TELEMETRY_INTERVAL` |
| `files` | `True` | `--no-files` |
| `logfire` | `if-token-present` | `--no-logfire`; set `False` to disable programmatically |
| `gpu` | `auto` | `--no-gpu`; set `False` to disable all GPU/allocator collection |
| `console` | `tee` | `--console {tee,off}` |
| `console_to_logfire` | `True` | `--no-console-to-logfire` |
| `log_level` | `logging.INFO` | `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` |
| `capture_user` | `True` | `--no-user` |
| `capture_git_diff` | `False` | `--git-diff` |
| `env_allowlist` | List below | `TRAIN_TELEMETRY_ENV_ALLOWLIST`, comma separated |
| `scrubbing` | Logfire defaults | Programmatic `ScrubbingOptions` |

The environment allowlist contains `SLURM_*`, `CUDA_VISIBLE_DEVICES`, `WORLD_SIZE`,
`RANK`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `OMP_NUM_THREADS` and
`PYTORCH_CUDA_ALLOC_CONF`. Entries ending in `_` match prefixes; others match exact
names. No full environment is captured. Git remotes have user credentials, query
strings and fragments removed. `capture_user=False` omits the username. Packages
are saved locally; the metadata log includes their count only.

## Captured metrics and logs

The host/process instruments use Logfire's system metrics integration: CPU
utilization, process RSS/thread count, host memory, disk IO and network IO. Custom
gauges add process-tree RSS/CPU/child count and filesystem usage/utilization for the
run directory's mount. Process objects are retained for CPU deltas, and custom
collector snapshots are shared for one second across readers.

NVIDIA metrics cover general/encoder/decoder utilization, memory usage/limit,
memory-controller utilization, temperature, power/limit, clocks, ECC errors and
GPU memory used by this process and its children. Unsupported per-device queries
are disabled after the first failure; transient errors are retried on later samples.
GPU series use `hw.id`, `hw.name`, `hw.vendor` and `gpu.index`. Experimental hardware
metric names are centralized in `_names.py`.

Allocator gauges inspect already imported torch. CUDA allocation/reservation/peak
queries require `torch.cuda.is_initialized()`; collection never initializes CUDA.
Available MPS backends expose allocated and driver memory (there is no public MPS
initialization-status API). Runtime metadata imports torch if installed. No model
parameters, training configuration, datasets, losses or artifacts are inspected.
Text that your script prints, including its loss/accuracy output, is captured as logs.

Console lines are emitted only at a newline, with carriage-return progress updates
collapsed to their last state. Partial lines remain in the raw file. Stdlib logging
and warnings are structured once; their formatted stream output stays in the raw
file but is not emitted again. Capture survives `logging.basicConfig(force=True)`
and leaves logger levels and handlers under the application's control. Loggers
starting with `logfire`, `opentelemetry`, `urllib3` or `requests` are excluded.

The runner preserves `SystemExit` codes; other exceptions exit 1, SIGINT/
`KeyboardInterrupt` exit 130 and SIGTERM exits 143. Run spans contain status,
exit code, duration and exception events. Background thread exceptions are error
logs and retain Python's existing exception hook behavior.

## Boundaries

- Calling `logfire.configure()` inside the script emits a warning and then replaces
  the collector's destinations. Configure through `TrainingRun` for complete files.
- Streams and logging/warning/thread/signal hooks are restored on exit. Logfire's
  process-wide configuration is left in place, shut down. Existing faulthandler
  configuration is preserved; its destination cannot be recovered through Python's
  public API, so the collector only enables it when it was previously disabled.
- Binary `sys.stdout.buffer` writes and native fd 1/2 writes bypass capture.
- Each process is an independent run. No DDP aggregation, replay, AMD collection,
  or Apple GPU utilization is included. Apple allocator memory is supported.
- SIGKILL and native fatal crashes cannot execute cleanup or update `run.json`.
- Sampling can miss very short GPU allocations and children that exit between ticks.
- Missing NVIDIA drivers or failed remote exporters do not fail training. File export
  remains independent; remote shutdown may wait for Logfire's bounded flush timeout.

## Verification and performance

```bash
uv run pytest tests/train_telemetry
uv run coverage run --rcfile=src/train_telemetry/pyproject.toml -m pytest tests/train_telemetry
uv run coverage report --rcfile=src/train_telemetry/pyproject.toml
uv run python src/train_telemetry/examples/benchmark.py --output /tmp/benchmark.json
```

Tests include a localhost OTLP receiver, protobuf decoding, scrubbing, metadata
sanitizing, process children, capture/restoration, MNIST success/crash, SIGINT/SIGTERM,
remote-only mode, exporter failure and fake NVIDIA/allocator backends. Install the
`example` extra to enable MNIST tests. Real NVIDIA hardware and a running Logfire
service require separate acceptance checks.

See [VALIDATION.md](VALIDATION.md) for measured performance and acceptance results.
