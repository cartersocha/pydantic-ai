# SPEC: Local metrics and log collection for training runs (`train_telemetry`)

Status: ready to implement · Target: a private fork of `pydantic/pydantic-ai` · Written 2026-09-28

## 0. Instructions for the implementing agent

- Work on a branch in the fork (`training-telemetry`) and build the milestones in section 9 in order, with one commit per milestone.
- After each milestone, run `make format lint typecheck` and `uv run pytest tests/train_telemetry`, then run that milestone's acceptance commands.
- Change nothing under `pydantic_ai_slim/`, `pydantic_evals/`, `pydantic_graph/`, `clai/` or `examples/`. The only shared files you may touch are the root `pyproject.toml` (workspace member, source, pytest marker) and `uv.lock`.
- Section 4 lists assumptions a spike already verified. If one turns out false against the pinned versions, stop and report it instead of working around it silently.
- Follow the repo's conventions: Python ≥3.10, full type hints (pyright), ruff, pytest with `inline-snapshot`, and Logfire's `capfire` fixture for telemetry assertions.

## 1. Goal

Collect everything a Weights & Biases run collects automatically about a training run, **except the training data itself**. The collection has to work locally, without changing the training script. There are three parts:

1. **Hardware and process metrics**, sampled every 15 s: CPU, memory, disk, network, process tree and GPU.
2. **Run metadata**: identity, host, OS, Python, hardware inventory, git, packages and an allowlist of environment variables.
3. **Logs**: stdout/stderr, stdlib `logging`, warnings, uncaught exceptions, and the run's start, end and status.

Everything is written to **two destinations at once**: local files in a run directory, and a local Logfire instance over OTLP (a remote one works too by changing the base URL).

## 2. Non-goals

- Training data: hyperparameters, loss/accuracy history, summaries, media, tables, artifacts, gradients.
- Visualizations and dashboards.
- W&B-style telemetry about the tool itself.
- Multi-node or DDP aggregation. Each process is its own run for now; see Stretch.
- Hardware other than NVIDIA: AMD, TPU, Trainium, and Apple GPU utilization. Apple unified-memory stats via torch are in scope.
- Output that bypasses Python's `sys.stdout`/`sys.stderr`, such as C/C++ writes to fd 1/2. This is a stretch goal.
- Upstreaming. pydantic-ai is an agent framework, and its `AGENTS.md` requires maintainer alignment before non-trivial contributions. The package stays isolated so rebasing onto upstream only touches the root `pyproject.toml` and `uv.lock`.

## 3. How it's used

Zero-code runner. This is the primary interface; it runs the script in-process with `runpy`, like `coverage run`:

```bash
uv run python -m train_telemetry train_mnist.py --epochs 1
uv run python -m train_telemetry --run-name cnn-baseline --interval 15 -m mypkg.train --lr 0.1
```

Programmatic:

```python
from train_telemetry import TrainingRun

with TrainingRun(name='cnn-baseline') as run:
    train()
print(run.id, run.dir, run.status)
```

Destination selection:

| Situation | Behaviour |
|---|---|
| `LOGFIRE_TOKEN` set (plus `LOGFIRE_BASE_URL` for a local instance) | Files **and** Logfire |
| No token | Files only (offline). Nothing leaves the machine. |
| `--no-files` | Logfire only |

## 4. Verified assumptions (spike, 2026-09-28)

A throwaway spike ran the design against `logfire==4.40.0` (the version in this repo's `uv.lock`), `opentelemetry-sdk 1.44.0` and `torch 2.14.0`. It used the MNIST script with fake data and an in-process OTLP receiver standing in for Logfire. It confirmed:

| # | Assumption | Result |
|---|---|---|
| V1 | `logfire.configure(resource_attributes={'service.instance.id': run_id})` overrides Logfire's random instance ID | ✅ The value appears on every span and metric sent to both destinations |
| V2 | `additional_span_processors` and `MetricsOptions(additional_readers=...)` receive the same data Logfire sends | ✅ Identical metric names in the files and at the fake Logfire |
| V3 | `OTEL_METRIC_EXPORT_INTERVAL`, when set **before** `configure`, also sets Logfire's own reader interval | ✅ A 2 s interval produced 19 `/v1/metrics` POSTs in about 20 s |
| V4 | `AdvancedOptions(base_url=...)` routes everything to a local URL. Logfire also sends `GET {base_url}/v1/info` at startup. | ✅ |
| V5 | OTLP protobuf encoders + `MessageToJson` give valid JSON lines that parse back into OTLP requests | ✅ `encode_spans` and `encode_metrics` from `opentelemetry.exporter.otlp.proto.common` |
| V6 | Logfire's scrubbing also applies to the file exporters | ✅ Lines containing `api_key` or `password` became `[Scrubbed due to '…']` in both places. The raw `console.log` is not scrubbed. |
| V7 | Wrapping `logging.Logger.callHandlers` keeps capturing after the script calls `logging.basicConfig(..., force=True)` | ✅ |
| V8 | Wrapping `logging.StreamHandler.emit` in a thread-local flag stops log lines being sent twice (once structured, once as console text) | ✅ |
| V9 | Collapsing `\r` into the last state per line turns a progress bar into a single record | ✅ |
| V10 | A crash keeps exit code 1, marks the run span `error` with an `exception` event, and puts the traceback in `console.log`. SIGTERM exits cleanly with all files flushed. | ✅ |
| V11 | While a long span is open, Logfire re-exports it several times with the same `spanId` and a growing end time | ⚠️ Anything reading `spans.jsonl` must keep only the **last** record per `spanId` |
| V12 | Epoch time overhead on a 2-vCPU VM | ⚠️ 5–8% slower, and the same with a 600 s interval, so sampling isn't the cause. Must be measured on real hardware in M6 (target ≤2%). |
| V13 | torch warns that `pynvml` is deprecated | Depend on **`nvidia-ml-py`**, which provides the `pynvml` module |

The spike couldn't test a real GPU (the VM has none; the NVML code path was exercised with a fake NVML object) or a real MNIST download (blocked in the spike environment).

## 5. Architecture

```
train_telemetry.__main__  ──runpy──►  user script (unchanged)
        │
   TrainingRun (context manager; one active per process)
        │
        ├── _export.configure_logfire()   logfire.configure(... console=False, resource_attributes, extra processor+reader, base_url)
        │        ├── Logfire OTLP exporter ─────────► local Logfire  (/v1/traces, /v1/metrics)
        │        ├── JsonlSpanExporter (BatchSpanProcessor) ─► runs/<run>/spans.jsonl
        │        └── JsonlMetricExporter (PeriodicExportingMetricReader) ─► runs/<run>/metrics.jsonl
        ├── _metadata.collect()      → metadata.json, requirements.txt, [diff.patch], "run metadata" log
        ├── _system_metrics.install() → logfire.instrument_system_metrics(config, base=None) + custom gauges
        ├── _gpu.install()           → NVML + torch allocator gauges (optional)
        ├── _console.install()       → stdout/stderr tee → console.log (raw) + Logfire logs
        ├── _logging.install()       → Logger.callHandlers patch → LogfireLoggingHandler; warnings hook
        └── root span "training run {run_name}" wraps the script; status/exit code set on exit
```

Key decisions and why:

- **The join key is `service.instance.id` = run ID.** It is a flat, indexed column on both Logfire's `records` and `metrics` tables. Logs emitted on background threads fall outside the run span's trace, but they still carry this key.
- **Pass `console=False` to `logfire.configure`.** Logfire's console exporter writes to stdout, which the tee captures, so leaving it on would create a feedback loop.
- **Use a tee instead of `logfire.instrument_print()`.** `instrument_print` only patches `builtins.print`. The tee also catches progress bars, `sys.stderr.write` and warnings.
- **Patch `callHandlers` instead of adding a root handler.** Any root handler turns the script's `logging.basicConfig()` into a no-op, which silently leaves the root level at WARNING (V7).
- **Write the files as OTLP-JSON lines**, so a run recorded offline can be replayed into Logfire later (stretch goal S1).
- **Every patch is undone on exit.** `TrainingRun` leaves no global state behind apart from the Logfire configuration.
- **Only one Logfire configuration.** A script that calls `logfire.configure()` itself replaces the collector's processors. Warn about this; see open question Q1.

## 6. Data catalog

### 6.1 Resource attributes (on every span and metric point)

| Attribute | Source |
|---|---|
| `service.name` | `RunConfig.service_name`, default `training` |
| `service.instance.id` | Run ID (12 hex characters) |
| `train.run.name` | `--run-name`, default: script or module stem |
| `deployment.environment.name` | `RunConfig.environment`, default `local` |
| `service.version` | Logfire's default: the git commit, if inside a repo |
| `host.name`, `host.arch`, `os.type`, `os.version`, `process.pid`, `telemetry.sdk.*` | Logfire defaults (V1) |

### 6.2 Metrics

Sampled once per export interval (default 15 s, matching W&B's current default). Values are point-in-time and not averaged.

**Host and process metrics**, from `logfire.instrument_system_metrics(config, base=None)`:

| Metric | Config value |
|---|---|
| `process.cpu.utilization` | `None` (Logfire's version, which avoids a 0 on the first reading) |
| `process.memory.usage` | `None` (RSS, bytes) |
| `process.thread.count` | `None` |
| `system.cpu.simple_utilization` | `None` |
| `system.memory.usage` | `['used', 'available', 'total']` |
| `system.memory.utilization` | `['available']` |
| `system.disk.io` | `['read', 'write']` |
| `system.network.io` | `['transmit', 'receive']` |

**Custom gauges**, via `logfire.metric_*_callback`:

| Metric | Unit | Notes |
|---|---|---|
| `train.process.tree.memory.rss` | By | This process plus all children (DataLoader workers) |
| `train.process.tree.cpu.utilization` | 1 | Sum of `cpu_percent` over the tree ÷ logical CPUs |
| `train.process.tree.children` | {process} | |
| `system.filesystem.utilization` | 1 | Mount holding the run directory. Attribute `system.filesystem.mountpoint`. |
| `system.filesystem.usage` | By | Same mount; attribute `system.filesystem.state` = `used` or `free` |

**NVIDIA GPU**, via `nvidia-ml-py`. One series per device, with attributes `hw.id` (GPU UUID), `hw.name`, `hw.vendor=nvidia` and `gpu.index`. The `hw.*` names follow OTel semconv, which is still in Development status.

| Metric | Unit | NVML call |
|---|---|---|
| `hw.gpu.utilization` | 1 | `nvmlDeviceGetUtilizationRates().gpu` (`hw.gpu.task=general`); `nvmlDeviceGetEncoderUtilization` / `nvmlDeviceGetDecoderUtilization` (`encoder` / `decoder`) |
| `hw.gpu.memory.usage` (UpDownCounter callback) | By | `nvmlDeviceGetMemoryInfo().used` |
| `hw.gpu.memory.limit` (UpDownCounter callback) | By | `.total` |
| `hw.gpu.memory.utilization` | 1 | `used / total` |
| `hw.errors` (Counter callback) | {error} | `nvmlDeviceGetTotalEccErrors`, with `hw.type=gpu`, `error.type=corrected` or `uncorrected` |
| `train.gpu.memory_controller.utilization` | 1 | `UtilizationRates().memory` (W&B's `gpu.memory`) |
| `train.gpu.temperature` | Cel | `nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)` |
| `train.gpu.power.usage` / `train.gpu.power.limit` | W | `nvmlDeviceGetPowerUsage` / `nvmlDeviceGetEnforcedPowerLimit`; both return mW, so divide by 1000 |
| `train.gpu.clock.frequency` | MHz | `nvmlDeviceGetClockInfo`, attribute `clock=sm`, `memory` or `graphics` |
| `train.gpu.process.memory.usage` | By | `nvmlDeviceGetComputeRunningProcesses`, summed over this run's process tree |

A call that raises `NVMLError_NotSupported` (ECC on consumer cards, for example) is turned off for that device after the first failure. If NVML can't be initialised, GPU metrics are skipped and the reason goes into the `run metadata` log.

**torch allocator.** Only if `torch` is already imported and the backend is initialised. It must never initialise CUDA itself.

| Metric | CUDA | MPS |
|---|---|---|
| `train.torch.memory.allocated` | `torch.cuda.memory_allocated(d)` | `torch.mps.current_allocated_memory()` |
| `train.torch.memory.reserved` | `torch.cuda.memory_reserved(d)` | `torch.mps.driver_allocated_memory()` |
| `train.torch.memory.peak_allocated` | `torch.cuda.max_memory_allocated(d)` | — |

**Sampling cost.** Each metric reader calls the observable callbacks, so with two readers every callback runs twice per interval. Cache each collector's snapshot for 1 s, so NVML and psutil are queried once per tick. Callbacks run on the reader thread and must never block or raise.

### 6.3 Run metadata

Collected once at start. It is written to `metadata.json`, and all of it except the package list is also sent as the `run metadata` log's `metadata` attribute.

| Section | Fields |
|---|---|
| `run` | id, name, started_at (UTC ISO), program or module, args, cwd, executable |
| `python` | version, implementation |
| `os` | `platform.platform()` |
| `host` | hostname, user (`getpass.getuser()`; opt out with `capture_user=False`) |
| `hardware` | CPU model, physical/logical cores, memory total, disk total/used for the run dir's mount, GPUs (name, UUID, memory total, driver version, CUDA driver version, compute capability from torch if available) |
| `torch` | version, `torch.version.cuda`, cuDNN version, `cuda_available`, `mps_available`, device count. Collected only if torch is importable. |
| `git` | commit, branch, remote (credentials stripped from the URL), dirty flag. A `diff.patch` against HEAD is written only if `capture_git_diff=True` (default off). |
| `env` | Allowlist only: `SLURM_*`, `CUDA_VISIBLE_DEVICES`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `OMP_NUM_THREADS`, `PYTORCH_CUDA_ALLOC_CONF`. Never the full environment. |
| `packages` | `name==version` for every installed distribution (`importlib.metadata`). Saved to `requirements.txt`; only the count goes in the log. |

### 6.4 Logs (Logfire records, `kind='log'`)

| Record | Template / level | Key attributes |
|---|---|---|
| Run span (a span, not a log) | `training run {run_name}`; error on failure | `train.run.id`, `train.run.status` (`ok`, `failed`, `interrupted`, `terminated`), `train.run.exit_code`, `train.run.duration_s`, exception event |
| Run metadata | `run metadata`, info | `metadata`, `package_count`, `gpu_status` |
| Console line | `{line}`, info | `line`, `train.log.source=console`, `train.console.stream` = `stdout` or `stderr`, tag `console` |
| stdlib logging | Record's own level | Logfire's defaults: `logfire.logger_name`, `code.filepath`, `code.lineno`, `code.function` |
| Warning | `{category}: {message}`, warn | `train.log.source=warnings`, `code.filepath`, `code.lineno` |
| Run finished | `run finished {status}`, info or error | status, exit code, duration, file paths |

Capture rules:
- The tee writes every byte to `console.log` unchanged.
- The tee sends a line to Logfire only when a `\n` completes it, after collapsing `\r`. It skips lines produced inside `StreamHandler.emit` or `warnings.showwarning`, since those were already sent in structured form.
- A thread-local re-entrancy guard stops Logfire's own output from being captured again.
- Loggers whose names start with `logfire`, `opentelemetry`, `urllib3` or `requests` are excluded from logging capture.
- The default capture level is INFO, configurable.
- Uncaught exceptions: `SystemExit(n)` keeps exit code `n`; `KeyboardInterrupt` gives `interrupted` and exit 130; SIGTERM gives `terminated` and exit 143; any other exception gives `failed` and exit 1, with the traceback printed to stderr as Python would.
- `threading.excepthook` sends exceptions from background threads as error logs.
- `faulthandler` writes to `faulthandler.log`, to catch hard crashes.

### 6.5 Run directory

`{run_dir}/{run_name}-{YYYYmmdd-HHMMSS}-{id[:8]}/`, default `./runs/`:

| File | Contents |
|---|---|
| `run.json` | id, name, timestamps, duration, status, exit code, destinations. Written at start and rewritten at the end. |
| `metadata.json` | Section 6.3 |
| `requirements.txt` | Package list |
| `diff.patch` | Opt-in |
| `console.log` | Raw, unscrubbed stdout and stderr, interleaved |
| `spans.jsonl` | One `ExportTraceServiceRequest` per line (OTLP-JSON). Includes all logs, because Logfire logs are spans. Keep the last record per `spanId` (V11). |
| `metrics.jsonl` | One `ExportMetricsServiceRequest` per line, using the same temporality as Logfire (delta for counters and histograms) |
| `logs.jsonl` | Written only if some library emits OTel log records. Set up through `AdvancedOptions(log_record_processors=...)`. |
| `faulthandler.log` | Empty unless the process crashed hard |

## 7. Configuration

`RunConfig` is a frozen dataclass. Its values come from, in order of priority, keyword arguments, CLI flags, `TRAIN_TELEMETRY_*` environment variables, then defaults.

| Field | Default | CLI flag / environment variable |
|---|---|---|
| `run_name` | Script stem | `--run-name` / `TRAIN_TELEMETRY_RUN_NAME` |
| `run_dir` | `./runs` | `--run-dir` / `TRAIN_TELEMETRY_RUN_DIR` |
| `service_name` | `training` | `TRAIN_TELEMETRY_SERVICE_NAME` |
| `environment` | `local` | `TRAIN_TELEMETRY_ENVIRONMENT` |
| `interval_s` | `15` | `--interval` / `TRAIN_TELEMETRY_INTERVAL` |
| `files` | `True` | `--no-files` |
| `logfire` | `'if-token-present'` | `--no-logfire`; `LOGFIRE_TOKEN`, `LOGFIRE_BASE_URL` |
| `gpu` | `'auto'` | `--no-gpu` |
| `console` | `'tee'` | `--console {tee,off}` |
| `console_to_logfire` | `True` | `--no-console-to-logfire` |
| `log_level` | `INFO` | `--log-level` |
| `capture_user` | `True` | `--no-user` |
| `capture_git_diff` | `False` | `--git-diff` |
| `env_allowlist` | Section 6.3 | `TRAIN_TELEMETRY_ENV_ALLOWLIST` (comma-separated prefixes) |

## 8. Package layout in the fork

```
pyproject.toml                          # + "src/train_telemetry" in [tool.uv.workspace].members and [tool.uv.sources]
                                        # + pytest marker "train_acceptance"
src/train_telemetry/
  pyproject.toml                        # name="train-telemetry"; requires-python>=3.10
                                        # deps: logfire[system-metrics]>=4.40, psutil
                                        # extras: nvidia=["nvidia-ml-py"], example=["torch>=2.13", "torchvision>=0.28"]
  README.md
  train_telemetry/
    __init__.py        # TrainingRun, RunConfig, __version__
    __main__.py        # CLI runner (argparse; supports `script.py` and `-m module`; passes REMAINDER args through)
    _run.py            # TrainingRun lifecycle, run span, status/exit code, signals, run.json
    _config.py         # RunConfig + env/CLI resolution
    _export.py         # configure_logfire(), JsonlSpanExporter, JsonlMetricExporter, JsonlLogExporter
    _metadata.py       # collect() -> dict; git helpers with URL sanitising
    _system_metrics.py # instrument_system_metrics config + process-tree + filesystem gauges
    _gpu.py            # NVML + torch allocator gauges, 1 s snapshot cache
    _console.py        # ConsoleTee
    _logging.py        # callHandlers / StreamHandler.emit / showwarning / threading.excepthook patches
    py.typed
  examples/
    train_mnist.py     # the test workload (provided with this spec)
tests/train_telemetry/
  conftest.py          # otlp_receiver fixture (in-process http.server that decodes OTLP protobuf)
  test_config.py  test_export.py  test_metadata.py  test_system_metrics.py
  test_gpu.py  test_console.py  test_logging.py  test_run.py
  test_acceptance.py   # @pytest.mark.train_acceptance, skipped if torch isn't installed
```

## 9. Milestones

Sizes: S is about half a day, M about a day, for one person without an agent.

| # | Scope | Size | Done when |
|---|---|---|---|
| M0 | Fork, branch, workspace member, empty package, `examples/train_mnist.py`, `make install` works | S | `uv run python src/train_telemetry/examples/train_mnist.py --data fake --limit-batches 20` exits 0; lint and typecheck pass |
| M1 | `RunConfig`, `TrainingRun`, CLI runner, Logfire configuration, JSONL exporters, run span, status and exit codes, SIGTERM/SIGINT, `run.json` | M | A1 (files only), A2, A7 pass; `otlp_receiver` test shows spans and metrics arriving with `service.instance.id` = run ID |
| M2 | Metadata, `requirements.txt`, git (with sanitising), env allowlist, `diff.patch` opt-in | S | `test_metadata.py` covers a temporary git repo, a dirty tree, a remote with credentials, and no git at all |
| M3 | System metrics, process tree, filesystem | S | A1 shows every host and process metric from 6.2; A8 shows the child count > 0 and tree RSS > process RSS |
| M4 | Console tee, logging patch, warnings, thread excepthook, faulthandler | M | Tests for `\r` collapsing, braces in lines, `basicConfig(force=True)`, no duplicates, scrubbing, `isatty`/`fileno` passthrough |
| M5 | NVML GPU metrics, torch allocator, MPS | M | Unit tests with a fake `pynvml` module (every metric, NotSupported handling, no-GPU path); A3 on a real GPU if one is available |
| M6 | Acceptance suite, README, overhead benchmark | S | A1–A8 pass; overhead ≤2% of step time on your machine, or the cause is profiled and written down |

## 10. Acceptance tests

Run these from the repo root. `W=src/train_telemetry/examples/train_mnist.py`

| ID | Command | Expected |
|---|---|---|
| A1 | `uv run python -m train_telemetry $W --data fake --epochs 1 --limit-batches 50` | Exit 0. The run dir has every file from 6.5. `spans.jsonl` has the run span with `train.run.status=ok`; console records for the `epoch … loss` lines; logging records from `train_mnist` with `logfire.logger_name`; one warning record, with no duplicates of any of these. `metrics.jsonl` holds all the host and process metric names from 6.2. |
| A2 | Same as A1, plus `--crash-at-step 10` | Exit 1; status `failed`; run span level error with an `exception` event; traceback in `console.log` |
| A3 | `uv run python -m train_telemetry $W --epochs 1` on a CUDA machine | `hw.gpu.utilization` > 0 at some point; `train.gpu.process.memory.usage` > 0; `train.torch.memory.allocated` > 0 |
| A4 | A1 with `LOGFIRE_TOKEN` and `LOGFIRE_BASE_URL` pointing at local Logfire | The SQL in section 11 returns the run, its logs and its metrics |
| A5 | Bare run of `$W --data fake --epochs 1 --limit-batches 300` vs. instrumented, 3 times each | Median in-loop epoch time overhead ≤2% |
| A6 | A1 with `LOGFIRE_BASE_URL` pointing at a closed port | Training unaffected, exit 0, files complete, no traceback in `console.log` from the exporter |
| A7 | Start A1 with `--epochs 3`, then send SIGTERM after 10 s | Exit 143; status `terminated`; all files flushed; `run.json` has `ended_at` |
| A8 | A1 with `--num-workers 2` | `train.process.tree.children` ≥ 2 while training |

Also run A1 once with real MNIST (`--data mnist`), which downloads about 12 MB.

## 11. Checking a run in Logfire (A4)

Replace `<RUN_ID>` with the ID printed at the end of the run and shown in `run.json`. All four queries were run against a Logfire project with a dummy run ID to confirm they parse and execute.

```sql
-- The run and its status
SELECT start_timestamp, duration, message, attributes->>'train.run.status' AS status
FROM records
WHERE service_instance_id = '<RUN_ID>' AND span_name = 'training run {run_name}'
  AND start_timestamp >= now() - interval '1 day'
LIMIT 5;

-- Its logs, in order
SELECT start_timestamp, level, attributes->>'train.log.source' AS source, message
FROM records
WHERE service_instance_id = '<RUN_ID>' AND kind = 'log'
  AND start_timestamp >= now() - interval '1 day'
ORDER BY start_timestamp
LIMIT 200;

-- Which metrics arrived
SELECT metric_name, count(*) AS points, max(recorded_timestamp) AS last_seen
FROM metrics
WHERE service_instance_id = '<RUN_ID>' AND recorded_timestamp >= now() - interval '1 day'
GROUP BY metric_name
ORDER BY metric_name
LIMIT 100;

-- GPU utilization per device, per minute
SELECT time_bucket('1 minute', recorded_timestamp) AS t, attributes->>'hw.id' AS gpu,
       avg(metric_avg(value)) AS utilization
FROM metrics
WHERE metric_name = 'hw.gpu.utilization' AND service_instance_id = '<RUN_ID>'
  AND recorded_timestamp >= now() - interval '1 day'
GROUP BY time_bucket('1 minute', recorded_timestamp), attributes->>'hw.id'
ORDER BY t
LIMIT 500;
```

## 12. Unit test notes

- Use Logfire's `capfire` fixture (`logfire.testing.CaptureLogfire`) for span and metric assertions, as the rest of this repo does. For the dual-export path, add an `otlp_receiver` fixture: a `ThreadingHTTPServer` on 127.0.0.1 that gunzips requests and decodes them with `ExportTraceServiceRequest` / `ExportMetricsServiceRequest`, and answers `200 {}` to every route, including `GET /v1/info`.
- Fake NVML: a module-like `SimpleNamespace` with the calls listed in 6.2, monkeypatched in as `sys.modules['pynvml']`. Include one device that raises `NVMLError_NotSupported` for ECC.
- Console tests write through a `ConsoleTee` that wraps `io.StringIO`.
- Every test must restore patched globals. Assert this with a fixture that snapshots `sys.stdout`, `sys.stderr`, `Logger.callHandlers`, `StreamHandler.emit` and `warnings.showwarning`.
- Timing-based tests use a 0.2 s interval and `logfire.force_flush()`, never sleeps longer than 1 s.

## 13. Risks and open questions

| ID | Item | Proposed handling |
|---|---|---|
| R1 | Overhead of 5–8% on a small VM that didn't depend on the sampling interval (V12) | Measure in M6 on real hardware. If it's over 2%, profile the background threads (Logfire's span re-export, batch processors) before tuning. |
| R2 | `hw.*` semconv is Development status and may be renamed | Keep all names in one `_names.py` so a rename is one edit |
| R3 | Scrubbing replaces whole console lines that contain words like `token`, `key` or `password` (V6), which can hide useful lines | Document it; expose `ScrubbingOptions(callback=...)` pass-through; `console.log` stays raw |
| R4 | Writes to `sys.stdout.buffer` and to fd 1/2 bypass the tee | Accepted for v1; stretch goal S2 |
| Q1 | The script calls `logfire.configure()` itself | v1 warns and continues. Option: accept `TrainingRun(logfire_instance=...)` and use `configure(local=True)`. |
| Q2 | Local Logfire URL and port | Supplied by the user through `LOGFIRE_BASE_URL`. Needs a write token from that instance. |
| Q3 | Should `capture_user` default to off before anything goes to a remote Logfire? | Default on for local; revisit if remote becomes the norm |

## 14. Stretch

- **S1**: `python -m train_telemetry replay runs/<run>`, which re-sends `spans.jsonl` and `metrics.jsonl` to an OTLP endpoint. Parse each line with `json_format.Parse` and POST it as protobuf. This covers runs recorded offline.
- **S2**: fd-level capture (`os.pipe` + `dup2` + a reader thread) to pick up C, CUDA and NCCL output.
- **S3**: AMD GPUs via `amdsmi`.
- **S4**: DDP. Share one run ID across ranks (rank 0 creates it and passes it through an environment variable), add a `train.rank` resource attribute, and use one run directory per rank.

## Appendix: W&B parity

| W&B automatic collection | Here |
|---|---|
| System metrics every 15 s (CPU, memory, disk, network, NVIDIA GPU) | ✅ 6.2 |
| Process-tree stats (`x_stats_track_process_tree`) | ✅ Always on |
| AMD, Apple GPU utilization, TPU, Trainium, IPU | ❌ Non-goals (Apple memory via torch ✅) |
| `wandb-metadata.json` (OS, Python, host, user, program, args, git, hardware, CUDA, Slurm) | ✅ 6.3 (email ❌) |
| `requirements.txt` | ✅ |
| `diff.patch` | ✅ Opt-in |
| Console capture (`output.log`) | ✅ `console.log` and Logfire records (Python-level only) |
| SDK usage telemetry and Sentry about the tool | ❌ Deliberately not collected |
