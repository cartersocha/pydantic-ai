# Validation

Measured on 2026-09-28, Apple M4, macOS 26.5.2, Python 3.11.0,
Logfire 4.40.0, OpenTelemetry SDK 1.44.0, torch 2.13.0 and torchvision 0.28.0.
The workspace pins torch 2.13.0; the attached spike used 2.14.0.

## Acceptance results

| Check | Result |
| --- | --- |
| Fake MNIST, 50 batches, default settings | Passed on MPS; exit 0 and complete files |
| MNIST success and simulated crash | Automated CLI tests pass; failure exits 1, with an error span, exception event and raw traceback |
| Real MNIST, 50 batches, 0.2 s sampling | Downloaded and trained successfully; exit 0, no callback tracebacks after the startup-race fix |
| Apple allocator metrics | Positive samples: allocated 25,863,424 B; driver allocated 124,551,168 B |
| Dual export | Localhost receiver decodes trace and metric protobufs with the same run ID as the files |
| Native OTel logs | Lazy `logs.jsonl`, valid protobuf round trip, scrubbing applied |
| SIGTERM and SIGINT | Real subprocess tests exit 143/130, update `run.json`, restore hooks and flush files |
| Unreachable Logfire endpoint | Exit 0, complete files, no exporter traceback in the raw console log |
| DataLoader with two workers | Real subprocess test captures at least two children; separate process-tree test verifies RSS greater than parent RSS |
| NVIDIA GPU | All metrics, unsupported queries, device loss and shutdown tested with a fake backend; real NVIDIA hardware unavailable on this Apple machine |
| Running Logfire service and SQL queries | Not attempted: no instance URL or write token supplied. Localhost OTLP transport is tested |
| Repository lint and full typecheck | Passed |
| Automated suite | 73 passed; one deliberate captured-warning test produces a warning summary |
| Package coverage | 100% statements and branches: 811 statements, 176 branches; protocol declarations and the interpreter entry guard excluded |

## Overhead: target not met on the fake workload

Three alternating bare and instrumented runs, 300 fake-data training batches each,
using the unchanged workload's default automatic device selection (MPS):

| Mode | Epoch times (seconds) | Median |
| --- | --- | --- |
| Bare script | 5.8, 5.8, 6.0 | 5.8 |
| Collected, default 15 s interval | 7.1, 7.1, 7.1 | 7.1 |
| Minimal `runpy` runner, no collector | 6.3, 6.2, 6.2 | 6.2 |
| Collected, 600 s interval | 7.1, 6.9, 6.8 | 6.9 |

The initial default-interval measurement is **22.4% overhead**, exceeding the spec's
2% target. The 600 s result is 19.0% over the original bare median. These are epoch
times, including evaluation, reported by the provided workload to 0.1 s resolution;
they exclude collector startup and shutdown. The workload is deliberately synthetic
and is not evidence of a 22% overhead on real training datasets.

### Profiling evidence

A separate `cProfile` run profiled the foreground and each Python background thread:

- The foreground recorded 29,265 `torch.manual_seed` calls and 87,798
  `traceback.format_stack` calls. `torchvision.datasets.FakeData` resets the RNG for
  each sample; uninitialized CUDA/XPU/MTIA backends save a formatted Python stack
  for each seed operation. Each extra `runpy`/CLI frame increases that repeated work.
- In the collected profile, stack formatting totaled about 11.6 s cumulatively;
  `torch.manual_seed` totaled 12.0 s of a 15.7 s profiled epoch. A minimal `runpy`
  profile totaled 11.5 s in `manual_seed` and a 15.3 s epoch. Profiling changes timings,
  so these are diagnostic profiles, not replacement benchmark measurements.
- Collector export/metric threads did about 13 ms of measured work combined;
  their other time was spent waiting. A 600 s interval did not remove the overhead.

The evidence identifies repeated foreground stack extraction as the main cost for
this workload. The runner retains `runpy` semantics and does not patch torch's RNG
or change the supplied training behavior. The spec allows M6 completion when an
above-target cause is profiled and documented; this document records that result.

Reproduce the primary benchmark:

```bash
uv sync --all-packages --extra example --extra nvidia
uv run python src/train_telemetry/examples/benchmark.py --batches 300 --repetitions 3 --output /tmp/benchmark.json
```

The measured JSON is in [benchmark-results.json](benchmark-results.json).

## Corrections to spike assumptions

- `/v1/info` must return `project_name` and `project_url`, rather than `{}`. The
  latter raises `KeyError` in Logfire's background token-validation thread.
- GPU callbacks must skip torch while its loader is initializing. This race was
  reproduced with 0.2 s sampling and fixed with deferred registration and an import
  readiness guard.
- `nvidia-ml-py` converts unavailable process-memory values to `None`; it does not
  expose a plain `NVML_VALUE_NOT_AVAILABLE` constant. The collector uses its actual
  Python API and a typed optional memory value.
- MPS has no public `is_initialized` function. Available-backend memory queries
  are used; CUDA remains explicitly guarded by `is_initialized()`.
- The package must be in the development dependency group for the root
  `uv run python -m train_telemetry` command. Workspace membership and a source
  declaration alone do not install it with the root project.
- Ruff/Pyright discovery paths must include the new package. These and the pytest
  marker live in the existing root configuration.
