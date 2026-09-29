"""GPU instrument names, units and kinds (hardware semconv is experimental)."""

from typing import Literal

GPU_METRICS: tuple[tuple[str, str, Literal['gauge', 'updown', 'counter']], ...] = (
    ('hw.gpu.utilization', '1', 'gauge'),
    ('hw.gpu.memory.usage', 'By', 'updown'),
    ('hw.gpu.memory.limit', 'By', 'updown'),
    ('hw.gpu.memory.utilization', '1', 'gauge'),
    ('hw.errors', '{error}', 'counter'),
    ('train.gpu.memory_controller.utilization', '1', 'gauge'),
    ('train.gpu.temperature', 'Cel', 'gauge'),
    ('train.gpu.power.usage', 'W', 'gauge'),
    ('train.gpu.power.limit', 'W', 'gauge'),
    ('train.gpu.clock.frequency', 'MHz', 'gauge'),
    ('train.gpu.process.memory.usage', 'By', 'gauge'),
    ('train.torch.memory.allocated', 'By', 'gauge'),
    ('train.torch.memory.reserved', 'By', 'gauge'),
    ('train.torch.memory.peak_allocated', 'By', 'gauge'),
)
