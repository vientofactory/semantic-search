"""Isolated, fixed-corpus BGE20 comparison; never imported by the HTTP service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

TEACHER = 'dragonkue/BGE-m3-ko'
REVISION = '7074d66aa46562342193ca4feb3d89bf9dad71b4'
DROPPED_LAYERS = frozenset({4, 8, 12, 16})


def layer_positions(count: int) -> tuple[int, ...]:
    if count != 24:
        raise ValueError(f'shadow expects 24 teacher layers, got {count}')
    return tuple(i for i in range(count) if i not in DROPPED_LAYERS)


def validate_checkpoint(meta: dict) -> None:
    expected = {
        'student': 'bge20',
        'teacher': TEACHER,
        'teacher_revision': REVISION,
        'input_dim': 1024,
        'output': 1024,
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('shadow checkpoint does not match the pinned teacher and projection')


@dataclass
class Timing:
    encode_ms: float
    index_ms: float
    serialize_ms: float
    total_ms: float


def measure(encode: Callable, rank: Callable, serialize: Callable) -> tuple[Timing, object]:
    start = perf_counter()
    vector = encode()
    encoded = perf_counter()
    hits = rank(vector)
    ranked = perf_counter()
    payload = serialize(hits)
    finished = perf_counter()
    return (
        Timing(
            (encoded - start) * 1000,
            (ranked - encoded) * 1000,
            (finished - ranked) * 1000,
            (finished - start) * 1000,
        ),
        payload,
    )
