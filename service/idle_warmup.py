"""Bounded background inference for an otherwise idle, ready CPU ONNX engine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter

from anyio import to_thread

from lawcast_semantic import config

from .logging import get_service_logger

logger = get_service_logger(__name__)


def memory_pressure_high() -> bool:
    """Avoid background page touching during Linux reclaim or near a cgroup limit."""
    try:
        pressure = Path('/proc/pressure/memory').read_text().splitlines()[0].split()
        if float(dict(item.split('=') for item in pressure[1:])['avg10']) >= 1.0:
            return True
    except (OSError, ValueError, KeyError, IndexError):
        pass
    try:
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        return current >= limit * 0.9
    except (OSError, ValueError):
        return False


def warm_once(state, interval: float) -> bool:
    """Skip active searches/updates and reuse one consistent index generation."""
    if memory_pressure_high():
        return False
    searcher = state.idle_searcher(interval)
    if searcher is None:
        return False
    started = perf_counter()
    try:
        vector = searcher.embedder.warm_query()
        if vector is None:
            return False
        searcher.index.search(vector, k=1)
    except Exception as exc:  # noqa: BLE001 - optional upkeep must not change serving readiness
        logger.warning('idle warmup failed: %s: %s', type(exc).__name__, exc)
        return False
    if config.ENVIRONMENT == 'development':
        logger.info('idle warmup completed total_ms=%.1f', (perf_counter() - started) * 1000)
    return True


async def run_warmup(state, interval: float, stop: asyncio.Event) -> None:
    """Run one worker at a time; shutdown waits for an active native call."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            await to_thread.run_sync(warm_once, state, interval)
