"""Idle upkeep must exercise inference without changing the serving contract."""

import asyncio
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from service.app import EngineState
from service.idle_warmup import memory_pressure_high, run_warmup, warm_once


def ready_state():
    vector = np.array([1.0, 0.0], dtype='float32')
    searcher = SimpleNamespace(
        embedder=SimpleNamespace(warm_query=Mock(return_value=vector)),
        index=SimpleNamespace(search=Mock()),
        chunk_ids=['one'],
        index_updated_at=None,
    )
    state = EngineState()
    state.record_device('cpu')
    state.mark_ready(searcher, 'model')
    state.last_search_finished = 0
    return state, searcher


@pytest.fixture(autouse=True)
def no_host_pressure(monkeypatch):
    monkeypatch.setattr('service.idle_warmup.memory_pressure_high', lambda: False)


@pytest.mark.parametrize('reason', ['loading', 'failed', 'updating', 'active', 'recent', 'mps'])
def test_warmup_skips_busy_unready_and_recent_searches(reason):
    state, searcher = ready_state()
    if reason in {'loading', 'failed'}:
        state.status = reason
    elif reason == 'updating':
        state.updating = True
    elif reason == 'active':
        state.search_started()
    elif reason == 'recent':
        state.search_started()
        state.search_finished()
    else:
        state.device = 'mps'
    assert not warm_once(state, 30)
    searcher.embedder.warm_query.assert_not_called()
    searcher.index.search.assert_not_called()


def test_warmup_touches_one_generation_without_mutating_readiness():
    state, searcher = ready_state()
    before = state.snapshot()
    assert warm_once(state, 30)
    searcher.embedder.warm_query.assert_called_once_with()
    searcher.index.search.assert_called_once_with(searcher.embedder.warm_query.return_value, k=1)
    assert state.snapshot() == before


def test_pressure_skips_background_inference(monkeypatch):
    state, searcher = ready_state()
    monkeypatch.setattr('service.idle_warmup.memory_pressure_high', lambda: True)
    assert not warm_once(state, 30)
    searcher.embedder.warm_query.assert_not_called()


def test_cached_traffic_does_not_starve_idle_model_upkeep():
    state, searcher = ready_state()
    searcher.embedder.last_query_inference_at = 0
    state.search_started()
    assert not warm_once(state, 30)
    state.search_finished()
    assert warm_once(state, 30)
    searcher.embedder.warm_query.assert_called_once()


def test_recent_native_inference_suppresses_unnecessary_upkeep():
    from time import monotonic

    state, searcher = ready_state()
    searcher.embedder.last_query_inference_at = monotonic()
    assert not warm_once(state, 30)
    searcher.embedder.warm_query.assert_not_called()


@pytest.mark.parametrize('failed', [False, True])
def test_missing_onnx_and_failures_never_change_serving_state(failed):
    state, searcher = ready_state()
    if failed:
        searcher.embedder.warm_query.side_effect = RuntimeError('probe failure')
    else:
        searcher.embedder.warm_query.return_value = None
    assert not warm_once(state, 30)
    assert state.status == 'ready'
    searcher.index.search.assert_not_called()


def test_warmup_does_not_hold_the_state_lock_during_inference():
    state, searcher = ready_state()
    searcher.embedder.warm_query.side_effect = lambda: state.snapshot() and np.ones(2)
    assert warm_once(state, 30)


@pytest.mark.parametrize('failed', [False, True])
def test_request_activity_is_released_after_success_and_failure(monkeypatch, failed):
    from service import app as app_module

    state, _ = ready_state()
    monkeypatch.setattr(app_module, 'STATE', state)

    async def work(*_):
        assert state.active_searches == 1
        assert state.idle_searcher(0) is None
        if failed:
            raise RuntimeError('search failure')
        return 'reply'

    monkeypatch.setattr(app_module, '_search_with_admission', work)
    if failed:
        with pytest.raises(RuntimeError, match='search failure'):
            asyncio.run(app_module.search_with_admission(object(), 'query', 1))
    else:
        assert asyncio.run(app_module.search_with_admission(object(), 'query', 1)) == 'reply'
    assert state.active_searches == 0
    assert state.idle_searcher(30) is None


def test_linux_pressure_and_cgroup_limit_guards(monkeypatch):
    def read(path):
        if str(path) == '/proc/pressure/memory':
            return 'some avg10=1.50 avg60=0.0 avg300=0.0 total=1\n'
        raise OSError

    monkeypatch.setattr('service.idle_warmup.Path.read_text', read)
    assert memory_pressure_high()
    values = {
        '/proc/pressure/memory': 'some avg10=0.00 avg60=0.0 avg300=0.0 total=1\n',
        '/sys/fs/cgroup/memory.max': '1000',
        '/sys/fs/cgroup/memory.current': '900',
    }
    monkeypatch.setattr('service.idle_warmup.Path.read_text', lambda path: values[str(path)])
    assert memory_pressure_high()
    values['/sys/fs/cgroup/memory.current'] = '800'
    assert not memory_pressure_high()
    values['/sys/fs/cgroup/memory.max'] = 'max'
    assert not memory_pressure_high()


def test_shutdown_joins_one_native_warmup_without_starting_another(monkeypatch):
    entered, released = Event(), Event()
    calls = []

    def work(*_):
        calls.append(1)
        entered.set()
        assert released.wait(5)

    monkeypatch.setattr('service.idle_warmup.warm_once', work)

    async def check():
        stop = asyncio.Event()
        task = asyncio.create_task(run_warmup(object(), 0.01, stop))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.sleep(0.02)
        assert not task.done()
        released.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(check())
    assert calls == [1]
