"""Tests for the semantic search sidecar service (no model download).

The engine loader is stubbed so validation, state handling and response
shapes are exercised without the model stack. The `phases` fixture drives
the REAL `load_engine` with fakes for the lazy model exports (design §6.1).
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Serving generation's artifact stamp reported by both /health and /search.
STUB_STAMP = '2026-10-02T00:00:00+00:00'


@pytest.fixture()
def client(monkeypatch):
    import service.app as app_module

    def stub_search(query: str, k: int):
        return [
            SimpleNamespace(
                chunk_id='2220607-0000',
                notice_num=2220607,
                subject='상가건물 임대차보호법 일부개정법률안',
                committee='법제사법위원회',
                section='주요내용',
                score=0.5863,
                text='점유를 회복할 필요가 있는 경우...',
            )
        ][:k]

    def stub_search_tiered(query: str, k: int):
        # The engine tiers hits; the sidecar only passes the two lists
        # through (weak example rides along so /search weakResults is real).
        weak = [
            SimpleNamespace(
                chunk_id='2220715-0000',
                notice_num=2220715,
                subject='주택 임대차 보증금 반환 특례법안',
                committee='국토교통위원회',
                section='제안이유',
                score=0.35,
                text='관련도가 낮은 결과 예시...',
            )
        ][:k]
        return SimpleNamespace(results=stub_search(query, k), weak_results=weak)

    def ready_loader(state, boot_repair=None):
        state.mark_ready(
            SimpleNamespace(
                search_tiered=stub_search_tiered,
                chunk_ids=['2220607-0000'],
                index_updated_at=STUB_STAMP,
            ),
            'stub-model',
        )

    monkeypatch.setattr(app_module, 'STATE', app_module.create_state())
    monkeypatch.setattr(app_module, 'load_engine', ready_loader)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_search_returns_ranked_hits(client):
    response = client.get('/search', params={'query': '세입자 보호', 'k': 3})
    assert response.status_code == 200
    body = response.json()
    assert body['query'] == '세입자 보호'
    assert body['model'] == 'stub-model'
    hit = body['results'][0]
    assert hit['chunkId'] == '2220607-0000'
    assert hit['noticeNum'] == 2220607
    assert hit['section'] == '주요내용'
    assert hit['score'] == pytest.approx(0.5863)
    assert 'text' in hit


def test_search_passes_weak_results_through_separately(client):
    """Clear hits land in `results`; the weak band rides in `weakResults`
    (never mixed into the main list) so the UI can hide it behind a reveal."""
    body = client.get('/search', params={'query': '세입자 보호', 'k': 3}).json()
    assert [hit['noticeNum'] for hit in body['results']] == [2220607]
    assert [hit['noticeNum'] for hit in body['weakResults']] == [2220715]
    assert body['weakResults'][0]['score'] == pytest.approx(0.35)
    # The two tiers never overlap on a notice.
    assert {hit['noticeNum'] for hit in body['results']}.isdisjoint(
        hit['noticeNum'] for hit in body['weakResults']
    )


def test_search_rejects_blank_and_overlong_query(client):
    assert client.get('/search', params={'query': '   '}).status_code == 400
    assert client.get('/search', params={'query': ''}).status_code in (400, 422)
    assert client.get('/search', params={'query': 'x' * 501}).status_code == 422


def test_search_rejects_out_of_range_k(client):
    assert client.get('/search', params={'query': '질의', 'k': 0}).status_code == 422
    assert client.get('/search', params={'query': '질의', 'k': 201}).status_code == 422
    # k is a chunk count: the backend over-requests up to MAX_K chunks so
    # chunk->notice dedup can fill its notice-level k.
    assert client.get('/search', params={'query': '질의', 'k': 150}).status_code == 200


def test_search_reports_engine_failure_as_503(monkeypatch):
    import service.app as app_module

    def failed_loader(state, boot_repair=None):
        state.mark_failed('OSError: artifacts missing')

    monkeypatch.setattr(app_module, 'STATE', app_module.create_state())
    monkeypatch.setattr(app_module, 'load_engine', failed_loader)
    with TestClient(app_module.app) as test_client:
        response = test_client.get('/search', params={'query': '세입자 보호'})
        assert response.status_code == 503
        assert 'OSError' in response.json()['detail']

        health = test_client.get('/health').json()
        assert health['status'] == 'failed'
        assert 'OSError' in health['error']


def test_search_reports_loading_state_as_503(monkeypatch):
    import service.app as app_module

    def never_finishes(state, boot_repair=None):
        pass  # keep the engine in its initial 'loading' state

    monkeypatch.setattr(app_module, 'STATE', app_module.create_state())
    monkeypatch.setattr(app_module, 'load_engine', never_finishes)
    with TestClient(app_module.app) as test_client:
        assert test_client.get('/search', params={'query': '질의'}).status_code == 503
        assert test_client.get('/health').json()['status'] == 'loading'


def test_health_shape(client):
    health = client.get('/health').json()
    assert health['status'] == 'ready'
    assert health['model'] == 'stub-model'
    assert health['indexedChunks'] == 1
    assert health['device'] is None  # stub loader never runs the model phase
    assert health['error'] is None
    assert health['updating'] is False  # no tick has run yet
    assert health['lastUpdateTriggeredAt'] is None


def test_health_exposes_tick_progress(client):
    """The tick trigger is queryable over HTTP while it runs (§5.2 additive
    fields); recording the outcome flips `updating` off."""
    import service.app as app_module

    app_module.STATE.record_tick_started()
    health = client.get('/health').json()
    assert health['updating'] is True
    assert datetime.fromisoformat(health['lastUpdateTriggeredAt'])  # ISO 8601

    app_module.STATE.record_update('changed')
    health = client.get('/health').json()
    assert health['updating'] is False
    assert health['lastUpdateResult'] == 'changed'


def test_tick_info_logs_reach_the_server_process():
    """run_update_cycle logs trigger/result at INFO; uvicorn's log config
    leaves the root logger at WARNING with no root handler, so the `service`
    package must carry its own level + handler or the live server process
    silently drops them (found by driving a real uvicorn instance)."""
    import service.app as app_module

    assert app_module.service_logger is logging.getLogger('service')  # applied at import
    assert app_module.service_logger.level == logging.INFO
    assert app_module.service_logger.handlers
    assert logging.getLogger('service.update_runner').isEnabledFor(logging.INFO)


def test_last_update_at_reported_on_health_and_search(client):
    """Both main responses carry the serving generation's artifact stamp,
    so the backend gets the time with the search it already requests."""
    assert client.get('/health').json()['lastUpdateAt'] == STUB_STAMP
    search = client.get('/search', params={'query': '세입자 보호'}).json()
    assert search['lastUpdateAt'] == STUB_STAMP


@pytest.fixture()
def phases():
    """Install deterministic stand-ins for lawcast_semantic's lazy exports.

    Prior state is snapshotted from `vars()` directly: a plain getattr on a
    PEP 562 export would import the real model stack (torch) into the suite.
    """
    import lawcast_semantic

    missing = object()
    saved = {
        name: vars(lawcast_semantic).get(name, missing)
        for name in ('KoreanEmbedder', 'SemanticSearcher')
    }
    cfg = SimpleNamespace(embed_error=None, outcomes=[], load_calls=[])

    class FakeEmbedder:
        model_name = 'stub-model'

        def __init__(self):
            if cfg.embed_error is not None:
                raise cfg.embed_error

    class FakeSearcher:
        @classmethod
        def load(cls, embedder, **_paths):
            cfg.load_calls.append(embedder)
            outcome = cfg.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    lawcast_semantic.KoreanEmbedder = FakeEmbedder
    lawcast_semantic.SemanticSearcher = FakeSearcher
    yield cfg
    for name, original in saved.items():
        if original is missing:
            delattr(lawcast_semantic, name)
        else:
            setattr(lawcast_semantic, name, original)


def test_load_engine_success_marks_ready(phases):
    import service.app as app_module

    loaded = SimpleNamespace(chunk_ids=['stub-0000'], chunks_by_id={}, index_updated_at=None)
    phases.outcomes.append(loaded)
    state = app_module.create_state()
    app_module.load_engine(state)

    assert state.status == 'ready'
    assert state.searcher is loaded
    assert state.model_name == 'stub-model'
    assert state.indexed_chunks == 1


def test_load_engine_stamps_the_resolved_device(phases):
    """`/health.device` reports the hardware the embedder actually runs on
    (auto-detected accelerator, or cpu — lawcast_semantic/device.py)."""
    import lawcast_semantic
    import service.app as app_module

    lawcast_semantic.KoreanEmbedder.device = 'mps'
    loaded = SimpleNamespace(chunk_ids=['stub-0000'], chunks_by_id={}, index_updated_at=None)
    phases.outcomes.append(loaded)
    state = app_module.create_state()
    app_module.load_engine(state)

    assert state.status == 'ready'
    assert state.device == 'mps'
    assert state.snapshot()['device'] == 'mps'


def test_device_is_reported_even_when_the_artifact_phase_fails(phases):
    """The device is stamped at the model phase, so a failed index load still
    says which hardware the model came up on (model ok vs model failed is the
    diagnosis an operator needs first)."""
    import lawcast_semantic
    import service.app as app_module

    lawcast_semantic.KoreanEmbedder.device = 'cuda'
    phases.outcomes.append(FileNotFoundError('faiss.index missing'))
    state = app_module.create_state()
    app_module.load_engine(state)

    assert state.status == 'failed'
    assert state.device == 'cuda'


def test_embedder_phase_failure_never_calls_repair(monkeypatch, phases):
    """Discrimination is the phase, not the exception type (§6.1).

    The same ValueError shape that phase 2 treats as repairable must go
    straight to `failed` when raised by the embedder phase.
    """
    import service.app as app_module

    monkeypatch.setattr(app_module.config, 'DB_PATH', '/data/lawcast.db')
    phases.embed_error = ValueError('broken')
    repair_calls = []

    def boot_repair(embedder, db_path):
        repair_calls.append((embedder, db_path))
        return True

    state = app_module.create_state()
    app_module.load_engine(state, boot_repair=boot_repair)

    assert state.status == 'failed'
    assert 'ValueError: broken' in state.error
    assert repair_calls == []
    assert phases.load_calls == []


def test_searcher_phase_failure_skips_repair_without_db_path(monkeypatch, phases):
    """Scheduling disabled (empty DB_PATH) => behavior exactly as before."""
    import service.app as app_module

    monkeypatch.setattr(app_module.config, 'DB_PATH', '')
    phases.outcomes.append(FileNotFoundError('chunks.jsonl missing'))
    repair_calls = []

    def boot_repair(embedder, db_path):
        repair_calls.append(db_path)
        return True

    state = app_module.create_state()
    app_module.load_engine(state, boot_repair=boot_repair)

    assert state.status == 'failed'
    assert 'chunks.jsonl missing' in state.error
    assert repair_calls == []
    assert len(phases.load_calls) == 1


def test_searcher_phase_failure_runs_repair_and_recovers(monkeypatch, phases):
    """Hook fires on the searcher phase only; True => re-load -> ready."""
    import service.app as app_module

    monkeypatch.setattr(app_module.config, 'DB_PATH', '/data/lawcast.db')
    loaded = SimpleNamespace(
        chunk_ids=['stub-0000', 'stub-0001'], chunks_by_id={}, index_updated_at=None
    )
    phases.outcomes.extend([FileNotFoundError('torn set'), loaded])
    repair_calls = []

    def boot_repair(embedder, db_path):
        repair_calls.append((embedder, db_path))
        return True

    state = app_module.create_state()
    app_module.load_engine(state, boot_repair=boot_repair)

    assert len(repair_calls) == 1
    embedder, db_path = repair_calls[0]
    assert db_path == '/data/lawcast.db'
    assert embedder.model_name == 'stub-model'
    assert len(phases.load_calls) == 2  # failed once, re-loaded after repair
    assert state.status == 'ready'
    assert state.searcher is loaded
    assert state.indexed_chunks == 2


def test_repair_refusal_marks_failed_with_original_error(monkeypatch, phases):
    import service.app as app_module

    monkeypatch.setattr(app_module.config, 'DB_PATH', '/data/lawcast.db')
    phases.outcomes.append(FileNotFoundError('torn set'))
    repair_calls = []

    def boot_repair(embedder, db_path):
        repair_calls.append(db_path)
        return False

    state = app_module.create_state()
    app_module.load_engine(state, boot_repair=boot_repair)

    assert repair_calls == ['/data/lawcast.db']
    assert state.status == 'failed'
    assert 'torn set' in state.error


def test_repair_hook_exception_is_contained(monkeypatch, phases):
    """A hook bug must never trap the boot thread in `loading` (§6.1)."""
    import service.app as app_module

    monkeypatch.setattr(app_module.config, 'DB_PATH', '/data/lawcast.db')
    phases.outcomes.append(FileNotFoundError('torn set'))

    def boot_repair(embedder, db_path):
        raise RuntimeError('runner bug')

    state = app_module.create_state()
    app_module.load_engine(state, boot_repair=boot_repair)

    assert state.status == 'failed'
    assert 'torn set' in state.error
    assert 'boot repair failed: RuntimeError: runner bug' in state.error


def test_import_service_app_stays_light():
    """§6.1: `import service.app` must not pull the model stack."""
    code = (
        'import sys\n'
        'import service.app\n'
        "heavy = [name for name in ('torch', 'faiss') if name in sys.modules]\n"
        "assert not heavy, f'heavy modules imported: {heavy}'\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
