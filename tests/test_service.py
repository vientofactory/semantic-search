"""Tests for the semantic search sidecar service (no model download).

The engine loader is stubbed so validation, state handling and response
shapes are exercised without the model stack.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


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

    def ready_loader(state):
        state.mark_ready(
            SimpleNamespace(search=stub_search, chunk_ids=['2220607-0000']), 'stub-model'
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


def test_search_rejects_blank_and_overlong_query(client):
    assert client.get('/search', params={'query': '   '}).status_code == 400
    assert client.get('/search', params={'query': ''}).status_code in (400, 422)
    assert client.get('/search', params={'query': 'x' * 501}).status_code == 422


def test_search_rejects_out_of_range_k(client):
    assert client.get('/search', params={'query': '질의', 'k': 0}).status_code == 422
    assert client.get('/search', params={'query': '질의', 'k': 51}).status_code == 422


def test_search_reports_engine_failure_as_503(monkeypatch):
    import service.app as app_module

    def failed_loader(state):
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

    def never_finishes(state):
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
    assert health['error'] is None
