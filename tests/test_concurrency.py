"""Concurrency tests for the HTTP sidecar (service/app.py).

Sync handlers run on anyio worker threads while `EngineState._lock` only
holds microsecond snapshots, so a long search, the background engine load
and generation swaps must never serialize concurrent requests behind each
other. The shared harness (server lifecycle, readiness wait, concurrent
fire) lives in `conftest.py`.
"""

import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep

import httpx
import pytest
from conftest import (
    SERVE_TIMEOUT,
    StubSearcher,
    fire_concurrently,
    ready_loader,
    real_sidecar_process,
    sidecar_server,
    wait_until_ready,
)

from lawcast_semantic import config


def test_concurrent_searches_overlap_and_all_succeed(monkeypatch):
    """All eight handlers reach the same rendezvous before any may return."""
    clients = 8
    rendezvous = threading.Barrier(clients, timeout=5)

    class RendezvousSearcher(StubSearcher):
        def search(self, query, k=5):
            rendezvous.wait()
            return super().search(query, k)

    with sidecar_server(monkeypatch, loader=ready_loader(RendezvousSearcher())) as base:
        wait_until_ready(base, timeout=5)
        responses, _, wall = fire_concurrently(
            base,
            [
                {'path': '/search', 'params': {'query': f'세입자 보호 {i}', 'k': 3}}
                for i in range(clients)
            ],
        )

    codes = [response.status_code for response in responses]
    print(f'\n[concurrent /search] rendezvous={clients} codes={codes} wall={wall:.2f}s')
    assert codes == [200] * clients
    assert all(response.json()['results'] for response in responses)


def test_health_responds_while_search_is_blocked(monkeypatch):
    """Health must finish before a deliberately blocked search is released."""
    entered = threading.Event()
    release = threading.Event()

    class BlockingSearcher(StubSearcher):
        def search(self, query, k=5):
            entered.set()
            assert release.wait(10)
            return super().search(query, k)

    with sidecar_server(monkeypatch, loader=ready_loader(BlockingSearcher())) as base:
        wait_until_ready(base, timeout=5)
        with (
            httpx.Client(base_url=base, timeout=SERVE_TIMEOUT, trust_env=False) as client,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            slow = pool.submit(client.get, '/search', params={'query': '느린 질의'})
            try:
                assert entered.wait(5)
                health = client.get('/health', timeout=3)
                assert health.status_code == 200
                assert health.json()['status'] == 'ready'
                assert not slow.done()
            finally:
                release.set()
            assert slow.result().status_code == 200


def test_loading_engine_fails_fast_under_concurrent_load(monkeypatch):
    """While the engine loads, concurrent /search must answer 503 immediately.

    The loader thread is held on an event: requests arriving now have to fail
    fast (design: no queueing behind the model load), and once the load
    finishes the same server must start serving 200s without a restart.
    """
    searcher = StubSearcher()
    load_started = threading.Event()
    release = threading.Event()

    def blocking_loader(state, boot_repair=None):
        load_started.set()
        release.wait(timeout=30)
        state.mark_ready(searcher, 'stub-model')

    with sidecar_server(monkeypatch, loader=blocking_loader) as base:
        assert load_started.wait(timeout=5), 'background load never started'
        responses, latencies, wall = fire_concurrently(
            base,
            [{'path': '/search', 'params': {'query': f'질의 {i}'}} for i in range(6)],
            timeout=10,
        )
        health = httpx.get(f'{base}/health', timeout=5).json()
        assert [response.status_code for response in responses] == [503] * 6
        assert health['status'] == 'loading'

        release.set()
        wait_until_ready(base, timeout=5, interval=0.05)
        after_load = httpx.get(f'{base}/search', params={'query': '로드 완료 후'}, timeout=5)
        print(
            f'\n[loading fail-fast] wall={wall:.2f}s '
            f'latencies={[round(latency, 2) for latency in latencies]} '
            f'post_load_status={after_load.status_code}',
        )
        assert after_load.status_code == 200

    release.set()


def test_searches_keep_succeeding_across_generation_swaps(monkeypatch):
    """Live generation swaps must not fail or stall in-flight search traffic.

    A background loop swaps in a fresh searcher generation (the same
    lock-protected `mark_ready` field update the loader and reload path use)
    while 16 requests hit /search; every request has to return 200 and the
    reported generation must reflect all swaps.
    """
    import service.app as app_module

    searcher = StubSearcher(delay=0.05)
    with sidecar_server(monkeypatch, loader=ready_loader(searcher)) as base:
        stop = threading.Event()
        swapped: list = []

        def swap_loop() -> None:
            while not stop.is_set():
                app_module.STATE.mark_ready(StubSearcher(delay=0.05), 'stub-model')
                swapped.append(True)
                sleep(0.01)

        swapper = threading.Thread(target=swap_loop, daemon=True)
        swapper.start()
        sleep(0.02)
        responses, latencies, wall = fire_concurrently(
            base,
            [{'path': '/search', 'params': {'query': f'질의 {i % 4}', 'k': 2}} for i in range(16)],
        )
        stop.set()
        swapper.join(timeout=5)
        health = httpx.get(f'{base}/health', timeout=5).json()

    codes = [response.status_code for response in responses]
    print(
        f'\n[swap under load] codes={codes} wall={wall:.2f}s '
        f'latencies={[round(latency, 2) for latency in latencies]} '
        f'swaps={len(swapped)} generation={health["generation"]}',
    )
    assert codes == [200] * 16
    assert len(swapped) >= 2, 'no swaps happened during traffic'
    assert health['generation'] == 1 + len(swapped)


def _real_engine_available() -> bool:
    """True when artifacts/ is a complete set AND the model is cached locally.

    Gating on both keeps CI (no artifacts, no model cache) and fresh clones
    free of model downloads; the live test then exercises the full stack.
    """
    artifacts_ready = all(
        path.exists() for path in (config.CHUNKS_PATH, config.FAISS_INDEX_PATH, config.ID_MAP_PATH)
    )
    if not artifacts_ready:
        return False
    from huggingface_hub import constants as hub_constants

    model_dir = f'models--{config.MODEL_NAME.replace("/", "--")}'
    return (Path(hub_constants.HF_HUB_CACHE) / model_dir).exists()


@pytest.mark.skipif(
    not _real_engine_available(),
    reason='needs artifacts/ + cached embedding model',
)
def test_real_engine_serves_concurrent_queries():
    """Compare real HTTP ranks under serial and overlapping MPS or CPU traffic.

    Timing is diagnostic; the deterministic worker and phase tests own the
    overlap assertions. A fast first finisher does not imply serialization.
    """
    queries = [
        '세입자 보호',
        '국가 연구시설 공동 활용',
        '전기차 충전 인프라 구축',
        '개인정보 유출 신고 절차',
        '기후위기 대응 예산 편성',
        '중소기업 자금 지원',
    ]
    ROUNDS = 3

    with real_sidecar_process() as (base, process, output_chunks):

        def fail_if_exited() -> None:
            if process.poll() is not None:
                output = b''.join(output_chunks).decode(errors='replace')
                pytest.fail(f'sidecar process exited early: {output[-2000:]}')

        health, server_up, ready_at = wait_until_ready(
            base,
            timeout=180,
            check=fail_if_exited,
        )

        # Warm-up outside the measured rounds: on a cold process the first
        # query pays lazy init (measured 2.7s) and the baseline stays
        # elevated for several queries after that (0.17s/query vs 0.12s
        # steady), so one warm-up query was not enough to reach steady state.
        with httpx.Client(timeout=60) as measure_client:
            warmup_started = perf_counter()
            for query in queries[:3]:
                response = measure_client.get(
                    f'{base}/search',
                    params={'query': query, 'k': 5},
                )
                assert response.status_code == 200
            warmup_latency = perf_counter() - warmup_started

            rounds = []
            for _ in range(ROUNDS):
                single_latencies = []
                expected = []
                for query in queries:
                    started = perf_counter()
                    response = measure_client.get(
                        f'{base}/search',
                        params={'query': query, 'k': 5},
                    )
                    single_latencies.append(perf_counter() - started)
                    assert response.status_code == 200
                    expected.append(_rank_signature(response.json()))
                sequential_sum = sum(single_latencies)
                responses, latencies, wall = fire_concurrently(
                    base,
                    [{'path': '/search', 'params': {'query': query, 'k': 5}} for query in queries],
                    timeout=60,
                )
                rounds.append(
                    {
                        'sum': sequential_sum,
                        'responses': responses,
                        'single': single_latencies,
                        'expected': expected,
                        'latencies': latencies,
                        'wall': wall,
                    },
                )

    # Phase timing: makes any slow run attributable (boot vs model load vs
    # warm-up vs the three measured rounds) instead of a bare total.
    print(
        f'\n[real phases] server_up={server_up:.2f}s '
        f'engine_load={ready_at - server_up:.2f}s '
        f'warmup={warmup_latency:.2f}s '
        f'rounds_wall={[round(item["wall"], 2) for item in rounds]} '
        f'measure_total={sum(item["wall"] + item["sum"] for item in rounds):.2f}s',
    )
    for index, item in enumerate(rounds, start=1):
        codes = [response.status_code for response in item['responses']]
        print(
            f'\n[real round {index}] device={health["device"]} '
            f'sequential_sum={item["sum"]:.2f}s wall={item["wall"]:.2f}s '
            f'p50={statistics.median(item["latencies"]):.3f}s '
            f'max={max(item["latencies"]):.3f}s codes={codes}',
        )
        assert codes == [200] * len(queries), f'round {index}: {codes}'
        assert all(response.json()['results'] for response in item['responses'])
        assert [_rank_signature(r.json()) for r in item['responses']] == item['expected']


def _rank_signature(body):
    return [[hit['chunkId'] for hit in body[tier]] for tier in ('results', 'weakResults')]


def test_mps_admission_leaves_a_worker_free_and_releases_before_faiss(monkeypatch):
    """Prove the worker boundary without using a latency ratio as a proxy."""
    from types import SimpleNamespace

    import numpy as np
    from anyio import to_thread

    from lawcast_semantic.search import SemanticSearcher
    from service import app as sidecar

    enter_first = threading.Event()
    release_first = threading.Event()
    enter_rank = threading.Event()
    release_rank = threading.Event()
    enter_next = threading.Event()

    def embed(text):
        if text == 'first':
            enter_first.set()
            assert release_first.wait(10)
            return np.array([1.0, 0.0], dtype=np.float32)
        if text == 'failure':
            raise ValueError('test failure')
        enter_next.set()
        return np.array([0.0, 1.0], dtype=np.float32)

    def rank(vector, k):
        if vector[0] == 1:
            enter_rank.set()
            assert release_rank.wait(10)
        return []

    embedder = SimpleNamespace(
        embed_query=embed, query_slots=threading.BoundedSemaphore(1), model_name='stub-model'
    )
    searcher = SemanticSearcher(embedder, None, [], {})
    monkeypatch.setattr(searcher, '_rank', rank)
    monkeypatch.setattr(sidecar.config, 'MPS_QUERY_CONCURRENCY', 1)

    def loader(state, boot_repair=None):
        state.record_device('mps')
        state.mark_ready(searcher, 'stub-model')

    async def worker_state():
        workers = to_thread.current_default_thread_limiter()
        workers.total_tokens = 2
        admission = sidecar.app.state.mps_admission.statistics()
        return {'workers': workers.borrowed_tokens, 'waiting': admission.tasks_waiting}

    monkeypatch.setattr(sidecar.app.router, 'routes', list(sidecar.app.router.routes))
    sidecar.app.get('/_test_workers')(worker_state)
    with sidecar_server(monkeypatch, loader=loader) as base:
        wait_until_ready(base, timeout=5)
        with httpx.Client(base_url=base, timeout=10, trust_env=False) as client:
            client.get('/_test_workers').raise_for_status()
            with ThreadPoolExecutor(max_workers=4) as pool:
                first = pool.submit(client.get, '/search', params={'query': 'first'})
                try:
                    assert enter_first.wait(5)
                    queued = [
                        pool.submit(client.get, '/search', params={'query': f'next {i}'})
                        for i in range(3)
                    ]
                    deadline = perf_counter() + 5
                    while perf_counter() < deadline:
                        state = client.get('/_test_workers').json()
                        if state['waiting'] == 3:
                            break
                        sleep(0.01)
                    assert state == {'workers': 1, 'waiting': 3}
                    assert client.get('/health', timeout=2).status_code == 200
                    release_first.set()
                    assert enter_rank.wait(5)
                    assert enter_next.wait(5), 'the first FAISS phase retained the embedding slot'
                finally:
                    release_first.set()
                    release_rank.set()
                assert first.result().status_code == 200
                assert all(job.result().status_code == 200 for job in queued)
            assert (
                client.get(
                    '/search', params={'query': 'failure'}, headers={'Connection': 'close'}
                ).status_code
                == 500
            )
            assert client.get('/search', params={'query': 'recovery'}).status_code == 200
            assert client.get('/_test_workers').json() == {'workers': 0, 'waiting': 0}


def test_split_lanes_bound_ranking_without_stalling_embedding_or_health(monkeypatch):
    """Keep both active budgets separate and pin a generation across the handoff."""
    from types import SimpleNamespace

    import numpy as np
    from anyio import to_thread

    from lawcast_semantic.search import SemanticSearcher
    from service import app as sidecar

    release_embed = threading.Event()
    release_rank = threading.Event()
    first_rank = threading.Event()
    lock = threading.Lock()
    counters = {'embed': 0, 'embed_max': 0, 'rank': 0, 'rank_max': 0, 'embedded': 0}

    def embed(text):
        with lock:
            counters['embed'] += 1
            counters['embed_max'] = max(counters['embed_max'], counters['embed'])
        try:
            assert release_embed.wait(10)
            return np.array([1.0], dtype=np.float32)
        finally:
            with lock:
                counters['embed'] -= 1
                counters['embedded'] += 1

    def rank(vector, k):
        with lock:
            counters['rank'] += 1
            counters['rank_max'] = max(counters['rank_max'], counters['rank'])
        first_rank.set()
        try:
            assert release_rank.wait(10)
            return []
        finally:
            with lock:
                counters['rank'] -= 1

    embedder = SimpleNamespace(embed_query=embed, model_name='stub-original')
    searcher = SemanticSearcher(embedder, None, [], {})
    monkeypatch.setattr(searcher, '_rank', rank)
    monkeypatch.setattr(sidecar.config, 'QUERY_EMBED_WORKERS', 2)
    monkeypatch.setattr(sidecar.config, 'QUERY_RANK_WORKERS', 1)

    def loader(state, boot_repair=None):
        state.record_device('cpu')
        state.mark_ready(searcher, 'stub-original')

    async def lane_state():
        embed_lane, _, rank_lane = sidecar.app.state.query_lanes
        return {
            'embed_active': embed_lane.borrowed_tokens,
            'embed_waiting': embed_lane.statistics().tasks_waiting,
            'rank_active': rank_lane.borrowed_tokens,
            'rank_waiting': rank_lane.statistics().tasks_waiting,
            'default_active': to_thread.current_default_thread_limiter().borrowed_tokens,
        }

    monkeypatch.setattr(sidecar.app.router, 'routes', list(sidecar.app.router.routes))
    sidecar.app.get('/_test_lanes')(lane_state)
    with sidecar_server(monkeypatch, loader=loader) as base:
        wait_until_ready(base, timeout=5)
        with (
            httpx.Client(base_url=base, timeout=10, trust_env=False) as client,
            ThreadPoolExecutor(max_workers=4) as pool,
        ):

            def wait_for(expected):
                deadline = perf_counter() + 5
                while perf_counter() < deadline:
                    actual = client.get('/_test_lanes').json()
                    if all(actual[k] == v for k, v in expected.items()):
                        return
                    sleep(0.01)
                pytest.fail(f'Expected lane state {expected}; got {actual}')

            jobs = [
                pool.submit(client.get, '/search', params={'query': f'probe {i}'}) for i in range(4)
            ]
            try:
                wait_for({'embed_active': 2, 'embed_waiting': 2, 'default_active': 0})
                assert client.get('/health', timeout=2).status_code == 200
                release_embed.set()
                assert first_rank.wait(5)
                wait_for(
                    {'embed_active': 0, 'rank_active': 1, 'rank_waiting': 3, 'default_active': 0}
                )
                assert client.get('/health', timeout=2).status_code == 200
                sidecar.STATE.mark_ready(StubSearcher(), 'stub-reloaded')
            finally:
                release_embed.set()
                release_rank.set()
            bodies = [job.result().raise_for_status().json() for job in jobs]
            assert all(body['model'] == 'stub-original' for body in bodies)
            assert client.get('/health').json()['model'] == 'stub-reloaded'
            assert counters['embed_max'] == 2
            assert counters['rank_max'] == 1
            assert counters['embedded'] == 4
