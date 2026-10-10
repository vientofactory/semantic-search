"""Concurrency tests for the HTTP sidecar (service/app.py).

Sync handlers run on anyio worker threads while `EngineState._lock` only
holds microsecond snapshots, so a long search, the background engine load
and generation swaps must never serialize concurrent requests behind each
other. The shared harness (server lifecycle, readiness wait, concurrent
fire) lives in `conftest.py`.
"""

import statistics
import threading
from pathlib import Path
from time import perf_counter, sleep, time

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
    """N simultaneous /search requests must overlap, not queue one by one.

    The stub searcher sleeps 0.4s per call: fully serialized the batch would
    take 8 x 0.4s = 3.2s, so a non-blocking sidecar must finish well under
    half that while actually running several searches at once.
    """
    delay, clients = 0.4, 8
    searcher = StubSearcher(delay=delay)
    with sidecar_server(monkeypatch, loader=ready_loader(searcher)) as base:
        responses, latencies, wall = fire_concurrently(
            base,
            [
                {'path': '/search', 'params': {'query': f'세입자 보호 {i}', 'k': 3}}
                for i in range(clients)
            ],
        )

    serial_budget = delay * clients
    codes = [response.status_code for response in responses]
    print(
        f'\n[concurrent /search] codes={codes} wall={wall:.2f}s '
        f'serial_budget={serial_budget:.2f}s '
        f'latencies={[round(latency, 2) for latency in latencies]} '
        f'max_in_flight={searcher.max_in_flight}',
    )
    assert codes == [200] * clients
    assert all(response.json()['results'] for response in responses)
    # Direct overlap proof: at least half the batch ran their search at once.
    assert searcher.max_in_flight >= clients // 2
    # Wall time must stay far below the fully-serialized budget.
    assert wall < serial_budget / 2
    assert max(latencies) < serial_budget / 2


def test_health_responds_fast_while_slow_search_runs(monkeypatch):
    """The state lock must not be held across a slow search (snapshot pattern).

    With a 1.5s search in flight, /health requests are measured individually:
    if `searcher.search()` ran under `EngineState._lock` each health check
    would stall for the rest of the search.
    """
    searcher = StubSearcher(delay=1.5)
    with sidecar_server(monkeypatch, loader=ready_loader(searcher)) as base:
        slow: dict = {}

        def slow_search() -> None:
            started = perf_counter()
            slow['response'] = httpx.get(
                f'{base}/search',
                params={'query': '느린 질의'},
                timeout=SERVE_TIMEOUT,
            )
            slow['latency'] = perf_counter() - started

        background = threading.Thread(target=slow_search, daemon=True)
        background.start()

        deadline = time() + 5
        while searcher.in_flight == 0 and time() < deadline:
            sleep(0.01)
        assert searcher.in_flight == 1, 'slow search never reached the handler'

        health_client = httpx.Client(timeout=5)
        try:
            health_latencies = []
            for _ in range(5):
                started = perf_counter()
                response = health_client.get(f'{base}/health')
                health_latencies.append(perf_counter() - started)
                assert response.status_code == 200
                assert response.json()['status'] == 'ready'
        finally:
            health_client.close()
        background.join(timeout=SERVE_TIMEOUT)

    print(
        f'\n[health during slow search] health_latencies='
        f'{[round(latency * 1000, 1) for latency in health_latencies]}ms '
        f'slow_search_latency={slow["latency"]:.2f}s',
    )
    assert slow['response'].status_code == 200
    # 0.5s vs the 1.5s search: health would block that long if a lock leaked.
    assert max(health_latencies) < 0.5


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
        assert max(latencies) < 1.0, f'503 responses queued: {[round(x, 2) for x in latencies]}'
        assert wall < 2.0

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
    """Real engine behind the production uvicorn CMD: concurrent batch vs baseline.

    Spawns the sidecar as a separate process (like Docker), waits for the
    engine to become ready, then measures three rounds of (sequential
    baseline, concurrent batch) on the warm process. Every batch must fully
    succeed, and no request may finish at its uncontended single-query
    speed — a serialized sidecar lets the FIRST request do exactly that
    while the last waits for the others (queue signature), whereas genuine
    overlap keeps every member several times slower than its baseline.
    """
    queries = [
        '세입자 보호',
        '국가 연구시설 공동 활용',
        '전기차 충전 인프라 구축',
        '개인정보 유출 신고 절차',
        '기후위기 대응 예산 편성',
        '중소기업 자금 지원',
    ]
    # Three rounds judged on their median: a single shot drifts both ways
    # (cold-round baseline elevation pushed the ratio down to 0.56, a
    # transient batch wall spike pushed it up to 0.79 against the 0.85
    # threshold) — the median drops either tail.
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
                baseline_started = perf_counter()
                for query in queries:
                    started = perf_counter()
                    response = measure_client.get(
                        f'{base}/search',
                        params={'query': query, 'k': 5},
                    )
                    single_latencies.append(perf_counter() - started)
                    assert response.status_code == 200
                baseline_wall = perf_counter() - baseline_started
                sequential_sum = sum(single_latencies)
                responses, latencies, wall = fire_concurrently(
                    base,
                    [{'path': '/search', 'params': {'query': query, 'k': 5}} for query in queries],
                    timeout=60,
                )
                rounds.append(
                    {
                        'sum': sequential_sum,
                        'client_delta': baseline_wall - sequential_sum,
                        'responses': responses,
                        'single': single_latencies,
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
    ratios = [item['wall'] / item['sum'] for item in rounds]
    max_over_sums = [max(item['latencies']) / item['sum'] for item in rounds]
    inflations = [
        (sum(item['latencies']) / len(item['latencies']))
        / (sum(item['single']) / len(item['single']))
        for item in rounds
    ]
    median_ratio = statistics.median(ratios)
    median_max_over_sum = statistics.median(max_over_sums)
    for index, item in enumerate(rounds, start=1):
        print(
            f'\n[real round {index}] sequential_sum={item["sum"]:.2f}s '
            f'client_delta={item["client_delta"]:.3f}s wall={item["wall"]:.2f}s '
            f'ratio={item["wall"] / item["sum"]:.2f} '
            f'codes={[response.status_code for response in item["responses"]]} '
            f'single={[round(latency, 2) for latency in item["single"]]} '
            f'concurrent={[round(latency, 2) for latency in item["latencies"]]}',
        )
    ladder_min_over_max = [
        round(min(item['latencies']) / max(item['latencies']), 2) for item in rounds
    ]
    print(
        f'\n[real margin] indexedChunks={health["indexedChunks"]} model={health["model"]} '
        f'ratios={[round(ratio, 2) for ratio in ratios]} '
        f'median_ratio={median_ratio:.2f} '
        f'ladder_min_over_max={ladder_min_over_max} '
        'gate=(min_latency>=2x_single) lock_signature=1x_single',
    )
    # GIL vs lock: a lock held across search() would queue every request, so
    # the last one waits for ALL others — max/sum ~= 1.0 (and ratio ~= 1.0)
    # on any machine. Observed max far below the sum with per-request latency
    # inflated ~3.7x (cpu) to ~5.2x (mps) means requests overlapped while
    # slowing each other down: CPU/GIL contention, not a lock. Lock absence
    # itself is proven directly by the stub tests (max_in_flight,
    # health-under-search).
    print(
        f'\n[real gil-vs-lock] inflation={[round(value, 1) for value in inflations]}x '
        f'median_max_over_sum={median_max_over_sum:.2f} (lock=1.00) '
        f'median_ratio={median_ratio:.2f} (lock=1.00)',
    )
    for index, item in enumerate(rounds, start=1):
        codes = [response.status_code for response in item['responses']]
        assert codes == [200] * len(queries), f'round {index}: {codes}'
        assert all(response.json()['results'] for response in item['responses'])
    # Queue detection, load-normalized: a sidecar that serialized searches
    # behind one lock releases them one by one, so the FIRST request finishes
    # at its uncontended single-query speed (min(batch) ~= 1x single) while
    # the last waits for all others. With genuine overlap nobody finishes
    # fast — every request pays the shared GIL / accelerator queue — so the
    # fastest batch member stays several times its own baseline: measured
    # 3.1-5.5x across cpu and mps runs, idle and busy. Threshold 2.0 keeps
    # >=1.5x headroom over the lowest honest observation and fires on the
    # ~1x lock signature on any hardware or machine load.
    #
    # Wall/sum ratio gates (0.85, then 1.0) were removed after measuring
    # them flake: honest rounds span 0.83-1.02 on a busy host while a
    # serialized sidecar also lands at ~1.0 — a sub-second wall clock has no
    # separating margin between the two signatures. Ratio stays a print-only
    # diagnostic below, alongside the latency ladder (min/max).
    for index, item in enumerate(rounds, start=1):
        baseline = statistics.median(item['single'])
        fastest = min(item['latencies'])
        assert fastest >= 2.0 * baseline, (
            f'round {index}: fastest request {fastest:.2f}s = '
            f'{fastest / baseline:.1f}x its uncontended {baseline:.2f}s — '
            'first-in-line speed implies queueing (search running under a lock?)'
        )
