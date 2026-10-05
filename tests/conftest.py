"""Shared test bootstrap and sidecar test harness.

Makes the side project root importable so test modules can `import
lawcast_semantic` directly. This is the only place in the suite that adjusts
sys.path, whatever way pytest is invoked.

The sidecar HTTP tests share their server lifecycle and request harness here
so any test module can reuse them: `sidecar_server` (in-process uvicorn with
a stub loader), `real_sidecar_process` (spawned production Dockerfile CMD),
`wait_until_ready`, and `fire_concurrently`.
"""

import os
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, sleep, time
from types import SimpleNamespace

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVE_TIMEOUT = 15.0

sys.path.insert(0, str(PROJECT_ROOT))


class StubSearcher:
    """Deterministic SemanticSearcher stand-in with an in-flight probe.

    `search()` sleeps `delay` seconds while counting concurrent callers, so
    tests can distinguish real overlap (max_in_flight > 1) from queued
    serialization and compare wall time against the serial budget.
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.chunk_ids = ['2220607-0000']
        self.chunks_by_id = {}
        self.index_updated_at = '2026-10-02T00:00:00+00:00'
        self.embedder = SimpleNamespace(model_name='stub-model')
        self.in_flight = 0
        self.max_in_flight = 0
        self._probe = threading.Lock()

    def search(self, query: str, k: int = 5):
        with self._probe:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                sleep(self.delay)
            return [
                SimpleNamespace(
                    chunk_id=self.chunk_ids[0],
                    notice_num=2220607,
                    subject='상가건물 임대차보호법 일부개정법률안',
                    committee='법제사법위원회',
                    section='주요내용',
                    score=0.5863,
                    text='점유를 회복할 필요가 있는 경우...',
                )
            ][:k]
        finally:
            with self._probe:
                self.in_flight -= 1

    def search_tiered(self, query: str, k: int = 5):
        """Sidecar entry point; the probe lives in `search`, so the tiered
        wrapper delegates to it and reports every hit as clear."""
        from lawcast_semantic.search import SearchResults

        return SearchResults(results=self.search(query, k), weak_results=[])


def ready_loader(searcher: StubSearcher):
    """Background loader that registers `searcher` immediately (design §6.1)."""

    def loader(state, boot_repair=None):
        state.mark_ready(searcher, 'stub-model')

    return loader


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextmanager
def sidecar_server(monkeypatch, *, loader=None):
    """Run the sidecar app under an in-process uvicorn server (stub tests).

    `loader` replaces the background engine load (`service.app.load_engine`);
    omitting it keeps the real loader. The update scheduler is disabled so no
    index tick interferes with timing measurements.
    """
    import uvicorn

    import service.app as app_module

    monkeypatch.setattr(app_module, 'STATE', app_module.create_state())
    if loader is not None:
        monkeypatch.setattr(app_module, 'load_engine', loader)
    monkeypatch.setattr(app_module.config, 'UPDATE_CRON', '')

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app_module.app,
            host='127.0.0.1',
            port=port,
            log_level='warning',
            access_log=False,
        ),
    )
    thread = threading.Thread(target=server.run, daemon=True, name='sidecar-under-test')
    thread.start()
    deadline = time() + SERVE_TIMEOUT
    while not server.started:
        assert thread.is_alive(), 'uvicorn exited during startup'
        assert time() < deadline, 'uvicorn did not start within timeout'
        sleep(0.02)
    try:
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        thread.join(timeout=SERVE_TIMEOUT)


@contextmanager
def real_sidecar_process():
    """Spawn `python -m uvicorn service.app:app` — the production Dockerfile CMD.

    Scheduling/boot-repair gates are disabled via env so no index tick runs
    during the measurement window; stdout is drained by a reader thread — an
    undrained PIPE fills at ~64KB and blocks the server.
    """
    port = _free_port()
    env = {
        **os.environ,
        'LAWCAST_SEMANTIC_UPDATE_CRON': '',
        'LAWCAST_SEMANTIC_DB_PATH': '',
        # The skip gate already requires the model in HF_HUB_CACHE; without
        # offline mode the loader still issues HEAD revalidations to
        # huggingface.co, and a degraded network adds retry backoffs to the
        # load (measured: engine_load 10s online vs 143s with an unreachable
        # hub) — a pre-measurement phase that must not depend on the network.
        'HF_HUB_OFFLINE': '1',
    }
    process = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'uvicorn',
            'service.app:app',
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--log-level',
            'warning',
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_chunks: list[bytes] = []

    def drain() -> None:
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(4096), b''):
            output_chunks.append(chunk)

    reader = threading.Thread(target=drain, daemon=True, name='sidecar-output-drain')
    reader.start()
    try:
        yield f'http://127.0.0.1:{port}', process, output_chunks
    finally:
        teardown_started = perf_counter()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        # Phase timing: a slow run must be attributable to a named phase.
        teardown = perf_counter() - teardown_started
        print(f'\n[real sidecar teardown] terminate_to_exit={teardown:.2f}s')


def wait_until_ready(
    base_url: str,
    *,
    timeout: float,
    interval: float = 0.3,
    check: Callable[[], None] | None = None,
) -> tuple[dict, float, float]:
    """Poll /health until the engine reports ready.

    Returns (health, seconds to first response, seconds to ready). `check`
    runs before every request so a caller can fail fast (e.g. the spawned
    sidecar already exited). Connection errors while the server is still
    binding are retried; a `failed` engine or a timeout fails the test.
    """
    started = perf_counter()
    with httpx.Client(timeout=5) as client:
        deadline = time() + timeout
        health: dict = {}
        server_up = None
        while time() < deadline:
            if check is not None:
                check()
            try:
                health = client.get(f'{base_url}/health').json()
            except httpx.HTTPError:
                sleep(interval)
                continue
            if server_up is None:
                server_up = perf_counter() - started
            if health.get('status') == 'ready':
                return health, server_up, perf_counter() - started
            if health.get('status') == 'failed':
                pytest.fail(f'engine failed to load: {health.get("error")}')
            sleep(interval)
    pytest.fail(f'engine did not become ready within {timeout}s: {health}')


def fire_concurrently(base_url: str, requests: list[dict], timeout: float = 20.0):
    """Fire one simultaneous GET per request spec from its own client thread.

    Every `httpx.Client` is built in the calling thread BEFORE the measured
    window: constructing one inside a worker costs ~30ms (CA-bundle parsing
    is Python work on the same GIL as an in-process server) and would
    contaminate `wall` and the per-request latencies with client-side setup.
    Returns (responses, latencies, wall_seconds).
    """
    clients = len(requests)
    http_clients = [httpx.Client(timeout=timeout) for _ in range(clients)]
    barrier = threading.Barrier(clients, timeout=timeout)
    responses: list = [None] * clients
    latencies = [0.0] * clients

    def worker(index: int) -> None:
        spec = requests[index]
        client = http_clients[index]
        barrier.wait()
        started = perf_counter()
        try:
            responses[index] = client.get(
                f'{base_url}{spec["path"]}',
                params=spec.get('params'),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced through the assertion below
            responses[index] = exc
        finally:
            latencies[index] = perf_counter() - started

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    wall_started = perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout + 5)
    wall = perf_counter() - wall_started
    for client in http_clients:
        client.close()

    failures = [response for response in responses if not isinstance(response, httpx.Response)]
    assert not failures, f'request errors under concurrency: {failures!r}'
    return responses, latencies, wall
