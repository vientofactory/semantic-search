"""Socket-level gRPC contract and deadline tests without model downloads."""

import asyncio
import threading

import grpc
import pytest
from anyio import CapacityLimiter, to_thread
from conftest import _free_port

from service import app as sidecar
from service import semantic_search_pb2 as pb
from service.grpc_server import SERVICE, start_server
from service.search_trace import trace_fields


def rpc(channel, name, request, response):
    return channel.unary_unary(
        f'/{SERVICE}/{name}',
        request_serializer=request.SerializeToString,
        response_deserializer=response.FromString,
    )


def test_health_search_presence_and_statuses(monkeypatch):
    async def run():
        monkeypatch.setattr(sidecar, 'STATE', sidecar.create_state())
        original = sidecar.search_with_admission
        empty = False
        traces = []

        async def dispatch(app, query, k):
            if not empty:
                return await original(app, query, k)
            traces.append(await to_thread.run_sync(trace_fields))
            return sidecar.SearchResponse(
                query=query,
                k=k,
                model=None,
                lastUpdateAt=None,
                candidateCount=0,
                results=[],
                weakResults=[],
                timings=sidecar.SearchTimings(
                    denseMs=1, annotateMs=0, totalMs=1, admissionMs=0.25, rankWaitMs=0
                ),
            )

        monkeypatch.setattr(sidecar, 'search_with_admission', dispatch)
        addr = f'127.0.0.1:{_free_port()}'
        server = await start_server(sidecar.app, addr)
        try:
            async with grpc.aio.insecure_channel(addr) as channel:
                health = rpc(channel, 'GetHealth', pb.HealthRequest, pb.HealthResponse)
                search = rpc(channel, 'Search', pb.SearchRequest, pb.SearchResponse)
                status = await health(pb.HealthRequest(), timeout=2)
                assert status.status == 'loading'
                assert status.indexedChunks == 0
                assert status.maxK == sidecar.config.MAX_K
                assert not status.HasField('lastUpdateAt')

                cases = [
                    (' ', 1, grpc.StatusCode.INVALID_ARGUMENT),
                    ('', 1, grpc.StatusCode.INVALID_ARGUMENT),
                    (
                        'x' * (sidecar.config.MAX_QUERY_LENGTH + 1),
                        1,
                        grpc.StatusCode.INVALID_ARGUMENT,
                    ),
                    ('세입자', 0, grpc.StatusCode.OUT_OF_RANGE),
                    ('세입자', sidecar.config.MAX_K + 1, grpc.StatusCode.OUT_OF_RANGE),
                    ('세입자', 1, grpc.StatusCode.UNAVAILABLE),
                ]
                for query, k, expected in cases:
                    with pytest.raises(grpc.aio.AioRpcError) as err:
                        await search(pb.SearchRequest(query=query, k=k), timeout=2)
                    assert err.value.code() == expected

                empty = True
                trace_id = 'abecd012-1234-4321-8765-abcdef987654'
                result = await search(
                    pb.SearchRequest(query='세입자'),
                    timeout=2,
                    metadata=(
                        ('x-lawcast-search-trace', trace_id),
                        ('x-lawcast-search-attempt', '3'),
                    ),
                )
                assert traces == [('grpc', trace_id, '3')]
                assert result.timings.admissionMs == 0.25
                assert result.timings.HasField('rankWaitMs') and result.timings.rankWaitMs == 0
                await search(
                    pb.SearchRequest(query='세입자'),
                    timeout=2,
                    metadata=(
                        ('x-lawcast-search-trace', 'invalid'),
                        ('x-lawcast-search-attempt', '3'),
                    ),
                )
                assert traces[-1] == ('grpc', '-', '-')
                assert result.k == sidecar.config.DEFAULT_K
                assert result.HasField('candidateCount') and result.candidateCount == 0
                assert not result.HasField('lastUpdateAt')
                assert result.results == result.weakResults == []
        finally:
            await server.stop(grace=0)

    asyncio.run(run())


def test_deadline_drops_queued_work_without_releasing_active_worker(monkeypatch):
    async def run():
        limiter = CapacityLimiter(1)
        queued = asyncio.Event()
        queue_released = asyncio.Event()
        started = asyncio.Event()
        release = threading.Event()
        worker_calls = []
        loop = asyncio.get_running_loop()

        def worker(query):
            worker_calls.append(query)
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)

        async def lane(app, query, k):
            if query == 'second':
                queued.set()
            try:
                await to_thread.run_sync(worker, query, limiter=limiter)
            finally:
                if query == 'second':
                    queue_released.set()

        monkeypatch.setattr(sidecar, 'search_with_admission', lane)
        addr = f'127.0.0.1:{_free_port()}'
        server = await start_server(sidecar.app, addr)
        try:
            async with grpc.aio.insecure_channel(addr) as channel:
                search = rpc(channel, 'Search', pb.SearchRequest, pb.SearchResponse)
                first = search(pb.SearchRequest(query='first'), timeout=1)
                await asyncio.wait_for(started.wait(), timeout=2)
                second = search(pb.SearchRequest(query='second'), timeout=1.5)
                await asyncio.wait_for(queued.wait(), timeout=2)
                for call in (first, second):
                    with pytest.raises(grpc.aio.AioRpcError) as err:
                        await call
                    assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                await asyncio.wait_for(queue_released.wait(), timeout=1)
                assert limiter.borrowed_tokens == 1
                assert worker_calls == ['first']
                release.set()
                async with asyncio.timeout(2):
                    while limiter.borrowed_tokens:
                        await asyncio.sleep(0.01)
                assert worker_calls == ['first']
        finally:
            release.set()
            await server.stop(grace=0)

    asyncio.run(run())
