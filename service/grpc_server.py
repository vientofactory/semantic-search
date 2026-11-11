"""Internal gRPC listener sharing the HTTP application's engine and worker lanes."""

import asyncio
import logging

import grpc
from anyio import CancelScope, current_time
from fastapi import FastAPI, HTTPException

from lawcast_semantic import config

from . import semantic_search_pb2 as pb
from .search_trace import search_trace

SERVICE = 'lawcast.semantic.v1.SemanticSearch'
logger = logging.getLogger(__name__)


async def start_server(app: FastAPI, bind: str) -> grpc.aio.Server:
    # Imported after the ASGI module finishes initializing.
    from .app import health, search_with_admission

    async def get_health(_request, _context):
        state = health()
        fields = ('status', 'indexedChunks', 'lastUpdateAt', 'lastUpdateTriggeredAt', 'maxK')
        return pb.HealthResponse(
            **{field: state[field] for field in fields if state[field] is not None}
        )

    async def search(request, context):
        k = request.k if request.HasField('k') else config.DEFAULT_K
        if not 1 <= len(request.query) <= config.MAX_QUERY_LENGTH:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'invalid query length')
        if not 1 <= k <= config.MAX_K:
            await context.abort(grpc.StatusCode.OUT_OF_RANGE, 'invalid chunk window')

        # grpc.aio cancels its asyncio handler directly. Translate that into an
        # AnyIO cancellation so waiting work is dropped, but an active native
        # worker retains its limiter token until it actually finishes.
        remaining = context.time_remaining()
        scope = CancelScope(
            deadline=current_time() + remaining if remaining is not None else float('inf')
        )

        async def run_search():
            with scope, search_trace('grpc', dict(context.invocation_metadata())):
                return await search_with_admission(app, request.query, k)

        work = asyncio.create_task(run_search())
        try:
            response = await asyncio.shield(work)
        except asyncio.CancelledError:
            scope.cancel()
            try:
                await work
            except Exception:
                logger.exception('Cancelled semantic search worker failed')
            raise
        except HTTPException as error:
            status = (
                grpc.StatusCode.INVALID_ARGUMENT
                if error.status_code == 400
                else grpc.StatusCode.UNAVAILABLE
            )
            await context.abort(status, str(error.detail))
        if scope.cancel_called:
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, 'search deadline exceeded')
        return pb.SearchResponse(**response.model_dump(exclude_none=True))

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE,
                {
                    'GetHealth': grpc.unary_unary_rpc_method_handler(
                        get_health,
                        request_deserializer=pb.HealthRequest.FromString,
                        response_serializer=pb.HealthResponse.SerializeToString,
                    ),
                    'Search': grpc.unary_unary_rpc_method_handler(
                        search,
                        request_deserializer=pb.SearchRequest.FromString,
                        response_serializer=pb.SearchResponse.SerializeToString,
                    ),
                },
            ),
        )
    )
    server.add_insecure_port(bind)
    await server.start()
    return server
