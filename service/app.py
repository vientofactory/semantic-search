"""Online adapter: HTTP and optional internal gRPC over one search engine.

Run from the side project root (the engine loads artifacts via
`lawcast_semantic.config` defaults):

    .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8300

The LawCast backend uses gRPC in Compose and falls back to keyword search
when the sidecar is unavailable. HTTP remains available for operations. The
engine loads once in a background thread at startup; while it loads,
`/search` answers 503 immediately instead of making requests wait. A failed load
is reported through `/health` and `/search` until the process is restarted.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from anyio import CapacityLimiter, from_thread, to_thread
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from lawcast_semantic import config  # env-only module: import stays light
from lawcast_semantic.hybrid import has_lexical_evidence
from lawcast_semantic.omp_env import use_single_threaded_omp

from .logging import get_service_logger
from .search_trace import search_trace, trace_fields

if TYPE_CHECKING:
    from lawcast_semantic.search import SearchResults
    from lawcast_semantic.timing import DensePhase

use_single_threaded_omp()

# Sidecar logs go through the dedicated logger (`service/logging.py`): the
# same `%(levelprefix)s` layout uvicorn's own lines use, with a `[sidecar]`
# context marker so the two sources never read as each other. The handler
# lives on the `service` base (children propagate); propagation stays on so
# pytest caplog and any root config still see every record.
service_logger = get_service_logger('service')

# Boot-repair hook contract (design §6.1): (embedder, db_path) -> whether the
# on-disk artifact set was repaired. The update runner supplies the real one;
# None (default) means no repair capability, matching pre-design behavior.
BootRepairHook = Callable[[Any, Path], bool]

# Request bounds are owned by lawcast_semantic.config (env-injectable via
# LAWCAST_SEMANTIC_MAX_QUERY_LENGTH / _MAX_K / _DEFAULT_K); re-exported here
# so the wire contract and tests keep a single import site.
MAX_QUERY_LENGTH = config.MAX_QUERY_LENGTH
# `/search` k counts CHUNKS, not notices: the backend over-requests chunks so
# chunk->notice dedup can still fill its notice-level k. The default must
# match SIDE_CAR_MAX_CHUNK_K in
# backend/src/modules/semantic-search/semantic-search.constants.ts (pinned by
# the contract test there).
MAX_K = config.MAX_K
DEFAULT_K = config.DEFAULT_K


class EngineState:
    """Process-wide engine holder with single-writer locking.

    Owns the loaded searcher; the HTTP handlers only take snapshots so a
    long model load never blocks a request. The update runner swaps a new
    generation in under the same lock (design §5.2).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: Literal['loading', 'ready', 'failed'] = 'loading'
        self.searcher: Any = None
        self.model_name: str | None = None
        # Device the embedding model actually runs on (auto-detected
        # accelerator or cpu) — reported by /health so an operator can see
        # which hardware the probe picked. None until the model loads.
        self.device: str | None = None
        self.indexed_chunks: int = 0
        self.error: str | None = None
        # §5.2 observability: what serves now, what the last tick did.
        self.generation = 0
        self.loaded_fingerprint: str | None = None
        self.reload_error: str | None = None
        # Index-update time of the serving generation: the id_map `updated_at`
        # stamped by VectorIndex.save, adopted at mark_ready/reload (durable
        # across restarts; owned by the artifact write, never by tick code).
        self.last_update_at: str | None = None
        self.last_update_result: str | None = None  # 'changed'|'unchanged'|'failed'|'skipped'
        self.last_update_error: str | None = None
        # Trigger moment of the latest tick: stamped at cycle entry so /health
        # can show a tick while it runs (record_update clears `updating`).
        self.last_update_triggered_at: str | None = None
        self.updating = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'status': self.status,
                'searcher': self.searcher,
                'model': self.model_name,
                'device': self.device,
                'indexedChunks': self.indexed_chunks,
                'error': self.error,
                'generation': self.generation,
                'reloadError': self.reload_error,
                'lastUpdateAt': self.last_update_at,
                'lastUpdateResult': self.last_update_result,
                'lastUpdateError': self.last_update_error,
                'lastUpdateTriggeredAt': self.last_update_triggered_at,
                'updating': self.updating,
            }

    def mark_ready(
        self,
        searcher: Any,
        model_name: str,
        fingerprint: str | None = None,
    ) -> None:
        """First successful load (generation 1).

        `fingerprint` is the set identity the load validated; None means
        unknown, which the first tick treats as a mismatch and verifies with
        one reload (§5.3). The run device is owned by `record_device`, which
        runs earlier (model phase), so a ready state never erases it.
        """
        with self._lock:
            self.searcher = searcher
            self.model_name = model_name
            self.indexed_chunks = len(searcher.chunk_ids)
            self.status = 'ready'
            self.error = None
            self.generation += 1
            self.loaded_fingerprint = fingerprint
            self.last_update_at = searcher.index_updated_at

    def reload(self, embedder: Any) -> bool:
        """Load-validate-swap a new generation while the old one serves (§5.2).

        `status` is never touched, so readiness holds for the whole swap. A
        validation failure discards the new searcher, keeps the previous
        generation and `generation` untouched, and records `reloadError` for
        the next tick to retry (§5.3).
        """
        from lawcast_semantic import SemanticSearcher
        from lawcast_semantic.chunking import compute_chunks_fingerprint

        try:
            searcher = SemanticSearcher.load(
                embedder,
                chunks_path=config.CHUNKS_PATH,
                index_path=config.FAISS_INDEX_PATH,
                id_map_path=config.ID_MAP_PATH,
            )
            fingerprint = compute_chunks_fingerprint(searcher.chunks_by_id.values())
        except Exception as exc:  # noqa: BLE001 - validation gates surface as any exception
            with self._lock:
                self.reload_error = f'{type(exc).__name__}: {exc}'
            return False
        with self._lock:
            self.searcher = searcher
            self.model_name = embedder.model_name
            self.indexed_chunks = len(searcher.chunk_ids)
            self.loaded_fingerprint = fingerprint
            self.reload_error = None
            self.generation += 1
            # Adopt the new generation's artifact stamp; a failed reload
            # (above) leaves the previous generation's time in place.
            self.last_update_at = searcher.index_updated_at
        return True

    def record_device(self, device: str | None) -> None:
        """Stamp the embedder's resolved run device (§ observability).

        Called as soon as the model loads — before the artifact phase — so
        /health still reports the detected hardware when the artifact load
        fails (model ok, index missing is a different diagnosis than model
        failed).
        """
        with self._lock:
            self.device = device

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = 'failed'
            self.error = error

    def record_tick_started(self) -> None:
        """Trigger moment of the latest tick: stamps the fired-at time and
        flips `updating` on; `record_update` brackets it off."""
        with self._lock:
            self.last_update_triggered_at = datetime.now(UTC).isoformat()
            self.updating = True

    def record_update(self, result: str, error: str | None = None) -> None:
        """Record one update tick's outcome for §5.2's additive /health fields.

        Only the tick vocabulary lives here: `lastUpdateAt` is owned by the
        artifact write (VectorIndex.save) and adopted at load/swap, so a
        failed or unchanged tick never moves it.
        """
        with self._lock:
            self.updating = False
            self.last_update_result = result
            self.last_update_error = error


class SearchHit(BaseModel):
    chunkId: str
    noticeNum: int
    subject: str
    committee: str
    section: str
    score: float
    text: str
    # Whether the (alias-expanded) query shares a content token with this
    # chunk's subject + text (lawcast_semantic.hybrid). The backend's
    # legacy no-result gate reads this flag on uncalibrated indexes.
    lexicalEvidence: bool


class DensePhaseTiming(BaseModel):
    startMs: float = Field(ge=0, description='Offset from the start of Dense, in milliseconds.')
    durationMs: float = Field(ge=0, description='Elapsed milliseconds for this phase.')


class EmbeddingPhaseTiming(BaseModel):
    startMs: float = Field(
        ge=0, description='Offset from the start of query embedding, in milliseconds.'
    )
    durationMs: float = Field(ge=0, description='Elapsed milliseconds for this phase.')


class EmbeddingStagesTiming(BaseModel):
    tokenization: EmbeddingPhaseTiming
    model: EmbeddingPhaseTiming
    cpuConversion: EmbeddingPhaseTiming


class DenseStagesTiming(BaseModel):
    embedding: DensePhaseTiming
    faiss: DensePhaseTiming
    classification: DensePhaseTiming
    embeddingStages: EmbeddingStagesTiming | None = None


class SearchTimings(BaseModel):
    denseMs: float = Field(ge=0, description='Tiered retrieval in milliseconds.')
    annotateMs: float = Field(ge=0, description='Hit mapping and annotation in milliseconds.')
    totalMs: float = Field(
        ge=0, description='Search pipeline in milliseconds, before serialization.'
    )
    denseStages: DenseStagesTiming | None = None
    admissionMs: float | None = Field(default=None, ge=0)
    rankWaitMs: float | None = Field(default=None, ge=0)


class SearchResponse(BaseModel):
    query: str
    k: int = Field(ge=1, le=MAX_K)
    model: str | None
    # Index-update time of the serving generation (id_map `updated_at`);
    # dual-owned with SemanticSidecarSearchResponse in
    # backend/src/modules/semantic-search/semantic-search.types.ts (pinned
    # by the contract spec there).
    lastUpdateAt: str | None
    # The window size above the absolute floor distinguishes evidence filtering
    # from index exhaustion. Lower-ranked hits cannot pass the floor either.
    # Null identifies the legacy cosine-only tiers.
    candidateCount: int | None = None
    results: list[SearchHit]
    weakResults: list[SearchHit]
    timings: SearchTimings


def _hit(
    result: Any,
    query: str,
    chunks_by_id: dict,
) -> SearchHit:
    """Map one engine SearchResult onto the wire hit shape.

    `chunks_by_id` is the serving searcher's chunk lookup; hits whose chunk
    is unknown (older test doubles) report no lexical evidence.
    """
    chunk = chunks_by_id.get(result.chunk_id)
    evidence = chunk is not None and has_lexical_evidence(query, chunk)
    return SearchHit(
        chunkId=result.chunk_id,
        noticeNum=result.notice_num,
        subject=result.subject,
        committee=result.committee,
        section=result.section,
        score=result.score,
        text=result.text,
        lexicalEvidence=evidence,
    )


def log_startup_config() -> None:
    """Log the env-injected system settings once at process startup.

    Runs at the top of `lifespan` (not per request, not per reload), so the
    operator can see exactly which values this process booted with. Values
    come from `lawcast_semantic.config` (env `LAWCAST_SEMANTIC_*` + optional
    `.env` file); the resolved run device is logged separately when the
    embedding model finishes loading.
    """
    service_logger.info(
        'semantic config model=%s device_request=%s batch=%s artifacts=%s '
        'queryLen=%s maxK=%s defaultK=%s minSim=%s clearSim=%s '
        'update=%s/%s',
        config.MODEL_NAME,
        config.DEVICE,
        config.EMBED_BATCH_SIZE,
        config.ARTIFACTS_DIR,
        config.MAX_QUERY_LENGTH,
        config.MAX_K,
        config.DEFAULT_K,
        config.MIN_SIMILARITY,
        config.CLEAR_SIMILARITY,
        'db' if config.DB_PATH else 'no-db',
        config.UPDATE_CRON if config.UPDATE_CRON else 'no-schedule',
    )


def load_engine(state: EngineState, boot_repair: BootRepairHook | None = None) -> None:
    """Load the embedding model + artifacts and register the searcher.

    Two phases with separate handlers (design §6.1): a model-load failure can
    never be repaired, so it goes straight to `failed`; an artifact-load
    failure runs the boot-repair hook when a DB path is configured.
    Discrimination is *which phase raised*, never the exception type.
    Importing the model stack inside the phases keeps `import service.app`
    light (no torch/faiss at module import) so validation paths and tests do
    not pay for the model stack. Readiness is logged without polling `/health`.
    """
    started = time.perf_counter()
    # Phase 1 — embedding model. Nothing downstream can succeed without it
    # (no vectors to embed, no set for reload() to validate): no repair hook.
    try:
        from lawcast_semantic import KoreanEmbedder

        embedder = KoreanEmbedder()
        state.record_device(getattr(embedder, 'device', None))
    except Exception as exc:  # noqa: BLE001 - any load failure must degrade cleanly
        state.mark_failed(f'{type(exc).__name__}: {exc}')
        service_logger.warning(
            'dense engine failed (model phase, %.1fs): %s', time.perf_counter() - started, exc
        )
        return

    # Phase 2 — artifacts. Failures here (validation gates or missing/corrupt
    # files) are the only repair candidates, and only when a DB path is
    # configured (scheduling disabled => behavior exactly as before).
    try:
        from lawcast_semantic import SemanticSearcher
        from lawcast_semantic.chunking import compute_chunks_fingerprint

        searcher = SemanticSearcher.load(embedder)
        # Record what the load validated so the first tick can tell "disk
        # changed" from "still in sync" (§5.3); content hash, order-free.
        fingerprint = compute_chunks_fingerprint(searcher.chunks_by_id.values())
    except Exception as exc:  # noqa: BLE001 - any load failure must degrade cleanly
        error = f'{type(exc).__name__}: {exc}'
        repaired = False
        if boot_repair is not None and config.DB_PATH:
            try:
                repaired = boot_repair(embedder, config.DB_PATH)
            except Exception as repair_exc:  # noqa: BLE001 - never trap the boot thread
                error += f'; boot repair failed: {type(repair_exc).__name__}: {repair_exc}'
        if not repaired:
            state.mark_failed(error)
            service_logger.warning(
                'dense engine failed (artifact phase, %.1fs): %s',
                time.perf_counter() - started,
                error,
            )
            return
        try:
            # The hook fixed the on-disk set: finish the normal load path.
            searcher = SemanticSearcher.load(embedder)
            fingerprint = compute_chunks_fingerprint(searcher.chunks_by_id.values())
        except Exception as repair_exc:  # noqa: BLE001 - re-load can also fail
            state.mark_failed(f'{type(repair_exc).__name__}: {repair_exc}')
            service_logger.warning(
                'dense engine failed (repair reload, %.1fs): %s',
                time.perf_counter() - started,
                repair_exc,
            )
            return
    state.mark_ready(searcher, embedder.model_name, fingerprint)
    service_logger.info(
        'dense engine ready (%s on %s, requested %s, %d chunks, %.1fs)',
        embedder.model_name,
        getattr(embedder, 'device', None),
        getattr(embedder, 'requested_device', None),
        len(searcher.chunk_ids),
        time.perf_counter() - started,
    )


def create_state() -> EngineState:
    return EngineState()


STATE = create_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Function-level import: `import service.app` stays light (§6.1) even
    # though the runner module is part of this service.
    from . import update_runner

    app.state.mps_admission = (
        CapacityLimiter(config.MPS_QUERY_CONCURRENCY) if config.MPS_QUERY_CONCURRENCY else None
    )
    app.state.query_lanes = None
    if config.QUERY_RANK_WORKERS:
        embed = config.QUERY_EMBED_WORKERS
        mps = min(embed, config.MPS_QUERY_CONCURRENCY or embed)
        app.state.query_lanes = (
            CapacityLimiter(embed),
            CapacityLimiter(mps),
            CapacityLimiter(config.QUERY_RANK_WORKERS),
        )
    log_startup_config()
    # The hook is passed unconditionally; load_engine invokes it only on the
    # searcher phase and only when config.DB_PATH is set (the §4.2 off gate).
    threading.Thread(
        target=load_engine,
        args=(STATE, update_runner.run_boot_repair),
        daemon=True,
        name='semantic-engine-load',
    ).start()
    if config.DB_PATH and config.UPDATE_CRON:
        threading.Thread(
            target=update_runner.start_scheduler,
            args=(STATE,),
            daemon=True,
            name='semantic-update-scheduler',
        ).start()
    grpc_server = None
    if bind := os.environ.get('LAWCAST_SEMANTIC_GRPC_BIND', '').strip():
        from .grpc_server import start_server

        grpc_server = await start_server(app, bind)
    try:
        yield
    finally:
        if grpc_server is not None:
            await grpc_server.stop(grace=5)


app = FastAPI(title='LawCast Semantic Search', lifespan=lifespan)


@app.get('/health')
def health() -> dict:
    snapshot = STATE.snapshot()
    return {
        'status': snapshot['status'],
        'model': snapshot['model'],
        # Embedding device the engine actually runs on (auto-detected
        # accelerator, or cpu) — additive field, existing consumers read
        # `status` only.
        'device': snapshot['device'],
        'indexedChunks': snapshot['indexedChunks'],
        'error': snapshot['error'],
        'reloadError': snapshot['reloadError'],
        'lastUpdateAt': snapshot['lastUpdateAt'],
        'lastUpdateResult': snapshot['lastUpdateResult'],
        'lastUpdateError': snapshot['lastUpdateError'],
        'lastUpdateTriggeredAt': snapshot['lastUpdateTriggeredAt'],
        'updating': snapshot['updating'],
        'generation': snapshot['generation'],
        # Chunk-window cap this sidecar enforces on `/search` (unit: chunks,
        # not notices). The backend reads it here instead of hardcoding the
        # cap, so changing LAWCAST_SEMANTIC_MAX_K only needs a sidecar
        # restart — the field name is pinned against the backend's
        # SemanticSidecarHealthResponse by the contract spec there.
        'maxK': config.MAX_K,
    }


@app.post('/reload')
def manual_reload() -> dict:
    """Manual load-validate-swap trigger (design §5.2.1) — hot reload, no restart.

    Reuses the scheduler's swap path (`EngineState.reload` under the artifact
    lock), so ops and host-pipeline runs (`01→03` on the host, then one curl)
    get the same atomic generation swap. 409 unless the engine is ready and no
    update is in flight; 503 when the new set fails validation — the old
    generation keeps serving (§5.2.4).
    """
    from . import update_runner

    snapshot = STATE.snapshot()
    if snapshot['status'] != 'ready':
        raise HTTPException(
            status_code=409,
            detail=f'reload requires a ready engine (status={snapshot["status"]})',
        )
    result = update_runner.run_reload(STATE, snapshot['searcher'].embedder)
    if result == 'skipped':
        raise HTTPException(status_code=409, detail='an index update is already in progress')
    if result == 'failed':
        raise HTTPException(
            status_code=503,
            detail=f'semantic reload failed: {STATE.snapshot()["reloadError"]}',
        )
    return health()


@app.get('/search', response_model=SearchResponse)
async def search_http(
    request: Request,
    query: str = Query(..., min_length=1, max_length=MAX_QUERY_LENGTH),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
) -> SearchResponse:
    with search_trace('http', request.headers):
        return await search_with_admission(request.app, query, k)


async def search_with_admission(app: FastAPI, query: str, k: int) -> SearchResponse:
    lanes = getattr(app.state, 'query_lanes', None)
    if lanes is not None:
        return await _search_with_worker_lanes(query, k, lanes)
    limiter = getattr(app.state, 'mps_admission', None)
    snapshot = _search_snapshot(query)
    started = time.perf_counter()

    def dispatch(on_embedding_finished=None):
        # Include both admission and AnyIO worker dispatch, measured at entry
        # to the worker rather than when the event loop submits it.
        return _search_response(
            query,
            k,
            snapshot,
            admission_started=started,
            admission_ms=(time.perf_counter() - started) * 1000,
            on_embedding_finished=on_embedding_finished,
        )

    if limiter is None or snapshot['device'] != 'mps':
        return await run_in_threadpool(dispatch)
    borrower = object()
    # Wait before borrowing an AnyIO worker, so queued searches do not starve
    # health checks or FAISS work. Active native work is not abandoned on cancellation.
    await limiter.acquire_on_behalf_of(borrower)
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            limiter.release_on_behalf_of(borrower)
            released = True

    try:
        return await run_in_threadpool(dispatch, lambda: from_thread.run_sync(release))
    finally:
        release()


async def _search_with_worker_lanes(
    query: str,
    k: int,
    lanes: tuple[CapacityLimiter, CapacityLimiter, CapacityLimiter],
) -> SearchResponse:
    # Keep embedding, ranking and annotation on a single index generation.
    snapshot = _search_snapshot(query)
    searcher = snapshot['searcher']
    embed, mps, rank = lanes
    started = time.perf_counter()

    def embed_work():
        admission_ms = (time.perf_counter() - started) * 1000
        return searcher.embed_tiered(query, k), admission_ms

    # Each lane acquires a capacity token before starting a worker. Requests
    # waiting for FAISS never retain an embedding token or an AnyIO worker.
    prepared, admission_ms = await to_thread.run_sync(
        embed_work, limiter=mps if snapshot['device'] == 'mps' else embed
    )
    rank_queued = time.perf_counter()

    def rank_work():
        rank_wait_ms = (time.perf_counter() - rank_queued) * 1000
        outcome = searcher.rank_tiered(query, prepared, k)
        return _format_search_response(
            query, k, snapshot, outcome, started, admission_ms, rank_wait_ms
        )

    return await to_thread.run_sync(rank_work, limiter=rank)


def search(query: str, k: int = DEFAULT_K) -> SearchResponse:
    """Synchronous search entry point for local tools and benchmarks."""
    return _search_response(query, k, _search_snapshot(query))


def _search_snapshot(query: str) -> dict:
    if not query.strip():
        raise HTTPException(status_code=400, detail='query must not be blank')
    snapshot = STATE.snapshot()
    if snapshot['status'] == 'loading':
        raise HTTPException(status_code=503, detail='semantic engine is still loading')
    if snapshot['status'] == 'failed':
        raise HTTPException(
            status_code=503,
            detail=f'semantic engine unavailable: {snapshot["error"]}',
        )
    return snapshot


def _search_response(
    query: str,
    k: int,
    snapshot: dict,
    *,
    admission_started: float | None = None,
    admission_ms: float = 0.0,
    on_embedding_finished: Callable[[], None] | None = None,
) -> SearchResponse:
    started = time.perf_counter() if admission_started is None else admission_started
    searcher = snapshot['searcher']
    if on_embedding_finished is None:
        outcome = searcher.search_tiered(query, k=k)
    else:
        outcome = searcher.search_tiered(
            query,
            k=k,
            on_embedding_finished=on_embedding_finished,
        )
    return _format_search_response(query, k, snapshot, outcome, started, admission_ms)


def _format_search_response(
    query: str,
    k: int,
    snapshot: dict,
    outcome: SearchResults,
    started: float,
    admission_ms: float,
    rank_wait_ms: float = 0.0,
) -> SearchResponse:
    dense_finished = time.perf_counter()
    chunks_by_id = getattr(snapshot['searcher'], 'chunks_by_id', None) or {}
    results = [_hit(result, query, chunks_by_id) for result in outcome.results]
    weak_results = [_hit(result, query, chunks_by_id) for result in outcome.weak_results]
    finished = time.perf_counter()
    stages = getattr(outcome, 'dense_stages', None)

    def phase_timing(phase: DensePhase) -> DensePhaseTiming:
        return DensePhaseTiming(
            startMs=(phase.started - started) * 1000.0,
            durationMs=(phase.finished - phase.started) * 1000.0,
        )

    embedding = stages.embedding_stages if stages is not None else None

    def embedding_phase(phase: DensePhase) -> EmbeddingPhaseTiming:
        return EmbeddingPhaseTiming(
            startMs=(phase.started - stages.embedding.started) * 1000.0,
            durationMs=(phase.finished - phase.started) * 1000.0,
        )

    timings = SearchTimings(
        denseMs=(dense_finished - started) * 1000.0,
        annotateMs=(finished - dense_finished) * 1000.0,
        totalMs=(finished - started) * 1000.0,
        admissionMs=admission_ms,
        rankWaitMs=rank_wait_ms,
        denseStages=DenseStagesTiming(
            embedding=phase_timing(stages.embedding),
            faiss=phase_timing(stages.faiss),
            classification=phase_timing(stages.classification),
            embeddingStages=EmbeddingStagesTiming(
                tokenization=embedding_phase(embedding.tokenization),
                model=embedding_phase(embedding.model),
                cpuConversion=embedding_phase(embedding.cpu_conversion),
            )
            if embedding is not None
            else None,
        )
        if stages is not None
        else None,
    )
    service_logger.info(
        'search-timing k=%d dense_ms=%.1f annotate_ms=%.1f total_ms=%.1f '
        'admission_ms=%.1f rank_wait_ms=%.1f transport=%s trace_id=%s attempt=%s',
        k,
        timings.denseMs,
        timings.annotateMs,
        timings.totalMs,
        admission_ms,
        rank_wait_ms,
        *trace_fields(),
    )
    return SearchResponse(
        query=query,
        k=k,
        model=snapshot['model'],
        lastUpdateAt=snapshot['lastUpdateAt'],
        candidateCount=outcome.candidate_count,
        results=results,
        weakResults=weak_results,
        timings=timings,
    )
