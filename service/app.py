"""Online adapter: HTTP sidecar exposing the semantic search library.

Run from the side project root (the engine loads artifacts via
`lawcast_semantic.config` defaults):

    .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8300

The LawCast backend calls this service over HTTP (same integration style as
its Ollama client) and falls back to keyword search when this service answers
503 or is unreachable. The engine loads once in a background thread at
startup; while it loads, `/search` answers 503 immediately instead of making
requests wait. A failed load is reported through `/health` and `/search` until
the process is restarted.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from lawcast_semantic import config  # env-only module: import stays light
from lawcast_semantic.hybrid import has_lexical_evidence
from lawcast_semantic.omp_env import use_single_threaded_omp

use_single_threaded_omp()

# Tick trigger/result logs are INFO records (run_update_cycle); uvicorn's log
# config leaves the root logger at WARNING with no root handler, which would
# silently drop them in the real server process (found by driving a live
# uvicorn instance). The `service` package carries its own level + handler;
# propagation stays on so pytest caplog and any root config still see them.
service_logger = logging.getLogger('service')
service_logger.setLevel(logging.INFO)
if not service_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    service_logger.addHandler(_handler)

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
        # Cross-encoder shadow scorer (§ rerank): loaded after the dense
        # engine is ready; None while disabled (`off`) or not yet loaded.
        self.reranker: Any = None
        self.rerank_error: str | None = None

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
                'reranker': self.reranker,
                'rerankError': self.rerank_error,
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

    def record_reranker(self, reranker: Any) -> None:
        """Register the loaded shadow scorer (serving never waits for it)."""
        with self._lock:
            self.reranker = reranker
            self.rerank_error = None

    def record_rerank_error(self, error: str) -> None:
        """A failed reranker load degrades to unscored shadow, never to `failed`."""
        with self._lock:
            self.rerank_error = error

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
    # no-result gate reads this flag; it never filters inside the sidecar so
    # chunk-window widening keeps working (see contract spec).
    lexicalEvidence: bool
    # Cross-encoder relevance logit for this chunk (`reranking.Reranker`),
    # scored over the served window's top hits when the shadow scorer is
    # loaded. None while the scorer is disabled or absent: the backend's
    # gate treats unknown as unscored (fail-open) and never as a reject.
    rerankScore: float | None = None


class SearchResponse(BaseModel):
    query: str
    k: int = Field(ge=1, le=MAX_K)
    model: str | None
    # Index-update time of the serving generation (id_map `updated_at`);
    # dual-owned with SemanticSidecarSearchResponse in
    # backend/src/modules/semantic-search/semantic-search.types.ts (pinned
    # by the contract spec there).
    lastUpdateAt: str | None
    # Clear hits (score >= CLEAR_SIMILARITY) and the weak band between
    # MIN_SIMILARITY and CLEAR_SIMILARITY; unrelated hits below the floor
    # are dropped by the engine and appear in neither list.
    results: list[SearchHit]
    weakResults: list[SearchHit]


def _hit(
    result: Any,
    query: str,
    chunks_by_id: dict,
    rerank_scores: dict[str, float] | None = None,
) -> SearchHit:
    """Map one engine SearchResult onto the wire hit shape.

    `chunks_by_id` is the serving searcher's chunk lookup; hits whose chunk
    is unknown (older test doubles) report no lexical evidence.
    `rerank_scores` maps chunk ids to cross-encoder logits (None while the
    shadow scorer is disabled); unscored hits report None, never a number.
    """
    chunk = chunks_by_id.get(result.chunk_id)
    evidence = chunk is not None and has_lexical_evidence(query, chunk)
    rerank_score = rerank_scores.get(result.chunk_id) if rerank_scores is not None else None
    return SearchHit(
        chunkId=result.chunk_id,
        noticeNum=result.notice_num,
        subject=result.subject,
        committee=result.committee,
        section=result.section,
        score=result.score,
        text=result.text,
        lexicalEvidence=evidence,
        rerankScore=rerank_score,
    )


def load_engine(state: EngineState, boot_repair: BootRepairHook | None = None) -> None:
    """Load the embedding model + artifacts and register the searcher.

    Two phases with separate handlers (design §6.1): a model-load failure can
    never be repaired, so it goes straight to `failed`; an artifact-load
    failure runs the boot-repair hook when a DB path is configured.
    Discrimination is *which phase raised*, never the exception type.
    Importing the model stack inside the phases keeps `import service.app`
    light (no torch/faiss at module import) so validation paths and tests do
    not pay for the model stack.
    """
    # Phase 1 — embedding model. Nothing downstream can succeed without it
    # (no vectors to embed, no set for reload() to validate): no repair hook.
    try:
        from lawcast_semantic import KoreanEmbedder

        embedder = KoreanEmbedder()
        state.record_device(getattr(embedder, 'device', None))
    except Exception as exc:  # noqa: BLE001 - any load failure must degrade cleanly
        state.mark_failed(f'{type(exc).__name__}: {exc}')
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
            return
        try:
            # The hook fixed the on-disk set: finish the normal load path.
            searcher = SemanticSearcher.load(embedder)
            fingerprint = compute_chunks_fingerprint(searcher.chunks_by_id.values())
        except Exception as repair_exc:  # noqa: BLE001 - re-load can also fail
            state.mark_failed(f'{type(repair_exc).__name__}: {repair_exc}')
            return
    state.mark_ready(searcher, embedder.model_name, fingerprint)
    maybe_load_reranker(state)


def maybe_load_reranker(state: EngineState) -> None:
    """Load the cross-encoder shadow scorer when `RERANK_MODE` enables it.

    Runs after the dense engine is ready so readiness never waits on the
    reranker; any failure degrades to unscored shadow and is recorded for
    the logs. The model stack is imported here, keeping module import light.
    """
    if config.RERANK_MODE == 'off':
        return
    try:
        from lawcast_semantic.reranking import load_reranker

        state.record_reranker(load_reranker())
        service_logger.info('rerank shadow scorer ready (%s)', config.RERANK_MODEL)
    except Exception as exc:  # noqa: BLE001 - shadow must never break serving
        state.record_rerank_error(f'{type(exc).__name__}: {exc}')
        service_logger.warning('rerank shadow scorer unavailable: %s', exc)


def score_rerank_window(reranker: Any, query: str, hits: list) -> dict[str, float]:
    """Score the served window's top hits and log the scores (no filtering).

    `hits` arrive in served rank order (clear tier then weak band); only the
    top `RERANK_TOP_K` are scored to bound the added CPU work. Returns a
    chunk-id -> logit map for the wire annotation; a scoring failure logs
    once and returns an empty map so the response still serves unscored.
    """
    from lawcast_semantic.reranking import shadow_inputs

    try:
        started = time.perf_counter()
        window = list(hits)[: config.RERANK_TOP_K]
        texts = shadow_inputs(query, window)
        if not texts:
            return {}
        scores = reranker.score(query, texts)
        rerank_ms = (time.perf_counter() - started) * 1000.0
        top = sorted(scores, reverse=True)[:5]
        service_logger.info(
            'rerank-shadow n=%d top1=%.3f top5=%s rerank_ms=%.1f',
            len(scores),
            top[0],
            ','.join(f'{value:.3f}' for value in top),
            rerank_ms,
        )
        return {hit.chunk_id: score for hit, score in zip(window, scores)}
    except Exception as exc:  # noqa: BLE001 - shadow must never break serving
        service_logger.warning('rerank-shadow scoring failed: %s', exc)
        return {}


def create_state() -> EngineState:
    return EngineState()


STATE = create_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Function-level import: `import service.app` stays light (§6.1) even
    # though the runner module is part of this service.
    from . import update_runner

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
    yield


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
def search(
    query: str = Query(..., min_length=1, max_length=MAX_QUERY_LENGTH),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
    rerank: bool | None = Query(
        default=None,
        description=(
            'Ask for cross-encoder logits on the served window. None (older '
            'backend) keeps the legacy behavior: score whenever the shadow '
            'scorer is loaded. Explicit false skips the scoring pass so '
            'callers that never consume `rerankScore` avoid ~1s of CPU.'
        ),
    ),
) -> SearchResponse:
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
    started = time.perf_counter()
    searcher = snapshot['searcher']
    outcome = searcher.search_tiered(query, k=k)
    dense_ms = (time.perf_counter() - started) * 1000.0
    chunks_by_id = getattr(searcher, 'chunks_by_id', None) or {}
    # The sidecar annotates, never filters (§1 of the gate plan): per-hit
    # rerank logits ride along so the backend can gate post-accumulation
    # without faking window exhaustion. None while the scorer is off/absent
    # or the caller opts out (`rerank=false`: the backend sends it when its
    # rerank leg is disabled and no score is consumed downstream).
    rerank_scores: dict[str, float] | None = None
    rerank_ms = 0.0
    want_rerank = True if rerank is None else rerank
    if want_rerank and config.RERANK_MODE != 'off' and snapshot.get('reranker') is not None:
        rerank_started = time.perf_counter()
        rerank_scores = score_rerank_window(
            snapshot['reranker'],
            query,
            [*outcome.results, *outcome.weak_results],
        )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
    annotate_started = time.perf_counter()
    response = SearchResponse(
        query=query,
        k=k,
        model=snapshot['model'],
        lastUpdateAt=snapshot['lastUpdateAt'],
        results=[_hit(result, query, chunks_by_id, rerank_scores) for result in outcome.results],
        weakResults=[
            _hit(result, query, chunks_by_id, rerank_scores) for result in outcome.weak_results
        ],
    )
    annotate_ms = (time.perf_counter() - annotate_started) * 1000.0
    service_logger.info(
        'search-timing k=%d dense_ms=%.1f annotate_ms=%.1f rerank_ms=%.1f total_ms=%.1f rerank=%s',
        k,
        dense_ms,
        annotate_ms,
        rerank_ms,
        (time.perf_counter() - started) * 1000.0,
        'scored' if rerank_scores is not None else 'skipped',
    )
    return response
