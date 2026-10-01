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

import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from lawcast_semantic import config  # env-only module: import stays light
from lawcast_semantic.omp_env import use_single_threaded_omp

use_single_threaded_omp()

# Boot-repair hook contract (design §6.1): (embedder, db_path) -> whether the
# on-disk artifact set was repaired. The update runner supplies the real one;
# None (default) means no repair capability, matching pre-design behavior.
BootRepairHook = Callable[[Any, Path], bool]

MAX_QUERY_LENGTH = 500
# `/search` k counts CHUNKS, not notices: the backend over-requests chunks so
# chunk->notice dedup can still fill its notice-level k. Must match
# SIDE_CAR_MAX_CHUNK_K in
# backend/src/modules/semantic-search/semantic-search.constants.ts (pinned by
# the contract test there).
MAX_K = 200
DEFAULT_K = 5


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
        self.indexed_chunks: int = 0
        self.error: str | None = None
        # §5.2 observability: what serves now, what the last tick did.
        self.generation = 0
        self.loaded_fingerprint: str | None = None
        self.reload_error: str | None = None
        self.last_update_at: str | None = None
        self.last_update_result: str | None = None  # 'changed'|'unchanged'|'failed'|'skipped'
        self.last_update_error: str | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'status': self.status,
                'searcher': self.searcher,
                'model': self.model_name,
                'indexedChunks': self.indexed_chunks,
                'error': self.error,
                'generation': self.generation,
                'reloadError': self.reload_error,
                'lastUpdateAt': self.last_update_at,
                'lastUpdateResult': self.last_update_result,
                'lastUpdateError': self.last_update_error,
            }

    def mark_ready(self, searcher: Any, model_name: str, fingerprint: str | None = None) -> None:
        """First successful load (generation 1).

        `fingerprint` is the set identity the load validated; None means
        unknown, which the first tick treats as a mismatch and verifies with
        one reload (§5.3).
        """
        with self._lock:
            self.searcher = searcher
            self.model_name = model_name
            self.indexed_chunks = len(searcher.chunk_ids)
            self.status = 'ready'
            self.error = None
            self.generation += 1
            self.loaded_fingerprint = fingerprint

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
        return True

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = 'failed'
            self.error = error

    def record_update(self, result: str, error: str | None = None) -> None:
        """Record one update tick's outcome for §5.2's additive /health fields."""
        with self._lock:
            self.last_update_at = datetime.now(UTC).isoformat()
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


class SearchResponse(BaseModel):
    query: str
    k: int = Field(ge=1, le=MAX_K)
    model: str | None
    results: list[SearchHit]


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
    if config.DB_PATH and config.UPDATE_INTERVAL_MINUTES > 0:
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
        'indexedChunks': snapshot['indexedChunks'],
        'error': snapshot['error'],
        # §5.2 additive fields: existing consumers read `status` only.
        'reloadError': snapshot['reloadError'],
        'lastUpdateAt': snapshot['lastUpdateAt'],
        'lastUpdateResult': snapshot['lastUpdateResult'],
        'lastUpdateError': snapshot['lastUpdateError'],
        'generation': snapshot['generation'],
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
    results = snapshot['searcher'].search(query, k=k)
    return SearchResponse(
        query=query,
        k=k,
        model=snapshot['model'],
        results=[
            SearchHit(
                chunkId=result.chunk_id,
                noticeNum=result.notice_num,
                subject=result.subject,
                committee=result.committee,
                section=result.section,
                score=result.score,
                text=result.text,
            )
            for result in results
        ],
    )
