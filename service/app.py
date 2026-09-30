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
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from lawcast_semantic.omp_env import use_single_threaded_omp

use_single_threaded_omp()

MAX_QUERY_LENGTH = 500
MAX_K = 50
DEFAULT_K = 5


class EngineState:
    """Process-wide engine holder with single-writer locking.

    Owns the loaded searcher; the HTTP handlers only take snapshots so a
    long model load never blocks a request.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: Literal['loading', 'ready', 'failed'] = 'loading'
        self.searcher: Any = None
        self.model_name: str | None = None
        self.indexed_chunks: int = 0
        self.error: str | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'status': self.status,
                'searcher': self.searcher,
                'model': self.model_name,
                'indexedChunks': self.indexed_chunks,
                'error': self.error,
            }

    def mark_ready(self, searcher: Any, model_name: str) -> None:
        with self._lock:
            self.searcher = searcher
            self.model_name = model_name
            self.indexed_chunks = len(searcher.chunk_ids)
            self.status = 'ready'
            self.error = None

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = 'failed'
            self.error = error


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


def load_engine(state: EngineState) -> None:
    """Load the embedding model + artifacts and register the searcher.

    Importing the model stack here keeps `import service.app` light (no torch)
    so validation paths and tests do not pay for the model stack.
    """
    try:
        from lawcast_semantic import KoreanEmbedder, SemanticSearcher

        embedder = KoreanEmbedder()
        searcher = SemanticSearcher.load(embedder)
        state.mark_ready(searcher, embedder.model_name)
    except Exception as exc:  # noqa: BLE001 - any load failure must degrade cleanly
        state.mark_failed(f'{type(exc).__name__}: {exc}')


def create_state() -> EngineState:
    return EngineState()


STATE = create_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(
        target=load_engine,
        args=(STATE,),
        daemon=True,
        name='semantic-engine-load',
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
    }


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
