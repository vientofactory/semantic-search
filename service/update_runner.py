"""Scheduled index refresh runner (design §2/§4/§5/§6 of the pipeline design).

Owns everything the sidecar does on a tick: read the backend DB through the
mode=ro datasource, diff it against the committed artifacts, apply the delta
through incremental's atomic write order, then load-validate-swap the serving
searcher (`EngineState.reload`). Also exports the boot-repair hook that
`load_engine` invokes on a phase-2 (artifact) failure.

All heavy imports stay function-level: `lawcast_semantic.incremental` pulls
faiss at module import and `import service.app` must stay light (§6.1).
Configuration gates live in `lawcast_semantic.config` (§4.2): an empty
DB_PATH turns scheduling and boot repair off entirely.
"""

from __future__ import annotations

import fcntl
import json
import logging
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any

from lawcast_semantic import config

if TYPE_CHECKING:
    from lawcast_semantic.incremental import UpdatePlan

    from .app import EngineState

logger = logging.getLogger(__name__)


@contextmanager
def _update_lock() -> Iterator[None]:
    """Hold the cross-process artifact lock (§6.2).

    Ticks and manual `flock`-guarded runs never interleave. Non-blocking: a
    held lock raises BlockingIOError for the caller to classify.
    """
    with open(config.ARTIFACTS_DIR / '.update.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _plan_for(db_path: str | Path, model_name: str) -> UpdatePlan:
    """Read the corpus (mode=ro, one connection) and diff it against the committed artifacts."""
    from lawcast_semantic.datasource import load_notices_from_db
    from lawcast_semantic.incremental import plan_update

    notices = load_notices_from_db(Path(db_path))
    return plan_update(
        notices,
        chunks_path=config.CHUNKS_PATH,
        embeddings_path=config.EMBEDDINGS_PATH,
        model_name=model_name,
    )


def _shrink_refused(plan: UpdatePlan) -> bool:
    """§6 shrink guard: refuse deletions over 20% AND over 100 notices.

    |old notices| = |new| − added + deleted (set identity), so the ratio
    needs no second read of chunks.jsonl. ALLOW_LARGE_DELETE overrides (§4.2).
    """
    if config.ALLOW_LARGE_DELETE:
        return False
    deleted = len(plan.deleted_notices)
    old_notices = (
        len({record['notice_num'] for record in plan.new_records})
        + deleted
        - len(plan.added_notices)
    )
    return deleted > 100 and deleted * 5 > old_notices


def _apply(plan: UpdatePlan, embed_texts, model_name: str):
    """Write through incremental's atomic order (npz → id_map → index → chunks)."""
    from lawcast_semantic.incremental import apply_update

    return apply_update(
        plan,
        embed_texts,
        model_name,
        chunks_path=config.CHUNKS_PATH,
        embeddings_path=config.EMBEDDINGS_PATH,
        index_path=config.FAISS_INDEX_PATH,
        id_map_path=config.ID_MAP_PATH,
    )


def _disk_fingerprint() -> str | None:
    """Set fingerprint committed in id_map.json; None when unreadable (= mismatch)."""
    try:
        payload = json.loads(config.ID_MAP_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    fingerprint = payload.get('chunks_fingerprint')
    return str(fingerprint) if fingerprint else None


def run_boot_repair(embedder: Any, db_path: str | Path) -> bool:
    """Boot-repair hook for `load_engine` phase 2 (§6.1).

    Runs one cycle with the already-loaded embedder and applies iff no
    embedding is needed: a torn set always plans with 0 embeds (the npz is
    write #1 and already carries the rows), while a plan needing vectors has
    no usable baseline — the manual bootstrap stays authoritative — and a
    guarded shrink means the mounted DB is suspect. Returns whether artifacts
    were actually rewritten (a no-op cycle repairs nothing, so the caller
    keeps its original error); exceptions propagate to the load thread, which
    contains them and fails the boot.
    """
    with _update_lock():
        plan = _plan_for(db_path, embedder.model_name)
        if plan.needs_embedding or _shrink_refused(plan):
            return False
        return _apply(plan, None, embedder.model_name).wrote_artifacts


def _tick(state: EngineState, embedder: Any) -> tuple[str, str | None]:
    plan = _plan_for(config.DB_PATH, embedder.model_name)
    if _shrink_refused(plan):
        error = (
            f'shrink guard: refusing to delete {len(plan.deleted_notices)} notices; '
            'check the DB, then set LAWCAST_SEMANTIC_ALLOW_LARGE_DELETE=true to override'
        )
        logger.error(error)
        return 'failed', error
    if plan.has_changes:
        _apply(plan, embedder.embed_texts if plan.needs_embedding else None, embedder.model_name)
    elif (disk := _disk_fingerprint()) is not None and disk == state.loaded_fingerprint:
        return 'unchanged', None  # memory and disk agree (§5.3)
    if not state.reload(embedder):
        return 'failed', state.reload_error  # old generation still serves (§5.2)
    return 'changed', None


def run_reload(state: EngineState, embedder: Any) -> str:
    """Manual load-validate-swap behind POST /reload (design §5.2.1).

    The scheduler's exact swap path (`EngineState.reload`), taken under the
    artifact lock so a reload can never read a half-written artifact set.
    Returns 'changed' (new generation serving), 'failed' (old generation
    keeps serving; see `reloadError`), or 'skipped' (a tick or a manual run
    holds the lock). Manual reloads leave the lastUpdate* tick fields
    untouched.
    """
    try:
        with _update_lock():
            return 'changed' if state.reload(embedder) else 'failed'
    except BlockingIOError:
        logger.warning('manual reload skipped: another holder owns .update.lock')
        return 'skipped'


def run_update_cycle(state: EngineState, embedder: Any) -> str:
    """One scheduled tick (§2) under the artifact lock; records the outcome.

    Returns the §5.2 result vocabulary: 'changed' (artifacts written and
    swapped in, or a stale memory reloaded), 'unchanged', 'failed', or
    'skipped' (a manual run holds the lock).
    """
    try:
        with _update_lock():
            result, error = _tick(state, embedder)
    except BlockingIOError:
        result, error = 'skipped', None
        logger.warning('update tick skipped: another holder owns .update.lock')
    except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the scheduler
        result, error = 'failed', f'{type(exc).__name__}: {exc}'
        logger.warning('update tick failed: %s', error)
    state.record_update(result, error)
    return result


def start_scheduler(state: EngineState) -> None:
    """Lifespan tick loop (§4.2): interval ±10% jitter, ready-gated.

    Sequential by construction (single-flight): one sleep → check → tick at a
    time. `loading` waits out the next sleep; `failed` stops the loop — boot
    repair is the only recovery path for a failed boot.
    """
    interval = config.UPDATE_INTERVAL_MINUTES * 60
    while True:
        sleep(interval * random.uniform(0.9, 1.1))
        snapshot = state.snapshot()
        if snapshot['status'] == 'failed':
            return
        if snapshot['status'] == 'ready':
            run_update_cycle(state, snapshot['searcher'].embedder)
