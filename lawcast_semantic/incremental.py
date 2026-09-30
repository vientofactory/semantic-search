"""Incremental artifact updates: sync the artifact set with the corpus.

Full rebuilds (scripts 01-02-03) re-embed every chunk (~40 min on the full
LawCast corpus); this module embeds only chunks whose embedded text changed,
reuses stored rows otherwise, and emits exactly the artifact set a full
rebuild would produce — same files, same formats, same `SemanticSearcher.load`
contract (the set fingerprint is a content invariant, so an incrementally
built set is indistinguishable from a rebuilt one, by design).

Consistency model (see agent_memories/07-incremental-indexing/plan.md):

- The chunk set derived from the corpus (`chunk_notices`) is the identity
  source of truth. Every run re-chunks the corpus (cheap, seconds) and diffs
  it against the committed artifacts, so change detection is content-derived
  and no state file can drift or corrupt.
- Each `embeddings.npz` row carries its provenance (`chunk_text_digests`), so
  a stored vector is reusable iff its digest matches the chunk's current
  embedded text. That holds under any crash interleaving: rerunning after a
  crash embeds exactly the chunks that lack a valid vector and converges on
  the full-rebuild artifacts.
- Artifacts are written embeddings.npz -> id_map.json -> faiss.index ->
  chunks.jsonl (temp file + atomic rename each). Every partial write leaves
  either the complete old generation (load passes) or a mix that fails the
  fingerprint validation. Recovery is simply re-running the update.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunking import (
    chunk_notices,
    compose_embedding_text,
    compute_chunk_text_digest,
    compute_chunks_fingerprint,
    load_chunks_jsonl,
    write_chunks_jsonl,
)
from .indexing import VectorIndex

# Optional embeddings.npz member: per-row SHA-256 provenance (parallel to
# chunk_ids). Absent in pre-incremental artifacts, which are then verified
# through the whole-set fingerprint instead.
CHUNK_DIGESTS_MEMBER = 'chunk_text_digests'


@dataclass(frozen=True)
class UpdatePlan:
    """Minimal work needed to bring the artifacts in line with the corpus."""

    new_records: tuple[dict, ...]
    embed_records: tuple[dict, ...]
    reused_chunk_ids: tuple[str, ...]
    dropped_chunk_ids: tuple[str, ...]
    added_notices: tuple[int, ...]
    updated_notices: tuple[int, ...]
    deleted_notices: tuple[int, ...]
    records_changed: bool

    @property
    def needs_embedding(self) -> bool:
        return bool(self.embed_records)

    @property
    def has_changes(self) -> bool:
        """True when the artifacts must be rewritten (new work or repair)."""
        return self.records_changed or bool(self.embed_records) or bool(self.dropped_chunk_ids)


@dataclass(frozen=True)
class UpdateReport:
    """What one `apply_update` run did (plan summary plus results)."""

    chunks_total: int
    embedded_count: int
    reused_count: int
    dropped_count: int
    added_notices: tuple[int, ...]
    updated_notices: tuple[int, ...]
    deleted_notices: tuple[int, ...]
    chunks_fingerprint: str
    wrote_artifacts: bool


@dataclass(frozen=True)
class _StoredState:
    """Provenance extracted from embeddings.npz: rows and what they embed."""

    ids: tuple[str, ...]
    digest_by_id: dict[str, str]


def _group_by_notice(records: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record['notice_num']), []).append(record)
    return grouped


def _load_stored_state(
    embeddings_path: Path,
    old_records: list[dict],
    model_name: str,
) -> _StoredState | None:
    """Read row provenance from embeddings.npz (None when it does not exist).

    Artifacts predating `chunk_text_digests` only tie rows to texts through
    the whole-set fingerprint: when the stored fingerprint matches
    `old_records`, the rows provably embed those texts and per-chunk digests
    are derived from them. Anything unprovable raises, because reusing rows
    whose source text is unknown could serve stale vectors.
    """
    if not embeddings_path.exists():
        return None
    with np.load(embeddings_path, allow_pickle=False) as payload:
        files = payload.files
        if 'chunk_ids' not in files:
            raise ValueError(f'{embeddings_path} has no chunk_ids member; run a full rebuild')
        stored_ids = [str(chunk_id) for chunk_id in payload['chunk_ids']]
        stored_model = str(payload['model_name'].item()) if 'model_name' in files else None
        if stored_model is not None and stored_model != model_name:
            raise ValueError(
                f'embeddings.npz was built with model {stored_model!r} but this update '
                f'targets {model_name!r}; vectors from two models cannot share one index '
                '— run a full rebuild (scripts/02_extract_embeddings.py and '
                'scripts/03_build_index.py)'
            )
        if CHUNK_DIGESTS_MEMBER in files:
            digests = [str(item) for item in payload[CHUNK_DIGESTS_MEMBER]]
            if len(digests) != len(stored_ids):
                raise ValueError(
                    f'{embeddings_path} has {len(digests)} {CHUNK_DIGESTS_MEMBER} entries '
                    f'for {len(stored_ids)} rows; run a full rebuild'
                )
            return _StoredState(tuple(stored_ids), dict(zip(stored_ids, digests)))
        stored_fingerprint = (
            str(payload['chunks_fingerprint'].item()) if 'chunks_fingerprint' in files else None
        )

    if stored_fingerprint is None:
        raise ValueError(
            f'{embeddings_path} carries neither {CHUNK_DIGESTS_MEMBER} nor '
            'chunks_fingerprint provenance; run a full rebuild'
        )
    if stored_fingerprint != compute_chunks_fingerprint(old_records):
        raise ValueError(
            f'{embeddings_path} predates {CHUNK_DIGESTS_MEMBER} and its chunk set '
            f'fingerprint {stored_fingerprint[:12]} does not match chunks.jsonl; '
            'its rows cannot be tied to texts, so an incremental update would risk '
            'reusing stale vectors — run a full rebuild (scripts/01-03)'
        )
    records_by_id = {record['chunk_id']: record for record in old_records}
    derived: dict[str, str] = {}
    for chunk_id in stored_ids:
        record = records_by_id.get(chunk_id)
        if record is not None:
            derived[chunk_id] = compute_chunk_text_digest(record)
    return _StoredState(tuple(stored_ids), derived)


def plan_update(
    notices: Iterable[dict],
    *,
    chunks_path: Path = config.CHUNKS_PATH,
    embeddings_path: Path = config.EMBEDDINGS_PATH,
    model_name: str = config.MODEL_NAME,
) -> UpdatePlan:
    """Diff the corpus against the committed artifacts and plan minimal work.

    The whole corpus is re-chunked (cheap) so change detection is
    content-derived: a chunk is re-embedded exactly when its embedded text
    differs from what its stored row was built from. Notices are only a
    reporting layer here — work granularity is the chunk row.
    """
    new_records = [chunk.to_dict() for chunk in chunk_notices(notices)]
    old_records = load_chunks_jsonl(chunks_path) if chunks_path.exists() else []
    stored = _load_stored_state(embeddings_path, old_records, model_name)

    digest_by_id = stored.digest_by_id if stored else {}
    stored_ids = stored.ids if stored else ()
    pairs = [(record, compute_chunk_text_digest(record)) for record in new_records]
    embed_records = tuple(
        record for record, digest in pairs if digest_by_id.get(record['chunk_id']) != digest
    )
    reused_chunk_ids = tuple(
        record['chunk_id']
        for record, digest in pairs
        if digest_by_id.get(record['chunk_id']) == digest
    )
    new_ids = {record['chunk_id'] for record in new_records}
    dropped_chunk_ids = tuple(dict.fromkeys(cid for cid in stored_ids if cid not in new_ids))

    old_by_notice = _group_by_notice(old_records)
    new_by_notice = _group_by_notice(new_records)
    added_notices = tuple(sorted(set(new_by_notice) - set(old_by_notice)))
    deleted_notices = tuple(sorted(set(old_by_notice) - set(new_by_notice)))
    updated_notices = tuple(
        sorted(
            notice_num
            for notice_num in set(old_by_notice) & set(new_by_notice)
            if compute_chunks_fingerprint(old_by_notice[notice_num])
            != compute_chunks_fingerprint(new_by_notice[notice_num])
        )
    )

    return UpdatePlan(
        new_records=tuple(new_records),
        embed_records=embed_records,
        reused_chunk_ids=reused_chunk_ids,
        dropped_chunk_ids=dropped_chunk_ids,
        added_notices=added_notices,
        updated_notices=updated_notices,
        deleted_notices=deleted_notices,
        records_changed=new_records != old_records,
    )


def _load_stored_rows(
    embeddings_path: Path,
) -> tuple[list[str], np.ndarray] | None:
    """Read (chunk ids in row order, matrix) from embeddings.npz for merging.

    Provenance was already decided in `plan_update`; this is only the raw
    matrix keyed by row position.
    """
    if not embeddings_path.exists():
        return None
    with np.load(embeddings_path, allow_pickle=False) as payload:
        stored_ids = [str(chunk_id) for chunk_id in payload['chunk_ids']]
        matrix = np.asarray(payload['embeddings'], dtype='float32')
    if matrix.ndim != 2 or matrix.shape[0] != len(stored_ids):
        raise ValueError(
            f'{embeddings_path} has {matrix.shape[0]} rows for {len(stored_ids)} '
            'chunk ids; run a full rebuild'
        )
    return stored_ids, matrix


def _merge_matrix(
    plan: UpdatePlan,
    new_vectors: np.ndarray,
    old_matrix: np.ndarray | None,
    stored_ids: Sequence[str],
) -> np.ndarray:
    """Compose the merged matrix in canonical (plan.new_records) row order."""
    if new_vectors.shape[0]:
        dimension = new_vectors.shape[1]
    elif old_matrix is not None and old_matrix.shape[1]:
        dimension = old_matrix.shape[1]
    else:
        raise ValueError('cannot determine the embedding dimension; run a full rebuild')
    if old_matrix is not None and old_matrix.shape[1] != dimension:
        raise ValueError(
            f'stored vectors have dimension {old_matrix.shape[1]} but the embedder '
            f'returns {dimension}; run a full rebuild'
        )

    row_by_id = {chunk_id: row for row, chunk_id in enumerate(stored_ids)}
    vector_by_id = {
        record['chunk_id']: new_vectors[row] for row, record in enumerate(plan.embed_records)
    }
    rows: list[np.ndarray] = []
    for record in plan.new_records:
        chunk_id = record['chunk_id']
        if chunk_id in vector_by_id:
            rows.append(vector_by_id[chunk_id])
            continue
        row = row_by_id.get(chunk_id)
        if row is None or old_matrix is None:
            raise ValueError(f'no stored vector for reused chunk {chunk_id}; run a full rebuild')
        rows.append(old_matrix[row])
    if not rows:
        return np.zeros((0, dimension), dtype='float32')
    return np.ascontiguousarray(np.vstack(rows), dtype='float32')


def _id_map_in_sync(id_map_path: Path, fingerprint: str) -> bool:
    try:
        payload = json.loads(id_map_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return payload.get('chunks_fingerprint') == fingerprint


def _write_artifacts(
    records: Sequence[dict],
    merged: np.ndarray,
    model_name: str,
    *,
    chunks_path: Path,
    embeddings_path: Path,
    index_path: Path,
    id_map_path: Path,
) -> None:
    """Write the whole artifact set (see module docstring for the order).

    The id_map is renamed before the index so a crash in between is caught by
    the load-time fingerprint and row-count checks; chunks.jsonl is the commit
    point and goes last.
    """
    chunk_ids = [record['chunk_id'] for record in records]
    fingerprint = compute_chunks_fingerprint(records)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends '.npz' unless the name already ends with it.
    npz_tmp = embeddings_path.with_name(embeddings_path.stem + '.tmp.npz')
    np.savez_compressed(
        npz_tmp,
        embeddings=merged,
        chunk_ids=np.asarray(chunk_ids),
        chunks_fingerprint=np.asarray(fingerprint),
        model_name=np.asarray(model_name),
        chunk_text_digests=np.asarray([compute_chunk_text_digest(record) for record in records]),
    )
    os.replace(npz_tmp, embeddings_path)

    index = VectorIndex(merged.shape[1])
    index.build(merged)
    index_tmp = index_path.with_name(index_path.name + '.tmp')
    id_map_tmp = id_map_path.with_name(id_map_path.name + '.tmp')
    index.save(
        index_tmp,
        id_map_tmp,
        chunk_ids,
        meta={'chunks_fingerprint': fingerprint, 'model_name': model_name},
    )
    os.replace(id_map_tmp, id_map_path)
    os.replace(index_tmp, index_path)

    chunks_tmp = chunks_path.with_name(chunks_path.name + '.tmp')
    write_chunks_jsonl(records, chunks_tmp)
    os.replace(chunks_tmp, chunks_path)


def apply_update(
    plan: UpdatePlan,
    embed_texts: Callable[[Sequence[str]], np.ndarray] | None,
    model_name: str = config.MODEL_NAME,
    *,
    chunks_path: Path = config.CHUNKS_PATH,
    embeddings_path: Path = config.EMBEDDINGS_PATH,
    index_path: Path = config.FAISS_INDEX_PATH,
    id_map_path: Path = config.ID_MAP_PATH,
) -> UpdateReport:
    """Execute a plan: embed the missing rows, merge, rewrite the artifact set.

    Idempotent — re-running on an unchanged corpus (or after a crash mid-write)
    embeds only chunks that still lack a valid row and converges on the exact
    artifacts a full rebuild would produce. `embed_texts` may be None only when
    the plan needs no embedding. A no-change run reports `wrote_artifacts=False`
    and touches nothing.
    """
    if plan.needs_embedding and embed_texts is None:
        raise ValueError('plan needs embedding but no embed_texts callable was given')

    fingerprint = compute_chunks_fingerprint(plan.new_records)
    wrote_artifacts = plan.has_changes or not _id_map_in_sync(id_map_path, fingerprint)
    if wrote_artifacts:
        if plan.needs_embedding:
            texts = [
                compose_embedding_text(record['subject'], record['text'])
                for record in plan.embed_records
            ]
            new_vectors = np.asarray(embed_texts(texts), dtype='float32')
            if new_vectors.ndim != 2 or new_vectors.shape[0] != len(texts):
                raise ValueError(
                    f'embed_texts returned shape {new_vectors.shape} for {len(texts)} texts'
                )
        else:
            new_vectors = np.zeros((0, 0), dtype='float32')

        stored_rows = _load_stored_rows(embeddings_path)
        stored_ids, old_matrix = stored_rows if stored_rows else ([], None)
        merged = _merge_matrix(plan, new_vectors, old_matrix, stored_ids)
        _write_artifacts(
            plan.new_records,
            merged,
            model_name,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            index_path=index_path,
            id_map_path=id_map_path,
        )

    return UpdateReport(
        chunks_total=len(plan.new_records),
        embedded_count=len(plan.embed_records),
        reused_count=len(plan.reused_chunk_ids),
        dropped_count=len(plan.dropped_chunk_ids),
        added_notices=plan.added_notices,
        updated_notices=plan.updated_notices,
        deleted_notices=plan.deleted_notices,
        chunks_fingerprint=fingerprint,
        wrote_artifacts=wrote_artifacts,
    )
