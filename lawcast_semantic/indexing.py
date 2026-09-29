"""Stage 3: FAISS index construction and persistence.

Uses `faiss.IndexFlatIP` over L2-normalized vectors, which makes inner
product equal to cosine similarity. Exact search is fine at LawCast scale
(tens of thousands of chunks); swap in an IVF/PQ index later if needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    """Cosine-similarity vector index with JSON chunk-id mapping."""

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.index = faiss.IndexFlatIP(self.dimension)

    @staticmethod
    def _as_normalized_matrix(embeddings: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(embeddings, dtype='float32')
        if vectors.ndim != 2:
            raise ValueError(f'expected 2-D embeddings, got shape {vectors.shape}')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(vectors / norms)

    def build(self, embeddings: np.ndarray) -> None:
        """(Re)build the index from an embeddings matrix."""
        vectors = self._as_normalized_matrix(embeddings)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f'embedding dim {vectors.shape[1]} does not match index dim {self.dimension}'
            )
        self.index.reset()
        self.index.add(vectors)

    def save(
        self,
        index_path: Path,
        id_map_path: Path,
        chunk_ids: list[str],
        meta: dict | None = None,
    ) -> None:
        """Persist the FAISS index and its row -> chunk_id mapping.

        `meta` (e.g. chunks_fingerprint / model_name) is stored alongside the
        mapping so consumers can detect stale or mismatched artifacts.
        """
        if len(chunk_ids) != self.index.ntotal:
            raise ValueError('chunk_ids length must match number of indexed vectors')
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        payload: dict = {'chunk_ids': list(chunk_ids)}
        if meta:
            payload.update(meta)
        id_map_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    @classmethod
    def load(cls, index_path: Path, id_map_path: Path) -> tuple['VectorIndex', dict]:
        """Load a persisted index and return (index, payload).

        payload contains at least 'chunk_ids'; remaining keys are the meta
        recorded at build time.
        """
        index = faiss.read_index(str(index_path))
        wrapper = cls(index.d)
        wrapper.index = index
        payload = json.loads(id_map_path.read_text(encoding='utf-8'))
        return wrapper, payload

    def search(
        self, query_vector: np.ndarray, k: int | None = None
    ) -> list[tuple[int, float]]:
        """Return (row_position, cosine_score) pairs sorted by score desc.

        k=None scores every vector. Ties are broken deterministically by row
        position; SemanticSearcher re-sorts by chunk_id before cutting top-k so
        duplicate-content chunks cannot flip order between runs.
        """
        if self.index.ntotal == 0:
            return []
        if k is not None and k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype='float32')
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores, positions = self.index.search(query, self.index.ntotal)
        hits = [(int(pos), float(score)) for pos, score in zip(positions[0], scores[0])]
        hits.sort(key=lambda hit: (-hit[1], hit[0]))
        return hits if k is None else hits[:k]
