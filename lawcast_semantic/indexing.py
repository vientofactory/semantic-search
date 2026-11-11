"""Stage 3: FAISS index construction and persistence.

The default FlatIP index scores normalized vectors exactly. The optional
SQ8 index scans scalar-quantized approximations of the same vectors.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

INDEX_TYPES = ('flat', 'sq8')


class VectorIndex:
    """Cosine-similarity vector index with JSON chunk-id mapping."""

    def __init__(self, dimension: int, index_type: str = 'flat') -> None:
        self.dimension = int(dimension)
        if index_type not in INDEX_TYPES:
            raise ValueError(f'unsupported index type {index_type!r}')
        self.index_type = index_type
        if index_type == 'sq8':
            self.index = faiss.IndexScalarQuantizer(
                self.dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
            )
        else:
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
        vectors = np.asarray(embeddings, dtype='float32')
        if vectors.ndim != 2:
            raise ValueError(f'expected 2-D embeddings, got shape {vectors.shape}')
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f'embedding dim {vectors.shape[1]} does not match index dim {self.dimension}'
            )
        self.index.reset()
        if len(vectors) == 0:
            return
        if self.index_type == 'sq8':
            vectors = self._as_normalized_matrix(vectors)
            self.index.train(vectors)
            self.index.add(vectors)
            return
        # FlatIP owns a copy. Bound normalization scratch instead of keeping
        # another complete corpus matrix alive during scheduled refreshes.
        for start in range(0, len(vectors), 4096):
            self.index.add(self._as_normalized_matrix(vectors[start : start + 4096]))

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
        # Single owner of the index-update timestamp: every writer (host
        # rebuilds via scripts/03, incremental runs via scripts/06, sidecar
        # ticks and boot repair) funnels through save(), so id_map.json
        # records when this set was last written. Stamped after merging
        # caller meta so the value always describes this write.
        payload['index_type'] = self.index_type
        payload['updated_at'] = datetime.now(UTC).isoformat()
        id_map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, index_path: Path, id_map_path: Path) -> tuple[VectorIndex, dict]:
        """Load a persisted index and return (index, payload).

        payload contains at least 'chunk_ids'; remaining keys are the meta
        recorded at build time, including 'updated_at' (UTC ISO 8601 stamp
        written by :meth:`save` — absent only for pre-stamp legacy sets).
        """
        index = faiss.read_index(str(index_path))
        payload = json.loads(id_map_path.read_text(encoding='utf-8'))
        index_type = payload.get('index_type', 'flat')
        if index.metric_type != faiss.METRIC_INNER_PRODUCT or not (
            (index_type == 'flat' and isinstance(index, faiss.IndexFlatIP))
            or (
                index_type == 'sq8'
                and isinstance(index, faiss.IndexScalarQuantizer)
                and index.sq.qtype == faiss.ScalarQuantizer.QT_8bit
            )
        ):
            raise ValueError(f'FAISS index does not match declared type {index_type!r}')
        wrapper = cls(index.d, index_type)
        wrapper.index = index
        return wrapper, payload

    def search(self, query_vector: np.ndarray, k: int | None = None) -> list[tuple[int, float]]:
        """Return (row_position, cosine_score) pairs sorted by score desc.

        k=None scores every vector; otherwise only the top-k rows are fetched
        from FAISS so a serving query never pays for (or Python-sorts) the
        whole index. Ties are broken deterministically by row position;
        SemanticSearcher re-sorts by chunk_id before cutting top-k so
        duplicate-content chunks cannot flip order between runs.
        """
        if self.index.ntotal == 0:
            return []
        if k is not None and k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype='float32')
        query /= max(float(np.linalg.norm(query)), 1e-12)
        # Push the window down into FAISS: a serving k (<= MAX_K) fetches a
        # handful of rows instead of all ntotal, which also keeps the GIL-held
        # Python sort below off the concurrent-request path.
        fetch = self.index.ntotal if k is None else min(k, self.index.ntotal)
        scores, positions = self.index.search(query, fetch)
        hits = [
            (int(pos), float(score)) for pos, score in zip(positions[0], scores[0]) if int(pos) >= 0
        ]
        hits.sort(key=lambda hit: (-hit[1], hit[0]))
        return hits
