"""Stage 4: query processing and semantic similarity search.

Loads the persisted chunks + FAISS index, embeds the query with the same
Korean model used at index time, and ranks chunks by cosine similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import config

from .chunking import compute_chunks_fingerprint
from .indexing import VectorIndex
from .preprocess import normalize_text


@dataclass
class SearchResult:
    """One ranked hit: similarity score plus the source chunk metadata."""

    chunk_id: str
    score: float
    notice_num: int
    subject: str
    committee: str
    section: str
    text: str


class SemanticSearcher:
    """Query -> embed -> FAISS top-k -> ranked SearchResult list."""

    def __init__(self, embedder, index, chunk_ids: list[str], chunks_by_id: dict) -> None:
        self.embedder = embedder
        self.index = index
        self.chunk_ids = chunk_ids
        self.chunks_by_id = chunks_by_id

    @classmethod
    def load(
        cls,
        embedder,
        chunks_path: Path = config.CHUNKS_PATH,
        index_path: Path = config.FAISS_INDEX_PATH,
        id_map_path: Path = config.ID_MAP_PATH,
    ) -> SemanticSearcher:
        """Load stage 1/3 artifacts and build a ready-to-query searcher.

        Validates that the artifacts belong together (chunk fingerprint and
        model provenance) so partial pipeline re-runs fail loudly instead of
        silently mixing stale scores with fresh text.
        """
        records = []
        with chunks_path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        chunks_by_id = {record['chunk_id']: record for record in records}
        index, meta = VectorIndex.load(index_path, id_map_path)

        expected_fingerprint = compute_chunks_fingerprint(records)
        if meta.get('chunks_fingerprint') != expected_fingerprint:
            raise ValueError(
                'chunks.jsonl is out of sync with the FAISS index '
                f'(fingerprint {expected_fingerprint[:12]} != '
                f'{str(meta.get("chunks_fingerprint"))[:12]}); '
                're-run scripts/02_extract_embeddings.py and scripts/03_build_index.py'
            )
        if meta.get('model_name') != embedder.model_name:
            raise ValueError(
                f"index was built with model '{meta.get('model_name')}' but the "
                f"query embedder is '{embedder.model_name}'; rebuild the index "
                'or query with the same --model'
            )
        if index.dimension != embedder.dimension:
            raise ValueError(
                f'index dimension {index.dimension} does not match embedder '
                f'dimension {embedder.dimension}; rebuild the index'
            )
        return cls(embedder, index, meta['chunk_ids'], chunks_by_id)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Process one query and return top-k chunks ranked by similarity.

        Equal scores are broken by chunk_id so ranking is reproducible even
        when the corpus contains duplicate content.
        """
        if k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        normalized = normalize_text(query)
        if not normalized:
            return []
        query_vector = self.embedder.embed_query(normalized)
        hits = self.index.search(query_vector, k=None)
        hits.sort(key=lambda hit: (-hit[1], self.chunk_ids[hit[0]]))

        results: list[SearchResult] = []
        for position, score in hits[:k]:
            chunk_id = self.chunk_ids[position]
            chunk = self.chunks_by_id[chunk_id]
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    notice_num=int(chunk['notice_num']),
                    subject=chunk['subject'],
                    committee=chunk['committee'],
                    section=chunk['section'],
                    text=chunk['text'],
                )
            )
        return results
