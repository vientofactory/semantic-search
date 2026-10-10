"""Stage 4: query processing and semantic similarity search.

Loads the persisted chunks + FAISS index, embeds the query with the same
Korean model used at index time, and ranks chunks by cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .aliases import expand_query
from .chunking import compute_chunks_fingerprint, load_chunks_jsonl
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


@dataclass
class SearchResults:
    """Tiered outcome of one query (see `SemanticSearcher.search_tiered`).

    `results` are clear hits (score >= CLEAR_SIMILARITY); `weak_results` are
    the weak band (MIN_SIMILARITY <= score < CLEAR_SIMILARITY) that callers
    serve separately so the UI can hide them behind an explicit reveal. Hits
    below MIN_SIMILARITY are unrelated to the query and dropped entirely.
    Both tiers keep ranked order.
    """

    results: list[SearchResult]
    weak_results: list[SearchResult]


class SemanticSearcher:
    """Query -> embed -> FAISS top-k -> ranked SearchResult list."""

    def __init__(
        self,
        embedder,
        index,
        chunk_ids: list[str],
        chunks_by_id: dict,
        index_updated_at: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunk_ids = chunk_ids
        self.chunks_by_id = chunks_by_id
        # id_map `updated_at` stamped by VectorIndex.save (None for legacy
        # sets built before the stamp existed); the sidecar adopts it as the
        # "index last updated" time of the generation this searcher serves.
        self.index_updated_at = index_updated_at

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
        records = load_chunks_jsonl(chunks_path)
        chunks_by_id = {record['chunk_id']: record for record in records}
        index, meta = VectorIndex.load(index_path, id_map_path)

        if index.index.ntotal != len(meta['chunk_ids']):
            raise ValueError(
                f'faiss index holds {index.index.ntotal} rows but id_map lists '
                f'{len(meta["chunk_ids"])} chunk ids; the index and id map are from '
                'different builds — re-run scripts/03_build_index.py'
            )

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
        return cls(
            embedder,
            index,
            meta['chunk_ids'],
            chunks_by_id,
            index_updated_at=meta.get('updated_at'),
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Process one query and return raw top-k chunks ranked by similarity.

        No relevance policy is applied here — callers that expose results to
        users go through `search_tiered`; tools (CLI, evaluation) that need
        the unfiltered ranking use this method directly.

        Equal scores are broken by chunk_id so ranking is reproducible even
        when the corpus contains duplicate content.

        Citizen aliases (중처법, 산안법, ...) are rewritten to their official
        bill names before embedding (see `aliases.expand_query`); the corpus
        has no such surface form, so without it those queries return nothing.
        """
        if k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        normalized = normalize_text(expand_query(query).text)
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

    def search_tiered(self, query: str, k: int = 5) -> SearchResults:
        """Rank like `search()`, then split the top-k window by relevance tier.

        The two thresholds come from `config.MIN_SIMILARITY` /
        `config.CLEAR_SIMILARITY`. Unrelated hits (below the floor) vanish
        entirely, so a query with no qualifying hit returns two empty lists
        rather than padded results; the tiers are cut from the same top-k
        window, so `len(results) + len(weak_results) <= k`.
        """
        clear: list[SearchResult] = []
        weak: list[SearchResult] = []
        for hit in self.search(query, k):
            if hit.score >= config.CLEAR_SIMILARITY:
                clear.append(hit)
            elif hit.score >= config.MIN_SIMILARITY:
                weak.append(hit)
        return SearchResults(results=clear, weak_results=weak)
