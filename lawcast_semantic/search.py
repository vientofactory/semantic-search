"""Stage 4: query processing and semantic similarity search.

Loads the persisted chunks + FAISS index, embeds the query with the same
Korean model used at index time, and ranks chunks by cosine similarity.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from . import config
from .aliases import expand_query
from .chunking import compute_chunks_fingerprint, load_chunks_jsonl
from .hybrid import has_lexical_evidence
from .indexing import VectorIndex
from .preprocess import normalize_text
from .relevance import CALIBRATED_MODEL, DEFAULT_CUTOFFS, RelevanceProfile, classify
from .timing import DensePhase, EmbeddingStages

# A related amendment is useful context only after the query has a clear anchor.
_WEAK_RECOVERY_ANCHOR = 0.60
_WEAK_RECOVERY_FLOOR = 0.45
_AMENDMENT_SUBJECT = re.compile(r'(.+?)\s*(?:일부개정|전부개정)\s*법률안(?:\s*\([^()]*\))*')


def _amended_law(subject: str) -> str | None:
    match = _AMENDMENT_SUBJECT.fullmatch(normalize_text(subject))
    return ''.join(match[1].split()) if match else None


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
class DenseStages:
    embedding: DensePhase
    faiss: DensePhase
    classification: DensePhase
    embedding_stages: EmbeddingStages | None = None


@dataclass
class SearchResults:
    """Tiered outcome of one query (see `SemanticSearcher.search_tiered`).

    Both tiers keep dense rank. When calibrated filtering runs,
    `candidate_count` counts candidates above the absolute floor before tiering.
    """

    results: list[SearchResult]
    weak_results: list[SearchResult]
    candidate_count: int | None = None
    dense_stages: DenseStages | None = None


@dataclass
class EmbeddedQuery:
    vector: np.ndarray
    phase: DensePhase
    stages: EmbeddingStages | None


class SemanticSearcher:
    """Query -> embed -> FAISS top-k -> ranked SearchResult list."""

    def __init__(
        self,
        embedder,
        index,
        chunk_ids: list[str],
        chunks_by_id: dict,
        index_updated_at: str | None = None,
        relevance_profile: RelevanceProfile | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunk_ids = chunk_ids
        self.chunks_by_id = chunks_by_id
        # id_map `updated_at` stamped by VectorIndex.save (None for legacy
        # sets built before the stamp existed); the sidecar adopts it as the
        # "index last updated" time of the generation this searcher serves.
        self.index_updated_at = index_updated_at
        self.relevance_profile = relevance_profile

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
            relevance_profile=(
                RelevanceProfile.from_index(index, meta['chunk_ids'], chunks_by_id)
                if config.RELEVANCE_ENABLED
                and embedder.model_name == CALIBRATED_MODEL
                and index.index_type == 'flat'
                else None
            ),
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Process one query and return raw top-k chunks ranked by similarity.

        No relevance policy is applied here — callers that expose results to
        users go through `search_tiered`; tools (CLI, evaluation) that need
        the unfiltered ranking use this method directly.

        Equal scores are broken by chunk_id so ranking is reproducible even
        when the corpus contains duplicate content.

        Citizen aliases (`중처법`, `산안법`, ...) are rewritten to their official
        bill names before embedding (see `aliases.expand_query`); the corpus
        has no such surface form, so without it those queries return nothing.
        """
        if k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        normalized = normalize_text(expand_query(query).text)
        if not normalized:
            return []
        return self._rank(self.embedder.embed_query(normalized), k)

    def _rank(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        # Fetch only the serving window from the index: the tiers below cut
        # the same top-k window, so scoring all ntotal rows only added a
        # GIL-held full-index sort to every concurrent request.
        hits = self.index.search(query_vector, k=k)
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

    def search_tiered(
        self,
        query: str,
        k: int = 5,
        *,
        admission_started: float | None = None,
        on_embedding_finished: Callable[[], None] | None = None,
    ) -> SearchResults:
        """Apply the generation's relevance policy without changing dense order."""
        try:
            prepared = self.embed_tiered(query, k, admission_started=admission_started)
        finally:
            if on_embedding_finished is not None:
                on_embedding_finished()
        return self.rank_tiered(query, prepared, k)

    def embed_tiered(
        self, query: str, k: int, *, admission_started: float | None = None
    ) -> EmbeddedQuery | SearchResults:
        """Prepare and embed once; the caller can schedule ranking separately."""
        if k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        normalized = normalize_text(expand_query(query).text)
        if not normalized:
            return SearchResults([], [], 0 if self.relevance_profile is not None else None)
        embed_started = perf_counter() if admission_started is None else admission_started
        slots = getattr(self.embedder, 'query_slots', None)
        # Direct library callers may also use this embedder; keep its native gate.
        with slots if slots is not None else nullcontext():
            profile_query = getattr(self.embedder, 'embed_query_profiled', None)
            if profile_query is None:
                vector, embedding_stages = self.embedder.embed_query(normalized), None
            else:
                vector, embedding_stages = profile_query(normalized)
        embed_finished = perf_counter()
        if not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError('query embedding is non-finite or zero')
        return EmbeddedQuery(vector, DensePhase(embed_started, embed_finished), embedding_stages)

    def rank_tiered(
        self, query: str, prepared: EmbeddedQuery | SearchResults, k: int
    ) -> SearchResults:
        """Rank a prepared query on the same immutable index generation."""
        if isinstance(prepared, SearchResults):
            return prepared
        vector = prepared.vector
        rank_started = perf_counter()
        hits = self._rank(vector, k)
        rank_finished = perf_counter()
        outcome = self.tier_candidates(query, vector, hits)
        outcome.dense_stages = DenseStages(
            embedding=prepared.phase,
            faiss=DensePhase(rank_started, rank_finished),
            classification=DensePhase(rank_finished, perf_counter()),
            embedding_stages=prepared.stages,
        )
        return outcome

    def tier_candidates(
        self, query: str, vector: np.ndarray, hits: list[SearchResult]
    ) -> SearchResults:
        """Classify a ranked window; replay can share retrieval with a baseline."""
        profile = self.relevance_profile
        if profile is not None:
            # Lower-ranked candidates cannot pass this absolute floor either.
            # Returning a short count lets the backend avoid widening such windows.
            floor = max(config.MIN_SIMILARITY, DEFAULT_CUTOFFS.weak_cosine)
            hits = [hit for hit in hits if hit.score >= floor]
        evidence = profile.for_query(query, vector) if profile is not None and hits else None
        clear, weak = [], []
        for hit in hits:
            if evidence is None:
                if hit.score >= config.CLEAR_SIMILARITY:
                    clear.append(hit)
                elif hit.score >= config.MIN_SIMILARITY:
                    weak.append(hit)
                continue
            tier = classify(
                evidence.measure(hit.score, self.chunks_by_id[hit.chunk_id]),
                config.MIN_SIMILARITY,
                config.CLEAR_SIMILARITY,
            )
            if tier == 'strong':
                clear.append(hit)
            elif tier == 'weak':
                weak.append(hit)
        if profile is None:
            weak.extend(self._recover_related_amendments(query, hits, clear, weak))
        return SearchResults(clear, weak, len(hits) if profile is not None else None)

    def _recover_related_amendments(
        self,
        query: str,
        hits: list[SearchResult],
        clear: list[SearchResult],
        weak: list[SearchResult],
    ) -> list[SearchResult]:
        """Recover same-law amendments in the weak tier without reopening empty queries."""
        if (
            config.MIN_SIMILARITY <= _WEAK_RECOVERY_FLOOR
            or not clear
            or clear[0].score < _WEAK_RECOVERY_ANCHOR
        ):
            return []
        law = _amended_law(clear[0].subject)
        if not law:
            return []
        served = {hit.notice_num for hit in (*clear, *weak)}
        recovered = []
        for hit in hits:
            if hit.score < _WEAK_RECOVERY_FLOOR:
                break
            if (
                hit.score >= config.MIN_SIMILARITY
                or hit.notice_num in served
                or _amended_law(hit.subject) != law
                or not has_lexical_evidence(query, {'subject': hit.subject, 'text': hit.text})
            ):
                continue
            recovered.append(hit)
            served.add(hit.notice_num)
        return recovered
