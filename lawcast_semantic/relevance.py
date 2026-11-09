"""Relevance policy fitted to the existing Korean BGE model and FlatIP index."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, log1p
from typing import Literal

import numpy as np

from .hybrid import chunk_tokens, lexical_tokens

CALIBRATED_MODEL = 'dragonkue/BGE-m3-ko'
BACKGROUND_SIZE = 512
BACKGROUND_SEED = 42
MIN_BACKGROUND_SIZE = 32
SIGMA_FLOOR = 0.03

Tier = Literal['strong', 'weak', 'discard']


@dataclass(frozen=True)
class Evidence:
    cosine: float
    z: float | None
    coverage: float


@dataclass(frozen=True)
class Cutoffs:
    strong_cosine: float = 0.51
    strong_z: float = 5.0
    strong_coverage: float = 0.35
    semantic_cosine: float = 0.55
    semantic_z: float = 5.5
    weak_cosine: float = 0.49
    weak_z: float = 5.5
    weak_semantic_cosine: float = 0.51
    weak_coverage: float = 0.35


DEFAULT_CUTOFFS = Cutoffs()


def classify(
    evidence: Evidence, min_cosine: float, clear_cosine: float, cutoffs: Cutoffs = DEFAULT_CUTOFFS
) -> Tier:
    """Require query-relative evidence; never promote a hit based on rank alone."""
    s, z, k = evidence.cosine, evidence.z, evidence.coverage
    if not isfinite(s) or s < min_cosine:
        return 'discard'
    has_z = z is not None and isfinite(z)
    if (
        has_z
        and s >= max(clear_cosine, cutoffs.strong_cosine)
        and z >= cutoffs.strong_z
        and (
            k >= cutoffs.strong_coverage
            or (s >= cutoffs.semantic_cosine and z >= cutoffs.semantic_z)
        )
    ):
        return 'strong'
    if s >= cutoffs.weak_cosine and (
        k >= cutoffs.weak_coverage
        or (has_z and s >= cutoffs.weak_semantic_cosine and z >= cutoffs.weak_z)
    ):
        return 'weak'
    return 'discard'


@dataclass(frozen=True)
class QueryEvidence:
    weights: dict[str, float]
    weight_sum: float
    background: tuple[float, float] | None

    def measure(self, score: float, chunk: dict) -> Evidence:
        coverage = 0.0
        if self.weight_sum:
            shared = set(chunk_tokens(chunk)).intersection(self.weights)
            # Sum in query order for stable boundary decisions across hash seeds.
            coverage = (
                sum(weight for token, weight in self.weights.items() if token in shared)
                / self.weight_sum
            )
        z = None
        if self.background is not None:
            mean, scale = self.background
            z = (score - mean) / scale
        return Evidence(score, z, coverage)


class RelevanceProfile:
    """Bounded reference vectors and sampled token frequencies, owned by one index."""

    def __init__(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        self.vectors = vectors
        self.vectors.setflags(write=False)
        self.size = len(chunks)
        self.df = Counter(token for chunk in chunks for token in chunk_tokens(chunk))

    @classmethod
    def from_index(cls, index, chunk_ids: list[str], chunks_by_id: dict) -> RelevanceProfile:
        positions = np.random.default_rng(BACKGROUND_SEED).choice(
            len(chunk_ids), size=min(BACKGROUND_SIZE, len(chunk_ids)), replace=False
        )
        positions.sort()
        vectors = index.index.reconstruct_batch(positions.astype(np.int64))
        return cls(vectors, [chunks_by_id[chunk_ids[i]] for i in positions])

    def for_query(self, query: str, vector: np.ndarray) -> QueryEvidence:
        weights = {}
        for token in lexical_tokens(query):
            df = self.df[token]
            # Unseen tokens still count in the denominator, without assuming
            # they are exceptionally rare in the full corpus.
            weights[token] = min(5.0, log1p((self.size - df + 0.5) / (df + 0.5))) if df else 3.0
        background = None
        if len(self.vectors) >= MIN_BACKGROUND_SIZE:
            # The inputs are L2-normalized by the query encoder and FlatIP builder.
            scores = np.einsum('ij,j->i', self.vectors, vector)
            if np.isfinite(scores).all():
                std = float(scores.std())
                if std > 1e-6:
                    background = (float(scores.mean()), max(std, SIGMA_FLOOR))
        return QueryEvidence(weights, sum(weights.values()), background)
