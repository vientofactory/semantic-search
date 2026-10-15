"""Stage 4c: cross-encoder rerank scorer (shadow only).

Wraps a sentence-transformers CrossEncoder (default
`nlpai-lab/KURE-Reranker-nano`, see `config.RERANK_*`) that scores
query-document pairs jointly. Scores are uncalibrated relevance logits, not
probabilities — the shadow path logs them for calibration and never filters.

Import discipline (§6.1): this module must stay light. The model stack is
imported inside `load_reranker`, never at module top, so `import
service.app` keeps passing the no-torch/no-faiss gate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from . import config


class Reranker:
    """Thin owner of a loaded cross-encoder and its model provenance."""

    def __init__(
        self,
        model_name: str,
        device: str,
        predictor: Callable[[list[list[str]]], Any],
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._predictor = predictor

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score one query against each candidate text, best-first input order."""
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        return [float(value) for value in self._predictor(pairs)]


def load_reranker(
    model_name: str = config.RERANK_MODEL,
    device: str = config.RERANK_DEVICE,
    max_length: int = config.RERANK_MAX_LENGTH,
) -> Reranker:
    """Load the cross-encoder onto `device` (imports the model stack here).

    `max_length` bounds the query-document truncation window; the served
    chunks are short (see `config.RERANK_MAX_LENGTH`), so a smaller window
    sheds padding/attention work without touching the model weights.
    """
    from sentence_transformers import CrossEncoder

    predictor = CrossEncoder(model_name, max_length=max_length, device=device)
    return Reranker(model_name, device, predictor.predict)


def shadow_inputs(
    query: str,
    hits: Sequence[Any],
    top_k: int = config.RERANK_TOP_K,
) -> list[str]:
    """Candidate texts for the shadow scorer: top-`top_k` hits in rank order."""
    from .chunking import compose_embedding_text

    return [compose_embedding_text(hit.subject, hit.text) for hit in list(hits)[:top_k]]
