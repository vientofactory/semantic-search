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

import logging
from collections.abc import Callable, Sequence
from typing import Any

from . import config
from .device import resolve_device

logger = logging.getLogger(__name__)


class Reranker:
    """Thin owner of a loaded cross-encoder and its model provenance."""

    def __init__(
        self,
        model_name: str,
        device: str,
        predictor: Callable[[list[list[str]]], Any],
        requested_device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.requested_device = requested_device if requested_device is not None else device
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
    """Load the cross-encoder onto the resolved device (imports the model stack here).

    `device` is the *requested* device (default `config.RERANK_DEVICE`,
    `cpu`): resolved through `lawcast_semantic.device.resolve_device`
    (CUDA > MPS > XPU > cpu) like the embedding model, so
    `LAWCAST_SEMANTIC_RERANK_DEVICE=auto` opts into accelerator probing.
    Keep `cpu` under concurrent load — concurrent embed + rerank passes on
    Apple MPS abort the process (Metal command-buffer race); `auto` is for
    single-query measurement only. An accelerator load must prove itself
    with a tiny probe prediction, because a device can construct the model
    fine and still fail on the first forward pass. Any failure on a non-cpu
    device falls back to cpu with a warning instead of aborting.

    `max_length` bounds the query-document truncation window; the served
    chunks are short (see `config.RERANK_MAX_LENGTH`), so a smaller window
    sheds padding/attention work without touching the model weights.
    """
    from sentence_transformers import CrossEncoder

    resolved = resolve_device(device)
    try:
        predictor = CrossEncoder(model_name, max_length=max_length, device=resolved)
        if resolved != 'cpu':
            _probe(predictor, resolved)
    except Exception as exc:  # noqa: BLE001 - any accelerator failure degrades to cpu
        if resolved == 'cpu':
            raise
        logger.warning(
            'rerank device %s unusable (%s: %s); falling back to cpu',
            resolved,
            type(exc).__name__,
            exc,
        )
        resolved = 'cpu'
        predictor = CrossEncoder(model_name, max_length=max_length, device='cpu')
    return Reranker(model_name, resolved, predictor.predict, requested_device=device)


def _probe(predictor: Any, device: str) -> None:
    """Score one short pair to prove the device actually works."""
    import numpy as np

    scores = np.asarray(predictor.predict([['device probe', 'device probe']]), dtype='float32')
    if scores.shape != (1,):
        raise RuntimeError(f'{device} probe returned shape {scores.shape}, expected (1,)')
    if not np.isfinite(scores).all():
        raise RuntimeError(f'{device} probe returned non-finite scores')


def shadow_inputs(
    query: str,
    hits: Sequence[Any],
    top_k: int = config.RERANK_TOP_K,
) -> list[str]:
    """Candidate texts for the shadow scorer: top-`top_k` hits in rank order."""
    from .chunking import compose_embedding_text

    return [compose_embedding_text(hit.subject, hit.text) for hit in list(hits)[:top_k]]
