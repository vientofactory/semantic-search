"""Stage 2: tokenization and embedding extraction.

Wraps a Korean-specialized sentence-transformers model (default:
`nlpai-lab/KURE-v1`, Korean retrieval-tuned bge-m3) and exposes
both the tokenizer step (token counts / truncation against the model's
max sequence length) and the embedding step (pooled vectors,
L2-normalized for cosine similarity).

The run device comes from `lawcast_semantic.device`: `auto` (the default)
picks hardware faster than CPU when present, and any accelerator failure
degrades to cpu with a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .device import resolve_device

logger = logging.getLogger(__name__)


class KoreanEmbedder:
    """Tokenizer + embedding extractor backed by one sentence-transformers model."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.device, self.model = self._load_model(model_name, device)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = int(self.model.max_seq_length)
        self.dimension = int(self.model.get_embedding_dimension())

    @staticmethod
    def _load_model(model_name: str, device: str) -> tuple[str, SentenceTransformer]:
        """Load the model on the resolved device, degrading to cpu.

        Both an `auto` pick and an explicit pin go through the same gate: the
        accelerator path must prove itself with a tiny probe forward pass,
        because a device can construct the model fine and still fail (or
        return non-finite vectors) on the first encode — and a mid-corpus
        failure would strand a long stage-2 run halfway through. Any failure
        on a non-cpu device falls back to cpu with a warning instead of
        aborting, since the device never changes the vectors' meaning.
        """
        resolved = resolve_device(device)
        try:
            model = SentenceTransformer(model_name, device=resolved)
            if resolved != 'cpu':
                KoreanEmbedder._probe(model, resolved)
        except Exception as exc:  # noqa: BLE001 - any accelerator failure degrades to cpu
            if resolved == 'cpu':
                raise
            logger.warning(
                'device %s unusable (%s: %s); falling back to cpu',
                resolved,
                type(exc).__name__,
                exc,
            )
            resolved = 'cpu'
            model = SentenceTransformer(model_name, device='cpu')
        return resolved, model

    @staticmethod
    def _probe(model: SentenceTransformer, device: str) -> None:
        """Encode one short text to prove the device actually works."""
        expected = int(model.get_embedding_dimension())
        encoded = model.encode(['device probe'], normalize_embeddings=True)
        vectors = np.asarray(encoded, dtype='float32')
        if vectors.shape != (1, expected):
            raise RuntimeError(
                f'{device} probe returned shape {vectors.shape}, expected (1, {expected})'
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError(f'{device} probe returned non-finite embeddings')

    def tokenize(self, text: str) -> dict:
        """Tokenize one text without truncation and report model-fit info."""
        # truncation=False so we can measure true length; verbose=False keeps
        # transformers from warning about long sequences (reported by us).
        input_ids = self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)[
            'input_ids'
        ]
        return {
            'token_count': len(input_ids),
            'truncated': len(input_ids) > self.max_seq_length,
            # First tokens decoded for inspection (special tokens included).
            'preview_tokens': self.tokenizer.convert_ids_to_tokens(input_ids[:12]),
        }

    def tokenize_report(self, texts: Iterable[str]) -> dict:
        """Aggregate token statistics for a corpus (used by stage 2 output)."""
        counts = [self.tokenize(text)['token_count'] for text in texts]
        truncated = sum(1 for count in counts if count > self.max_seq_length)
        return {
            'model_name': self.model_name,
            'max_seq_length': self.max_seq_length,
            'text_count': len(counts),
            'tokens_total': int(sum(counts)),
            'tokens_mean': round(float(np.mean(counts)), 1) if counts else 0.0,
            'tokens_max': int(max(counts)) if counts else 0,
            'truncated_count': truncated,
        }

    def embed_texts(
        self,
        texts: Iterable[str],
        batch_size: int = config.EMBED_BATCH_SIZE,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed texts into a float32 matrix (L2-normalized by default).

        An empty input yields a well-shaped (0, dimension) matrix so an empty
        corpus can flow through the pipeline without special-casing downstream.
        """
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dimension), dtype='float32')
        vectors = self.model.encode(
            text_list,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype='float32')

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a 1-D float32 vector."""
        return self.embed_texts([text])[0]
