"""Stage 2: tokenization and embedding extraction.

Wraps a Korean-specialized sentence-transformers model (default:
`jhgan/ko-sbert-sts`, Korean Sentence-BERT fine-tuned on KorSTS) and exposes
both the tokenizer step (token counts / truncation against the model's
max sequence length) and the embedding step (768-dim pooled vectors,
L2-normalized for cosine similarity).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class KoreanEmbedder:
    """Tokenizer + embedding extractor backed by one sentence-transformers model."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = int(self.model.max_seq_length)
        self.dimension = int(self.model.get_embedding_dimension())

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
