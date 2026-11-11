"""Stage 2: tokenization and embedding extraction.

Wraps a Korean-specialized sentence-transformers model (default:
`dragonkue/BGE-m3-ko`, Korean retrieval-tuned bge-m3) and exposes
both the tokenizer step (token counts / truncation against the model's
max sequence length) and the embedding step (pooled vectors,
L2-normalized for cosine similarity).

The run device comes from `lawcast_semantic.device`: `auto` (the default)
picks hardware faster than CPU when present, and any accelerator failure
degrades to cpu with a warning.
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config
from .device import resolve_device
from .query_onnx import MODEL, OnnxQueryEncoder
from .timing import DensePhase, EmbeddingStages

logger = logging.getLogger(__name__)


_preprocess_spans: ContextVar[list[DensePhase] | None] = ContextVar(
    'query_preprocess_spans', default=None
)


class _ProfiledSentenceTransformer(SentenceTransformer):
    def preprocess(self, *args, **kwargs):
        spans = _preprocess_spans.get()
        if spans is None:
            return super().preprocess(*args, **kwargs)
        started = perf_counter()
        features = super().preprocess(*args, **kwargs)
        spans.append(DensePhase(started, perf_counter()))
        return features


class KoreanEmbedder:
    """Tokenizer + embedding extractor backed by one sentence-transformers model."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self._torch_lock = Lock()
        self._torch_users = 0
        self._torch_persistent = False
        self._torch_revision = None
        self.model = None
        self._query_onnx = self._load_query_onnx(config.QUERY_ONNX_PATH)
        if self._query_onnx is None:
            self.device, self.model = self._load_model(model_name, device)
            self._torch_persistent = True
            self.tokenizer = self.model.tokenizer
            self.max_seq_length = int(self.model.max_seq_length)
            self.dimension = int(self.model.get_embedding_dimension())
        else:
            self.device = 'cpu'
            self._torch_revision = self._query_onnx.revision
            self.tokenizer = self._query_onnx.tokenizer
            self.max_seq_length = self._query_onnx.max_seq_length
            self.dimension = self._query_onnx.dimension
        # Share slots across index generations that reuse this embedder.
        self.query_slots = (
            BoundedSemaphore(config.MPS_QUERY_CONCURRENCY)
            if self.device == 'mps' and config.MPS_QUERY_CONCURRENCY
            else None
        )
        self._query_cache_size = config.QUERY_EMBED_CACHE_SIZE
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = Lock()

    def _load_query_onnx(self, path: str) -> OnnxQueryEncoder | None:
        if not path:
            return None
        if self.model_name != MODEL or resolve_device(self.requested_device) != 'cpu':
            logger.warning('query ONNX requires the CPU dragonkue/BGE-m3-ko model; using PyTorch')
            return None
        try:
            encoder = OnnxQueryEncoder(Path(path), config.QUERY_ONNX_THREADS)
        except Exception as exc:  # noqa: BLE001 - optional runtime must not block the original path
            logger.warning(
                'query ONNX unavailable; using PyTorch (%s: %s)', type(exc).__name__, exc
            )
            return None
        logger.info(
            'query-encoder=onnx cpu_threads=%d probe_max_abs_delta=%.9g torch_weights=deferred',
            config.QUERY_ONNX_THREADS,
            encoder.probe_delta,
        )
        return encoder

    @contextmanager
    def _torch_model(self) -> Iterator[SentenceTransformer]:
        # Keep the legacy eager path lock-free, including standalone tools.
        if self.model is not None and getattr(self, '_torch_persistent', True):
            yield self.model
            return
        with self._torch_lock:
            if self.model is None:
                reason = 'query-fallback' if self._torch_persistent else 'documents'
                logger.info('loading PyTorch weights reason=%s', reason)
                _, self.model = self._load_model(self.model_name, 'cpu', self._torch_revision)
            self._torch_users += 1
            model = self.model
        try:
            yield model
        finally:
            released = False
            with self._torch_lock:
                self._torch_users -= 1
                if not self._torch_users and not self._torch_persistent:
                    self.model = None
                    released = True
            if released:
                # Model/tokenizer cycles can otherwise keep the weight tensors alive
                # until an unrelated request happens to trigger cyclic collection.
                del model
                gc.collect()
                logger.info('released document PyTorch weights')

    @staticmethod
    def _load_model(
        model_name: str, device: str, revision: str | None = None
    ) -> tuple[str, SentenceTransformer]:
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
        kwargs = {'revision': revision} if revision else {}
        try:
            model = _ProfiledSentenceTransformer(model_name, device=resolved, **kwargs)
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
            model = _ProfiledSentenceTransformer(model_name, device='cpu', **kwargs)
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
        with self._torch_model() as model:
            try:
                vectors = model.encode(
                    text_list,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
            finally:
                del model
        return np.asarray(vectors, dtype='float32')

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a 1-D float32 vector."""
        if getattr(self, '_query_onnx', None) is not None:
            return self.embed_query_profiled(text)[0]
        return self.embed_texts([text])[0]

    def embed_query_profiled(self, text: str) -> tuple[np.ndarray, EmbeddingStages | None]:
        """Reuse exact model vectors for repeated normalized queries and wider windows."""
        if not self._query_cache_size:
            return self._encode_query_profiled(text)
        with self._query_cache_lock:
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
        if cached is not None:
            # FAISS normalizes its input in place. Never expose a shared vector.
            return cached.copy(), None

        # Never serialize unrelated model forwards behind a cache lock.
        vector, stages = self._encode_query_profiled(text)
        if np.isfinite(vector).all() and np.any(vector):
            with self._query_cache_lock:
                self._query_cache[text] = vector.copy()
                self._query_cache.move_to_end(text)
                if len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)
        return vector, stages

    def _encode_query_profiled(self, text: str) -> tuple[np.ndarray, EmbeddingStages | None]:
        """Run canonical encoding, separating device work from host conversion."""
        encoder = self._query_onnx
        if encoder is not None:
            try:
                return encoder.encode(text)
            except Exception as exc:  # noqa: BLE001 - a broken native session can fall back
                with self._torch_lock:
                    if self._query_onnx is encoder:
                        logger.warning(
                            'query ONNX failed; switching to PyTorch (%s)', type(exc).__name__
                        )
                        self._torch_persistent = True
                        self._query_onnx = None
                # Release the failed session before loading another set of weights.
                del encoder
        spans: list[DensePhase] = []
        token = _preprocess_spans.set(spans)
        try:
            with self._torch_model() as model:
                (vector,) = model.encode(
                    [text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=False,
                )
        finally:
            _preprocess_spans.reset(token)
        # Wait for a stream marker, not the whole device. Otherwise an async
        # forward would be charged to the blocking host copy. This is wall time.
        if vector.device.type == 'mps':
            complete = torch.mps.Event()
            complete.record()
            complete.synchronize()
        elif vector.device.type in {'cuda', 'xpu'}:
            getattr(torch, vector.device.type).current_stream(vector.device).synchronize()
        conversion_started = perf_counter()
        output = vector.to(device='cpu', dtype=torch.float32).numpy()
        finished = perf_counter()
        if len(spans) != 1:
            return output, None
        return output, EmbeddingStages(
            tokenization=spans[0],
            model=DensePhase(spans[0].finished, conversion_started),
            cpu_conversion=DensePhase(conversion_started, finished),
        )
