"""Exact query-vector cache tests with a stub encoder and no model weights."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

from lawcast_semantic import config
from lawcast_semantic.timing import DensePhase, EmbeddingStages

STAGES = EmbeddingStages(*(DensePhase(1.0, 2.0) for _ in range(3)))


@pytest.fixture
def make_embedder(monkeypatch):
    from lawcast_semantic import embedding

    def build(capacity, encode):
        model = SimpleNamespace(
            tokenizer=object(), max_seq_length=256, get_embedding_dimension=lambda: 2
        )
        monkeypatch.setattr(config, 'QUERY_EMBED_CACHE_SIZE', capacity)
        monkeypatch.setattr(
            embedding.KoreanEmbedder, '_load_model', staticmethod(lambda *_: ('cpu', model))
        )
        embedder = embedding.KoreanEmbedder()
        monkeypatch.setattr(embedder, '_encode_query_profiled', encode)
        return embedder

    return build


def test_cache_is_bounded_lru_and_never_exposes_shared_vectors(make_embedder):
    calls = []

    def encode(text):
        calls.append(text)
        return np.array([0.6, 0.8], dtype=np.float32), STAGES

    embedder = make_embedder(2, encode)
    original, stages = embedder.embed_query_profiled('a')
    assert stages is STAGES
    original[:] = 0
    embedder.embed_query_profiled('b')
    reused, stages = embedder.embed_query_profiled('a')
    assert stages is None
    np.testing.assert_array_equal(reused, np.array([0.6, 0.8], dtype=np.float32))
    reused[:] = 1
    embedder.embed_query_profiled('c')
    np.testing.assert_array_equal(
        embedder.embed_query_profiled('a')[0], np.array([0.6, 0.8], dtype=np.float32)
    )
    embedder.embed_query_profiled('b')
    assert calls == ['a', 'b', 'c', 'b']


@pytest.mark.parametrize('value', [0.0, float('nan'), float('inf')])
def test_invalid_vectors_are_not_cached(make_embedder, value):
    calls = []

    def encode(text):
        calls.append(text)
        return np.full(2, value, dtype=np.float32), STAGES

    embedder = make_embedder(2, encode)
    for _ in range(2):
        embedder.embed_query_profiled('a')
    assert calls == ['a', 'a']


def test_disabled_cache_and_new_model_both_encode_again(make_embedder):
    calls = []

    def encode(text):
        calls.append(text)
        return np.ones(2, dtype=np.float32), STAGES

    disabled = make_embedder(0, encode)
    for _ in range(2):
        assert disabled.embed_query_profiled('a')[1] is STAGES
    first = make_embedder(1, encode)
    second = make_embedder(1, encode)
    first.embed_query_profiled('a')
    assert second.embed_query_profiled('a')[1] is STAGES
    assert first.embed_query_profiled('a')[1] is None
    assert calls == ['a'] * 4


def test_warmup_bypasses_lru_and_never_loads_fallback(make_embedder):
    calls = []

    def encode(text):
        calls.append(text)
        return np.array([0.6, 0.8], dtype=np.float32), STAGES

    embedder = make_embedder(2, encode)
    embedder.embed_query_profiled('cached')
    embedder._query_onnx = SimpleNamespace(encode=encode)
    first = embedder.warm_query()
    second = embedder.warm_query()
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 3
    assert list(embedder._query_cache) == ['cached']
    embedder._query_onnx = None
    assert embedder.warm_query() is None
    assert len(calls) == 3


def test_warmup_failure_does_not_retire_onnx(make_embedder):
    embedder = make_embedder(0, lambda _: (np.ones(2), STAGES))

    def fail(_):
        raise RuntimeError('native failure')

    encoder = SimpleNamespace(encode=fail)
    embedder._query_onnx = encoder
    with pytest.raises(RuntimeError, match='native failure'):
        embedder.warm_query()
    assert embedder._query_onnx is encoder
    assert not embedder._torch_users


def test_only_native_inference_and_upkeep_refresh_the_activity_clock(make_embedder, monkeypatch):
    from lawcast_semantic import embedding

    embedder = make_embedder(2, lambda _: (np.ones(2), STAGES))
    embedder._query_onnx = SimpleNamespace(encode=lambda _: (np.ones(2), STAGES))
    monkeypatch.setattr(embedding, 'monotonic', lambda: 123.0)
    embedding.KoreanEmbedder._encode_query_profiled(embedder, 'fresh')
    assert embedder.last_query_inference_at == 123.0
    embedder._query_cache['cached'] = np.ones(2)
    monkeypatch.setattr(embedding, 'monotonic', lambda: 456.0)
    embedder.embed_query_profiled('cached')
    assert embedder.last_query_inference_at == 123.0
    embedder.warm_query()
    assert embedder.last_query_inference_at == 456.0


def test_unrelated_cache_misses_do_not_serialize_model_work(make_embedder):
    concurrent = Barrier(2, timeout=5)

    def encode(text):
        concurrent.wait()
        return np.array([float(text), 1.0], dtype=np.float32), STAGES

    embedder = make_embedder(2, encode)
    with ThreadPoolExecutor(2) as workers:
        first, second = workers.map(embedder.embed_query_profiled, ('1', '2'))
    assert first[0].tolist() == [1.0, 1.0]
    assert second[0].tolist() == [2.0, 1.0]
    assert embedder.embed_query_profiled('1')[1] is None
    assert embedder.embed_query_profiled('2')[1] is None
