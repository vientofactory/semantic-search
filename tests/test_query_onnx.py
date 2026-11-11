"""Lazy query and document routing with no model downloads."""

import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from lawcast_semantic import config


@pytest.fixture
def setup_encoder(monkeypatch):
    from lawcast_semantic import embedding

    def build(*, graph_failure=False, document_gate=None):
        loads = []
        refs = []
        corpus_calls = []
        query_calls = []
        unavailable = Event()

        class Model:
            tokenizer = object()
            max_seq_length = 32

            def get_embedding_dimension(self):
                return 2

            def encode(self, texts, **kwargs):
                if kwargs.get('convert_to_numpy') is False:
                    import torch

                    return torch.tensor([[1.0, 0.0]])
                corpus_calls.extend(texts)
                if document_gate:
                    document_gate[0].set()
                    assert document_gate[1].wait(timeout=5)
                return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

        def torch_load(*args):
            loads.append(args)
            model = Model()
            model.cycle = model
            refs.append(weakref.ref(model))
            return 'cpu', model

        def encode(text):
            query_calls.append(text)
            if unavailable.is_set():
                raise RuntimeError('injected query failure')
            return np.array([0.6, 0.8], dtype=np.float32), None

        runner = SimpleNamespace(
            tokenizer=object(),
            max_seq_length=32,
            dimension=2,
            revision='a' * 40,
            probe_delta=0.0,
            encode=encode,
        )

        def load(*args):
            if graph_failure:
                raise FileNotFoundError('missing test bundle')
            return runner

        monkeypatch.setattr(embedding.KoreanEmbedder, '_load_model', staticmethod(torch_load))
        monkeypatch.setattr(embedding, 'OnnxQueryEncoder', load)
        monkeypatch.setattr(config, 'QUERY_ONNX_PATH', '/tmp/test-model.onnx')
        monkeypatch.setattr(config, 'QUERY_EMBED_CACHE_SIZE', 2)
        return SimpleNamespace(
            embedder=embedding.KoreanEmbedder('dragonkue/BGE-m3-ko', device='cpu'),
            loads=loads,
            refs=refs,
            corpus_calls=corpus_calls,
            query_calls=query_calls,
            unavailable=unavailable,
        )

    return build


def test_queries_start_without_torch_and_document_batches_release_weights(setup_encoder):
    env = setup_encoder()
    assert env.loads == []
    env.embedder.embed_query_profiled('보호')
    env.embedder.embed_query_profiled('보호')
    env.embedder.embed_query('보호')
    assert env.query_calls == ['보호']
    assert env.loads == []
    assert env.embedder.embed_texts([]).shape == (0, 2)
    assert env.loads == []

    np.testing.assert_array_equal(env.embedder.embed_texts(['입법예고']), [[1.0, 0.0]])
    assert env.loads == [('dragonkue/BGE-m3-ko', 'cpu', 'a' * 40)]
    assert env.corpus_calls == ['입법예고']
    assert env.embedder.model is None
    assert env.refs[0]() is None
    env.embedder.embed_query_profiled('다른 질의')
    assert env.query_calls == ['보호', '다른 질의']
    assert len(env.loads) == 1


def test_missing_bundle_uses_existing_torch_path(setup_encoder):
    env = setup_encoder(graph_failure=True)
    assert len(env.loads) == 1
    assert env.embedder._query_onnx is None
    assert env.embedder.embed_query_profiled('보호')[0].tolist() == [1.0, 0.0]


def test_runtime_fallback_reuses_document_model_without_duplicate_load(setup_encoder):
    started, release = Event(), Event()
    env = setup_encoder(document_gate=(started, release))
    with ThreadPoolExecutor(2) as pool:
        document = pool.submit(env.embedder.embed_texts, ['입법예고'])
        try:
            assert started.wait(timeout=5)
            env.unavailable.set()
            fallback = pool.submit(env.embedder.embed_query_profiled, '보호')
            assert fallback.result(timeout=5)[0].tolist() == [1.0, 0.0]
        finally:
            release.set()
        np.testing.assert_array_equal(document.result(timeout=5), [[1.0, 0.0]])
    assert len(env.loads) == 1
    assert env.embedder.model is not None
    assert env.embedder._query_onnx is None
    env.embedder.embed_query_profiled('다른 질의')
    assert env.query_calls == ['보호']
    assert len(env.loads) == 1
