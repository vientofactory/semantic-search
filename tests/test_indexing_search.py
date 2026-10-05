"""Tests for stages 3-4: FAISS round-trip, ranking determinism, artifact sync.

Uses deterministic fake vectors so the tests run without downloading the
embedding model.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lawcast_semantic.chunking import compute_chunks_fingerprint, write_chunks_jsonl
from lawcast_semantic.indexing import VectorIndex
from lawcast_semantic.search import SemanticSearcher


class StubEmbedder:
    """Returns a pre-programmed vector per query for deterministic tests."""

    def __init__(
        self,
        vector: np.ndarray,
        model_name: str = 'stub-model',
        dimension: int = 4,
    ) -> None:
        self.vector = vector
        self.model_name = model_name
        self.dimension = dimension

    def embed_query(self, text: str) -> np.ndarray:
        return self.vector


def make_vectors() -> np.ndarray:
    return np.eye(4, dtype='float32')


def make_records(count: int = 4) -> list[dict]:
    return [
        {
            'chunk_id': f'chunk-{i}',
            'notice_num': 100 + i,
            'subject': f'법률안 {i}',
            'committee': '교육위원회',
            'section': '제안이유',
            'text': f'내용 {i}',
        }
        for i in range(count)
    ]


def build_artifacts(
    tmp_path: Path,
    records: list[dict],
    vectors: np.ndarray,
    model_name: str = 'stub-model',
) -> tuple[Path, Path, Path]:
    """Write chunks.jsonl + faiss.index + id_map.json exactly like stage 1+3."""
    index = VectorIndex(vectors.shape[1])
    index.build(vectors)
    chunk_ids = [record['chunk_id'] for record in records]
    index_path = tmp_path / 'faiss.index'
    id_map_path = tmp_path / 'id_map.json'
    index.save(
        index_path,
        id_map_path,
        chunk_ids,
        meta={
            'chunks_fingerprint': compute_chunks_fingerprint(records),
            'model_name': model_name,
        },
    )
    chunks_path = tmp_path / 'chunks.jsonl'
    write_chunks_jsonl(records, chunks_path)
    return chunks_path, index_path, id_map_path


def test_index_round_trip_and_nearest_neighbor(tmp_path: Path):
    vectors = make_vectors()
    index = VectorIndex(vectors.shape[1])
    index.build(vectors)
    chunk_ids = [f'chunk-{i}' for i in range(len(vectors))]
    index_path = tmp_path / 'faiss.index'
    id_map_path = tmp_path / 'id_map.json'
    index.save(index_path, id_map_path, chunk_ids)

    loaded, payload = VectorIndex.load(index_path, id_map_path)
    assert payload['chunk_ids'] == chunk_ids
    hits = loaded.search(vectors[2], k=2)
    assert hits[0][0] == 2  # nearest neighbor of vector 2 is itself
    assert hits[0][1] > 0.99


def test_search_rejects_nonpositive_k(tmp_path: Path):
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)
    searcher = SemanticSearcher.load(
        StubEmbedder(vectors[0]),
        chunks_path=chunks_path,
        index_path=index_path,
        id_map_path=id_map_path,
    )
    with pytest.raises(ValueError, match='k must be >= 1'):
        searcher.search('질의', k=0)
    with pytest.raises(ValueError, match='k must be >= 1'):
        searcher.index.search(vectors[0], k=-1)


def test_build_rejects_wrong_dimension():
    index = VectorIndex(4)
    with pytest.raises(ValueError, match='does not match index dim'):
        index.build(np.ones((3, 2), dtype='float32'))


def test_searcher_ranks_and_maps_metadata(tmp_path: Path):
    """Equal-score ties must resolve to stable chunk order (chunk-3, chunk-0)."""
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)

    searcher = SemanticSearcher.load(
        StubEmbedder(vectors[3]),
        chunks_path=chunks_path,
        index_path=index_path,
        id_map_path=id_map_path,
    )
    results = searcher.search('세 번째와 유사한 질의', k=2)
    assert [result.chunk_id for result in results] == ['chunk-3', 'chunk-0']
    assert results[0].score >= results[1].score
    assert results[0].subject == '법률안 3'
    assert results[0].notice_num == 103


def test_duplicate_content_ranks_by_chunk_id(tmp_path: Path):
    """Identical vectors (duplicate content) must not flip order between runs."""
    records = make_records()
    vectors = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype='float32',
    )
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)

    searcher = SemanticSearcher.load(
        StubEmbedder(vectors[0]),
        chunks_path=chunks_path,
        index_path=index_path,
        id_map_path=id_map_path,
    )
    results = searcher.search('중복 내용 질의', k=3)
    assert [result.chunk_id for result in results] == ['chunk-0', 'chunk-1', 'chunk-2']


def test_searcher_rejects_stale_chunks(tmp_path: Path):
    """Re-running stage 1 alone must not silently mix stale scores with new text."""
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)

    records[1]['text'] = '완전히 다른 텍스트로 바뀌었지만 임베딩은 그대로임'
    write_chunks_jsonl(records, chunks_path)
    with pytest.raises(ValueError, match='out of sync'):
        SemanticSearcher.load(
            StubEmbedder(vectors[0]),
            chunks_path=chunks_path,
            index_path=index_path,
            id_map_path=id_map_path,
        )


def test_searcher_rejects_stale_subject_change(tmp_path: Path):
    """The subject is embedded as context, so editing it invalidates vectors too."""
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)

    records[1]['subject'] = '완전히 다른 제목'
    write_chunks_jsonl(records, chunks_path)
    with pytest.raises(ValueError, match='out of sync'):
        SemanticSearcher.load(
            StubEmbedder(vectors[0]),
            chunks_path=chunks_path,
            index_path=index_path,
            id_map_path=id_map_path,
        )


def test_searcher_rejects_index_id_map_row_mismatch(tmp_path: Path):
    """An index and id map from different builds must fail loudly, not mislabel hits."""
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)

    payload = json.loads(id_map_path.read_text(encoding='utf-8'))
    payload['chunk_ids'] = payload['chunk_ids'] + ['chunk-extra']
    id_map_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='different builds'):
        SemanticSearcher.load(
            StubEmbedder(vectors[0]),
            chunks_path=chunks_path,
            index_path=index_path,
            id_map_path=id_map_path,
        )


def test_searcher_rejects_model_mismatch(tmp_path: Path):
    """Same-dim different-model queries must fail loudly, not return garbage."""
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(
        tmp_path, records, vectors, model_name='stub-model'
    )
    with pytest.raises(ValueError, match='model'):
        SemanticSearcher.load(
            StubEmbedder(vectors[0], model_name='other-model'),
            chunks_path=chunks_path,
            index_path=index_path,
            id_map_path=id_map_path,
        )


def test_searcher_rejects_dimension_mismatch(tmp_path: Path):
    records = make_records()
    vectors = make_vectors()
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)
    with pytest.raises(ValueError, match='dimension'):
        SemanticSearcher.load(
            StubEmbedder(vectors[0], dimension=8),
            chunks_path=chunks_path,
            index_path=index_path,
            id_map_path=id_map_path,
        )


def test_empty_query_returns_nothing(tmp_path: Path):
    vectors = make_vectors()
    index = VectorIndex(vectors.shape[1])
    index.build(vectors)
    searcher = SemanticSearcher(StubEmbedder(vectors[0]), index, [], {})
    assert searcher.search('   ', k=3) == []


def _tiered_searcher(tmp_path: Path, vectors: np.ndarray, query: np.ndarray) -> SemanticSearcher:
    records = make_records(len(vectors))
    chunks_path, index_path, id_map_path = build_artifacts(tmp_path, records, vectors)
    return SemanticSearcher.load(
        StubEmbedder(query),
        chunks_path=chunks_path,
        index_path=index_path,
        id_map_path=id_map_path,
    )


def test_search_tiered_splits_clear_weak_and_unrelated(tmp_path: Path, monkeypatch):
    """score >= CLEAR -> clear, MIN <= score < CLEAR -> weak, below MIN -> dropped."""
    from lawcast_semantic import config

    monkeypatch.setattr(config, 'MIN_SIMILARITY', 0.25)
    monkeypatch.setattr(config, 'CLEAR_SIMILARITY', 0.45)
    # Query vector is row 0 (e_0), so each row's cosine score is known:
    # 1.0 (clear), 0.35 (weak), 0.0 and 0.1 (unrelated, dropped).
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.35, 0.9367497, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 0.9949874, 0.0],
        ],
        dtype='float32',
    )
    searcher = _tiered_searcher(tmp_path, vectors, vectors[0])

    outcome = searcher.search_tiered('질의', k=4)

    assert [hit.chunk_id for hit in outcome.results] == ['chunk-0']
    assert outcome.results[0].score == pytest.approx(1.0, abs=1e-5)
    assert [hit.chunk_id for hit in outcome.weak_results] == ['chunk-1']
    assert outcome.weak_results[0].score == pytest.approx(0.35, abs=1e-5)
    # Unrelated hits appear in neither tier.
    listed = {hit.chunk_id for hit in outcome.results + outcome.weak_results}
    assert listed.isdisjoint({'chunk-2', 'chunk-3'})


def test_search_tiered_returns_nothing_when_everything_is_unrelated(tmp_path: Path, monkeypatch):
    """The 'no clear results' contract: nothing qualifies -> two empty lists."""
    from lawcast_semantic import config

    monkeypatch.setattr(config, 'MIN_SIMILARITY', 0.25)
    monkeypatch.setattr(config, 'CLEAR_SIMILARITY', 0.45)
    # All rows score below the 0.25 floor against query vector e_0.
    vectors = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.2, 0.9797959, 0.0, 0.0],
        ],
        dtype='float32',
    )
    searcher = _tiered_searcher(
        tmp_path, vectors, np.asarray([1.0, 0.0, 0.0, 0.0], dtype='float32')
    )

    outcome = searcher.search_tiered('질의', k=4)

    assert outcome.results == []
    assert outcome.weak_results == []
