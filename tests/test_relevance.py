from unittest.mock import Mock

import numpy as np

from lawcast_semantic.indexing import VectorIndex
from lawcast_semantic.relevance import DEFAULT_CUTOFFS, Evidence, RelevanceProfile, classify
from lawcast_semantic.search import SemanticSearcher


def test_inclusive_boundaries_and_paraphrase_path():
    c = DEFAULT_CUTOFFS
    assert (
        classify(Evidence(c.strong_cosine, c.strong_z, c.strong_coverage), 0.25, 0.45) == 'strong'
    )
    assert classify(Evidence(c.semantic_cosine, c.semantic_z, 0), 0.25, 0.45) == 'strong'
    assert classify(Evidence(c.weak_cosine, c.weak_z, c.weak_coverage), 0.25, 0.45) == 'weak'
    assert classify(Evidence(0.99, 0.0, 0.0), 0.25, 0.45) == 'discard'
    assert classify(Evidence(float('nan'), 100, 1), 0.25, 0.45) == 'discard'


def test_degenerate_background_allows_only_exact_evidence_and_never_promotes():
    vectors = np.ones((64, 1), dtype=np.float32)
    profile = RelevanceProfile(vectors, [{'text': '반복'}] * len(vectors))
    stats = profile.for_query('반복 희귀 희귀', np.ones(1, dtype=np.float32))
    assert stats.background is None
    assert stats.weights['반복'] < stats.weights['희귀']
    assert stats.measure(0.9, {'text': '희귀어'}).coverage == 0
    assert stats.measure(0.9, {'text': '희귀'}).coverage < 1
    assert stats.measure(0.9, {'text': '희귀 반복'}).coverage == 1
    assert classify(stats.measure(0.9, {'text': '희귀 반복'}), 0.25, 0.45) == 'weak'


def test_profile_uses_generation_vectors_and_search_embeds_once():
    count = 64
    index = VectorIndex(count)
    index.build(np.eye(count, dtype=np.float32))
    ids = [str(i) for i in range(count)]
    records = {
        key: {
            'notice_num': i,
            'subject': '주제',
            'committee': '',
            'section': '',
            'text': '본문',
        }
        for i, key in enumerate(ids)
    }
    profile = RelevanceProfile.from_index(index, ids, records)
    embedder = Mock(spec=['embed_query'])
    embedder.embed_query.return_value = np.ones(count, dtype=np.float32) / np.sqrt(count)
    engine = SemanticSearcher(embedder, index, ids, records, relevance_profile=profile)

    result = engine.search_tiered('무관한 입력', k=5)

    assert result.candidate_count == 0
    assert result.results == result.weak_results == []
    embedder.embed_query.assert_called_once_with('무관한 입력')


def test_above_floor_candidates_remain_counted_after_relevance_filtering():
    vectors = np.ones((64, 1), dtype=np.float32)
    index = VectorIndex(1)
    index.build(vectors)
    ids = [str(i) for i in range(len(vectors))]
    records = {
        key: dict(notice_num=i, subject='반복', committee='', section='', text='반복')
        for i, key in enumerate(ids)
    }
    profile = RelevanceProfile.from_index(index, ids, records)
    embedder = Mock(spec=['embed_query'])
    embedder.embed_query.return_value = np.ones(1, dtype=np.float32)
    engine = SemanticSearcher(embedder, index, ids, records, relevance_profile=profile)

    result = engine.search_tiered('무관한 입력', k=5)

    assert result.candidate_count == 5
    assert result.results == result.weak_results == []
