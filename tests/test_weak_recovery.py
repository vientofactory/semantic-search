import pytest

from lawcast_semantic import config
from lawcast_semantic.search import SearchResult, SemanticSearcher


def hit(notice, score, subject, text='', suffix=0):
    return SearchResult(f'{notice}-{suffix}', score, notice, subject, '', '', text)


@pytest.fixture
def searcher(monkeypatch):
    monkeypatch.setattr(config, 'MIN_SIMILARITY', 0.49)
    monkeypatch.setattr(config, 'CLEAR_SIMILARITY', 0.51)
    return SemanticSearcher(None, None, [], {})


def test_recovers_distinct_same_law_amendment_only_as_weak(searcher):
    anchor = hit(1, 0.60, '여성기업지원에 관한 법률 일부개정법률안')
    existing_weak = hit(2, 0.49, '여성기업지원에 관한 법률 일부개정법률안')
    another_law = hit(3, 0.48, '중소기업창업 지원법 일부개정법률안')
    candidate = hit(4, 0.45, '여성기업지원에 관한 법률 전부개정법률안 (대안)')
    hits = [
        anchor,
        existing_weak,
        another_law,
        hit(1, 0.47, anchor.subject, suffix=1),
        hit(2, 0.46, existing_weak.subject, suffix=1),
        candidate,
        hit(5, 0.449, candidate.subject),
    ]

    result = searcher.tier_candidates('여성 CEO 창업 지원', None, hits)

    assert result.results == [anchor]
    assert result.weak_results == [existing_weak, candidate]
    assert result.candidate_count is None


@pytest.mark.parametrize('anchor_score', [0.48, 0.59])
def test_never_recovers_below_floor_without_high_confidence_anchor(searcher, anchor_score):
    title = '여성기업지원에 관한 법률 일부개정법률안'
    result = searcher.tier_candidates(
        '여성 CEO 창업 지원', None, [hit(1, anchor_score, title), hit(2, 0.47, title)]
    )
    assert result.weak_results == []
    assert len(result.results) == int(anchor_score >= 0.51)


def test_same_law_still_requires_lexical_evidence(searcher):
    title = '물환경보전법 일부개정법률안'
    result = searcher.tier_candidates(
        '폐수 비용', None, [hit(1, 0.64, title, '폐수 비용'), hit(2, 0.47, title)]
    )
    assert result.weak_results == []
