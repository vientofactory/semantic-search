"""Tests for stage 4b: lexical evidence gating and RRF fusion.

Uses plain chunk dicts and stub hits so no embedding model is needed. The
separation test mirrors the measured score overlap in
agent_memories/20-semantic-query-quality-eval/ §10.4: a noise query can score
higher than a valid vague query, so any absolute cosine floor accepts both or
rejects both, while shared surface tokens still separate them.
"""

from types import SimpleNamespace

import pytest

from lawcast_semantic.hybrid import (
    chunk_tokens,
    filter_by_lexical_evidence,
    has_lexical_evidence,
    lexical_tokens,
    overlap_count,
    rrf_fuse,
)


def make_chunk(chunk_id: str, subject: str, text: str) -> dict:
    return {
        'chunk_id': chunk_id,
        'notice_num': 1,
        'subject': subject,
        'committee': '위원회',
        'section': '제안이유',
        'text': text,
    }


def test_lexical_tokens_drop_stopwords_and_single_chars():
    tokens = lexical_tokens('법률안 개정에 관한 주요 내용')
    assert '개정에' in tokens or '개정' in tokens
    assert '관한' not in tokens
    assert '법률안' not in tokens
    assert all(len(token) >= 2 for token in tokens)


def test_lexical_tokens_deduplicate_and_expand_aliases():
    tokens = lexical_tokens('중처법 중처법')
    assert '중대재해' in tokens
    assert tokens.count('중대재해') == 1


def test_overlap_count_counts_shared_tokens():
    assert overlap_count(['임대차', '보호'], ['임대차', '세입자']) == 1
    assert overlap_count(['점심', '메뉴'], ['임대차', '보호']) == 0


def test_has_lexical_evidence_matches_subject_or_text():
    chunk = make_chunk('c1', '주택임대차보호법 일부개정법률안', '임차인의 보증금을 보호한다')
    assert has_lexical_evidence('임대차 보호', chunk) is True
    assert has_lexical_evidence('오늘 점심 메뉴', chunk) is False
    assert has_lexical_evidence('   ', chunk) is False


def test_filter_preserves_order_and_drops_unrelated():
    chunks = {
        'c1': make_chunk('c1', '주택임대차보호법', '임차인 보호'),
        'c2': make_chunk('c2', '동물보호법', '유기동물 처벌'),
    }
    hits = [SimpleNamespace(chunk_id='c1'), SimpleNamespace(chunk_id='c2')]
    kept = filter_by_lexical_evidence('임대차 보증금 대책', hits, chunks)
    assert [hit.chunk_id for hit in kept] == ['c1']


def test_filter_returns_empty_for_unrelated_query():
    chunks = {
        'c1': make_chunk('c1', '주택임대차보호법', '임차인 보호'),
        'c2': make_chunk('c2', '동물보호법', '유기동물 처벌'),
    }
    hits = [SimpleNamespace(chunk_id='c1'), SimpleNamespace(chunk_id='c2')]
    assert filter_by_lexical_evidence('오늘 점심 메뉴', hits, chunks) == []


def test_no_cosine_floor_separates_overlapping_scores():
    """Any absolute floor fails where lexical evidence succeeds.

    Measured overlap (§10.4): noise `오늘 점심 메뉴` top-1 0.536 scores above
    valid `그거 어떻게 되나요` 0.480. Sweep every floor between them: the floor
    either keeps both windows or empties both, never just the noise one.
    """
    noise_top, valid_top = 0.536, 0.480
    for floor in (0.25, 0.45, 0.50, 0.55):
        noise_kept = noise_top >= floor
        valid_kept = valid_top >= floor
        assert (noise_kept, valid_kept) != (False, True)

    chunks = {'c1': make_chunk('c1', '환경보호를 위한 공원법', '자연공원을 보호한다')}
    hits = [SimpleNamespace(chunk_id='c1')]
    assert filter_by_lexical_evidence('오늘 점심 메뉴', hits, chunks) == []
    assert filter_by_lexical_evidence('환경 보호', hits, chunks) != []


def test_rrf_fuse_prefers_ids_ranked_high_in_both_lists():
    fused = rrf_fuse([['a', 'b'], ['b', 'c']])
    assert fused['b'] > fused['a']
    assert fused['b'] > fused['c']
    same_order = rrf_fuse([['a', 'b', 'c'], ['a', 'b', 'c']])
    assert same_order['a'] > same_order['b'] > same_order['c']


def test_rrf_fuse_rejects_nonpositive_k():
    with pytest.raises(ValueError, match='k must be >= 1'):
        rrf_fuse([['a']], k=0)


def test_chunk_tokens_combine_subject_and_text():
    chunk = make_chunk('c1', '산업안전보건법', '사업장의 안전을 강화한다')
    tokens = chunk_tokens(chunk)
    assert '산업안전보건법' in tokens
    assert '사업장의' in tokens or '안전을' in tokens
