"""Tests for stage 4a: citizen alias expansion.

Covers the alias table contract, the standalone/particle boundary rules, and the
wiring that makes `SemanticSearcher.search` embed the expanded query text.
"""

import numpy as np

from lawcast_semantic.aliases import ALIASES, expand_query
from lawcast_semantic.search import SemanticSearcher


def test_alias_expands_to_official_bill_name():
    assert expand_query('중처법').text == '중대재해 처벌 등에 관한 법률'
    assert expand_query('산안법').text == '산업안전보건법'
    assert expand_query('전상법').text == '전자상거래 등에서의 소비자보호에 관한 법률'


def test_matched_reports_declared_alias():
    assert expand_query('중처법').matched == ('중처법',)
    assert expand_query('오늘 날씨').matched == ()


def test_expansion_is_in_place_and_keeps_surrounding_text():
    expansion = expand_query('중처법 위반 처벌 강화')
    assert expansion.text == '중대재해 처벌 등에 관한 법률 위반 처벌 강화'
    assert expansion.matched == ('중처법',)


def test_multiple_aliases_in_one_query_expand_in_occurrence_order():
    expansion = expand_query('산안법과 중처법')
    assert expansion.text == '산업안전보건법과 중대재해 처벌 등에 관한 법률'
    assert expansion.matched == ('산안법', '중처법')


def test_repeated_alias_is_reported_once():
    expansion = expand_query('중처법 중처법')
    assert expansion.text == '중대재해 처벌 등에 관한 법률 중대재해 처벌 등에 관한 법률'
    assert expansion.matched == ('중처법',)


def test_trailing_particle_is_matched_and_re_emitted():
    assert expand_query('중처법은 개정되나요').text == ('중대재해 처벌 등에 관한 법률은 개정되나요')
    assert expand_query('산안법상 의무').text == '산업안전보건법상 의무'
    assert expand_query('상증법으로').text == '상속세 및 증여세법으로'


def test_alias_embedded_in_a_longer_word_is_left_alone():
    """A boundary rule, not a substring rule: the acronym must stand alone."""
    expansion = expand_query('산안법률 개정')
    assert expansion.text == '산안법률 개정'
    assert expansion.matched == ()


def test_unknown_query_passes_through_unchanged():
    query = '세입자 보호 대책 알려줘'
    expansion = expand_query(query)
    assert expansion.text == query
    assert expansion.matched == ()


def test_latin_alias_matches_any_case():
    official = '인공지능 발전과 신뢰 기반 조성 등에 관한 기본법'
    assert expand_query('AI 기본법').text == official
    assert expand_query('ai 기본법').text == official
    assert expand_query('Ai기본법').text == official


def test_multiword_alias_tolerates_missing_space():
    assert expand_query('인공지능기본법').text == (
        '인공지능 발전과 신뢰 기반 조성 등에 관한 기본법'
    )


def test_empty_and_none_queries_return_empty_expansion():
    assert expand_query('') == expand_query(None)
    assert expand_query(None).text == ''
    assert expand_query(None).matched == ()


def test_alias_table_integrity():
    """Every entry must rewrite something, and none may point at its own key."""
    for alias, official in ALIASES.items():
        assert alias.strip(), 'alias keys must not be blank'
        assert official.strip(), f'{alias} maps to a blank official name'
        assert alias != official, f'{alias} maps to itself'
        assert alias.casefold() not in official.casefold(), (
            f'{alias} is contained in its own canonical name; the entry is a no-op or loops'
        )


class _RecordingEmbedder:
    """Captures the text handed to `embed_query` and returns a fixed vector."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.model_name = 'recording-model'
        self.dimension = 4

    def embed_query(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.zeros(4, dtype='float32')


class _EmptyIndex:
    """Stands in for VectorIndex: the assertion is about the embedded text."""

    def search(self, query_vector, k=None):
        del query_vector, k  # signature parity with VectorIndex.search
        return []


def test_searcher_embeds_the_expanded_query_text():
    embedder = _RecordingEmbedder()
    searcher = SemanticSearcher(embedder, _EmptyIndex(), [], {})

    assert searcher.search('중처법 위반', k=3) == []
    assert embedder.texts == ['중대재해 처벌 등에 관한 법률 위반']


def test_searcher_leaves_plain_queries_untouched():
    embedder = _RecordingEmbedder()
    searcher = SemanticSearcher(embedder, _EmptyIndex(), [], {})

    searcher.search('세입자 보호', k=3)
    assert embedder.texts == ['세입자 보호']


def test_searcher_does_not_embed_a_whitespace_query():
    embedder = _RecordingEmbedder()
    searcher = SemanticSearcher(embedder, _EmptyIndex(), [], {})

    assert searcher.search('   ', k=3) == []
    assert embedder.texts == []
