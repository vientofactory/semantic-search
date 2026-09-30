"""Tests for stage 1a: normalization and section detection."""

import unicodedata

from lawcast_semantic.preprocess import (
    DEFAULT_SECTION,
    detect_sections,
    normalize_text,
    split_paragraphs,
)


def test_normalize_preserves_line_breaks():
    text = '제안이유\n최근 ESG 공시가 확산되고 있음.\n\n\n\n기업가치에 영향을 줌.'
    normalized = normalize_text(text)
    assert '\n' in normalized
    assert '\n\n\n' not in normalized
    assert normalized.split('\n')[0] == '제안이유'


def test_normalize_collapses_horizontal_whitespace_and_strips_html():
    text = '<p>대한민국은</p>   \t 지방소멸\t위기에  직면함.'
    normalized = normalize_text(text)
    assert '<p>' not in normalized
    assert '  ' not in normalized
    assert normalized.endswith('직면함.')


def test_normalize_applies_nfc():
    decomposed = unicodedata.normalize('NFD', '한글 텍스트')
    assert normalize_text(decomposed) == '한글 텍스트'


def test_normalize_handles_empty_values():
    assert normalize_text(None) == ''
    assert normalize_text('') == ''


def test_split_paragraphs_drops_blank_lines():
    assert split_paragraphs('가. 항목\n\n나. 항목\n') == ['가. 항목', '나. 항목']


def test_detect_sections_groups_by_header():
    text = (
        '의안번호 1234\n'
        '제안이유\n'
        '최근 문제가 있음.\n'
        '주요내용\n'
        '가. 첫 번째 내용임.\n'
        '나. 두 번째 내용임.\n'
    )
    sections = detect_sections(text)
    assert [name for name, _ in sections] == [DEFAULT_SECTION, '제안이유', '주요내용']
    assert sections[0][1] == '의안번호 1234'
    assert sections[1][1] == '최근 문제가 있음.'
    assert '가. 첫 번째 내용임.' in sections[2][1]


def test_detect_sections_drops_leading_empty_default_section():
    # Text starting with a header has no preamble, so no 'body' section.
    sections = detect_sections('제안이유\n최근 문제가 있음.')
    assert [name for name, _ in sections] == ['제안이유']


def test_detect_sections_combined_header():
    text = '제안이유 및 주요내용\n\n최근 ESG 공시가 확산되고 있음.'
    sections = detect_sections(text)
    assert sections[0][0] == '제안이유 및 주요내용'
    assert 'ESG' in sections[0][1]
