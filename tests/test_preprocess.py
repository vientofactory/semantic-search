"""Tests for stage 1a: normalization and section detection."""

import unicodedata

from lawcast_semantic.preprocess import (
    DEFAULT_SECTION,
    detect_sections,
    normalize_text,
    split_paragraphs,
    split_section_header,
    strip_glued_prefixes,
    strip_item_marker,
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


def test_normalize_text_keeps_structure_tokens():
    """`normalize_text` is shared with the QUERY path, so it must not delete text.

    Both transforms are indexing-only: a user is allowed to type '가.' or paste
    a header, and silently editing the query would change what they searched.
    """
    assert normalize_text('제안이유\n가. 항목임.') == '제안이유\n가. 항목임.'


def test_split_section_header_accepts_glued_content():
    """The corpus writes the label glued to its paragraph far more often than
    on a line of its own; that label is 11 characters of corpus-wide boilerplate
    and must not reach the embedder."""
    label, tail = split_section_header('제안이유 및 주요내용 현행법은 미비함.')
    assert label == '제안이유 및 주요내용'
    assert tail == '현행법은 미비함.'


def test_split_section_header_accepts_the_alternative_qualifier():
    """Committee alternatives qualify the label ('대안의 제안이유')."""
    assert split_section_header('대안의 제안이유 및 주요내용') == ('제안이유 및 주요내용', '')
    assert split_section_header('대안의 주요내용') == ('주요내용', '')


def test_split_section_header_rejects_an_enumeration_prefix():
    """'가.' opens ordinary body items too, so it is not accepted as a header.

    '가. 주요 내용 첫 번째 항목임.' is a body item; reading its second word as
    a label would delete real text. The 14 enumerated headers in a 3,000-notice
    sample are not worth that, so they keep their label as content.
    """
    assert split_section_header('가. 제안이유') is None
    assert split_section_header('2. 대안의 제안이유') is None
    assert split_section_header('가. 주요 내용 첫 번째 항목임.') is None


def test_split_section_header_strips_brackets_and_spacing_variants():
    assert split_section_header('【제안이유】') == ('제안이유', '')
    assert split_section_header('[주요내용] 내용임.') == ('주요내용', '내용임.')
    assert split_section_header('주요 내용') == ('주요 내용', '')
    assert split_section_header('제안이유및주요내용 현행법은') == ('제안이유및주요내용', '현행법은')


def test_split_section_header_ignores_ordinary_prose():
    """A sentence that merely starts with the same characters is not a header.

    Deleting it would remove real text, so the label must be followed by
    whitespace or the line end.
    """
    assert split_section_header('주요 내용은 다음과 같음.') is None
    assert split_section_header('계약의 주요 내용') is None
    assert split_section_header('현행법은 미비함.') is None
    assert split_section_header('') is None


def test_detect_sections_keeps_the_glued_paragraph():
    text = '의안번호 1234\n제안이유 및 주요내용 현행법은 미비함.\n주요내용\n가. 첫째'
    sections = detect_sections(text)
    assert [name for name, _ in sections] == ['body', '제안이유 및 주요내용', '주요내용']
    assert sections[0][1] == '의안번호 1234'
    # The glued paragraph survives as the new section's body, minus the label.
    assert sections[1][1] == '현행법은 미비함.'
    # Item markers are chunking's business, not section detection's.
    assert sections[2][1] == '가. 첫째'


def test_strip_item_marker_forms():
    assert strip_item_marker('가. 협회의 업무 범위') == '협회의 업무 범위'
    assert strip_item_marker('바. 검사의 사건 처리') == '검사의 사건 처리'
    assert strip_item_marker('2) 상속인의 주소') == '상속인의 주소'
    assert strip_item_marker('① 첫째') == '첫째'
    assert strip_item_marker('(나) 항목') == '항목'
    assert strip_item_marker('ㅇ 개요') == '개요'
    assert strip_item_marker('가.') == ''


def test_strip_item_marker_leaves_prose_alone():
    assert strip_item_marker('현행법은 미비함.') == '현행법은 미비함.'
    # A syllable from the ordinal set only counts as a marker with '.' or ')'.
    assert strip_item_marker('가나다라 어쩌고') == '가나다라 어쩌고'
    assert strip_item_marker('다음과 같음') == '다음과 같음'


def test_strip_glued_prefixes_removes_a_sentence_glued_label():
    """The corpus glues labels and markers to the PREVIOUS sentence, where
    line-level detection cannot see them."""
    assert strip_glued_prefixes('경감하려는 것임(안 제80조의2). 참고사항 이 법률안은 임.') == (
        '경감하려는 것임(안 제80조의2). 이 법률안은 임.'
    )
    assert strip_glued_prefixes('의견을 제출하도록 함(안 제245조의6). 바. 검사의 통지 대상을') == (
        '의견을 제출하도록 함(안 제245조의6). 검사의 통지 대상을'
    )
    assert strip_glued_prefixes('확인할 수 있음. 주요내용 다음 항목임.') == (
        '확인할 수 있음. 다음 항목임.'
    )


def test_strip_glued_prefixes_leaves_paragraph_initial_text_alone():
    """A paragraph-initial '가.' or '주요 내용' is a body item, not a prefix.

    The line-level pass owns the line start, `detect_sections` owns a
    paragraph-initial label, and a marker's own '.' must not be mistaken for the
    sentence boundary a glued prefix follows.
    """
    assert strip_glued_prefixes('가. 주요 내용 첫 번째 항목임.') == '가. 주요 내용 첫 번째 항목임.'
    assert strip_glued_prefixes('주요 내용 첫 번째 항목임.') == '주요 내용 첫 번째 항목임.'
    assert strip_glued_prefixes('참고사항 이 법률안은 임.') == '참고사항 이 법률안은 임.'
    assert strip_glued_prefixes('일반 문장임. 그리고 다음 문장임.') == (
        '일반 문장임. 그리고 다음 문장임.'
    )
