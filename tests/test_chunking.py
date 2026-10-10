"""Tests for stage 1b: section-aware chunking."""

import pytest

from lawcast_semantic.chunking import (
    chunk_notice,
    chunk_notices,
    compose_embedding_text,
    split_sentences,
)


def make_notice(proposal_reason: str, notice_num: int = 100) -> dict:
    return {
        'notice_num': notice_num,
        'subject': '테스트법 일부개정법률안',
        'committee': '교육위원회',
        'proposal_reason': proposal_reason,
    }


def test_split_sentences():
    assert split_sentences('첫 문장임. 두 번째 문장임.\n새 줄 내용.') == [
        '첫 문장임.',
        '두 번째 문장임.',
        '새 줄 내용.',
    ]


def test_chunk_ids_are_unique_and_ordered():
    notice = make_notice('제안이유\n' + '\n'.join(f'문장 {i} 번째 내용임.' for i in range(30)))
    chunks = chunk_notice(notice, max_chars=100, overlap_chars=20, min_chars=10)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunks) > 1
    assert len(set(ids)) == len(ids)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_respect_max_chars():
    notice = make_notice('본문\n' + '긴 문장이지만 경계에서 분리됨. ' * 40)
    chunks = chunk_notice(notice, max_chars=120, overlap_chars=30, min_chars=10)
    assert all(chunk.char_count <= 120 for chunk in chunks)


def test_overlap_repeats_trailing_context():
    notice = make_notice(
        '본문\n' + '\n'.join(f'각 문장은 고유한 단어 {i} 를 포함함.' for i in range(20))
    )
    chunks = chunk_notice(notice, max_chars=120, overlap_chars=40, min_chars=10)
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, 'consecutive chunks should share overlap content'


def test_section_names_propagate_to_chunks():
    notice = make_notice(
        '제안이유\n이유 내용이 충분히 길어서 하나의 청크를 이룸.\n'
        '주요내용\n가. 주요 내용 첫 번째 항목임.'
    )
    chunks = chunk_notice(notice, max_chars=200, overlap_chars=20, min_chars=5)
    assert {chunk.section for chunk in chunks} == {'제안이유', '주요내용'}


def test_chunk_text_drops_section_label_and_item_markers():
    """Boilerplate must not reach the embedder: the label is corpus-wide constant
    and the markers are pure enumeration, so both only add tokens that make every
    chunk look alike. The label survives in the chunk's `section` metadata."""
    notice = make_notice(
        '제안이유 및 주요내용 현행법은 미비함.\n가. 첫째 항목임.\n나. 둘째 항목임.'
    )
    chunks = chunk_notice(notice, max_chars=200, overlap_chars=20, min_chars=5)
    assert chunks
    assert {chunk.section for chunk in chunks} == {'제안이유 및 주요내용'}
    body = '\n'.join(chunk.text for chunk in chunks)
    assert '제안이유' not in body
    assert '주요내용' not in body
    assert '첫째 항목임.' in body
    assert '둘째 항목임.' in body
    # No chunk starts with a marker (the marker's '.' used to split '가.' off as
    # its own sentence and could open a chunk).
    for chunk in chunks:
        assert not chunk.text.split('\n')[0].startswith(('가.', '나.', '다.'))


def test_inline_label_after_a_sentence_is_removed():
    """'... 것임(안 제80조의2). 참고사항 이 법률안은 ...' glues the label after a
    sentence, so it is not at a line start; removing it is per sentence.

    The paragraph is made longer than the body budget on purpose so the chunker
    actually splits it into sentences, which is when a mid-paragraph label is
    visible as one.
    """
    notice = make_notice(
        '제안이유 및 주요내용 경감함. 참고사항 이 법률안은 대표발의한 법률안임.'
        ' 그리고 다른 내용도 함께 있음.'
    )
    chunks = chunk_notice(notice, max_chars=60, overlap_chars=5, min_chars=5)
    body = '\n'.join(chunk.text for chunk in chunks)
    assert '참고사항' not in body
    assert '이 법률안은 대표발의한 법률안임.' in body
    assert '경감함.' in body


def test_paragraph_initial_label_like_text_is_kept():
    """A body sentence that happens to start with a label word is content.

    '가. 주요 내용 첫 번째 항목임.' loses only its marker; the words '주요 내용'
    belong to the sentence and must survive, even though the same token is a
    section label elsewhere.
    """
    notice = make_notice('주요내용\n가. 주요 내용 첫 번째 항목임.')
    chunks = chunk_notice(notice, max_chars=200, overlap_chars=20, min_chars=5)
    body = '\n'.join(chunk.text for chunk in chunks)
    assert '주요 내용 첫 번째 항목임.' in body


def test_marker_stripping_does_not_bite_into_content():
    """Only the marker itself goes: the rest of the line stays verbatim."""
    notice = make_notice('본문\n가. 협회의 업무 범위에 정책 건의를 추가함(안 제14조제1항).')
    chunks = chunk_notice(notice, max_chars=200, overlap_chars=20, min_chars=5)
    body = '\n'.join(chunk.text for chunk in chunks)
    assert '협회의 업무 범위에 정책 건의를 추가함(안 제14조제1항).' in body


def test_empty_proposal_reason_falls_back_to_subject():
    chunks = chunk_notice(make_notice(''))
    assert len(chunks) == 1
    assert '테스트법' in chunks[0].text


def test_chunk_notices_covers_all_notices():
    notices = [
        make_notice('제안이유\n내용 하나.', notice_num=1),
        make_notice('제안이유\n내용 둘.', notice_num=2),
    ]
    chunks = chunk_notices(notices)
    assert {chunk.notice_num for chunk in chunks} == {1, 2}


def test_chunks_respect_max_chars_with_large_units():
    """Overlap carry + a near-limit unit must not push a chunk over max_chars."""
    para1 = ' '.join(f'작은 문장 하나둘 {i} 이다.' for i in range(25))
    para2 = '가' * 250  # no [.!?] -> hard split into max_chars pieces
    notice = make_notice(f'본문\n{para1}\n{para2}')
    chunks = chunk_notice(notice, max_chars=200, overlap_chars=50, min_chars=40)
    assert chunks
    assert all(chunk.char_count <= 200 for chunk in chunks)


def test_below_floor_leftover_is_merged_not_dropped():
    """A below-floor chunk is a leftover that did not fit beside its neighbour.

    It used to hit `len(text) < CHUNK_MIN_CHARS and chunks -> drop` and vanish
    from the index. The text must survive: here one punctuation-free sentence is
    too long for the budget, so `_paragraph_units` hard-splits it into a
    budget-sized piece and a leftover that used to be dropped on its own.
    """
    body = '가' * 120
    chunks = chunk_notice(make_notice(body), max_chars=100, overlap_chars=30, min_chars=40)
    assert len(chunks) == 1
    assert chunks[0].text.replace('\n', '') == body
    # The merge is the one place the char budget may be exceeded, by at most
    # min_chars (the fragment plus the newline that joins it).
    assert chunks[0].char_count <= 100 + 40


def test_below_floor_trailing_paragraph_joins_the_chunk_in_front():
    """The trailing fragment keeps the sentence it belongs to as its context."""
    notice = make_notice('가' * 105 + '\n짧은 조항임(안 제3조).')
    chunks = chunk_notice(notice, max_chars=120, overlap_chars=30, min_chars=40)
    assert len(chunks) == 1
    text = chunks[0].text
    assert '가' * 105 in text
    assert '짧은 조항임(안 제3조).' in text
    assert text.index('가' * 105) < text.index('짧은 조항임(안 제3조).')


def test_below_floor_fragment_does_not_duplicate_the_overlap_carry():
    """Only units that are not already in the previous chunk are appended.

    A merged chunk repeats the carry units of the chunk in front of it by
    construction (that is the overlap), but it must not append them a second
    time: appending the fragment's whole text would double the carry inside one
    chunk and spend the merge budget on text the index already has.
    """
    notice = make_notice(
        '본문\n' + '\n'.join(f'문장 {i} 번째 내용임(안 제{i}조).' for i in range(12))
    )
    chunks = chunk_notice(notice, max_chars=90, overlap_chars=40, min_chars=40)
    assert len(chunks) >= 2
    for chunk in chunks:
        lines = [line for line in chunk.text.split('\n') if line]
        assert len(lines) == len(set(lines)), f'chunk repeats a unit: {chunk.text!r}'
    body = '\n'.join(chunk.text for chunk in chunks)
    for i in range(12):
        assert f'문장 {i} 번째 내용임(안 제{i}조).' in body


def test_short_section_after_a_long_one_is_kept():
    """A section's only chunk is content, even when an earlier section packed first."""
    notice = make_notice('제안이유\n' + '가' * 150 + '\n참고사항\n짧은 참고임.')
    chunks = chunk_notice(notice, max_chars=120, overlap_chars=30, min_chars=40)
    body = '\n'.join(chunk.text for chunk in chunks)
    assert '짧은 참고임.' in body
    assert '참고사항' in {chunk.section for chunk in chunks}


def test_duplicate_notice_num_rejected():
    """Duplicate notice_num would silently collide chunk ids across notices."""
    notices = [
        make_notice('제안이유\n내용 하나.', notice_num=7),
        make_notice('제안이유\n내용 둘.', notice_num=7),
    ]
    with pytest.raises(ValueError, match='duplicate notice_num'):
        chunk_notices(notices)


def test_compose_embedding_text_puts_subject_first():
    """Title vocabulary must enter the embedding input as context."""
    assert compose_embedding_text('공급망 안정화 지원법', '위기품목 수급 안정 내용') == (
        '공급망 안정화 지원법\n위기품목 수급 안정 내용'
    )


def test_compose_embedding_text_without_subject_keeps_text():
    assert compose_embedding_text('', '본문만 있음') == '본문만 있음'
    assert compose_embedding_text('   ', '본문만 있음') == '본문만 있음'


def test_compose_embedding_text_does_not_duplicate_fallback_subject():
    """Fallback chunks already embed the subject; composing must not repeat it."""
    chunks = chunk_notice(make_notice(''))
    composed = compose_embedding_text(chunks[0].subject, chunks[0].text)
    assert composed == chunks[0].text


def test_subject_context_respects_char_budget():
    """A long title shrinks the body budget; the embedding input still fits."""
    notice = make_notice('본문\n' + '긴 문장이지만 경계에서 분리됨. ' * 40)
    notice['subject'] = '경제안보를 위한 공급망 안정화 지원 기본법 일부개정법률안' * 2
    chunks = chunk_notice(notice, max_chars=120, overlap_chars=30, min_chars=10)
    assert chunks
    for chunk in chunks:
        assert len(compose_embedding_text(chunk.subject, chunk.text)) <= 120
