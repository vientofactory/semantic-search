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
