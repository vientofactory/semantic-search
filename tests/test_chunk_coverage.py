"""Regression guard: chunking must never drop a notice's content.

A chunk the packer has already built is TEXT. Any rule that decides such a chunk
is "too small" has to MERGE it into a neighbouring chunk: deleting it makes that
text unreachable — it stays in `proposalReason`, but no vector covers it, so no
query can return it. That is what `CHUNK_MIN_CHARS` used to do (see
`agent_memories/22-chunk-floor-content-loss/`), and it is invisible in chunk
counts, sizes and fingerprints, so it needs a guard of its own.

The oracle is re-derived from the source text rather than by asking the chunker
what it packed: it walks the same sections and paragraphs (the `preprocess`
transforms are what define "content" for the index) and requires every
paragraph's text to appear in that notice's chunk set. Matching is
whitespace-insensitive because the packer only ever rewrites whitespace — it
joins units with newlines and re-joins sentences of a long paragraph with
spaces — so no packing change can satisfy this by construction.

Fixture provenance: `tests/fixtures/notices-chunk-coverage.jsonl` holds 11 real
`notice_archives` records captured on 2026-10-10 from the newest 3,000 notices
with `proposalReason > 300` chars, selected because the pre-merge floor lost
content in them — the five notices the fix was verified on by hand, the shortest
(`).`, 2 chars) and longest (39 chars) fragments the sample dropped, a fragment
lost mid-notice, a notice whose whole below-floor chunk was a section, and three
controls that were never affected. Rebuild it with
`_workspace/build_chunk_coverage_fixture.py` (host-local, gitignored).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lawcast_semantic import config
from lawcast_semantic.chunking import chunk_notice, compose_embedding_text
from lawcast_semantic.datasource import load_notices_from_db
from lawcast_semantic.preprocess import (
    detect_sections,
    normalize_text,
    strip_glued_prefixes,
    strip_item_marker,
)

FIXTURE_PATH = Path(__file__).resolve().parent / 'fixtures' / 'notices-chunk-coverage.jsonl'
# The corpus (5.4 GB, gitignored) is not in CI, so the wide sweep skips there and
# the committed fixture above is what runs everywhere.
CORPUS_PATH = Path(__file__).resolve().parents[2] / 'backend' / 'lawcast.db'
CORPUS_LIMIT = 2000

_WHITESPACE_RE = re.compile(r'\s+')

FIXTURE_NOTICES: list[dict] = [
    json.loads(line)
    for line in FIXTURE_PATH.read_text(encoding='utf-8').splitlines()
    if line.strip()
]


def expected_paragraphs(notice: dict) -> list[str]:
    """Every paragraph of content the notice's chunks must cover."""
    normalized = normalize_text(notice.get('proposal_reason'))
    if not normalized:
        return []
    paragraphs: list[str] = []
    for _section, body in detect_sections(normalized):
        for paragraph in body.split('\n'):
            if not paragraph.strip():
                continue
            cleaned = strip_glued_prefixes(strip_item_marker(paragraph.strip())).strip()
            if cleaned:
                paragraphs.append(cleaned)
    return paragraphs


def squash(text: str) -> str:
    return _WHITESPACE_RE.sub('', text)


def covered_text(chunks: list) -> str:
    """All chunk text of one notice, whitespace-insensitively joined."""
    return squash(''.join(chunk.text for chunk in chunks))


def assert_content_is_covered(notice: dict, **kwargs) -> list:
    """Chunk one notice and require every paragraph of it to stay findable."""
    chunks = chunk_notice(notice, **kwargs)
    assert chunks, f'notice {notice["notice_num"]} produced no chunk at all'
    expected = expected_paragraphs(notice)
    assert expected, 'guard input has no content paragraphs to check'
    covered = covered_text(chunks)
    missing = [paragraph for paragraph in expected if squash(paragraph) not in covered]
    assert not missing, (
        f'notice {notice["notice_num"]} dropped {len(missing)} paragraph(s) from the '
        f'index: {missing[:3]!r}'
    )
    return chunks


SUBJECT = '테스트법 일부개정법률안'
# The body budget `chunk_notice` derives from that subject (200 - subject - newline).
BODY_BUDGET = max(config.CHUNK_MAX_CHARS - (len(SUBJECT) + 1), config.CHUNK_MIN_CHARS)


def make_notice(
    proposal_reason: str,
    subject: str = SUBJECT,
    notice_num: int = 100,
) -> dict:
    return {
        'notice_num': notice_num,
        'subject': subject,
        'committee': '교육위원회',
        'proposal_reason': proposal_reason,
    }


# Shapes that put a leftover at every packing boundary the merge has to handle.
# The two budget-relative entries are sized from the body budget on purpose: they
# are the shapes that produced the bug — a sentence overflowing the budget by a
# few characters, and a short paragraph right after a chunk filled to the budget.
GENERATED_SHAPES = {
    'punctuation-free sentence longer than the budget': '가' * 320,
    'sentence just over the budget (tiny tail)': '가' * (BODY_BUDGET + 5),
    'short paragraph after a full chunk': '나' * BODY_BUDGET + '\n짧음.',
    'short paragraph after a partly filled chunk': '나' * 150 + '\n짧은 조항임.',
    'short section after a long one': '제안이유\n' + '가' * 150 + '\n참고사항\n짧음.',
    'tiny notice, whole body below the floor': '짧음.',
    'content before the first section header': '서두 문장임.\n제안이유\n' + '가' * 120,
    'many short paragraphs': '\n'.join(f'{index}번째 항목임.' for index in range(40)),
    'label and marker glue': '제안이유 및 주요내용 현행법은 미비함.\n가. 첫째 항목임.',
}

# Deliberately loose budgets: the invariant must hold for any configuration.
PARAM_SETS = (
    {'max_chars': 200, 'overlap_chars': 50, 'min_chars': 40},
    {'max_chars': 100, 'overlap_chars': 30, 'min_chars': 40},
    {'max_chars': 60, 'overlap_chars': 10, 'min_chars': 20},
)


@pytest.mark.parametrize('notice', FIXTURE_NOTICES, ids=lambda notice: str(notice['notice_num']))
def test_real_notices_keep_every_paragraph(notice: dict):
    assert_content_is_covered(notice)


def test_fixture_exercises_the_shapes_that_used_to_lose_content():
    """The guard is worth what its inputs cover: assert the fixture can bite.

    Measured on the fixture as captured: 10/11 notices put a paragraph over the
    body budget (the long-sentence hard split that produced the tail leftovers),
    11/11 carry a section label, 2/11 an item marker, and 2/11 still keep a
    sub-floor chunk (a section with nothing in front of it).
    """
    hard_split = 0  # a paragraph too long for the body budget -> tail leftover
    below_floor = 0  # a sub-floor chunk survives (nothing in front of it)
    labelled = 0  # a real section label, so stripping runs and `section` is set
    marked = 0  # a line-leading item marker, so `strip_item_marker` runs
    for notice in FIXTURE_NOTICES:
        subject = (notice.get('subject') or '').strip()
        budget = max(config.CHUNK_MAX_CHARS - (len(subject) + 1), config.CHUNK_MIN_CHARS)
        if any(len(paragraph) > budget for paragraph in expected_paragraphs(notice)):
            hard_split += 1
        if any(chunk.char_count < config.CHUNK_MIN_CHARS for chunk in chunk_notice(notice)):
            below_floor += 1
        normalized = normalize_text(notice['proposal_reason'])
        if any(name != 'body' for name, _ in detect_sections(normalized)):
            labelled += 1
        lines = [line.strip() for line in normalized.split('\n') if line.strip()]
        if any(strip_item_marker(line) != line for line in lines):
            marked += 1
    assert len(FIXTURE_NOTICES) >= 10
    assert hard_split >= 8, 'too few fixture notices exercise the long-sentence hard split'
    assert below_floor >= 1, 'no fixture notice keeps a sub-floor chunk'
    assert labelled >= 8, 'too few fixture notices carry a section label'
    assert marked >= 1, 'no fixture notice carries an item marker'


@pytest.mark.parametrize('params', PARAM_SETS, ids=lambda params: f'max{params["max_chars"]}')
@pytest.mark.parametrize('name', sorted(GENERATED_SHAPES))
def test_generated_shapes_keep_every_paragraph(name: str, params: dict):
    assert_content_is_covered(make_notice(GENERATED_SHAPES[name], notice_num=1), **params)


@pytest.mark.parametrize('name', sorted(GENERATED_SHAPES))
def test_generated_shapes_stay_within_the_budget_plus_merge_allowance(name: str):
    """The merge is the only overshoot: at most min_chars, fragment + newline."""
    limit = config.CHUNK_MAX_CHARS + config.CHUNK_MIN_CHARS
    for chunk in chunk_notice(make_notice(GENERATED_SHAPES[name], notice_num=1)):
        assert chunk.char_count > 0
        assert chunk.char_count <= limit
        assert len(compose_embedding_text(chunk.subject, chunk.text)) <= limit


def test_below_floor_chunks_are_only_a_sections_first_chunk():
    """A sub-floor chunk means "nothing in front to merge into", nothing else."""
    inputs = [
        *FIXTURE_NOTICES,
        *(make_notice(text, notice_num=1) for text in GENERATED_SHAPES.values()),
    ]
    for notice in inputs:
        seen_sections: set[str] = set()
        for chunk in chunk_notice(notice):
            is_section_first = chunk.section not in seen_sections
            seen_sections.add(chunk.section)
            if chunk.char_count < config.CHUNK_MIN_CHARS:
                assert is_section_first, (
                    f'{chunk.chunk_id} is below the floor but is not the first chunk of '
                    f'section {chunk.section!r}: it should have been merged into the chunk '
                    'in front of it'
                )


def test_a_notice_whose_body_is_only_boilerplate_still_gets_a_chunk():
    """A label-only body strips to an empty section; the fallback still indexes it.

    Nothing may silently disappear from the index: a notice whose sections all
    clean away has to end up with one chunk carrying its subject.
    """
    chunks = chunk_notice(make_notice('제안이유', notice_num=1))
    assert len(chunks) == 1
    assert chunks[0].text.startswith('테스트법 일부개정법률안')


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason='corpus DB not present (CI has no lawcast.db)')
def test_real_corpus_keeps_every_paragraph():
    """Wide sweep over real notices, bounded so it stays a test and not a job."""
    notices = load_notices_from_db(
        CORPUS_PATH, where='LENGTH(proposalReason) > 300', limit=CORPUS_LIMIT
    )
    assert len(notices) >= 1000, 'corpus sweep loaded too few notices to be meaningful'
    missing: list[tuple[int, str]] = []
    for notice in notices:
        covered = covered_text(chunk_notice(notice))
        missing.extend(
            (int(notice['notice_num']), paragraph)
            for paragraph in expected_paragraphs(notice)
            if squash(paragraph) not in covered
        )
    assert not missing, f'{len(missing)} paragraph(s) unsearchable, e.g. {missing[:3]!r}'
