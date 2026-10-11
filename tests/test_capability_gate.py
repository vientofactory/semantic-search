"""Regression gate: a chunking rule must not cost a notice a retrieval capability.

`test_chunking.py` pins the chunker's shape, `test_chunk_coverage.py` pins that
no paragraph text is dropped, and `test_chunk_coverage_retrieval.py` pins that a
dropped clause is reachable. None of them asked the question this module asks:
when a rule REPLACES a chunk instead of adding one, WHICH notices lose a
capability? Aggregate metrics do not answer it — the first AI-summary rule raised
title recall@1 from 0.167 to 0.333 while deleting the title-bearing chunk of the
405 notices whose `proposal_reason` is empty (name-benchmark coverage 80/133 ->
48/133; agent_memories/24-ai-summary-chunking §4).

The oracle lives in `lawcast_semantic.capabilities`: per notice, the structural
capabilities its chunk set has (`indexed`, `title_text`, `section:<label>`,
`derived:<field>`), each re-derived from the notice record. This module is the
gate built on it:

  1. `derived_field_diff` — the field-level gate. Registering a field in
     `DERIVED_NOTICE_FIELDS` puts it under it: chunking the same notices WITHOUT
     the field must be a capability-subset of chunking WITH it. A field that
     swaps a chunk out shows up as `lost`, so a new enrichment cannot ship as a
     net capability loss.
  2. the summary swap is reproduced here (`summary_swap`) and must be CAUGHT —
     a gate is worth what its inputs cover.
  3. the retrieval layer asks the same question through the real search path
     (`SemanticSearcher` in `tests/retrieval_probe.py`): a notice whose chunk set
     carries its title must return that title-bearing chunk from its own title
     query inside the window `/search` serves.

Fixture `tests/fixtures/notices-capability-gate.jsonl` — 24 real notices captured
2026-10-11, one row per shape the gate must distinguish: 8 with an empty
`proposal_reason` and a summary (the class the swap cost), 4 empty without a
summary, 8 ordinary notices with a long body, 4 with a short body (all with a
summary — the corpus has no long-body notice without one: the 106 `not_supported`
rows are exactly the summary-less empty-reason notices). Rebuild it with
`_workspace/build_capability_fixture.py` (host-local, gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_chunk_coverage_retrieval import QUERY_WINDOW, run_probe

from lawcast_semantic.capabilities import (
    DERIVED_NOTICE_FIELDS,
    DERIVED_PREFIX,
    IDENTITY_FIELDS,
    INDEXED,
    LEARNING_FIELD,
    SECTION_PREFIX,
    TITLE_TEXT,
    capability_index,
    capability_index_from_records,
    chunk_records,
    compare_capability_indexes,
    derived_field_diff,
    notice_capabilities,
    squash,
    strip_derived,
    unregistered_fields,
)
from lawcast_semantic.chunking import SUMMARY_SECTION, chunk_notice
from lawcast_semantic.datasource import load_notices_from_db, record_from_row
from lawcast_semantic.preprocess import detect_sections, normalize_text

FIXTURE_PATH = Path(__file__).resolve().parent / 'fixtures' / 'notices-capability-gate.jsonl'
CORPUS_PATH = Path(__file__).resolve().parents[2] / 'backend' / 'lawcast.db'
CORPUS_LIMIT = 600
SUMMARY_CAPABILITY = f'{DERIVED_PREFIX}ai_summary'

NOTICES: list[dict] = [
    json.loads(line)
    for line in FIXTURE_PATH.read_text(encoding='utf-8').splitlines()
    if line.strip()
]
NOTICES_BY_SHAPE: dict[str, list[dict]] = {}
for _notice in NOTICES:
    NOTICES_BY_SHAPE.setdefault(_notice['shape'], []).append(_notice)
# The notices whose chunk set MUST carry their own title: the shapes with no
# `proposal_reason` to chunk, i.e. the class the summary swap cost. Pinned to the
# fixture (not to the live rule) on purpose — a set derived from the chunker
# under test would shrink with the regression instead of failing on it.
TITLE_NOTICES = [
    notice
    for notice in NOTICES
    if notice['shape'] in ('empty_reason_summary', 'empty_reason_plain')
]


def make_notice(
    proposal_reason: str,
    *,
    ai_summary: str = '',
    notice_num: int = 2220607,
) -> dict:
    return {
        'notice_num': notice_num,
        'subject': '테스트법 일부개정법률안',
        'proposer_category': '의원발의',
        'committee': '법제사법위원회',
        'proposal_reason': proposal_reason,
        'ai_summary': ai_summary,
    }


# Synthetic notices for the shapes the fixture cannot exercise on demand, so
# the field-level gate still runs in CI (no corpus needed). Distinct notice_num
# values: the gate keys its index by notice.
GENERATED_NOTICES = [
    make_notice('', ai_summary='이 법안은 미비점을 보완하려는 개정안입니다.', notice_num=1),
    make_notice('', notice_num=2),
    make_notice('제안이유\n' + '가' * 300, ai_summary='요약문입니다.', notice_num=3),
    make_notice('제안이유 및 주요내용\n현행법은 미비함.', notice_num=4),
    make_notice('제안이유 및 주요내용\n현행법은 미비함.', ai_summary='요약문입니다.', notice_num=5),
    make_notice('제안이유\n' + '나' * 80, ai_summary='요약문입니다.', notice_num=6),
    make_notice('제안이유\n' + '다' * 60, ai_summary='요약문입니다.', notice_num=7),
]


def summary_swap(notice: dict) -> list[dict]:
    """The first AI-summary rule, reproduced: the summary REPLACES the fallback.

    For a notice whose `proposal_reason` yields no section, the shipped rule
    keeps `('body', subject)` and appends the summary; the first implementation
    let the summary chunk take its place, so the only chunk left was the summary
    paraphrase and the notice's title text was gone from the index.
    """
    records = [chunk.to_dict() for chunk in chunk_notice(notice)]
    if (
        not (notice.get('proposal_reason') or '').strip()
        and (notice.get('ai_summary') or '').strip()
    ):
        kept = [record for record in records if record['section'] == SUMMARY_SECTION]
        for index, record in enumerate(kept):  # a real replacement renumbers its rows
            record['chunk_index'] = index
            record['chunk_id'] = f'{record["notice_num"]}-{index:04d}'
        return kept
    return records


def title_probes(notices: list[dict]) -> list[dict]:
    """One probe per notice, asking for the notice by its own subject."""
    return [
        {
            'label': f'title notice {notice["notice_num"]}',
            'query': notice['subject'],
            'notice_num': notice['notice_num'],
        }
        for notice in notices
    ]


def title_carriers(result: dict, notice: dict) -> list[dict]:
    """Hits from this notice whose text carries the notice's own title."""
    subject = squash(notice['subject'])
    return [
        hit
        for hit in result['hits']
        if hit['notice_num'] == notice['notice_num'] and subject in squash(hit['text'])
    ]


# --- the registry: a new derived field cannot enter the pipeline unregistered ---


def test_a_new_notice_record_field_cannot_enter_unregistered():
    """The canonical record's fields must be exactly the declared registry.

    `datasource.record_from_row` is the only record producer, so this fails the
    moment a field is added there without being declared here — and an
    undeclared derived field is a field no gate covers (see `unregistered_fields`
    and the module docstring of `lawcast_semantic.capabilities`).
    """
    record = record_from_row(
        (2220607, '법률안', '의원발의', '법제사법위원회', '제안이유 본문', '요약문')
    )
    assert unregistered_fields(record) == set()
    assert set(record) == {*IDENTITY_FIELDS, LEARNING_FIELD, *DERIVED_NOTICE_FIELDS}


# --- the oracle: capability classes are re-derived from the record ---


def test_fixture_carries_every_shape_the_gate_distinguishes():
    assert set(NOTICES_BY_SHAPE) == {
        'empty_reason_summary',
        'empty_reason_plain',
        'body_summary',
        'short_body_summary',
    }
    assert len(NOTICES_BY_SHAPE['empty_reason_summary']) >= 8
    assert len(NOTICES_BY_SHAPE['empty_reason_plain']) >= 4
    assert len(NOTICES_BY_SHAPE['body_summary']) >= 4
    for notice in NOTICES:
        if notice['shape'] == 'empty_reason_summary':
            assert not notice['proposal_reason'].strip()
            assert notice['ai_summary'].strip()


def test_each_shape_has_the_capabilities_it_exists_to_prove():
    """Pins what the shapes mean, so a fixture rebuild cannot silently lose them."""
    for notice in NOTICES_BY_SHAPE['empty_reason_summary']:
        assert notice_capabilities(notice) == {INDEXED, TITLE_TEXT, SUMMARY_CAPABILITY}
    for notice in NOTICES_BY_SHAPE['empty_reason_plain']:
        assert notice_capabilities(notice) == {INDEXED, TITLE_TEXT}
    for shape in ('body_summary', 'short_body_summary'):
        for notice in NOTICES_BY_SHAPE[shape]:
            capabilities = notice_capabilities(notice)
            assert INDEXED in capabilities
            assert SUMMARY_CAPABILITY in capabilities
            assert any(name.startswith('section:') for name in capabilities)


def test_reporting_an_existing_chunk_set_matches_re_chunking_it():
    """The report path (stored artifacts) and the gate path (the rule) agree.

    `scripts/07_capability_report.py` measures a chunk set that already exists
    while this module measures what a rule produces; if those two handed the
    oracle different things, the release check would be reporting a different
    index than the gate guards.
    """
    stored = {int(notice['notice_num']): chunk_records(notice) for notice in NOTICES}
    assert capability_index_from_records(NOTICES, stored) == capability_index(NOTICES)


def test_derived_capability_is_re_derived_from_the_text_not_the_metadata():
    """A labeled chunk is not a capability: the field's TEXT has to be findable."""
    notice = make_notice(
        '제안이유\n현행법은 미비함.', ai_summary='이 법안은 미비점을 보완하려는 개정안입니다.'
    )
    assert SUMMARY_CAPABILITY in notice_capabilities(notice)
    dropped = lambda record_notice, **kwargs: [  # noqa: E731 - a one-off rule variant
        record
        for record in chunk_records(record_notice, **kwargs)
        if squash(record_notice['ai_summary']) not in squash(record['text'])
    ]
    assert SUMMARY_CAPABILITY not in notice_capabilities(notice, chunker=dropped)


def test_every_source_section_stays_carried_by_a_chunk():
    """A section `proposal_reason` yields must keep a chunk carrying its label.

    Re-derived from the source (not from a frozen expectation), so a packing or
    relabeling change fails here instead of only weakening the aggregate
    metrics. `detect_sections` drops a section whose body cleans away, so the
    invariant never demands a chunk the chunker cannot produce.
    """
    for notice in [*NOTICES, *GENERATED_NOTICES]:
        expected = {
            f'{SECTION_PREFIX}{label}'
            for label, _body in detect_sections(normalize_text(notice.get(LEARNING_FIELD)))
        }
        assert expected <= set(notice_capabilities(notice)), (
            f'notice {notice["notice_num"]} lost a section the source text yields: '
            f'{sorted(expected - set(notice_capabilities(notice)))}'
        )


# --- the gate: an additive field must not take a capability away ---


def test_the_shipped_rule_adds_the_derived_field_without_losing_anything():
    diff = derived_field_diff(NOTICES)
    assert not diff.is_regression, diff.render()
    assert diff.gained[SUMMARY_CAPABILITY] == {
        notice['notice_num'] for notice in NOTICES if notice['ai_summary'].strip()
    }, diff.render()


def test_every_registered_derived_field_reaches_the_index():
    """Registering a field commits the pipeline to indexing it for every notice."""
    for field in DERIVED_NOTICE_FIELDS:
        missing = [
            notice['notice_num']
            for notice in NOTICES
            if (notice.get(field) or '').strip()
            and f'{DERIVED_PREFIX}{field}' not in notice_capabilities(notice)
        ]
        assert not missing, f'{field} text is not covered by any chunk for {missing[:5]}'


def test_synthetic_shapes_are_additive_without_a_corpus():
    """The field-level gate must hold for shapes the fixture may not cover."""
    diff = derived_field_diff(GENERATED_NOTICES)
    assert not diff.is_regression, diff.render()
    assert diff.gained[SUMMARY_CAPABILITY] == {1, 3, 5, 6, 7}, diff.render()


# --- the gate has to bite: reproduce the regression it exists for ---


def test_the_gate_catches_the_summary_swap():
    """Baseline (no derived field) vs the swap: title_text loss on the empty-reason class."""
    fallback_only = capability_index(strip_derived(notice) for notice in NOTICES)
    swapped = capability_index(NOTICES, chunker=lambda notice, **kwargs: summary_swap(notice))
    diff = compare_capability_indexes(fallback_only, swapped)
    assert set(diff.lost) == {TITLE_TEXT}, diff.render()
    assert diff.lost[TITLE_TEXT] == {
        notice['notice_num'] for notice in NOTICES_BY_SHAPE['empty_reason_summary']
    }, diff.render()
    assert set(diff.gained) == {SUMMARY_CAPABILITY}, diff.render()


def test_the_shipped_rule_passes_the_same_comparison_the_swap_fails():
    fallback_only = capability_index(strip_derived(notice) for notice in NOTICES)
    diff = compare_capability_indexes(fallback_only, capability_index(NOTICES))
    assert not diff.is_regression, diff.render()
    assert set(diff.gained) == {SUMMARY_CAPABILITY}, diff.render()


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason='corpus DB not present (CI has no lawcast.db)')
def test_the_derived_field_is_additive_across_the_real_corpus():
    """Wide sweep: every summary-less-reason notice plus a long-body sample.

    The 405 empty-`proposal_reason` notices are exactly the class the swap cost,
    so they are all read (CORPUS_LIMIT is above that count) rather than sampled.
    """
    empty_reason = load_notices_from_db(
        CORPUS_PATH,
        where='(proposalReason IS NULL OR LENGTH(TRIM(proposalReason)) = 0)',
        limit=CORPUS_LIMIT,
    )
    bodies = load_notices_from_db(
        CORPUS_PATH, where='LENGTH(proposalReason) > 300', limit=CORPUS_LIMIT
    )
    assert len(empty_reason) >= 400, 'the empty-reason class is missing from the sweep'
    assert len(bodies) >= CORPUS_LIMIT, 'the body sample is smaller than requested'
    diff = derived_field_diff([*empty_reason, *bodies])
    assert not diff.is_regression, diff.render()
    assert len(diff.gained[SUMMARY_CAPABILITY]) >= 400, diff.render()


# --- the retrieval layer: the capability must survive the real search path ---


@pytest.fixture(scope='module')
def title_retrieval(tmp_path_factory) -> dict[str, dict]:
    """Index the fixture notices out of process and probe their titles once."""
    directory = tmp_path_factory.mktemp('capability-gate-titles')
    return run_probe(directory / 'artifacts', title_probes(TITLE_NOTICES), notices=NOTICES)


@pytest.fixture(scope='module')
def swapped_title_retrieval(tmp_path_factory) -> dict[str, dict]:
    """The same probes against the swap's chunk set (rule A/B through one path)."""
    directory = tmp_path_factory.mktemp('capability-gate-swapped')
    chunks = [record for notice in NOTICES for record in summary_swap(notice)]
    return run_probe(directory / 'artifacts', title_probes(TITLE_NOTICES), chunks=chunks)


@pytest.mark.parametrize('notice', TITLE_NOTICES, ids=lambda notice: str(notice['notice_num']))
def test_a_title_bearing_notice_is_reachable_by_its_own_title(title_retrieval, notice):
    """A capability that no query can reach is not a capability.

    The notice has to return a hit of its own that CARRIES the title inside the
    window `/search` serves by default (QUERY_WINDOW, pinned against the sidecar
    default by `test_chunk_coverage_retrieval.test_window_matches_the_sidecar_default`).
    """
    result = title_retrieval[f'title notice {notice["notice_num"]}']
    assert title_carriers(result, notice), (
        f'notice {notice["notice_num"]} carries its title in the index but no query reaches it: '
        f'window notice order {result["notice_order"][:QUERY_WINDOW]}'
    )


def test_the_retrieval_layer_sees_the_summary_swap(swapped_title_retrieval):
    """Sensitivity of the check above: under the swap its class cannot answer.

    The 8 empty-`proposal_reason` notices lose their title-bearing chunk, so no
    wound in the window can carry their title; the notices the swap does not
    touch (no summary) keep theirs. Without this, the previous test could be
    vacuous.
    """
    lost = {notice['notice_num'] for notice in NOTICES_BY_SHAPE['empty_reason_summary']}
    assert lost
    for notice in TITLE_NOTICES:
        result = swapped_title_retrieval[f'title notice {notice["notice_num"]}']
        if notice['notice_num'] in lost:
            assert not title_carriers(result, notice), (
                f'the swap left a title-bearing hit for {notice["notice_num"]}, so this probe '
                'cannot see the regression'
            )
        else:
            assert title_carriers(result, notice), (
                f'the swap changed notice {notice["notice_num"]}, which has no summary to swap'
            )
