"""Capability oracle: what a notice's chunk set can still be found by.

Chunking changes are argued on aggregate retrieval metrics (recall@k over a
labeled set). Those metrics are nearly blind to one failure mode: a rule that
REPLACES a notice's chunks can raise them while deleting a capability for a
subset of notices. Adding the AI summary section first did exactly that — for
the 405 notices whose `proposal_reason` is empty, the summary chunk replaced the
subject-only fallback chunk, so title recall@1 rose 0.167 -> 0.333 while those
notices lost their exact-title hit (name-benchmark coverage 80/133 -> 48/133,
top-1 16/133 -> 4/133; see agent_memories/24-ai-summary-chunking §4).

This module turns "which capabilities does THIS notice have" into a value, so a
candidate rule can be compared against the rule it replaces. A capability is a
per-notice, structural property of the chunk set, re-derived from the notice
record rather than read off the chunker's own bookkeeping:

    indexed          the notice produced at least one chunk
    title_text       some chunk's text carries the subject
    section:<label>  a chunk carries each section `proposal_reason` yields
    derived:<field>  a chunk carries each non-empty derived field's text

`title_text` is deliberately the strict form. `chunking.compose_embedding_text`
already prepends the subject to every chunk's embedding INPUT, so title
vocabulary is present whenever a notice has any chunk at all. What the summary
swap deleted was the chunk whose TEXT is the title — the score-1.0 answer a name
query used to get. A notice whose body repeats its bill name still has the
capability; the empty-`proposal_reason` fallback is the shape that depends on it,
and so does a notice whose SUMMARY text names the bill (measured on the live
set: the summary build carries `title_text` for 3 more notices than its
summary-less rollback set). Every containment check is whitespace-insensitive
(`squash`), matching the packer, which only ever rewrites whitespace.

Three consumers use this oracle:

  - `derived_field_diff(notices)` is the gate for an additive derived field:
    the same notices chunked WITHOUT the fields must be a capability-subset of
    chunking WITH them (`strip_derived` builds that baseline). A field that
    swaps a chunk out instead of adding one shows up as a loss.
  - `compare_capability_indexes(before, after)` compares any two chunk sets, so
    a packing-rule change can be A/B'd the same way.
  - `capability_index_from_records(notices, records_by_notice)` measures an
    artifact set that already exists, which is what makes a release check
    possible: `scripts/07_capability_report.py` reports the served index and
    diffs it against a saved baseline per notice.

The registry is what keeps the gate from being bypassed: a field added to the
notice record is reported by `unregistered_fields` until it is declared, so a
new derived field cannot enter the pipeline without landing under the gate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .chunking import chunk_notice
from .preprocess import detect_sections, normalize_text

# Notice-record fields that identify a notice instead of carrying searchable
# content: no chunk is derived from them.
IDENTITY_FIELDS = ('notice_num', 'subject', 'proposer_category', 'committee')
# The field the index learns from — the legislation text itself.
LEARNING_FIELD = 'proposal_reason'
# Fields DERIVED from a notice (a summary of the same bill, and any future
# enrichment). Every entry is covered by `derived_field_diff`; registering one
# here is what puts it under the additive gate.
DERIVED_NOTICE_FIELDS = ('ai_summary',)

# A chunker maps one notice to the chunk records the gate inspects.
Chunker = Callable[..., list[dict]]

_WHITESPACE_RE = re.compile(r'\s+')

INDEXED = 'indexed'
TITLE_TEXT = 'title_text'
SECTION_PREFIX = 'section:'
DERIVED_PREFIX = 'derived:'


def squash(text: str) -> str:
    """Whitespace-insensitive form used for every containment check."""
    return _WHITESPACE_RE.sub('', text or '')


def chunk_records(notice: Mapping, **kwargs) -> list[dict]:
    """Default chunker: `chunk_notice` in the wire format (`Chunk.to_dict`)."""
    return [chunk.to_dict() for chunk in chunk_notice(notice, **kwargs)]


def strip_derived(notice: Mapping) -> dict:
    """The same notice without any derived field (the pre-field baseline)."""
    return {key: value for key, value in notice.items() if key not in DERIVED_NOTICE_FIELDS}


def unregistered_fields(record: Mapping) -> set[str]:
    """Record fields that are neither identity, learning text nor a declared derived field.

    `datasource.record_from_row` is the canonical record producer, so a non-empty
    result means a field entered the pipeline without entering the registry —
    and a derived field the gate does not know about is a field nothing checks.
    """
    return set(record) - set(IDENTITY_FIELDS) - {LEARNING_FIELD} - set(DERIVED_NOTICE_FIELDS)


def record_capabilities(notice: Mapping, records: Iterable[Mapping]) -> frozenset[str]:
    """Capabilities of one notice's chunk records, re-derived from the notice."""
    records = list(records)
    if not records:
        return frozenset()
    covered = squash(''.join(record['text'] for record in records))
    chunk_sections = {record.get('section') for record in records}
    capabilities = {INDEXED}
    subject = squash(notice.get('subject') or '')
    if subject and subject in covered:
        capabilities.add(TITLE_TEXT)
    normalized = normalize_text(notice.get(LEARNING_FIELD))
    # The LABELS come from the source, their CARRIERS from the chunk set: a rule
    # that relabels or merges a section into another must lose the class, so the
    # capability is not satisfied by the source text alone.
    for label, _body in detect_sections(normalized) if normalized else []:
        if label in chunk_sections:
            capabilities.add(f'{SECTION_PREFIX}{label}')
    for field in DERIVED_NOTICE_FIELDS:
        text = squash(notice.get(field) or '')
        if text and text in covered:
            capabilities.add(f'{DERIVED_PREFIX}{field}')
    return frozenset(capabilities)


def notice_capabilities(
    notice: Mapping,
    *,
    chunker: Chunker = chunk_records,
    **chunk_kwargs,
) -> frozenset[str]:
    """The capabilities this notice has under the rule `chunker` implements."""
    return record_capabilities(notice, chunker(notice, **chunk_kwargs))


def capability_index(
    notices: Iterable[Mapping],
    *,
    chunker: Chunker = chunk_records,
    **chunk_kwargs,
) -> dict[str, set[int]]:
    """Map each capability to the notices holding it — the unit of comparison."""
    index: dict[str, set[int]] = {}
    for notice in notices:
        notice_num = int(notice['notice_num'])
        for capability in notice_capabilities(notice, chunker=chunker, **chunk_kwargs):
            index.setdefault(capability, set()).add(notice_num)
    return index


def capability_index_from_records(
    notices: Iterable[Mapping],
    records_by_notice: Mapping[int, Iterable[Mapping]],
) -> dict[str, set[int]]:
    """Capability index of an ALREADY-BUILT chunk set (the served artifacts).

    `capability_index` re-chunks each notice, so it answers "what would this
    rule produce". An operational report has to answer "what does the index we
    are serving cover" instead — `scripts/07_capability_report.py` groups the
    artifact's own chunk records by notice and passes them here, so a release
    A/B compares two artifact sets rather than two code revisions. The two paths
    agree on the same chunk set (`tests/test_capability_gate.py`).
    """

    def stored(notice: Mapping, **_kwargs) -> list[dict]:
        return list(records_by_notice.get(int(notice['notice_num']), ()))

    return capability_index(notices, chunker=stored)


@dataclass(frozen=True)
class CapabilityDiff:
    """Baseline vs candidate capability coverage, per capability class."""

    baseline: Mapping[str, set[int]]
    candidate: Mapping[str, set[int]]

    @property
    def lost(self) -> dict[str, set[int]]:
        """Capabilities the candidate dropped, with the notices that had them."""
        return {
            capability: notices - self.candidate.get(capability, set())
            for capability, notices in self.baseline.items()
            if notices - self.candidate.get(capability, set())
        }

    @property
    def gained(self) -> dict[str, set[int]]:
        """Capabilities the candidate added that the baseline did not have."""
        return {
            capability: notices - self.baseline.get(capability, set())
            for capability, notices in self.candidate.items()
            if notices - self.baseline.get(capability, set())
        }

    @property
    def is_regression(self) -> bool:
        return bool(self.lost)

    def render(self, *, sample: int = 5) -> str:
        """Human-readable diff, for a test failure or a measurement log."""
        lines = ['capability diff (baseline -> candidate):']
        if not self.lost:
            lines.append('  lost: none')
        for capability, notices in sorted(self.lost.items()):
            sample_nums = sorted(notices)[:sample]
            lines.append(f'  LOST   {capability:24} {len(notices):5} notices  {sample_nums}')
        if not self.gained:
            lines.append('  gained: none')
        for capability, notices in sorted(self.gained.items()):
            sample_nums = sorted(notices)[:sample]
            lines.append(f'  gained {capability:24} {len(notices):5} notices  {sample_nums}')
        return '\n'.join(lines)


def compare_capability_indexes(
    baseline: Mapping[str, set[int]],
    candidate: Mapping[str, set[int]],
) -> CapabilityDiff:
    """Compare two capability indexes by class (the general A/B shape)."""
    return CapabilityDiff(baseline=baseline, candidate=candidate)


def derived_field_diff(
    notices: Iterable[Mapping],
    *,
    chunker: Chunker = chunk_records,
    **chunk_kwargs,
) -> CapabilityDiff:
    """Gate for an additive field: chunk without the derived fields, then with them.

    The baseline is the same notices and the same rule with every
    `DERIVED_NOTICE_FIELDS` value stripped, so the diff isolates what the fields
    cost. `lost` must be empty: an enrichment may add chunks, never take one
    away.
    """
    notices = list(notices)
    baseline = capability_index(
        (strip_derived(notice) for notice in notices), chunker=chunker, **chunk_kwargs
    )
    candidate = capability_index(notices, chunker=chunker, **chunk_kwargs)
    return compare_capability_indexes(baseline, candidate)
