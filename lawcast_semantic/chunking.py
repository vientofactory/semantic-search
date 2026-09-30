"""Stage 1b: section-aware chunking of cleaned legislation text.

Chunking strategy:
  1. detect sections (제안이유 / 주요내용 / ...) and their paragraphs
  2. turn each paragraph into units (long paragraphs are split at sentence
     boundaries; pathological sentences at raw chars)
  3. greedily pack consecutive units into chunks, carrying the tail of the
     previous chunk as overlap so context survives the boundary

Chunk text keeps newline-separated paragraphs intact (the same line-break
structure LawCast preserves in `proposalReason`). The char budget bounds the
full embedding input (subject context + body, see `compose_embedding_text`),
so the body budget shrinks by the subject length and every chunk stays inside
the model token window regardless of title length.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .preprocess import detect_sections, normalize_text

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


@dataclass
class Chunk:
    """One embeddable unit of a legislation notice."""

    chunk_id: str
    notice_num: int
    subject: str
    committee: str
    section: str
    chunk_index: int
    text: str
    char_count: int
    sentence_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks_jsonl(path: Path) -> list[dict]:
    """Read chunk records from a chunks.jsonl artifact (blank lines skipped).

    Records are returned as plain dicts (the `Chunk.to_dict` wire format);
    this module owns that format, so every stage reads it through here.
    """
    records: list[dict] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_chunks_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write chunk records as chunks.jsonl (one `Chunk.to_dict` per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences at [.!?] boundaries and newlines."""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def compose_embedding_text(subject: str, text: str) -> str:
    """Embedding input for one chunk: subject as title context + body text.

    The subject carries title vocabulary (e.g. '공급망 안정화') that the body
    may never repeat, so queries phrased in title words still match. Queries
    embed their own text and carry no subject context.
    """
    subject = (subject or '').strip()
    if not subject or text.startswith(subject):
        # Empty subject -> body only; fallback chunks already include it.
        return text
    return f'{subject}\n{text}'


def _paragraph_units(paragraph: str, max_chars: int) -> list[str]:
    """Split one paragraph into units of at most max_chars characters."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    units: list[str] = []
    buffer = ''
    for sentence in split_sentences(paragraph):
        if len(sentence) > max_chars:
            # Pathological sentence without usable boundaries: hard split.
            if buffer:
                units.append(buffer)
                buffer = ''
            units.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        candidate = f'{buffer} {sentence}'.strip()
        if len(candidate) > max_chars:
            if buffer:
                units.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        units.append(buffer)
    return units


def _pack_units(
    units: list[str],
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Greedily pack units into chunks, joining them with newlines.

    Each new chunk repeats the trailing units of the previous chunk until
    overlap_chars is reached, preserving cross-boundary context. The overlap
    carry is trimmed from the front when necessary so every chunk stays within
    max_chars even when the next unit is itself close to the limit.
    """
    chunks: list[str] = []
    carry: list[str] = []  # trailing units of the previous chunk
    fresh: list[str] = []  # units appended on top of the carry

    def joined_length(parts: list[str]) -> int:
        return sum(len(part) for part in parts) + max(len(parts) - 1, 0)

    for unit in units:
        if fresh and joined_length(carry + fresh + [unit]) > max_chars:
            chunks.append('\n'.join(carry + fresh))
            previous = carry + fresh
            carry, carry_len = [], 0
            for part in reversed(previous):
                if carry_len + len(part) > overlap_chars:
                    break
                carry.insert(0, part)
                carry_len += len(part)
            fresh = []
        # A unit close to max_chars can push the chunk over the limit together
        # with the carry; drop carry units from the front until it fits.
        while carry and joined_length(carry + [unit]) > max_chars:
            carry.pop(0)
        fresh.append(unit)
    if fresh:
        chunks.append('\n'.join(carry + fresh))
    return chunks


def chunk_notice(
    notice: dict,
    max_chars: int = config.CHUNK_MAX_CHARS,
    overlap_chars: int = config.CHUNK_OVERLAP_CHARS,
    min_chars: int = config.CHUNK_MIN_CHARS,
) -> list[Chunk]:
    """Chunk a single notice record (dict with notice_num/subject/committee/proposal_reason).

    Notices with an empty `proposal_reason` fall back to a single chunk built
    from the subject so every notice stays searchable.

    `max_chars` bounds the composed embedding input (subject + body): the body
    budget shrinks by the subject context length so title length cannot push a
    chunk past the model token window.
    """
    notice_num = int(notice['notice_num'])
    subject = (notice.get('subject') or '').strip()
    committee = (notice.get('committee') or '').strip()

    normalized = normalize_text(notice.get('proposal_reason'))
    sections = detect_sections(normalized) if normalized else []

    # The subject is prepended as embedding context, spending part of the
    # char budget that keeps chunks inside the model window.
    context_len = len(subject) + 1 if subject else 0
    body_budget = max(max_chars - context_len, min_chars)

    chunks: list[Chunk] = []
    chunk_index = 0
    for section_name, body in sections:
        units: list[str] = []
        for paragraph in body.split('\n'):
            if paragraph.strip():
                units.extend(_paragraph_units(paragraph.strip(), body_budget))
        for text in _pack_units(units, body_budget, overlap_chars):
            if len(text) < min_chars and chunks:
                # Drop tiny trailing fragments once the notice has real chunks.
                continue
            chunks.append(
                Chunk(
                    chunk_id=f'{notice_num}-{chunk_index:04d}',
                    notice_num=notice_num,
                    subject=subject,
                    committee=committee,
                    section=section_name,
                    chunk_index=chunk_index,
                    text=text,
                    char_count=len(text),
                    sentence_count=len(split_sentences(text)),
                )
            )
            chunk_index += 1

    if not chunks:
        fallback_text = f'{subject}\n{normalized}'.strip()
        chunks.append(
            Chunk(
                chunk_id=f'{notice_num}-{chunk_index:04d}',
                notice_num=notice_num,
                subject=subject,
                committee=committee,
                section='body',
                chunk_index=chunk_index,
                text=fallback_text,
                char_count=len(fallback_text),
                sentence_count=len(split_sentences(fallback_text)),
            )
        )
    return chunks


def chunk_notices(
    notices: Iterable[dict],
    max_chars: int = config.CHUNK_MAX_CHARS,
    overlap_chars: int = config.CHUNK_OVERLAP_CHARS,
    min_chars: int = config.CHUNK_MIN_CHARS,
) -> list[Chunk]:
    """Chunk an iterable of notice records into a flat chunk list.

    Raises ValueError on duplicate notice_num values: chunk ids are derived
    from notice_num, so duplicates would silently collide and make the search
    stage return metadata for the wrong record.
    """
    chunks: list[Chunk] = []
    seen_notice_nums: set[int] = set()
    for notice in notices:
        notice_num = int(notice['notice_num'])
        if notice_num in seen_notice_nums:
            raise ValueError(f'duplicate notice_num in input: {notice_num}')
        seen_notice_nums.add(notice_num)
        chunks.extend(
            chunk_notice(
                notice,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )
        )
    return chunks


def _chunk_digest_bytes(record: dict) -> bytes:
    """Canonical per-chunk byte string: chunk_id NUL embedded-text NEWLINE."""
    embedded = compose_embedding_text(record['subject'], record['text'])
    return record['chunk_id'].encode('utf-8') + b'\0' + embedded.encode('utf-8') + b'\n'


def compute_chunk_text_digest(record: dict) -> str:
    """SHA-256 of one chunk's id + embedded text — per-row provenance atom.

    `compute_chunks_fingerprint` is exactly the hash of these atoms (sorted by
    chunk_id) over a whole set, so this digest pins one stored vector to the
    text it was computed from. Incremental updates reuse a stored row iff its
    digest still matches (see `lawcast_semantic.incremental`).
    """
    return hashlib.sha256(_chunk_digest_bytes(record)).hexdigest()


def compute_chunks_fingerprint(records: Iterable[dict]) -> str:
    """Stable fingerprint of chunk ids + embedded text.

    Guards pipeline artifact consistency: if chunks.jsonl changes after the
    embeddings/index were built, the fingerprint no longer matches and the
    search stage can refuse stale artifacts instead of silently mixing stale
    scores with new text. The embedded text is the subject-context composition
    (`compose_embedding_text`), so editing either the body or the subject
    invalidates the fingerprint — both change vectors. Metadata that is not
    embedded (e.g. committee) stays excluded: it is read from chunks.jsonl at
    query time and may be refreshed without re-embedding.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda record: record['chunk_id']):
        digest.update(_chunk_digest_bytes(record))
    return digest.hexdigest()
