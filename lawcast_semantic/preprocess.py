"""Stage 1a: legislation text cleaning and structure detection.

Input is the `proposalReason` field of LawCast `notice_archives` rows (see
`datasource.py` — the corpus text is the DB column as stored, not HTML-derived):
Korean proposal texts with section headers ("제안이유", "주요내용", ...) and
enumerated items ("가.", "나.", "1)", ...).

The corpus rarely writes a header as a line of its own. It usually GLUES the
label to the paragraph it introduces:

    제안이유 및 주요내용 현행법은 ...

so a header is modelled here as a line PREFIX, not a whole line (see
`SECTION_HEADER_RE`). Recognizing it is what keeps the 11-character label out of
the embedded text — it is the single most common line-start token in the corpus
and identical across thousands of notices, so embedding it only pulls unrelated
bills closer together.

`strip_item_marker` is the same idea for enumeration (`가. 협회의 업무`): the
marker is not content, it is never part of a query, and its `.` would otherwise
be read as a sentence boundary by the chunker. Both transforms belong to the
INDEXING path only — applying them to a user query would delete text the user
typed — so `normalize_text` deliberately stays transform-free.
"""

from __future__ import annotations

import re
import unicodedata

# Residual-markup guard only: DB `proposalReason` values are stored as plain
# text, so this is a no-op for well-formed data. It never "extracts" text from
# HTML — it drops stray tags if a source ever leaks markup into the column.
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')
_HORIZONTAL_SPACE_RE = re.compile(r'[ \t\u3000]+')
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')

# Enumerated item markers at the start of a line or sentence. The ordinal
# syllable set is the 14 Korean enumerators (가나다라마바사아자차카타파하); the
# numeric, parenthesized and circled forms cover the rest of what the corpus
# writes. A marker is only a marker when whitespace or the line end follows it,
# so ordinary prose that merely starts with the same characters is untouched.
_ITEM_ORDINALS = '가나다라마바사아자차카타파하'
_ITEM_MARKER_CORE = (
    rf'(?:[{_ITEM_ORDINALS}]\s*[.)]'
    rf'|\d{{1,3}}\s*[.)]'
    rf'|[①-⑳]'
    rf'|[\[\(][{_ITEM_ORDINALS}\d]{{1,3}}[\]\)]'
    rf'|[ㅇ○●◦•])'
)
ITEM_MARKER_RE = re.compile(rf'^\s*{_ITEM_MARKER_CORE}(?:\s+|$)')

# Header tokens exactly as the corpus writes them. Whitespace inside a token is
# optional so "주요내용"/"주요 내용" and "참고사항"/"참고 사항" are one case.
SECTION_HEADER_CORE = (
    r'(?:제안\s*이유(?:\s*(?:및|·|,)\s*주요\s*내용)?|주요\s*내용|주요\s*골자|'
    r'참고\s*사항|검토\s*의견)'
)

# A header may carry a bracket and the '대안의' qualifier (committee alternative
# bills write "대안의 제안이유") before its core. It must be followed by
# whitespace or the line end: '주요 내용은 ...' is a sentence that happens to
# start with the same characters, and stripping it would delete real text.
#
# An enumeration marker is deliberately NOT accepted here. '가. 제안이유' does
# occur, but '가.' also opens ordinary body items, and '가. 주요 내용 첫 번째
# 항목임.' is content that would be misread as a label — the 14 enumerated
# headers in a 3,000-notice sample are not worth that. Such a line keeps its
# label as 4 characters of content, which chunking then treats as body.
SECTION_HEADER_RE = re.compile(
    rf'^\s*(?:[\[\(【]\s*)?(?:대안\s*의\s*)?(?:[\[\(【]\s*)?'
    rf'(?P<header>{SECTION_HEADER_CORE})[\]\)】]?(?=\s|$)'
)

DEFAULT_SECTION = 'body'


def strip_item_marker(line: str) -> str:
    """Drop a leading enumeration marker from one line ('가. 내용' -> '내용').

    A no-op for lines that do not start with a marker, so callers can apply it
    unconditionally to paragraphs, sentences and chunk text.
    """
    return ITEM_MARKER_RE.sub('', line, count=1)


def split_section_header(paragraph: str) -> tuple[str, str] | None:
    """Split a section header off the front of a paragraph.

    Returns `(label, remaining_text)` — `remaining_text` is empty for a header
    that occupies the whole line — or None when the paragraph is ordinary
    prose. The label is what the corpus wrote, kept verbatim for the chunk
    metadata; it never enters the embedded text.
    """
    match = SECTION_HEADER_RE.match(paragraph)
    if not match:
        return None
    return match.group('header').strip(), paragraph[match.end() :].strip()


# A marker or label glued to the END of a sentence instead of standing on its
# own line:
#
#     ... 것임(안 제80조의2). 참고사항 이 법률안은 ...
#     ... 함(안 제245조의6). 바. 검사의 사건 처리 ...
#
# Only a sentence boundary can precede one here, and the two-character
# lookbehind excludes a marker's own period ('가. 주요 내용 ...' is a body item
# whose label-looking words are content, not a glued label).
_GLUED_PREFIX_RE = re.compile(
    rf'(?<![{_ITEM_ORDINALS}\d][.!?])(?<=[.!?])\s+'
    rf'(?:{_ITEM_MARKER_CORE}|{SECTION_HEADER_CORE})(?=\s|$)'
)


def strip_glued_prefixes(text: str) -> str:
    """Drop markers/labels glued after a sentence end, keeping the separator.

    Applied to a whole paragraph before it is split into sentences, because
    which sentence a chunk boundary lands on is decided later: a prefix removed
    only from the start of a packed unit would survive whenever the packer
    grouped it with the sentence before it. A paragraph-initial label is not
    touched — `detect_sections` consumes that one (it is what opens a section),
    and a body sentence like '주요 내용 첫 번째 항목임.' must stay intact.
    """
    # The pattern consumes the separator before the prefix as well as the
    # prefix, so an empty replacement leaves the sentence break intact
    # ('것임. 참고사항 이' -> '것임. 이').
    return _GLUED_PREFIX_RE.sub('', text)


def normalize_text(text: str | None) -> str:
    """Clean raw proposal text without destroying its structure.

    - Unicode NFC normalization (Korean compatibility jamo safety)
    - drop residual markup / zero-width characters (guard, not extraction)
    - collapse horizontal whitespace, but PRESERVE line breaks: newlines are
      meaningful paragraph structure in `proposalReason` and must not be
      collapsed into spaces
    - collapse 3+ consecutive newlines down to a blank line
    """
    if not text:
        return ''
    cleaned = unicodedata.normalize('NFC', text)
    cleaned = _HTML_TAG_RE.sub(' ', cleaned)
    cleaned = _ZERO_WIDTH_RE.sub('', cleaned)
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    cleaned = _HORIZONTAL_SPACE_RE.sub(' ', cleaned)
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    cleaned = _MULTI_NEWLINE_RE.sub('\n\n', cleaned)
    return cleaned.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split normalized text into non-empty paragraphs (one per line)."""
    return [line.strip() for line in text.split('\n') if line.strip()]


def detect_sections(text: str) -> list[tuple[str, str]]:
    """Group normalized proposal text into (section_name, body) pairs.

    A paragraph that starts with a section header opens a new section; text
    following the label on the same line belongs to that section's body, so the
    glued form ("제안이유 및 주요내용 현행법은 ...") keeps its content and loses
    only the label. Content before the first header belongs to
    DEFAULT_SECTION. Empty sections are dropped.
    """
    sections: list[tuple[str, list[str]]] = [(DEFAULT_SECTION, [])]
    for paragraph in split_paragraphs(text):
        header = split_section_header(paragraph)
        if header:
            label, tail = header
            sections.append((label, [tail] if tail else []))
        else:
            sections[-1][1].append(paragraph)

    return [(name, '\n'.join(paras)) for name, paras in sections if paras]
