"""Stage 1a: legislation text cleaning and structure detection.

Input is the `proposalReason` field of LawCast `notice_archives` rows (see
`datasource.py` — the corpus text is the DB column as stored, not HTML-derived):
Korean proposal texts with standalone section headers ("제안이유", "주요내용", ...)
and enumerated items ("가.", "나.", "1)", ...) separated by newlines.
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

# Standalone section headers that appear at line starts in proposalReason.
SECTION_HEADER_RE = re.compile(
    r'^(제안이유(\s*및\s*주요내용)?|주요\s*내용|주요\s*골자|참고사항|검토의견)\s*$'
)

DEFAULT_SECTION = 'body'


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

    Lines matching SECTION_HEADER_RE start a new section; content before the
    first header belongs to DEFAULT_SECTION. Empty sections are dropped.
    """
    sections: list[tuple[str, list[str]]] = [(DEFAULT_SECTION, [])]
    for paragraph in split_paragraphs(text):
        if SECTION_HEADER_RE.match(paragraph):
            sections.append((paragraph.strip(), []))
        else:
            sections[-1][1].append(paragraph)

    return [(name, '\n'.join(paras)) for name, paras in sections if paras]
