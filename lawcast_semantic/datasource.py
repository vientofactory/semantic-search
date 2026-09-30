"""Stage 0: learning-corpus data source — LawCast DB `proposalReason`.

The corpus this project learns (chunks, embeds, indexes) comes from the
`proposalReason` column of `notice_archives`: the 제안이유/주요내용 text that
LawCast already parsed and stored in SQLite. This module is the single place
that touches the database, and it deliberately extracts no text from HTML —
`source_html` snapshots are never read and no page scraping happens here.
Text cleaning is the preprocessing stage's job (see `preprocess.py`).

Canonical record schema consumed by every later stage:
    notice_num, subject, proposer_category, committee, proposal_reason

Two access paths produce the same records:
  - `load_notices_from_db` / `extract_sample`: straight from `notice_archives`
  - `load_notices_jsonl` / `write_notices_jsonl`: a JSONL snapshot of the same
    records (reproducible offline runs, e.g. the evaluation corpus)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# The learning text is `proposalReason` exactly as stored. No other column
# (including `source_html`) is part of the embedding corpus.
NOTICE_SELECT_COLUMNS = 'noticeNum, subject, proposerCategory, committee, proposalReason'

# Stratified sampling by proposalReason length so chunking is exercised across
# short/medium/long notices (SQLite `length()` counts characters for TEXT).
LENGTH_BUCKETS = {
    'short': 'length(proposalReason) BETWEEN 80 AND 400',
    'medium': 'length(proposalReason) BETWEEN 401 AND 1200',
    'long': 'length(proposalReason) > 1200',
}


def record_from_row(row: tuple) -> dict:
    """Map one `notice_archives` row to the canonical notice record."""
    notice_num, subject, proposer_category, committee, proposal_reason = row
    return {
        'notice_num': int(notice_num),
        'subject': subject or '',
        'proposer_category': proposer_category or '',
        'committee': committee or '',
        'proposal_reason': proposal_reason or '',
    }


def load_notices_from_db(
    db_path: Path,
    *,
    where: str | None = None,
    limit: int | None = None,
    order: str = 'noticeNum DESC',
) -> list[dict]:
    """Load notice records from `notice_archives` (read-only).

    `where` is a raw SQL condition over `notice_archives` (same style as
    LENGTH_BUCKETS), `limit` caps rows after `order`. The record text is the
    stored `proposalReason` value — no HTML parsing or text extraction.
    """
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        query = f'SELECT {NOTICE_SELECT_COLUMNS} FROM notice_archives'
        params: list = []
        if where:
            query += f' WHERE {where}'
        query += f' ORDER BY {order}'
        if limit is not None:
            query += ' LIMIT ?'
            params.append(limit)
        rows = connection.execute(query, params).fetchall()
        return [record_from_row(row) for row in rows]
    finally:
        connection.close()


def extract_sample(db_path: Path, per_bucket: int) -> list[dict]:
    """Fetch up to `per_bucket` notices per proposalReason length bucket.

    Buckets are queried newest-first and deduplicated across buckets, so a
    record carries the first (shortest) bucket it matched.
    """
    records: list[dict] = []
    seen: set[int] = set()
    for bucket, condition in LENGTH_BUCKETS.items():
        for record in load_notices_from_db(db_path, where=condition, limit=per_bucket):
            if record['notice_num'] in seen:
                continue
            seen.add(record['notice_num'])
            records.append({**record, 'length_bucket': bucket})
    return records


def load_notices_jsonl(path: Path) -> list[dict]:
    """Load notice records from a JSONL snapshot (see `write_notices_jsonl`)."""
    records: list[dict] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_notices_jsonl(records: list[dict], path: Path) -> None:
    """Write notice records as JSONL (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
