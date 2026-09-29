"""Extract a small, diverse sample of legislation notices from the LawCast DB.

Reads `notice_archives` from the LawCast SQLite database (read-only) and
writes JSONL records that stage 1 consumes. Sample is stratified by
`proposalReason` length (short/medium/long) so chunking is exercised.

Data source: `backend/lawcast.db` (the backend's development database, kept
in sync with the current `notice_archives` schema). The root `lawcast_prod.db`
is a production snapshot with a slightly older schema and can still be used
via `--db ../../lawcast_prod.db`.

Usage:
    python scripts/extract_sample_data.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SELECT_COLUMNS = 'noticeNum, subject, proposerCategory, committee, proposalReason'

LENGTH_BUCKETS = {
    'short': 'length(proposalReason) BETWEEN 80 AND 400',
    'medium': 'length(proposalReason) BETWEEN 401 AND 1200',
    'long': 'length(proposalReason) > 1200',
}


def extract_sample(db_path: Path, per_bucket: int) -> list[dict]:
    """Fetch up to `per_bucket` notices per length bucket, deduplicated."""
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    records: list[dict] = []
    seen: set[int] = set()
    try:
        for bucket, condition in LENGTH_BUCKETS.items():
            rows = connection.execute(
                f'SELECT {SELECT_COLUMNS} FROM notice_archives '
                f'WHERE {condition} ORDER BY noticeNum DESC LIMIT ?',
                (per_bucket,),
            ).fetchall()
            for notice_num, subject, proposer_category, committee, proposal_reason in rows:
                if notice_num in seen:
                    continue
                seen.add(notice_num)
                records.append(
                    {
                        'notice_num': int(notice_num),
                        'subject': subject or '',
                        'proposer_category': proposer_category or '',
                        'committee': committee or '',
                        'proposal_reason': proposal_reason or '',
                        'length_bucket': bucket,
                    }
                )
    finally:
        connection.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--db', type=Path, default=config.PROJECT_ROOT.parent / 'backend' / 'lawcast.db'
    )
    parser.add_argument('--per-bucket', type=int, default=4)
    parser.add_argument('--out', type=Path, default=config.SAMPLE_DATA_PATH)
    args = parser.parse_args()

    records = extract_sample(args.db, args.per_bucket)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'sample notices : {len(records)}')
    print(f'source db      : {args.db}')
    for bucket in LENGTH_BUCKETS:
        count = sum(1 for record in records if record['length_bucket'] == bucket)
        print(f'  {bucket:<6} : {count}')
    print(f'written to     : {args.out}')
    print(f'extracted_at   : {datetime.now(UTC).isoformat()}')


if __name__ == '__main__':
    main()
