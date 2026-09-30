"""Extract a small, diverse sample of legislation notices from the LawCast DB.

Thin CLI over `lawcast_semantic.datasource`: reads `notice_archives` from the
LawCast SQLite database (read-only) and writes JSONL records that stage 1
consumes. The learning text is the `proposalReason` column exactly as stored
(no HTML extraction). Sample is stratified by `proposalReason` length
(short/medium/long) so chunking is exercised.

Data source: `backend/lawcast.db` (the backend's development database, kept
in sync with the current `notice_archives` schema). The root `lawcast_prod.db`
is a production snapshot with a slightly older schema and can still be used
via `--db ../../lawcast_prod.db`.

Usage:
    python scripts/extract_sample_data.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.datasource import (  # noqa: E402
    LENGTH_BUCKETS,
    extract_sample,
    write_notices_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--db', type=Path, default=config.PROJECT_ROOT.parent / 'backend' / 'lawcast.db'
    )
    parser.add_argument('--per-bucket', type=int, default=4)
    parser.add_argument('--out', type=Path, default=config.SAMPLE_DATA_PATH)
    args = parser.parse_args()

    records = extract_sample(args.db, args.per_bucket)
    write_notices_jsonl(records, args.out)

    print(f'sample notices : {len(records)}')
    print(f'source db      : {args.db}')
    for bucket in LENGTH_BUCKETS:
        count = sum(1 for record in records if record['length_bucket'] == bucket)
        print(f'  {bucket:<6} : {count}')
    print(f'written to     : {args.out}')
    print(f'extracted_at   : {datetime.now(UTC).isoformat()}')


if __name__ == '__main__':
    main()
