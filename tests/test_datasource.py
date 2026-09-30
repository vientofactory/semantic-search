"""Tests for stage 0: the learning-corpus data source (DB `proposalReason`)."""

import sqlite3
from pathlib import Path

from lawcast_semantic.datasource import (
    extract_sample,
    load_notices_from_db,
    load_notices_jsonl,
    write_notices_jsonl,
)

LONG_REASON = '법' * 1300  # > 1200 chars
MEDIUM_REASON = '법' * 600  # 401-1200 chars
SHORT_REASON = '법' * 100  # 80-400 chars


def make_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Create a temp DB with the notice_archives columns the datasource touches.

    `source_html` is included to prove the datasource reads `proposalReason`
    only — HTML snapshots never enter the learning corpus.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / 'lawcast.db'
    connection = sqlite3.connect(db_path)
    connection.execute(
        'CREATE TABLE notice_archives ('
        'noticeNum integer PRIMARY KEY, '
        'subject varchar(500) NOT NULL, '
        'proposerCategory varchar(100) NOT NULL, '
        'committee varchar(200) NOT NULL, '
        "proposalReason text NOT NULL DEFAULT '', "
        'source_html text)'
    )
    connection.executemany(
        'INSERT INTO notice_archives '
        '(noticeNum, subject, proposerCategory, committee, proposalReason, source_html) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        rows,
    )
    connection.commit()
    connection.close()
    return db_path


def test_load_notices_from_db_reads_proposal_reason_as_learning_text(tmp_path: Path):
    db_path = make_db(
        tmp_path,
        [(1, '테스트법안', '의원', '교육위원회', '제안이유\n본문 내용', '<html>ignored</html>')],
    )
    records = load_notices_from_db(db_path)
    assert records == [
        {
            'notice_num': 1,
            'subject': '테스트법안',
            'proposer_category': '의원',
            'committee': '교육위원회',
            'proposal_reason': '제안이유\n본문 내용',
        }
    ]


def test_load_notices_from_db_never_reads_source_html(tmp_path: Path):
    """The learning text must come from proposalReason, not HTML extraction."""
    db_path = make_db(
        tmp_path,
        [(1, '법안', '의원', '위원회', 'DB에 저장된 제안이유', '<p>HTML에서 추출된 텍스트</p>')],
    )
    record = load_notices_from_db(db_path)[0]
    assert record['proposal_reason'] == 'DB에 저장된 제안이유'
    assert 'HTML에서 추출된' not in record['proposal_reason']


def test_load_notices_from_db_handles_empty_values_and_empty_db(tmp_path: Path):
    # Empty-string columns mirror the real NOT NULL DEFAULT '' schema.
    db_path = make_db(tmp_path, [(1, '법안', '', '', '', None)])
    assert load_notices_from_db(db_path) == [
        {
            'notice_num': 1,
            'subject': '법안',
            'proposer_category': '',
            'committee': '',
            'proposal_reason': '',
        }
    ]
    empty_db = make_db(tmp_path / 'empty', [])
    assert load_notices_from_db(empty_db) == []


def test_extract_sample_stratifies_by_proposal_reason_length(tmp_path: Path):
    db_path = make_db(
        tmp_path,
        [
            (1, '장문', '의원', '위원회', LONG_REASON, None),
            (2, '중문', '의원', '위원회', MEDIUM_REASON, None),
            (3, '단문', '의원', '위원회', SHORT_REASON, None),
            (4, '제외', '의원', '위원회', '너무 짧음', None),  # < 80 chars: no bucket
        ],
    )
    records = extract_sample(db_path, per_bucket=10)
    by_num = {record['notice_num']: record for record in records}
    assert set(by_num) == {1, 2, 3}
    assert by_num[1]['length_bucket'] == 'long'
    assert by_num[2]['length_bucket'] == 'medium'
    assert by_num[3]['length_bucket'] == 'short'
    assert by_num[1]['proposal_reason'] == LONG_REASON


def test_jsonl_snapshot_roundtrip(tmp_path: Path):
    db_path = make_db(tmp_path, [(1, '법안', '의원', '위원회', SHORT_REASON, None)])
    records = extract_sample(db_path, per_bucket=10)
    snapshot = tmp_path / 'sample.jsonl'
    write_notices_jsonl(records, snapshot)
    assert load_notices_jsonl(snapshot) == records
