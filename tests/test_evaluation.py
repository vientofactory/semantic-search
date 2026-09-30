"""Tests for retrieval evaluation: metrics and eval-set integrity.

Runs without loading the embedding model (pure metric math + data checks).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lawcast_semantic.evaluation import notice_rank, summarize_ranks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notice_rank_is_one_based_over_deduplicated_notices():
    results = [SimpleNamespace(notice_num=num) for num in (101, 102, 101, 103)]
    assert notice_rank(results, 101) == 1
    assert notice_rank(results, 102) == 2
    assert notice_rank(results, 103) == 3
    assert notice_rank(results, 999) is None


def test_summarize_ranks_recall_and_mrr():
    summary = summarize_ranks([1, 2, None], ks=(1, 3))
    assert summary['count'] == 3
    assert summary['recall@1'] == pytest.approx(1 / 3)
    assert summary['recall@3'] == pytest.approx(2 / 3)
    assert summary['mrr'] == pytest.approx((1 + 0.5 + 0) / 3)


def test_eval_set_matches_corpus_vocabulary():
    """Title entries must be subject-only vocabulary, body entries body-only.

    Pins what the kind labels mean: a title query is answerable through the
    subject context (its key phrase never occurs in the body), a body query
    through body content alone (key phrase never occurs in the subject).
    """
    corpus = {}
    with (PROJECT_ROOT / 'data' / 'sample_notices.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                corpus[record['notice_num']] = record

    entries = []
    with (PROJECT_ROOT / 'data' / 'eval_queries.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))

    assert entries
    assert {entry['kind'] for entry in entries} == {'title', 'body'}
    assert sum(1 for entry in entries if entry['kind'] == 'title') >= 3
    assert sum(1 for entry in entries if entry['kind'] == 'body') >= 3
    queries = [entry['query'] for entry in entries]
    assert len(set(queries)) == len(queries)

    for entry in entries:
        record = corpus[entry['notice_num']]
        phrase = entry['key_phrase']
        in_subject = phrase in record['subject']
        in_body = phrase in (record.get('proposal_reason') or '')
        if entry['kind'] == 'title':
            assert in_subject and not in_body, entry
        else:
            assert in_body and not in_subject, entry
