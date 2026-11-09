from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return import_module('12_eval_relevance')


def test_minimum_only_drops_weak_without_altering_strong(replay):
    hits = [
        SimpleNamespace(notice_num=i, score=score)
        for i, score in enumerate((0.55, 0.45, 0.449, 0.25), start=1)
    ]

    assert replay.cosine_window(hits, 0.25, 0.45, 5, widening=True) == {
        'strong': [1, 2],
        'weak': [3, 4],
    }
    assert replay.cosine_window(hits, 0.45, 0.45, 5, widening=True) == {
        'strong': [1, 2],
        'weak': [],
    }


def test_prefix_widens_on_duplicate_notices_and_keeps_inclusive_thresholds(replay):
    hits = [SimpleNamespace(notice_num=1, score=0.51)] * 15
    hits.extend(SimpleNamespace(notice_num=i, score=0.51) for i in range(2, 6))
    hits.extend(SimpleNamespace(notice_num=i, score=0.49) for i in range(6, 10))

    assert replay.cosine_window(hits, 0.49, 0.51, 5, widening=True) == {
        'strong': [1, 2, 3, 4, 5],
        'weak': [6, 7, 8, 9],
    }


def test_topic_pairing_does_not_treat_queries_in_a_topic_as_independent(replay):
    rows = [
        {'group': group, 'baseline': False, 'raised': True}
        for group in ('lease', 'election')
        for _ in range(4)
    ]
    result = replay.cluster_test(rows, 'raised', lambda _row, outcome: outcome)

    assert result == {'topic_deltas': {'lease': 4, 'election': 4}, 'p': 0.5}


def test_hard_negatives_are_unique_and_linked_to_twelve_topics(replay):
    rows = replay.cosine_cases()
    hard = [row for row in rows if row.get('anchor_notice')]

    assert len({replay.normalized(row['query']) for row in rows}) == len(rows)
    assert len(hard) == 48
    assert len({(row['group'], row['anchor_notice']) for row in hard}) == 12
    assert all(row['negative'] and not row.get('targets') for row in hard)
