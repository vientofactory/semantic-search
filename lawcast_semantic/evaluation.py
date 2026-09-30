"""Quantitative retrieval evaluation: notice-level rank metrics.

Ground truth is a notice while results are chunks, so ranks are computed on the
notice-deduplicated result list: recall@k is "correct notice in the top-k
distinct notices", MRR averages 1/rank over queries (unretrieved = 0).
"""

from __future__ import annotations


def notice_rank(results, notice_num: int) -> int | None:
    """1-based rank of notice_num in notice-deduplicated results, None if absent."""
    seen: list[int] = []
    for result in results:
        num = int(result.notice_num)
        if num not in seen:
            seen.append(num)
            if num == int(notice_num):
                return len(seen)
    return None


def summarize_ranks(ranks: list[int | None], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """recall@k and MRR over per-query ranks (None = not retrieved)."""
    total = len(ranks)
    summary = {'count': total}
    for k in ks:
        summary[f'recall@{k}'] = sum(1 for rank in ranks if rank is not None and rank <= k) / total
    summary['mrr'] = sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / total
    return summary
