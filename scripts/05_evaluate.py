"""Evaluate retrieval quality on a labeled query set.

Computes notice-level recall@k and MRR for every query in the eval set
(data/eval_queries.jsonl), overall and per query kind (title-vocabulary vs
body-vocabulary), through the real pipeline artifacts.

Usage:
    python scripts/05_evaluate.py [--eval data/eval_queries.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic.omp_env import use_single_threaded_omp  # noqa: E402

use_single_threaded_omp()

from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.embedding import KoreanEmbedder  # noqa: E402
from lawcast_semantic.evaluation import notice_rank, summarize_ranks  # noqa: E402
from lawcast_semantic.search import SemanticSearcher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', type=Path, default=config.EVAL_QUERIES_PATH)
    parser.add_argument('--model', default=config.MODEL_NAME)
    args = parser.parse_args()

    entries = []
    with args.eval.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    if not entries:
        parser.error(f'eval set is empty: {args.eval}')

    embedder = KoreanEmbedder(args.model)
    try:
        searcher = SemanticSearcher.load(embedder)
    except ValueError as exc:
        # Stale/mismatched artifacts are user errors: report like --k validation.
        parser.error(str(exc))
    depth = len(searcher.chunk_ids)

    rows = []
    for entry in entries:
        results = searcher.search(entry['query'], k=depth)
        rows.append({**entry, 'rank': notice_rank(results, entry['notice_num'])})

    kinds = sorted({row['kind'] for row in rows})
    summaries = {
        kind: summarize_ranks([row['rank'] for row in rows if row['kind'] == kind])
        for kind in kinds
    }
    summaries['all'] = summarize_ranks([row['rank'] for row in rows])

    print(
        f'eval set           : {len(rows)} queries '
        f'({", ".join(f"{k} {summaries[k]['count']}" for k in kinds)})'
    )
    print(f'search depth       : {depth} chunks (full ranking, notice-deduplicated)\n')
    print(f'{"kind":6} {"n":>3} {"recall@1":>9} {"recall@3":>9} {"recall@5":>9} {"mrr":>7}')
    for kind in [*kinds, 'all']:
        s = summaries[kind]
        print(
            f'{kind:6} {s["count"]:>3} {s["recall@1"]:>9.3f} {s["recall@3"]:>9.3f} '
            f'{s["recall@5"]:>9.3f} {s["mrr"]:>7.3f}'
        )
    misses = [row for row in rows if row['rank'] != 1]
    print(f'\nqueries not ranked 1st ({len(misses)}):')
    for row in misses:
        rank = 'miss' if row['rank'] is None else str(row['rank'])
        print(f'  [{row["kind"]}] "{row["query"]}" -> notice {row["notice_num"]}, rank {rank}')


if __name__ == '__main__':
    main()
