"""Stage 4c: probe the full corpus with unrelated queries to test filtering.

Runs every query in `data/unrelated_queries.jsonl` through the real dense
index (top-k window), the tier split (`search_tiered`) and the lexical-evidence
gate (`hybrid.filter_by_lexical_evidence`), and records per-query evidence for
filter analysis: top-1 score, clear/weak counts, evidence-kept counts,
per-hit token overlaps, and the top hits with their shared tokens.

Categories in the query file: unrelated families (`expect: empty`) that should
produce no clear hits and ideally no evidence-kept hits, `vague-boundary`
queries that sit on the valid/noise border (measured either way, never used to
justify a stricter gate), and `valid-control` queries that must stay served
(the false-reject guard for any proposed filter change).

Usage:
    cd semantic-search && .venv/bin/python scripts/08_unrelated_probe.py
    cd semantic-search && .venv/bin/python scripts/08_unrelated_probe.py \\
        --k 30 --out /tmp/unrelated_out.json

Output: JSON document to stdout (or `--out`); a human-readable category
summary goes to stderr so piping stdout still yields pure JSON.
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
from lawcast_semantic.hybrid import (  # noqa: E402
    chunk_tokens,
    filter_by_lexical_evidence,
    lexical_tokens,
    overlap_count,
)
from lawcast_semantic.search import SemanticSearcher  # noqa: E402


def load_queries(path: Path) -> list[dict]:
    """Read the probe set; every line needs query/category/expect."""
    entries = []
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        for key in ('query', 'category', 'expect'):
            if key not in entry:
                raise ValueError(f'{path}:{lineno} misses {key!r}')
        entries.append(entry)
    if not entries:
        raise ValueError(f'probe set is empty: {path}')
    return entries


def probe(searcher: SemanticSearcher, entry: dict, k: int) -> dict:
    """Run one query through dense, tiers and the lexical gate."""
    query = entry['query']
    hits = searcher.search(query, k=k)
    tiered = searcher.search_tiered(query, k=k)
    kept = filter_by_lexical_evidence(query, hits, searcher.chunks_by_id)

    query_tokens = lexical_tokens(query)
    per_hit = []
    for hit in hits:
        doc_tokens = chunk_tokens(searcher.chunks_by_id[hit.chunk_id])
        shared = sorted(
            {token for token in query_tokens for doc in doc_tokens if token in doc or doc in token}
        )
        per_hit.append(
            {
                'chunk_id': hit.chunk_id,
                'notice_num': hit.notice_num,
                'score': round(float(hit.score), 4),
                'overlap': overlap_count(query_tokens, doc_tokens),
                'shared': shared,
                'clear': hit.score >= config.CLEAR_SIMILARITY,
            }
        )
    kept_ids = {hit.chunk_id for hit in kept}
    for row in per_hit:
        row['evidence'] = row['chunk_id'] in kept_ids

    top1 = float(hits[0].score) if hits else 0.0
    return {
        'query': query,
        'category': entry['category'],
        'expect': entry['expect'],
        'top1': round(top1, 4),
        'window': len(hits),
        'clear': len(tiered.results),
        'weak': len(tiered.weak_results),
        'keptChunks': len(kept),
        'keptNotices': len({hit.notice_num for hit in kept}),
        'maxOverlap': max((row['overlap'] for row in per_hit), default=0),
        'queryTokens': query_tokens,
        'hits': per_hit,
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate leak/false-reject rates per category."""
    summary: dict[str, dict] = {}
    for row in rows:
        bucket = summary.setdefault(
            row['category'], {'n': 0, 'clearServed': 0, 'keptAny': 0, 'top1Sum': 0.0}
        )
        bucket['n'] += 1
        bucket['clearServed'] += 1 if row['clear'] > 0 else 0
        bucket['keptAny'] += 1 if row['keptChunks'] > 0 else 0
        bucket['top1Sum'] += row['top1']
    for bucket in summary.values():
        bucket['avgTop1'] = round(bucket.pop('top1Sum') / bucket['n'], 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=config.DATA_DIR / 'unrelated_queries.jsonl')
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--out', type=Path, default=None, help='also save the JSON document here')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be a positive integer')

    entries = load_queries(args.queries)
    print(f'loading embedder ({args.model})...', file=sys.stderr)
    embedder = KoreanEmbedder(args.model)
    print('loading searcher...', file=sys.stderr)
    try:
        searcher = SemanticSearcher.load(embedder)
    except ValueError as exc:
        parser.error(str(exc))

    rows = [probe(searcher, entry, args.k) for entry in entries]
    document = {
        'k': args.k,
        'model': embedder.model_name,
        'indexedChunks': len(searcher.chunk_ids),
        'summary': summarize(rows),
        'rows': rows,
    }
    if args.out is not None:
        args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'saved {args.out}', file=sys.stderr)

    print(f'indexedChunks={len(searcher.chunk_ids)} queries={len(rows)}', file=sys.stderr)
    for category, bucket in sorted(document['summary'].items()):
        print(
            f'  {category:22} n={bucket["n"]:>2} '
            f'avgTop1={bucket["avgTop1"]:.3f} '
            f'clearServed={bucket["clearServed"]} keptAny={bucket["keptAny"]}',
            file=sys.stderr,
        )
    print(json.dumps(document, ensure_ascii=False))


if __name__ == '__main__':
    main()
