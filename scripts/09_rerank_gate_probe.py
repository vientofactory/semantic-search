"""Stage 4d: sweep rerank cutoffs over probe windows with the backend rule.

Reads a `scripts/08_unrelated_probe.py` JSON document (dense windows plus
lexical evidence per hit), scores each window with the cross-encoder shadow
scorer, and applies the backend gate rule *exactly* as
`SemanticSearchService` does in enforce mode:

- query-level bypass: dense top-1 >= `--bypass` serves the window ungated;
- otherwise per-hit keep <=> (rerank >= cutoff) OR (bypassed AND evidence),
  unscored tail falls back to evidence, notice-OR over kept chunks.

Reports, per cutoff, unrelated queries fully emptied versus valid queries
damaged (kept-any empty, plus labeled targets lost), so the cutoff decision
rests on the shipped rule rather than the calibration approximation.

Usage:
    cd semantic-search && .venv/bin/python scripts/08_unrelated_probe.py \\
        --queries /tmp/combined_queries.jsonl --out /tmp/combined_out.json
    cd semantic-search && .venv/bin/python scripts/09_rerank_gate_probe.py \\
        --in /tmp/combined_out.json --out /tmp/rerank_gate_out.json

Output: JSON document to `--out` (or stdout); a cutoff table to stderr.
CPU-only by default to simulate the production sidecar image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.chunking import compose_embedding_text, load_chunks_jsonl  # noqa: E402
from lawcast_semantic.reranking import load_reranker  # noqa: E402

CUTOFFS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)


def apply_rule(row: dict, cutoff: float, bypass: float) -> set[int]:
    """Notices served under the backend enforce rule at this cutoff."""
    bypassed = row['top1'] >= bypass
    if bypassed:
        return {hit['notice_num'] for hit in row['hits']}
    kept: set[int] = set()
    for hit in row['hits']:
        rerank = hit.get('rerank')
        if rerank is None:
            hit_kept = bool(hit['evidence'])
        else:
            hit_kept = rerank >= cutoff
        if hit_kept:
            kept.add(hit['notice_num'])
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in', dest='input', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--rk', type=int, default=20, help='hits scored per window')
    parser.add_argument('--bypass', type=float, default=0.6)
    parser.add_argument('--model', default=config.RERANK_MODEL)
    parser.add_argument('--device', default=config.RERANK_DEVICE)
    args = parser.parse_args()
    if args.rk < 1:
        parser.error('--rk must be a positive integer')

    document = json.loads(args.input.read_text(encoding='utf-8'))
    rows = document['rows']
    chunks = {record['chunk_id']: record for record in load_chunks_jsonl(config.CHUNKS_PATH)}
    print(f'windows={len(rows)} chunks={len(chunks)}', file=sys.stderr)

    print(f'loading reranker ({args.model} on {args.device})...', file=sys.stderr)
    reranker = load_reranker(args.model, args.device)

    scored_ms = 0
    for row in rows:
        window = row['hits'][: args.rk]
        texts = []
        for hit in window:
            chunk = chunks[hit['chunk_id']]
            texts.append(compose_embedding_text(chunk['subject'], chunk['text']))
        t0 = time.time()
        scores = reranker.score(row['query'], texts)
        scored_ms += int((time.time() - t0) * 1000)
        for hit, score in zip(window, scores):
            hit['rerank'] = round(float(score), 4)
        for hit in row['hits'][args.rk :]:
            hit['rerank'] = None
    print(f'scored {len(rows)} windows in {scored_ms}ms', file=sys.stderr)

    table = []
    for cutoff in CUTOFFS:
        entry: dict[str, object] = {'cutoff': cutoff}
        for family, predicate in (
            ('unrelated', lambda r: r['expect'] == 'empty'),
            ('labeled', lambda r: r['category'] == 'valid-labeled'),
            ('control', lambda r: r['category'] == 'valid-control'),
            ('vague', lambda r: r['category'] == 'vague-boundary'),
        ):
            members = [row for row in rows if predicate(row)]
            kept_map = {row['query']: apply_rule(row, cutoff, args.bypass) for row in members}
            entry[f'{family}_n'] = len(members)
            entry[f'{family}_emptied'] = sum(1 for kept in kept_map.values() if not kept)
            entry[f'{family}_kept'] = sum(1 for kept in kept_map.values() if kept)
        labeled_lost = sorted(
            row['query']
            for row in rows
            if row['category'] == 'valid-labeled'
            and row.get('notice_num') is not None
            and int(row['notice_num']) not in apply_rule(row, cutoff, args.bypass)
        )
        entry['labeled_targets_lost'] = labeled_lost
        table.append(entry)

    print(
        f'{"cutoff":>7} | {"unrelatedEmpty":>14} | {"labeledKept":>11} '
        f'{"targetsLost":>11} | {"controlKept":>11} | {"vagueKept":>9}',
        file=sys.stderr,
    )
    for entry in table:
        print(
            f'{entry["cutoff"]:>7} | {entry["unrelated_emptied"]:>3}/{entry["unrelated_n"]:<9} | '
            f'{entry["labeled_kept"]:>3}/{entry["labeled_n"]:<7} '
            f'{len(entry["labeled_targets_lost"]):>11} | '
            f'{entry["control_kept"]:>3}/{entry["control_n"]:<7} | '
            f'{entry["vague_kept"]:>3}/{entry["vague_n"]}',
            file=sys.stderr,
        )

    payload = {
        'input': str(args.input),
        'model': args.model,
        'device': args.device,
        'rk': args.rk,
        'bypass': args.bypass,
        'scoredMs': scored_ms,
        'table': table,
        'rows': [
            {
                'query': row['query'],
                'category': row['category'],
                'expect': row['expect'],
                'top1': row['top1'],
                'bypassed': row['top1'] >= args.bypass,
                'hits': [
                    {
                        'notice_num': hit['notice_num'],
                        'score': hit['score'],
                        'evidence': hit['evidence'],
                        'rerank': hit.get('rerank'),
                    }
                    for hit in row['hits']
                ],
            }
            for row in rows
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.write_text(text, encoding='utf-8')
        print(f'saved {args.out}', file=sys.stderr)
    else:
        print(text)


if __name__ == '__main__':
    main()
