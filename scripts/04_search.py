"""Stage 4: run semantic search queries against the FAISS index.

Loads all pipeline artifacts, embeds each query with the same Korean model,
and prints top-k results ranked by cosine similarity.

Usage:
    python scripts/04_search.py --query "ESG 공시 의무화" [--query "..."] [--k 5]
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
from lawcast_semantic.search import SemanticSearcher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', action='append', required=True, help='repeatable')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--json', action='store_true', help='machine-readable output')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be a positive integer')

    embedder = KoreanEmbedder(args.model)
    try:
        searcher = SemanticSearcher.load(embedder)
    except ValueError as exc:
        # Stale/mismatched artifacts are user errors: report like --k validation.
        parser.error(str(exc))
    print(f'model: {args.model} | indexed chunks: {len(searcher.chunk_ids)}\n')

    for query in args.query:
        results = searcher.search(query, k=args.k)
        if args.json:
            print(
                json.dumps(
                    {
                        'query': query,
                        'results': [vars(result) for result in results],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            continue
        print(f'query: {query}')
        for rank, result in enumerate(results, start=1):
            excerpt = result.text.replace('\n', ' ')[:80]
            print(
                f'  {rank}. score={result.score:.4f} notice={result.notice_num} '
                f'[{result.section}] {result.subject}'
            )
            print(f'     {excerpt}...')
        print()


if __name__ == '__main__':
    main()
