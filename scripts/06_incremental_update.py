"""Stage 6 (incremental): bring the artifact set in line with the corpus.

Detects new/changed/deleted notices against the committed artifacts, embeds
only the chunks whose embedded text changed, reuses stored rows otherwise, and
rewrites the artifact set so it equals a full rebuild of the current corpus —
the output passes `SemanticSearcher.load` unchanged. Deletes are handled: rows
whose notices left the corpus are dropped. A no-change run reports in seconds
and never loads the embedding model. Crash recovery is re-running this script;
design and consistency rules: agent_memories/07-incremental-indexing/plan.md.

Usage:
    python scripts/06_incremental_update.py
        [--db ../backend/lawcast.db | --input data/sample_notices.jsonl]
        [--plan-only] [--model nlpai-lab/KURE-v1]
        [--chunks artifacts/chunks.jsonl] [--embeddings artifacts/embeddings.npz]
        [--index artifacts/faiss.index] [--id-map artifacts/id_map.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic.omp_env import use_single_threaded_omp  # noqa: E402

use_single_threaded_omp()  # faiss and torch share this process; see omp_env.py

from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.datasource import load_notices_from_db, load_notices_jsonl  # noqa: E402
from lawcast_semantic.incremental import apply_update, plan_update  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        '--input',
        type=Path,
        default=config.SAMPLE_DATA_PATH,
        help='JSONL snapshot of DB proposalReason records (default)',
    )
    source.add_argument(
        '--db',
        type=Path,
        help='LawCast DB path: sync straight from notice_archives.proposalReason',
    )
    parser.add_argument('--chunks', type=Path, default=config.CHUNKS_PATH)
    parser.add_argument('--embeddings', type=Path, default=config.EMBEDDINGS_PATH)
    parser.add_argument('--index', type=Path, default=config.FAISS_INDEX_PATH)
    parser.add_argument('--id-map', type=Path, default=config.ID_MAP_PATH)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument(
        '--plan-only',
        action='store_true',
        help='print the change plan as JSON and exit without writing artifacts',
    )
    args = parser.parse_args()

    if args.db:
        notices = load_notices_from_db(args.db)
        source_label = f'{args.db} (notice_archives.proposalReason)'
    else:
        notices = load_notices_jsonl(args.input)
        source_label = str(args.input)

    try:
        plan = plan_update(
            notices,
            chunks_path=args.chunks,
            embeddings_path=args.embeddings,
            model_name=args.model,
        )

        if args.plan_only:
            print(
                json.dumps(
                    {
                        'notices': len(notices),
                        'chunks_total': len(plan.new_records),
                        'added_notices': list(plan.added_notices),
                        'updated_notices': list(plan.updated_notices),
                        'deleted_notices': list(plan.deleted_notices),
                        'to_embed': len(plan.embed_records),
                        'reused': len(plan.reused_chunk_ids),
                        'dropped': len(plan.dropped_chunk_ids),
                        'has_changes': plan.has_changes,
                    },
                    ensure_ascii=False,
                )
            )
            return

        # Deferred: loading the model is the slow part and a no-change run skips it.
        embed_texts = None
        if plan.needs_embedding:
            from lawcast_semantic.embedding import KoreanEmbedder

            embedder = KoreanEmbedder(args.model)
            embed_texts = embedder.embed_texts

        report = apply_update(
            plan,
            embed_texts,
            args.model,
            chunks_path=args.chunks,
            embeddings_path=args.embeddings,
            index_path=args.index,
            id_map_path=args.id_map,
        )
    except ValueError as exc:
        # Stale/mismatched artifacts are user errors: report like --k validation.
        sys.exit(f'error: {exc}')

    print(f'notices read        : {len(notices)}')
    print(f'source              : {source_label}')
    print(f'chunks (corpus)     : {report.chunks_total}')
    print(f'added notices       : {len(report.added_notices)} {report.added_notices[:10]}')
    print(f'updated notices     : {len(report.updated_notices)} {report.updated_notices[:10]}')
    print(f'deleted notices     : {len(report.deleted_notices)} {report.deleted_notices[:10]}')
    print(f'chunks to embed     : {report.embedded_count}')
    print(f'chunks reused       : {report.reused_count}')
    print(f'rows dropped        : {report.dropped_count}')
    if report.wrote_artifacts:
        print(f'artifacts           : rewritten ({args.chunks.parent})')
    else:
        print('artifacts           : unchanged (already in sync)')
    print(f'chunks fingerprint  : {report.chunks_fingerprint[:12]}')


if __name__ == '__main__':
    main()
