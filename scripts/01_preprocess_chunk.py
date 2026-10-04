"""Stage 1: preprocess + chunk legislation notices.

Loads notice records (JSONL snapshot by default, or straight from the LawCast
DB's `proposalReason` field with --db), cleans each `proposal_reason`, splits
into section-aware chunks, and writes chunks JSONL for stage 2.

Usage:
    python scripts/01_preprocess_chunk.py
        [--input data/sample_notices.jsonl | --db lawcast.db]
        [--out artifacts/chunks.jsonl]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.chunking import chunk_notices, write_chunks_jsonl  # noqa: E402
from lawcast_semantic.datasource import (  # noqa: E402
    load_notices_from_db,
    load_notices_jsonl,
)
from lawcast_semantic.preprocess import detect_sections, normalize_text  # noqa: E402


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
        help='LawCast DB path: learn directly from notice_archives.proposalReason',
    )
    parser.add_argument('--out', type=Path, default=config.CHUNKS_PATH)
    args = parser.parse_args()

    if args.db:
        notices = load_notices_from_db(args.db)
        source_label = f'{args.db} (notice_archives.proposalReason)'
    else:
        notices = load_notices_jsonl(args.input)
        source_label = str(args.input)

    chunks = chunk_notices(notices)

    section_counts: dict[str, int] = {}
    for chunk in chunks:
        section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1

    write_chunks_jsonl([chunk.to_dict() for chunk in chunks], args.out)

    sizes = [chunk.char_count for chunk in chunks]
    print(f'notices read        : {len(notices)}')
    print(f'source              : {source_label}')
    print(f'chunks created      : {len(chunks)}')
    if sizes:
        print(
            'chunk chars         : '
            f'mean={statistics.mean(sizes):.0f} min={min(sizes)} max={max(sizes)}'
        )
    else:
        print('chunk chars         : n/a (no chunks)')
    print(f'sections detected   : {section_counts}')
    print(f'written to          : {args.out}')

    if notices:
        # Show the first section structure of one notice as a sanity check.
        demo = normalize_text(notices[0]['proposal_reason'])
        demo_sections = detect_sections(demo)
        print('\nsection structure demo (first notice):')
        for name, body in demo_sections:
            print(f'  [{name}] {len(body)} chars')

    if chunks:
        preview = chunks[0]
        print(f'\nchunk preview ({preview.chunk_id}, {preview.char_count} chars):')
        for line in preview.text.split('\n')[:3]:
            print(f'  | {line[:100]}')


if __name__ == '__main__':
    main()
