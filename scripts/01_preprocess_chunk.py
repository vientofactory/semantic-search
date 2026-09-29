"""Stage 1: preprocess + chunk legislation notices.

Reads sample notices JSONL, cleans each `proposal_reason`, splits into
section-aware chunks, and writes chunks JSONL for stage 2.

Usage:
    python scripts/01_preprocess_chunk.py [--input data/sample_notices.jsonl] [--out artifacts/chunks.jsonl]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lawcast_semantic import chunk_notices  # noqa: E402
from lawcast_semantic.preprocess import detect_sections, normalize_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=config.SAMPLE_DATA_PATH)
    parser.add_argument('--out', type=Path, default=config.CHUNKS_PATH)
    args = parser.parse_args()

    notices = []
    with args.input.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                notices.append(json.loads(line))

    chunks = chunk_notices(notices)

    section_counts: dict[str, int] = {}
    for chunk in chunks:
        section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')

    sizes = [chunk.char_count for chunk in chunks]
    print(f'notices read        : {len(notices)}')
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
