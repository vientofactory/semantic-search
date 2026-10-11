"""Operational report: capability coverage of the artifact set being served.

Answers one question for a release: **which retrieval capabilities does the
index cover, per notice class** — and, against a saved baseline, which ones did
this release take away. It is the artifact-side counterpart of
`tests/test_capability_gate.py`: the gate compares two RULES on the same corpus,
this compares two ARTIFACT SETS (e.g. the live index against
`artifacts/backup-pre-ai-summary/`) per notice, so a swap cannot hide behind an
equal count (405 notices losing `title_text` while 405 gain `derived:ai_summary`
nets to zero on a count line, and to a reported loss here).

No model and no FAISS are needed — every capability is a property of the chunk
text plus the source record — so a full-corpus run is seconds and cannot hit the
faiss/torch OpenMP conflict. Two integrity lines come for free from the same
reads: whether `chunks.jsonl` and `id_map.json` describe the same rows, and
whether the chunk fingerprint still matches the one the vectors were built with
(the stale-set hazard of agent_memories/24-ai-summary-chunking §8.1, where an
old image silently rewrote the index back to a summary-less set).

The source side is the CURRENT corpus (`--db` or a JSONL snapshot), so a DB that
moved on since the index was built shows up as reduced coverage: the report
describes the index against the source it is supposed to serve, agreement
included.

Usage:
    # before a release / deploy: the artifact set being served now
    python scripts/07_capability_report.py --db ../backend/lawcast.db --save /tmp/cap-before.json

    # after (or against the rollback set), one line of deltas:
    python scripts/07_capability_report.py --db ../backend/lawcast.db --compare /tmp/cap-before.json
    python scripts/07_capability_report.py --artifacts artifacts/backup-pre-ai-summary \\
        --db ../backend/lawcast.db --compare /tmp/cap-before.json

    # after a deploy: compare the DEPLOYED code's chunk set (no artifacts are
    # written) against the pre-deploy baseline. This is the CI release guard of
    # `deploy-semantic-search.yml`: `01_preprocess_chunk.py` rebuilds the chunks
    # the deployed code would serve, and `--chunks` reports/compares that
    # candidate instead of the served artifact set (integrity is n/a — a
    # candidate set is not the served one).
    python scripts/01_preprocess_chunk.py --db ../backend/lawcast.db --out /tmp/candidate.jsonl
    python scripts/07_capability_report.py --chunks /tmp/candidate.jsonl \\
        --db ../backend/lawcast.db --compare /tmp/cap-before.json

Exit codes: 0 = no capability lost and the artifact set is consistent,
1 = a capability class was lost or the set is inconsistent, 2 = user error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.capabilities import (  # noqa: E402
    DERIVED_PREFIX,
    INDEXED,
    SECTION_PREFIX,
    TITLE_TEXT,
    capability_index_from_records,
    compare_capability_indexes,
)
from lawcast_semantic.chunking import (  # noqa: E402
    compute_chunks_fingerprint,
    load_chunks_jsonl,
)
from lawcast_semantic.datasource import load_notices_from_db, load_notices_jsonl  # noqa: E402

# Capability classes are printed in this order: the two corpus-wide ones first,
# then the section labels and the derived fields alphabetically.
CLASS_ORDER = (INDEXED, TITLE_TEXT, SECTION_PREFIX, DERIVED_PREFIX)


def class_order(capability: str) -> tuple[int, str]:
    for rank, prefix in enumerate(CLASS_ORDER):
        if capability == prefix or capability.startswith(prefix):
            return (rank, capability)
    return (len(CLASS_ORDER), capability)


def notice_count(count: int) -> str:
    return f'{count} notice' if count == 1 else f'{count} notices'


def render_counts(index: Mapping[str, set[int]]) -> str:
    """One line: `indexed=N title_text=N section:<label>=N derived:<field>=N`."""
    return ' '.join(
        f'{capability}={len(index[capability])}' for capability in sorted(index, key=class_order)
    )


def group_records_by_notice(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record['notice_num']), []).append(record)
    return grouped


def artifact_integrity(records: list[dict], id_map_path: Path) -> tuple[bool, str]:
    """(ok, line) for the artifact set: row agreement and fingerprint match.

    The reported capabilities come from `chunks.jsonl`, so these two checks are
    what ties them to the index the sidecar actually serves.
    """
    try:
        payload = json.loads(id_map_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return False, f'MISMATCH — {id_map_path.name} unreadable ({exc})'
    row_ids = [record['chunk_id'] for record in records]
    stored_ids = payload.get('chunk_ids')
    if stored_ids is None:
        return False, f'MISMATCH — {id_map_path.name} carries no chunk_ids'
    same_rows = len(row_ids) == len(stored_ids) and set(row_ids) == set(stored_ids)
    fingerprint = compute_chunks_fingerprint(records)
    stored_fingerprint = payload.get('chunks_fingerprint')
    fingerprint_ok = stored_fingerprint == fingerprint
    if same_rows and fingerprint_ok:
        return True, f'OK ({len(row_ids)} rows == id_map rows, fingerprint {fingerprint[:12]})'
    reasons = []
    if not same_rows:
        reasons.append(
            f'chunks.jsonl lists {len(row_ids)} rows but id_map.json {len(stored_ids)} ids'
        )
    if not fingerprint_ok:
        reasons.append(
            f'chunk fingerprint {fingerprint[:12]} != id_map {str(stored_fingerprint)[:12]}'
        )
    return (
        False,
        f'MISMATCH — {"; ".join(reasons)} (stale set: re-run scripts/06_incremental_update.py)',
    )


def load_baseline(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        sys.exit(f'error: cannot read baseline report {path}: {exc}')
    if not isinstance(payload.get('capabilities'), dict):
        sys.exit(f'error: {path} is not a capability report (no "capabilities" object)')
    return payload


def report_capabilities(notices: list[dict], index: Mapping[str, set[int]]) -> dict:
    return {
        'generated_at': datetime.now(UTC).isoformat(),
        'notices': len(notices),
        'capabilities': {capability: sorted(index[capability]) for capability in index},
    }


def print_comparison(
    baseline: dict, index: Mapping[str, set[int]], notice_universe: set[int]
) -> bool:
    """Print the one-line delta and the release verdict; True when nothing was lost."""
    baseline_index = {
        capability: set(nums) for capability, nums in baseline['capabilities'].items()
    }
    baseline_universe = set(baseline.get('notice_nums') or ())
    for nums in baseline_index.values():
        baseline_universe |= nums
    common = baseline_universe & notice_universe
    restricted_baseline = {
        capability: nums & common for capability, nums in baseline_index.items() if nums & common
    }
    restricted_candidate = {
        capability: nums & common for capability, nums in index.items() if nums & common
    }
    diff = compare_capability_indexes(restricted_baseline, restricted_candidate)

    changed = sorted({*diff.lost, *diff.gained}, key=class_order)
    deltas = (
        ' | '.join(
            f'{capability} {len(restricted_baseline.get(capability, ()))}'
            f' -> {len(restricted_candidate.get(capability, ()))}'
            for capability in changed
        )
        or 'none'
    )
    print(f'capability deltas   : {deltas}')
    print(
        f'notices compared    : {len(common)} common '
        f'(+{len(notice_universe - baseline_universe)} new, '
        f'-{len(baseline_universe - notice_universe)} gone)'
    )
    if diff.is_regression:
        detail = '; '.join(
            f'{capability} ({notice_count(len(nums))}, e.g. {sorted(nums)[:3]})'
            for capability, nums in sorted(diff.lost.items(), key=lambda item: class_order(item[0]))
        )
        print(f'release check       : LOST — {detail}')
        return False
    print('release check       : OK — no capability class lost')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--artifacts',
        type=Path,
        default=config.ARTIFACTS_DIR,
        help='artifact set to report on (default: the configured live set)',
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        '--db',
        type=Path,
        help='LawCast DB: report the capabilities of its notices (default: '
        'LAWCAST_SEMANTIC_DB_PATH)',
    )
    source.add_argument(
        '--input',
        type=Path,
        help='JSONL notice snapshot instead of the DB (offline/CI use)',
    )
    parser.add_argument(
        '--chunks',
        type=Path,
        help='candidate chunk set to report instead of the served artifacts (the '
        "deployed code's would-be index, e.g. from 01_preprocess_chunk.py --db). "
        'Integrity is reported as n/a: a candidate set is not a served artifact set',
    )
    parser.add_argument(
        '--save',
        type=Path,
        help='write the report as JSON (the --compare baseline of a later release)',
    )
    parser.add_argument(
        '--compare',
        type=Path,
        help='baseline report JSON: print the per-notice capability delta and exit 1 on a loss',
    )
    args = parser.parse_args()

    db_path = args.db or (Path(config.DB_PATH) if config.DB_PATH else None)
    if args.input is None and db_path is None:
        sys.exit('error: no corpus source: pass --db or --input, or set LAWCAST_SEMANTIC_DB_PATH')
    if args.input is None and not db_path.exists():
        sys.exit(f'error: corpus DB not found: {db_path}')

    chunks_path = args.chunks or args.artifacts / 'chunks.jsonl'
    if not chunks_path.exists():
        if args.chunks:
            sys.exit(f'error: candidate chunk set not found: {chunks_path}')
        sys.exit(f'error: no chunks.jsonl in {args.artifacts}')

    if args.input is not None:
        notices = load_notices_jsonl(args.input)
        source_label = str(args.input)
    else:
        notices = load_notices_from_db(db_path)
        source_label = f'{db_path} (notice_archives.proposalReason)'

    records = load_chunks_jsonl(chunks_path)
    notices_by_num = {int(notice['notice_num']) for notice in notices}
    index = capability_index_from_records(notices, group_records_by_notice(records))

    print(f'artifacts           : {args.artifacts}')
    print(f'source              : {source_label}')
    print(
        f'corpus              : {len(notices)} notices, {len(notices_by_num)} distinct notice_num'
    )
    indexed_count = len(index.get(INDEXED, ()))
    if args.chunks:
        # A candidate set is produced by chunking the corpus with the incoming
        # code, so it has no id_map/fingerprint to tie it to a served index.
        print(
            f'candidate chunks    : {chunks_path} '
            f'({len(records)} chunks, {indexed_count} notices indexed)'
        )
        print('index integrity     : n/a (candidate chunk set, not a served artifact set)')
        integrity_ok = True
    else:
        print(f'artifact            : {len(records)} chunks, {indexed_count} notices indexed')
        integrity_ok, integrity_line = artifact_integrity(records, args.artifacts / 'id_map.json')
        print(f'index integrity     : {integrity_line}')
    print(f'capabilities        : {render_counts(index)}')

    if args.save:
        payload = report_capabilities(notices, index)
        payload.update(
            {
                'artifacts': str(args.artifacts),
                'source': source_label,
                'chunks': len(records),
                'notice_nums': sorted(notices_by_num),
                'integrities_ok': integrity_ok,
            }
        )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        print(f'saved               : {args.save} (baseline for a later --compare)')

    lost = False
    if args.compare:
        baseline = load_baseline(args.compare)
        baseline_label = f'{baseline.get("notices")} notices, {baseline.get("source")}'
        print(f'baseline            : {args.compare} ({baseline_label})')
        lost = not print_comparison(baseline, index, notices_by_num)

    if lost or not integrity_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
