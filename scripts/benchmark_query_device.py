"""Compare native CPU/MPS on the same local model and artifacts.

Run from the sidecar directory:
    .venv/bin/python scripts/benchmark_query_device.py > /tmp/query-device.json

Each worker loads the normal serving engine and runs the real /search handler.
The first request is after readiness (including the accelerator probe); warm
requests rotate the four queries. No server or scheduler is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ('비트코인 과세', '전기차 충전소 늘리기', '전세사기 피해자 보호', '인앱결제 수수료')


def worker(device: str, first: int, rounds: int) -> dict:
    sys.path.insert(0, str(ROOT))
    from service import app as sidecar

    sidecar.service_logger.setLevel(logging.WARNING)
    sidecar.load_engine(sidecar.STATE)
    state = sidecar.STATE.snapshot()
    if state['status'] != 'ready' or state['device'] != device:
        raise RuntimeError(f'Requested {device}; readiness: {sidecar.health()}')
    searcher = state['searcher']
    rank = searcher._rank
    raw_hits = []

    def capture_rank(vector, k):
        nonlocal raw_hits
        raw_hits = rank(vector, k)
        return raw_hits

    searcher._rank = capture_rank
    sequence = [QUERIES[first]] + [
        QUERIES[(first + step + 1) % len(QUERIES)] for step in range(rounds * len(QUERIES))
    ]
    samples = []
    for i, query in enumerate(sequence):
        response = sidecar.search(query, k=sidecar.MAX_K)
        timings = response.timings.model_dump()
        if not timings['denseStages']['embeddingStages']:
            raise RuntimeError('Expected query embedding stage timings')
        samples.append(
            {
                'query': query,
                'phase': 'first' if i == 0 else 'repeat',
                'timings': timings,
                'raw': [[hit.chunk_id, hit.score] for hit in raw_hits[:10]],
                'clear': [[hit.chunkId, hit.score] for hit in response.results],
                'weak': [[hit.chunkId, hit.score] for hit in response.weakResults],
            }
        )
    # Check instrumented vs stock encoding with identical retrieval settings.
    original_profile = searcher.embedder.embed_query_profiled
    searcher.embedder.embed_query_profiled = None
    parity = {}
    for query in QUERIES:
        baseline = sidecar.search(query, k=sidecar.MAX_K)
        stock = [hit.chunk_id for hit in raw_hits[:10]]
        recorded = next(sample for sample in samples if sample['query'] == query)
        parity[query] = (
            stock == [chunk for chunk, _ in recorded['raw']]
            and [hit.chunkId for hit in baseline.results]
            == [chunk for chunk, _ in recorded['clear']]
            and [hit.chunkId for hit in baseline.weakResults]
            == [chunk for chunk, _ in recorded['weak']]
        )
    searcher.embedder.embed_query_profiled = original_profile
    return {
        'health': sidecar.health(),
        'fingerprint': sidecar.STATE.loaded_fingerprint,
        'k': sidecar.MAX_K,
        'samples': samples,
        'stockRankParity': parity,
    }


def hashes() -> dict:
    sys.path.insert(0, str(ROOT))
    from lawcast_semantic import config

    result = {}
    for path in (config.CHUNKS_PATH, config.FAISS_INDEX_PATH, config.ID_MAP_PATH):
        with path.open('rb') as handle:
            result[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=int, default=12, help='Fresh processes per device')
    parser.add_argument(
        '--rounds', type=int, default=2, help='Four-query rounds after first search'
    )
    parser.add_argument('--worker', choices=('cpu', 'mps'))
    parser.add_argument('--first', type=int, default=0)
    args = parser.parse_args()
    if args.sessions < 1 or args.rounds < 1:
        parser.error('sessions and rounds must be positive')
    if args.worker:
        print(json.dumps(worker(args.worker, args.first, args.rounds), ensure_ascii=False))
        return
    os.chdir(ROOT)
    before = hashes()
    workers = []
    for session in range(args.sessions):
        for device in ('cpu', 'mps') if session % 2 == 0 else ('mps', 'cpu'):
            print(f'Run {session + 1}/{args.sessions} on {device}', file=sys.stderr)
            env = {
                **os.environ,
                'LAWCAST_SEMANTIC_DEVICE': device,
                'HF_HUB_OFFLINE': '1',
                'OMP_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false',
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    '--worker',
                    device,
                    '--first',
                    str(session % len(QUERIES)),
                    '--rounds',
                    str(args.rounds),
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise RuntimeError(
                    f'{device} worker exited {completed.returncode}: {completed.stderr}'
                )
            workers.append(json.loads(completed.stdout.splitlines()[-1]))
    if before != hashes():
        raise RuntimeError('Index artifacts changed during benchmark')
    if len({(run['health']['model'], run['fingerprint'], run['k']) for run in workers}) != 1:
        raise RuntimeError('Runs did not use the same model, chunks and retrieval window')
    print(json.dumps({'artifactSha256': before, 'runs': workers}, ensure_ascii=False))


if __name__ == '__main__':
    main()
