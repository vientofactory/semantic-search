"""Measure real FAISS files and export identical candidate windows for board replay.

Run encode once, then rank once per index in independent processes. The split
also avoids loading two OpenMP runtimes together on macOS.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SETS = ('eval_search_priorities', 'eval_queries', 'eval_holdout_queries', 'unrelated_queries')


def read_jsonl(path):
    return [
        json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()
    ]


def dataset():
    rows = []
    for name in SETS:
        for original in read_jsonl(ROOT / 'data' / f'{name}.jsonl'):
            row = dict(original, dataset=name)
            if 'relevant' not in row:
                row['relevant'] = [row['notice_num']] if 'notice_num' in row else []
            rows.append(row)
    return rows


def encode(output):
    from lawcast_semantic.aliases import expand_query
    from lawcast_semantic.embedding import KoreanEmbedder
    from lawcast_semantic.preprocess import normalize_text

    rows = dataset()
    queries = [normalize_text(expand_query(row['query']).text) for row in rows]
    model = KoreanEmbedder(device='cpu')
    times = []
    model.embed_query(queries[0])
    for query in queries:
        start = perf_counter()
        model.embed_query(query)
        times.append((perf_counter() - start) * 1000)
    vectors = model.embed_texts(queries, batch_size=16)
    np.savez(output, vectors=vectors, queries=np.array([row['query'] for row in rows]), times=times)
    latency = np.percentile(times, [50, 95])
    print(f'encoded {len(rows)} queries on {model.device}; p50/p95 ms: {latency}')


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def rank(args):
    from lawcast_semantic.indexing import VectorIndex

    rows = dataset()
    matrix = np.load(args.vectors, allow_pickle=False)
    if matrix['queries'].tolist() != [row['query'] for row in rows]:
        raise ValueError('query set differs from the encoded vector set')
    vectors = matrix['vectors']
    before_rss = rss_bytes()
    start = perf_counter()
    index, meta = VectorIndex.load(args.index, args.id_map)
    load_ms = (perf_counter() - start) * 1000
    index_rss = rss_bytes() - before_rss
    records = {row['chunk_id']: row for row in read_jsonl(args.chunks)}
    chunk_ids = meta['chunk_ids']
    if len(chunk_ids) != index.index.ntotal or set(chunk_ids) != records.keys():
        raise ValueError('candidate and chunk data are from different builds')
    from lawcast_semantic.chunking import compute_chunks_fingerprint

    if compute_chunks_fingerprint(list(records.values())) != meta['chunks_fingerprint']:
        raise ValueError('candidate index fingerprint differs from chunk snapshot')
    for vector in vectors[:8]:
        index.search(vector, k=args.k)
    order = np.random.default_rng(42).permutation(len(vectors))
    timings = []
    results = [None] * len(rows)
    for iteration in range(args.repeats):
        for i in order:
            start = perf_counter()
            found = index.search(vectors[i], k=args.k)
            timings.append((perf_counter() - start) * 1000)
            if iteration != 0:
                continue
            found.sort(key=lambda hit: (-hit[1], chunk_ids[hit[0]]))
            hits = []
            for pos, score in found:
                chunk = records[chunk_ids[pos]]
                hits.append(
                    {
                        'chunkId': chunk['chunk_id'],
                        'noticeNum': chunk['notice_num'],
                        'score': score,
                        'section': chunk['section'],
                        'evidence': chunk['section'] != 'AI 요약'
                        and bool(rows[i].get('evidence_terms'))
                        and all(term in chunk['text'] for term in rows[i]['evidence_terms']),
                    }
                )
            results[i] = dict(rows[i], hits=hits)
    stats = dict(
        index_type=index.index_type,
        index_bytes=args.index.stat().st_size,
        load_ms=load_ms,
        index_peak_rss_delta_bytes=index_rss,
        queries=len(rows),
        repeats=args.repeats,
        k=args.k,
        milliseconds=dict(zip(('p50', 'p95'), np.percentile(timings, [50, 95]), strict=True)),
    )
    args.out.write_text(
        json.dumps(dict(stats=stats, results=results), ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    encoder = commands.add_parser('encode')
    encoder.add_argument('--out', type=Path, required=True)
    ranker = commands.add_parser('rank')
    ranker.add_argument('--vectors', type=Path, required=True)
    ranker.add_argument('--index', type=Path, required=True)
    ranker.add_argument('--id-map', type=Path, required=True)
    ranker.add_argument('--chunks', type=Path, default=ROOT / 'artifacts/chunks.jsonl')
    ranker.add_argument('--k', type=int, default=100)
    ranker.add_argument('--repeats', type=int, default=3)
    ranker.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'encode':
        encode(args.out)
    else:
        rank(args)


if __name__ == '__main__':
    main()
