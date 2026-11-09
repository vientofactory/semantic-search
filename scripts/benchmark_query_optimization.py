"""Replay the real sidecar search path without starting a listener or update job.

Run from the sidecar directory with a cached model and existing FAISS artifacts.
Every variant uses the same ordered query sets and read-only index. Optional
ONNX input must be an export of the same model, with its SentenceTransformer
modules and tokenizer preserved. Serving configuration is never changed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Semaphore, local
from time import perf_counter

from benchmark_query_device import QUERIES, ROOT, hashes

VARIANTS = (
    'mps-detail',
    'mps-stock',
    'mps-fp16',
    'mps-gated',
    'mps-gated-detail',
    'cpu-detail',
    'cpu-stock',
    'cpu-onnx',
)


def cases() -> list[dict]:
    queries = {q: {'query': q, 'group': 'browser'} for q in QUERIES}
    for filename, limit, group in (
        ('data/eval_queries.jsonl', 16, 'labeled'),
        ('data/eval_holdout_queries.jsonl', 24, 'holdout'),
        ('data/eval_relevance_holdout.jsonl', 12, 'negative'),
        ('tests/fixtures/notices-chunk-coverage-queries.jsonl', 8, 'clause'),
    ):
        for line in (ROOT / filename).read_text().splitlines()[:limit]:
            row = json.loads(line)
            q = row['query']
            queries.setdefault(q, {'query': q, 'target': row.get('notice_num'), 'group': group})
    return list(queries.values())


def worker(args: argparse.Namespace) -> dict:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(ROOT))
    from lawcast_semantic.embedding import KoreanEmbedder, _ProfiledSentenceTransformer
    from lawcast_semantic.search import SemanticSearcher
    from service import app as sidecar

    variant = args.worker
    device = variant.split('-')[0]
    sidecar.service_logger.setLevel(logging.WARNING)
    if variant == 'cpu-onnx':
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.onnx_threads
        options.inter_op_num_threads = 1
        embedder = KoreanEmbedder.__new__(KoreanEmbedder)
        embedder.model_name = sidecar.config.MODEL_NAME
        embedder.device = 'cpu'
        embedder.model = _ProfiledSentenceTransformer(
            str(args.onnx),
            backend='onnx',
            device='cpu',
            model_kwargs={
                'provider': 'CPUExecutionProvider',
                'export': False,
                'session_options': options,
            },
        )
        embedder.dimension = int(embedder.model.get_embedding_dimension())
        KoreanEmbedder._probe(embedder.model, 'onnx')
        searcher = SemanticSearcher.load(embedder)
        sidecar.STATE.record_device('cpu')
        sidecar.STATE.mark_ready(searcher, embedder.model_name)
    else:
        sidecar.load_engine(sidecar.STATE)
        state = sidecar.STATE.snapshot()
        if state['status'] != 'ready' or state['device'] != device:
            raise RuntimeError(f'{variant}: {sidecar.health()}')
        searcher = state['searcher']
        embedder = searcher.embedder
        if variant == 'mps-fp16':
            embedder.model.half()
            if next(embedder.model.parameters()).dtype.is_floating_point:
                assert str(next(embedder.model.parameters()).dtype) == 'torch.float16'
            KoreanEmbedder._probe(embedder.model, 'mps-fp16')

    thread_data = local()
    rank = searcher._rank

    def capture_rank(vector, k):
        hits = rank(vector, k)
        thread_data.raw = [(h.chunk_id, h.notice_num, h.score) for h in hits]
        return hits

    searcher._rank = capture_rank
    gate = Semaphore(args.gate)
    stock = embedder.embed_query
    if variant == 'mps-gated':

        def gated(query):
            started = perf_counter()
            with gate:
                thread_data.queue_ms = (perf_counter() - started) * 1000
                return stock(query)

        embedder.embed_query = gated
    if variant == 'mps-gated-detail':
        slots = embedder.query_slots
        if slots is None:
            raise RuntimeError('Expected MPS query admission to be enabled')

        class MeasuredSlot:
            def __enter__(self):
                started = perf_counter()
                slots.__enter__()
                thread_data.queue_ms = (perf_counter() - started) * 1000

            def __exit__(self, *exc):
                return slots.__exit__(*exc)

        embedder.query_slots = MeasuredSlot()
    if not variant.endswith('-detail'):
        embedder.embed_query_profiled = None

    k = min(args.k, sidecar.MAX_K)
    for query in ('보호', '개인정보 보호 정책', '공공 데이터 접근성과 정보 공개의 범위'):
        sidecar.search(query, k=k)

    def sample(row, barrier):
        thread_data.queue_ms = 0
        if barrier:
            barrier.wait(timeout=10)
        started = perf_counter()
        response = sidecar.search(row['query'], k=k)
        wall = (perf_counter() - started) * 1000
        timings = response.timings
        stages = timings.denseStages
        if not stages:
            raise RuntimeError('Missing dense timing')
        raw = thread_data.raw
        return {
            **row,
            'wallMs': wall,
            'embeddingMs': stages.embedding.durationMs,
            'queueMs': thread_data.queue_ms,
            'faissMs': stages.faiss.durationMs,
            'detail': stages.embeddingStages.model_dump() if stages.embeddingStages else None,
            'raw': raw,
            'clear': [(h.chunkId, h.noticeNum) for h in response.results],
            'weak': [(h.chunkId, h.noticeNum) for h in response.weakResults],
        }

    cells = {}
    for concurrency in args.concurrency:
        queries = cases()
        random.Random(args.seed).shuffle(queries)
        samples = []
        cell_started = perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for i in range(0, len(queries), concurrency):
                wave = queries[i : i + concurrency]
                barrier = Barrier(len(wave)) if len(wave) > 1 else None
                jobs = [pool.submit(sample, row, barrier) for row in wave]
                samples.extend(job.result() for job in jobs)
        cells[concurrency] = {
            'elapsedMs': (perf_counter() - cell_started) * 1000,
            'samples': samples,
        }
    return {
        'variant': variant,
        'seed': args.seed,
        'k': k,
        'health': sidecar.health(),
        'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS[:-1])
    parser.add_argument('--worker', choices=VARIANTS)
    parser.add_argument('--concurrency', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--onnx', type=Path)
    parser.add_argument('--onnx-threads', type=int, default=2)
    parser.add_argument('--gate', type=int, default=1)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--k', type=int, default=50)
    args = parser.parse_args()
    if min(args.gate, args.onnx_threads, args.k) < 1:
        parser.error('gate, onnx-threads and k must be positive')
    if (args.worker == 'cpu-onnx' or 'cpu-onnx' in args.variants) and not args.onnx:
        parser.error('cpu-onnx requires --onnx')
    if args.worker:
        print(json.dumps(worker(args), ensure_ascii=False))
        return
    before = hashes()
    runs = []
    failures = {}
    for variant in args.variants:
        print(
            f'{variant}: seed {args.seed}, concurrency {args.concurrency}',
            file=sys.stderr,
            flush=True,
        )
        proc = subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], '--worker', variant],
            cwd=ROOT,
            env={
                **os.environ,
                'LAWCAST_SEMANTIC_DEVICE': variant.split('-')[0],
                'LAWCAST_SEMANTIC_MPS_QUERY_CONCURRENCY': (
                    str(args.gate) if variant == 'mps-gated-detail' else '0'
                ),
                'HF_HUB_OFFLINE': '1',
                'OMP_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false',
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode:
            failures[variant] = {'exit': proc.returncode, 'stderr': proc.stderr[-6000:]}
            print(f'{variant}: exit {proc.returncode}', file=sys.stderr, flush=True)
            continue
        runs.append(json.loads(proc.stdout.splitlines()[-1]))
    if before != hashes():
        raise RuntimeError('Index artifacts changed during benchmark')
    print(
        json.dumps(
            {'artifactSha256': before, 'runs': runs, 'failures': failures}, ensure_ascii=False
        )
    )
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
