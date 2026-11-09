"""Compare two local sidecar revisions through actual HTTP, not in-process calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import httpx
from benchmark_query_device import QUERIES, ROOT, hashes


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_until_ready(base, process, device):
    with httpx.Client(base_url=base, timeout=3, trust_env=False) as client:
        deadline = perf_counter() + 120
        while perf_counter() < deadline and process.poll() is None:
            try:
                health = client.get('/health').raise_for_status().json()
                if health['status'] == 'ready':
                    if health['device'] != device:
                        raise RuntimeError(f'Expected native {device}: {health}')
                    return health
                if health['status'] == 'failed':
                    raise RuntimeError(str(health))
            except httpx.TransportError:
                pass
            sleep(0.1)
    raise RuntimeError(f'Sidecar not ready (process status: {process.poll()})')


async def replay(base: str, rounds: int):
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with (
        httpx.AsyncClient(base_url=base, trust_env=False, limits=limits, timeout=35) as client,
        httpx.AsyncClient(base_url=base, trust_env=False, timeout=35) as probe,
    ):

        async def search(query):
            started = perf_counter()
            response = await client.get('/search', params={'query': query, 'k': 50})
            elapsed = (perf_counter() - started) * 1000
            body = response.raise_for_status().json()
            return {
                'query': query,
                'httpMs': elapsed,
                'timings': body['timings'],
                'ranks': [
                    [h['chunkId'] for h in body[tier]] for tier in ('results', 'weakResults')
                ],
            }

        for query in QUERIES * 2:
            await search(query)

        result = {}
        for concurrency in (1, 4, 48):
            search_rows, health_rows, walls = [], [], []
            for round_no in range(rounds):
                stop = asyncio.Event()

                async def probe_health():
                    while not stop.is_set():
                        started = perf_counter()
                        res = await probe.get('/health')
                        health_rows.append((perf_counter() - started) * 1000)
                        res.raise_for_status()
                        await asyncio.sleep(0.01)

                monitor = asyncio.create_task(probe_health())
                queries = [
                    QUERIES[(i + round_no) % len(QUERIES)] for i in range(max(12, concurrency))
                ]
                started = perf_counter()
                try:
                    for i in range(0, len(queries), concurrency):
                        search_rows.extend(
                            await asyncio.gather(*(search(q) for q in queries[i : i + concurrency]))
                        )
                    walls.append((perf_counter() - started) * 1000)
                finally:
                    stop.set()
                    await monitor
            result[concurrency] = {
                'search': search_rows,
                'healthMs': health_rows,
                'waveMs': walls,
            }
        return result


def trial(root: Path, rounds: int, *, device, embed_workers, rank_workers, mps_concurrency):
    port = free_port()
    base = f'http://127.0.0.1:{port}'
    with TemporaryDirectory() as folder, (Path(folder) / 'uvicorn.log').open('w+') as log:
        process = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'uvicorn',
                'service.app:app',
                '--host',
                '127.0.0.1',
                '--port',
                str(port),
                '--log-level',
                'warning',
                '--no-access-log',
            ],
            cwd=root,
            env={
                **os.environ,
                'LAWCAST_SEMANTIC_ENV_FILE': str(ROOT / '.env'),
                'LAWCAST_SEMANTIC_ARTIFACTS_DIR': str(ROOT / 'artifacts'),
                'LAWCAST_SEMANTIC_DEVICE': device,
                'LAWCAST_SEMANTIC_MPS_QUERY_CONCURRENCY': str(mps_concurrency),
                'LAWCAST_SEMANTIC_QUERY_EMBED_WORKERS': str(embed_workers),
                'LAWCAST_SEMANTIC_QUERY_RANK_WORKERS': str(rank_workers),
                'LAWCAST_SEMANTIC_DB_PATH': '',
                'LAWCAST_SEMANTIC_UPDATE_CRON': '',
                'OMP_NUM_THREADS': '1',
                'HF_HUB_OFFLINE': '1',
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            health = wait_until_ready(base, process, device)
            return {'health': health, 'cells': asyncio.run(replay(base, rounds))}
        except Exception:
            log.seek(0)
            print(log.read()[-6000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--sessions', type=int, default=2)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    p.add_argument('--embed-workers', type=int, default=4)
    p.add_argument('--rank-workers', type=int, default=2)
    p.add_argument('--mps-concurrency', type=int, default=1)
    args = p.parse_args()
    if min(args.embed_workers, args.rank_workers) < 1 or args.mps_concurrency < 0:
        p.error('worker limits must be positive and MPS concurrency nonnegative')
    if min(args.rounds, args.sessions) < 1:
        p.error('sessions and rounds must be positive')
    before = hashes()
    runs = []
    for session in range(args.sessions):
        order = ('baseline', 'modified') if session % 2 == 0 else ('modified', 'baseline')
        for arm in order:
            print(f'Session {session + 1}: {arm}', file=sys.stderr, flush=True)
            rank_workers = args.rank_workers if arm == 'modified' else 0
            run = trial(
                args.baseline if arm == 'baseline' else ROOT,
                args.rounds,
                device=args.device,
                embed_workers=args.embed_workers,
                rank_workers=rank_workers,
                mps_concurrency=args.mps_concurrency,
            )
            runs.append({'session': session + 1, 'arm': arm, **run})
    if hashes() != before:
        raise RuntimeError('Index files changed during the experiment')
    print(json.dumps({'artifactHashes': before, 'runs': runs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
