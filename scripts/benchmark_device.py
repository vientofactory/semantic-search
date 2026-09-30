"""One-off benchmark: measure embedding throughput on cpu vs mps.

Usage: python scripts/benchmark_device.py [num_texts]

Extrapolates full-corpus embedding time from measured throughput.
Not part of the production pipeline; kept for reproducibility of the
device selection decision documented in the agent memories.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run-from-source bootstrap
from lawcast_semantic.embedding import KoreanEmbedder  # noqa: E402


def bench(device: str, texts: list[str], batch_size: int) -> float:
    client = KoreanEmbedder(device=device)
    start = time.perf_counter()
    client.embed_texts(texts, batch_size=batch_size)
    return time.perf_counter() - start


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    texts = [
        f'입법예고 안건 {i}: 행정기관이 추진하는 제도 개선에 대한 의견을 수렴합니다. '
        '관계 기관 협의를 거쳐 시행령 일부를 개정하고자 합니다.'
        for i in range(n)
    ]
    batch_size = int(os.environ.get('LAWCAST_SEMANTIC_BATCH', '32'))
    total_chunks = 96754

    for device in ('cpu', 'mps'):
        try:
            elapsed = bench(device, texts, batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f'{device}: FAILED ({exc})')
            continue
        rate = n / elapsed
        projected = total_chunks / rate
        print(
            f'{device}: {n} texts in {elapsed:.2f}s -> {rate:.2f} texts/s '
            f'-> projected {total_chunks} chunks in {projected / 60:.1f} min'
        )


if __name__ == '__main__':
    main()
