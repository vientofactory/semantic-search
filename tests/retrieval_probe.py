"""Out-of-process retrieval probe for the chunk-coverage guard.

Run as a subprocess by `tests/test_chunk_coverage_retrieval.py`. It builds the
stage 1/3 artifacts for the task's notices with a deterministic hashing embedder,
loads them through the real `SemanticSearcher`, and prints one `RESULT {json}`
line per probe.

Why its own process: on this host a faiss search performed before torch is
imported into the same interpreter aborts the process (`OMP: Error #15`, the two
libraries statically link their own libomp). The pytest process collects the
guard before `tests/test_entrypoints.py` imports the embedding stack, so every
faiss call lives here instead. Stub mode never imports torch at all, and the
faiss-only process cannot hit the conflict.

Usage:
  python tests/retrieval_probe.py --artifacts-dir DIR --task task.json [--k 5]

Task JSON:
  {
    "notices": [<notice record>...],             # indexed with the stub embedder
    "probes": [{"label": str, "query": str, "notice_num": int}]
  }

Output: one line per probe, `RESULT ` followed by `{"label", "notice_num",
"notice_order", "hit_count", "hits"}`. `hits` is the ranked window (bounded by
`--k`, the window the sidecar's `/search` serves by default) carrying each hit's
notice, score and full text; `notice_order` is the notice-deduplicated ranking of
that window — the same order `scripts/05_evaluate.py` reads its rank metrics off.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

# The parent runs this file as a script, so the project root has to be on
# sys.path before `lawcast_semantic` can be imported (conftest does the same for
# the pytest process).
# isort: off
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lawcast_semantic.chunking import (  # noqa: E402 - after the sys.path bootstrap
    chunk_notice,
    compose_embedding_text,
    compute_chunks_fingerprint,
    write_chunks_jsonl,
)
from lawcast_semantic.indexing import VectorIndex  # noqa: E402
from lawcast_semantic.search import SemanticSearcher  # noqa: E402

# isort: on

_WHITESPACE_RE = re.compile(r'\s+')


class HashingEmbedder:
    """Deterministic bag-of-character-3-grams embedder: no model, no download.

    A fixture query is a substring of the chunk text it must reach, so n-gram
    overlap puts the containing chunk on top while unrelated notices stay close
    to orthogonal. `zlib.crc32` keeps the buckets stable across processes
    (Python's `hash()` is salted per run), so a failure here is reproducible.
    """

    model_name = 'hashing-ngram-stub'
    dimension = 512

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype='float32')
        squashed = _WHITESPACE_RE.sub('', text)
        for index in range(max(len(squashed) - 2, 0)):
            bucket = zlib.crc32(squashed[index : index + 3].encode('utf-8')) % self.dimension
            vector[bucket] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def embed_texts(self, texts) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])


def build_artifacts(artifacts_dir: Path, notices: list[dict], embedder) -> list[dict]:
    """Write chunks.jsonl + faiss.index + id_map.json for these notices."""
    records = [chunk.to_dict() for notice in notices for chunk in chunk_notice(notice)]
    index = VectorIndex(embedder.dimension)
    index.build(
        embedder.embed_texts(
            [compose_embedding_text(record['subject'], record['text']) for record in records]
        )
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    index.save(
        artifacts_dir / 'faiss.index',
        artifacts_dir / 'id_map.json',
        [record['chunk_id'] for record in records],
        meta={
            'chunks_fingerprint': compute_chunks_fingerprint(records),
            'model_name': embedder.model_name,
        },
    )
    write_chunks_jsonl(records, artifacts_dir / 'chunks.jsonl')
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description='Answer retrieval probes in a faiss-only process.')
    parser.add_argument('--artifacts-dir', required=True, help='where stub artifacts are written')
    parser.add_argument('--task', required=True, help='task JSON (notices + probes)')
    parser.add_argument('--k', type=int, default=5, help='ranked window size per probe')
    arguments = parser.parse_args()

    artifacts_dir = Path(arguments.artifacts_dir)
    task = json.loads(Path(arguments.task).read_text(encoding='utf-8'))
    embedder = HashingEmbedder()
    notices = task.get('notices') or []
    assert notices, 'the task carries no notices to index'
    records = build_artifacts(artifacts_dir, notices, embedder)

    searcher = SemanticSearcher.load(
        embedder,
        chunks_path=artifacts_dir / 'chunks.jsonl',
        index_path=artifacts_dir / 'faiss.index',
        id_map_path=artifacts_dir / 'id_map.json',
    )
    print(f'INDEXED {len(records)} chunks from {len(notices)} notices', flush=True)

    for probe in task['probes']:
        results = searcher.search(probe['query'], k=arguments.k)
        order: list[int] = []
        for result in results:
            notice_num = int(result.notice_num)
            if notice_num not in order:
                order.append(notice_num)
        print(
            'RESULT '
            + json.dumps(
                {
                    'label': probe['label'],
                    'notice_num': int(probe['notice_num']),
                    'notice_order': order,
                    'hit_count': len(results),
                    'hits': [
                        {
                            'notice_num': int(result.notice_num),
                            'score': round(float(result.score), 6),
                            'text': result.text,
                        }
                        for result in results
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()
