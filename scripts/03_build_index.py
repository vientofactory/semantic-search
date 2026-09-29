"""Stage 3: build and persist the FAISS index.

Loads the embedding matrix from stage 2, builds a cosine-similarity FAISS
index (IndexFlatIP over L2-normalized vectors), and saves it together with
the row -> chunk_id mapping.

Usage:
    python scripts/03_build_index.py [--embeddings artifacts/embeddings.npz] [--out artifacts/faiss.index]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lawcast_semantic import VectorIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, default=config.EMBEDDINGS_PATH)
    parser.add_argument('--out', type=Path, default=config.FAISS_INDEX_PATH)
    parser.add_argument('--id-map', type=Path, default=config.ID_MAP_PATH)
    args = parser.parse_args()

    payload = np.load(args.embeddings, allow_pickle=False)
    embeddings = payload['embeddings']
    chunk_ids = [str(chunk_id) for chunk_id in payload['chunk_ids']]
    # Provenance metadata from stage 2 travels into the id map so stage 4 can
    # reject stale or model-mismatched artifacts.
    meta = {
        key: payload[key].item()
        for key in ('chunks_fingerprint', 'model_name')
        if key in payload
    }

    index = VectorIndex(embeddings.shape[1])
    index.build(embeddings)
    index.save(args.out, args.id_map, chunk_ids, meta=meta)

    print(f'vectors indexed     : {index.index.ntotal}')
    print(f'index type          : IndexFlatIP (cosine over normalized vectors)')
    print(f'index dim           : {index.dimension}')
    print(f'index file          : {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)')
    print(f'id map              : {args.id_map} ({len(chunk_ids)} entries)')


if __name__ == '__main__':
    main()
