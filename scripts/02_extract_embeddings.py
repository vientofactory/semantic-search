"""Stage 2: tokenize chunks and extract embeddings.

Loads chunks JSONL, reports tokenizer statistics (token counts vs. the
model's max sequence length), embeds every chunk with the Korean
sentence-transformers model, and saves the embedding matrix.

Usage:
    python scripts/02_extract_embeddings.py
        [--chunks artifacts/chunks.jsonl] [--out artifacts/embeddings.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lawcast_semantic import KoreanEmbedder  # noqa: E402
from lawcast_semantic.chunking import compute_chunks_fingerprint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', type=Path, default=config.CHUNKS_PATH)
    parser.add_argument('--out', type=Path, default=config.EMBEDDINGS_PATH)
    parser.add_argument('--model', default=config.MODEL_NAME)
    args = parser.parse_args()

    records = []
    with args.chunks.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    chunk_ids = [record['chunk_id'] for record in records]
    texts = [record['text'] for record in records]

    print(f'model              : {args.model} (first use downloads it from HuggingFace)')
    embedder = KoreanEmbedder(args.model, device=config.DEVICE)
    print(f'max_seq_length     : {embedder.max_seq_length}')
    print(f'embedding dim      : {embedder.dimension}')

    report = embedder.tokenize_report(texts)
    print(f'tokenizer report   : {report}')

    if texts:
        sample_tokens = embedder.tokenize(texts[0])
        print(f'token preview      : {sample_tokens["preview_tokens"]}')
    else:
        print('token preview      : n/a (no chunks)')

    embeddings = embedder.embed_texts(texts, batch_size=config.EMBED_BATCH_SIZE)
    fingerprint = compute_chunks_fingerprint(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        chunk_ids=np.asarray(chunk_ids),
        chunks_fingerprint=np.asarray(fingerprint),
        model_name=np.asarray(args.model),
    )
    print(f'embedding matrix   : {embeddings.shape} float32')
    if len(embeddings):
        norms = np.linalg.norm(embeddings, axis=1)
        print(f'L2 norms           : mean={norms.mean():.4f} min={norms.min():.4f}')
    print(f'chunks fingerprint : {fingerprint[:12]}')
    print(f'written to         : {args.out}')


if __name__ == '__main__':
    main()
