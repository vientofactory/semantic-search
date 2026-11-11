"""Prepare tokenizer and offline parity data for an existing FP32 BGE graph."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from lawcast_semantic.omp_env import use_single_threaded_omp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--revision', required=True, help='Exact BGE-m3-ko model commit')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('--revision must be an exact 40-character commit')
    if args.threads < 1:
        parser.error('--threads must be positive')
    use_single_threaded_omp()

    from sentence_transformers import SentenceTransformer

    from lawcast_semantic.query_onnx import (
        MODEL,
        PROBES,
        OnnxQueryEncoder,
        file_digest,
        manifest_path,
    )

    graph = args.graph.resolve()
    manifest = manifest_path(graph)
    tokenizer_dir = graph.parent / 'tokenizer'
    if manifest.exists() or tokenizer_dir.exists():
        parser.error('prepare a new bundle directory instead of replacing existing validation')
    weights = graph.with_name(f'{graph.name}_data')
    hashes = {path.name: file_digest(path) for path in (graph, weights)}
    model = SentenceTransformer(MODEL, revision=args.revision, device='cpu')
    probes = [
        model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        for text in PROBES
    ]
    validation = {
        'version': 1,
        'model': MODEL,
        'revision': args.revision,
        'dimension': int(model.get_embedding_dimension()),
        'maxSeqLength': int(model.max_seq_length),
        'sha256': hashes,
        'probeVectors': probes,
    }
    with TemporaryDirectory(dir=graph.parent) as scratch:
        staged = Path(scratch) / 'tokenizer'
        model.tokenizer.save_pretrained(staged)
        for path in staged.iterdir():
            hashes[f'tokenizer/{path.name}'] = file_digest(path)
        del model
        staged.rename(tokenizer_dir)
        try:
            encoder = OnnxQueryEncoder(graph, args.threads, validation=validation)
            pending = Path(scratch) / manifest.name
            pending.write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
            pending.rename(manifest)
        except BaseException:
            tokenizer_dir.rename(staged)
            raise
    print(f'Validated {manifest}; max absolute probe delta={encoder.probe_delta:.9g}')


if __name__ == '__main__':
    main()
