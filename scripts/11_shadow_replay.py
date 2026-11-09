"""Compare teacher and BGE20 on explicit replay queries inside a one-shot container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

SHADOW_ENABLED_ENV = 'LAWCAST_SEMANTIC_DISTILLATION_SHADOW_ENABLED'
WEIGHTS = Path(__file__).with_name('assets') / 'bge20-projection.pt'
WEIGHT_SHA256 = 'dfb3ef6afa71450814fc5ccab38997d20899618bb0e63a832eb8532e248a24c7'


def emit(event, **data):
    print(json.dumps({'event': event, **data}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', required=True, type=Path)
    parser.add_argument('--k', default=50, type=int)
    parser.add_argument('--repeat', default=2, type=int)
    args = parser.parse_args()
    # Check the process environment before config can read a dotenv file.
    if os.environ.get(SHADOW_ENABLED_ENV) != '1':
        parser.error(f'distillation shadow replay is disabled; set {SHADOW_ENABLED_ENV}=1')

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from lawcast_semantic.omp_env import use_single_threaded_omp

    use_single_threaded_omp()

    from lawcast_semantic import config
    from lawcast_semantic.aliases import expand_query
    from lawcast_semantic.preprocess import normalize_text
    from lawcast_semantic.shadow import (
        REVISION,
        TEACHER,
        layer_positions,
        measure,
        validate_checkpoint,
    )

    rows = json.loads(args.queries.read_text(encoding='utf-8'))
    if not (1 <= len(rows) <= 50 and 1 <= args.repeat <= 5 and 1 <= args.k <= config.MAX_K):
        parser.error('use 1–50 fixed probes, 1–5 repetitions and a valid serving window')
    if any(not isinstance(row['query'], str) or not row['query'].strip() for row in rows):
        parser.error('each probe requires a non-empty query')
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHT_SHA256:
        raise ValueError('shadow projection checksum mismatch')

    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(WEIGHTS, weights_only=True, map_location='cpu')
    validate_checkpoint(checkpoint['meta'])

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.residual = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1024))

        def forward(self, x):
            return F.normalize(self.linear(x) + self.residual(x), dim=-1)

    model_path = snapshot_download(TEACHER, revision=REVISION, local_files_only=True)
    model = SentenceTransformer(model_path, device='cpu', local_files_only=True)
    trunk = model[0].auto_model
    full_layers = trunk.encoder.layer
    student_layers = nn.ModuleList([full_layers[i] for i in layer_positions(len(full_layers))])
    head = Projection()
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    head.eval()

    from lawcast_semantic.search import SearchResult, SemanticSearcher
    from service.app import SearchResponse, _hit

    class TeacherMetadata:
        model_name = TEACHER
        dimension = 1024

    searcher = SemanticSearcher.load(TeacherMetadata())

    @torch.inference_mode()
    def encode(query, student):
        trunk.encoder.layer = student_layers if student else full_layers
        try:
            vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            return head(torch.from_numpy(vector))[0].numpy() if student else vector[0]
        finally:
            trunk.encoder.layer = full_layers

    def rank(vector):
        pairs = searcher.index.search(vector, k=args.k)
        pairs.sort(key=lambda pair: (-pair[1], searcher.chunk_ids[pair[0]]))
        clear, weak = [], []
        for pos, score in pairs:
            if score < config.MIN_SIMILARITY:
                continue
            chunk_id = searcher.chunk_ids[pos]
            row = searcher.chunks_by_id[chunk_id]
            hit = SearchResult(
                chunk_id=chunk_id,
                score=score,
                notice_num=row['notice_num'],
                subject=row['subject'],
                committee=row['committee'],
                section=row['section'],
                text=row['text'],
            )
            (clear if score >= config.CLEAR_SIMILARITY else weak).append(hit)
        return clear, weak

    def run(row, student):
        raw = row['query']
        normalized = normalize_text(expand_query(raw).text)

        def serialize(hits):
            response = SearchResponse(
                query=raw,
                k=args.k,
                model=TEACHER,
                lastUpdateAt=searcher.index_updated_at,
                results=[_hit(hit, raw, searcher.chunks_by_id) for hit in hits[0]],
                weakResults=[_hit(hit, raw, searcher.chunks_by_id) for hit in hits[1]],
            )
            response.model_dump_json()
            return response

        return measure(
            lambda: encode(normalized, student),
            rank,
            serialize,
        )

    emit(
        'shadow-start',
        teacher=TEACHER,
        revision=REVISION,
        index_type=searcher.index.index_type,
        indexed_chunks=len(searcher.chunk_ids),
        index_updated_at=searcher.index_updated_at,
        query_count=len(rows),
        repeat=args.repeat,
        k=args.k,
    )
    for student in (False, True):
        run(rows[0], student)
    for repeat in range(args.repeat):
        for ordinal, row in enumerate(rows):
            result = {}
            for student in (False, True) if (repeat + ordinal) % 2 == 0 else (True, False):
                timing, payload = run(row, student)
                result['student' if student else 'teacher'] = {
                    **{k: round(v, 3) for k, v in asdict(timing).items()},
                    'notice_top5': list(dict.fromkeys(x.noticeNum for x in payload.results))[:5],
                }
            emit('shadow-pair', repeat=repeat + 1, probe=ordinal + 1, **result)
    emit('shadow-finish', pairs=args.repeat * len(rows))


if __name__ == '__main__':
    main()
