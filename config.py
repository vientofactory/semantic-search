"""Central configuration for the LawCast semantic-search side project.

All paths are resolved relative to this file so every script can be run
from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

# Input / output artifacts (each pipeline stage reads the previous stage's output).
SAMPLE_DATA_PATH = DATA_DIR / 'sample_notices.jsonl'
CHUNKS_PATH = ARTIFACTS_DIR / 'chunks.jsonl'
EMBEDDINGS_PATH = ARTIFACTS_DIR / 'embeddings.npz'
FAISS_INDEX_PATH = ARTIFACTS_DIR / 'faiss.index'
ID_MAP_PATH = ARTIFACTS_DIR / 'id_map.json'

# Korean-specialized sentence embedding model. jhgan/ko-sbert-sts is a Korean
# Sentence-BERT fine-tuned on KorSTS (Korean semantic textual similarity),
# 768-dim, native sentence-transformers support. See README for the full
# model-selection rationale. Override with LAWCAST_SEMANTIC_MODEL to swap.
MODEL_NAME = os.environ.get('LAWCAST_SEMANTIC_MODEL', 'jhgan/ko-sbert-sts')
DEVICE = os.environ.get('LAWCAST_SEMANTIC_DEVICE', 'cpu')
EMBED_BATCH_SIZE = int(os.environ.get('LAWCAST_SEMANTIC_BATCH', '32'))

# Chunking parameters (character based; stage 2 reports token-level fit).
# Calibrated for jhgan/ko-sbert-sts (max_seq_length=128 tokens): Korean legal
# prose tokenizes at roughly 1.8-1.9 chars/token, so 200 chars keeps chunks
# inside the model window with headroom (no silent truncation).
CHUNK_MAX_CHARS = 200
CHUNK_OVERLAP_CHARS = 50
CHUNK_MIN_CHARS = 40
