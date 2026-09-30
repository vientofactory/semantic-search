"""Configuration for the LawCast semantic-search library.

Single owner of pipeline settings and artifact locations. The library never
reads configuration from anywhere else, so importing `lawcast_semantic` is
sufficient for consumers (e.g. the LawCast backend integration) — no root-level
modules, no environment wiring beyond the LAWCAST_SEMANTIC_* variables below.

All paths are resolved relative to the package so every script and consumer
resolves the same locations from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# The side project root (parent of this package); data/ and artifacts/ live here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
# Override with LAWCAST_SEMANTIC_ARTIFACTS_DIR to serve artifacts from another
# location (e.g. a mounted volume in the Docker sidecar) without moving the
# code layout. Empty value falls back to the in-tree default.
ARTIFACTS_DIR = Path(
    os.environ.get('LAWCAST_SEMANTIC_ARTIFACTS_DIR') or PROJECT_ROOT / 'artifacts'
).expanduser()

# Input / output artifacts (each pipeline stage reads the previous stage's output).
SAMPLE_DATA_PATH = DATA_DIR / 'sample_notices.jsonl'
EVAL_QUERIES_PATH = DATA_DIR / 'eval_queries.jsonl'
CHUNKS_PATH = ARTIFACTS_DIR / 'chunks.jsonl'
EMBEDDINGS_PATH = ARTIFACTS_DIR / 'embeddings.npz'
FAISS_INDEX_PATH = ARTIFACTS_DIR / 'faiss.index'
ID_MAP_PATH = ARTIFACTS_DIR / 'id_map.json'

# Korean-specialized sentence embedding model. nlpai-lab/KURE-v1 is a Korean
# retrieval embedding model (bge-m3 base, 1024-dim, 8192-token window) that
# topped the Korean retrieval benchmarks in the A/B that adopted it — see
# README for the model-selection rationale and comparison numbers.
# Override with LAWCAST_SEMANTIC_MODEL to swap (e.g. jhgan/ko-sbert-sts).
MODEL_NAME = os.environ.get('LAWCAST_SEMANTIC_MODEL', 'nlpai-lab/KURE-v1')
DEVICE = os.environ.get('LAWCAST_SEMANTIC_DEVICE', 'cpu')
EMBED_BATCH_SIZE = int(os.environ.get('LAWCAST_SEMANTIC_BATCH', '32'))

# Scheduled refresh gates (agent_memories/08-semantic-search-production-deploy/
# incremental-update-pipeline-design.md §4.2). All three default to disabled:
# an empty DB_PATH turns off both the scheduler and the boot-repair hook, so
# host dev runs and tests behave exactly as before.
DB_PATH = os.environ.get('LAWCAST_SEMANTIC_DB_PATH', '')
UPDATE_INTERVAL_MINUTES = int(os.environ.get('LAWCAST_SEMANTIC_UPDATE_INTERVAL_MINUTES', '60'))
# Truthy spellings are defined exactly once, here (design §4.2): only '1',
# 'true', 'yes', 'on' (case/space-insensitive) enable the override; every
# other value fails safe with the shrink guard closed.
ALLOW_LARGE_DELETE = os.environ.get('LAWCAST_SEMANTIC_ALLOW_LARGE_DELETE', '').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}

# Chunking parameters (character based; stage 2 reports token-level fit).
# Calibrated for jhgan/ko-sbert-sts (max_seq_length=128 tokens): Korean legal
# prose tokenizes at roughly 1.8-1.9 chars/token, so 200 chars keeps chunks
# inside the model window with headroom (no silent truncation). CHUNK_MAX_CHARS
# bounds the full embedding input (subject context + body), so the body budget
# shrinks by the subject length.
CHUNK_MAX_CHARS = 200
CHUNK_OVERLAP_CHARS = 50
CHUNK_MIN_CHARS = 40
