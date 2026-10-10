"""Configuration for the LawCast semantic-search library.

Single owner of pipeline settings and artifact locations. The library reads
configuration from the process environment (LAWCAST_SEMANTIC_* below) and an
optional `.env` file in the project root, so importing `lawcast_semantic` is
sufficient for consumers (e.g. the LawCast backend integration) — no
root-level modules, no environment wiring beyond those variables.

All paths are resolved relative to the package so every script and consumer
resolves the same locations from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# The side project root (parent of this package); data/ and artifacts/ live here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Optional `.env` file (see .env.example): loaded once at import, BEFORE any
# env read below, so every consumer — scripts, the FastAPI sidecar, the Docker
# image — shares the same file-based defaults. Precedence is standard dotenv:
# the process environment always wins (compose `environment:` and shell
# exports override file values), applied via setdefault. Path override:
# LAWCAST_SEMANTIC_ENV_FILE; an empty value disables file loading entirely
# (same empty=off convention as DB_PATH/UPDATE_CRON). A missing file is the
# normal case (CI clone, host without .env) and fails safe silently.
ENV_FILE_ENV = 'LAWCAST_SEMANTIC_ENV_FILE'


def _parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines: blanks/# comments, optional `export`, quotes."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching quotes; unquoted values keep inner `#`
        # (cron expressions and paths may legally contain it).
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env_file() -> None:
    """Load the optional .env file into os.environ without overriding it."""
    override = os.environ.get(ENV_FILE_ENV)
    if override is not None and not override.strip():
        return  # empty LAWCAST_SEMANTIC_ENV_FILE disables file loading
    path = Path(override).expanduser() if override else PROJECT_ROOT / '.env'
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return  # absent file is the default host/CI case
    for key, value in _parse_env_file(text).items():
        os.environ.setdefault(key, value)


load_env_file()

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
# Cron expression driving the scheduled index update (standard 5-field:
# minute hour day-of-month month day-of-week, evaluated in the process's local
# time — compose sets TZ=Asia/Seoul). Default keeps design §4.2's hourly
# cadence; an empty value disables scheduling entirely (replaces the retired
# LAWCAST_SEMANTIC_UPDATE_INTERVAL_MINUTES, where `0` disabled). The expression
# is parsed by the scheduler (service/update_runner.py), not here, so this
# module stays an env-only, import-light config owner.
UPDATE_CRON = os.environ.get('LAWCAST_SEMANTIC_UPDATE_CRON', '0 * * * *').strip()
# Truthy spellings are defined exactly once, here (design §4.2): only '1',
# 'true', 'yes', 'on' (case/space-insensitive) enable the override; every
# other value fails safe with the shrink guard closed.
ALLOW_LARGE_DELETE = os.environ.get('LAWCAST_SEMANTIC_ALLOW_LARGE_DELETE', '').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}

# Relevance tiers for query results (stage 4 cosine similarity, range
# [-1, 1]). MIN_SIMILARITY is the relatedness floor: hits scoring below it
# are unrelated to the query and dropped entirely. CLEAR_SIMILARITY separates
# the clear results (shown by default) from the weak band between the two
# thresholds, which callers serve separately so the UI can hide them behind
# an explicit reveal. MIN_SIMILARITY must not exceed CLEAR_SIMILARITY.
MIN_SIMILARITY = float(os.environ.get('LAWCAST_SEMANTIC_MIN_SIMILARITY', '0.25'))
CLEAR_SIMILARITY = float(os.environ.get('LAWCAST_SEMANTIC_CLEAR_SIMILARITY', '0.45'))
if MIN_SIMILARITY > CLEAR_SIMILARITY:
    raise ValueError(
        'LAWCAST_SEMANTIC_MIN_SIMILARITY must be <= LAWCAST_SEMANTIC_CLEAR_SIMILARITY '
        f'(got {MIN_SIMILARITY} > {CLEAR_SIMILARITY})'
    )

# Chunking parameters (character based; stage 2 reports token-level fit).
# Calibrated for jhgan/ko-sbert-sts (max_seq_length=128 tokens): Korean legal
# prose tokenizes at roughly 1.8-1.9 chars/token, so 200 chars keeps chunks
# inside the model window with headroom (no silent truncation). CHUNK_MAX_CHARS
# bounds the full embedding input (subject context + body), so the body budget
# shrinks by the subject length. CHUNK_MIN_CHARS is the size the packer packs
# AGAINST, never a filter: a packed chunk below it is a leftover that did not
# fit beside its neighbour, and it is merged into the chunk in front of it
# rather than dropped (dropping it made that text unsearchable — see
# agent_memories/22-chunk-floor-content-loss/). Only that merge may exceed the
# budget, by at most CHUNK_MIN_CHARS characters.
CHUNK_MAX_CHARS = 200
CHUNK_OVERLAP_CHARS = 50
CHUNK_MIN_CHARS = 40
