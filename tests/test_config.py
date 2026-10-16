"""Config contract tests: artifact locations, env overrides, and the
scheduled-refresh gates (design §4.2).

Run in subprocess isolation (same style as test_entrypoints) so module-level
env reads in `lawcast_semantic.config` are exercised exactly as a fresh
process (script, sidecar, container) would see them.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERRIDE_ENV = 'LAWCAST_SEMANTIC_ARTIFACTS_DIR'
ENV_FILE_ENV = 'LAWCAST_SEMANTIC_ENV_FILE'
# Env vars the tests must control explicitly so defaults stay deterministic
# (a developer's shell exports and any project-root .env must not leak in).
STRIPPED_ENV = {
    OVERRIDE_ENV,
    'LAWCAST_SEMANTIC_DB_PATH',
    'LAWCAST_SEMANTIC_UPDATE_CRON',
    'LAWCAST_SEMANTIC_ALLOW_LARGE_DELETE',
    'LAWCAST_SEMANTIC_DEVICE',
    'LAWCAST_SEMANTIC_BATCH',
    'LAWCAST_SEMANTIC_MODEL',
    'LAWCAST_SEMANTIC_MIN_SIMILARITY',
    'LAWCAST_SEMANTIC_CLEAR_SIMILARITY',
    'LAWCAST_SEMANTIC_MAX_QUERY_LENGTH',
    'LAWCAST_SEMANTIC_MAX_K',
    'LAWCAST_SEMANTIC_DEFAULT_K',
    'LAWCAST_SEMANTIC_RERANK_MODEL',
    'LAWCAST_SEMANTIC_RERANK_MODE',
    'LAWCAST_SEMANTIC_RERANK_TOP_K',
    'LAWCAST_SEMANTIC_RERANK_DEVICE',
    'LAWCAST_SEMANTIC_RERANK_MAX_LENGTH',
}


def run_config(*names: str, env_overrides: dict | None = None) -> list[str]:
    """Print the requested config attributes from a fresh interpreter."""
    code = 'import lawcast_semantic.config as c\n' + ''.join(f'print(c.{name})\n' for name in names)
    # Strip any inherited override so the default case is deterministic.
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    # Disable .env loading unless a test opts in, so a developer's
    # project-root .env can never shift defaults in the subprocess.
    env[ENV_FILE_ENV] = ''
    env.update(env_overrides or {})
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_artifacts_default_to_project_root():
    """Without the env var, artifacts resolve under PROJECT_ROOT/artifacts."""
    lines = run_config(
        'ARTIFACTS_DIR', 'CHUNKS_PATH', 'EMBEDDINGS_PATH', 'FAISS_INDEX_PATH', 'ID_MAP_PATH'
    )
    artifacts = PROJECT_ROOT / 'artifacts'
    assert lines == [
        str(artifacts),
        str(artifacts / 'chunks.jsonl'),
        str(artifacts / 'embeddings.npz'),
        str(artifacts / 'faiss.index'),
        str(artifacts / 'id_map.json'),
    ]


def test_artifacts_dir_env_override(tmp_path: Path):
    """LAWCAST_SEMANTIC_ARTIFACTS_DIR relocates every artifact path (e.g. a mounted volume)."""
    mounted = tmp_path / 'mounted'
    lines = run_config(
        'ARTIFACTS_DIR',
        'CHUNKS_PATH',
        'FAISS_INDEX_PATH',
        env_overrides={OVERRIDE_ENV: str(mounted)},
    )
    assert lines == [
        str(mounted),
        str(mounted / 'chunks.jsonl'),
        str(mounted / 'faiss.index'),
    ]


def test_artifacts_dir_empty_env_falls_back_to_default():
    """An empty override is ignored rather than resolving to the cwd."""
    lines = run_config('ARTIFACTS_DIR', env_overrides={OVERRIDE_ENV: ''})
    assert lines == [str(PROJECT_ROOT / 'artifacts')]


def test_refresh_gates_default_to_disabled():
    """Design §4.2: empty DB_PATH (the off gate), hourly cron, delete guard off."""
    lines = run_config('DB_PATH', 'UPDATE_CRON', 'ALLOW_LARGE_DELETE')
    assert lines == ['', '0 * * * *', 'False']


def test_similarity_tiers_default_to_floor_and_clear_threshold():
    """Relatedness floor (below = dropped) and clear-results threshold
    (between the two = weak band, hidden behind a reveal in the UI)."""
    lines = run_config('MIN_SIMILARITY', 'CLEAR_SIMILARITY')
    assert lines == ['0.25', '0.45']


def test_similarity_tier_env_overrides():
    lines = run_config(
        'MIN_SIMILARITY',
        'CLEAR_SIMILARITY',
        env_overrides={
            'LAWCAST_SEMANTIC_MIN_SIMILARITY': '0.3',
            'LAWCAST_SEMANTIC_CLEAR_SIMILARITY': '0.6',
        },
    )
    assert lines == ['0.3', '0.6']


def run_config_failure(**env_overrides: str) -> str:
    """Import the config with invalid settings; it must refuse instead of
    serving a silently wrong surface. Returns stderr for message assertions."""
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    env[ENV_FILE_ENV] = ''
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, '-c', 'import lawcast_semantic.config'],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0, 'config imported despite invalid settings'
    return result.stderr


def test_rerank_shadow_defaults_to_off_with_bounded_cpu_window():
    """Shadow scoring is opt-in; the default model/window/device are the
    calibrated pilot values (nano, top-20, cpu for prod parity)."""
    lines = run_config(
        'RERANK_MODEL', 'RERANK_MODE', 'RERANK_TOP_K', 'RERANK_DEVICE', 'RERANK_MAX_LENGTH'
    )
    assert lines == ['nlpai-lab/KURE-Reranker-nano', 'off', '20', 'cpu', '512']


def test_rerank_invalid_mode_fails_fast():
    stderr = run_config_failure(LAWCAST_SEMANTIC_RERANK_MODE='sometimes')
    assert 'RERANK_MODE' in stderr


def test_rerank_max_length_override_and_floor():
    """The truncation window is env-tunable; zero refuses to import."""
    lines = run_config(
        'RERANK_MAX_LENGTH', env_overrides={'LAWCAST_SEMANTIC_RERANK_MAX_LENGTH': '256'}
    )
    assert lines == ['256']
    stderr = run_config_failure(LAWCAST_SEMANTIC_RERANK_MAX_LENGTH='0')
    assert 'RERANK_MAX_LENGTH' in stderr


def test_min_similarity_above_clear_fails_fast():
    """A floor above the clear threshold would silently invert the tiers,
    so the config refuses to import instead."""
    stderr = run_config_failure(
        LAWCAST_SEMANTIC_MIN_SIMILARITY='0.6',
        LAWCAST_SEMANTIC_CLEAR_SIMILARITY='0.3',
    )
    assert 'MIN_SIMILARITY must be <=' in stderr


def test_sidecar_request_bounds_default_to_the_wire_contract_values():
    """The sidecar's `/search` bounds live here (env-injectable); the code
    defaults are the pinned wire contract: query <= 500 chars, chunk window
    1..100 defaulting to 5 (MAX_K's default is pinned against the backend's
    SIDE_CAR_MAX_CHUNK_K by the cross-language contract spec)."""
    lines = run_config('MAX_QUERY_LENGTH', 'MAX_K', 'DEFAULT_K')
    assert lines == ['500', '100', '5']


def test_sidecar_request_bounds_env_overrides():
    """Operators can tune the API surface without a code change."""
    lines = run_config(
        'MAX_QUERY_LENGTH',
        'MAX_K',
        'DEFAULT_K',
        env_overrides={
            'LAWCAST_SEMANTIC_MAX_QUERY_LENGTH': '120',
            'LAWCAST_SEMANTIC_MAX_K': '33',
            'LAWCAST_SEMANTIC_DEFAULT_K': '10',
        },
    )
    assert lines == ['120', '33', '10']


def test_default_k_above_max_k_fails_fast():
    """A default window wider than the cap would make `/search` reject its
    own default, so the config refuses to import instead."""
    stderr = run_config_failure(LAWCAST_SEMANTIC_MAX_K='3', LAWCAST_SEMANTIC_DEFAULT_K='5')
    assert 'LAWCAST_SEMANTIC_DEFAULT_K must be between 1 and LAWCAST_SEMANTIC_MAX_K' in stderr


def test_non_positive_request_bounds_fail_fast():
    """Zero/negative caps would reject every possible request."""
    stderr = run_config_failure(LAWCAST_SEMANTIC_MAX_K='0')
    assert 'LAWCAST_SEMANTIC_MAX_K must be >= 1' in stderr
    stderr = run_config_failure(LAWCAST_SEMANTIC_MAX_QUERY_LENGTH='0')
    assert 'LAWCAST_SEMANTIC_MAX_QUERY_LENGTH must be >= 1' in stderr
    stderr = run_config_failure(LAWCAST_SEMANTIC_DEFAULT_K='0')
    assert 'LAWCAST_SEMANTIC_DEFAULT_K must be between 1 and LAWCAST_SEMANTIC_MAX_K' in stderr


def test_non_integer_request_bound_fails_fast():
    """A typo in an int env var must not collapse into a confusing failure
    inside FastAPI — the offending variable is named at import time."""
    stderr = run_config_failure(LAWCAST_SEMANTIC_MAX_K='many')
    assert 'LAWCAST_SEMANTIC_MAX_K' in stderr


def test_refresh_db_path_and_cron_env_overrides():
    """compose sets DB_PATH; operators can tune the schedule via cron expression."""
    lines = run_config(
        'DB_PATH',
        'UPDATE_CRON',
        env_overrides={
            'LAWCAST_SEMANTIC_DB_PATH': '/data/lawcast.db',
            'LAWCAST_SEMANTIC_UPDATE_CRON': '17 * * * *',
        },
    )
    assert lines == ['/data/lawcast.db', '17 * * * *']


def test_update_cron_empty_keeps_scheduling_disabled():
    """Design §4.2: empty expression also disables (falsy for the lifespan gate)."""
    lines = run_config(
        'UPDATE_CRON',
        env_overrides={'LAWCAST_SEMANTIC_UPDATE_CRON': ''},
    )
    assert lines == ['']


def test_update_cron_is_stripped():
    """Whitespace around the expression is cosmetic, not a validation failure."""
    lines = run_config(
        'UPDATE_CRON',
        env_overrides={'LAWCAST_SEMANTIC_UPDATE_CRON': '  */10 * * * * '},
    )
    assert lines == ['*/10 * * * *']


def test_allow_large_delete_true_spellings():
    """§4.2's exact accepted set — case- and space-insensitive."""
    for value in ('1', 'true', 'True', ' TRUE ', 'yes', 'on'):
        (line,) = run_config(
            'ALLOW_LARGE_DELETE',
            env_overrides={'LAWCAST_SEMANTIC_ALLOW_LARGE_DELETE': value},
        )
        assert line == 'True', value


def test_allow_large_delete_fails_safe_on_anything_else():
    """Typos and every unknown value keep the shrink guard closed."""
    for value in ('', '0', 'false', 'no', 'ye', 'on?', 'nonsense'):
        (line,) = run_config(
            'ALLOW_LARGE_DELETE',
            env_overrides={'LAWCAST_SEMANTIC_ALLOW_LARGE_DELETE': value},
        )
        assert line == 'False', value


def test_env_file_supplies_values_when_process_env_is_silent(tmp_path: Path):
    """A .env file is read at import: comments, export, quotes, cron spaces."""
    env_file = tmp_path / '.env'
    env_file.write_text(
        '# operator overrides\n'
        'export LAWCAST_SEMANTIC_DEVICE=mps\n'
        'LAWCAST_SEMANTIC_BATCH=8\n'
        "LAWCAST_SEMANTIC_UPDATE_CRON='17 * * * *'\n"
        'LAWCAST_SEMANTIC_MODEL="jhgan/ko-sbert-sts"\n',
        encoding='utf-8',
    )
    lines = run_config(
        'DEVICE',
        'EMBED_BATCH_SIZE',
        'UPDATE_CRON',
        'MODEL_NAME',
        env_overrides={ENV_FILE_ENV: str(env_file)},
    )
    assert lines == ['mps', '8', '17 * * * *', 'jhgan/ko-sbert-sts']


def test_env_file_never_overrides_the_process_environment(tmp_path: Path):
    """Precedence: compose `environment:` / shell exports beat the file."""
    env_file = tmp_path / '.env'
    env_file.write_text('LAWCAST_SEMANTIC_DEVICE=mps\n', encoding='utf-8')
    lines = run_config(
        'DEVICE',
        env_overrides={ENV_FILE_ENV: str(env_file), 'LAWCAST_SEMANTIC_DEVICE': 'cpu'},
    )
    assert lines == ['cpu']


def test_missing_env_file_fails_safe(tmp_path: Path):
    """A configured-but-absent path keeps the code defaults (CI clone case)."""
    lines = run_config(
        'DEVICE',
        'UPDATE_CRON',
        env_overrides={ENV_FILE_ENV: str(tmp_path / 'nope.env')},
    )
    assert lines == ['auto', '0 * * * *']


def test_empty_env_file_path_disables_loading(tmp_path: Path):
    """Empty LAWCAST_SEMANTIC_ENV_FILE = off, even when a file exists."""
    env_file = tmp_path / '.env'
    env_file.write_text('LAWCAST_SEMANTIC_DEVICE=mps\n', encoding='utf-8')
    lines = run_config('DEVICE', env_overrides={ENV_FILE_ENV: ''})
    assert lines == ['auto'], f'file was read despite being disabled: {env_file.exists()}'


def test_device_defaults_to_auto_detection():
    """No pin anywhere = probe for better-than-CPU hardware at model load
    (lawcast_semantic/device.py), never a silent hardcoded cpu."""
    (line,) = run_config('DEVICE')
    assert line == 'auto'


def test_device_pin_is_kept_verbatim():
    """An explicit pin (including 'cpu') must survive so operators can force
    a device — the probe only runs for 'auto'."""
    for value in ('cpu', 'mps', 'cuda:1', 'auto'):
        (line,) = run_config('DEVICE', env_overrides={'LAWCAST_SEMANTIC_DEVICE': value})
        assert line == value, value
