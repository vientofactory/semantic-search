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
    assert lines == ['cpu', '0 * * * *']


def test_empty_env_file_path_disables_loading(tmp_path: Path):
    """Empty LAWCAST_SEMANTIC_ENV_FILE = off, even when a file exists."""
    env_file = tmp_path / '.env'
    env_file.write_text('LAWCAST_SEMANTIC_DEVICE=mps\n', encoding='utf-8')
    lines = run_config('DEVICE', env_overrides={ENV_FILE_ENV: ''})
    assert lines == ['cpu'], f'file was read despite being disabled: {env_file.exists()}'
