"""Entry-point tests: run the pipeline scripts as subprocesses.

Covers boundary inputs at the CLI surface (empty input, invalid --k) without
loading the embedding model.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / 'scripts'


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_package_imports_stay_light():
    """Light library imports must not load the model stack (CLI/library split)."""
    code = (
        'import sys\n'
        'import lawcast_semantic, lawcast_semantic.chunking, lawcast_semantic.datasource\n'
        "print([m for m in ('torch', 'sentence_transformers', 'faiss') if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'


def test_stage1_handles_empty_input(tmp_path: Path):
    empty_input = tmp_path / 'empty_notices.jsonl'
    empty_input.write_text('', encoding='utf-8')
    out = tmp_path / 'chunks.jsonl'
    result = run_script('01_preprocess_chunk.py', '--input', str(empty_input), '--out', str(out))
    assert result.returncode == 0, result.stderr
    assert 'Traceback' not in result.stderr
    assert 'chunks created      : 0' in result.stdout
    assert out.read_text(encoding='utf-8') == ''


def test_stage1_handles_empty_proposal_reason(tmp_path: Path):
    """A notice with no proposal_reason falls back to the subject."""
    sample = tmp_path / 'notices.jsonl'
    sample.write_text(
        '{"notice_num": 1, "subject": "빈 공고", "committee": "", "proposal_reason": ""}\n',
        encoding='utf-8',
    )
    out = tmp_path / 'chunks.jsonl'
    result = run_script('01_preprocess_chunk.py', '--input', str(sample), '--out', str(out))
    assert result.returncode == 0, result.stderr
    assert 'chunks created      : 1' in result.stdout
    assert '빈 공고' in out.read_text(encoding='utf-8')


def test_stage1_learns_from_db_proposal_reason(tmp_path: Path):
    """--db chunks straight from notice_archives.proposalReason (no snapshot)."""
    import sqlite3

    db_path = tmp_path / 'lawcast.db'
    connection = sqlite3.connect(db_path)
    connection.execute(
        'CREATE TABLE notice_archives ('
        'noticeNum integer PRIMARY KEY, subject varchar(500) NOT NULL, '
        'proposerCategory varchar(100) NOT NULL, committee varchar(200) NOT NULL, '
        "proposalReason text NOT NULL DEFAULT '')"
    )
    connection.execute(
        "INSERT INTO notice_archives VALUES (1, 'DB기반법안', '의원', '위원회', "
        "'제안이유\nDB proposalReason에서 직접 학습하는 본문 내용입니다.')"
    )
    connection.commit()
    connection.close()

    out = tmp_path / 'chunks.jsonl'
    result = run_script('01_preprocess_chunk.py', '--db', str(db_path), '--out', str(out))
    assert result.returncode == 0, result.stderr
    assert 'Traceback' not in result.stderr
    assert 'notices read        : 1' in result.stdout
    assert f'{db_path} (notice_archives.proposalReason)' in result.stdout
    assert 'DB proposalReason에서 직접 학습' in out.read_text(encoding='utf-8')


def test_stage1_rejects_db_and_input_together(tmp_path: Path):
    result = run_script(
        '01_preprocess_chunk.py',
        '--db',
        str(tmp_path / 'lawcast.db'),
        '--input',
        str(tmp_path / 'notices.jsonl'),
    )
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'not allowed with argument' in result.stderr


def test_stage6_rejects_db_and_input_together(tmp_path: Path):
    result = run_script(
        '06_incremental_update.py',
        '--db',
        str(tmp_path / 'lawcast.db'),
        '--input',
        str(tmp_path / 'notices.jsonl'),
    )
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'not allowed with argument' in result.stderr


def test_search_rejects_nonpositive_k_without_loading_model():
    result = run_script('04_search.py', '--query', '테스트', '--k', '0')
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert '--k must be a positive integer' in result.stderr


def test_stage5_rejects_empty_eval_set_without_loading_model(tmp_path: Path):
    empty_eval = tmp_path / 'eval.jsonl'
    empty_eval.write_text('', encoding='utf-8')
    result = run_script('05_evaluate.py', '--eval', str(empty_eval))
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'eval set is empty' in result.stderr


def test_search_reports_stale_artifacts_without_traceback(monkeypatch, capsys):
    """Artifact mismatch is a user error: clean error line, like --k validation.

    Drives the script's main() with a stubbed embedder/searcher so the check
    runs without loading the embedding model.
    """
    script = SCRIPTS / '04_search.py'
    spec = importlib.util.spec_from_file_location('search_cli_under_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class StubSearcher:
        @classmethod
        def load(cls, embedder):
            raise ValueError('chunks.jsonl is out of sync with the FAISS index')

    monkeypatch.setattr(module, 'KoreanEmbedder', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'SemanticSearcher', StubSearcher)
    monkeypatch.setattr(sys, 'argv', ['04_search.py', '--query', '테스트'])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code != 0
    err = capsys.readouterr().err
    assert 'error: chunks.jsonl is out of sync' in err
    assert 'Traceback' not in err
