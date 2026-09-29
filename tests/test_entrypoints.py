"""Entry-point tests: run the pipeline scripts as subprocesses.

Covers boundary inputs at the CLI surface (empty input, invalid --k) without
loading the embedding model.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / 'scripts'


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_stage1_handles_empty_input(tmp_path: Path):
    empty_input = tmp_path / 'empty_notices.jsonl'
    empty_input.write_text('', encoding='utf-8')
    out = tmp_path / 'chunks.jsonl'
    result = run_script(
        '01_preprocess_chunk.py', '--input', str(empty_input), '--out', str(out)
    )
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


def test_search_rejects_nonpositive_k_without_loading_model():
    result = run_script('04_search.py', '--query', '테스트', '--k', '0')
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert '--k must be a positive integer' in result.stderr
