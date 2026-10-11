"""Entry-point tests: run the pipeline scripts as subprocesses.

Covers boundary inputs at the CLI surface (empty input, invalid --k) without
loading the embedding model.
"""

import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from lawcast_semantic.chunking import (
    chunk_notice,
    compute_chunks_fingerprint,
    write_chunks_jsonl,
)
from lawcast_semantic.datasource import load_notices_from_db, write_notices_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / 'scripts'


def run_script(name: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
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
    """--db uses notice_archives.proposalReason plus the joined AI summary."""
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
    connection.execute(
        'CREATE TABLE notice_archive_snapshot_states ('
        'notice_num integer PRIMARY KEY, ai_summary text)'
    )
    connection.execute(
        'INSERT INTO notice_archive_snapshot_states VALUES (?, ?)',
        (1, '국민의 권익을 보호하려는 개정안입니다.'),
    )
    connection.commit()
    connection.close()

    out = tmp_path / 'chunks.jsonl'
    result = run_script('01_preprocess_chunk.py', '--db', str(db_path), '--out', str(out))
    assert result.returncode == 0, result.stderr
    assert 'Traceback' not in result.stderr
    assert 'notices read        : 1' in result.stdout
    assert f'{db_path} (notice_archives.proposalReason)' in result.stdout
    chunks = out.read_text(encoding='utf-8')
    assert 'DB proposalReason에서 직접 학습' in chunks
    assert '국민의 권익을 보호하려는 개정안입니다.' in chunks
    assert '"section": "AI 요약"' in chunks


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


# --- stage 7: capability coverage report (the release check) ---

# Four notice shapes covering every capability the report distinguishes:
#   1 empty proposal_reason + a summary that does NOT name the bill -> its title
#     capability hangs on the subject fallback chunk alone,
#   2 a body section + a summary                                  -> section, derived,
#   3 an empty proposal_reason and no summary                     -> title fallback,
#   4 an empty proposal_reason + a summary that NAMES the bill     -> keeps a title
#     carrier even when the fallback chunk is taken away.
CAPABILITY_CORPUS = (
    (1, '요약법 일부개정법률안', '', '국민의 권익을 보호하려는 개정안입니다.'),
    (
        2,
        '본문법 일부개정법률안',
        '제안이유\n현행법은 미비하므로 개선이 필요합니다.',
        '요약문입니다.',
    ),
    (3, '빈공고법 일부개정법률안', '', ''),
    (
        4,
        '이름담김법 일부개정법률안',
        '',
        '이름담김법 일부개정법률안은 처벌을 강화하려는 개정안입니다.',
    ),
)
CAPABILITY_LINE = 'indexed=4 title_text=3 section:제안이유=1 derived:ai_summary=3'


def build_capability_corpus(tmp_path: Path) -> Path:
    """LawCast-shaped DB with the four shapes above (read mode=ro by the loader)."""
    db_path = tmp_path / 'lawcast.db'
    connection = sqlite3.connect(db_path)
    connection.execute(
        'CREATE TABLE notice_archives ('
        'noticeNum integer PRIMARY KEY, subject varchar(500) NOT NULL, '
        'proposerCategory varchar(100) NOT NULL, committee varchar(200) NOT NULL, '
        "proposalReason text NOT NULL DEFAULT '')"
    )
    connection.execute(
        'CREATE TABLE notice_archive_snapshot_states ('
        'notice_num integer PRIMARY KEY, ai_summary text)'
    )
    for notice_num, subject, reason, summary in CAPABILITY_CORPUS:
        connection.execute(
            'INSERT INTO notice_archives VALUES (?, ?, ?, ?, ?)',
            (notice_num, subject, '의원', '위원회', reason),
        )
        connection.execute(
            'INSERT INTO notice_archive_snapshot_states VALUES (?, ?)', (notice_num, summary)
        )
    connection.commit()
    connection.close()
    return db_path


def build_capability_artifacts(
    db_path: Path, artifacts: Path, *, drop_title_chunk: bool = False
) -> Path:
    """A chunk set + id_map for the corpus; drop_title_chunk rebuilds the swap.

    `drop_title_chunk` mirrors the first AI-summary rule: for a notice whose
    `proposal_reason` is empty, the summary chunk REPLACES the subject fallback
    (the chunk set the report must still recognise as a capability loss).
    """
    records: list[dict] = []
    for notice in load_notices_from_db(db_path):
        chunks = [chunk.to_dict() for chunk in chunk_notice(notice)]
        if drop_title_chunk and not notice['proposal_reason'].strip() and notice['ai_summary']:
            chunks = chunks[1:]
        records.extend(chunks)
    write_chunks_jsonl(records, artifacts / 'chunks.jsonl')
    (artifacts / 'id_map.json').write_text(
        json.dumps(
            {
                'chunk_ids': [record['chunk_id'] for record in records],
                'chunks_fingerprint': compute_chunks_fingerprint(records),
                'model_name': 'stub-model',
                'updated_at': '2026-10-12T00:00:00+00:00',
            },
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    return artifacts


def test_stage7_reports_the_served_index_capabilities(tmp_path: Path):
    db_path = build_capability_corpus(tmp_path)
    artifacts = build_capability_artifacts(db_path, tmp_path / 'artifacts')
    result = run_script(
        '07_capability_report.py', '--artifacts', str(artifacts), '--db', str(db_path)
    )
    assert result.returncode == 0, result.stderr
    assert 'Traceback' not in result.stderr
    assert 'corpus              : 4 notices' in result.stdout
    assert re.search(r'artifact\s+: \d+ chunks, 4 notices indexed', result.stdout)
    assert 'index integrity     : OK' in result.stdout
    assert re.search(rf'capabilities\s+: {re.escape(CAPABILITY_LINE)}', result.stdout), (
        result.stdout
    )
    assert 'release check' not in result.stdout


def test_stage7_accepts_a_jsonl_snapshot_and_requires_a_source(tmp_path: Path):
    db_path = build_capability_corpus(tmp_path)
    artifacts = build_capability_artifacts(db_path, tmp_path / 'artifacts')
    snapshot = tmp_path / 'notices.jsonl'
    write_notices_jsonl(load_notices_from_db(db_path), snapshot)
    result = run_script(
        '07_capability_report.py', '--artifacts', str(artifacts), '--input', str(snapshot)
    )
    assert result.returncode == 0, result.stderr
    assert f'source              : {snapshot}' in result.stdout
    assert re.search(rf'capabilities\s+: {re.escape(CAPABILITY_LINE)}', result.stdout)

    # No source anywhere: a clean error, not a traceback (env cleared so a
    # developer's .env or shell export cannot supply one).
    missing = run_script(
        '07_capability_report.py',
        '--artifacts',
        str(artifacts),
        env={
            **os.environ,
            'LAWCAST_SEMANTIC_DB_PATH': '',
            'LAWCAST_SEMANTIC_ENV_FILE': '',
        },
    )
    assert missing.returncode != 0
    assert 'Traceback' not in missing.stderr
    assert 'error: no corpus source' in missing.stderr


def test_stage7_compare_flags_the_summary_swap(tmp_path: Path):
    """The report's release check sees the regression the aggregate metrics hid."""
    db_path = build_capability_corpus(tmp_path)
    baseline = tmp_path / 'baseline.json'
    before = build_capability_artifacts(db_path, tmp_path / 'before')
    saved = run_script(
        '07_capability_report.py',
        '--artifacts',
        str(before),
        '--db',
        str(db_path),
        '--save',
        str(baseline),
    )
    assert saved.returncode == 0, saved.stderr
    assert f'saved               : {baseline}' in saved.stdout

    after = build_capability_artifacts(db_path, tmp_path / 'after', drop_title_chunk=True)
    result = run_script(
        '07_capability_report.py',
        '--artifacts',
        str(after),
        '--db',
        str(db_path),
        '--compare',
        str(baseline),
    )
    assert result.returncode == 1, result.stdout + result.stderr
    # Notice 1 loses its carrier; notice 4 keeps one (its summary names the bill),
    # so the count delta must be 3 -> 2 with exactly one notice reported.
    assert 'capability deltas   : title_text 3 -> 2' in result.stdout, result.stdout
    assert 'release check       : LOST — title_text (1 notice, e.g. [1])' in result.stdout


def test_stage7_compare_is_clean_for_an_unchanged_set(tmp_path: Path):
    db_path = build_capability_corpus(tmp_path)
    artifacts = build_capability_artifacts(db_path, tmp_path / 'artifacts')
    baseline = tmp_path / 'baseline.json'
    assert (
        run_script(
            '07_capability_report.py',
            '--artifacts',
            str(artifacts),
            '--db',
            str(db_path),
            '--save',
            str(baseline),
        ).returncode
        == 0
    )
    result = run_script(
        '07_capability_report.py',
        '--artifacts',
        str(artifacts),
        '--db',
        str(db_path),
        '--compare',
        str(baseline),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'capability deltas   : none' in result.stdout
    assert 'release check       : OK — no capability class lost' in result.stdout


def test_stage7_flags_a_stale_artifact_set(tmp_path: Path):
    """A fingerprint mismatch means the reported coverage is not the served index."""
    db_path = build_capability_corpus(tmp_path)
    artifacts = build_capability_artifacts(db_path, tmp_path / 'artifacts')
    id_map_path = artifacts / 'id_map.json'
    payload = json.loads(id_map_path.read_text(encoding='utf-8'))
    payload['chunks_fingerprint'] = 'stale-fingerprint-from-an-older-build'
    id_map_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    result = run_script(
        '07_capability_report.py', '--artifacts', str(artifacts), '--db', str(db_path)
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'index integrity     : MISMATCH' in result.stdout
    assert 'chunk fingerprint' in result.stdout


def test_stage7_compares_a_candidate_chunk_set(tmp_path: Path):
    """The CI release guard compares the DEPLOYED code's chunk set (--chunks).

    The baseline is an artifact set (with id_map); the candidate is a bare
    chunk set, exactly what `01_preprocess_chunk.py --db` writes after a deploy.
    """
    db_path = build_capability_corpus(tmp_path)
    baseline = tmp_path / 'baseline.json'
    before = build_capability_artifacts(db_path, tmp_path / 'before')
    saved = run_script(
        '07_capability_report.py',
        '--artifacts',
        str(before),
        '--db',
        str(db_path),
        '--save',
        str(baseline),
    )
    assert saved.returncode == 0, saved.stderr

    swapped = build_capability_artifacts(db_path, tmp_path / 'candidate', drop_title_chunk=True)
    candidate = swapped / 'chunks.jsonl'
    result = run_script(
        '07_capability_report.py',
        '--chunks',
        str(candidate),
        '--db',
        str(db_path),
        '--compare',
        str(baseline),
    )
    assert result.returncode == 1, result.stdout + result.stderr
    # A candidate set is not a served set: no id_map agreement to report.
    assert f'candidate chunks    : {candidate} (5 chunks, 4 notices indexed)' in result.stdout, (
        result.stdout
    )
    assert 'index integrity     : n/a (candidate chunk set, not a served artifact set)' in (
        result.stdout
    )
    assert 'release check       : LOST — title_text (1 notice, e.g. [1])' in result.stdout


def test_stage7_candidate_set_reports_without_an_id_map(tmp_path: Path):
    """A --chunks candidate needs no id_map/fingerprint; it is not the served set."""
    db_path = build_capability_corpus(tmp_path)
    artifacts = build_capability_artifacts(db_path, tmp_path / 'artifacts')
    candidate = tmp_path / 'candidate.jsonl'
    candidate.write_text((artifacts / 'chunks.jsonl').read_text(encoding='utf-8'), encoding='utf-8')
    result = run_script('07_capability_report.py', '--chunks', str(candidate), '--db', str(db_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'index integrity     : n/a' in result.stdout
    assert re.search(rf'capabilities\s+: {re.escape(CAPABILITY_LINE)}', result.stdout)


def test_stage7_missing_candidate_chunk_set_is_a_clean_error(tmp_path: Path):
    db_path = build_capability_corpus(tmp_path)
    result = run_script(
        '07_capability_report.py', '--chunks', str(tmp_path / 'nope.jsonl'), '--db', str(db_path)
    )
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'error: candidate chunk set not found' in result.stderr


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
