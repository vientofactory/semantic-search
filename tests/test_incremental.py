"""Tests for incremental artifact updates (stage 6 library layer).

Runs entirely with a deterministic stub embedder (same text -> same vector) so
full-rebuild vs incremental equivalence can be compared exactly, including
add/update/delete scenarios, crash-window repair, and idempotence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lawcast_semantic.chunking import (
    chunk_notices,
    compose_embedding_text,
    compute_chunk_text_digest,
    compute_chunks_fingerprint,
    load_chunks_jsonl,
    write_chunks_jsonl,
)
from lawcast_semantic.incremental import CHUNK_DIGESTS_MEMBER, apply_update, plan_update
from lawcast_semantic.indexing import VectorIndex
from lawcast_semantic.search import SemanticSearcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / 'scripts'

MODEL = 'stub-model'


class StubEmbedder:
    """Deterministic text -> vector map that records every embedded text."""

    model_name = MODEL
    dimension = 8

    def __init__(self):
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts):
        texts = list(texts)
        self.embedded_texts.extend(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype='float32')
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')
        return np.random.default_rng(seed).random(StubEmbedder.dimension).astype('float32')


def make_notice(num: int, *, reason: str | None = None) -> dict:
    sentence = f'{num}번 법률안의 제안이유를 설명하는 문장입니다. '
    return {
        'notice_num': num,
        'subject': f'법률안 제{num}호',
        'proposer_category': '의원',
        'committee': '교육위원회',
        'proposal_reason': reason if reason is not None else '제안이유\n' + sentence * 12,
    }


def artifact_paths(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {
        'chunks_path': root / 'chunks.jsonl',
        'embeddings_path': root / 'embeddings.npz',
        'index_path': root / 'faiss.index',
        'id_map_path': root / 'id_map.json',
    }


def build_full_artifacts(notices: list[dict], root: Path, *, legacy_npz: bool = False) -> dict:
    """Replicate scripts 01-03 (a full rebuild) into `root`; returns the paths.

    `legacy_npz=True` writes the pre-provenance npz format (no
    chunk_text_digests) to exercise the set-fingerprint fallback.
    """
    paths = artifact_paths(root)
    records = [chunk.to_dict() for chunk in chunk_notices(notices)]
    embedder = StubEmbedder()
    texts = [compose_embedding_text(record['subject'], record['text']) for record in records]
    embeddings = embedder.embed_texts(texts)
    fingerprint = compute_chunks_fingerprint(records)
    members = dict(
        embeddings=embeddings,
        chunk_ids=np.asarray([record['chunk_id'] for record in records]),
        chunks_fingerprint=np.asarray(fingerprint),
        model_name=np.asarray(MODEL),
    )
    if not legacy_npz:
        members[CHUNK_DIGESTS_MEMBER] = np.asarray(
            [compute_chunk_text_digest(record) for record in records]
        )
    np.savez_compressed(paths['embeddings_path'], **members)
    index = VectorIndex(embeddings.shape[1])
    index.build(embeddings)
    index.save(
        paths['index_path'],
        paths['id_map_path'],
        [record['chunk_id'] for record in records],
        meta={'chunks_fingerprint': fingerprint, 'model_name': MODEL},
    )
    write_chunks_jsonl(records, paths['chunks_path'])
    return paths


def run_update(notices: list[dict], stub: StubEmbedder, paths: dict, model: str = MODEL):
    """The script's plan-then-apply flow, driven with the stub embedder."""
    plan = plan_update(
        notices,
        chunks_path=paths['chunks_path'],
        embeddings_path=paths['embeddings_path'],
        model_name=model,
    )
    return apply_update(plan, stub.embed_texts, model, **paths)


def load_searcher(paths: dict) -> SemanticSearcher:
    return SemanticSearcher.load(
        StubEmbedder(),
        chunks_path=paths['chunks_path'],
        index_path=paths['index_path'],
        id_map_path=paths['id_map_path'],
    )


def result_snapshot(searcher: SemanticSearcher, queries: list[str]) -> list:
    return [
        [(result.chunk_id, result.score) for result in searcher.search(query, 10)]
        for query in queries
    ]


def test_incremental_equals_full_rebuild(tmp_path: Path):
    """Old artifacts + incremental update must equal a full rebuild exactly."""
    batch_a = [make_notice(1), make_notice(2), make_notice(3)]
    batch_b = [make_notice(4), make_notice(5), make_notice(6)]
    full_paths = build_full_artifacts(batch_a + batch_b, tmp_path / 'full')
    inc_paths = build_full_artifacts(batch_a, tmp_path / 'inc')

    stub = StubEmbedder()
    report = run_update(batch_a + batch_b, stub, inc_paths)
    assert report.added_notices == (4, 5, 6)
    assert report.embedded_count == len(chunk_notices(batch_b))
    assert report.reused_count == len(chunk_notices(batch_a))

    assert inc_paths['chunks_path'].read_bytes() == full_paths['chunks_path'].read_bytes()

    with (
        np.load(full_paths['embeddings_path']) as full_npz,
        np.load(inc_paths['embeddings_path']) as inc_npz,
    ):
        assert np.array_equal(inc_npz['embeddings'], full_npz['embeddings'])
        assert [str(x) for x in inc_npz['chunk_ids']] == [str(x) for x in full_npz['chunk_ids']]
        full_fp = str(full_npz['chunks_fingerprint'].item())
        inc_fp = str(inc_npz['chunks_fingerprint'].item())
        assert inc_fp == full_fp
        assert [str(x) for x in inc_npz[CHUNK_DIGESTS_MEMBER]] == [
            str(x) for x in full_npz[CHUNK_DIGESTS_MEMBER]
        ]

    queries = ['법률안 제안이유 설명', '주요내용 개정', '교육위원회 심사']
    assert result_snapshot(load_searcher(inc_paths), queries) == result_snapshot(
        load_searcher(full_paths), queries
    )


def test_update_scenario_reembeds_only_changed_notice(tmp_path: Path):
    paths = build_full_artifacts([make_notice(1), make_notice(2)], tmp_path)
    changed_reason = '제안이유\n개정된 제안이유로 탄소중립 이행 근거를 신설합니다. ' * 12
    updated = [make_notice(1), make_notice(2, reason=changed_reason)]

    stub = StubEmbedder()
    report = run_update(updated, stub, paths)
    assert report.updated_notices == (2,)
    assert report.added_notices == () and report.deleted_notices == ()

    changed_chunks = chunk_notices([make_notice(2, reason=changed_reason)])
    assert report.embedded_count == len(changed_chunks)
    assert len(stub.embedded_texts) == len(changed_chunks)
    assert all('법률안 제2호' in text for text in stub.embedded_texts)

    records = load_chunks_jsonl(paths['chunks_path'])
    notice_2 = [record for record in records if record['notice_num'] == 2]
    assert notice_2 and all('개정된 제안이유' in record['text'] for record in notice_2)
    assert not any('제안이유를 설명하는 문장' in record['text'] for record in notice_2)
    load_searcher(paths)  # must not raise: incremental output passes validation


def test_delete_scenario_removes_chunks_and_rows(tmp_path: Path):
    notices = [make_notice(1), make_notice(2), make_notice(3)]
    paths = build_full_artifacts(notices, tmp_path)

    stub = StubEmbedder()
    report = run_update([make_notice(1), make_notice(2)], stub, paths)
    assert report.deleted_notices == (3,)
    assert report.dropped_count == len(chunk_notices([make_notice(3)]))
    assert report.embedded_count == 0 and stub.embedded_texts == []

    records = load_chunks_jsonl(paths['chunks_path'])
    assert {record['notice_num'] for record in records} == {1, 2}
    with np.load(paths['embeddings_path']) as payload:
        assert len(payload['chunk_ids']) == len(records)
        assert not any(str(chunk_id).startswith('3-') for chunk_id in payload['chunk_ids'])

    hits = load_searcher(paths).search('법률안', k=50)
    assert hits and all(hit.notice_num != 3 for hit in hits)


def test_initial_build_without_prior_artifacts_passes_load(tmp_path: Path):
    paths = artifact_paths(tmp_path)
    stub = StubEmbedder()
    report = run_update([make_notice(1)], stub, paths)
    assert report.wrote_artifacts and report.added_notices == (1,)
    load_searcher(paths)


def test_model_mismatch_refuses_update(tmp_path: Path):
    paths = build_full_artifacts([make_notice(1)], tmp_path)
    notices = [make_notice(1), make_notice(2)]
    with pytest.raises(ValueError, match='full rebuild'):
        plan_update(
            notices,
            chunks_path=paths['chunks_path'],
            embeddings_path=paths['embeddings_path'],
            model_name='other-model',
        )
    with pytest.raises(ValueError, match='full rebuild'):
        run_update(notices, StubEmbedder(), paths, model='other-model')


def test_noop_run_writes_nothing_and_embeds_nothing(tmp_path: Path):
    notices = [make_notice(1), make_notice(2)]
    paths = build_full_artifacts(notices, tmp_path)
    before = {key: path.read_bytes() for key, path in paths.items()}

    stub = StubEmbedder()
    report = run_update(notices, stub, paths)
    assert not report.wrote_artifacts
    assert report.embedded_count == 0 and stub.embedded_texts == []
    assert {key: path.read_bytes() for key, path in paths.items()} == before


def test_crash_before_commit_repairs_without_reembedding(tmp_path: Path):
    """Crash window: npz/index/id_map new, chunks.jsonl still old -> rerun heals.

    Every stored row self-describes its text (chunk_text_digests), so the rerun
    re-embeds nothing and simply completes the write.
    """
    old_notices = [make_notice(1), make_notice(2)]
    new_notices = old_notices + [make_notice(3)]
    old_paths = build_full_artifacts(old_notices, tmp_path / 'old')
    new_paths = build_full_artifacts(new_notices, tmp_path / 'new')
    for key in ('embeddings_path', 'index_path', 'id_map_path'):
        shutil.copyfile(new_paths[key], old_paths[key])

    stub = StubEmbedder()
    report = run_update(new_notices, stub, old_paths)
    assert report.wrote_artifacts
    assert report.embedded_count == 0
    assert stub.embedded_texts == []

    reference = build_full_artifacts(new_notices, tmp_path / 'reference')
    assert old_paths['chunks_path'].read_bytes() == reference['chunks_path'].read_bytes()
    queries = ['질의 하나', '질의 둘']
    assert result_snapshot(load_searcher(old_paths), queries) == result_snapshot(
        load_searcher(reference), queries
    )


def test_crash_before_npz_repairs_by_reembedding_missing(tmp_path: Path):
    """The opposite skew: chunks.jsonl new, stored rows stale -> only the
    chunks without a valid row are embedded."""
    old_notices = [make_notice(1), make_notice(2)]
    new_notices = old_notices + [make_notice(3)]
    paths = build_full_artifacts(old_notices, tmp_path)
    write_chunks_jsonl(
        [chunk.to_dict() for chunk in chunk_notices(new_notices)], paths['chunks_path']
    )

    stub = StubEmbedder()
    report = run_update(new_notices, stub, paths)
    assert report.embedded_count == len(chunk_notices([make_notice(3)]))
    assert len(stub.embedded_texts) == report.embedded_count
    load_searcher(paths)


def test_legacy_npz_without_digests_falls_back_to_set_fingerprint(tmp_path: Path):
    old_notices = [make_notice(1), make_notice(2)]
    new_notices = old_notices + [make_notice(3)]
    paths = build_full_artifacts(old_notices, tmp_path, legacy_npz=True)

    stub = StubEmbedder()
    report = run_update(new_notices, stub, paths)
    assert report.embedded_count == len(chunk_notices([make_notice(3)]))
    with np.load(paths['embeddings_path']) as payload:
        assert CHUNK_DIGESTS_MEMBER in payload.files
    load_searcher(paths)


def test_legacy_npz_with_skewed_chunks_refuses(tmp_path: Path):
    """Unprovenanced rows from a different chunk set must not be reused."""
    old_notices = [make_notice(1), make_notice(2)]
    paths = build_full_artifacts(old_notices, tmp_path, legacy_npz=True)
    write_chunks_jsonl(
        [chunk.to_dict() for chunk in chunk_notices(old_notices + [make_notice(3)])],
        paths['chunks_path'],
    )
    with pytest.raises(ValueError, match='full rebuild'):
        plan_update(
            old_notices,
            chunks_path=paths['chunks_path'],
            embeddings_path=paths['embeddings_path'],
            model_name=MODEL,
        )


def test_delete_all_yields_empty_valid_artifacts(tmp_path: Path):
    paths = build_full_artifacts([make_notice(1), make_notice(2)], tmp_path)
    stub = StubEmbedder()
    report = run_update([], stub, paths)
    assert report.chunks_total == 0
    assert report.dropped_count == len(chunk_notices([make_notice(1), make_notice(2)]))
    assert load_searcher(paths).search('질의', k=5) == []


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def script_args(paths: dict, corpus: Path, model: str = MODEL) -> tuple[str, ...]:
    return (
        '--input',
        str(corpus),
        '--model',
        model,
        '--chunks',
        str(paths['chunks_path']),
        '--embeddings',
        str(paths['embeddings_path']),
        '--index',
        str(paths['index_path']),
        '--id-map',
        str(paths['id_map_path']),
    )


def write_notices_jsonl(path: Path, notices: list[dict]) -> None:
    path.write_text(
        ''.join(json.dumps(notice, ensure_ascii=False) + '\n' for notice in notices),
        encoding='utf-8',
    )


def test_script_noop_run_never_loads_the_model(tmp_path: Path):
    """An in-sync run must finish without touching the (unstubbed) model."""
    notices = [make_notice(1), make_notice(2)]
    paths = build_full_artifacts(notices, tmp_path)
    corpus = tmp_path / 'notices.jsonl'
    write_notices_jsonl(corpus, notices)

    result = run_script('06_incremental_update.py', *script_args(paths, corpus))
    assert result.returncode == 0, result.stderr
    assert 'Traceback' not in result.stderr
    assert 'artifacts           : unchanged' in result.stdout
    assert 'chunks to embed     : 0' in result.stdout


def test_script_plan_only_prints_json_without_artifacts(tmp_path: Path):
    corpus = tmp_path / 'notices.jsonl'
    write_notices_jsonl(corpus, [make_notice(1)])
    paths = artifact_paths(tmp_path / 'artifacts')

    result = run_script('06_incremental_update.py', '--plan-only', *script_args(paths, corpus))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['has_changes'] is True
    assert plan['to_embed'] == len(chunk_notices([make_notice(1)]))
    assert plan['added_notices'] == [1]
    assert not paths['chunks_path'].exists()


def test_script_reports_model_mismatch_cleanly(tmp_path: Path):
    paths = build_full_artifacts([make_notice(1)], tmp_path)
    corpus = tmp_path / 'notices.jsonl'
    write_notices_jsonl(corpus, [make_notice(1)])

    result = run_script(
        '06_incremental_update.py',
        '--plan-only',
        *script_args(paths, corpus, model='other-model'),
    )
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'error: ' in result.stderr and 'full rebuild' in result.stderr
