"""Tests for stage 4c: cross-encoder rerank shadow scorer.

The real model is never downloaded here: scoring is verified through an
injected predictor, and the light-import gate runs the import in a fresh
interpreter (torch/faiss must stay out).
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from lawcast_semantic.reranking import Reranker, shadow_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_hits(count: int = 3):
    return [SimpleNamespace(subject=f'법률안 {i}', text=f'내용 {i}') for i in range(count)]


def test_score_pairs_query_with_each_text_in_order():
    seen = []

    def predictor(pairs):
        seen.extend(pairs)
        return [0.5, -1.0, 2.0][: len(pairs)]

    reranker = Reranker('stub-reranker', 'cpu', predictor)
    assert reranker.score('질의', ['a', 'b', 'c']) == [0.5, -1.0, 2.0]
    assert [pair[0] for pair in seen] == ['질의'] * 3
    assert [pair[1] for pair in seen] == ['a', 'b', 'c']


def test_score_empty_window_scores_nothing():
    assert Reranker('stub', 'cpu', lambda pairs: []).score('질의', []) == []


def test_shadow_inputs_truncates_to_top_k_in_rank_order():
    texts = shadow_inputs('질의', make_hits(5), top_k=2)
    assert len(texts) == 2
    assert '법률안 0' in texts[0]
    assert '법률안 1' in texts[1]


def test_load_reranker_off_mode_loads_nothing(monkeypatch):
    """`maybe_load_reranker` in `off` mode must not touch the model stack."""

    import service.app as app_module

    monkeypatch.setattr(app_module.config, 'RERANK_MODE', 'off')

    def fail_load(*args, **kwargs):
        raise AssertionError('model stack touched while off')

    stub_module = SimpleNamespace(load_reranker=fail_load)
    monkeypatch.setitem(sys.modules, 'lawcast_semantic.reranking', stub_module)
    state = app_module.create_state()
    app_module.maybe_load_reranker(state)
    assert state.snapshot()['reranker'] is None


def test_reranker_load_failure_degrades_to_unscored_shadow(monkeypatch):
    import lawcast_semantic.reranking as reranking_module
    import service.app as app_module

    def boom(*args, **kwargs):
        raise ValueError('no weights')

    monkeypatch.setattr(app_module.config, 'RERANK_MODE', 'shadow')
    monkeypatch.setattr(reranking_module, 'load_reranker', boom)
    state = app_module.create_state()
    app_module.maybe_load_reranker(state)
    snapshot = state.snapshot()
    assert snapshot['reranker'] is None
    assert 'no weights' in snapshot['rerankError']


def test_import_reranking_stays_light():
    """`import lawcast_semantic.reranking` must not pull torch/faiss."""
    code = (
        'import sys\n'
        'import lawcast_semantic.reranking\n'
        "heavy = [name for name in ('torch', 'faiss') if name in sys.modules]\n"
        "assert not heavy, f'heavy modules imported: {heavy}'\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_load_reranker_uses_config_defaults():
    """Model/device wiring runs in a subprocess: the model stack must not
    load in the pytest process (faiss-before-torch aborts it)."""
    code = (
        'import lawcast_semantic.config as config\n'
        'from lawcast_semantic import reranking\n'
        'from sentence_transformers import CrossEncoder\n'
        'import inspect\n'
        'source = inspect.getsource(reranking.load_reranker)\n'
        "assert 'CrossEncoder' in source\n"
        "assert 'max_length' in source\n"
        "assert config.RERANK_DEVICE == 'cpu'\n"
        'assert config.RERANK_TOP_K >= 1\n'
        'assert config.RERANK_MAX_LENGTH >= 1\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
