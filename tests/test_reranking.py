"""Tests for stage 4c: cross-encoder rerank shadow scorer.

The real model is never downloaded here: scoring is verified through an
injected predictor, and the light-import gate runs the import in a fresh
interpreter (torch/faiss must stay out).
"""

import os
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
        "assert 'resolve_device' in source\n"
        "assert config.RERANK_DEVICE == 'cpu'\n"
        'assert config.RERANK_TOP_K >= 1\n'
        'assert config.RERANK_MAX_LENGTH >= 1\n'
    )
    # Isolate from a developer `.env` / shell exports: the default case must
    # be deterministic (same pattern as test_config.run_config).
    env = {k: v for k, v in os.environ.items() if not k.startswith('LAWCAST_SEMANTIC_RERANK_')}
    env['LAWCAST_SEMANTIC_ENV_FILE'] = ''
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_load_reranker_resolves_auto_and_falls_back_to_cpu(monkeypatch):
    """`auto` resolves through device.py; a broken accelerator reloads on cpu."""
    import lawcast_semantic.reranking as reranking_module

    seen_devices: list[str] = []

    class StubPredictor:
        def __init__(self, model_name, max_length=None, device='cpu'):
            seen_devices.append(device)
            if device != 'cpu':
                raise RuntimeError(f'{device} kernel is missing')
            self.predict = lambda pairs: [0.1] * len(pairs)

    monkeypatch.setattr(reranking_module, 'CrossEncoder', StubPredictor, raising=False)
    # `CrossEncoder` is imported inside `load_reranker` from
    # `sentence_transformers`; stub that import path too.
    import sys
    import types

    stub_package = types.ModuleType('sentence_transformers')
    stub_package.CrossEncoder = StubPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', stub_package)

    def fake_resolve(requested: str) -> str:
        return 'mps' if requested == 'auto' else requested

    monkeypatch.setattr(reranking_module, 'resolve_device', fake_resolve)
    reranker = reranking_module.load_reranker('stub-reranker', 'auto')
    assert reranker.device == 'cpu'
    assert reranker.requested_device == 'auto'
    assert seen_devices == ['mps', 'cpu']
    assert reranker.score('질의', ['a', 'b']) == [0.1, 0.1]


def test_load_reranker_keeps_resolved_accelerator(monkeypatch):
    """A working accelerator pin is used as-is after the probe passes."""
    import lawcast_semantic.reranking as reranking_module

    seen_devices: list[str] = []

    class StubPredictor:
        def __init__(self, model_name, max_length=None, device='cpu'):
            seen_devices.append(device)
            self.predict = lambda pairs: [0.2] * len(pairs)

        def predict_pairs(self, pairs):
            return self.predict(pairs)

    import sys
    import types

    stub_package = types.ModuleType('sentence_transformers')
    stub_package.CrossEncoder = StubPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', stub_package)
    monkeypatch.setattr(reranking_module, 'resolve_device', lambda requested: requested)
    # Probe calls `predictor.predict([[q, d]])`; route it to the stub.
    original_probe = reranking_module._probe
    monkeypatch.setattr(
        reranking_module, '_probe', lambda predictor, device: seen_devices.append(f'probe:{device}')
    )
    reranker = reranking_module.load_reranker('stub-reranker', 'mps')
    assert reranker.device == 'mps'
    assert reranker.requested_device == 'mps'
    assert seen_devices[0] == 'mps'
    monkeypatch.setattr(reranking_module, '_probe', original_probe)
