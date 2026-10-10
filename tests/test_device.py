"""Device-selection contract: `lawcast_semantic/device.py` + `KoreanEmbedder`.

Resolution tests stay in-process against a *fake* torch injected into
`sys.modules` — importing the real torch would load the model stack into a
process that must stay light (the macOS torch+faiss libomp clash documented
behind `tests/retrieval_probe.py`), and fake backends are the only way to
test detection deterministically on hosts with and without accelerators.

The embedder's load / probe / cpu-fallback behavior runs in a subprocess
with a stubbed `SentenceTransformer` — the same isolation style as
`tests/test_entrypoints.py`. Every case pins a device explicitly, so the
assertions hold on any host regardless of its real hardware.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lawcast_semantic.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fake_torch(cuda: bool = False, mps: bool = False, xpu: bool = False) -> SimpleNamespace:
    """A torch stand-in whose availability APIs answer from the flags."""

    def check(available: bool):
        return lambda: available

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=check(cuda)),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=check(mps))),
        xpu=SimpleNamespace(is_available=check(xpu)),
    )


# --- requested device -> resolved device ---------------------------------


@pytest.mark.parametrize(
    'requested,expected',
    [
        ('cpu', 'cpu'),  # the pin operators use to stay on cpu
        ('mps', 'mps'),
        ('cuda', 'cuda'),
        ('cuda:1', 'cuda:1'),
        ('CUDA:0', 'cuda:0'),  # normalized, not rejected
        ('  mps  ', 'mps'),
    ],
)
def test_explicit_pin_is_honoured_without_probing(requested, expected):
    assert resolve_device(requested) == expected


def test_auto_prefers_cuda_over_every_other_candidate(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(cuda=True, mps=True, xpu=True))
    assert resolve_device('auto') == 'cuda'


def test_auto_picks_mps_when_there_is_no_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(mps=True))
    assert resolve_device('auto') == 'mps'


def test_auto_picks_xpu_when_only_an_intel_gpu_exists(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(xpu=True))
    assert resolve_device('auto') == 'xpu'


def test_auto_falls_back_to_cpu_when_no_accelerator_is_usable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch())
    assert resolve_device('auto') == 'cpu'


@pytest.mark.parametrize('requested', [None, '', '   ', 'AUTO', ' Auto '])
def test_auto_spellings_all_trigger_the_probe(requested, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(cuda=True))
    assert resolve_device(requested) == 'cuda'


def test_auto_uses_cpu_when_torch_cannot_be_imported(monkeypatch):
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert resolve_device('auto') == 'cpu'


def test_broken_cuda_probe_degrades_to_the_next_candidate(monkeypatch, caplog):
    """A driver that raises must not break resolution — it skips to mps."""
    broken = fake_torch(mps=True)
    broken.cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError('driver mismatch'))
    monkeypatch.setitem(sys.modules, 'torch', broken)
    with caplog.at_level('WARNING', logger='lawcast_semantic.device'):
        assert resolve_device('auto') == 'mps'
    assert 'device probe cuda failed' in caplog.text


def test_auto_logs_the_detected_device(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(mps=True))
    with caplog.at_level('INFO', logger='lawcast_semantic.device'):
        assert resolve_device('auto') == 'mps'
    assert 'using mps' in caplog.text


# --- KoreanEmbedder: load / probe / cpu fallback (subprocess, stub model) ---

# Stub model injected in place of sentence_transformers' SentenceTransformer.
# `fail-construct` raises for any non-cpu device at load; `fail-encode`
# constructs fine but answers the probe with NaN vectors — both are real
# accelerator failure modes (missing kernel / broken numerics).
EMBEDDER_STUB = """
import sys
import types

import numpy as np

import lawcast_semantic.embedding as embedding

mode, requested = sys.argv[1], sys.argv[2]


class StubModel:
    def __init__(self, model_name, device='cpu', **kwargs):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = 512
        self.tokenizer = types.SimpleNamespace()
        if mode == 'fail-construct' and device != 'cpu':
            raise RuntimeError(f'{device} kernel is missing')

    def get_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        if mode == 'fail-encode' and self.device != 'cpu':
            return np.full((len(texts), 4), np.nan, dtype='float32')
        return np.ones((len(texts), 4), dtype='float32')


embedding.SentenceTransformer = StubModel

if mode == 'probe':
    import torch

    detected = embedding.resolve_device('auto')
    capabilities = {
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
    }
    print(f'auto={detected} capabilities={capabilities}')
else:
    embedder = embedding.KoreanEmbedder('stub/model', device=requested)
    vectors = embedder.embed_texts(['임베딩 검증'])
    print(f'requested={embedder.requested_device} resolved={embedder.device} shape={vectors.shape}')
"""


def run_embedder(mode: str, requested: str) -> list[str]:
    """Run the stubbed embedder in a fresh interpreter; return stdout lines."""
    result = subprocess.run(
        [sys.executable, '-c', EMBEDDER_STUB, mode, requested],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_embedder_keeps_a_working_accelerator_pin():
    """A device that loads and probes clean is the device that is used."""
    (line,) = run_embedder('ok', 'cuda')
    assert line == 'requested=cuda resolved=cuda shape=(1, 4)'


def test_embedder_falls_back_to_cpu_when_the_accelerator_cannot_load():
    (line,) = run_embedder('fail-construct', 'cuda')
    assert line == 'requested=cuda resolved=cpu shape=(1, 4)'


def test_embedder_falls_back_to_cpu_when_the_probe_returns_non_finite_vectors():
    """A device that constructs but yields NaN must be caught *before* a long
    corpus run strands halfway — hence the probe at load time."""
    (line,) = run_embedder('fail-encode', 'cuda')
    assert line == 'requested=cuda resolved=cpu shape=(1, 4)'


def test_cpu_pin_is_used_as_is_without_probing():
    """`cpu` never runs the probe, so even a probe that would fail is moot."""
    (line,) = run_embedder('fail-encode', 'cpu')
    assert line == 'requested=cpu resolved=cpu shape=(1, 4)'


def test_auto_probe_agrees_with_torch_capabilities():
    """End-to-end check of the real detection path against the real torch on
    this host: whatever resolve_device('auto') returns must match the
    preference order (cuda > mps > cpu) of the capabilities it reports."""
    (line,) = run_embedder('probe', 'auto')
    resolved = line.split('auto=')[1].split(' ')[0]
    capabilities = line.split('capabilities=')[1]
    if "'cuda': True" in capabilities:
        assert resolved == 'cuda'
    elif "'mps': True" in capabilities:
        assert resolved == 'mps'
    else:
        assert resolved == 'cpu'
