"""Compute-device selection for the embedding model.

Single owner of "which hardware should the model run on": `config.DEVICE`
carries the *requested* device and this module turns it into the device to
actually load on.

- An explicit pin (`cpu`, `cuda`, `cuda:1`, `mps`, `xpu`, ...) is honoured
  so an operator can force a choice — including forcing `cpu`.
- `auto` (the default) probes torch for hardware faster than CPU, in
  preference order CUDA > Apple MPS > Intel XPU, and returns `cpu` when no
  accelerator is usable.

Detection is lazy: torch is imported inside `resolve_device`, so importing
this module stays as light as `lawcast_semantic.config` (the library's
light-import contract, pinned by `tests/test_entrypoints.py`). The device
only decides where the forward pass runs — it never changes the vectors'
meaning, so switching devices never invalidates embeddings or FAISS
artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# The requested-device spelling that means "probe the hardware".
AUTO = 'auto'


def _cuda_available(torch_module) -> bool:
    return bool(torch_module.cuda.is_available())


def _mps_available(torch_module) -> bool:
    # torch.backends.mps only exists on builds/platforms that know it.
    mps = getattr(getattr(torch_module, 'backends', None), 'mps', None)
    return mps is not None and bool(mps.is_available())


def _xpu_available(torch_module) -> bool:
    xpu = getattr(torch_module, 'xpu', None)
    return xpu is not None and bool(xpu.is_available())


# Probe order: discrete accelerator first (CUDA), then Apple Silicon, then
# Intel GPU. Each backend has its own availability API and each check is
# isolated: a broken driver must degrade to the next candidate, never raise.
_PROBES: tuple[tuple[str, Callable], ...] = (
    ('cuda', _cuda_available),
    ('mps', _mps_available),
    ('xpu', _xpu_available),
)


def best_available_device(torch_module) -> str:
    """Return the fastest usable accelerator on `torch_module`, else `cpu`."""
    for name, probe in _PROBES:
        try:
            if probe(torch_module):
                return name
        except Exception as exc:  # noqa: BLE001 - a broken backend must not break resolution
            logger.warning('device probe %s failed (%s: %s)', name, type(exc).__name__, exc)
    return 'cpu'


def resolve_device(requested: str | None = None) -> str:
    """Turn a requested device (`config.DEVICE`) into the device to load on.

    A non-`auto` request is returned as an explicit pin (trimmed/lowercased).
    `auto` (also the result of an empty request) probes torch for hardware
    faster than CPU and falls back to `cpu` when torch itself cannot be
    imported.
    """
    pin = str(requested if requested is not None else AUTO).strip().lower()
    if not pin:
        pin = AUTO
    if pin != AUTO:
        return pin
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - no torch means no accelerator, not a failure
        logger.warning(
            'auto device detection unavailable (%s: %s); using cpu', type(exc).__name__, exc
        )
        return 'cpu'
    device = best_available_device(torch)
    if device != 'cpu':
        logger.info('auto device detection: better-than-CPU hardware found, using %s', device)
    else:
        logger.info('auto device detection: no accelerator available, using cpu')
    return device
