"""Single owner of the OpenMP workaround for the model stack.

torch (sentence-transformers) and faiss each bundle their own ``libomp.dylib``
on macOS, so one process that loads both crashes at search time (the failure
point is non-deterministic). Measured on this stack: ``faiss.omp_set_num_threads(1)``
fails (the call itself initializes the second runtime), ``KMP_DUPLICATE_LIB_OK=TRUE``
alone fails 0/3 runs, and ``OMP_NUM_THREADS=1`` alone passes 3/3 with faiss top-k
exactly matching the numpy ground truth. Single-threaded exact search is
latency-bound (~0.11 s/query), so entrypoints that load both engines call
``use_single_threaded_omp()`` before importing torch/faiss. torch-only (stage 2)
and faiss-only (stage 3) pipelines keep full threading.
"""

from __future__ import annotations

import os


def use_single_threaded_omp() -> None:
    """Prefer single-threaded OpenMP; a pre-set OMP_NUM_THREADS is respected."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
