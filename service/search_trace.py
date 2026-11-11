"""Bounded request correlation shared by the HTTP and gRPC search adapters."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

TRACE_HEADER = 'x-lawcast-search-trace'
ATTEMPT_HEADER = 'x-lawcast-search-attempt'
_TRACE_ID = re.compile(r'[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}', re.IGNORECASE)
_ATTEMPT = re.compile(r'[1-9][0-9]{0,5}')
_trace: ContextVar[tuple[str, str, str]] = ContextVar('search_trace', default=('direct', '-', '-'))


@contextmanager
def search_trace(transport: str, headers: Mapping[str, str | bytes]) -> Iterator[None]:
    trace = headers.get(TRACE_HEADER)
    attempt = headers.get(ATTEMPT_HEADER)
    if not (
        isinstance(trace, str)
        and _TRACE_ID.fullmatch(trace)
        and isinstance(attempt, str)
        and _ATTEMPT.fullmatch(attempt)
    ):
        trace, attempt = '-', '-'
    token = _trace.set((transport, trace, attempt))
    try:
        yield
    finally:
        _trace.reset(token)


def trace_fields() -> tuple[str, str, str]:
    return _trace.get()
