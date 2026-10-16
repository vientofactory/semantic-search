"""Dedicated sidecar logger on uvicorn's log format.

All sidecar logs (`service.*`) and the library logs the sidecar depends on
(`lawcast_semantic.*`: device probe, rerank fallback) share uvicorn's own
line layout: `%(levelprefix)s %(message)s` rendered by uvicorn's
`DefaultFormatter`. A `[sidecar]` context marker sits in front of every
message so a sidecar line never reads as a uvicorn line:

    INFO:     Started server process [43431]    <- uvicorn
    INFO:     [sidecar] dense engine ready ...  <- sidecar

`get_service_logger(name)` keeps the requested logger name (tests pin
`service` and `service.update_runner` via caplog) and attaches the handler
once to the `service` base only — children propagate, so no line is emitted
twice. Propagation stays on so pytest caplog still sees every record.
"""

from __future__ import annotations

import copy
import logging

# Context marker in front of every sidecar message; uvicorn's own lines
# carry no marker, so the two sources stay visually distinct.
CONTEXT = '[sidecar]'
_FORMAT = '%(levelprefix)s ' + CONTEXT + ' %(message)s'


class _PlainUvicornStyleFormatter(logging.Formatter):
    """Level-prefix layout identical to uvicorn's, without colours.

    Fallback when uvicorn itself cannot be imported (light-import tests):
    `levelprefix` is the level name padded exactly like uvicorn's
    `ColourizedFormatter` builds it (`INFO:     `, `WARNING:  `).
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        levelname = record.levelname
        record.__dict__['levelprefix'] = levelname + ':' + ' ' * (8 - len(levelname))
        return super().formatMessage(record)


def _formatter() -> logging.Formatter:
    try:
        from uvicorn.logging import DefaultFormatter

        return DefaultFormatter(fmt=_FORMAT)
    except Exception:  # noqa: BLE001 - uvicorn missing; keep the same layout uncoloured
        return _PlainUvicornStyleFormatter(fmt=_FORMAT)


def _ensure_handler(logger: logging.Logger) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_formatter())
    logger.addHandler(handler)


def get_service_logger(name: str) -> logging.Logger:
    """Return the named sidecar logger, configuring the shared handler once."""
    base = logging.getLogger('service')
    base.setLevel(logging.INFO)
    _ensure_handler(base)
    # Library logs the sidecar depends on (device probe, rerank fallback)
    # would otherwise fall to the root logger, which uvicorn leaves at
    # WARNING with no handler: INFO detection lines vanish and warnings
    # print bare. Route them through the same format.
    library = logging.getLogger('lawcast_semantic')
    library.setLevel(logging.INFO)
    _ensure_handler(library)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
