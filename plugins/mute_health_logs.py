"""Mute health-probe log noise in docling-serve.

Installed into site-packages as ``dcc_mute_health_logs.py`` and auto-imported at
interpreter startup through a ``.pth`` file, so no docling-serve source file has
to be touched. Upstream can change freely; only the logger names and the probe
message below are assumptions, and both are configurable via env vars.

Two log records are suppressed per probe:

    {"logger": "docling_serve.app",  "message": "Health check requested"}
    {"logger": "uvicorn.access",     "message": "1.2.3.4:5 - \\"GET /health HTTP/1.1\\" 200"}

Env vars:
    DCC_MUTE_HEALTH_LOGS=0        disable this patch entirely
    DCC_MUTE_HEALTH_PATHS         comma-separated request paths to mute
                                  (default: /health)
"""

import logging
import os

_ACCESS_LOGGERS = ("uvicorn.access",)
_APP_LOGGERS = ("docling_serve.app",)
_APP_MESSAGE_MARKER = "Health check requested"


def _muted_paths() -> tuple[str, ...]:
    raw = os.environ.get("DCC_MUTE_HEALTH_PATHS", "/health")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


class _MuteAccessPaths(logging.Filter):
    """Drop uvicorn access records for the given request paths."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self.paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn logs '%s - "%s %s HTTP/%s" %s' with
        # args = (client_addr, method, full_path, http_version, status_code)
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path = args[2].split("?", 1)[0]
            return path not in self.paths
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(f'"GET {path} ' in message for path in self.paths)


class _MuteMessage(logging.Filter):
    """Drop records whose message contains a marker string."""

    def __init__(self, marker: str) -> None:
        super().__init__()
        self.marker = marker

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return self.marker not in record.getMessage()
        except Exception:
            return True


def install() -> None:
    if os.environ.get("DCC_MUTE_HEALTH_LOGS", "1").lower() in ("0", "false", "no"):
        return

    paths = _muted_paths()
    if not paths:
        return

    # Filters live on the Logger objects, not on handlers, so they survive
    # docling-serve's setup_logging() (which only swaps handlers and levels).
    for name in _ACCESS_LOGGERS:
        logging.getLogger(name).addFilter(_MuteAccessPaths(paths))
    for name in _APP_LOGGERS:
        logging.getLogger(name).addFilter(_MuteMessage(_APP_MESSAGE_MARKER))


try:
    install()
except Exception:  # never break interpreter startup
    pass
