"""
logger_setup.py — Centralised logging configuration.

Two outputs are produced simultaneously:
  1. Human-readable rotating log file  → logs/session_<timestamp>.log
  2. JSON-lines structured log file    → logs/session_<timestamp>.jsonl
  3. Coloured console output           → stderr

The JSON-lines file is machine-readable and used by the visualiser to
reconstruct the full agentic reasoning timeline.

Usage
-----
    from logger_setup import setup_logging, log_event
    setup_logging("logs")

    log_event("hyperparameter_suggestion", {"iteration": 1, "reasoning": "..."})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# ANSI colour codes (works on Linux/macOS; Windows needs colorama)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
GREY   = "\033[90m"

LEVEL_COLOURS = {
    "DEBUG":    GREY,
    "INFO":     CYAN,
    "WARNING":  YELLOW,
    "ERROR":    RED,
    "CRITICAL": BOLD + RED,
}

# ---------------------------------------------------------------------------
# Module-level session ID (set by setup_logging)
# ---------------------------------------------------------------------------

_session_id: str  = ""
_jsonl_path: str  = ""
_jsonl_fh:   Any  = None


# ---------------------------------------------------------------------------
# Custom formatter — coloured console
# ---------------------------------------------------------------------------

class ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour  = LEVEL_COLOURS.get(record.levelname, "")
        level   = f"{colour}{record.levelname:<8}{RESET}"
        module  = f"{GREY}{record.name:<20}{RESET}"
        message = super().format(record)
        # Strip the duplicated levelname/module from the full message
        # (we only want the final %(message)s part)
        msg_only = record.getMessage()
        ts       = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{GREY}{ts}{RESET} {level} {module} {msg_only}"


# ---------------------------------------------------------------------------
# JSON-lines handler — writes one JSON object per log record
# ---------------------------------------------------------------------------

class JSONLinesHandler(logging.Handler):
    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self._fh   = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts":      datetime.fromtimestamp(record.created).isoformat(),
            "level":   record.levelname,
            "module":  record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields added by log_event()
        for key in ("event", "data", "session_id", "iteration"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        try:
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._fh.close()
        super().close()


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir:      str,
    session_name: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level:    int = logging.DEBUG,
) -> str:
    """
    Configure the root logger.

    Parameters
    ----------
    log_dir       : Directory where log files will be written (created if absent).
    session_name  : Base name for log files (default: auto-generated timestamp).
    console_level : Minimum log level for console output.
    file_level    : Minimum log level for file output.

    Returns the session name (useful for cross-referencing logs).
    """
    global _session_id, _jsonl_path, _jsonl_fh

    os.makedirs(log_dir, exist_ok=True)

    if session_name is None:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    _session_id = session_name
    log_path    = os.path.join(log_dir, f"{session_name}.log")
    jsonl_path  = os.path.join(log_dir, f"{session_name}.jsonl")
    _jsonl_path = jsonl_path

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(ColouredFormatter())
    root.addHandler(ch)

    # ── File handler (plain text, rotating) ──────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    ))
    root.addHandler(fh)

    # ── JSON-lines handler ────────────────────────────────────────────────
    jh = JSONLinesHandler(jsonl_path)
    jh.setLevel(file_level)
    root.addHandler(jh)

    logging.getLogger(__name__).info(
        "Logging started. session=%s  log=%s  jsonl=%s",
        session_name, log_path, jsonl_path,
    )

    return session_name


# ---------------------------------------------------------------------------
# Structured event logger
# ---------------------------------------------------------------------------

def log_event(
    event:     str,
    data:      Dict[str, Any],
    level:     int            = logging.INFO,
    iteration: Optional[int]  = None,
    logger_name: str          = "agent",
) -> None:
    """
    Write a structured event to the log.

    These events are distinct from ordinary log lines — they carry a
    machine-readable 'event' tag and a 'data' payload, making them easy
    to filter and visualise.

    Parameters
    ----------
    event      : Short event name (e.g., 'hyperparameter_suggestion').
    data       : Arbitrary dict payload.
    level      : Python logging level.
    iteration  : Agent iteration number (optional).
    logger_name: Which logger to use.
    """
    log = logging.getLogger(logger_name)
    extra: Dict[str, Any] = {
        "event":      event,
        "data":       data,
        "session_id": _session_id,
    }
    if iteration is not None:
        extra["iteration"] = iteration

    log.log(level, "[EVENT] %s — %s", event, _summarise(data), extra=extra)


def _summarise(data: dict, max_len: int = 120) -> str:
    """One-line summary of a data dict for the human-readable log."""
    try:
        s = json.dumps(data, default=str)
        return s if len(s) <= max_len else s[:max_len] + "…"
    except Exception:
        return str(data)[:max_len]


# ---------------------------------------------------------------------------
# JSONL reader (used by visualiser)
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> list:
    """Load all records from a JSON-lines log file."""
    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def get_events(path: str, event_name: Optional[str] = None) -> list:
    """Filter JSONL records to those with a specific event tag."""
    records = read_jsonl(path)
    if event_name:
        records = [r for r in records if r.get("event") == event_name]
    return records


def get_jsonl_path() -> str:
    return _jsonl_path
