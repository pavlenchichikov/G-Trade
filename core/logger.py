"""Centralized logging configuration for G-Trade.

Usage in any module:
    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")

Logs go to both console (INFO) and file gtrade.log (DEBUG).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_LOG_DIR, "gtrade.log")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _setup_root():
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — INFO level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # File handler — DEBUG level, rotating
    try:
        fh = RotatingFileHandler(
            _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)
    except OSError:
        # Can't write to log file (read-only FS, permissions, etc.)
        pass

    # Suppress noisy third-party loggers
    for name in ("urllib3", "matplotlib", "PIL", "h5py", "absl"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Initializes root logging on first call."""
    _setup_root()
    return logging.getLogger(name)
