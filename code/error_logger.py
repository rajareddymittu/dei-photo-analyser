"""Centralized error and activity logging for the photo intelligence platform."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG_PATH = LOG_DIR / "photo_ai_errors.log"
APP_LOG_PATH = LOG_DIR / "photo_ai.log"

_LOGGER_INITIALIZED = False


def get_logger(name: str = "photo_ai") -> logging.Logger:
    """Return a logger configured with rotating file handlers and console output."""
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)

    if not _LOGGER_INITIALIZED:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. Error log handler: records WARNING, ERROR, CRITICAL only
        error_handler = RotatingFileHandler(
            ERROR_LOG_PATH,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # 2. General app log handler: records INFO and above
        app_handler = RotatingFileHandler(
            APP_LOG_PATH,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(formatter)
        logger.addHandler(app_handler)

        # 3. StreamHandler for stdout
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        _LOGGER_INITIALIZED = True

    return logger


logger = get_logger("photo_ai")


def log_error(msg: str, exc: Exception | None = None, context: dict[str, Any] | None = None) -> None:
    """Convenience helper to record an error with traceback and contextual details."""
    extra_info = f" | Context: {context}" if context else ""
    if exc is not None:
        logger.error(f"{msg}{extra_info} — Exception: {exc}", exc_info=True)
    else:
        logger.error(f"{msg}{extra_info}", exc_info=True)


def log_warning(msg: str, exc: Exception | None = None, context: dict[str, Any] | None = None) -> None:
    """Convenience helper to record a warning."""
    extra_info = f" | Context: {context}" if context else ""
    if exc is not None:
        logger.warning(f"{msg}{extra_info} — Exception: {exc}")
    else:
        logger.warning(f"{msg}{extra_info}")


def get_recent_errors(max_lines: int = 50) -> list[str]:
    """Read the latest lines from the error log file for display."""
    if not ERROR_LOG_PATH.is_file():
        return []
    try:
        with open(ERROR_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-max_lines:]
    except OSError:
        return []
