"""Structured logging configuration for SwitchGen."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# Log format with timestamp, module, level, and message
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)s - %(message)s"


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
    simple_format: bool = False,
) -> logging.Logger:
    """Configure application-wide logging.

    Args:
        level: Log level (default: from SWITCHGEN_LOG_LEVEL env var or INFO)
        log_file: Optional file path for persistent logging
        simple_format: Use simplified format without timestamps

    Returns:
        The root switchgen logger
    """
    # Determine log level from environment or parameter
    if level is None:
        env_level = os.environ.get("SWITCHGEN_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    # Get the switchgen root logger
    logger = logging.getLogger("switchgen")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose format
    fmt = LOG_FORMAT_SIMPLE if simple_format else LOG_FORMAT

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (e.g., "switchgen.core.engine")

    Returns:
        Logger instance for the module
    """
    return logging.getLogger(name)
