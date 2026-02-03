from pathlib import Path
from typing import Any

from loguru import logger

_logger_initialized = False


def init_logger(log_file: str | None = None) -> None:
    """Initialize logger. If log_file is None, logging is disabled."""
    global _logger_initialized

    if _logger_initialized:
        return

    if log_file is None:
        logger.remove()
        _logger_initialized = True
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
    )

    _logger_initialized = True
    logger.info("Logger initialized")


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


# logger.py
