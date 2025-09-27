"""
Structured logging configuration for ZauriScore.

This module provides a flexible logging system with support for both console
and file output, with optional JSON formatting.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, cast

from ..config import settings

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGING_AVAILABLE = True
except ImportError:
    JSON_LOGGING_AVAILABLE = False

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that includes additional context."""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, 
                  message_dict: Dict[str, Any]) -> None:
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)
        log_record["logger"] = record.name
        log_record["level"] = record.levelname
        log_record["timestamp"] = record.created

def setup_logger(
    name: str = "zauriscore",
    log_level: Optional[Union[str, int]] = None,
    log_file: Optional[Union[str, Path]] = None,
    json_format: Optional[bool] = None,
) -> logging.Logger:
    """
    Configure and return a logger with the specified settings.

    Args:
        name: Logger name (default: "zauriscore")
        log_level: Logging level (default: from settings)
        log_file: Optional file path for logging (default: from settings)
        json_format: Whether to use JSON formatting (default: from settings)

    Returns:
        Configured logger instance
    """
    # Use settings if not overridden
    if log_level is None:
        log_level = settings.get('LOG_LEVEL', 'INFO')
    if json_format is None:
        json_format = False  # Default to False if not specified
    if log_file is None:
        log_file = None  # No default log file in settings

    # Convert string log level to numeric
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(cast(int, log_level))

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    if json_format and JSON_LOGGING_AVAILABLE:
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
        )
    else:
        if json_format and not JSON_LOGGING_AVAILABLE:
            logger.warning(
                "JSON logging requested but python-json-logger is not installed. "
                "Falling back to standard logging format.")
        formatter = logging.Formatter(settings.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(cast(int, log_level))
    console_handler.setFormatter(formatter)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(cast(int, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

# Default logger instance
logger = setup_logger()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name, inheriting the root logger's configuration.

    Args:
        name: Logger name (default: None, uses the root logger)

    Returns:
        Configured logger instance
    """
    if name is None:
        return logger
    return logging.getLogger(f"zauriscore.{name}")
