"""
Logging utilities for Oncorag2.

This module provides a consistent logging configuration for the entire package.
"""

import logging
import os
import sys
from typing import Optional


def configure_logging(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console: bool = True
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, or None for no file logging
        console: Whether to log to console

    Returns:
        Configured logger
    """
    # Get level from string
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logger = logging.getLogger("oncorag2")
    logger.setLevel(numeric_level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Name of the module

    Returns:
        Logger instance
    """
    return logging.getLogger(f"oncorag2.{name}")