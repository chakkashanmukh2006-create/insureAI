"""
Logging configuration for the Insurance AI system.

Provides a pre-configured logger with stdout streaming and
consistent formatting across all modules.
"""

import logging
import sys


def setup_logger(name: str = "insurance_ai") -> logging.Logger:
    """
    Create and configure a logger instance.

    Sets up a logger with INFO level, stdout stream handler,
    and a consistent timestamp-name-level-message format.

    Args:
        name: The logger name. Defaults to 'insurance_ai'.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Module-level logger instance for convenient import
logger = setup_logger()
