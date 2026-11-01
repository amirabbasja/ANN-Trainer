import os
import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration for the project using loguru.

    Args:
        level: Logging level (INFO, DEBUG, WARNING, ERROR, etc.)
        format_string: Custom format string for log messages
    """
    # Remove default handler
    logger.remove()

    format_string = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - "
        "<level>{level}</level> - "
        "<white>{message}</white> - "
        "<blue>{file}:{line}</blue>"
    )

    log_level = os.environ.get("LRAEP_LOG_LEVEL", "INFO")

    # Add custom handler with specified level and format
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Makes it thread-safe and can help with color output
    )


def get_logger(name: str):
    """
    Get a logger instance for the given name.
    With loguru, we return the logger bound with the module name.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        Logger instance bound with the module name
    """
    return logger.bind(name=name)


# Initialize logging on import
should_log = os.environ.get("LRAEP_LOG_LEVEL", "INFO").upper() != "NONE"
if should_log:
    setup_logging()
