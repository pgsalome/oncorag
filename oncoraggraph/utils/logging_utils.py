import logging
from loguru import logger

# Configure third-party logging noise suppression
logger.remove()
logging.getLogger("PyRuSH").setLevel(logging.CRITICAL)
logging.getLogger("PyRuSH.PyRuSHSentencizer").setLevel(logging.CRITICAL)
logging.getLogger("medspacy").setLevel(logging.CRITICAL)
logging.getLogger("spacy").setLevel(logging.CRITICAL)
logger = logging.getLogger("sentence_transformers")
logger.handlers.clear()
logger.setLevel(logging.CRITICAL)

_DEBUG_MODE = False
_QUIET_MODE = False
_PREFIXES = {
    "HEADER": "=" * 70,
    "SUBHEADER": "-" * 70,
    "STEP": "→",
    "SUCCESS": "✓",
    "WARNING": "⚠",
    "ERROR": "✗",
    "INFO": "ℹ",
}


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable verbose debug logging."""
    global _DEBUG_MODE
    _DEBUG_MODE = enabled


def is_debug_mode() -> bool:
    """Return whether debug logging is enabled."""
    return _DEBUG_MODE


def set_quiet_mode(enabled: bool) -> None:
    """Silence informational logs when enabled."""
    global _QUIET_MODE
    _QUIET_MODE = enabled


def log(message: str, level: str = "INFO", *, debug: bool = False) -> None:
    """
    Print a formatted log message.

    Args:
        message: The message to log.
        level: Logging level key for prefix selection.
        debug: If True, only emit when debug mode is enabled.
    """
    if debug and not _DEBUG_MODE:
        return

    if _QUIET_MODE and level not in {"WARNING", "ERROR"}:
        return

    prefix = _PREFIXES.get(level, "")
    if level in ("HEADER", "SUBHEADER"):
        print(f"\n{prefix}\n{message}\n{prefix}")
    else:
        print(f"{prefix} {message}")
