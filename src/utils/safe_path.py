"""Safe path construction utilities to prevent path traversal attacks."""

import logging
from pathlib import Path

from werkzeug.security import safe_join

logger = logging.getLogger(__name__)


def safe_path_join(base_dir: str, filename: str) -> Path:
    """Safely join a base directory with a filename, preventing path traversal.

    Args:
        base_dir: The trusted base directory.
        filename: The untrusted filename to join (e.g. from parsed annotation files).

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If the filename would escape the base directory.
    """
    result = safe_join(str(base_dir), filename)
    if result is None:
        logger.error(
            "Security Alert: Path traversal attempt blocked — base=%s, filename=%s",
            base_dir,
            filename,
        )
        raise ValueError(f"Invalid filename: {filename}")
    return Path(result).resolve()
