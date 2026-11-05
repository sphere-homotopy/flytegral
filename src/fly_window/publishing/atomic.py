"""Atomic file replacement tolerant of short-lived Windows sharing conflicts."""
from __future__ import annotations

from pathlib import Path
import sys
import time


_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
_WINDOWS_SHARING_ERRORS = frozenset((5, 32, 33))


def replace_with_retry(source: Path, target: Path) -> None:
    """Replace atomically, waiting at most 3.15 seconds for Windows readers.

    Never unlink the destination: a persistent error leaves its old contents
    intact and propagates to the caller. Other errors and platforms fail fast.
    """
    for attempt in range(len(_RETRY_DELAYS_SECONDS) + 1):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            if (
                sys.platform != "win32"
                or getattr(error, "winerror", None) not in _WINDOWS_SHARING_ERRORS
                or attempt == len(_RETRY_DELAYS_SECONDS)
            ):
                raise
            time.sleep(_RETRY_DELAYS_SECONDS[attempt])
