"""Windows-safe file replacement.

``os.replace`` onto a file another process has open (the web server polling status.json, an
antivirus scan, Explorer's preview) fails on Windows with "Access is denied". Those holds last
milliseconds, so a short retry is enough.
"""

import os
import time


def replace(src: str, dst: str, tries: int = 40, wait: float = 0.05, required: bool = True) -> bool:
    """``os.replace`` with retries. If it still fails: raise when ``required``, otherwise drop
    ``src`` and return False (for files that are refreshed again moments later)."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            if i == tries - 1:
                if required:
                    raise
                try:
                    os.remove(src)
                except OSError:
                    pass
                return False
            time.sleep(wait * (1 + i // 10))
    return False
