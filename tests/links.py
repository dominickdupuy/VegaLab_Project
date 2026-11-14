"""A link a test can make on any machine: a symlink, or its nearest stand-in on Windows.

Windows refuses ``os.symlink`` to an account without the symlink privilege (WinError
1314) unless Developer Mode is on, which a partner team's machine need not have (#10).
There a directory is linked by a junction and a file by a hard link; neither needs a
privilege. Both are a second path to the same bytes, which is all these tests ask of a
link: the data audit knows the team panel by its content, not by how a path reaches it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def link(dst: Path, src: Path) -> None:
    """Make ``dst`` a link to ``src``: a symlink where the OS allows one."""
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if sys.platform != "win32":
            raise
        if src.is_dir():
            import _winapi

            _winapi.CreateJunction(str(src), str(dst))
        else:
            os.link(src, dst)
