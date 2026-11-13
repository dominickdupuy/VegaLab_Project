"""The landing page's text: the README's opening paragraphs.

Lane: Ilan Danial (machinery). Moved from ``optspread.dashboard_data``, which
re-exports it.
"""

from __future__ import annotations

import re
from pathlib import Path

# -- the landing page --------------------------------------------------------- #


def readme_summary(readme: Path) -> tuple[str, ...]:
    """The README's opening paragraphs, before its first section heading.

    Sentences that point at the README or the project document are dropped: on the
    dashboard they would refer to a document the reader is not looking at.
    """
    if not readme.is_file():
        return ()
    paragraphs: list[str] = []
    current: list[str] = []
    for line in readme.read_text().splitlines():
        if line.startswith("## "):
            break
        if line.startswith("# "):
            continue
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    out: list[str] = []
    for para in paragraphs:
        sentences = re.split(r"(?<=\.)\s+", para)
        kept = [s for s in sentences if "README" not in s and "project document" not in s]
        if kept:
            out.append(" ".join(kept))
    return tuple(out)
