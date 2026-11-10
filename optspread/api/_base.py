"""Shared pieces of the API payloads: the frozen base model and the JSON-safe conversions.

Lane: Ilan Danial (machinery).

NaN is not JSON, so a figure that is undefined is sent as null (``_f``, ``_fl``,
``_finite_json`` for a free-form dict). ``_inside`` keeps a key that came from the browser
inside the root it names a path under; the run and the sweep payloads both use it.
``_source_suffix`` is the part of a cached file's name that says which data a run read.
Moved from ``optspread.webapi`` and ``optspread.webviews``, which re-export it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


def _f(x: float | None) -> float | None:
    """A float for JSON: NaN and infinities become null."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fl(values: Iterable[float]) -> list[float | None]:
    return [_f(v) for v in values]


def _iso(values: NDArray[np.datetime64]) -> list[str]:
    return [str(d) for d in values.astype("datetime64[D]")]


class Out(BaseModel):
    model_config = ConfigDict(frozen=True)


def _inside(root: Path, key: str) -> Path | None:
    """``root / key`` resolved, or None when it leaves the root or is not a path at all.

    Judged on the resolved forms, but returned under ``root`` as the caller spells it
    (relative, a short name, a link), so a path made from it is relative to that root as
    every path read from the root's own listing is."""
    try:
        resolved_root = root.resolve()
        target = (root / key).resolve()
        if not target.is_relative_to(resolved_root):
            return None
        return root / target.relative_to(resolved_root)
    except (OSError, ValueError):  # a NUL byte, or a name the filesystem refuses
        return None


def _source_suffix(source: str) -> str:
    """``""`` for the synthetic world (or a run recorded before sources were), else
    ``.<source>``: what a cached file's name adds after the run id or grid hash, so a
    real-data run's file never stands in for the synthetic one's."""
    return "" if source in ("", "synthetic") else f".{source}"


def _finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    return value
