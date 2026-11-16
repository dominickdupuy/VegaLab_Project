"""A synthetic dataset frozen to disk: the exact paths an agent trains and is scored on.

A ``SynthSpec`` names a generator, the market it runs in, the split sizes and a seed.
``build_dataset`` generates every path once, for the signal world and its null twin, and
writes ``paths.npz`` plus a ``dataset.json`` manifest carrying the spec, the path seeds and the
npz digest, under ``<root>/synth/datasets/<hash>/``. The hash is the spec's content
hash, so the same spec is always the same directory and is never rebuilt.

``load_dataset`` refuses a directory whose npz no longer matches its manifest: the
point of freezing is that "the data the agent saw" is a checkable claim. ``check_dataset``
makes the same check without reading the paths, for a caller that only names the dataset,
and digests the npz once per version of the file.

Two builds of one spec at once (two browser tabs, or the server and a sweep's worker) are
safe: a build holds the directory's lock while it writes, the second finds the first's
dataset when it gets the lock, and each file is written to a temporary name of its own
and moved into place, so no reader sees half a file.

Path seeds: train ``seed .. seed+n_train-1``, held-out ``seed+HELDOUT_OFFSET ..``. Both
splits are drawn from the same generator but share no path, and the held-out paths do not
depend on ``n_train``: datasets that differ only in their number of train paths are scored
on the same held-out paths, and the smaller train set is the start of the larger one, which
is what a sweep over the number of train paths needs.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from filelock import BaseFileLock, Timeout, UnixFileLock, WindowsFileLock
from pydantic import BaseModel, ConfigDict, Field

from optspread.alpha.generators import FloatArray, GeneratorParams, generate, twin_of
from optspread.config import MarketConfig, config_hash
from optspread.fileio import atomic_write_bytes, atomic_write_text
from optspread.manifest import file_sha256, git_sha
from optspread.manifest import panel_file_sha256 as cached_sha256
from optspread.rng import SeededStepRNG

Split = Literal["train", "heldout"]
World = Literal["signal", "twin"]
SPLITS: tuple[Split, ...] = ("train", "heldout")
WORLDS: tuple[World, ...] = ("signal", "twin")
FIELDS = ("spot", "iv", "signal")
DATASETS_DIR = Path("alpha") / "datasets"
# Not "manifest.json": that name marks a run directory, and a dataset is not a run.
PATHS_FILE, MANIFEST_FILE = "paths.npz", "dataset.json"
HELDOUT_OFFSET = 2**20  # past the largest train split (n_train <= 2000)
_HASH = re.compile(r"^[0-9a-f]{16}$")
BUILD_LOCK = ".build.lock"  # beside the files; held while a build writes them
BUILD_LOCK_TIMEOUT = 300.0  # seconds; the largest build allowed takes about 20
# The operating system's lock, as ``optspread.runs.rundir.root_lock`` takes it (no soft
# fallback, which would take a leftover lock file for a live holder).
_HARD_LOCK: type[BaseFileLock] = WindowsFileLock if sys.platform == "win32" else UnixFileLock


class DatasetError(ValueError):
    """A dataset that is absent, malformed, or no longer matches its manifest."""


class SynthSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generator: GeneratorParams
    sigma: float = Field(default=0.20, gt=0.0, le=1.5)
    n_days: int = Field(default=63, ge=5, le=252)
    n_train: int = Field(default=64, ge=1, le=2000)
    n_heldout: int = Field(default=256, ge=2, le=5000)
    seed: int = Field(default=0, ge=0, lt=2**30)

    def market(self) -> MarketConfig:
        return MarketConfig(sigma=self.sigma, n_days=self.n_days)

    def seeds(self, split: Split) -> range:
        if split == "train":
            return range(self.seed, self.seed + self.n_train)
        start = self.seed + HELDOUT_OFFSET
        return range(start, start + self.n_heldout)


def spec_hash(spec: SynthSpec) -> str:
    return config_hash(spec)


def _key(world: World, split: Split, field: str) -> str:
    return f"{world}_{split}_{field}"


@dataclass(frozen=True, slots=True)
class SynthDataset:
    spec: SynthSpec
    hash: str
    dir: Path
    arrays: dict[str, FloatArray]

    def _get(self, world: World, split: Split, field: str) -> FloatArray:
        return self.arrays[_key(world, split, field)]

    def spot(self, world: World, split: Split) -> FloatArray:
        return self._get(world, split, "spot")

    def iv(self, world: World, split: Split) -> FloatArray:
        return self._get(world, split, "iv")

    def signal(self, world: World, split: Split) -> FloatArray:
        return self._get(world, split, "signal")

    def n_paths(self, split: Split) -> int:
        return int(self.spot("signal", split).shape[0])


def generate_arrays(
    spec: SynthSpec, splits: dict[Split, range] | None = None
) -> dict[str, FloatArray]:
    """Every path of every split in both worlds, as ``(n_paths, n_days+1)`` arrays."""
    market = spec.market()
    out: dict[str, FloatArray] = {}
    for split, seeds in (splits or {s: spec.seeds(s) for s in SPLITS}).items():
        worlds: tuple[tuple[World, GeneratorParams], ...] = (
            ("signal", spec.generator),
            ("twin", twin_of(spec.generator)),
        )
        for world, params in worlds:
            paths = [generate(params, market, SeededStepRNG(s)) for s in seeds]
            for field in FIELDS:
                out[_key(world, split, field)] = np.stack([getattr(p, field) for p in paths])
    return out


def _dir(root: Path, h: str) -> Path:
    if not _HASH.match(h):
        raise DatasetError(f"{h!r} is not a dataset hash (16 hex characters)")
    return root / DATASETS_DIR / h


@contextmanager
def _build_lock(d: Path) -> Iterator[None]:
    """Hold the dataset directory's lock for the body. Where it cannot be taken (no kernel
    locks, or another build holds it past the timeout) the body runs without it: every
    file is still written whole and moved into place, so the worst case is the same
    dataset written twice."""
    lock = _HARD_LOCK(d / BUILD_LOCK, timeout=BUILD_LOCK_TIMEOUT, fallback_to_soft=False)
    try:
        lock.acquire()
    except (Timeout, OSError, NotImplementedError):
        held = False
    else:
        held = True
    try:
        yield
    finally:
        if held:
            lock.release()


def _on_disk(root: Path, h: str) -> SynthDataset | None:
    if not (_dir(root, h) / MANIFEST_FILE).is_file():
        return None
    try:
        return load_dataset(root, h)
    except DatasetError:
        return None  # a damaged directory is rebuilt from its spec


def build_dataset(spec: SynthSpec, root: Path) -> SynthDataset:
    """Generate and write the dataset, or return the one already on disk for this spec."""
    h = spec_hash(spec)
    d = _dir(root, h)
    if (found := _on_disk(root, h)) is not None:
        return found
    d.mkdir(parents=True, exist_ok=True)
    with _build_lock(d):
        if (found := _on_disk(root, h)) is not None:
            return found  # another build of this spec wrote it while this one waited
        arrays = generate_arrays(spec)
        buf = io.BytesIO()
        np.savez(buf, **arrays)  # type: ignore[arg-type]
        data = buf.getvalue()
        atomic_write_bytes(d / PATHS_FILE, data)
        manifest: dict[str, Any] = {
            "hash": h,
            "spec": spec.model_dump(mode="json"),
            "seeds": {s: [spec.seeds(s).start, spec.seeds(s).stop] for s in SPLITS},
            "paths_sha256": hashlib.sha256(data).hexdigest(),
            "git_sha": git_sha(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        # the manifest last: a dataset exists once its manifest does
        text = json.dumps(manifest, indent=2, sort_keys=True)
        atomic_write_text(d / MANIFEST_FILE, text, newline="\n")
    return SynthDataset(spec, h, d, arrays)


def _manifest(root: Path, h: str) -> tuple[Path, dict[str, Any], SynthSpec]:
    d = _dir(root, h)
    try:
        manifest = json.loads((d / MANIFEST_FILE).read_text())
        spec = SynthSpec.model_validate(manifest["spec"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetError(f"no dataset {h} under {root / DATASETS_DIR}") from exc
    if spec_hash(spec) != h:
        raise DatasetError(f"dataset {h}: its manifest spec hashes to {spec_hash(spec)}")
    return d, manifest, spec


def _check_digest(d: Path, h: str, manifest: dict[str, Any], digest: Callable[[Path], str]) -> None:
    try:
        actual = digest(d / PATHS_FILE)
    except OSError as exc:
        raise DatasetError(f"dataset {h}: {PATHS_FILE} cannot be read") from exc
    if actual != manifest.get("paths_sha256"):
        raise DatasetError(f"dataset {h}: {PATHS_FILE} does not match its manifest digest")


def load_dataset(root: Path, h: str) -> SynthDataset:
    """The dataset with its paths, after digesting the npz in full against its manifest."""
    d, manifest, spec = _manifest(root, h)
    _check_digest(d, h, manifest, file_sha256)
    with np.load(d / PATHS_FILE) as npz:
        arrays = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files}
    return SynthDataset(spec, h, d, arrays)


def check_dataset(root: Path, h: str) -> SynthSpec:
    """The dataset's spec, checked as ``load_dataset`` checks it, without reading its paths.

    The npz digest is computed once per version of the file (same inode, size and
    modification time) and reused after, so a route that only names the dataset does
    not hash it on every call; whatever trains or scores on the paths reads them through
    ``load_dataset``, which digests them in full.
    """
    d, manifest, spec = _manifest(root, h)
    _check_digest(d, h, manifest, cached_sha256)
    return spec


def list_datasets(root: Path) -> list[dict[str, Any]]:
    """Manifests of every dataset on disk, newest first."""
    base = root / DATASETS_DIR
    out = []
    for m in base.glob(f"*/{MANIFEST_FILE}") if base.is_dir() else []:
        try:
            out.append(json.loads(m.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda m: str(m.get("created_at", "")), reverse=True)


def preview(ds: SynthDataset, n: int = 20) -> dict[str, Any]:
    """The first ``n`` held-out paths in both worlds, for plotting."""

    def world(w: World) -> dict[str, list[list[float]]]:
        return {f: ds._get(w, "heldout", f)[:n].round(6).tolist() for f in FIELDS}

    return {
        "days": list(range(ds.spec.n_days + 1)),
        "signal": world("signal"),
        "twin": world("twin"),
    }
