"""A synthetic dataset frozen to disk: the exact paths an agent trains and is scored on.

A ``SynthSpec`` names a generator, the market it runs in, the split sizes and a seed.
``build_dataset`` generates every path once, for the signal world and its null twin, and
writes ``paths.npz`` plus a ``dataset.json`` manifest carrying the spec, the path seeds and the
npz digest, under ``<root>/synth/datasets/<hash>/``. The hash is the spec's content
hash, so the same spec is always the same directory and is never rebuilt.

``load_dataset`` refuses a directory whose npz no longer matches its manifest: the
point of freezing is that "the data the agent saw" is a checkable claim.

Path seeds: train ``seed .. seed+n_train-1``, held-out the next ``n_heldout``. Both
splits are drawn from the same generator but share no path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from optspread.alpha.generators import FloatArray, GeneratorParams, generate, twin_of
from optspread.config import MarketConfig, config_hash
from optspread.manifest import file_sha256, git_sha
from optspread.rng import SeededStepRNG

Split = Literal["train", "heldout"]
World = Literal["signal", "twin"]
SPLITS: tuple[Split, ...] = ("train", "heldout")
WORLDS: tuple[World, ...] = ("signal", "twin")
FIELDS = ("spot", "iv", "signal")
DATASETS_DIR = Path("alpha") / "datasets"
# Not "manifest.json": that name marks a run directory, and a dataset is not a run.
PATHS_FILE, MANIFEST_FILE = "paths.npz", "dataset.json"
_HASH = re.compile(r"^[0-9a-f]{16}$")


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
        start = self.seed + self.n_train
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


def build_dataset(spec: SynthSpec, root: Path) -> SynthDataset:
    """Generate and write the dataset, or return the one already on disk for this spec."""
    h = spec_hash(spec)
    d = _dir(root, h)
    if (d / MANIFEST_FILE).is_file():
        try:
            return load_dataset(root, h)
        except DatasetError:
            pass  # a damaged directory is rebuilt from its spec
    arrays = generate_arrays(spec)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{PATHS_FILE}.tmp.npz"
    np.savez(tmp, **arrays)  # type: ignore[arg-type]
    tmp.replace(d / PATHS_FILE)
    manifest: dict[str, Any] = {
        "hash": h,
        "spec": spec.model_dump(mode="json"),
        "seeds": {s: [spec.seeds(s).start, spec.seeds(s).stop] for s in SPLITS},
        "paths_sha256": file_sha256(d / PATHS_FILE),
        "git_sha": git_sha(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (d / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), newline="\n")
    return SynthDataset(spec, h, d, arrays)


def load_dataset(root: Path, h: str) -> SynthDataset:
    d = _dir(root, h)
    try:
        manifest = json.loads((d / MANIFEST_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise DatasetError(f"no dataset {h} under {root / DATASETS_DIR}") from exc
    if file_sha256(d / PATHS_FILE) != manifest.get("paths_sha256"):
        raise DatasetError(f"dataset {h}: {PATHS_FILE} does not match its manifest digest")
    spec = SynthSpec.model_validate(manifest["spec"])
    if spec_hash(spec) != h:
        raise DatasetError(f"dataset {h}: its manifest spec hashes to {spec_hash(spec)}")
    with np.load(d / PATHS_FILE) as npz:
        arrays = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files}
    return SynthDataset(spec, h, d, arrays)


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
