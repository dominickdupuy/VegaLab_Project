"""How much data, or training, does an arm need to take a planted edge without overfitting?

Lane: Dominick Dupuy.

A sweep runs one benchmark per value on one axis and keeps each one's headline figures:

* ``n_train``: the number of frozen train paths. Each value is its own dataset, built from
  the open one with only ``n_train`` changed, so every point is scored on the same held-out
  paths and each train set is the start of the next (``optspread.alpha.dataset``). The
  oracle is chosen once for all of them (``benchmark.choose_oracle``).
* ``total_timesteps``: the training budget, on the open dataset.

Overfitting shows as an overfit gap and a noise fit that shrink as ``n_train`` grows, and a
capture ratio that rises to meet them. Every point is an ordinary benchmark, written and
reused like one, so a sweep that is run again recomputes nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optspread.alpha.benchmark import (
    MAX_SEEDS,
    Arm,
    BenchStat,
    SynthTrainer,
    learned_trainer,
    run_benchmark,
    stages,
)
from optspread.alpha.dataset import build_dataset, load_dataset
from optspread.config import LEARNED_ARMS, config_hash

SweepAxis = Literal["n_train", "total_timesteps"]
SWEEP_DIR = Path("alpha") / "sweeps"
SWEEP_FILE = "sweep.json"
MIN_TIMESTEPS = 2048


class AlphaSweepSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(pattern=r"^[0-9a-f]{16}$")  # the open dataset: every other setting
    arm: Arm = "ppo"
    axis: SweepAxis = "n_train"
    values: tuple[int, ...] = Field(min_length=2, max_length=8)
    total_timesteps: int | None = Field(default=None, ge=MIN_TIMESTEPS, le=5_000_000)
    n_seeds: int = Field(default=1, ge=1, le=MAX_SEEDS)

    @model_validator(mode="after")
    def _consistent(self) -> AlphaSweepSpec:
        values = list(self.values)
        if values != sorted(set(values)):
            raise ValueError("values must be distinct and increasing")
        learned = self.arm in LEARNED_ARMS
        if self.axis == "total_timesteps":
            if not learned:
                raise ValueError("a sweep over the training budget needs a learned arm")
            if self.total_timesteps is not None:
                raise ValueError("a sweep over the training budget takes its budget from values")
            if values[0] < MIN_TIMESTEPS or values[-1] > 5_000_000:
                raise ValueError(f"training budgets must lie in [{MIN_TIMESTEPS}, 5,000,000]")
        else:
            if learned != (self.total_timesteps is not None):
                raise ValueError("total_timesteps is set exactly when the arm is learned")
            if values[0] < 1 or values[-1] > 2000:
                raise ValueError("train path counts must lie in [1, 2000]")
        if not learned and self.n_seeds != 1:
            raise ValueError("a scripted arm is deterministic: n_seeds is 1")
        return self

    def budget(self, value: int) -> int | None:
        return value if self.axis == "total_timesteps" else self.total_timesteps

    @property
    def stages(self) -> int:
        return len(self.values) * (stages(self.arm, self.n_seeds) + 1)  # + its own line


class AlphaSweepPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: int
    dataset: str
    n_train: int
    total_timesteps: int | None
    attainable: BenchStat
    captured: BenchStat
    capture_ratio: BenchStat
    overfit_gap: BenchStat
    noise_fit: BenchStat | None


class AlphaSweep(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    spec: AlphaSweepSpec
    generator: str
    datasets: tuple[str, ...]
    points: tuple[AlphaSweepPoint, ...]


def sweep_path(root: Path, spec: AlphaSweepSpec) -> Path:
    return root / SWEEP_DIR / config_hash(spec) / SWEEP_FILE


def load_sweep(path: Path) -> AlphaSweep:
    return AlphaSweep.model_validate_json(path.read_text())


def list_sweeps(root: Path, dataset: str) -> list[AlphaSweep]:
    """Every sweep that ran on ``dataset`` (as its base or one of its points), newest first."""
    files = sorted(
        (root / SWEEP_DIR).glob(f"*/{SWEEP_FILE}"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    out = []
    for f in files:
        try:
            sweep = load_sweep(f)
        except (OSError, ValueError):
            continue
        if dataset in sweep.datasets:
            out.append(sweep)
    return out


def run_sweep(
    spec: AlphaSweepSpec,
    root: Path,
    trainer: SynthTrainer = learned_trainer,
    log: Callable[[str], None] | None = None,
) -> AlphaSweep:
    """One benchmark per value, each written and reused as a benchmark; then ``sweep.json``."""
    path = sweep_path(root, spec)
    if path.is_file():
        return load_sweep(path)
    say = log if log is not None else (lambda _: None)
    base = load_dataset(root, spec.dataset)
    points = []
    for value in spec.values:
        ds = (
            build_dataset(base.spec.model_copy(update={"n_train": value}), root)
            if spec.axis == "n_train"
            else base
        )
        budget = spec.budget(value)
        say(f"point {spec.axis} = {value:,}: dataset {ds.hash}")
        b = run_benchmark(ds, spec.arm, budget, root, trainer, log, spec.n_seeds)
        points.append(
            AlphaSweepPoint(
                value=value,
                dataset=ds.hash,
                n_train=ds.n_paths("train"),
                total_timesteps=budget,
                attainable=b.attainable,
                captured=b.captured,
                capture_ratio=b.capture_ratio,
                overfit_gap=b.overfit_gap,
                noise_fit=b.noise_fit,
            )
        )
    datasets = tuple(dict.fromkeys([base.hash, *(p.dataset for p in points)]))
    sweep = AlphaSweep(
        key=config_hash(spec),
        spec=spec,
        generator=base.spec.generator.kind,
        datasets=datasets,
        points=tuple(points),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(sweep.model_dump_json(indent=2), newline="\n")
    tmp.replace(path)
    return sweep
