"""How much data, or training, does an arm need to take a planted edge without overfitting?

Lane: Dominick Dupuy.

A sweep runs one benchmark per value on one axis and keeps each one's headline figures:

* ``n_train``: the number of frozen train paths. Each value is its own dataset, built from
  the open one with only ``n_train`` changed, so every point is scored on the same held-out
  paths and each train set is the start of the next (``optspread.alpha.dataset``). The
  oracle is chosen once for all of them (``benchmark.choose_oracle``).
* ``total_timesteps``: the training budget, on the open dataset.
* ``edge``: the planted edge's size (``EDGE_KNOB``: the premium of ``const_vrp``, the
  premium's s.d. of ``mr_vrp``, the drift of ``drift``), zero for the null. Each value is
  its own dataset, the open one with only that knob changed, on the same path seeds, so
  every size shares its spot shocks, its twin and its held-out paths. ``check_edge``
  refuses, before anything runs, a size outside the knob's range and an ``mr_vrp`` base
  whose ``mean_pts`` is not 0: its premium s.d. at 0 still leaves that constant premium,
  so size 0 would not be the null the floor's curve is drawn through. Each size's captured
  edge is tested per seed and the sizes give a power curve and the arm's detection floor
  (``optspread.alpha.floor``, the study's machinery). A floor needs two or more training
  seeds for a learner; a deterministic arm is read on path blocks.

Overfitting shows as an overfit gap and a noise fit that shrink as ``n_train`` grows, and a
capture ratio that rises to meet them. Every point is an ordinary benchmark, written and
reused like one, so a sweep that is run again recomputes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from optspread.alpha.benchmark import (
    MAX_SEEDS,
    Arm,
    Benchmark,
    BenchStat,
    SynthTrainer,
    check_arm,
    is_learned,
    learned_trainer,
    run_benchmark,
    stages,
)
from optspread.alpha.dataset import SynthDataset, SynthSpec, build_dataset, load_dataset
from optspread.alpha.floor import EdgeFloor, EdgeTest, Unit, edge_floor, seed_test
from optspread.alpha.generators import GeneratorParams, MrVrpParams
from optspread.alpha.plugins import NAME_PATTERN
from optspread.alpha.plugins import find as find_plugin
from optspread.config import config_hash
from optspread.fileio import atomic_write_text

SweepAxis = Literal["n_train", "total_timesteps", "edge"]
STORED = {"stored": True}  # validation context of a result read back from disk
# The knob an edge-size sweep moves, per generator, and its unit in words. ``noise`` has
# no edge and ``spot_mr``'s half-life is no size (it is the null only at infinity).
EDGE_KNOB: dict[str, tuple[str, str]] = {
    "const_vrp": ("premium_pts", "vol points"),
    "mr_vrp": ("vol_pts", "vol points of premium s.d."),
    "drift": ("bps_per_day", "bps a day"),
}
SWEEP_DIR = Path("alpha") / "sweeps"
SWEEP_FILE = "sweep.json"
MIN_TIMESTEPS = 2048


def _stored(info: ValidationInfo) -> bool:
    return isinstance(info.context, dict) and bool(info.context.get("stored"))


class AlphaSweepSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(pattern=r"^[0-9a-f]{16}$")  # the open dataset: every other setting
    arm: Arm = Field(default="ppo", pattern=NAME_PATTERN)
    axis: SweepAxis = "n_train"
    # Integers on the train-size and budget axes; edge sizes may be fractional. A union, so
    # an integer stays one and a sweep written before ``edge`` keeps its hash.
    values: tuple[int | float, ...] = Field(min_length=2, max_length=8)
    total_timesteps: int | None = Field(default=None, ge=MIN_TIMESTEPS, le=5_000_000)
    n_seeds: int = Field(default=1, ge=1, le=MAX_SEEDS)

    @field_validator("arm")
    @classmethod
    def _known(cls, arm: str, info: ValidationInfo) -> str:
        # A sweep read back from disk keeps its arm even if that plug-in is gone now.
        return arm if _stored(info) else check_arm(arm)

    @field_validator("values")
    @classmethod
    def _integral(
        cls, values: tuple[int | float, ...], info: ValidationInfo
    ) -> tuple[int | float, ...]:
        if info.data.get("axis") == "edge":
            return values
        if not all(float(v).is_integer() for v in values):
            raise ValueError("train path counts and training budgets are whole numbers")
        return tuple(int(v) for v in values)

    @model_validator(mode="after")
    def _consistent(self, info: ValidationInfo) -> AlphaSweepSpec:
        if _stored(info):
            return self
        values = list(self.values)
        if values != sorted(set(values)):
            raise ValueError("values must be distinct and increasing")
        learned = is_learned(self.arm)
        if self.axis == "edge":
            if learned != (self.total_timesteps is not None):
                raise ValueError("total_timesteps is set exactly when the arm is learned")
            if values[0] < 0:
                raise ValueError("edge sizes are zero (the null) or positive")
            if values[-1] <= 0:
                raise ValueError("an edge-size sweep needs at least one positive size")
            if learned and self.n_seeds < 2:
                raise ValueError(
                    "a learner's detection floor needs at least 2 training seeds per size"
                )
        elif self.axis == "total_timesteps":
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

    def budget(self, value: int | float) -> int | None:
        return int(value) if self.axis == "total_timesteps" else self.total_timesteps

    @property
    def stages(self) -> int:
        per = len(self.values) * (stages(self.arm, self.n_seeds) + 1)  # + its own line
        return per + (self.axis == "edge")  # + the floor's


class AlphaSweepPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: int | float
    dataset: str
    n_train: int
    total_timesteps: int | None
    attainable: BenchStat
    captured: BenchStat
    capture_ratio: BenchStat
    overfit_gap: BenchStat
    noise_fit: BenchStat | None
    # edge sweeps only: the figures the detection test read, and its result
    figures: tuple[float, ...] = Field(default_factory=tuple)
    test: EdgeTest | None = None


class AlphaSweep(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    spec: AlphaSweepSpec
    generator: str
    datasets: tuple[str, ...]
    points: tuple[AlphaSweepPoint, ...]
    # edge sweeps only: the knob moved, its unit, and the arm's detection floor
    edge_knob: str | None = None
    edge_unit: str | None = None
    floor: EdgeFloor | None = None


def edge_knob(generator: GeneratorParams) -> tuple[str, str]:
    """The knob an edge-size sweep moves on ``generator``, and its unit; ValueError if none."""
    if generator.kind not in EDGE_KNOB:
        raise ValueError(
            f"the {generator.kind} generator has no edge size to sweep; an edge-size sweep "
            f"runs on {', '.join(EDGE_KNOB)}"
        )
    return EDGE_KNOB[generator.kind]


def zero_is_null(generator: GeneratorParams) -> bool:
    """Whether ``generator`` with its edge knob at 0 plants nothing (is its own twin):
    ``mr_vrp`` only with ``mean_pts`` 0, the other knobs always."""
    return not isinstance(generator, MrVrpParams) or generator.mean_pts == 0.0


def with_edge(spec: SynthSpec, value: float) -> SynthSpec:
    """``spec`` with only its generator's edge knob set to ``value`` (validated)."""
    knob, _ = edge_knob(spec.generator)
    params: dict[str, Any] = {**spec.generator.model_dump(), knob: value}
    return SynthSpec.model_validate({**spec.model_dump(), "generator": params})


def check_edge(base: SynthSpec, values: Sequence[float]) -> tuple[str, str]:
    """The knob an edge-size sweep over ``values`` moves on ``base``, and its unit, after
    checking it can run: a ValueError, saying why, for a generator with no edge size, an
    ``mr_vrp`` base with a nonzero ``mean_pts``, or a size outside the knob's range."""
    knob, unit = edge_knob(base.generator)
    g = base.generator
    if not zero_is_null(g):
        assert isinstance(g, MrVrpParams)
        raise ValueError(
            f"an edge-size sweep on mr_vrp moves vol_pts, which is the null only with mean_pts "
            f"0; this dataset's mean_pts is {g.mean_pts:g}, a premium that stays at every size, "
            "so size 0 would not be the null. Sweep a dataset with mean_pts 0 (or const_vrp's "
            "premium_pts for a constant premium)"
        )
    for v in values:
        try:
            with_edge(base, float(v))
        except ValidationError as exc:
            why = exc.errors()[0]["msg"]
            raise ValueError(f"edge size {v:g} is out of range for {knob}: {why}") from None
    return knob, unit


def detection_figures(b: Benchmark) -> tuple[tuple[float, ...], Unit]:
    """What a size's detection test reads: the per-seed captured edge of a learner, the
    per-block captured edge of a deterministic arm."""
    if b.total_timesteps is not None:
        return tuple(s.captured for s in b.seeds), "seeds"
    return b.blocks, "path blocks"


def sweep_key(spec: AlphaSweepSpec) -> str:
    """The spec's hash, and for a plug-in arm its code digest: edited code, a new sweep."""
    found = find_plugin(spec.arm)
    return config_hash(spec) + (f"-{found.digest}" if found is not None else "")


def sweep_path(root: Path, spec: AlphaSweepSpec) -> Path:
    return root / SWEEP_DIR / sweep_key(spec) / SWEEP_FILE


def load_sweep(path: Path) -> AlphaSweep:
    return AlphaSweep.model_validate_json(path.read_text(), context=STORED)


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
        try:
            return load_sweep(path)
        except (OSError, ValueError):
            pass  # unreadable: run again (each point is a benchmark, reused if it is whole)
    say = log if log is not None else (lambda _: None)
    base = load_dataset(root, spec.dataset)
    knob = check_edge(base.spec, spec.values) if spec.axis == "edge" else None
    points = []
    tests = []
    for value in spec.values:
        ds: SynthDataset
        if spec.axis == "n_train":
            ds = build_dataset(base.spec.model_copy(update={"n_train": int(value)}), root)
        elif spec.axis == "edge":
            ds = build_dataset(with_edge(base.spec, float(value)), root)
        else:
            ds = base
        budget = spec.budget(value)
        say(f"point {spec.axis} = {value:,}: dataset {ds.hash}")
        b = run_benchmark(ds, spec.arm, budget, root, trainer, log, spec.n_seeds)
        figures: tuple[float, ...] = ()
        test = None
        if knob is not None:
            figures, unit = detection_figures(b)
            if len(figures) >= 2:
                test = seed_test(float(value), figures, unit)
                tests.append(test)
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
                figures=figures,
                test=test,
            )
        )
    datasets = tuple(dict.fromkeys([base.hash, *(p.dataset for p in points)]))
    sweep = AlphaSweep(
        key=sweep_key(spec),
        spec=spec,
        generator=base.spec.generator.kind,
        datasets=datasets,
        points=tuple(points),
        edge_knob=knob[0] if knob else None,
        edge_unit=knob[1] if knob else None,
        floor=edge_floor(tests, knob[1], zero_is_null=zero_is_null(base.spec.generator))
        if knob
        else None,
    )
    if sweep.floor is not None:
        say(f"floor: {sweep.floor.reading}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, sweep.model_dump_json(indent=2), newline="\n")
    return sweep
