"""The variance-components pilot: the grid at zero and one premium, and its matrices.

Lane: Ilan Danial (machinery).

The pilot is the grid of ``optspread.experiment.grid`` at zero and one premium, for the
learner and its scripted control. Its S x N contrast matrices are what
``scripts/pilot_budget.py`` reads, written beside the cells with a manifest of their own.
Moved from ``optspread.orchestrate``, which re-exports it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from optspread.analysis import NULL_PREMIUM
from optspread.config import ArmName, RunConfig, TrainConfig
from optspread.data import SourceName, make_source
from optspread.experiment.grid import GridConfig, GridRecord, block_config
from optspread.manifest import DataRecord, RunManifest

# -- the pilot ------------------------------------------------------------------ #

# The detection event's contrasts at the pilot premium (each arm at p minus itself at 0).
PILOT_MATRIX = "matrix.npy"  # the learner: what scripts/pilot_budget.py --matrix reads
PILOT_ORACLE_MATRIX = "matrix_oracle.npy"  # the negative control, tau^2 = 0 by construction


def pilot_grid(
    base: RunConfig,
    n_seeds: int,
    n_episodes: int,
    premium: float,
    train: TrainConfig,
) -> GridConfig:
    """The variance-components pilot: the learner and its scripted control, each at zero
    and at ``premium``, which is what the detection event's contrast needs (#26)."""
    if premium == NULL_PREMIUM:
        raise ValueError(
            "the pilot measures the contrast against zero premium, so its premium must "
            f"differ from {NULL_PREMIUM:g}"
        )
    return GridConfig(
        base=base.model_copy(update={"n_episodes": n_episodes}),
        premiums=(NULL_PREMIUM, premium),
        arms=("ppo", "delta_seller"),
        n_seeds=n_seeds,
        train=train,
    )


def write_pilot_matrices(
    record: GridRecord,
    root: Path,
    *,
    source: SourceName = "synthetic",
) -> dict[ArmName, Path]:
    """Save each arm's ``(S, N)`` contrast at the pilot's premium beside its cells.

    The contrast is the detection event's: the arm at the pilot premium minus the arm
    at zero (``GridRecord.contrast``). Each arm's raw matrix in each world is saved
    too, as ``matrix_<arm>_p<premium>.npy``. The matrices get a manifest of their own
    at ``root``, keyed by the learned arm's first block config at the pilot premium,
    recording every matrix's digest, so the M2 report can refuse a matrix that
    changed after it was written, and the trainer the learned arm's cells record. Its
    data record is the cells' (``source``, on its ``eval`` split).
    """
    premium = next(p for p in record.grid.premiums if p != NULL_PREMIUM)
    names: dict[ArmName, str] = {"ppo": PILOT_MATRIX, "delta_seller": PILOT_ORACLE_MATRIX}
    paths: dict[ArmName, Path] = {}
    raw: list[Path] = []
    for arm in record.grid.arms:
        path = root / names.get(arm, f"matrix_{arm}.npy")
        np.save(path, record.contrast(root, premium, arm))
        paths[arm] = path
        for p in record.grid.premiums:
            raw.append(root / f"matrix_{arm}_p{p:g}.npy")
            np.save(raw[-1], record.matrix(root, p, arm))
    lead: ArmName = "ppo" if "ppo" in record.grid.arms else record.grid.arms[0]
    # The learned arm's trainer, when every one of its cells records the same one.
    trainers = {RunManifest.read(root / c.run_id).trainer for c in record.cells if c.arm == "ppo"}
    trainer = trainers.pop() if len(trainers) == 1 else None
    data = DataRecord.of(make_source(source, record.grid.base.market, split="eval"))
    manifest = RunManifest.create(
        block_config(record.grid, premium, lead, 0), data=data, trainer=trainer
    )
    for path in [*paths.values(), *raw]:
        manifest.add_artifact(root, path)
    manifest.write(root)
    return paths
