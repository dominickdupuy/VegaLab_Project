"""Dominick Dupuy: the A6 rule that turns the pilot into the confirmatory S and N (#6).

Every number is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optspread.power import VarianceComponents, power_t, variance_components
from scripts import confirmatory_design as cd


def components(tau: float, sigma: float, seed: int = 0) -> VarianceComponents:
    rng = np.random.default_rng(seed)
    return variance_components(
        rng.normal(0.0, tau, size=(8, 1)) + rng.normal(0.0, sigma, size=(8, 200))
    )


def test_the_design_is_the_cheapest_that_reaches_80_percent_exactly() -> None:
    vc = components(300.0, 4_000.0)
    design, best, _ = cd.choose(500.0, vc, 1_000_000, 1_000.0, 1_800.0)
    assert design.power >= cd.TARGET and design in best
    assert design.core_hours == min(c.core_hours for c in best)
    for c in best:  # each N's S is the smallest that reaches the target
        assert power_t(500.0, vc.tau2, vc.sigma2, c.n_seeds - 1, c.n_episodes) < cd.TARGET


def test_the_fallbacks_are_the_preregistered_ones() -> None:
    vc = components(300.0, 4_000.0)
    no_edge, _, why = cd.choose(-10.0, vc, 1_000_000, 1_000.0, 1_800.0)
    assert (no_edge.n_seeds, no_edge.n_episodes) == cd.FALLBACK_NO_EDGE and "fallback" in why
    hopeless, _, why = cd.choose(1e-3, vc, 1_000_000, 1_000.0, 1_800.0)
    assert (hopeless.n_seeds, hopeless.n_episodes) == (cd.MAX_SEEDS, 1000) and "no S" in why


def test_the_sweep_cost_counts_every_cell() -> None:
    # 6 premiums x S blocks; the ppo cell trains 1000 s, each cell evaluates N x 63 / 1800 s.
    hours = cd.sweep_core_hours(10, 200, 1_000_000, 1_000.0, 1_800.0)
    assert hours == pytest.approx(6 * 10 * (1000 + 2 * 200 * 63 / 1800) / 3600)


def test_the_command_reads_the_pilot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(1)
    np.save(tmp_path / "matrix.npy", rng.normal(0, 300, (8, 1)) + rng.normal(0, 4000, (8, 200)))
    np.save(tmp_path / "matrix_oracle.npy", 500 + rng.normal(0, 1000, (8, 200)))
    assert cd.main(["--pilot-root", str(tmp_path), "--timesteps", "1000000"]) == 0
    assert "design: S = " in capsys.readouterr().out
