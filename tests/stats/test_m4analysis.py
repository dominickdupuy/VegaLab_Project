"""Dominick Dupuy: the M4 analysis of the sweep (#12, #13, #14), on grids small enough to run.

The confirmatory sweep is not on this machine, so the analysis is tested on a tiny grid
the real orchestrator writes (the fixed-delta seller and the fake learner of
``tests/fake_learner.py``), read back through the same ``GridRecord``. The statistics
are tested where their answer is known: the noncentral t inversions against their
defining quantiles and their coverage, the floor and the gap on curves built by hand,
Brown-Forsythe on groups of known spread, and the bootstrap's tolerance on normal seed
means and on a configuration no normal sample produces. Every number is synthetic.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fake_learner import fake_trainer
from scipy.stats import nct

from optspread import m4analysis as m4
from optspread.analysis import decide_contrast
from optspread.artifacts import RUN_FILENAME
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.config import config_hash as hash_of
from optspread.orchestrate import (
    GridCell,
    GridConfig,
    GridRecord,
    grid_cells,
    grid_filename,
    read_grid,
    run_grid,
)
from optspread.power import power_t, t_critical, t_noncentrality, var_grand_mean
from scripts import m4_analysis

SHORT = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
TINY = GridConfig(
    base=SHORT, premiums=(0.0, 2.0, 4.0), n_seeds=3, train=TrainConfig(total_timesteps=64, seed=10)
)
FAST = {"n_boot": 200, "replications": 40}


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("m4") / "runs"
    run_grid(TINY, root, trainer=fake_trainer)
    return root


def tiny_record(root: Path) -> GridRecord:
    return read_grid(root / grid_filename(TINY))


# -- the design ------------------------------------------------------------- #


@pytest.mark.invariant
def test_the_confirmatory_grid_is_the_a7_sweep_and_its_hash_has_not_moved() -> None:
    assert hash_of(m4.CONFIRMATORY_GRID) == m4.CONFIRMATORY_GRID_HASH == "7d9d53d340006f02"
    assert grid_filename(m4.CONFIRMATORY_GRID) == "grid-7d9d53d340006f02.json"
    g = m4.CONFIRMATORY_GRID
    assert (g.n_seeds, g.base.n_episodes, g.base.seed, g.base.cost.enabled) == (
        5,
        500,
        100_000,
        True,
    )
    assert g.train is not None and (g.train.total_timesteps, g.train.seed) == (1_000_000, 1000)
    assert g.premiums == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0) and g.arms == ("delta_seller", "ppo")
    assert hash_of(RunConfig()) == "17c15aa74ce473a9"  # the freeze: nothing re-keyed


def fake_record(grid: GridConfig, grid_hash: str | None = None) -> GridRecord:
    cells = tuple(
        GridCell(
            premium=c.premium,
            arm=c.arm,
            block=c.block,
            run_id=c.run_id,
            config_hash=hash_of(c.config),
            seed=c.config.seed,
            n_episodes=c.config.n_episodes,
            train_seed=None if c.config.train is None else c.config.train.seed,
            mean_pnl=0.0,
            se_pnl=0.0,
            mean_cost=0.0,
            mean_trades=0.0,
        )
        for c in grid_cells(grid)
    )
    return GridRecord(grid_hash=grid_hash or hash_of(grid), grid=grid, cells=cells)


@pytest.mark.invariant
def test_the_confirmatory_record_has_no_design_problem_and_any_other_grid_does() -> None:
    assert m4.design_problems(fake_record(m4.CONFIRMATORY_GRID)) == ([], [])
    hard, soft = m4.design_problems(fake_record(TINY))
    assert hard == [] and "not the confirmatory design" in soft[0]


@pytest.mark.invariant
def test_a_record_that_cannot_be_analysed_is_refused_whatever_the_flag(tmp_path: Path) -> None:
    hard, _ = m4.design_problems(fake_record(TINY, grid_hash="0" * 16))
    assert "is not its config's" in hard[0]
    no_null = TINY.model_copy(update={"premiums": (2.0, 4.0)})
    hard, _ = m4.design_problems(fake_record(no_null))
    assert any("no 0 premium" in h for h in hard)
    only_null = TINY.model_copy(update={"premiums": (0.0,)})
    assert any("no positive premium" in h for h in m4.design_problems(fake_record(only_null))[0])
    short = fake_record(TINY)
    short = short.model_copy(update={"cells": short.cells[:-1]})
    assert any("exactly its grid's cells" in h for h in m4.design_problems(short)[0])
    with pytest.raises(m4.RefusedGridError, match="is not its config's"):
        m4.analyse(fake_record(TINY, grid_hash="0" * 16), tmp_path, exploratory=True)


# -- the noncentral t ------------------------------------------------------- #


@pytest.mark.invariant
def test_power_at_is_power_t_written_in_the_noncentrality() -> None:
    tau2, sigma2, S, N = 3.0e4, 1.0e8, 5, 500
    for delta in (100.0, 500.0, 1_500.0):
        lam = delta / math.sqrt(var_grand_mean(tau2, sigma2, S, N))
        assert m4.power_at(lam, S) == pytest.approx(power_t(delta, tau2, sigma2, S, N), abs=1e-12)
    assert m4.power_at(0.0, S) == 0.05
    assert (m4.power_at(math.inf, S), m4.power_at(-math.inf, S)) == (1.0, 0.0)


@pytest.mark.invariant
def test_lambda_80_is_where_the_power_is_80_percent() -> None:
    lam = m4.noncentrality_for_power(0.80, 5)
    assert m4.power_at(lam, 5) == pytest.approx(0.80, abs=1e-9)
    assert m4.noncentrality_for_power(0.80, 50) < lam  # more df, a lower bar
    with pytest.raises(ValueError, match="target power"):
        m4.noncentrality_for_power(0.04, 5)


@pytest.mark.invariant
@pytest.mark.parametrize("t_obs", [-3.0, 0.0, 2.0, 40.0, 1e4])
def test_the_noncentrality_interval_inverts_the_noncentral_t(t_obs: float) -> None:
    lo, mid, hi = m4.noncentrality_interval(t_obs, 5)
    assert lo < mid < hi
    for lam, q in ((lo, 0.975), (mid, 0.5), (hi, 0.025)):
        assert nct.cdf(t_obs, 4, lam) == pytest.approx(q, abs=1e-6)


@pytest.mark.behaviour
def test_an_infinite_t_has_no_interval_and_a_bad_level_is_refused() -> None:
    assert m4.noncentrality_interval(math.inf, 5) == (m4.NONCENTRALITY_CAP,) * 3
    assert m4.noncentrality_interval(-math.inf, 5)[0] == -m4.NONCENTRALITY_CAP
    with pytest.raises(ValueError, match="level"):
        m4.noncentrality_interval(1.0, 5, level=1.0)


@pytest.mark.invariant
def test_the_band_covers_and_the_point_is_median_unbiased() -> None:
    # The section 8 discipline on the curve's own interval: t from nct(4, lambda = 2).
    rng = np.random.default_rng(3)
    draws = nct.rvs(4, 2.0, size=300, random_state=rng)
    bands = [m4.noncentrality_interval(float(t), 5) for t in draws]
    covered = np.mean([lo <= 2.0 <= hi for lo, _, hi in bands])
    below = np.mean([mid < 2.0 for _, mid, _ in bands])
    assert abs(covered - 0.95) <= 4 * math.sqrt(0.95 * 0.05 / 300)
    assert abs(below - 0.5) <= 4 * math.sqrt(0.25 / 300)
    assert np.mean(draws) / 2.0 > 1.15  # why the t itself is not the point estimate


# -- the curve, the floor and the gap ---------------------------------------- #

PREMIUMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def curve_of(lams: list[float], spread: float = 1.0) -> list[m4.CurvePoint]:
    return [
        m4.CurvePoint(
            p,
            p > 0.0,
            lam,
            lam - spread if p > 0.0 else 0.0,
            lam + spread if p > 0.0 else 0.0,
            m4.power_at(lam, 5),
            m4.power_at(lam - spread if p > 0.0 else 0.0, 5),
            m4.power_at(lam + spread if p > 0.0 else 0.0, 5),
            None,
        )
        for p, lam in zip(PREMIUMS, lams, strict=True)
    ]


LAM80 = m4.noncentrality_for_power(0.80, 5)


@pytest.mark.invariant
def test_the_floor_is_the_first_crossing_of_the_interpolated_noncentrality() -> None:
    floor = m4.detection_floor(curve_of([0.0, 0.5, 1.0, 2.0, 4.0, 8.0]), 5)
    assert floor.premium == pytest.approx(1.0 + (LAM80 - 2.0) / 2.0)
    assert floor.status == "reached" and floor.bracket == (1.0, 2.0) and not floor.recrossings
    assert floor.lo is not None and floor.hi is not None
    assert floor.lo < floor.premium < floor.hi  # the band's upper edge crosses first
    assert floor.max_power == pytest.approx(m4.power_at(8.0, 5))


@pytest.mark.invariant
def test_a_floor_under_the_first_sweep_point_rests_on_the_anchor() -> None:
    floor = m4.detection_floor(curve_of([0.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 5)
    assert floor.premium == pytest.approx(0.25 * LAM80 / 5.0)
    assert floor.status == "below_grid" and floor.bracket == (0.0, 0.25)


@pytest.mark.invariant
def test_a_floor_never_reached_is_above_the_grid_and_a_dip_is_reported() -> None:
    never = m4.detection_floor(curve_of([0.0, 0.1, 0.2, 0.4, 0.8, 1.6], spread=0.5), 5)
    assert never.premium is None and never.status == "not_reached"
    assert never.bracket == (4.0, None) and never.hi is None
    assert never.max_power == pytest.approx(m4.power_at(1.6, 5))
    dip = m4.detection_floor(curve_of([0.0, 1.0, 4.0, 2.0, 5.0, 6.0]), 5)
    assert dip.bracket == (0.25, 0.5) and dip.recrossings == (1.0,)
    with pytest.raises(ValueError, match="measured"):
        m4.detection_floor(curve_of([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])[:1], 5)


@pytest.mark.behaviour
def test_the_interpolated_curve_meets_the_knots() -> None:
    curve = curve_of([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    dense = m4.interpolate(curve, 5)
    assert len(dense) == 81 and dense[0] == (0.0, 0.05, 0.05, 0.05)
    at = {round(p, 6): (w, lo, hi) for p, w, lo, hi in dense}
    for c in curve:
        assert at[c.premium] == pytest.approx((c.power, c.power_lo, c.power_hi))
    powers = [w for _, w, _, _ in dense]
    assert powers == sorted(powers)  # a rising noncentrality draws a rising curve


@pytest.mark.invariant
def test_the_gap_is_the_horizontal_distance_at_80_percent() -> None:
    oracle = m4.detection_floor(curve_of([0.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 5)
    learned = m4.detection_floor(curve_of([0.0, 0.5, 1.0, 2.0, 4.0, 8.0]), 5)
    never = m4.detection_floor(curve_of([0.0, 0.1, 0.2, 0.4, 0.8, 1.6], spread=0.5), 5)
    assert oracle.premium is not None and oracle.hi is not None
    assert learned.premium is not None and learned.lo is not None
    gap = m4.gap_between("delta_seller", oracle, "ppo", learned, 4.0)
    assert gap.kind == "point"
    assert gap.value == pytest.approx(learned.premium - oracle.premium)
    assert gap.bracket == (1.0 - 0.25, 2.0 - 0.0)
    assert gap.band_range[0] == pytest.approx(learned.lo - oracle.hi)
    bound = m4.gap_between("delta_seller", oracle, "ppo", never, 4.0)
    assert bound.kind == "lower_bound"
    assert bound.value == pytest.approx(4.0 - oracle.premium)
    assert bound.bracket == (4.0 - 0.25, None)
    assert m4.gap_between("delta_seller", never, "ppo", learned, 4.0).kind == "undefined"


# -- homogeneity (#13) ------------------------------------------------------ #


def groups(sds: list[float], seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    # (S, N) matrices whose seed means have the given spreads.
    return [rng.normal(0.0, sd, size=(12, 1)) + rng.normal(0.0, 1e-3, size=(12, 4)) for sd in sds]


@pytest.mark.invariant
def test_brown_forsythe_sees_a_spread_that_moves_and_not_one_that_does_not() -> None:
    same = m4.brown_forsythe("contrast", [1.0, 2.0, 4.0], groups([5.0, 5.0, 5.0]))
    moved = m4.brown_forsythe("contrast", [1.0, 2.0, 4.0], groups([1.0, 5.0, 50.0]))
    assert same.rejected is False and moved.rejected is True
    assert moved.df == (2, 33) and len(moved.levels) == 3
    assert moved.levels[2].seed_mean_variance > moved.levels[0].seed_mean_variance


@pytest.mark.behaviour
def test_brown_forsythe_without_spread_or_with_one_group_is_undefined() -> None:
    flat = [np.ones((4, 3)), np.ones((4, 3)) * 2.0]
    assert m4.brown_forsythe("arm", [0.0, 1.0], flat).p_value is None
    one = m4.brown_forsythe("arm", [1.0], groups([1.0]))
    assert one.statistic is None and one.rejected is None


# -- the bootstrap (#14) ------------------------------------------------------ #


@pytest.mark.invariant
def test_normal_seed_means_agree_with_the_t_test() -> None:
    means = np.random.default_rng(5).normal(1.0, 1.0, size=5)
    level = m4.bootstrap_level(1.0, list(means), 0.01, n_boot=2_000, replications=300)
    assert level.agrees and level.reference[0] < level.check.gap < level.reference[1]
    assert level.check.n_seeds == 5 and level.check.n_boot == 2_000
    # the percentile interval is narrower than t at S = 5, and normal data say by how much
    assert level.width_ratio < 1.0 and 0.5 < level.reference_width_ratio < 0.85
    assert level.t_lower == pytest.approx(level.check.estimate - t_critical(5) * level.check.se)
    again = m4.bootstrap_level(1.0, list(means), 0.01, n_boot=2_000, replications=300)
    assert again == level  # seeded, so the report rebuilds the same


@pytest.mark.invariant
def test_one_outlying_seed_is_a_divergence_normal_data_do_not_produce() -> None:
    level = m4.bootstrap_level(
        2.0, [0.0, 0.0, 0.0, 0.0, 10.0], 0.01, n_boot=4_000, replications=400
    )
    assert not level.agrees and level.p_value < 0.01
    assert level.check.gap < level.reference[0]


@pytest.mark.behaviour
def test_a_decisive_level_agrees_trivially() -> None:
    level = m4.bootstrap_level(
        4.0, [100.0, 101.0, 99.0, 100.5, 99.5], 0.01, n_boot=500, replications=50
    )
    assert level.check.rate == 1.0 and level.agrees and level.boot_detected


@pytest.mark.invariant
def test_the_bootstrap_rule_is_anticonservative_at_five_seeds() -> None:
    size = m4.bootstrap_rule_size(5, n_boot=1_000, replications=600)
    assert size.size > 0.05 + 3 * size.mc_se and size.n_seeds == 5


# -- end to end on a tiny grid ------------------------------------------------- #


@pytest.mark.invariant
def test_a_grid_that_is_not_the_confirmatory_design_is_refused(tiny_root: Path) -> None:
    with pytest.raises(m4.RefusedGridError, match="not the confirmatory design"):
        m4.analyse(tiny_record(tiny_root), tiny_root, **FAST)


@pytest.mark.invariant
def test_the_analysis_runs_the_preregistered_event_on_every_arm_and_premium(
    tiny_root: Path,
) -> None:
    record = tiny_record(tiny_root)
    result = m4.analyse(record, tiny_root, exploratory=True, **FAST)
    assert not result.confirmatory and len(result.deviations) == 1
    assert result.blocks == (0, 1, 2) and [a.arm for a in result.arms] == ["delta_seller", "ppo"]
    for a in result.arms:
        assert [r.premium for r in a.results] == [2.0, 4.0]
        for r in a.results:
            want = decide_contrast(record.contrast(tiny_root, r.premium, a.arm))
            assert (r.decision.estimate, r.decision.se) == pytest.approx((want.estimate, want.se))
            assert r.decision.rejected == want.rejected
            assert np.mean(r.seed_means) == pytest.approx(want.estimate)
        assert [c.premium for c in a.curve] == [0.0, 2.0, 4.0] and not a.curve[0].measured
        assert [h.on for h in a.homogeneity] == ["contrast", "arm"]
        assert a.homogeneity[1].premiums == (0.0, 2.0, 4.0) and len(a.bootstrap) == 2
        assert a.bootstrap[0].threshold == pytest.approx(0.025)
    assert result.gap is not None and result.rule_size.n_seeds == 3
    assert result.sweep_git_shas and result.arm("ppo").role == "learned"


@pytest.mark.behaviour
def test_the_sidecar_is_strict_json_with_the_documented_shape(tiny_root: Path) -> None:
    result = m4.analyse(tiny_record(tiny_root), tiny_root, exploratory=True, **FAST)
    prov = m4.Provenance(date(2026, 11, 19), "$VEGALAB_DATA/runs", "grid-x.json", "abc", False, {})

    def no_constants(name: str) -> None:
        raise AssertionError(f"{name} in the sidecar")

    data = json.loads(result.render_json(prov), parse_constant=no_constants)
    assert data["schema_version"] == 1 and data["milestone"] == "M4"
    assert data["report_date"] == "2026-11-19" and data["confirmatory"] is False
    assert data["provenance"]["confirmatory_grid_hash"] == "7d9d53d340006f02"
    assert data["design"]["premiums"] == [0.0, 2.0, 4.0] and data["arm_order"] == [
        "delta_seller",
        "ppo",
    ]
    arm = data["arms"]["ppo"]
    assert set(arm) == {"role", "decisions", "curve", "floor", "homogeneity", "bootstrap"}
    assert set(arm["decisions"][0]) >= {"premium", "estimate", "se", "t", "lower_bound", "detected"}
    assert (
        len(arm["curve"]["interpolated"]) == 81 and arm["curve"]["points"][0]["measured"] is False
    )
    assert set(arm["homogeneity"]) == {"contrast", "arm"}
    assert arm["bootstrap"]["rule_in_force"] in ("t", "bootstrap")
    assert data["gap"]["kind"] in ("point", "lower_bound", "undefined")
    assert data["settings"]["df"] == 2 and data["bootstrap_rule_size"]["n_seeds"] == 3


@pytest.mark.invariant
def test_an_incomplete_grid_is_refused_and_analysed_on_its_whole_blocks_when_told(
    tiny_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    shutil.copytree(tiny_root, root)
    record = tiny_record(root)
    victim = record.cell(4.0, "ppo", 1)
    (root / victim.run_id / RUN_FILENAME).unlink()
    with pytest.raises(m4.RefusedGridError, match="incomplete: 1 of 18 cells"):
        m4.analyse(record, root, **FAST)
    result = m4.analyse(record, root, exploratory=True, **FAST)
    assert result.blocks == (0, 2) and result.n_seeds == 2
    assert result.arms[0].results[0].decision.components.n_seeds == 2
    assert (
        result.to_dict(m4.Provenance(date.today(), "r", "g", "s", False, {}))["provenance"][
            "cells_used"
        ]
        == 12
    )
    for gone in (0, 2):
        (root / record.cell(0.0, "delta_seller", gone).run_id / RUN_FILENAME).unlink()
    with pytest.raises(m4.RefusedGridError, match="needs at least 2"):
        m4.analyse(record, root, exploratory=True, **FAST)


# -- the command --------------------------------------------------------------- #


@pytest.mark.behaviour
def test_the_command_writes_a_labelled_exploratory_report_with_portable_roots(
    tiny_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VEGALAB_DATA", str(tiny_root.parent))
    grid = tiny_root / grid_filename(TINY)
    out = tiny_root.parent / "report" / "M4_ANALYSIS"
    args = [
        "--root",
        str(tiny_root),
        "--grid",
        str(grid),
        "--n-boot",
        "200",
        "--replications",
        "200",
    ]
    assert m4_analysis.main(args, on=date(2026, 11, 19)) == 2
    assert "--exploratory analyses it anyway" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        m4_analysis.main([*args, "--exploratory"])  # no --out: docs/ is for the real one
    assert m4_analysis.main([*args, "--exploratory", "--out", str(out)], on=date(2026, 11, 19)) == 0
    text = out.with_suffix(".md").read_text()
    assert (
        text.startswith("# M4 analysis")
        and "**EXPLORATORY: not the confirmatory analysis.**" in text
    )
    assert "Run 2026-11-19" in text and "under `$VEGALAB_DATA/runs`" in text
    assert str(tiny_root) not in text
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["provenance"]["root"] == "$VEGALAB_DATA/runs"
    assert (
        data["provenance"]["grid_file"] == grid.name
        and len(data["provenance"]["analysis_git_sha"]) == 40
    )
    assert m4_analysis.main([*args, "--exploratory", "--out", "-"]) == 0
    assert "## Bootstrap validation (#14)" in capsys.readouterr().out


@pytest.mark.behaviour
def test_the_command_defaults_to_the_confirmatory_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert m4_analysis.main(["--root", str(tmp_path)]) == 2
    assert "grid-7d9d53d340006f02.json" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # too few to ever reach p < 0.01: a tolerance never applied
        m4_analysis.main(["--root", str(tmp_path), "--replications", "100"])


@pytest.mark.behaviour
def test_the_report_names_a_confirmatory_run_a_gap_and_a_divergence(tiny_root: Path) -> None:
    result = m4.analyse(tiny_record(tiny_root), tiny_root, exploratory=True, **FAST)
    prov = m4.Provenance(
        date(2026, 11, 19), "$VEGALAB_DATA/runs", "grid-x.json", "a" * 40, True, {}
    )
    diverged = m4.bootstrap_level(
        4.0, [0.0, 0.0, 0.0, 0.0, 10.0], 0.05, n_boot=500, replications=100
    )
    ppo = replace(result.arm("ppo"), bootstrap=(diverged,))
    reached = m4.detection_floor(curve_of([0.0, 0.5, 1.0, 2.0, 4.0, 8.0]), 5)
    seller = replace(result.arm("delta_seller"), floor=reached)
    point = m4.gap_between("delta_seller", reached, "ppo", reached, 4.0)
    text = replace(result, deviations=(), arms=(seller, ppo), gap=point).render(prov)
    assert "**Confirmatory**: the A7 design" in text and "(dirty)" in text
    assert "**Gap** (ppo floor minus delta_seller floor): 0 vol points." in text
    assert "between the sweep points 1 and 2" in text
    assert "diverges at 4 vol points" in text and "the bootstrap rule" in text
    assert ppo.rule_in_force == "bootstrap" and ppo.diverged_at == (4.0,)
    never = m4.detection_floor(curve_of([0.0, 0.1, 0.2, 0.4, 0.8, 1.6], spread=0.5), 5)
    bound = m4.gap_between("delta_seller", reached, "ppo", never, 4.0)
    assert "only a lower bound" in replace(result, gap=bound).render(prov)
    assert "it has not" in replace(result, gap=None).render(prov)
    assert "upper edge" not in replace(result, arms=(replace(seller, floor=never),)).render(prov)
    wide = m4.detection_floor(curve_of([0.0, 0.1, 0.2, 0.4, 0.8, 1.6], spread=2.0), 5)
    assert wide.premium is None and wide.lo is not None
    text = replace(result, arms=(replace(seller, floor=wide),)).render(prov)
    assert f"the band's upper edge reaches it at {wide.lo:.3g}" in text
    below = m4.detection_floor(curve_of([0.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 5)
    dip = m4.detection_floor(curve_of([0.0, 1.0, 4.0, 2.0, 5.0, 6.0]), 5)
    for floor, words in (
        (below, "below the first sweep point"),
        (dip, "falls back under 80% at 1"),
    ):
        assert words in replace(result, arms=(replace(seller, floor=floor), ppo)).render(prov)


# -- the audit's fixes (S1, S3, S5, S7, N3, N5) ------------------------------------ #


@pytest.mark.invariant
def test_one_solver_for_the_80_percent_noncentrality_and_a_knot_on_it_is_the_floor() -> None:
    assert m4.noncentrality_for_power(0.80, 5) == t_noncentrality(5, 0.05, 0.80)
    on_knot = m4.detection_floor(curve_of([0.0, 1.0, LAM80, 5.0, 6.0, 7.0]), 5)
    assert on_knot.premium == 0.5 and on_knot.status == "reached"  # exactly, not 0.49999...
    assert m4.crossing([0.0, 1.0], [0.0, 3.0], 3.0) == 1.0


@pytest.mark.invariant
def test_each_bootstrap_level_resamples_with_its_own_seed() -> None:
    seeds = {m4.level_seed(0, arm, p) for arm in ("delta_seller", "ppo") for p in PREMIUMS[1:]}
    assert len(seeds) == 10  # no two levels or arms share resample indices
    assert m4.level_seed(0, "ppo", 1.0) == m4.level_seed(0, "ppo", 1.0)
    assert m4.level_seed(1, "ppo", 1.0) != m4.level_seed(0, "ppo", 1.0)
    means = [1.0, 2.0, 0.5, 3.0, 1.5]
    a = m4.bootstrap_level(1.0, means, 0.01, n_boot=500, replications=20, seed=1)
    b = m4.bootstrap_level(2.0, means, 0.01, n_boot=500, replications=20, seed=2)
    assert a.seed == 1 and not np.array_equal(
        m4.resampled_means(np.array(means), 500, a.seed),
        m4.resampled_means(np.array(means), 500, b.seed),
    )


@pytest.mark.invariant
def test_brown_forsythe_is_calibrated_at_the_analysis_s_and_has_little_power_at_five() -> None:
    cal = m4.brown_forsythe_calibration(5, 5, ratios=(25.0,), replications=1_000, seed=3)
    assert cal.groups == 5 and cal.per_group == 5 and cal.alpha == 0.05
    assert cal.size < 0.02  # far under its alpha with five means a group
    assert cal.power[0][0] == 25.0 and cal.power[0][1] < 0.6  # weak even at a 25-fold spread
    assert cal.mc_se > 0.0
    big = m4.brown_forsythe_calibration(5, 40, ratios=(25.0,), replications=300, seed=3)
    assert big.power[0][1] > 0.95  # the test works where it has the means to
    with pytest.raises(ValueError, match="at least 2 groups"):
        m4.brown_forsythe_calibration(1, 5)


@pytest.mark.invariant
def test_the_plan_is_a7s_and_the_design_is_rechecked_on_the_sweeps_own_variances(
    tiny_root: Path,
) -> None:
    plan = m4.planned_design(5, 500)
    assert round(plan.power, 3) == m4.A7_POWER and round(plan.floor, 1) == m4.A7_FLOOR
    # The audit's figure: PPO's sweep contrast at 1 vol point gives the design 0.584.
    assert power_t(m4.A7_DELTA, 4.088e5, 1.311e8, 5, 500) == pytest.approx(0.584, abs=1e-3)
    result = m4.analyse(tiny_record(tiny_root), tiny_root, exploratory=True, **FAST)
    for a in result.arms:
        for d, r in zip(result.design_power(a), a.results, strict=True):
            vc = r.decision.components
            assert (d.premium, d.tau2, d.sigma2) == (r.premium, vc.tau2, vc.sigma2)
            assert d.power == power_t(m4.A7_DELTA, vc.tau2, vc.sigma2, 3, TINY.base.n_episodes)
    prov = m4.Provenance(date(2026, 11, 19), "r", "g", "a" * 40, False, {})
    data = result.to_dict(prov)
    check = data["design_check"]
    assert check["planned"]["source"] == "A7" and check["planned"]["delta"] == 824.3
    assert [d["premium"] for d in check["arms"]["ppo"]] == [2.0, 4.0]
    text = result.render(prov)
    assert "## The power the design had (A7's plan against the sweep's variances)" in text
    assert "These are pilot estimates, not what the sweep achieved." in text


@pytest.mark.invariant
def test_the_report_and_the_sidecar_say_the_rules_were_written_after_the_sweep(
    tiny_root: Path,
) -> None:
    result = m4.analyse(tiny_record(tiny_root), tiny_root, exploratory=True, **FAST)
    prov = m4.Provenance(date(2026, 11, 19), "r", "g", "a" * 40, False, {})
    rules = result.to_dict(prov)["rules"]
    assert rules == {
        "written": "2026-10-28",
        "sweep_ran": "2026-10-26",
        "written_after_sweep": True,
        "statement": m4.RULES_TIMING,
    }
    assert "after its raw arm means were posted in issue #9" in m4.RULES_TIMING
    assert "not pre-registered" in m4.RULES_TIMING and "A9 (issue #37)" in m4.RULES_TIMING
    text = result.render(prov)
    assert f"**When these rules were written.** {m4.RULES_TIMING}" in text
    assert "before the sweep" not in text and "before any sweep" not in (m4.__doc__ or "")
    assert "a non-rejection here does not show that tau^2 is the same" in text
    assert "The test's own size and power at 2 groups of 3" in text
    assert "a switch changes the verdict only" in text
    assert "read each floor as about a tenth low" in text
    data = result.to_dict(prov)
    assert [c["groups"] for c in data["homogeneity_calibration"]] == [2, 3]
    assert all(isinstance(b["seed"], int) for b in data["arms"]["ppo"]["bootstrap"]["levels"])


@pytest.mark.behaviour
def test_a_floor_below_the_grid_lists_the_noncentrality_per_vol_point(tiny_root: Path) -> None:
    result = m4.analyse(tiny_record(tiny_root), tiny_root, exploratory=True, **FAST)
    prov = m4.Provenance(date(2026, 11, 19), "r", "g", "a" * 40, False, {})
    seller = result.arm("delta_seller")
    low = m4.Floor(0.05, 0.03, 0.2, "below_grid", (0.0, 2.0), (), 1.0)
    per = [c.noncentrality / c.premium for c in seller.curve if c.measured]
    gap = m4.gap_between("delta_seller", low, "ppo", result.arm("ppo").floor, 4.0)
    text = replace(result, arms=(replace(seller, floor=low), result.arm("ppo")), gap=gap).render(
        prov
    )
    assert f"the measured noncentrality per vol point is {per[0]:.3g}, {per[1]:.3g}" in text
    assert "the floor holds the first of these down to zero" in text
    assert "so the gap inherits its interpolation from the anchor at zero" in text
