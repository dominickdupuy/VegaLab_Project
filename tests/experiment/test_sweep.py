"""Ilan Danial: sweep orchestration on the planted-premium axis, with paired seeds (week 5).

A sweep is a grid of ``RunConfig`` cells (premium x arm) run on one shared seed range
and written as ordinary run directories, so every cell is reproducible from its config
hash and seed alone and the dashboard reads them with the same reader as any run.

Pre-registered predictions for the oracle curve, committed before the module existed
and before the first sweep was run (the test IS the prediction):

    P1  FLAT is exactly zero at every premium: no position, no cost, no P&L.
    P2  The fixed-delta seller collects a planted premium: on paired seeds, P&L at a
        planted 2 vol points exceeds P&L at 0 vol points, with a one-sided paired
        t-statistic of at least 3 over 20 seeds (the same bar the Wave 0 gate uses for
        "zero"), so the seller's mean is increasing along the grid.
    P3  A second run of the same sweep is bit-identical.

The pilot matrix for the variance-components pilot lives here too: on a scripted policy,
which has no training variance, the estimated between-seed component must be at or
near zero. That is the pilot's negative control.

Every number is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optspread.artifacts import RUN_FILENAME, SchemaVersionError, read_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, config_hash
from optspread.harness import EpisodeResult, build_env, run_episode, run_episodes
from optspread.manifest import MANIFEST_FILENAME
from optspread.policies import DeltaSellerPolicy, Policy
from optspread.power import variance_components
from optspread.sweep import (
    SWEEP_FILENAME,
    SweepConfig,
    cell_config,
    pilot_matrix,
    read_sweep,
    run_sweep,
)

SHORT = RunConfig(
    seed=3000,
    n_episodes=20,
    market=MarketConfig(n_days=42, tenor_days=(14, 28)),
    env=EnvConfig(episode_days=42),
    cost=CostConfig(enabled=True),
)
SWEEP = SweepConfig(base=SHORT, premiums=(0.0, 2.0), arms=("flat", "delta_seller"))


# -- the orchestration contract --------------------------------------------- #


def test_a_cell_changes_only_the_premium_and_the_arm() -> None:
    cfg = cell_config(SWEEP, 2.0, "delta_seller")
    assert cfg.market.iv_premium_vol_pts == 2.0
    assert cfg.policy == "delta_seller"
    assert cfg.market.model_copy(update={"iv_premium_vol_pts": 0.0}) == SHORT.market
    assert (cfg.seed, cfg.n_episodes, cfg.env, cfg.cost) == (
        SHORT.seed,
        SHORT.n_episodes,
        SHORT.env,
        SHORT.cost,
    )


def test_cells_are_run_configs_keyed_by_hash_and_written_as_run_directories(
    tmp_path: Path,
) -> None:
    record = run_sweep(SWEEP, tmp_path)
    assert record.sweep_hash == config_hash(SWEEP)
    assert len(record.cells) == 4
    for cell in record.cells:
        cfg = cell_config(SWEEP, cell.premium, cell.policy)
        assert cell.config_hash == config_hash(cfg)
        assert cell.run_id == f"{config_hash(cfg)}-s{SHORT.seed}"
        cell_dir = tmp_path / cell.run_id
        assert (cell_dir / RUN_FILENAME).exists() and (cell_dir / MANIFEST_FILENAME).exists()
        run = read_run(cell_dir)
        assert run.config_hash == cell.config_hash
        assert run.n_episodes == cell.n_episodes == SHORT.n_episodes
        assert run.mean_pnl == cell.mean_pnl and run.se_pnl == cell.se_pnl
    assert (tmp_path / SWEEP_FILENAME).exists()


@pytest.mark.slow
def test_arms_at_one_premium_share_their_episode_seeds_exactly(tmp_path: Path) -> None:
    record = run_sweep(SWEEP, tmp_path)
    expected = list(range(SHORT.seed, SHORT.seed + SHORT.n_episodes))
    for cell in record.cells:
        rows = read_run(tmp_path / cell.run_id).episodes
        assert [row.seed for row in rows] == expected, cell.run_id


@pytest.mark.slow
def test_p3_a_second_run_is_bit_identical(tmp_path: Path) -> None:
    a = run_sweep(SWEEP, tmp_path / "a")
    b = run_sweep(SWEEP, tmp_path / "b")
    assert a == b
    assert a.render() == b.render()
    for cell in a.cells:
        assert (tmp_path / "a" / cell.run_id / RUN_FILENAME).read_bytes() == (
            tmp_path / "b" / cell.run_id / RUN_FILENAME
        ).read_bytes()
    assert (tmp_path / "a" / SWEEP_FILENAME).read_bytes() == (
        tmp_path / "b" / SWEEP_FILENAME
    ).read_bytes()


@pytest.mark.slow
def test_a_sweep_resumes_from_valid_cells_and_recomputes_only_the_missing(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def counting(cfg: RunConfig, policy: Policy | None = None) -> tuple[EpisodeResult, ...]:
        calls.append(config_hash(cfg))
        return run_episodes(cfg, policy)

    first = run_sweep(SWEEP, tmp_path, runner=counting)
    assert len(calls) == 4

    # A cell whose results file is gone is recomputed; the other three are read back.
    victim = first.cells[1]
    (tmp_path / victim.run_id / RUN_FILENAME).unlink()
    second = run_sweep(SWEEP, tmp_path, runner=counting)
    assert calls[4:] == [victim.config_hash]
    assert second == first

    # A cell whose results no longer match its manifest is not trusted either.
    other = first.cells[2]
    (tmp_path / other.run_id / RUN_FILENAME).write_bytes(
        (tmp_path / first.cells[3].run_id / RUN_FILENAME).read_bytes()
    )
    third = run_sweep(SWEEP, tmp_path, runner=counting)
    assert calls[5:] == [other.config_hash]
    assert third == first


def test_the_sweep_record_round_trips_and_refuses_an_unknown_schema(tmp_path: Path) -> None:
    record = run_sweep(SWEEP, tmp_path)
    assert read_sweep(tmp_path) == record
    bumped = record.model_copy(update={"schema_version": record.schema_version + 1})
    (tmp_path / SWEEP_FILENAME).write_text(bumped.model_dump_json())
    with pytest.raises(SchemaVersionError):
        read_sweep(tmp_path)


def test_a_sweep_config_must_have_distinct_premiums_and_at_least_one_arm() -> None:
    with pytest.raises(ValueError, match="premiums"):
        SweepConfig(base=SHORT, premiums=(1.0, 1.0), arms=("flat",))
    with pytest.raises(ValueError, match="arms"):
        SweepConfig(base=SHORT, premiums=(0.0,), arms=())
    with pytest.raises(ValueError, match="premiums"):
        SweepConfig(base=SHORT, premiums=(), arms=("flat",))


# -- the pre-registered predictions ----------------------------------------- #


def test_p1_flat_is_exactly_zero_at_every_premium(tmp_path: Path) -> None:
    record = run_sweep(SWEEP, tmp_path)
    for premium in SWEEP.premiums:
        cell = record.cell(premium, "flat")
        assert cell.mean_pnl == 0.0 and cell.se_pnl == 0.0 and cell.mean_cost == 0.0
        assert cell.mean_trades == 0.0
        assert all(row.pnl == 0.0 for row in read_run(tmp_path / cell.run_id).episodes)


def test_p2_the_fixed_delta_seller_collects_a_planted_premium(tmp_path: Path) -> None:
    record = run_sweep(SWEEP, tmp_path)
    at_zero = read_run(tmp_path / record.cell(0.0, "delta_seller").run_id).episodes
    at_two = read_run(tmp_path / record.cell(2.0, "delta_seller").run_id).episodes
    assert [r.seed for r in at_zero] == [r.seed for r in at_two], "paired by construction"
    paired = np.array([b.pnl - a.pnl for a, b in zip(at_zero, at_two, strict=True)])
    n = len(paired)
    t = float(paired.mean() / (paired.std(ddof=1) / np.sqrt(n)))
    assert paired.mean() > 0.0, paired
    assert t >= 3.0, f"paired t = {t:.2f} over {n} seeds"
    series = record.series("delta_seller")
    assert series == (
        record.cell(0.0, "delta_seller").mean_pnl,
        record.cell(2.0, "delta_seller").mean_pnl,
    )
    assert series[0] < series[1]


# -- the pilot matrix: the scripted negative control --------------------------- #


def test_pilot_matrix_blocks_use_disjoint_path_seeds() -> None:
    S, N = 4, 5
    matrix = pilot_matrix(SHORT, S, N, lambda _: DeltaSellerPolicy())
    assert matrix.shape == (S, N)
    env = build_env(SHORT)
    for i in range(S):
        for j in range(N):
            direct = run_episode(env, DeltaSellerPolicy(), SHORT.seed + i * N + j)
            assert matrix[i, j] == direct.pnl, (i, j)


@pytest.mark.slow
def test_a_scripted_policy_shows_no_seed_effect_across_a_seed_sweep() -> None:
    # Under no seed effect E[MS_between] = E[MS_within], so (MS_between - MS_within)/N has
    # mean 0 and the clamped estimate tau2_hat is small relative to sigma2_hat: with S = 4
    # and N = 5 its SD is about sqrt(2/3 + 2/16) sigma2 / 5 = 0.18 sigma2 for normal data,
    # so the positive part averages well under 0.1 sigma2; option P&L is skewed, so the
    # bound is left at 0.5 with that factor of five in hand. The clamp fires about half
    # the time, so over 14 base seeds at least one clamp is expected (1 - 2^-14 otherwise).
    S, N = 4, 5
    ratios: list[float] = []
    clamped = 0
    for k in range(14):
        base = SHORT.model_copy(update={"seed": 10_000 + 1_000 * k})
        vc = variance_components(pilot_matrix(base, S, N, lambda _: DeltaSellerPolicy()))
        assert vc.sigma2 > 0.0
        ratios.append(vc.tau2 / vc.sigma2)
        clamped += int(vc.clamped)
    assert float(np.mean(ratios)) < 0.5, ratios
    assert clamped >= 1, ratios
