"""Dominick Dupuy: P&L attribution and its residual (project document 10.9, #16).

Three claims, each checked against something that does not come from the module:

* the identity ``delta + gamma + vega + theta + carry + cost + residual == pnl``
  holds at every step, episode and run, to the one rounding of the subtraction that
  defines the residual;
* the residual is small for the scripted arms, and it is small for the right reason:
  it is the remainder of a second-order expansion, so it shrinks as the expansion
  gets better (a longer tenor, where one day is a smaller step in ``sigma sqrt(tau)``);
* one long call over one day, worked by hand from the Black-Scholes formulas.

And the attribution decomposes the run it says it does: every re-simulated episode
equals its ``run.json`` row, and the run directory is left exactly as it was.
Every number here is synthetic.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread.artifacts import RUN_FILENAME, EpisodeRow, read_run, write_run
from optspread.attribution import (
    ATTRIBUTION_ENV,
    PNL_RTOL,
    REPO_ROOT,
    Terms,
    attribute_config,
    attribute_run,
    attribute_step,
    default_root,
    dump_attribution,
    leg_terms,
    output_name,
    read_attribution,
    write_attribution,
)
from optspread.config import (
    CostConfig,
    EnvConfig,
    MarketConfig,
    PolicyName,
    RunConfig,
    TrainConfig,
    config_hash,
)
from optspread.harness import run_episodes, summarize
from optspread.market import SyntheticChainProvider
from optspread.orchestrate import execute_run
from optspread.policies import Policy
from optspread.realdata.panel import K_GRID, TENOR_GRID
from optspread.realdata.provider import HistoricalChainProvider, SurfaceStore
from optspread.train import POLICY_FILENAME
from optspread.types import Chain, OptionLeg, Position
from scripts.attribution import main

# The tolerance on the residual share, |residual| over gross mark-to-market, for the
# scripted arms in the default synthetic world. The residual is the third-order
# remainder of the expansion; in Black-Scholes the natural step is dS / (S sigma
# sqrt(tau)) = sqrt(dt / tau), about 0.15 one day into a 42-day structure, so the
# remainder runs at a few percent of the move (measured 1.5% to 2.6% over 20 seeds;
# always_on is the high one because it holds through the expiry day, where a leg's
# value stops being smooth). 5% is that with margin; a sign or unit error in any one
# greek (vega per point, theta per day, a short leg's sign) moves it far past it,
# vega's only where the surface moves (MovingSurface below: the default one is flat).
RESIDUAL_SHARE_TOL = 0.05
# Accounting: the env's equity change less the legs' marks and the cost is float
# rounding on a $1M book (about 1e-10); anything near a cent is a booking the
# attribution does not see.
ACCOUNTING_TOL = 1e-6

SCRIPTED: tuple[PolicyName, ...] = ("always_on", "delta_seller", "random")
SMALL = RunConfig(
    policy="delta_seller",
    seed=8000,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21), iv_premium_vol_pts=1.0),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)


def close_enough(a: float, b: float) -> bool:
    """Equal to the rounding of one subtraction at the larger magnitude."""
    return abs(a - b) <= 2 * math.ulp(max(abs(a), abs(b), 1.0))


def identity_holds(t: Terms) -> bool:
    return close_enough(t.explained + t.residual, t.pnl)


def write_scripted(config: RunConfig, run_dir: Path) -> Path:
    episodes = run_episodes(config)
    write_run(run_dir, config, summarize(episodes), episodes)
    return run_dir


# -- a hand-worked case -------------------------------------------------------- #


def one_leg_chains(spot1: float = 101.0, vol1: float = 0.21) -> tuple[Chain, Chain]:
    def chain(t: int, spot: float, vol: float) -> Chain:
        strikes = np.array([90.0, 100.0, 110.0])
        return Chain(t, spot, 0.0, strikes, np.array([63]), np.full((1, 3), vol))

    return chain(0, 100.0, 0.20), chain(1, spot1, vol1)


def test_one_long_call_over_one_day_matches_the_hand_calculation() -> None:
    # S = K = 100, r = 0, sigma = 0.20, T = 63/252 = 0.25: d1 = 0.5 sigma sqrt(T) = 0.05,
    # N(d1) = 0.5199388, phi(d1) = 0.3984439. Per share: Delta = N(d1);
    # Gamma = phi / (S sigma sqrt T) = 0.03984439; Vega = S phi sqrt T = 19.922196 per
    # unit vol; Theta = -S phi sigma / (2 sqrt T) = -7.968878 per year. The day moves
    # S by +1, sigma by +0.01 (one vol point) and t by 1/252 of a year; times 100.
    before, after = one_leg_chains()
    leg = OptionLeg("C", 100.0, 63, +1, before.price("C", 100.0, 63))
    t = leg_terms(leg, before, after)
    assert t.delta == pytest.approx(0.5199388058 * 1.0 * 100, rel=1e-9)
    assert t.gamma == pytest.approx(0.5 * 0.0398443914 * 1.0 * 100, rel=1e-9)
    assert t.vega == pytest.approx(19.9221957047 * 0.01 * 100, rel=1e-9)
    assert t.theta == pytest.approx(-7.9688782819 / 252 * 100, rel=1e-9)
    # The actual move, from C(101, 0.21, 62/252) - C(100, 0.20, 63/252) by erf: 70.566.
    assert t.market == pytest.approx(70.5659322660, rel=1e-9)
    assert t.residual == pytest.approx(70.5659322660 - 70.7460426, abs=1e-6)  # vanna, volga
    assert abs(t.residual) < 0.003 * abs(t.market)
    assert identity_holds(t)

    short = leg_terms(OptionLeg("C", 100.0, 63, -2, leg.entry_price), before, after)
    for name in ("delta", "gamma", "vega", "theta", "market", "residual"):
        assert getattr(short, name) == pytest.approx(-2 * getattr(t, name), rel=1e-12)


def test_a_step_adds_its_legs_then_charges_its_cost() -> None:
    before, after = one_leg_chains(spot1=99.0, vol1=0.20)
    legs = (
        OptionLeg("P", 90.0, 63, -1, before.price("P", 90.0, 63)),
        OptionLeg("C", 110.0, 63, -1, before.price("C", 110.0, 63)),
    )
    per_leg = [leg_terms(leg, before, after) for leg in legs]
    market = sum(t.market for t in per_leg)
    step = attribute_step(4, Position(2, legs, 4), before, after, market - 12.5, 12.5)
    assert (step.day, step.action_id, step.n_legs) == (4, 2, 2)
    assert step.terms.cost == -12.5 and step.terms.carry == 0.0 and step.terms.vega == 0.0
    assert step.terms.delta == pytest.approx(sum(t.delta for t in per_leg), rel=1e-12)
    assert step.terms.residual == pytest.approx(sum(t.residual for t in per_leg), abs=1e-9)
    assert identity_holds(step.terms)

    flat = attribute_step(4, None, before, after, -3.0, 3.0)
    assert flat.n_legs == 0 and flat.terms.residual == 0.0 and flat.terms.cost == -3.0


# -- the identity and the residual on full episodes --------------------------- #


@pytest.mark.parametrize("premium", [0.0, 2.0])
@pytest.mark.parametrize("arm", SCRIPTED)
def test_the_identity_holds_and_the_residual_is_small_for_scripted_arms(
    arm: PolicyName, premium: float
) -> None:
    config = RunConfig(policy=arm, n_episodes=8, market=MarketConfig(iv_premium_vol_pts=premium))
    result = attribute_config(config)
    assert identity_holds(result.totals)
    for ep in result.episodes:
        assert identity_holds(ep.totals)
        assert ep.n_steps == config.env.episode_days == len(ep.steps)
        for s in ep.steps:
            assert identity_holds(s.terms)
            assert abs(s.terms.pnl - s.terms.market - s.terms.cost) < ACCOUNTING_TOL
            assert abs(s.terms.residual - (s.terms.market - s.terms.greeks)) < ACCOUNTING_TOL
            assert s.terms.carry == 0.0  # the book books no interest
            assert s.terms.vega == 0.0  # the synthetic surface never moves
    assert 0.0 < result.residual_share < RESIDUAL_SHARE_TOL
    assert result.totals.theta > 0 > result.totals.gamma  # a short-vol book's sign pattern


def test_the_flat_arm_decomposes_to_nothing() -> None:
    result = attribute_config(RunConfig(policy="flat", n_episodes=2))
    assert result.totals == Terms() and result.gross == 0.0 and result.residual_share == 0.0


def test_the_residual_is_a_remainder_it_shrinks_as_the_step_gets_relatively_smaller() -> None:
    # One trading day against a longer tenor is a smaller step in sigma sqrt(tau), so
    # the remainder must fall as tau grows: at least as sqrt(dt / tau) does (x0.5 per 4x
    # of tenor, x0.25 over 16x). Measured over 10 seeds: 0.0148 -> 0.0074 -> 0.0013.
    shares = []
    for tenors in ((21, 42), (84, 168), (336, 672)):
        market = MarketConfig(sigma=0.10, tenor_days=tenors, n_strikes_each_side=30)
        config = RunConfig(policy="delta_seller", n_episodes=10, market=market)
        shares.append(attribute_config(config).residual_share)
    assert shares[0] > shares[1] > shares[2] > 0.0
    assert shares[1] < 0.75 * shares[0] and shares[2] < 0.25 * shares[0]


class MovingSurface(SyntheticChainProvider):
    """The synthetic path under a surface whose level moves every day: vega has work."""

    def _chain(self, spot: float) -> Chain:
        chain = super()._chain(spot)
        shift = 0.02 * math.sin(0.7 * self.day)
        return Chain(
            chain.t, chain.spot, chain.r, chain.strikes, chain.expiry_days, chain.ivs + shift
        )


def test_vega_takes_the_surface_move_and_the_residual_stays_small() -> None:
    config = RunConfig(policy="delta_seller", n_episodes=6)
    result = attribute_config(config, provider=MovingSurface(config.market))
    steps = [s for e in result.episodes for s in e.steps if s.n_legs]
    assert any(abs(s.terms.vega) > 100.0 for s in steps)
    assert all(identity_holds(s.terms) for s in steps)
    assert result.residual_share < RESIDUAL_SHARE_TOL


def test_a_real_shaped_surface_runs_through_the_historical_provider() -> None:
    # Skew, a level random walk and a daily rate change: vanna and rho now leave the
    # expansion too, so the bound is looser than on the flat synthetic surface
    # (measured 3.3% to 4.1% on this store).
    rng = np.random.default_rng(3)
    n = 200
    level = 0.18 + np.cumsum(rng.normal(0.0, 0.005, n))
    spot = 5000.0 * np.exp(np.cumsum(rng.normal(0.0, level / np.sqrt(252))))
    k, tenor = np.asarray(K_GRID), np.asarray(TENOR_GRID, dtype=np.float64)
    ivs = level[:, None, None] - 0.3 * k[None, None, :] + 0.02 * np.sqrt(tenor / 365)[None, :, None]
    days = tuple(
        str(np.datetime64("2020-03-02") + np.timedelta64(i, "D")) for i in range(n)
    )  # in the eval range
    store = SurfaceStore(days, spot, 0.03 + rng.normal(0, 0.001, n), ivs)
    config = RunConfig(policy="always_on", n_episodes=4)
    result = attribute_config(
        config, provider=HistoricalChainProvider(config.market, store), source="optionmetrics"
    )
    assert result.source == "optionmetrics" and result.reproduces is None
    assert result.totals.vega != 0.0 and identity_holds(result.totals)
    assert result.residual_share < 2 * RESIDUAL_SHARE_TOL
    assert output_name(result) == f"{result.run_id}.optionmetrics.json"


# -- the run it decomposes ----------------------------------------------------- #


def snapshot(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run_dir.iterdir())}


@pytest.mark.parametrize("arm", SCRIPTED)
def test_a_run_directory_is_reproduced_episode_for_episode_and_left_alone(
    arm: PolicyName, tmp_path: Path
) -> None:
    run_dir = write_scripted(SMALL.model_copy(update={"policy": arm}), tmp_path / "run")
    before = snapshot(run_dir)
    result = attribute_run(run_dir)
    assert result.reproduces is True and all(e.reproduces for e in result.episodes)
    assert result.run_dir == str(run_dir) and len(result.episodes) == SMALL.n_episodes
    record = json.loads((run_dir / RUN_FILENAME).read_text())
    assert [e.totals.pnl for e in result.episodes] == [row["pnl"] for row in record["episodes"]]
    assert snapshot(run_dir) == before


def test_a_subset_of_seeds_reproduces_on_its_own(tmp_path: Path) -> None:
    run_dir = write_scripted(SMALL.model_copy(update={"policy": "random"}), tmp_path / "run")
    result = attribute_run(run_dir, seeds=[SMALL.seed + 2])
    assert [e.seed for e in result.episodes] == [SMALL.seed + 2] and result.reproduces is True
    with pytest.raises(ValueError, match="not episodes of this run"):
        attribute_run(run_dir, seeds=[SMALL.seed + SMALL.n_episodes])


def test_an_edited_run_is_refused_and_a_different_run_does_not_reproduce(tmp_path: Path) -> None:
    run_dir = write_scripted(SMALL, tmp_path / "run")
    rows = {row.seed: row for row in read_run(run_dir).episodes}
    assert attribute_config(SMALL, rows=rows).reproduces is True
    doctored = {s: r.model_copy(update={"pnl": r.pnl + 1.0}) for s, r in rows.items()}
    assert attribute_config(SMALL, rows=doctored).reproduces is False
    assert main([str(run_dir), "--out", str(tmp_path / "out")]) == 0
    record = json.loads((run_dir / RUN_FILENAME).read_text())
    record["mean_pnl"] = 1e9
    (run_dir / RUN_FILENAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="no longer match"):
        attribute_run(run_dir)


def test_a_run_from_another_platform_agrees_without_reproducing(tmp_path: Path) -> None:
    """The last digits a different libm rounds differently: agrees, not bit for bit."""
    from scripts.attribution import render

    run_dir = write_scripted(SMALL, tmp_path / "run")
    rows = {row.seed: row for row in read_run(run_dir).episodes}
    exact = attribute_config(SMALL, rows=rows)
    assert exact.reproduces is True and exact.agrees is True
    assert all(e.agrees for e in exact.episodes) and "(reproduces)" in render(exact)

    def nudged(rel: float) -> dict[int, EpisodeRow]:
        return {
            s: r.model_copy(update={"pnl": r.pnl + rel * max(1.0, abs(r.pnl))})
            for s, r in rows.items()
        }

    close = attribute_config(SMALL, rows=nudged(PNL_RTOL / 4))  # 3.5e-10 dollars on Dom's
    assert close.reproduces is False and close.agrees is True
    assert "(agrees to within PNL_RTOL, not bit for bit)" in render(close)
    far = attribute_config(SMALL, rows=nudged(PNL_RTOL * 4))
    assert far.reproduces is False and far.agrees is False and "DOES NOT AGREE" in render(far)
    fewer = {s: r.model_copy(update={"n_trades": r.n_trades + 1}) for s, r in rows.items()}
    assert attribute_config(SMALL, rows=fewer).agrees is False  # a different trade never agrees
    missing = attribute_config(SMALL, rows={})
    assert missing.reproduces is False and missing.agrees is False
    assert attribute_config(SMALL).agrees is None  # no run to compare


LEARNED = SMALL.model_copy(
    update={"policy": "ppo", "train": TrainConfig(total_timesteps=64, seed=2)}
)


def fake_loader(path: Path, config: RunConfig) -> Policy:
    assert path.name == POLICY_FILENAME and path.read_text().startswith("fake learner")
    assert config.train is not None
    return FakeLearned(config.train.seed)


def test_a_learned_run_is_attributed_with_its_saved_policy(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    result = attribute_run(tmp_path, load_policy=fake_loader)
    assert result.policy == "ppo" and result.reproduces is True
    (tmp_path / POLICY_FILENAME).unlink()
    with pytest.raises(ValueError, match=r"policy\.zip"):  # the manifest hashed it
        attribute_run(tmp_path, load_policy=fake_loader)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["artifacts"][POLICY_FILENAME]  # a manifest that never hashed the policy
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="is missing"):
        attribute_run(tmp_path, load_policy=fake_loader)


def test_a_learned_arm_needs_its_policy() -> None:
    with pytest.raises(ValueError, match="trained policy"):
        attribute_config(LEARNED)
    result = attribute_config(LEARNED, policy=FakeLearned(2))
    assert result.reproduces is None and len(result.episodes) == LEARNED.n_episodes


def test_the_tap_must_be_bound_before_the_path_moves() -> None:
    from optspread.attribution import _Tap
    from optspread.rng import SeededStepRNG

    tap = _Tap(SyntheticChainProvider(MarketConfig()))
    tap.reset(SeededStepRNG(0))
    assert tap.n_days == 63 and not tap.done
    with pytest.raises(RuntimeError, match="bound"):
        tap.step()


# -- the JSON the dashboard reads --------------------------------------------- #


def test_the_json_round_trips_and_can_leave_the_steps_out(tmp_path: Path) -> None:
    result = attribute_config(SMALL)
    path = write_attribution(result, tmp_path / "out" / output_name(result))
    assert path.name == f"{result.run_id}.json"
    assert read_attribution(path) == result
    raw = json.loads(path.read_text())
    assert raw["components"] == ["delta", "gamma", "vega", "theta", "carry", "cost", "residual"]
    assert set(raw["episodes"][0]["steps"][0]["terms"]) == set(Terms.model_fields)
    lean = json.loads(dump_attribution(result, steps=False))
    assert lean["totals"] == raw["totals"] and lean["episodes"][0]["steps"] == []


def test_attribution_is_never_written_into_a_run_directory(tmp_path: Path) -> None:
    run_dir = write_scripted(SMALL, tmp_path / "run")
    with pytest.raises(ValueError, match="run directory"):
        write_attribution(attribute_run(run_dir), run_dir / "attribution.json")


def test_another_schema_version_is_refused(tmp_path: Path) -> None:
    result = attribute_config(RunConfig(policy="flat", n_episodes=1))
    path = tmp_path / "a.json"
    path.write_text(result.model_copy(update={"schema_version": 99}).model_dump_json())
    with pytest.raises(ValueError, match="schema v99"):
        read_attribution(path)


def test_the_default_root_is_outside_every_run_and_can_be_moved(tmp_path: Path) -> None:
    assert default_root({}) == REPO_ROOT / "attribution"
    assert default_root({ATTRIBUTION_ENV: str(tmp_path)}) == tmp_path


# -- the command ------------------------------------------------------------- #


def test_the_command_attributes_runs_and_configs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    write_scripted(SMALL, runs / "a")
    write_scripted(SMALL.model_copy(update={"policy": "always_on"}), runs / "b")
    out = tmp_path / "attr"
    assert main([str(runs), "--out", str(out), "--first", "2"]) == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 2
    assert all(len(read_attribution(f).episodes) == 2 for f in files)
    assert "reproduces" in capsys.readouterr().out

    assert main(["--policy", "random", "--episodes", "2", "--out", str(out), "--no-steps"]) == 0
    config = tmp_path / "cfg.json"
    config.write_text(SMALL.model_dump_json())
    assert main(["--config", str(config), "--out", str(out), "--seeds", "8001"]) == 0
    manifest = runs / "a" / "manifest.json"
    assert main(["--config", str(manifest), "--out", str(out)]) == 0
    from_config = read_attribution(out / f"{config_hash(SMALL)}-s{SMALL.seed}.json")
    assert from_config.reproduces is None and from_config.run_dir is None  # no run compared


def test_the_command_says_no(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--policy", "flat"])
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--source", "optionmetrics"])
    with pytest.raises(SystemExit):
        main(["--policy", "flat", "--first", "0"])
    assert main([str(tmp_path / "absent")]) == 1
    # conftest points VEGALAB_DATA at an empty folder, so the real source is unavailable.
    assert main(["--policy", "flat", "--source", "optionmetrics", "--out", str(tmp_path)]) == 1
    assert "cannot attribute" in capsys.readouterr().err
    execute_run(LEARNED, tmp_path / "ppo", fake_trainer)
    monkeypatch.setitem(sys.modules, "torch", None)  # the train extra, absent
    assert main([str(tmp_path / "ppo"), "--out", str(tmp_path / "out")]) == 1
    assert "train extra" in capsys.readouterr().err
