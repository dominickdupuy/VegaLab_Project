"""Dominick Dupuy: the action mix, which structures a policy trades (the SDP's question).

Four claims, each checked against something that does not come from the module:

* the re-simulation is the run: every episode equals its ``run.json`` row (bit for
  bit here, on the machine that wrote it), the run directory is left exactly as it
  was, and a run that was edited or made on other data is refused;
* the record is the book's: the trades it implies (opens, policy closes, breach and
  horizon closes; expiry settles in cash and is no trade) add up to the env's own
  ``n_trades``, and the days its holdings span are the days a structure was held;
* the scripted arms trade what their code says: the fixed-delta seller opens iron
  condors at the configured deltas and tenor and rolls them, always-on holds to
  expiry, flat never trades, and an account too small for a structure is refused;
* a learned run is read with its saved policy (the stand-in learner here), a missing
  learner is refused with a message that says what to install, and a grid gives the
  mix per cell and pooled per arm and premium.

Every number here is synthetic.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread import webapi
from optspread.actionmix import (
    ACTIONMIX_ENV,
    EVENTS,
    PNL_RTOL,
    REPO_ROOT,
    GridActionMix,
    LearnerUnavailableError,
    RunActionMix,
    _Tap,
    default_root,
    dump_mix,
    find_grid,
    mix_config,
    mix_grid,
    mix_run,
    output_name,
    read_grid_mix,
    read_mix,
    summarize_mix,
    write_mix,
)
from optspread.actions import ACTION_NAMES, FLAT, SHORT_STRANGLE
from optspread.artifacts import RUN_FILENAME, read_run, write_run
from optspread.config import (
    CostConfig,
    EnvConfig,
    MarketConfig,
    PolicyName,
    RunConfig,
    TrainConfig,
)
from optspread.harness import run_episodes, summarize
from optspread.manifest import MANIFEST_FILENAME
from optspread.market import SyntheticChainProvider
from optspread.orchestrate import GridConfig, execute_run, run_grid
from optspread.policies import AlwaysOnPolicy, Policy, RandomPolicy
from optspread.rng import SeededStepRNG
from optspread.train import POLICY_FILENAME
from scripts.action_mix import main

SMALL = RunConfig(
    policy="delta_seller",
    seed=8000,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21), iv_premium_vol_pts=1.0),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
# An account just large enough for one short strangle: opens are refused on some days
# and a structure that clears the check is often liquidated by the breach rule later
# (measured over 6 seeds: 14 opens, 274 refusals, 12 breaches, 1 expiry, 1 horizon).
TIGHT = RunConfig(
    policy="always_on",
    n_episodes=6,
    env=EnvConfig(initial_cash=55_000.0),
    cost=CostConfig(enabled=True),
)
SCRIPTED: tuple[PolicyName, ...] = ("always_on", "delta_seller", "random", "flat")
LEARNED = SMALL.model_copy(
    update={"policy": "ppo", "train": TrainConfig(total_timesteps=64, seed=2)}
)


def write_scripted(config: RunConfig, run_dir: Path) -> Path:
    episodes = run_episodes(config)
    write_run(run_dir, config, summarize(episodes), episodes)
    return run_dir


def snapshot(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def fake_loader(path: Path, config: RunConfig) -> Policy:
    assert path.name == POLICY_FILENAME and path.read_text().startswith("fake learner")
    assert config.train is not None
    return FakeLearned(config.train.seed)


def book_consistent(result: RunActionMix) -> None:
    """The record against the env's own counts, episode by episode."""
    for ep in result.episodes:
        assert len(ep.actions) == len(ep.held) == len(ep.events) == ep.n_steps
        for action, held, event in zip(ep.actions, ep.held, ep.events, strict=True):
            if event in ("hold", "open", "switch"):
                assert held == action != FLAT
            else:
                assert held == FLAT
        trades = ep.n_opens + sum(h.exit != "expiry" for h in ep.holdings)
        assert trades == ep.n_trades
        assert ep.n_opens == len(ep.holdings)
        assert sum(h.days for h in ep.holdings) == sum(h != FLAT for h in ep.held)
        for h in ep.holdings:
            assert h.days == h.exit_day - h.open_day > 0
            assert ep.held[h.open_day] == h.action_id
            assert h.structure == ACTION_NAMES[h.action_id]
            if h.exit == "horizon":
                assert h.exit_day == ep.n_steps
            if h.exit == "expiry":
                assert h.exit_day == h.expiry_day


# -- the run it reads ---------------------------------------------------------- #


@pytest.mark.parametrize("arm", SCRIPTED)
def test_a_run_directory_is_reproduced_episode_for_episode_and_left_alone(
    arm: PolicyName, tmp_path: Path
) -> None:
    run_dir = write_scripted(SMALL.model_copy(update={"policy": arm}), tmp_path / "run")
    before = snapshot(run_dir)
    result = mix_run(run_dir)
    assert result.reproduces is True and result.agrees is True
    assert result.max_abs_pnl_diff == 0.0
    assert all(e.reproduces and e.pnl_diff == 0.0 for e in result.episodes)
    record = json.loads((run_dir / RUN_FILENAME).read_text())
    assert [e.pnl for e in result.episodes] == [row["pnl"] for row in record["episodes"]]
    assert result.run_dir == str(run_dir) and result.source == "synthetic"
    book_consistent(result)
    assert snapshot(run_dir) == before


def test_agreement_allows_the_last_digits_and_nothing_more(tmp_path: Path) -> None:
    run_dir = write_scripted(SMALL, tmp_path / "run")
    rows = {row.seed: row for row in read_run(run_dir).episodes}
    nudged = {
        s: r.model_copy(update={"pnl": r.pnl + 0.1 * PNL_RTOL * max(1.0, abs(r.pnl))})
        for s, r in rows.items()
    }
    near = mix_config(SMALL, rows=nudged)
    assert near.reproduces is False and near.agrees is True
    assert 0.0 < (near.max_abs_pnl_diff or 0.0) < 1e-6
    off = {s: r.model_copy(update={"pnl": r.pnl + 1.0}) for s, r in rows.items()}
    assert mix_config(SMALL, rows=off).agrees is False
    traded = {s: r.model_copy(update={"n_trades": r.n_trades + 1}) for s, r in rows.items()}
    assert mix_config(SMALL, rows=traded).agrees is False
    missing = mix_config(SMALL, rows={SMALL.seed: rows[SMALL.seed]})
    assert missing.agrees is False and missing.episodes[1].pnl_diff is None


def test_a_subset_of_seeds_and_an_edited_run(tmp_path: Path) -> None:
    run_dir = write_scripted(SMALL.model_copy(update={"policy": "random"}), tmp_path / "run")
    result = mix_run(run_dir, seeds=[SMALL.seed + 2])
    assert [e.seed for e in result.episodes] == [SMALL.seed + 2] and result.reproduces is True
    with pytest.raises(ValueError, match="not episodes of this run"):
        mix_run(run_dir, seeds=[SMALL.seed + SMALL.n_episodes])
    record = json.loads((run_dir / RUN_FILENAME).read_text())
    record["mean_pnl"] = 1e9
    (run_dir / RUN_FILENAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="no longer match"):
        mix_run(run_dir)


def test_a_run_is_read_on_its_recorded_source_and_refused_on_another_panel(
    tmp_path: Path,
) -> None:
    episodes = run_episodes(SMALL, source="optionmetrics_stub")
    run_dir = tmp_path / "stub"
    write_run(run_dir, SMALL, summarize(episodes), episodes, source="optionmetrics_stub")
    result = mix_run(run_dir)
    assert result.source == "optionmetrics_stub" and result.reproduces is True
    assert output_name(result) == f"{result.run_id}.optionmetrics_stub.json"
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    manifest["data"]["panel"] = {"panel/surface_grid.parquet": "0" * 64}
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="panel is not the one"):
        mix_run(run_dir)


# -- what the scripted arms trade ---------------------------------------------- #


def test_the_fixed_delta_seller_opens_condors_at_its_deltas_and_rolls_them() -> None:
    config = RunConfig(policy="delta_seller", n_episodes=6, cost=CostConfig(enabled=True))
    result = mix_config(config)
    s = result.summary
    book_consistent(result)
    assert s.n_episodes == 6 and s.n_steps == 6 * 63
    assert set(s.actions) == set(ACTION_NAMES) and set(s.events) == set(EVENTS)
    assert s.actions["SHORT_STRANGLE"] == s.actions["PUT_CREDIT_SPREAD"] == 0
    assert s.structures["IRON_CONDOR"].n_opens == s.n_opens and s.n_refused == 0
    # 42-day condors closed at 5 days to expiry: open on day 0, roll on day 37, reopen
    # on day 38 and close at the horizon, so two opens and one idle day per episode.
    assert s.opens_per_episode == 2.0 and s.held["FLAT"] == 6
    assert s.exits == {"policy": 6, "expiry": 0, "margin": 0, "horizon": 6}
    ic = s.structures["IRON_CONDOR"]
    assert ic.mean_tenor_days == 42.0 and ic.open_share == 1.0
    assert ic.mean_short_abs_delta == pytest.approx(config.env.short_delta, abs=0.02)
    assert ic.mean_long_abs_delta == pytest.approx(config.env.wing_delta, abs=0.01)
    assert (ic.mean_credit or 0.0) > 0.0 and (ic.mean_margin or 0.0) > 0.0
    assert s.structures["SHORT_STRANGLE"].mean_tenor_days is None
    first = result.episodes[0].holdings[0]
    assert (first.open_day, first.exit_day, first.exit) == (0, 37, "policy")
    assert [leg.qty for leg in first.legs] == [-1, -1, 1, 1]
    assert [leg.right for leg in first.legs] == ["P", "C", "P", "C"]
    assert first.net_delta == pytest.approx(sum(leg.qty * leg.delta for leg in first.legs))
    assert s.by_day[0].opens == 6 and s.by_day[37].actions[FLAT] == 6
    assert all(sum(d.actions) == sum(d.held) == 6 for d in s.by_day)
    assert s.holding_days is not None and s.holding_days.max == 37.0


def test_always_on_holds_to_expiry_and_flat_never_trades() -> None:
    always = mix_config(RunConfig(policy="always_on", n_episodes=3))
    book_consistent(always)
    assert always.summary.exits["expiry"] == always.summary.n_opens - 3
    assert always.summary.events["idle"] == 0
    flat = mix_config(RunConfig(policy="flat", n_episodes=2))
    assert flat.summary.events["idle"] == flat.summary.n_steps == 126
    assert flat.summary.held_share["FLAT"] == 1.0 and flat.summary.holding_days is None
    assert all(st.n_opens == 0 and st.open_share == 0.0 for st in flat.summary.structures.values())


def test_a_small_account_is_refused_and_breached_and_the_record_says_which() -> None:
    result = mix_config(TIGHT, policy=AlwaysOnPolicy(SHORT_STRANGLE))
    s = result.summary
    book_consistent(result)
    assert s.n_refused > 0 and s.events["refused"] == s.n_refused
    assert s.refused_share == s.n_refused / (s.n_opens + s.n_refused)
    assert s.exits["margin"] > 0 and s.structures["SHORT_STRANGLE"].mean_long_abs_delta is None
    margin = [h for e in result.episodes for h in e.holdings if h.exit == "margin"]
    assert all(h.exit_day < h.expiry_day for h in margin)
    assert sum(d.refused for d in s.by_day) == s.n_refused


def test_a_random_policy_shows_every_event_and_the_book_still_adds_up() -> None:
    config = TIGHT.model_copy(update={"policy": "random", "n_episodes": 8})
    result = mix_config(config, policy=RandomPolicy())
    book_consistent(result)
    events = result.summary.events
    assert all(events[name] > 0 for name in ("idle", "hold", "open", "close", "switch"))
    assert events["refused"] > 0
    assert sum(result.summary.action_share.values()) == pytest.approx(1.0)


def test_a_learned_run_is_read_with_its_saved_policy(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path / "ppo", fake_trainer)
    result = mix_run(tmp_path / "ppo", load_policy=fake_loader)
    assert result.policy == "ppo" and result.reproduces is True
    # Training seed 2 is the stand-in's condor seller (seed 3 sells put spreads).
    assert result.summary.actions["IRON_CONDOR"] > 0 == result.summary.actions["PUT_CREDIT_SPREAD"]
    other = LEARNED.model_copy(update={"train": TrainConfig(total_timesteps=64, seed=3)})
    spreads = mix_config(other, policy=FakeLearned(3)).summary.structures["PUT_CREDIT_SPREAD"]
    assert spreads.open_share == 1.0 and spreads.mean_long_abs_delta is not None
    with pytest.raises(ValueError, match="trained policy"):
        mix_config(LEARNED)


def test_a_missing_learner_or_policy_is_refused_with_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "ppo"
    execute_run(LEARNED, run_dir, fake_trainer)
    monkeypatch.setitem(sys.modules, "torch", None)  # the train extra, absent
    with pytest.raises(LearnerUnavailableError, match=r"stable-baselines3.*--arms delta_seller"):
        mix_run(run_dir)
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    del manifest["artifacts"][POLICY_FILENAME]  # a manifest that never hashed the policy
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    (run_dir / POLICY_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="is missing"):
        mix_run(run_dir, load_policy=fake_loader)


def test_the_tap_must_be_bound_before_the_path_moves() -> None:
    tap = _Tap(SyntheticChainProvider(MarketConfig()))
    tap.reset(SeededStepRNG(0))
    assert tap.n_days == 63 and not tap.done
    with pytest.raises(RuntimeError, match="bound"):
        tap.step()


# -- a grid -------------------------------------------------------------------- #

GRID_BASE = SMALL.model_copy(update={"seed": 7000, "n_episodes": 3})
GRID = GridConfig(
    base=GRID_BASE,
    premiums=(0.0, 2.0),
    n_seeds=2,
    train=TrainConfig(total_timesteps=64, seed=10),
)


@pytest.fixture()
def grid_root(tmp_path: Path) -> Path:
    root = tmp_path / "sweep"
    run_grid(GRID, root, trainer=fake_trainer)
    return root


def test_a_grid_gives_the_mix_per_cell_and_pooled_per_arm_and_premium(grid_root: Path) -> None:
    before = snapshot(grid_root)
    grid, cells = mix_grid(grid_root, load_policy=fake_loader)
    assert grid.reproduces is True and grid.agrees is True
    assert len(cells) == len(grid.cells) == 2 * 2 * 2 and len(grid.pooled) == 2 * 2
    assert grid.arms == ("delta_seller", "ppo") and grid.premiums == (0.0, 2.0)
    for p in grid.pooled:
        mine = [c for c in grid.cells if (c.premium, c.arm) == (p.premium, p.arm)]
        assert p.run_ids == tuple(c.run_id for c in mine)
        assert p.summary.n_episodes == 2 * GRID_BASE.n_episodes
        assert p.summary.n_opens == sum(c.summary.n_opens for c in mine)
    seller = grid.pooled_cell(2.0, "delta_seller").summary.structures["IRON_CONDOR"]
    assert seller.open_share == 1.0
    # The stand-in learner's blocks are a condor seller (seed 10) and a spread seller (11).
    ppo = grid.pooled_cell(2.0, "ppo").summary
    assert (
        ppo.structures["IRON_CONDOR"].n_opens > 0
        and ppo.structures["PUT_CREDIT_SPREAD"].n_opens > 0
    )
    with pytest.raises(KeyError):
        grid.pooled_cell(1.0, "ppo")
    assert snapshot(grid_root) == before


@pytest.mark.spawns_processes
def test_a_grid_can_keep_the_scripted_arm_alone_and_run_on_workers(grid_root: Path) -> None:
    serial, _ = mix_grid(grid_root, arms=["delta_seller"], first=2)
    assert serial.arms == ("delta_seller",) and all(c.arm == "delta_seller" for c in serial.cells)
    assert all(p.summary.n_episodes == 2 * 2 for p in serial.pooled)
    parallel, _ = mix_grid(
        grid_root / f"grid-{serial.grid_hash}.json", arms=["delta_seller"], first=2, workers=2
    )
    assert parallel.model_copy(update={"grid_file": serial.grid_file}) == serial
    with pytest.raises(ValueError, match="not in this grid"):
        mix_grid(grid_root, arms=["random"])
    shutil.rmtree(grid_root / serial.cells[0].run_id)
    with pytest.raises(FileNotFoundError, match="is not under"):
        mix_grid(grid_root, arms=["delta_seller"])


def test_finding_the_grid_file(grid_root: Path, tmp_path: Path) -> None:
    (only,) = grid_root.glob("grid-*.json")
    assert find_grid(grid_root) == only == find_grid(only)
    with pytest.raises(FileNotFoundError, match="none"):
        find_grid(tmp_path / "nowhere")
    (grid_root / "grid-0000000000000000.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="expected one"):
        find_grid(grid_root)


# -- the JSON ------------------------------------------------------------------ #


def test_the_json_round_trips_and_can_be_lean(tmp_path: Path, grid_root: Path) -> None:
    result = mix_config(SMALL)
    path = write_mix(result, tmp_path / "out" / output_name(result))
    assert path.name == f"{result.run_id}.json" and read_mix(path) == result
    raw = json.loads(path.read_text())
    assert raw["action_names"] == list(ACTION_NAMES) and raw["schema_version"] == 1
    assert set(raw["summary"]["structures"]) == {
        "IRON_CONDOR",
        "SHORT_STRANGLE",
        "PUT_CREDIT_SPREAD",
    }
    lean = json.loads(dump_mix(result, episodes=False))
    assert lean["summary"] == raw["summary"] and lean["episodes"][0]["actions"] == []
    with pytest.raises(ValueError, match="lean episode"):
        summarize_mix(result.lean().episodes)
    grid, _ = mix_grid(grid_root, arms=["delta_seller"], first=1)
    gpath = write_mix(grid, tmp_path / "out" / output_name(grid))
    assert gpath.name == f"grid-{grid.grid_hash}.json" and read_grid_mix(gpath) == grid
    assert dump_mix(grid, episodes=False) == dump_mix(grid)


def test_the_mix_is_never_written_into_a_run_or_a_sweep_root(
    tmp_path: Path, grid_root: Path
) -> None:
    run_dir = write_scripted(SMALL, tmp_path / "run")
    result = mix_run(run_dir)
    with pytest.raises(ValueError, match="run directory"):
        write_mix(result, run_dir / "mix.json")
    with pytest.raises(ValueError, match="holds run directories"):
        write_mix(result, grid_root / "mix.json")


def test_another_schema_version_is_refused(tmp_path: Path) -> None:
    result = mix_config(RunConfig(policy="flat", n_episodes=1))
    path = tmp_path / "a.json"
    path.write_text(result.model_copy(update={"schema_version": 99}).model_dump_json())
    with pytest.raises(ValueError, match="schema v99"):
        read_mix(path)
    grid = GridActionMix(
        schema_version=99,
        grid_hash="0",
        grid_file="",
        source="synthetic",
        arms=(),
        premiums=(),
        n_seeds=0,
        reproduces=True,
        agrees=True,
        cells=(),
        pooled=(),
    )
    path.write_text(grid.model_dump_json())
    with pytest.raises(ValueError, match="schema v99"):
        read_grid_mix(path)


def test_the_default_root_is_outside_every_run_and_can_be_moved(tmp_path: Path) -> None:
    assert default_root({}) == REPO_ROOT / "actionmix"
    assert default_root({ACTIONMIX_ENV: str(tmp_path)}) == tmp_path
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "actionmix/" in ignored


# -- the command and make -------------------------------------------------------- #


def test_the_command_reads_runs_grids_and_configs(
    tmp_path: Path, grid_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    write_scripted(SMALL, runs / "a")
    write_scripted(SMALL.model_copy(update={"policy": "always_on"}), runs / "b")
    out = tmp_path / "mix"
    assert main([str(runs), "--out", str(out), "--first", "2"]) == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 2 and all(len(read_mix(f).episodes) == 2 for f in files)
    assert "reproduces" in capsys.readouterr().out

    assert (
        main(
            ["--grid", str(grid_root), "--arms", "delta_seller", "--first", "1", "--out", str(out)]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "grid " in printed and "p=2" in printed and "IC" in printed
    assert len(list(out.glob("grid-*.json"))) == 1

    lean = tmp_path / "lean"
    assert main(["--policy", "random", "--episodes", "2", "--out", str(lean), "--lean"]) == 0
    (only,) = lean.glob("*.json")
    assert read_mix(only).episodes[0].actions == () and read_mix(only).reproduces is None
    config = tmp_path / "cfg.json"
    config.write_text(SMALL.model_dump_json())
    assert main(["--config", str(config), "--out", str(out), "--seeds", "8001"]) == 0
    assert main(["--config", str(runs / "a" / MANIFEST_FILENAME), "--out", str(out)]) == 0
    stub = tmp_path / "stub"
    assert (
        main(
            [
                "--policy",
                "flat",
                "--episodes",
                "1",
                "--source",
                "optionmetrics_stub",
                "--out",
                str(stub),
            ]
        )
        == 0
    )
    assert next(stub.glob("*.json")).name.endswith(".optionmetrics_stub.json")


def test_the_command_says_no(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for argv in (
        [],
        [str(tmp_path), "--policy", "flat"],
        [str(tmp_path), "--source", "optionmetrics"],
        ["--grid", str(tmp_path), "--source", "optionmetrics"],
        ["--policy", "flat", "--first", "0"],
        ["--grid", str(tmp_path), "--seeds", "1"],
        ["--policy", "flat", "--arms", "ppo"],
        ["--grid", str(tmp_path), "--workers", "0"],
    ):
        with pytest.raises(SystemExit):
            main(argv)
    assert main([str(tmp_path / "absent")]) == 1
    # conftest points VEGALAB_DATA at an empty folder, so the real source is unavailable.
    assert main(["--policy", "flat", "--source", "optionmetrics", "--out", str(tmp_path)]) == 1
    assert "cannot read the action mix" in capsys.readouterr().err
    assert main(["--grid", str(tmp_path / "nowhere")]) == 1
    run_dir = write_scripted(SMALL, tmp_path / "runs" / "a")
    record = json.loads((run_dir / RUN_FILENAME).read_text())
    record["episodes"][0]["pnl"] += 1.0  # rehashed below, so it reads as a different run
    (run_dir / RUN_FILENAME).write_text(json.dumps(record))
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    from optspread.manifest import file_sha256

    manifest["artifacts"][RUN_FILENAME] = file_sha256(run_dir / RUN_FILENAME)
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert main([str(run_dir), "--out", str(tmp_path / "out")]) == 1
    assert "DOES NOT REPRODUCE" in capsys.readouterr().out
    execute_run(LEARNED, tmp_path / "ppo", fake_trainer)
    monkeypatch.setitem(sys.modules, "torch", None)  # the train extra, absent
    assert main([str(tmp_path / "ppo"), "--out", str(tmp_path / "out")]) == 1
    assert "stable-baselines3" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_the_make_target_passes_its_args_through(tmp_path: Path) -> None:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    (tmp_path / "actionmix").mkdir()  # the folder it writes must not make it up to date
    out = subprocess.run(
        [
            "make",
            "-n",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "actionmix",
            "ARGS=--grid /tmp/r",
            "PY=py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "py -m scripts.action_mix --grid /tmp/r"


# -- the dashboard's payload ----------------------------------------------------- #


@pytest.fixture()
def mix_root(tmp_path: Path, grid_root: Path) -> Path:
    out = tmp_path / "mixout"
    grid, cells = mix_grid(grid_root, arms=["delta_seller"], first=1)
    write_mix(grid, out / output_name(grid))
    for cell in cells:
        write_mix(cell, out / output_name(cell))
    return out


def test_the_payload_reads_a_run_or_a_grid_and_says_what_is_missing(mix_root: Path) -> None:
    (grid_file,) = mix_root.glob("grid-*.json")
    grid_hash = grid_file.stem.removeprefix("grid-")
    run_file = next(p for p in mix_root.glob("*.json") if p != grid_file)
    rid = run_file.stem
    run = webapi.action_mix(mix_root, run=f"sweeps/tiny/{rid}")
    assert run.kind == "run" and run.run is not None and run.grid is None
    assert run.run.run_id == rid and run.run.episodes[0].actions == ()  # lean
    assert run.run.summary == read_mix(run_file).summary and run.path == str(run_file)
    grid = webapi.action_mix(mix_root, grid=grid_hash)
    assert grid.kind == "grid" and grid.grid is not None and grid.grid.grid_hash == grid_hash
    assert webapi.action_mix(mix_root, grid=grid_file.name).grid == grid.grid
    with pytest.raises(KeyError):
        webapi.action_mix(mix_root, run="0123abcd-s1")
    with pytest.raises(KeyError):
        webapi.action_mix(mix_root, grid="0123abcd")
    with pytest.raises(KeyError):
        webapi.action_mix(mix_root, run=rid, source="optionmetrics")
    for kwargs in (
        {},
        {"run": rid, "grid": grid_hash},
        {"run": "../../etc/passwd"},
        {"grid": "../x"},
        {"run": rid, "source": "elsewhere"},
    ):
        with pytest.raises(ValueError):
            webapi.action_mix(mix_root, **kwargs)


def test_the_payload_defaults_to_the_actionmix_root(
    mix_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ACTIONMIX_ENV, str(mix_root))
    (grid_file,) = mix_root.glob("grid-*.json")
    assert webapi.action_mix(grid=grid_file.name).path == str(grid_file)


def test_the_route_serves_the_payload(
    mix_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_docs: Path
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from optspread.jobs import JobManager
    from optspread.server import Settings, create_app

    monkeypatch.setenv(ACTIONMIX_ENV, str(mix_root))
    runs, web = tmp_path / "runs", tmp_path / "web"
    runs.mkdir()
    web.mkdir()
    settings = Settings(runs, gate_docs, tmp_path / "data", web, REPO_ROOT / "README.md")
    manager = JobManager(runs, executor=ThreadPoolExecutor(1))
    (grid_file,) = mix_root.glob("grid-*.json")
    with TestClient(create_app(settings, manager), base_url="http://127.0.0.1") as client:
        ok = client.get("/api/runs/actionmix", params={"grid": grid_file.name})
        assert ok.status_code == 200 and ok.json()["kind"] == "grid"
        assert ok.json()["grid"]["pooled"][0]["arm"] == "delta_seller"
        missing = client.get("/api/runs/actionmix", params={"run": "0123abcd-s1"})
        assert missing.status_code == 404 and "make actionmix" in missing.json()["detail"]
        assert client.get("/api/runs/actionmix").status_code == 422


def test_a_config_on_the_real_source_is_mixed_on_its_evaluation_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The docstring says "its evaluation range": the eval split, not the source's
    default, train (data audit MINOR-9), from the function and from the command."""
    import optspread.actionmix as actionmix
    import scripts.action_mix as command
    from optspread.data import make_source

    asked: list[str] = []

    def spy(name: str, config: MarketConfig | None = None, *, split: str = "train") -> object:
        asked.append(split)
        return make_source("synthetic", config, split=split)  # type: ignore[arg-type]

    monkeypatch.setattr(actionmix, "make_source", spy)
    monkeypatch.setattr(command, "make_source", spy)
    mix_config(RunConfig(policy="flat", n_episodes=1), source="optionmetrics")
    out = tmp_path / "mix"
    args = ["--policy", "flat", "--episodes", "1", "--source", "optionmetrics", "--out", str(out)]
    assert main(args) == 0
    assert asked == ["eval", "eval"]
