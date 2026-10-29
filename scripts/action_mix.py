"""Which structures a policy trades: the action mix of runs, a grid, or a config.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.action_mix runs/pilot/<run id>             one run
    python -m scripts.action_mix runs/pilot --first 50           a folder of runs
    python -m scripts.action_mix --grid <root> --arms delta_seller --workers 8
    python -m scripts.action_mix --config cfg.json               a RunConfig (or a manifest)
    python -m scripts.action_mix --policy delta_seller --premium 2 --episodes 10

A run directory is re-simulated from its manifest, on the source it records, checked
against its ``run.json`` episode by episode, and never written into; a ppo run needs
its ``policy.zip`` and the ``train`` extra, and is refused with a message that says
so when Stable-Baselines3 cannot be imported. ``--grid`` takes a sweep root (or its
``grid-<hash>.json``) and reads every cell of the arms kept (``--arms``), writing each
cell's result and the grid's, which gives the mix per cell and pooled per arm and
premium. A config (or the ``--policy`` defaults) runs on the synthetic world unless
``--source`` names another. Results go to ``<out>/<run id>.json`` and
``<out>/grid-<hash>.json`` (``.<source>.json`` off the synthetic world), ``--out``
defaulting to ``optspread.actionmix.default_root()``; the schema is the module
docstring of ``optspread.actionmix``. ``--lean`` leaves out each episode's
day-by-day lists and holdings. Exit status 1 when a run does not agree with its
``run.json`` or nothing was read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from optspread.actionmix import (
    STRUCTURES,
    GridActionMix,
    LearnerUnavailableError,
    MixSummary,
    RunActionMix,
    default_root,
    mix_config,
    mix_grid,
    mix_run,
    output_name,
    write_mix,
)
from optspread.config import ArmName, MarketConfig, PolicyName, RunConfig
from optspread.data import DataUnavailableError, SourceName, make_source
from optspread.manifest import RunManifest
from optspread.replay import find_runs
from scripts.attribution import read_config

SHORT = {"FLAT": "flat", "IRON_CONDOR": "IC", "SHORT_STRANGLE": "SS", "PUT_CREDIT_SPREAD": "PCS"}


def _verdict(reproduces: bool | None, agrees: bool | None) -> str:
    if reproduces:
        return "reproduces"
    if agrees:
        return "agrees to 1e-9, not bit for bit"
    return "no run to compare" if agrees is None else "DOES NOT REPRODUCE"


def _mix_line(s: MixSummary) -> str:
    held = "  ".join(f"{SHORT[k]} {v:.0%}" for k, v in s.held_share.items())
    days = "-" if s.holding_days is None else f"{s.holding_days.mean:.1f}"
    return (
        f"held {held} | opens/ep {s.opens_per_episode:.2f}  refused/ep "
        f"{s.refused_per_episode:.2f}  holding days {days}"
    )


def _opens_line(s: MixSummary) -> str:
    parts = []
    for name in STRUCTURES:
        st = s.structures[name]
        if st.n_opens:
            parts.append(
                f"{SHORT[name]} {st.n_opens} (tenor {st.mean_tenor_days:.0f}d, "
                f"short |d| {st.mean_short_abs_delta:.3f})"
            )
    exits = ", ".join(f"{k} {v}" for k, v in s.exits.items() if v)
    return f"opens {'; '.join(parts) or 'none'} | exits {exits or 'none'}"


def render(result: RunActionMix) -> str:
    diff = "" if result.max_abs_pnl_diff is None else f"; max |dP&L| {result.max_abs_pnl_diff:.2g}"
    return (
        f"{result.run_id}  {result.policy}  premium {result.premium_vol_pts:g}  "
        f"{result.source}  {len(result.episodes)} episodes  "
        f"({_verdict(result.reproduces, result.agrees)}{diff})\n"
        f"    {_mix_line(result.summary)}\n"
        f"    {_opens_line(result.summary)}"
    )


def render_grid(grid: GridActionMix) -> str:
    """Per premium and arm, the pooled mix: the table the question is read from."""
    lines = [f"grid {grid.grid_hash}  {grid.source}  ({_verdict(grid.reproduces, grid.agrees)})"]
    for p in grid.pooled:
        lines.append(f"  p={p.premium:<5g} {p.arm:<13} {_mix_line(p.summary)}")
        lines.append(f"  {'':<20}{_opens_line(p.summary)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("paths", type=Path, nargs="*", help="run directories or their folder")
    parser.add_argument("--grid", type=Path, help="a sweep root or its grid-<hash>.json")
    parser.add_argument("--arms", nargs="+", choices=get_args(ArmName), help="grid arms kept")
    parser.add_argument("--workers", type=int, default=1, help="processes for --grid")
    parser.add_argument("--config", type=Path, help="a RunConfig JSON file or a manifest.json")
    parser.add_argument(
        "--policy", choices=get_args(PolicyName), help="a default RunConfig for this scripted arm"
    )
    parser.add_argument("--premium", type=float, default=0.0, help="planted vol points")
    parser.add_argument("--episodes", type=int, default=10, help="episodes for --policy")
    parser.add_argument("--seed", type=int, default=0, help="first seed for --policy")
    parser.add_argument("--seeds", type=int, nargs="+", help="read only these seeds")
    parser.add_argument("--first", type=int, help="read only the first N episodes of each run")
    parser.add_argument(
        "--source", choices=get_args(SourceName), default="synthetic", help="for a config"
    )
    parser.add_argument("--out", type=Path, help="output folder (default: default_root())")
    parser.add_argument("--lean", action="store_true", help="no per-day lists or holdings")
    args = parser.parse_args(argv)
    modes = [m for m in (args.paths, args.grid, args.config, args.policy) if m]
    if len(modes) != 1:
        parser.error("give run directories, or one of --grid / --config / --policy")
    if args.source != "synthetic" and (args.paths or args.grid is not None):
        parser.error("a run records the source it ran on; --source is for configs")
    if args.first is not None and (args.first < 1 or args.seeds is not None):
        parser.error("--first takes a positive count and does not combine with --seeds")
    if args.grid is not None and args.seeds is not None:
        parser.error("a grid's cells have their own seeds; use --first")
    if args.arms is not None and args.grid is None:
        parser.error("--arms selects the arms of a --grid")
    if args.workers < 1:
        parser.error("--workers takes a positive count")
    out: Path = args.out if args.out is not None else default_root()

    def seeds_of(config: RunConfig) -> list[int] | None:
        if args.first is not None:
            return list(range(config.seed, config.seed + min(args.first, config.n_episodes)))
        return list(args.seeds) if args.seeds is not None else None

    results: list[RunActionMix] = []
    grid: GridActionMix | None = None
    try:
        if args.grid is not None:
            grid, cells = mix_grid(
                args.grid, arms=args.arms, first=args.first, workers=args.workers
            )
            results.extend(cells)
        elif args.paths:
            for path in args.paths:
                runs = find_runs(path) if path.is_dir() else []
                if not runs:
                    print(f"no run directories at {path}", file=sys.stderr)
                    return 1
                for run_dir in runs:
                    config = RunManifest.read(run_dir).config
                    results.append(mix_run(run_dir, seeds_of(config)))
        else:
            if args.config is not None:
                config = read_config(args.config)
            else:
                config = RunConfig(
                    policy=args.policy,
                    seed=args.seed,
                    n_episodes=args.episodes,
                    market=MarketConfig(iv_premium_vol_pts=args.premium),
                )
            source: SourceName = args.source
            provider = make_source(source, config.market, split="eval").provider()
            results.append(mix_config(config, seeds_of(config), provider=provider, source=source))
    except LearnerUnavailableError as exc:
        print(f"cannot load the policy: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, DataUnavailableError) as exc:
        print(f"cannot read the action mix: {exc}", file=sys.stderr)
        return 1

    for result in results:
        target = write_mix(result, out / output_name(result), episodes=not args.lean)
        if grid is None:
            print(render(result))
            print(f"    -> {target}", flush=True)
    if grid is not None:
        target = write_mix(grid, out / output_name(grid))
        print(render_grid(grid))
        print(f"    -> {target} (and {len(results)} cell files beside it)", flush=True)
    return 1 if any(r.agrees is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
