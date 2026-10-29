"""Attribute runs' P&L to delta, gamma, vega, theta, carry, cost and a residual.

Lane: Dominick Dupuy (statistical spine). Project document 10.9.

    python -m scripts.attribution runs/pilot/<run id>            one run
    python -m scripts.attribution runs/pilot --seeds 0 1 2        a folder of runs
    python -m scripts.attribution --config cfg.json               a RunConfig (or a manifest)
    python -m scripts.attribution --policy delta_seller --premium 2 --episodes 10
    python -m scripts.attribution --policy always_on --episodes 5 --source optionmetrics

A run directory is re-simulated from its manifest, checked against its ``run.json``
episode by episode, and never written into; a ppo run needs its ``policy.zip`` and
the ``train`` extra. A config (or the ``--policy`` defaults) runs on the synthetic
world unless ``--source optionmetrics`` puts it on the real SPX surface. Each result
is written as ``<out>/<run id>.json`` (``.optionmetrics.json`` off the synthetic
world), ``--out`` defaulting to ``optspread.attribution.default_root()``; the
schema is the module docstring of ``optspread.attribution``. Exit status 1 when a run
does not agree with its run.json (within ``PNL_RTOL``; a run made on another platform
agrees without reproducing bit for bit, and says so) or nothing was attributed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from optspread.attribution import (
    COMPONENTS,
    RunAttribution,
    Source,
    attribute_config,
    attribute_run,
    default_root,
    output_name,
    write_attribution,
)
from optspread.config import MarketConfig, PolicyName, RunConfig
from optspread.data import DataUnavailableError, OptionMetricsSource
from optspread.manifest import MANIFEST_FILENAME, RunManifest
from optspread.replay import find_runs


def render(result: RunAttribution) -> str:
    t = result.totals
    parts = "  ".join(f"{n} {getattr(t, n):,.0f}" for n in COMPONENTS[:-1])
    if result.reproduces is None:
        verdict = "no run to compare"
    elif result.reproduces:
        verdict = "reproduces"
    elif result.agrees:
        verdict = "agrees to within PNL_RTOL, not bit for bit"
    else:
        verdict = "DOES NOT AGREE"
    return (
        f"{result.run_id}  {result.policy}  premium {result.premium_vol_pts:g}  "
        f"{result.source}  {len(result.episodes)} episodes  "
        f"({verdict})\n"
        f"    pnl {t.pnl:,.0f} = {parts}  residual {t.residual:,.2f}\n"
        f"    |residual| {result.abs_residual:,.0f} of gross {result.gross:,.0f} "
        f"= {result.residual_share:.2%}; largest step {result.max_abs_step_residual:,.2f}"
    )


def read_config(path: Path) -> RunConfig:
    """A ``RunConfig`` JSON file, or a run's ``manifest.json`` (its config is taken)."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and path.name == MANIFEST_FILENAME:
        raw = raw["config"]
    return RunConfig.model_validate(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("paths", type=Path, nargs="*", help="run directories or their folder")
    parser.add_argument("--config", type=Path, help="a RunConfig JSON file or a manifest.json")
    parser.add_argument(
        "--policy", choices=get_args(PolicyName), help="a default RunConfig for this scripted arm"
    )
    parser.add_argument("--premium", type=float, default=0.0, help="planted vol points")
    parser.add_argument("--episodes", type=int, default=10, help="episodes for --policy")
    parser.add_argument("--seed", type=int, default=0, help="first seed for --policy")
    parser.add_argument("--seeds", type=int, nargs="+", help="attribute only these seeds")
    parser.add_argument("--first", type=int, help="attribute only the first N episodes")
    parser.add_argument("--source", choices=("synthetic", "optionmetrics"), default="synthetic")
    parser.add_argument("--out", type=Path, help="output folder (default: default_root())")
    parser.add_argument("--no-steps", action="store_true", help="write totals, not steps")
    args = parser.parse_args(argv)
    configs = [p for p in (args.config, args.policy) if p is not None]
    if bool(args.paths) == bool(configs) or len(configs) > 1:
        parser.error("give run directories, or one of --config / --policy")
    if args.paths and args.source != "synthetic":
        parser.error("a run directory was produced on the synthetic world; --source is for configs")
    if args.first is not None and (args.first < 1 or args.seeds is not None):
        parser.error("--first takes a positive count and does not combine with --seeds")
    out: Path = args.out if args.out is not None else default_root()

    def seeds_of(config: RunConfig) -> list[int] | None:
        if args.first is not None:
            return list(range(config.seed, config.seed + min(args.first, config.n_episodes)))
        return list(args.seeds) if args.seeds is not None else None

    results: list[RunAttribution] = []
    try:
        if args.paths:
            for path in args.paths:
                runs = find_runs(path) if path.is_dir() else []
                if not runs:
                    print(f"no run directories at {path}", file=sys.stderr)
                    return 1
                for run_dir in runs:
                    config = RunManifest.read(run_dir).config
                    results.append(attribute_run(run_dir, seeds_of(config)))
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
            source: Source = args.source
            provider = (
                OptionMetricsSource(config.market, split="eval").provider()
                if source == "optionmetrics"
                else None
            )
            results.append(
                attribute_config(config, seeds_of(config), provider=provider, source=source)
            )
    except (OSError, ValueError, DataUnavailableError) as exc:
        print(f"cannot attribute: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:  # a ppo run's policy.zip needs stable-baselines3
        print(f"cannot load the policy: {exc} (install the train extra)", file=sys.stderr)
        return 1

    for result in results:
        target = write_attribution(result, out / output_name(result), steps=not args.no_steps)
        print(render(result))
        print(f"    -> {target}", flush=True)
    return 1 if any(r.reproduces is False and not r.agrees for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
