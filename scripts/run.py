"""Run one arm once and write it as an ordinary run directory: the single-run command.

Lane: Ilan Danial (machinery).

    python -m scripts.run --policy delta_seller --premium 1 --episodes 200 --seed 0
    python -m scripts.run --policy delta_seller --episodes 200 --source optionmetrics
    python -m scripts.run --policy ppo --timesteps 200000 --episodes 200 --source optionmetrics

One ``RunConfig`` (costs on unless ``--no-costs``), evaluated on path seeds ``--seed``
to ``--seed + --episodes - 1`` and written through ``write_run`` to
``<root>/<run id>``, with its source recorded in the manifest. A second call with the
same arguments reads the run back rather than recomputing it. ``--root`` defaults to
``$VEGALAB_DATA/runs/single`` when ``VEGALAB_DATA`` is set, else ``runs/single``, and
for the real source to the ``optionmetrics`` folder of that runs root
(``.../runs/optionmetrics/single``): the source is not in the run id, so a real-data run
and its synthetic twin would otherwise share a directory.

On the real source (``--source optionmetrics``, the panel under ``$VEGALAB_DATA/panel``)
the episodes are windows from the ``eval`` split (2019-01-01 to 2022-08-24), and a PPO
arm is trained on ``train`` (to 2018-12-31) first, so training and evaluation never
share a date; those are the dates pre-registration amendment A8 fixed, and nothing
after them is read. The planted premium is not applied there.
``python -m scripts.replay <run dir>`` reruns the run on the source and split its
manifest records.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

from optspread.config import ArmName, CostConfig, MarketConfig, RunConfig, TrainConfig
from optspread.experiment.cli import (
    add_source_arguments,
    chosen_source,
    describe,
    premium_note,
    runs_root,
)
from optspread.experiment.schedule import run_single
from optspread.train import Trainer, train_ppo

ARM_NAMES: tuple[str, ...] = get_args(ArmName)
SINGLE_DIR = "single"


def main(argv: Sequence[str] | None = None, trainer: Trainer = train_ppo) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--policy", choices=ARM_NAMES, default="delta_seller")
    parser.add_argument(
        "--premium", type=float, default=0.0, help="planted premium, vol points (generator only)"
    )
    parser.add_argument("--episodes", type=int, default=100, help="evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="first evaluation path seed")
    parser.add_argument(
        "--timesteps",
        type=int,
        help="PPO steps (ppo only; A4 fixes 1000000 for the pilot and sweep)",
    )
    parser.add_argument("--train-seed", type=int, default=0, help="PPO training seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-costs", action="store_true", help="switch spread and commission off")
    parser.add_argument(
        "--root",
        type=Path,
        help="where the run directory goes (default: $VEGALAB_DATA/runs/single when "
        "VEGALAB_DATA is set, else runs/single; .../runs/optionmetrics/single on the real "
        "source)",
    )
    add_source_arguments(parser)
    args = parser.parse_args(argv)

    policy = cast(ArmName, str(args.policy))
    if (policy == "ppo") != (args.timesteps is not None):
        parser.error(
            "--timesteps is given exactly when --policy is ppo (A4: 1000000); it has no default"
        )
    source = chosen_source(parser, args)
    train = None
    if args.timesteps is not None:
        train = TrainConfig(
            total_timesteps=int(args.timesteps), seed=int(args.train_seed), device=args.device
        )
    config = RunConfig(
        market=MarketConfig(iv_premium_vol_pts=float(args.premium)),
        cost=CostConfig(enabled=not args.no_costs),
        policy=policy,
        seed=int(args.seed),
        n_episodes=int(args.episodes),
        train=train,
    )
    root = runs_root(args.root, source, SINGLE_DIR)
    if float(args.premium) != 0.0 and (note := premium_note(source)):
        print(note, file=sys.stderr)
    print(f"{policy} on {describe(source)}", flush=True)
    try:
        run_dir, record, reused = run_single(config, root, trainer=trainer, source=source)
    except (FileExistsError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"{record.run_id}  {'reused' if reused else 'written'}  {run_dir}\n"
        f"mean P&L {record.mean_pnl:,.0f}  se {record.se_pnl:,.0f}  t {record.t_stat:.2f}  "
        f"cost {record.mean_cost:,.0f}  trades {record.mean_trades:.1f}  "
        f"({record.n_episodes} episodes)"
    )
    print(f"replay: python -m scripts.replay {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
