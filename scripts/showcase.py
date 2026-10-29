"""The showcase: runs that exercise every view of the dashboard, and the catalog that names them.

Lane: Hemdutt Rao (market layer and data adapter).

    python -m scripts.showcase [--root runs/showcase] [--dry-run] [--skip-ppo]

Each run answers one question a trader or a researcher would ask, and the runs of a
question share their paths (seed 20,000, 200 episodes unless noted), so any two of them
compare as a pair: the difference between two settings is far more precise than either
run's mean. The questions:

- strikes: short delta 0.10, 0.16, 0.25, 0.35, at premiums 0, 1 and 2
- wings: wing delta 0.02, 0.05, 0.10
- tenor: the 21-day against the 42-day expiry, at premiums 0 and 1
- roll: at expiry, 5 and 10 days before
- costs: off, default, twice the spreads, twice the commission, at premiums 0 and 1
- account: $50k, $100k and $1M of cash
- regime: a 10%, 20% and 30% vol world
- real: the main settings and the scripted arms on real SPX (A8's evaluation dates)
- arms: all five arms at premiums 0, 0.5, 1, 2 and 4
- sample: 50, 200 and 500 episodes
- training: PPO trained 100,000, 200,000 and 500,000 steps

Runs go through the dashboard's own job path (``optspread.jobs.execute_job``), so each
records its equity paths, and a PPO run its policy. ``<root>/catalog.json`` lists every
run's key, group, question, label and settings; the Runs page and the workflows read it.
The real-data runs need the panel under ``$VEGALAB_DATA/panel``. A second call reuses
every run already on disk. The confirmatory sweep is not part of it: copy a sweep root
into ``--root`` to show it beside the showcase.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from optspread.jobs import JobSpec, execute_job

SEED = 20_000
EPISODES = 200
PPO_STEPS = 200_000


def plan() -> list[tuple[str, str, str, dict[str, Any]]]:
    """Every showcase run as (group, question, label, JobSpec settings), PPO last."""
    out: list[tuple[str, str, str, dict[str, Any]]] = []

    def add(group: str, question: str, label: str, **kw: Any) -> None:
        out.append((group, question, label, {"n_episodes": EPISODES, "seed": SEED, **kw}))

    q = "Which short-strike delta captures the most edge, and at what tail risk?"
    for d in (0.10, 0.16, 0.25, 0.35):
        for p in (0.0, 1.0, 2.0):
            add(
                "strikes",
                q,
                f"seller short delta {d:.2f}, premium {p:g}",
                policy="delta_seller",
                short_delta=d,
                iv_premium_vol_pts=p,
            )
    q = "How does wing width trade return against max loss?"
    for w in (0.02, 0.05, 0.10):
        add(
            "wings",
            q,
            f"seller wing delta {w:.2f}",
            policy="delta_seller",
            wing_delta=w,
            iv_premium_vol_pts=1.0,
        )
    q = "Short-dated or longer-dated condors?"
    for e, name in ((0, "21-day"), (1, "42-day")):
        for p in (0.0, 1.0):
            add(
                "tenor",
                q,
                f"seller {name} expiry, premium {p:g}",
                policy="delta_seller",
                expiry_idx=e,
                iv_premium_vol_pts=p,
            )
    q = "When should the seller roll?"
    for r in (0, 5, 10):
        add(
            "roll",
            q,
            f"seller roll {r} days before expiry",
            policy="delta_seller",
            roll_dte=r,
            iv_premium_vol_pts=1.0,
        )
    q = "How fast do trading costs eat the edge?"
    for p in (0.0, 1.0):
        add(
            "costs",
            q,
            f"seller costs off, premium {p:g}",
            policy="delta_seller",
            costs_enabled=False,
            iv_premium_vol_pts=p,
        )
        add(
            "costs",
            q,
            f"seller default costs, premium {p:g}",
            policy="delta_seller",
            iv_premium_vol_pts=p,
        )
        add(
            "costs",
            q,
            f"seller 2x spreads, premium {p:g}",
            policy="delta_seller",
            half_spread_frac=0.015,
            min_half_spread=0.30,
            iv_premium_vol_pts=p,
        )
        add(
            "costs",
            q,
            f"seller 2x commission, premium {p:g}",
            policy="delta_seller",
            commission_per_contract=1.30,
            iv_premium_vol_pts=p,
        )
    q = "When does margin start refusing trades?"
    for c in (50_000.0, 100_000.0, 1_000_000.0):
        add(
            "account",
            q,
            f"seller with ${c:,.0f} cash",
            policy="delta_seller",
            initial_cash=c,
            iv_premium_vol_pts=1.0,
        )
    q = "How does the volatility regime change the result?"
    for sg in (0.10, 0.20, 0.30):
        add(
            "regime",
            q,
            f"seller in a {sg:.0%} vol world, premium 1",
            policy="delta_seller",
            sigma=sg,
            iv_premium_vol_pts=1.0,
        )
    q = "How do these settings do on real SPX history (2019 to 2022-08)?"
    for d in (0.10, 0.16, 0.25):
        add(
            "real",
            q,
            f"seller on real data, short delta {d:.2f}",
            policy="delta_seller",
            short_delta=d,
            source="optionmetrics",
        )
    for w in (0.02, 0.10):
        add(
            "real",
            q,
            f"seller on real data, wing delta {w:.2f}",
            policy="delta_seller",
            wing_delta=w,
            source="optionmetrics",
        )
    add(
        "real",
        q,
        "seller on real data, 21-day expiry",
        policy="delta_seller",
        expiry_idx=0,
        source="optionmetrics",
    )
    for arm in ("flat", "always_on", "random"):
        add("real", q, f"{arm} on real data", policy=arm, source="optionmetrics")
    q = "Which strategies detect the premium, and which stay flat when there is none?"
    for arm in ("flat", "always_on", "random", "delta_seller"):
        for p in (0.0, 0.5, 1.0, 2.0, 4.0):
            add("arms", q, f"{arm}, premium {p:g}", policy=arm, iv_premium_vol_pts=p)
    q = "How many episodes does a study need?"
    for n in (50, 200, 500):
        add(
            "sample",
            q,
            f"seller premium 0.5 with {n} episodes",
            policy="delta_seller",
            iv_premium_vol_pts=0.5,
            n_episodes=n,
        )
    q = "Which strategies detect the premium, and which stay flat when there is none?"
    for p in (0.0, 0.5, 1.0, 2.0, 4.0):
        add(
            "arms",
            q,
            f"ppo, premium {p:g}",
            policy="ppo",
            iv_premium_vol_pts=p,
            total_timesteps=PPO_STEPS,
        )
    q = "How much training does PPO need?"
    for ts in (100_000, 200_000, 500_000):
        add(
            "training",
            q,
            f"ppo premium 2 trained {ts:,} steps",
            policy="ppo",
            iv_premium_vol_pts=2.0,
            total_timesteps=ts,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("runs/showcase"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--skip-ppo", action="store_true", help="the scripted runs only")
    args = ap.parse_args(argv)
    runs = [r for r in plan() if not (args.skip_ppo and r[3].get("policy") == "ppo")]
    for group, _, label, kw in runs:
        JobSpec(**kw)  # refuse a bad setting before anything runs
        if args.dry_run:
            print(f"[{group}] {label}")
    if args.dry_run:
        print(f"{len(runs)} runs")
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    for i, (group, question, label, kw) in enumerate(runs):
        start = time.time()
        key = execute_job(f"showcase-{i}", JobSpec(**kw).model_dump_json(), str(args.root))
        catalog.append(
            {"key": key, "group": group, "question": question, "label": label, "settings": kw}
        )
        (args.root / "catalog.json").write_text(
            json.dumps({"seed": SEED, "runs": catalog}, indent=1), encoding="utf-8"
        )
        print(
            f"{i + 1}/{len(runs)} [{group}] {label} -> {key} ({time.time() - start:.0f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
