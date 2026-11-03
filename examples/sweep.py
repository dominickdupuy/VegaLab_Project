"""
VegaLab sweep example — market parameter × arm comparison.

Run:
    python examples/sweep.py

This script demonstrates the VegaLab research loop:
  1. Define a synthetic market with configurable parameters (alpha_bps, vol_of_vol, skew_slope)
  2. Run RL and scripted baseline arms on the same synthetic market
  3. Compare mean final P&L across arms and alpha_bps values

Researchers use this to verify that:
  - Learned arms (PPO) can discover edge when alpha_bps > 0
  - Scripted baselines behave predictably under different regimes
  - vol_of_vol and skew_slope change the difficulty of the learning problem

Key API notes:
  - execute_job(job_id, spec_json, runs_root) runs a job and returns the relative
    path to the run directory; it does NOT return a RunRecord directly.
  - read_run(run_dir) reads run.json from the run directory and returns a RunRecord
    with mean_pnl (mean P&L across all episodes), se_pnl, t_stat, etc.
  - Runs with the same config hash + seed are cached on disk and not recomputed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from optspread.artifacts import read_run
from optspread.jobs import JOBS_DIR, JobSpec, execute_job, ppo_available

# --- sweep grid ----------------------------------------------------------------- #

# Scripted arms are always available; PPO requires stable_baselines3 and optspread.train.
# a2c and dqn are also learned arms — add them if you want a fuller comparison.
_ppo_ok = ppo_available()
ARMS: list[str] = ["flat", "delta_seller"] + (["ppo"] if _ppo_ok else [])

ALPHA_VALUES: list[int] = [0, 25, 75]

# Shared config — small n_episodes for speed; raise to 200+ for stable results.
# tenor_days must be a sorted tuple of distinct positive integers.
SHARED: dict = dict(
    n_days=63,
    n_episodes=50,
    tenor_days=(21,),
    # expiry_idx=0 because we only have one tenor; the default EnvConfig.expiry_idx=1
    # would reference a non-existent second tenor and fail validation.
    expiry_idx=0,
    vol_of_vol=0.0,
    skew_slope=0.0,
    seed=42,
)

# total_timesteps for PPO — raise for better policy quality at the cost of runtime
PPO_TIMESTEPS = 10_000


# --- helpers -------------------------------------------------------------------- #


def mean_pnl(runs_root: Path, rel: str) -> float:
    """Read the mean episode P&L from a completed run directory.

    execute_job writes run.json to runs_root / rel; RunRecord.mean_pnl is the
    mean of (final_equity - initial_equity) across all evaluated episodes.
    """
    record = read_run(runs_root / rel)
    return record.mean_pnl


# --- main ----------------------------------------------------------------------- #


def main() -> None:
    if not _ppo_ok:
        print(
            "Note: PPO is not available (stable_baselines3 not installed or "
            "optspread.train missing). Running scripted arms only.\n"
        )

    # Use a temporary directory so sweep runs do not pollute the project tree.
    # Change to a persistent path (e.g. Path("runs/sweep")) if you want caching.
    with tempfile.TemporaryDirectory(prefix="vegalab_sweep_") as tmp:
        runs_root = Path(tmp)
        # execute_job writes progress files to runs_root/.jobs — create it first.
        (runs_root / JOBS_DIR).mkdir(parents=True, exist_ok=True)

        results: dict[tuple[str, int], float] = {}
        total = len(ARMS) * len(ALPHA_VALUES)
        done = 0

        for arm in ARMS:
            for alpha in ALPHA_VALUES:
                done += 1
                is_learned = arm in ("ppo", "a2c", "dqn")
                print(
                    f"[{done}/{total}] Running arm={arm!r} alpha_bps={alpha} ...",
                    flush=True,
                )
                spec = JobSpec(
                    policy=arm,
                    alpha_bps=float(alpha),
                    # total_timesteps is required for learned arms, must be None otherwise
                    total_timesteps=PPO_TIMESTEPS if is_learned else None,
                    **SHARED,
                )
                # execute_job is the programmatic entrypoint used by the dashboard worker.
                # It returns the path to the run directory, relative to runs_root.
                rel = execute_job(
                    f"sweep-{arm}-alpha{alpha}",
                    spec.model_dump_json(),
                    str(runs_root),
                )
                results[(arm, alpha)] = mean_pnl(runs_root, rel)

    # --- print table ------------------------------------------------------------ #
    col_w = 15
    n_days = SHARED["n_days"]
    n_ep = SHARED["n_episodes"]
    tenor = SHARED["tenor_days"]
    vov = SHARED["vol_of_vol"]

    print()
    print(
        f"Sweep: arm × alpha_bps  "
        f"(n_days={n_days}, n_episodes={n_ep}, tenor={tenor}, vol_of_vol={vov})"
    )
    sep_len = 14 + col_w * len(ALPHA_VALUES)
    print("=" * sep_len)
    header = f"{'arm':<14}" + "".join(f"{'alpha_bps='+str(a):<{col_w}}" for a in ALPHA_VALUES)
    print(header)
    print("-" * sep_len)
    for arm in ARMS:
        row = f"{arm:<14}" + "".join(
            f"{results[(arm, a)]:>+{col_w}.3f}" for a in ALPHA_VALUES
        )
        print(row)

    print()
    print("Notes:")
    print("  alpha_bps=0  : zero-drift synthetic market — only theta-selling has structural edge")
    print("  alpha_bps=25 : ~SPX-like long bias — long-biased arms and learned policies gain")
    print("  alpha_bps=75 : strong directional signal — clear edge for any long arm")
    if _ppo_ok:
        print("  PPO is a learned policy trained fresh each run.")
    else:
        print("  Install stable_baselines3 and optspread.train to include the PPO arm.")
    print(f"  Increase n_episodes (currently {n_ep}) and PPO total_timesteps ({PPO_TIMESTEPS})")
    print("  for more stable, publication-quality results.")
    print("  Try vol_of_vol=0.3 or skew_slope=+0.3 (equity put skew) to stress-test robustness.")
    print("  P&L values are in raw dollar units (not normalised by initial equity).")


if __name__ == "__main__":
    main()
