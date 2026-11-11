"""API payloads for the study: the design ``docs/PRE_REGISTRATION.md`` fixes, the real premium
on the sweep's axis (``docs/VRP_MAPPING.json``) and the M4 analysis (``docs/M4_ANALYSIS.json``).

Lane: Ilan Danial (machinery).

The design's power and floor are recomputed by ``optspread.power`` rather than copied; the
real premium is read off the M4 analysis' power curves with ``optspread.vrp.power_at``.
Moved from ``optspread.webviews``, which re-exports it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from optspread import power as pw
from optspread.api._base import Out, _from_attrs

# -- the study design (docs/PRE_REGISTRATION.md) ------------------------------------ #

PREREG_FILE = "PRE_REGISTRATION.md"
# A7: the pilot's PPO contrast components, and the fixed-delta seller's (the control).
A7_TAU2, A7_SIGMA2 = 51_838.0, 105_996_836.0
A7_SELLER_TAU2, A7_SELLER_SIGMA2 = 1_026.0, 828_609.0
# A6: the effect to detect, the seller's contrast at 1 vol point, in $ per episode. It is
# also the dashboard's conversion between dollars and vol points of planted premium.
A6_DELTA = 824.3
# A7: steps/s/core measured on the pilot machine (train: PPO; eval: environment steps).
A7_TRAIN_RATE, A7_EVAL_RATE = 804.0, 1_538.0


class MeasuredDesignOut(Out):
    """The design's power and floor at the sweep's own variance components (M4's S3).

    ``docs/M4_ANALYSIS.json`` ``design_check.arms.<arm>`` at its reference premium: the
    same S, N, alpha and effect as A7's plan, with the contrast's measured tau^2 and
    sigma^2 in place of the pilot's. This, not the plan, is the power the design had.
    """

    arm: str
    premium: float
    tau2: float
    sigma2: float
    power: float
    floor: float  # $ per episode
    floor_vol_pts: float
    file: str


class StudyOut(Out):
    """The confirmatory design as the pre-registration fixes it, with its power recomputed.

    ``power`` and ``floor`` are A7's plan (the pilot's PPO variances); ``measured`` is
    the same design read with the sweep's own variances, when the analysis is written.
    """

    test: str
    alpha: float
    event: str
    n_seeds: int
    n_episodes: int
    train_steps: int
    episode_days: int
    premiums: list[float]
    arms: list[str]
    grid_hash: str
    delta_per_vol_pt: float
    tau2: float
    sigma2: float
    seller_tau2: float
    seller_sigma2: float
    target_power: float
    power: float
    floor: float  # $ per episode, exact t at S and N
    floor_vol_pts: float
    train_rate: float
    eval_rate: float
    core_hours: float
    doc: str
    source: str
    measured: MeasuredDesignOut | None = None


def measured_design(docs: Path, arm: str = "ppo") -> MeasuredDesignOut | None:
    """The design check of ``docs/M4_ANALYSIS.json`` for ``arm`` at its reference premium,
    or None when the analysis is not there or does not carry one."""
    raw, _ = _read_json(docs / M4_FILE)
    if raw is None:
        return None
    try:
        check = raw["design_check"]
        ref = float(check["reference_premium"])
        rows = check["arms"]
        pick = arm if arm in rows else next(iter(rows))
        row = next(r for r in rows[pick] if float(r["premium"]) == ref)
        floor = float(row["floor"])
        return MeasuredDesignOut(
            arm=str(pick),
            premium=ref,
            tau2=float(row["tau2"]),
            sigma2=float(row["sigma2"]),
            power=float(row["power"]),
            floor=floor,
            floor_vol_pts=floor / A6_DELTA,
            file=M4_FILE,
        )
    except (KeyError, StopIteration, TypeError, ValueError):
        return None


def sweep_core_hours(
    n_seeds: int,
    n_episodes: int,
    train_steps: int,
    episode_days: int,
    premiums: int,
    arms: Sequence[str],
    train_rate: float = A7_TRAIN_RATE,
    eval_rate: float = A7_EVAL_RATE,
) -> float:
    """Core-hours for a premium x arm x S grid: evaluation for every arm, training for ppo."""
    per_block = sum(
        n_episodes * episode_days / eval_rate + (train_steps / train_rate if a == "ppo" else 0.0)
        for a in arms
    )
    return premiums * n_seeds * per_block / 3600.0


def study(docs: Path | None = None) -> StudyOut:
    """The design; with ``docs``, also its power as the analysis measured it."""
    from optspread.m4analysis import CONFIRMATORY_GRID, CONFIRMATORY_GRID_HASH, TARGET_POWER

    g = CONFIRMATORY_GRID
    s, n = g.n_seeds, g.base.n_episodes
    steps = g.train.total_timesteps if g.train is not None else 0
    floor = pw.detectable_effect_t(A7_TAU2, A7_SIGMA2, s, n, 0.05, TARGET_POWER)
    return StudyOut(
        test="one-sided Student t on the S seed means, S \u2212 1 df",  # a true minus
        alpha=0.05,
        event="each arm at premium p against itself at 0, paired by seed block",
        n_seeds=s,
        n_episodes=n,
        train_steps=steps,
        episode_days=g.base.env.episode_days,
        premiums=sorted(g.premiums),
        arms=list(g.arms),
        grid_hash=CONFIRMATORY_GRID_HASH,
        delta_per_vol_pt=A6_DELTA,
        tau2=A7_TAU2,
        sigma2=A7_SIGMA2,
        seller_tau2=A7_SELLER_TAU2,
        seller_sigma2=A7_SELLER_SIGMA2,
        target_power=TARGET_POWER,
        power=pw.power_t(A6_DELTA, A7_TAU2, A7_SIGMA2, s, n, 0.05),
        floor=floor,
        floor_vol_pts=floor / A6_DELTA,
        train_rate=A7_TRAIN_RATE,
        eval_rate=A7_EVAL_RATE,
        core_hours=sweep_core_hours(s, n, steps, g.base.env.episode_days, len(g.premiums), g.arms),
        doc=PREREG_FILE,
        source=(
            "docs/PRE_REGISTRATION.md D1, A1, A2, A6, A7; optspread.m4analysis.CONFIRMATORY_GRID"
            + ("; measured: docs/M4_ANALYSIS.json design_check" if docs is not None else "")
        ),
        measured=measured_design(docs) if docs is not None else None,
    )


# -- the real premium on the sweep's axis (docs/VRP_MAPPING.json) ------------------- #

VRP_FILE = "VRP_MAPPING.json"
M4_FILE = "M4_ANALYSIS.json"


class VrpOut(Out):
    available: bool
    file: str
    mapping: dict[str, Any] | None
    error: str = ""


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, ""
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"{path.name} does not read: {exc}"
    if not isinstance(raw, dict):
        return None, f"{path.name} is not a JSON object"
    return raw, ""


def vrp_mapping(docs: Path) -> VrpOut:
    raw, error = _read_json(docs / VRP_FILE)
    if raw is not None and "headline" not in raw:
        raw, error = None, f"{VRP_FILE} has no headline"
    return VrpOut(available=raw is not None, file=VRP_FILE, mapping=raw, error=error)


# -- the M4 analysis (docs/M4_ANALYSIS.json) --------------------------------------- #


class MappedPowerOut(Out):
    """One arm's power curve read at the real premium, at its interval's ends and over it.

    Read as the M4 gate reads it (``m4gate.read_curve``): from the curve's knots through
    ``vrp.power_at_noncentrality``, a curve that dips by sampling noise read as it is;
    a sidecar without knots is read linearly off its ``interpolated`` list
    (``vrp.power_at``).
    """

    arm: str
    premium: float
    low: float
    high: float
    power: float | None
    power_low: float | None
    power_high: float | None
    clipped: bool
    reason: str = ""  # why the curve was not read, when it was not
    power_min: float | None = None  # the lowest power over the interval
    power_max: float | None = None  # the highest
    band_min: float | None = None  # the curve's band over the interval, from its knots
    band_max: float | None = None


class M4Out(Out):
    available: bool
    file: str
    analysis: dict[str, Any] | None
    vrp: dict[str, Any] | None  # {estimate, interval, label, file}
    mapped: list[MappedPowerOut]
    waiting_for: str
    error: str = ""


M4_WAITING = "the analysis of the confirmatory sweep: `make m4` writes docs/M4_ANALYSIS.json"


def _mapped(
    arm: str,
    curve: Any,
    estimate: float,
    interval: tuple[float, float],
    n_seeds: int | None = None,
    alpha: float | None = None,
) -> MappedPowerOut:
    from optspread.m4gate import read_curve
    from optspread.vrp import ALPHA, power_at

    lo, hi = sorted(interval)
    band: tuple[float | None, float | None] = (None, None)
    try:
        points = list(curve.get("points") or ())
        if points and n_seeds is not None:
            m, band = read_curve(
                points, n_seeds, ALPHA if alpha is None else alpha, lo, hi, estimate
            )
        else:
            pts = [(float(p["premium"]), float(p["power"])) for p in curve["interpolated"]]
            m = power_at([x for x, _ in pts], [y for _, y in pts], estimate, (lo, hi))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return MappedPowerOut(
            arm=arm,
            premium=estimate,
            low=lo,
            high=hi,
            power=None,
            power_low=None,
            power_high=None,
            clipped=False,
            reason=str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
    return _from_attrs(MappedPowerOut, m, arm=arm, band_min=band[0], band_max=band[1])


def m4_analysis(docs: Path) -> M4Out:
    """The sidecar as ``scripts/m4_analysis.py`` wrote it, with the real premium read off it."""
    from optspread.m4analysis import SCHEMA_VERSION as M4_SCHEMA

    raw, error = _read_json(docs / M4_FILE)
    version = raw.get("schema_version") if raw is not None else None
    if raw is not None and version != M4_SCHEMA:
        raw, error = None, f"{M4_FILE} is schema v{version}, not v{M4_SCHEMA}"
    vrp = vrp_mapping(docs).mapping
    real: dict[str, Any] | None = None
    if vrp is not None:
        try:
            inputs = vrp["power_inputs"]
            real = {
                "estimate": float(inputs["estimate"]),
                "interval": [float(x) for x in inputs["interval"]],
                "label": str(vrp.get("headline", {}).get("label", "")),
                "file": VRP_FILE,
            }
        except (KeyError, TypeError, ValueError):
            real = None
    mapped: list[MappedPowerOut] = []
    if raw is not None and real is not None:
        lo, hi = real["interval"]
        try:
            n_seeds: int | None = int(raw["design"]["n_seeds"])
        except (KeyError, TypeError, ValueError):
            n_seeds = None
        settings = raw.get("settings") or {}
        alpha = settings.get("alpha") if isinstance(settings, dict) else None
        for arm in raw.get("arm_order", []):
            curve = raw.get("arms", {}).get(arm, {}).get("curve", {})
            mapped.append(
                _mapped(
                    str(arm),
                    curve,
                    real["estimate"],
                    (lo, hi),
                    n_seeds,
                    float(alpha) if isinstance(alpha, int | float) else None,
                )
            )
    return M4Out(
        available=raw is not None,
        file=M4_FILE,
        analysis=raw,
        vrp=real,
        mapped=mapped,
        waiting_for=M4_WAITING,
        error=error,
    )
