"""API payloads for launching jobs: the knobs a job takes, their options, and a job's status.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi``, which re-exports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from optspread.api._base import Out, _from_attrs
from optspread.dashboard.jobs.queue import JobState, JobStatus
from optspread.dashboard.jobs.spec import (
    LAUNCH_DIR,
    PRESETS,
    JobSpec,
    available_arms,
    available_sources,
)
from optspread.realdata import SPLITS
from optspread.runs.rundir import SOURCES, source_root

# -- jobs ------------------------------------------------------------------------- #

# The input a knob takes, one per branch of _knob.
KnobKind = Literal["number", "integer", "boolean", "choice", "tenors"]
# The launch form's sections: the groups in _GROUPS, and "Run" for a knob it does not name.
KnobGroup = Literal["Arm", "Market", "Costs", "Structure", "Account", "Run", "Training", "Data"]


class KnobOut(Out):
    name: str
    kind: KnobKind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    choices: list[str] = Field(default_factory=list)
    group: KnobGroup
    help: str = ""
    label: str = ""  # what the form calls it; ``name`` is the JobSpec field
    synthetic_only: bool = False  # read by the generator only; a real chain ignores it


# What the form calls each JobSpec field.
KNOB_LABELS: dict[str, str] = {
    "policy": "Policy",
    "iv_premium_vol_pts": "Planted premium (vol pts)",
    "sigma": "Path volatility",
    "r": "Risk-free rate",
    "n_days": "Days per episode",
    "tenor_days": "Tenors (trading days)",
    "n_strikes_each_side": "Strikes each side",
    "strike_step": "Strike step (points)",
    "costs_enabled": "Costs on",
    "min_half_spread": "Minimum half-spread",
    "half_spread_frac": "Half-spread, share of mid",
    "commission_per_contract": "Commission per contract ($)",
    "short_delta": "Short delta",
    "wing_delta": "Wing delta",
    "expiry_idx": "Opening tenor (0 = nearest)",
    "roll_dte": "Roll at days to expiry",
    "initial_cash": "Initial cash ($)",
    "seed": "First seed",
    "n_episodes": "Episodes",
    "total_timesteps": "Training steps",
    "source": "Data source",
}
# The generator's knobs a real chain does not read: its surface, path and carry are the
# day's own (``realdata.provider.HistoricalChainProvider._chain``); strikes, tenors and
# the episode length still apply.
SYNTHETIC_ONLY = frozenset({"iv_premium_vol_pts", "sigma", "r"})


class PresetOut(Out):
    name: str
    label: str
    description: str
    spec: dict[str, Any]


class JobOptionsOut(Out):
    arms: list[str]
    ppo_available: bool
    sources: list[str] = Field(default_factory=lambda: ["synthetic"])  # whose data is here
    # where a job on each source lands, relative to the runs root
    landing: dict[str, str] = Field(default_factory=dict)
    # A8: the dates a real-data job trains (PPO) and is evaluated on, ISO [first, last]
    splits: dict[str, list[str]] = Field(default_factory=dict)
    knobs: list[KnobOut]
    presets: list[PresetOut]
    defaults: dict[str, Any]
    ppo_preset: PresetOut | None = None  # listed whether or not the learner is installed


_GROUPS: dict[str, tuple[KnobGroup, str]] = {
    "policy": (
        "Arm",
        "Which policy runs: the scripted arms, or PPO when the learner is installed.",
    ),
    "iv_premium_vol_pts": ("Market", "Planted premium in vol points. Zero is the null world."),
    "sigma": ("Market", "Annualised vol driving the path, decimal."),
    "r": ("Market", "Risk-free rate, decimal."),
    "n_days": ("Market", "Path and episode length in trading days."),
    "tenor_days": ("Market", "Tenors listed at every step, trading days."),
    "n_strikes_each_side": ("Market", "Strikes listed either side of the money."),
    "strike_step": ("Market", "Strike spacing in index points."),
    "costs_enabled": ("Costs", "Quoted half-spread plus commission; off for the fair-world check."),
    "min_half_spread": ("Costs", "Floor on the half-spread, index points per share."),
    "half_spread_frac": ("Costs", "Half-spread as a fraction of mid."),
    "commission_per_contract": ("Costs", "Dollars per contract."),
    "short_delta": ("Structure", "Delta of the short strikes."),
    "wing_delta": ("Structure", "Delta of the long wings (further out)."),
    "expiry_idx": ("Structure", "Which listed tenor new structures open at (0 = nearest)."),
    "roll_dte": ("Structure", "The seller closes at or below this many days to expiry."),
    "initial_cash": ("Account", "Starting cash, dollars."),
    "seed": ("Run", "First episode seed; episodes use seed, seed+1, ..."),
    "n_episodes": ("Run", "Evaluation episodes."),
    "total_timesteps": (
        "Training",
        "PPO training steps; the pilot and the sweep train 1,000,000.",
    ),
    "source": (
        "Data",
        "The generator, or the real SPX panel (evaluated on 2019-01-01..2022-08-24, PPO "
        "trained on the dates before); not in the run id, recorded in the manifest.",
    ),
}
_UNGROUPED: tuple[KnobGroup, str] = ("Run", "")


def _knob(name: str, info: Any, arms: Sequence[str]) -> KnobOut:
    group, text = _GROUPS.get(name, _UNGROUPED)
    default = info.default
    annotation = str(info.annotation)
    bounds: dict[str, float | None] = {
        "minimum": None,
        "maximum": None,
        "exclusive_minimum": None,
        "exclusive_maximum": None,
    }
    for meta in info.metadata:
        for attr, key in (
            ("ge", "minimum"),
            ("le", "maximum"),
            ("gt", "exclusive_minimum"),
            ("lt", "exclusive_maximum"),
        ):
            value = getattr(meta, attr, None)
            if value is not None:
                bounds[key] = float(value)
    kind: KnobKind
    if name == "policy":
        kind, choices = "choice", list(arms)
    elif name == "source":
        kind, choices = "choice", list(SOURCES)
    elif name == "tenor_days":
        kind, choices, default = "tenors", [], list(default)
    elif "bool" in annotation:
        kind, choices = "boolean", []
    elif "int" in annotation:
        kind, choices = "integer", []
    else:
        kind, choices = "number", []
    return KnobOut(
        name=name,
        kind=kind,
        default=default,
        choices=choices,
        group=group,
        help=text,
        label=KNOB_LABELS.get(name, name.replace("_", " ")),
        synthetic_only=name in SYNTHETIC_ONLY,
        **bounds,
    )


def job_options() -> JobOptionsOut:
    arms = available_arms()
    knobs = [_knob(name, info, arms) for name, info in JobSpec.model_fields.items()]
    if "ppo" not in arms:
        knobs = [k for k in knobs if k.name != "total_timesteps"]
    return JobOptionsOut(
        arms=list(arms),
        ppo_available="ppo" in arms,
        sources=list(available_sources()),
        landing={s: (source_root(Path(), s) / LAUNCH_DIR).as_posix() for s in SOURCES},
        splits={k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        knobs=knobs,
        presets=[_from_attrs(PresetOut, p, spec=p.spec.model_dump(mode="json")) for p in PRESETS],
        defaults=JobSpec().model_dump(mode="json"),
    )


class JobOut(Out):
    job_id: str
    state: JobState
    phase: str
    done: int
    total: int
    fraction: float
    run_id: str
    run_key: str
    created_at: str
    policy: str
    error: str
    log_tail: list[str]
    source: str = "synthetic"


def job_out(status: JobStatus) -> JobOut:
    return JobOut.model_validate(status, from_attributes=True)
