"""API payloads for the leak detector: the demo, its floor, and its per-field controls.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi`` (the demo) and
``optspread.webviews`` (the floor, the fields and their coverage), which re-export it.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import ConfigDict, Field

from optspread.api._base import Out, _f, _from_attrs
from optspread.dashboard.leakdemo import LeakDemo
from optspread.leaks import LeakReport

# -- the leak demo ---------------------------------------------------------------- #


class FeatureOut(Out):
    name: str
    tested: bool
    t_stat: float | None
    p_value: float | None
    reason: str


class LeakSideOut(Out):
    flagged: bool
    worst_feature: str
    worst_t: float | None
    p_value: float | None
    threshold: float
    n_pairs: int
    n_tested: int
    past_unchanged: bool  # the perturbation property: resampling the future leaves t alone
    features: list[FeatureOut]
    t_critical: float | None = None  # the |t| a field must exceed to be flagged


class LeakDemoOut(Out):
    eps: float
    seeds: list[int]
    clean: LeakSideOut
    poisoned: LeakSideOut
    caught: bool
    seconds: float
    criterion: str
    source: str


def leak_t_critical(threshold: float, n_pairs: int) -> float | None:
    """The two-sided |t| at which a field's p falls below the Bonferroni threshold."""
    from scipy import stats

    if n_pairs <= 2 or not 0.0 < threshold < 1.0:
        return None
    return _f(float(stats.t.isf(threshold / 2.0, df=n_pairs - 2)))


def _side(report: LeakReport, past_unchanged: bool) -> LeakSideOut:
    return LeakSideOut(
        flagged=report.flagged,
        worst_feature=report.worst_feature,
        worst_t=_f(report.worst_t),
        p_value=_f(report.p_value),
        threshold=report.threshold,
        n_pairs=report.n_pairs,
        n_tested=report.n_features_tested,
        past_unchanged=past_unchanged,
        t_critical=leak_t_critical(report.threshold, report.n_pairs),
        features=[
            FeatureOut(
                name=f.name,
                tested=f.tested,
                t_stat=_f(f.t_stat) if f.tested else None,
                p_value=_f(f.p_value) if f.tested else None,
                reason=f.reason,
            )
            for f in report.features
        ],
    )


def leak_demo(demo: LeakDemo) -> LeakDemoOut:
    return _from_attrs(
        LeakDemoOut,
        demo,
        clean=_side(demo.clean, demo.clean_past_unchanged),
        poisoned=_side(demo.poisoned, demo.poisoned_past_unchanged),
        source="optspread.leaks: detect_leak and perturbation_check, run live on default_env",
    )


class LeakDemoParams(Out):
    eps: float = Field(default=0.5, gt=0.0, le=2.0)
    n_seeds: int = Field(default=10, ge=3, le=40)


# -- the leak detector's floor and its per-field controls --------------------------- #

FLOOR_EPS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025)


class LeakFloorParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    n_seeds: int = Field(default=10, ge=3, le=40)
    eps_grid: list[float] = Field(
        default_factory=lambda: list(FLOOR_EPS), min_length=2, max_length=10
    )


class LeakFloorOut(Out):
    eps: list[float]  # descending, as leaks.leak_floor plants them
    flagged: list[bool]
    floor: float | None  # smallest eps still caught
    seeds: list[int]
    seconds: float
    source: str


def leak_floor_out(p: LeakFloorParams) -> LeakFloorOut:
    from optspread.leaks import default_env, leak_floor

    if any(not 0.0 < e <= 2.0 for e in p.eps_grid):
        raise ValueError("every eps must be in (0, 2]")
    seeds = list(range(p.n_seeds))
    start = time.perf_counter()
    report = leak_floor(default_env, p.eps_grid, seeds)
    return LeakFloorOut(
        eps=list(report.eps_grid),
        flagged=list(report.flagged),
        floor=report.floor,
        seeds=seeds,
        seconds=time.perf_counter() - start,
        source="optspread.leaks.leak_floor on default_env: spot look-ahead at each eps",
    )


class LeakFieldParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    field: str = "ret_1d"
    eps: float = Field(default=0.25, gt=0.0, le=2.0)
    n_seeds: int = Field(default=10, ge=3, le=40)


class LeakFieldOut(Out):
    field: str
    eps: float
    scale: float  # the field's own standard deviation under the probe
    seeds: list[int]
    flagged: bool
    named_correctly: bool  # the worst feature is the field that was poisoned
    worst_feature: str
    p_value: float | None
    threshold: float
    features: list[FeatureOut]
    seconds: float
    source: str
    t_critical: float | None = None  # the |t| a field must exceed to be flagged


def leak_fields() -> list[str]:
    from optspread.env import OBS_NAMES

    return list(OBS_NAMES)


def _poisoned_env(field_name: str, eps: float, scale: float) -> Any:
    from optspread.leaks import default_env, poison_field

    return poison_field(default_env(), field_name, eps, scale)


def leak_field_out(p: LeakFieldParams) -> LeakFieldOut:
    """Look-ahead planted in one field, eps of its own sd: does the detector fire on it?"""
    from functools import partial

    from optspread.leaks import default_env, detect_leak, field_scales

    if p.field not in leak_fields():
        raise ValueError(f"{p.field!r} is not an observation field")
    seeds = list(range(p.n_seeds))
    start = time.perf_counter()
    scale = field_scales(default_env, seeds)[p.field]
    report = detect_leak(partial(_poisoned_env, p.field, p.eps, scale), seeds)
    side = _side(report, past_unchanged=True)
    return LeakFieldOut(
        field=p.field,
        eps=p.eps,
        scale=scale,
        seeds=seeds,
        flagged=report.flagged,
        named_correctly=report.flagged and report.worst_feature == p.field,
        worst_feature=report.worst_feature,
        p_value=_f(report.p_value),
        threshold=report.threshold,
        features=side.features,
        seconds=time.perf_counter() - start,
        source="optspread.leaks.poison_field and detect_leak on default_env",
        t_critical=side.t_critical,
    )


class FieldControlOut(Out):
    field: str
    flagged: bool
    named_correctly: bool
    p_value: float | None


class LeakCoverageOut(Out):
    eps: float
    seeds: list[int]
    clean_flagged: bool
    passed: bool
    uncovered: list[str]
    misattributed: list[str]
    controls: list[FieldControlOut]
    seconds: float
    source: str


def leak_coverage_out(eps: float, n_seeds: int) -> LeakCoverageOut:
    from optspread.leaks import default_env, field_coverage

    seeds = list(range(n_seeds))
    start = time.perf_counter()
    rep = field_coverage(default_env, seeds, eps)
    return LeakCoverageOut(
        eps=eps,
        seeds=seeds,
        clean_flagged=rep.clean.flagged,
        passed=rep.passed,
        uncovered=list(rep.uncovered),
        misattributed=list(rep.misattributed),
        controls=[_from_attrs(FieldControlOut, c, p_value=_f(c.p_value)) for c in rep.controls],
        seconds=time.perf_counter() - start,
        source="optspread.leaks.field_coverage on default_env",
    )
