"""The M4 gate items: one small evidence function per deliverable.

Lane: Dominick Dupuy (statistical spine).

The M4 row of the schedule (project document section 12) is the power curve
analysis; bootstrap validation; the tau^2 homogeneity test; the VRP mapping with
propagated uncertainty; the dashboard's attribution and power curve views; and the
collaboration report. Section 10.9 adds the attribution residual: a near-zero
residual "is reported as such rather than only plotted", and section 13 claims it,
so it is an item here too, read from the attribution of the sweep's own cells.

Each function returns a ``GateItem`` and never raises for a missing deliverable.
Nothing here recomputes the analysis: the first three items read the sidecar
``scripts/m4_analysis.py`` writes (``docs/M4_ANALYSIS.json``, its schema in
``optspread.m4analysis``) and hold it to the rules it was written under, so the
report says what the committed analysis says, and fails where it is not the
confirmatory one. The analysis items share one check (``analysis_problem``): the
bootstrap, tau^2 and VRP items are not read off an exploratory or incomplete grid.

The VRP item is where the uncertainty is propagated: the mapped premium's point and
both ends of its interval (``docs/VRP_MAPPING.json``) are read off each arm's power
curve through ``vrp.power_at_noncentrality``, on the curve's knots and through the
analysis's own interpolation (noncentrality linear in premium from the anchor at
zero, exact noncentral-t power), and the curve's band is read the same way from the
knots' band ends. A curve estimated at S = 5 dips by sampling noise, so the item reads
it as it is and reports the lowest and highest power over the interval, not only the
powers at its ends. The sidecar's dense ``interpolated`` curve, which the dashboard
draws, must be the knots' own reading at every one of its points; a sidecar where it is
not, or knots that are not a curve, fail the item, with the dense curve's linear
reading beside the failure so the number is in the report either way.

The residual item reads ``agrees`` (the attribution's P&L and cost within
``attribution.PNL_RTOL`` of run.json, the same trades and steps) and reports
``reproduces`` (bit for bit) beside it: a cell attributed on another platform agrees
without reproducing. It cannot check vega on the synthetic sweep, where the surface is
flat and the vega term is zero, and says so; it reports any attribution under the root
whose vega term is not zero (real chains, a moving surface) and its residual share.

The dashboard is another lane's (Ilan Danial's), so its item imports the
server by the names below and matches routes loosely, by name: an API path holding
``attribution``, and one with an ``m4`` segment or a ``power-curve``. The evidence
states what matched. The project document names no file for the collaboration
report; ``COLLABORATION_FILES`` are the names looked for.

``scripts/m4_gate_report.py`` assembles these into ``docs/M4_GATE_REPORT.md``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from optspread import m4analysis as m4
from optspread import vrp
from optspread.attribution import RunAttribution, read_attribution
from optspread.config import config_hash
from optspread.experiment.grid import GridConfig, grid_cells, grid_filename, read_grid
from optspread.gatereport import GateItem
from optspread.m2gate import DOMINICK, HEMDUTT, ILAN, _load, _not_cleared

ANALYSIS_FILE = "M4_ANALYSIS.json"
VRP_FILE = "VRP_MAPPING.json"
DATA_MANIFEST_FILE = "DATA_MANIFEST.json"
PANEL_DAILY = "panel/spx_daily.parquet"  # the table the mapping reads, as the manifest keys it
COLLABORATION_FILES = ("COLLABORATION.md", "COLLABORATION_REPORT.md")
DENSE_ATOL = 1e-9  # the sidecar's interpolated curve against its knots' exact reading
TEAM = "All members"

SERVER_MODULE = "optspread.server"
APP_FACTORY = "create_app"  # (Settings) -> FastAPI; building it starts nothing
SETTINGS_CLASS = "Settings"  # (runs, docs, data, web, readme)
ATTRIBUTION_ROUTE = re.compile(r"attribution", re.IGNORECASE)
POWER_CURVE_ROUTE = re.compile(r"(^|/)m4(/|$)|power[-_]?curve", re.IGNORECASE)
SITE_SUFFIXES = (".html", ".js", ".txt")  # what a static export's pages and chunks are

# The residual share (the summed absolute residual over gross mark-to-market) a cell's attribution
# must stay under: tests/resim/test_attribution.py's RESIDUAL_SHARE_TOL. The residual is the
# third-order remainder of the expansion, measured at 1.5% to 2.6% for the scripted
# arms; a sign or unit error in delta, gamma or theta moves it far past 5%. One in vega
# does not on the synthetic sweep, whose surface is flat (optspread.attribution).
RESIDUAL_SHARE_TOL = 0.05

FLOOR_STATUSES = ("reached", "below_grid", "not_reached")
GAP_KINDS = ("point", "lower_bound", "undefined")
RULES = ("t", "bootstrap")


def _strict(name: str) -> float:
    raise ValueError(f"{name} is not JSON")


def read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    """``(object, "")``, or ``(None, why not)``. NaN and Infinity are refused, as strict JSON."""
    if not path.is_file():
        return None, f"no {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_strict)
    except (OSError, ValueError) as exc:
        return None, f"{path} is not readable JSON: {_one_line(exc)}"
    if not isinstance(data, dict):
        return None, f"{path} is not a JSON object"
    return data, ""


def _one_line(exc: Exception) -> str:
    """An error as one line: a report row cannot hold a newline (pydantic's errors have them)."""
    return " ".join(str(exc).split())


def _vp(x: Any) -> str:
    return "--" if not isinstance(x, int | float) else f"{x:.3g}"


def _p(x: Any) -> str:
    return "--" if not isinstance(x, int | float) else f"{x:.3f}"


def _premiums(values: Sequence[Any]) -> str:
    return ", ".join(f"{p:g}" for p in values)


# -- the analysis every analysis item reads ----------------------------------- #


def analysis_problem(data: dict[str, Any], grid: GridConfig) -> str:
    """What makes ``data`` not the complete confirmatory analysis of ``grid``; "" if nothing."""
    want = config_hash(grid)
    prov, design = data.get("provenance") or {}, data.get("design") or {}
    if data.get("schema_version") != m4.SCHEMA_VERSION or data.get("milestone") != "M4":
        return (
            f"schema {data.get('schema_version')!r} of milestone {data.get('milestone')!r}, "
            f"not schema {m4.SCHEMA_VERSION} of M4"
        )
    if data.get("confirmatory") is not True or data.get("deviations"):
        shown = "; ".join(str(d) for d in data.get("deviations") or ()) or "no reason given"
        return f"the analysis is exploratory, not confirmatory: {shown}"
    if prov.get("grid_hash") != want or prov.get("confirmatory_grid_hash") != want:
        return (
            f"the analysis is of grid {prov.get('grid_hash')} (its confirmatory grid "
            f"{prov.get('confirmatory_grid_hash')}), not {want} ({grid_filename(grid)})"
        )
    cells = len(grid_cells(grid))
    if prov.get("cells_total") != cells or prov.get("cells_used") != cells:
        return (
            f"incomplete: {prov.get('cells_used')} of {prov.get('cells_total')} cells used, "
            f"the grid has {cells}"
        )
    shape = (design.get("n_seeds"), design.get("premiums"), design.get("arms"))
    if shape != (grid.n_seeds, sorted(grid.premiums), list(grid.arms)):
        return f"the design read (S, premiums, arms) = {shape} is not the grid's"
    arms = data.get("arms") or {}
    missing = [a for a in grid.arms if a not in arms or a not in (data.get("arm_order") or ())]
    if missing:
        return f"no analysis of the {', '.join(missing)} arm"
    return ""


def load_analysis(docs_dir: Path, grid: GridConfig) -> tuple[dict[str, Any] | None, str]:
    """The confirmatory analysis under ``docs_dir``, or ``(None, what failed first)``."""
    data, why = read_json(docs_dir / ANALYSIS_FILE)
    if data is None:
        return None, why
    why = analysis_problem(data, grid)
    return (None, why) if why else (data, "")


def _analysis_line(data: dict[str, Any]) -> str:
    prov, design = data["provenance"], data["design"]
    shas = ", ".join(str(s)[:8] for s in prov.get("sweep_git_shas") or ()) or "?"
    return (
        f"{ANALYSIS_FILE} of {data.get('report_date')}, grid {prov['grid_hash']} "
        f"({prov['cells_used']} of {prov['cells_total']} cells, S = {design['n_seeds']} x "
        f"N = {design.get('n_episodes')}), analysis code {str(prov.get('analysis_git_sha'))[:8]}"
        f"{' (dirty)' if prov.get('analysis_git_dirty') else ''}, sweep cells written at {shas}"
        f"{' (dirty)' if prov.get('sweep_git_dirty') else ''}"
    )


# -- Dominick Dupuy: the power curve, the bootstrap, tau^2 -------------------- #

CURVE_CRITERION = (
    f"docs/{ANALYSIS_FILE}, written by scripts.m4_analysis, is strict JSON of schema "
    f"{m4.SCHEMA_VERSION}, confirmatory (no deviations) and of the A7 grid "
    f"{m4.CONFIRMATORY_GRID_HASH} with every cell used ({len(grid_cells(m4.CONFIRMATORY_GRID))} of "
    f"{len(grid_cells(m4.CONFIRMATORY_GRID))}) and the grid's design; "
    "each arm has a power curve with a point at every premium and a floor whose status "
    f"is one of {', '.join(FLOOR_STATUSES)}; and the gap (PPO's floor minus the "
    f"fixed-delta seller's) is reported with its kind ({', '.join(GAP_KINDS)})"
)


def _floor_text(arm: str, floor: dict[str, Any]) -> str:
    status = floor.get("status")
    if status == "not_reached":
        return (
            f"{arm} floor not reached, above {_vp((floor.get('bracket') or [None])[0])} vol "
            f"points (largest power {_p(floor.get('max_power'))})"
        )
    lo, hi = floor.get("bracket") or [None, None]
    return (
        f"{arm} floor {_vp(floor.get('premium'))} vol points (band {_vp(floor.get('lo'))} to "
        f"{_vp(floor.get('hi'))}, {status}, between sweep points {_vp(lo)} and {_vp(hi)})"
    )


def _gap_text(gap: dict[str, Any]) -> str:
    band, bracket = gap.get("band_range") or [None, None], gap.get("bracket") or [None, None]
    return (
        f"gap ({gap.get('learned')} minus {gap.get('oracle')}) {gap.get('kind')} "
        f"{_vp(gap.get('value'))} vol points (band range {_vp(band[0])} to {_vp(band[1])}, "
        f"sweep points {_vp(bracket[0])} to {_vp(bracket[1])})"
    )


def power_curve_item(docs_dir: Path, grid: GridConfig = m4.CONFIRMATORY_GRID) -> GateItem:
    name = "Power curve analysis"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, CURVE_CRITERION, evidence, why)

    data, why = load_analysis(docs_dir, grid)
    if data is None:
        return fail(f"read {docs_dir / ANALYSIS_FILE}", why)
    head = _analysis_line(data)
    floors: list[str] = []
    for arm in data["arm_order"]:
        a = data["arms"][arm]
        points = [p.get("premium") for p in (a.get("curve") or {}).get("points") or ()]
        if points != sorted(grid.premiums):
            return fail(head, f"the {arm} curve has points at {points}, not at every premium")
        floor = a.get("floor") or {}
        if floor.get("status") not in FLOOR_STATUSES:
            return fail(head, f"the {arm} floor has status {floor.get('status')!r}")
        floors.append(_floor_text(arm, floor))
    gap = data.get("gap")
    if not isinstance(gap, dict) or gap.get("kind") not in GAP_KINDS:
        return fail(f"{head}; {'; '.join(floors)}", f"no gap with a kind in {GAP_KINDS}")
    evidence = f"{head}; {'; '.join(floors)}; {_gap_text(gap)}"
    rules = data.get("rules") or {}
    if rules.get("written_after_sweep"):
        evidence += (
            f"; the curve, floor and gap rules were written {rules.get('written')}, after the "
            f"sweep of {rules.get('sweep_ran')}, and are not pre-registered until A9 (#37)"
        )
    return GateItem(name, DOMINICK, CURVE_CRITERION, evidence, passed=True)


BOOTSTRAP_CRITERION = (
    "on the confirmatory analysis (the power-curve item's), every arm's bootstrap section "
    "ran under the rule m4analysis states (section 5.3, #14; written after the sweep and "
    "before the analysis read it, to be recorded as A9, #37): at least "
    f"{m4.N_BOOT:,} resamples of the seed means at each premium above zero; a level agrees "
    f"when its reference p-value is at least {m4.AGREEMENT_ALPHA:g} over the arm's levels "
    "(Bonferroni), with enough normal replications that the threshold can be reached; every "
    "level's agrees flag follows from its p-value; and the rule in force is the t test when "
    "every level agrees and the bootstrap rule where any diverges. The item reports which "
    "rule is in force per arm; either one passes"
)


def _bootstrap_problem(arm: str, a: dict[str, Any], settings: dict[str, Any]) -> str:
    boot = a.get("bootstrap") or {}
    levels = boot.get("levels") or []
    wanted = [d.get("premium") for d in a.get("decisions") or ()]
    got = [b.get("premium") for b in levels]
    if not levels or got != wanted:
        return f"{arm}: bootstrap levels at {got}, the decisions at {wanted}"
    threshold = float(settings["agreement_alpha"]) / len(levels)
    if 2.0 / (int(settings["reference_replications"]) + 1) > threshold:
        return (
            f"{settings['reference_replications']} normal replications cannot reach a p-value "
            f"under {threshold:g}: the tolerance could never be applied"
        )
    for b in levels:
        p, at = b.get("p_value"), f"{arm} at {b.get('premium'):g}"
        if not isinstance(p, int | float) or not math.isclose(
            b.get("threshold", math.nan), threshold, rel_tol=1e-9
        ):
            return (
                f"{at}: threshold {b.get('threshold')} or p-value {p}, not p against {threshold:g}"
            )
        if b.get("agrees") is not (p >= threshold):
            return f"{at}: agrees is {b.get('agrees')} with p = {p:.4g} against {threshold:g}"
    diverged = [b["premium"] for b in levels if not b["agrees"]]
    rule = "bootstrap" if diverged else "t"
    if boot.get("rule_in_force") != rule or list(boot.get("diverged_at") or ()) != diverged:
        return (
            f"{arm}: rule in force {boot.get('rule_in_force')!r} diverged at "
            f"{boot.get('diverged_at')}, where its levels give {rule!r} at {diverged}"
        )
    return ""


def bootstrap_item(docs_dir: Path, grid: GridConfig = m4.CONFIRMATORY_GRID) -> GateItem:
    name = "Bootstrap validation"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, BOOTSTRAP_CRITERION, evidence, why)

    data, why = load_analysis(docs_dir, grid)
    if data is None:
        return fail(f"read {docs_dir / ANALYSIS_FILE}", f"the M4 analysis is not usable: {why}")
    settings = data.get("settings") or {}
    n_boot = settings.get("n_boot")
    if settings.get("agreement_alpha") != m4.AGREEMENT_ALPHA:
        why = f"agreement alpha {settings.get('agreement_alpha')}, not {m4.AGREEMENT_ALPHA:g}"
        return fail(_analysis_line(data), why)
    if not isinstance(n_boot, int) or n_boot < m4.N_BOOT:
        return fail(_analysis_line(data), f"{n_boot} resamples, fewer than {m4.N_BOOT:,}")
    rules: list[str] = []
    for arm in data["arm_order"]:
        a = data["arms"][arm]
        why = _bootstrap_problem(arm, a, settings)
        if why:
            return fail(_analysis_line(data), why)
        boot = a["bootstrap"]
        k = len(boot["levels"])
        if boot["rule_in_force"] == "t":
            rules.append(f"{arm}: all {k} levels agree, the t test is in force")
        else:
            rules.append(
                f"{arm}: diverges at {_premiums(boot['diverged_at'])} vol points of {k} levels, "
                "the bootstrap rule (one-sided 95% percentile lower bound above zero) is in force"
            )
    size = data.get("bootstrap_rule_size") or {}
    evidence = (
        f"{n_boot:,} resamples, {settings['reference_replications']:,} normal replications, "
        f"agreement at p >= {settings['agreement_alpha']:g} over each arm's levels; "
        + "; ".join(rules)
        + f"; the bootstrap rule's own size at S = {size.get('n_seeds')}: "
        f"{_p(size.get('size'))} (MC se {_p(size.get('mc_se'))}) against alpha "
        f"{settings.get('alpha')}"
    )
    return GateItem(name, DOMINICK, BOOTSTRAP_CRITERION, evidence, passed=True)


HOMOGENEITY_CRITERION = (
    "on the confirmatory analysis, Brown-Forsythe (median-centred Levene) across premiums "
    "on seed means was run for every arm twice, on the contrast's seed means at every "
    "premium above zero and on the arm's own at every premium, each with its df, p-value "
    f"and verdict at alpha {m4.HOMOGENEITY_ALPHA:g} consistent with one another and its "
    "per-level split of tau^2 and sigma^2 / N. The item passes on the test having been run "
    "and reported, whatever it finds: a rejection is a finding to report (section 5.4), "
    "not a failure of the gate"
)


def _homogeneity_problem(
    arm: str, h: dict[str, Any], on: str, premiums: list[float], S: int
) -> str:
    at = f"{arm} ({on})"
    if "brown-forsythe" not in str(h.get("test", "")).lower():
        return f"{at}: the test is {h.get('test')!r}, not Brown-Forsythe"
    if list(h.get("premiums") or ()) != premiums:
        return f"{at}: run on premiums {h.get('premiums')}, not {premiums}"
    k = len(premiums)
    if list(h.get("df") or ()) != [k - 1, k * S - k]:
        return f"{at}: df {h.get('df')}, not [{k - 1}, {k * S - k}] for {k} groups of {S}"
    if [lv.get("premium") for lv in h.get("levels") or ()] != premiums:
        return f"{at}: no per-level split of tau^2 and sigma^2 / N at every premium"
    p = h.get("p_value")
    want = None if p is None else p < m4.HOMOGENEITY_ALPHA
    if h.get("rejected") is not want:
        return f"{at}: rejected is {h.get('rejected')} with p = {p}"
    return ""


def _homogeneity_text(h: dict[str, Any]) -> str:
    verdict = {None: "undefined", True: "equal variance rejected", False: "not rejected"}
    return (
        f"W = {_p(h.get('statistic'))}, df {h['df'][0]}, {h['df'][1]}, p = {_p(h.get('p_value'))}, "
        f"{verdict[h.get('rejected')]}"
    )


def homogeneity_item(docs_dir: Path, grid: GridConfig = m4.CONFIRMATORY_GRID) -> GateItem:
    name = "tau^2 homogeneity test"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, HOMOGENEITY_CRITERION, evidence, why)

    data, why = load_analysis(docs_dir, grid)
    if data is None:
        return fail(f"read {docs_dir / ANALYSIS_FILE}", f"the M4 analysis is not usable: {why}")
    everything = sorted(grid.premiums)
    positive = [p for p in everything if p > 0.0]
    found: list[str] = []
    for arm in data["arm_order"]:
        tests = data["arms"][arm].get("homogeneity") or {}
        parts = []
        for on, premiums in (("contrast", positive), ("arm", everything)):
            h = tests.get(on)
            if not isinstance(h, dict):
                return fail(_analysis_line(data), f"{arm}: no homogeneity test on the {on}")
            why = _homogeneity_problem(arm, h, on, premiums, grid.n_seeds)
            if why:
                return fail(_analysis_line(data), why)
            parts.append(f"on the {on} {_homogeneity_text(h)}")
        found.append(f"{arm} {'; '.join(parts)}")
    evidence = (
        f"Brown-Forsythe on seed means at alpha {m4.HOMOGENEITY_ALPHA:g}, {grid.n_seeds} a group: "
        + "; ".join(found)
    )
    calibration = [c for c in data.get("homogeneity_calibration") or () if isinstance(c, dict)]
    for c in calibration:
        rates = "/".join(_p(w.get("rate")) for w in c.get("power") or ())
        ratios = "/".join(f"{w.get('ratio'):g}" for w in c.get("power") or ())
        evidence += (
            f"; at {c.get('groups')} groups of {c.get('per_group')} the test's simulated size "
            f"is {_p(c.get('size'))} and its power {rates} at {ratios}-fold variance, so a "
            "non-rejection is weak evidence that tau^2 is constant"
        )
    return GateItem(name, DOMINICK, HOMOGENEITY_CRITERION, evidence, passed=True)


# -- Hemdutt Rao: the VRP mapping, read off the curves ------------------------- #

VRP_CRITERION = (
    f"docs/{VRP_FILE} is strict JSON whose source_sha256 is the sha256 docs/"
    f"{DATA_MANIFEST_FILE} records for {PANEL_DAILY} (the panel build it was read from), "
    "with a finite mapped premium inside its interval and a sweep axis equal to the "
    "analysis's premiums; and on the confirmatory analysis each arm's power curve is read "
    "from its knots by vrp.power_at_noncentrality, through the analysis's own "
    "interpolation (noncentrality linear in premium from the anchor at zero, exact "
    "noncentral t), at the mapped premium, at both ends of its interval and over it, with "
    "the range the curve's band spans over that interval; the sidecar's interpolated curve "
    "must be that reading at each of its points. A curve that dips is read as it is; knots "
    "that are not a curve are not a pass"
)


def _linear(curve: list[dict[str, Any]], at: Sequence[float]) -> str:
    """The dense curve read linearly, for the evidence where its knots cannot be read."""
    try:
        xs = [float(c["premium"]) for c in curve]
        ys = [float(c["power"]) for c in curve]
    except (KeyError, TypeError, ValueError):
        return "--"
    if not xs:
        return "--"
    return ", ".join(f"{float(np.interp(x, xs, ys)):.3f} at {x:+.3g}" for x in at)


def read_curve(
    points: list[dict[str, Any]], n_seeds: int, alpha: float, lo: float, hi: float, at: float
) -> tuple[vrp.MappedPower, tuple[float, float]]:
    """The curve at ``at`` and over ``[lo, hi]`` from its knots, and its band's range there.

    Raises ``ValueError`` or ``TypeError`` for knots that are not a curve.
    """
    xs = [float(c["premium"]) for c in points]

    def knots(key: str) -> list[float]:
        return [float(c[key]) for c in points]

    mid = vrp.power_at_noncentrality(xs, knots("noncentrality"), n_seeds, at, (lo, hi), alpha)
    low = vrp.power_at_noncentrality(xs, knots("noncentrality_lo"), n_seeds, at, (lo, hi), alpha)
    high = vrp.power_at_noncentrality(xs, knots("noncentrality_hi"), n_seeds, at, (lo, hi), alpha)
    return mid, (low.power_min, high.power_max)


def dense_mismatch(
    points: list[dict[str, Any]], dense: list[dict[str, Any]], n_seeds: int, alpha: float
) -> str:
    """Where the sidecar's interpolated curve is not its knots' reading; "" if nowhere."""
    for d in dense:
        p = float(d["premium"])
        mid, _ = read_curve(points, n_seeds, alpha, p, p, p)
        if not math.isclose(float(d["power"]), mid.power, rel_tol=0.0, abs_tol=DENSE_ATOL):
            return f"at {p:g} the interpolated curve holds {d['power']}, its knots {mid.power}"
    return ""


def vrp_item(docs_dir: Path, grid: GridConfig = m4.CONFIRMATORY_GRID) -> GateItem:
    name = "VRP mapping with propagated uncertainty"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, HEMDUTT, VRP_CRITERION, evidence, why)

    mapping, why = read_json(docs_dir / VRP_FILE)
    if mapping is None:
        return fail("no mapping", why)
    manifest, why = read_json(docs_dir / DATA_MANIFEST_FILE)
    if manifest is None:
        return fail(f"{VRP_FILE} read", why)
    entry = (manifest.get("files") or {}).get(PANEL_DAILY) or {}
    recorded, digest = entry.get("sha256"), mapping.get("source_sha256")
    source = (
        f"{VRP_FILE} of {mapping.get('generated')} from {PANEL_DAILY} sha256 {str(digest)[:16]}"
    )
    if recorded is None:
        return fail(source, f"{DATA_MANIFEST_FILE} records no sha256 for {PANEL_DAILY}")
    if digest != recorded:
        why = (
            f"the mapping was read from a panel with sha256 {str(digest)[:16]}, the build "
            f"{DATA_MANIFEST_FILE} records is {str(recorded)[:16]}"
        )
        return fail(source, why)
    inputs = mapping.get("power_inputs") or {}
    estimate, interval = inputs.get("estimate"), list(inputs.get("interval") or ())
    if not isinstance(estimate, int | float) or len(interval) != 2:
        return fail(source, f"no finite mapped premium and interval: {inputs}")
    if not all(isinstance(x, int | float) for x in interval):
        return fail(source, f"no finite mapped premium and interval: {inputs}")
    point = float(estimate)
    lo, hi = sorted(float(x) for x in interval)
    if not lo <= point <= hi:
        return fail(
            source, f"the mapped premium {estimate:+.3g} lies outside [{lo:+.3g}, {hi:+.3g}]"
        )
    head = (
        f"{source}, the build {DATA_MANIFEST_FILE} records; mapped premium {estimate:+.3g} vol "
        f"points, 95% [{lo:+.3g}, {hi:+.3g}]"
    )
    data, why = load_analysis(docs_dir, grid)
    if data is None:
        return fail(head, f"the M4 analysis is not usable: {why}")
    axis = mapping.get("sweep_axis")
    if axis is not None and list(axis) != data["design"]["premiums"]:
        why = f"the mapping's sweep axis {axis} is not the analysis's {data['design']['premiums']}"
        return fail(head, why)
    bracket = vrp.bracket(point, data["design"]["premiums"])
    head += f", between sweep points {_vp(bracket[0])} and {_vp(bracket[1])}"
    reads: list[str] = []
    clipped = False
    n_seeds = int(data["design"]["n_seeds"])
    alpha = float((data.get("settings") or {}).get("alpha", vrp.ALPHA))
    for arm in data["arm_order"]:
        curve = data["arms"][arm].get("curve") or {}
        points = list(curve.get("points") or ())
        dense = list(curve.get("interpolated") or ())
        try:
            mid, band = read_curve(points, n_seeds, alpha, lo, hi, point)
            mismatch = dense_mismatch(points, dense, n_seeds, alpha)
        except (KeyError, TypeError, ValueError) as exc:
            evidence = "; ".join([head, *reads, f"{arm} on the interpolated curve, linearly: "])
            return fail(
                evidence + _linear(dense, (point, lo, hi)),
                f"the {arm} curve's knots cannot be read: {_one_line(exc)}",
            )
        if mismatch:
            evidence = "; ".join([head, *reads])
            return fail(evidence, f"the {arm} sidecar is not one curve: {mismatch}")
        clipped = clipped or mid.clipped
        reads.append(
            f"{arm} power {mid.power:.3f} at the mapped premium, {mid.power_low:.3f} to "
            f"{mid.power_high:.3f} at its interval's ends and {mid.power_min:.3f} to "
            f"{mid.power_max:.3f} over it ({band[0]:.3f} to {band[1]:.3f} within the "
            "curve's band)"
        )
    evidence = f"{head}; " + "; ".join(reads)
    if clipped:
        evidence += "; an end outside the curve's grid is held at the curve's end value"
    return GateItem(name, HEMDUTT, VRP_CRITERION, evidence, passed=True)


# -- Ilan Danial: the dashboard's views ---------------------------------------- #

DASHBOARD_CRITERION = (
    f"{SERVER_MODULE}.{APP_FACTORY} builds the dashboard's app, and the app routes an API "
    "path holding 'attribution' (the attribution view) and one with an 'm4' segment or "
    "holding 'power-curve' (the power curve view), matched by name and named in the "
    "evidence; and the built site (web/out) has its index.html and calls each matched "
    "path from a built page or script"
)


def _api_paths(app: Any) -> list[str]:
    paths = (getattr(route, "path", "") for route in getattr(app, "routes", ()))
    return sorted({p for p in paths if isinstance(p, str) and p.startswith("/api/")})


def _callers(web_dir: Path, path: str) -> list[str]:
    """The built files that hold ``path`` (its part before any ``{parameter}``)."""
    needle = path.split("{", 1)[0]
    found = []
    for f in sorted(web_dir.rglob("*")):
        if f.suffix in SITE_SUFFIXES and f.is_file():
            try:
                if needle in f.read_text(encoding="utf-8", errors="replace"):
                    found.append(f.relative_to(web_dir).as_posix())
            except OSError:
                continue
    return found


def dashboard_item(runs_root: Path, docs_dir: Path, web_dir: Path) -> GateItem:
    name = "Dashboard attribution and power curve views"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, ILAN, DASHBOARD_CRITERION, evidence, why)

    factory, why = _load(SERVER_MODULE, APP_FACTORY)
    settings, why_settings = _load(SERVER_MODULE, SETTINGS_CLASS)
    if factory is None or settings is None:
        return fail("no dashboard server", why or why_settings)
    try:
        # The panel is read lazily, so a folder with no panel is enough to build the app.
        app = factory(settings(runs_root, docs_dir, docs_dir, web_dir, docs_dir / "README.md"))
        paths = _api_paths(app)
    except Exception as exc:  # a broken server is a finding, not a crash of the report
        return fail(f"{SERVER_MODULE}.{APP_FACTORY} was called", f"it raised {exc!r}")
    views = {
        "attribution": [p for p in paths if ATTRIBUTION_ROUTE.search(p)],
        "power curve": [p for p in paths if POWER_CURVE_ROUTE.search(p)],
    }
    matched = "; ".join(
        f"{view} view: {', '.join(found) or 'no route'}" for view, found in views.items()
    )
    evidence = f"{len(paths)} API routes; {matched}"
    absent = [view for view, found in views.items() if not found]
    if absent:
        return fail(evidence, f"the app routes no {' and no '.join(absent)} endpoint")
    if not (web_dir / "index.html").is_file():
        return fail(evidence, f"no built site at {web_dir} (run make web)")
    called: list[str] = []
    for view, found in views.items():
        callers = {path: _callers(web_dir, path) for path in found}
        if not any(callers.values()):
            why = f"the built site at {web_dir} never calls the {view} route ({', '.join(found)})"
            return fail(f"{evidence}; site {web_dir}", why)
        for path, files in callers.items():
            if files:
                shown = ", ".join(files[:3]) + (", ..." if len(files) > 3 else "")
                called.append(f"{path} from {shown}")
    evidence += f"; the built site at {web_dir} calls " + "; ".join(called)
    return GateItem(name, ILAN, DASHBOARD_CRITERION, evidence, passed=True)


# -- all members: the collaboration report ------------------------------------- #

COLLABORATION_CRITERION = (
    "the collaboration report due at Evaluation 3 is in docs/ as "
    + " or ".join(COLLABORATION_FILES)
    + " (the project document names no file) and holds more than whitespace"
)


def collaboration_item(docs_dir: Path) -> GateItem:
    name = "Collaboration report"
    for filename in COLLABORATION_FILES:
        path = docs_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            why = f"{path} is empty"
            return _not_cleared(name, TEAM, COLLABORATION_CRITERION, f"{path} found", why)
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        evidence = (
            f"{path}: {len(text.encode()):,} bytes, {len(text.splitlines()):,} lines, "
            f"beginning '{title[:80]}'"
        )
        return GateItem(name, TEAM, COLLABORATION_CRITERION, evidence, passed=True)
    why = f"no {' or '.join(COLLABORATION_FILES)} under {docs_dir}"
    return _not_cleared(name, TEAM, COLLABORATION_CRITERION, "not written", why)


# -- Dominick Dupuy: the attribution residual ---------------------------------- #

RESIDUAL_CRITERION = (
    "for every arm of the A7 grid at least one of its cells (a run the grid record under "
    "the runs root lists) has an attribution under the attribution root made from its run "
    "directory that agrees with its run.json episode by episode (P&L and cost within "
    "attribution.PNL_RTOL, the same trades and steps; bit-for-bit reproduction is reported "
    "beside it, since a cell attributed on another platform agrees without reproducing); "
    "no attribution of a grid cell fails to agree; and every one of them has a residual "
    f"share (the summed absolute residual over gross mark-to-market) under "
    f"{RESIDUAL_SHARE_TOL:.0%} (section 10.9; the scripted arms measure 1.5% to 2.6%). The "
    "evidence states whether vega was exercised: on the flat synthetic surface it is zero "
    "and this residual cannot see a vega error"
)


def attributions(attribution_root: Path) -> tuple[list[RunAttribution], int]:
    """Every readable attribution under ``attribution_root``, and how many were refused."""
    found: list[RunAttribution] = []
    refused = 0
    if not attribution_root.is_dir():
        return found, refused
    for path in sorted(attribution_root.glob("*.json")):
        try:
            found.append(read_attribution(path))
        except (OSError, ValueError):  # pydantic's ValidationError is a ValueError
            refused += 1
    return found, refused


def residual_item(
    runs_root: Path, attribution_root: Path, grid: GridConfig = m4.CONFIRMATORY_GRID
) -> GateItem:
    name = "Attribution residual near zero"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, RESIDUAL_CRITERION, evidence, why)

    grid_path = runs_root / grid_filename(grid)
    if not grid_path.is_file():
        why = f"no grid record at {grid_path}, so no sweep cell to attribute"
        return fail(f"attribution root {attribution_root}", why)
    try:
        record = read_grid(grid_path)
    except (OSError, ValueError) as exc:  # SchemaVersionError and pydantic's are ValueErrors
        return fail(
            f"attribution root {attribution_root}", f"{grid_path} is refused: {_one_line(exc)}"
        )
    cells = {c.run_id: c for c in record.cells}
    found, refused = attributions(attribution_root)
    of_grid = [r for r in found if r.run_id in cells and r.run_dir is not None]
    head = (
        f"{len(of_grid)} attribution(s) of {grid_path.name}'s {len(cells)} cells under "
        f"{attribution_root} ({len(found)} readable, {refused} refused)"
    )
    broken = [r.run_id for r in of_grid if r.agrees is not True]
    if broken:
        return fail(
            head,
            f"{broken[0]} does not agree with its run.json under attribution (or was written "
            "before the agrees verdict; attribute it again)",
        )
    parts: list[str] = []
    for arm in grid.arms:
        mine = [r for r in of_grid if cells[r.run_id].arm == arm]
        if not mine:
            return fail(head, f"no attribution of a {arm} cell under {attribution_root}")
        worst = max(mine, key=lambda r: r.residual_share)
        premiums = sorted({cells[r.run_id].premium for r in mine})
        exact = sum(r.reproduces is True for r in mine)
        parts.append(
            f"{arm}: {len(mine)} cell(s) at premiums {_premiums(premiums)}, residual share at "
            f"most {worst.residual_share:.2%} of gross ({worst.run_id}, largest step residual "
            f"${worst.max_abs_step_residual:,.2f}), every one agreeing with its run.json "
            f"({exact} of {len(mine)} bit for bit)"
        )
        if worst.residual_share >= RESIDUAL_SHARE_TOL:
            why = (
                f"{worst.run_id} ({arm} at {cells[worst.run_id].premium:g}) has a residual share "
                f"of {worst.residual_share:.2%}, not under {RESIDUAL_SHARE_TOL:.0%}"
            )
            return fail(f"{head}; {'; '.join(parts)}", why)
    evidence = f"{head}; " + "; ".join(parts) + "; " + vega_text(of_grid, found)
    return GateItem(name, DOMINICK, RESIDUAL_CRITERION, evidence, passed=True)


def vega_text(of_grid: Sequence[RunAttribution], found: Sequence[RunAttribution]) -> str:
    """Whether the residual above could see a vega error, and what under the root can."""
    grid_ids = {r.run_id for r in of_grid}
    flat = all(r.totals.vega == 0.0 for r in of_grid)
    text = (
        "vega is zero on every attributed sweep cell (the synthetic surface does not move), "
        "so this residual checks delta, gamma and theta, not vega; vega's sign and units are "
        "checked by tests/test_attribution.py on a moving surface"
        if flat
        else "vega is not zero on the attributed sweep cells, so the residual checks it too"
    )
    moving = [r for r in found if r.run_id not in grid_ids and r.totals.vega != 0.0]
    if not moving:
        return text + "; no other attribution under the root exercises vega"
    worst = max(moving, key=lambda r: r.residual_share)
    sources = ", ".join(sorted({r.source for r in moving}))
    return text + (
        f"; {len(moving)} other attribution(s) under the root exercise vega ({sources}), "
        f"residual share at most {worst.residual_share:.2%} ({worst.run_id})"
    )
