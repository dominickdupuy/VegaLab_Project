"""The M2 gate items: one small evidence function per deliverable.

Lane: Dominick Dupuy (statistical spine).

The M2 row of the schedule (project document section 12) is the training and
evaluation harness with hyperparameters fixed; the reproducibility layer with
hashed manifests and replay; the variance-components pilot; the analysis pipeline
validated against a known analytic case; the pre-registration committed and
tagged; and the dashboard run browser and gate board live on real manifests.
Week 6 (issue #3, W6-4) adds the real data adapter.

Each function returns a ``GateItem`` and never raises for a missing deliverable.
Three of them depend on code being written in parallel (the PPO harness and replay,
W6-5; the dashboard, W6-6), so each imports what it needs by the names below and
reads NOT CLEARED with the import that failed when it is absent. Wiring the real
modules in is a matter of changing a name here, not the logic.

The pre-registration convention: the confirmatory analysis is fixed by the
earliest annotated or lightweight tag matching ``prereg-*`` (``prereg-m2`` at the
M2 meeting). A purpose-named tag rather than a release tag, because a ``v0.2``
says nothing about which file it freezes. After the tag the file may only grow:
the item checks that HEAD's copy begins with the tagged text byte for byte, which
is the "amendments are appended, never edited in place" rule made checkable.

``scripts/m2_gate_report.py`` assembles these into ``docs/M2_GATE_REPORT.md``.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
from collections.abc import Mapping, Sized
from pathlib import Path
from typing import Any

import numpy as np

from optspread.data import conformance_suite
from optspread.gatereport import GateItem
from optspread.manifest import MANIFEST_FILENAME, IncompleteManifestError, RunManifest, file_sha256
from optspread.power import detectable_effect, frontier, variance_components
from optspread.validation import (
    DEFAULT_CASE,
    DEFAULT_DELTAS,
    PASS_CRITERION,
    POISONED_PIPELINES,
    validate_pipeline,
)

# -- the names the parallel work is expected to land under ------------------- #

TRAIN_MODULE = "optspread.train"
TRAIN_FUNCTION = "train_ppo"
HYPERPARAMETERS = "frozen_hyperparameters"  # () -> record whose .values maps PPO kwargs (D3)
REPLAY_MODULE = "optspread.replay"
REPLAY_FUNCTION = "replay_matches"  # (run_dir: Path) -> (matches, reason)
DASHBOARD_MODULE = "optspread.dashboard_data"
RUNS_FUNCTION = "list_runs"  # (runs_root: Path) -> Sized, one entry per run
GATES_FUNCTION = "list_gate_reports"  # (docs_dir: Path) -> Sized, one entry per sidecar

PILOT_MATRIX = "matrix.npy"
PREREG_PATH = "docs/PRE_REGISTRATION.md"
PREREG_TAG_PATTERN = "prereg-*"
LEARNER_VERSIONS = ("torch", "stable_baselines3")

DOMINICK, ILAN, HEMDUTT = "Dominick Dupuy", "Ilan Danial", "Hemdutt Rao"


def _not_cleared(name: str, owner: str, criterion: str, evidence: str, why: str) -> GateItem:
    return GateItem(name, owner, criterion, evidence, passed=False, failed_first=why)


def _load(module: str, attr: str) -> tuple[Any, str]:
    """``(module.attr, "")``, or ``(None, what failed first)``."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module.startswith(exc.name):
            ours = module.startswith("optspread.")
            return None, f"{module} {'does not exist yet' if ours else 'is not installed'}"
        return None, f"{module} cannot import {exc.name} (is the train extra installed?)"
    except Exception as exc:  # a broken module is a finding, not a crash of the report
        return None, f"{module} failed to import: {exc!r}"
    value = getattr(mod, attr, None)
    if value is None:
        return None, f"{module} has no {attr}"
    return value, ""


def run_manifests(runs_root: Path) -> tuple[list[tuple[Path, RunManifest]], int]:
    """Every run directory under ``runs_root`` with a valid manifest, and how many were refused."""
    valid: list[tuple[Path, RunManifest]] = []
    refused = 0
    if not runs_root.is_dir():
        return valid, refused
    for path in sorted(runs_root.rglob(MANIFEST_FILENAME)):
        try:
            valid.append((path.parent, RunManifest.read(path.parent)))
        except (OSError, ValueError):  # IncompleteManifestError is a ValueError
            refused += 1
    return valid, refused


# -- Ilan Danial: harness, replay, dashboard --------------------------------- #

PPO_CRITERION = (
    f"{TRAIN_MODULE} exposes a {TRAIN_FUNCTION} entry point and one pinned "
    f"{HYPERPARAMETERS} mapping (D3); every pinned key is a Stable-Baselines3 PPO "
    "argument; and at least one run under the runs root has a valid manifest recording "
    "the learner's library versions"
)


def ppo_harness_item(runs_root: Path) -> GateItem:
    name = "PPO harness at fixed D3 hyperparameters"
    train, why = _load(TRAIN_MODULE, TRAIN_FUNCTION)
    if train is None or not callable(train):
        why = why or f"{TRAIN_MODULE}.{TRAIN_FUNCTION} is not callable"
        return _not_cleared(name, ILAN, PPO_CRITERION, "no training entry point", why)
    record, why = _load(TRAIN_MODULE, HYPERPARAMETERS)
    pinned = getattr(record(), "values", None) if callable(record) else record
    if not isinstance(pinned, Mapping) or not pinned:
        why = why or f"{HYPERPARAMETERS} is not a non-empty mapping"
        return _not_cleared(name, ILAN, PPO_CRITERION, "the hyperparameters are not pinned", why)
    ppo, why = _load("stable_baselines3", "PPO")
    if ppo is None:
        return _not_cleared(name, ILAN, PPO_CRITERION, f"{len(pinned)} pinned keys", why)
    parameters = inspect.signature(ppo.__init__).parameters
    unknown = sorted(str(k) for k in pinned if k not in parameters)
    if unknown:
        return _not_cleared(
            name,
            ILAN,
            PPO_CRITERION,
            f"{len(pinned)} pinned keys",
            f"not PPO arguments: {', '.join(unknown)}",
        )
    changed = sorted(str(k) for k, v in pinned.items() if parameters[str(k)].default != v)
    trained = [
        m for _, m in run_manifests(runs_root)[0] if all(v in m.versions for v in LEARNER_VERSIONS)
    ]
    evidence = (
        f"{len(pinned)} pinned PPO arguments, {len(changed)} off the library default"
        f"{' (' + ', '.join(changed) + ')' if changed else ''}; {len(trained)} training "
        f"run(s) under {runs_root}"
    )
    if not trained:
        return _not_cleared(
            name,
            ILAN,
            PPO_CRITERION,
            evidence,
            f"no training run with a valid manifest under {runs_root}",
        )
    return GateItem(name, ILAN, PPO_CRITERION, evidence, passed=True)


REPLAY_CRITERION = (
    f"{REPLAY_MODULE}.{REPLAY_FUNCTION} re-runs a recorded run from its manifest (config "
    "hash and seed) and reproduces every artifact digest the manifest records"
)


def replay_item(runs_root: Path, limit: int = 1) -> GateItem:
    name = "Deterministic replay from hash"
    replay, why = _load(REPLAY_MODULE, REPLAY_FUNCTION)
    if replay is None:
        return _not_cleared(name, ILAN, REPLAY_CRITERION, "no replay entry point", why)
    runs, _ = run_manifests(runs_root)
    if not runs:
        return _not_cleared(
            name,
            ILAN,
            REPLAY_CRITERION,
            "nothing to replay",
            f"no valid run manifest under {runs_root}",
        )
    replayed: list[str] = []
    for run_dir, manifest in runs[:limit]:
        try:
            result = replay(run_dir)
        except Exception as exc:
            return _not_cleared(
                name,
                ILAN,
                REPLAY_CRITERION,
                f"replaying {manifest.run_id}",
                f"replay raised {exc!r}",
            )
        # A (matches, reason) pair is truthy whatever it says, so it is unpacked first.
        matched, reason = (
            (bool(result[0]), str(result[1])) if isinstance(result, tuple) else (bool(result), "")
        )
        if not matched:
            reason = reason or "a replayed artifact digest differs from its manifest"
            return _not_cleared(
                name,
                ILAN,
                REPLAY_CRITERION,
                f"replaying {manifest.run_id}",
                f"{manifest.run_id}: {reason}",
            )
        replayed.append(f"{manifest.run_id} ({len(manifest.artifacts)} artifacts)")
    evidence = (
        f"replayed {', '.join(replayed)} of {len(runs)} runs under {runs_root}; digests match"
    )
    return GateItem(name, ILAN, REPLAY_CRITERION, evidence, passed=True)


DASHBOARD_CRITERION = (
    f"the dashboard's data layer ({DASHBOARD_MODULE}) lists every run directory under the "
    "runs root that has a valid manifest, of which there is at least one, and every gate "
    "report JSON sidecar under docs/"
)


def dashboard_item(runs_root: Path, docs_dir: Path) -> GateItem:
    name = "Dashboard run browser and gate board"
    list_runs, why = _load(DASHBOARD_MODULE, RUNS_FUNCTION)
    list_gates, why_gates = _load(DASHBOARD_MODULE, GATES_FUNCTION)
    if list_runs is None or list_gates is None:
        return _not_cleared(
            name, ILAN, DASHBOARD_CRITERION, "no dashboard data layer", why or why_gates
        )
    runs, _ = run_manifests(runs_root)
    sidecars = sorted(docs_dir.glob("*_GATE_REPORT.json"))
    try:
        listed: Sized = list_runs(runs_root)
        boards: Sized = list_gates(docs_dir)
        n_listed, n_boards = len(listed), len(boards)
    except Exception as exc:
        return _not_cleared(
            name, ILAN, DASHBOARD_CRITERION, "the data layer was called", f"it raised {exc!r}"
        )
    evidence = (
        f"run browser lists {n_listed} of {len(runs)} manifests under {runs_root}; gate "
        f"board reads {n_boards} of {len(sidecars)} sidecars"
    )
    if not runs:
        why = f"no valid run manifest under {runs_root}: nothing real to browse"
    elif n_listed != len(runs):
        why = f"the run browser lists {n_listed} runs, {len(runs)} have valid manifests"
    elif n_boards != len(sidecars):
        why = f"the gate board reads {n_boards} reports, {len(sidecars)} sidecars exist"
    else:
        return GateItem(name, ILAN, DASHBOARD_CRITERION, evidence, passed=True)
    return _not_cleared(name, ILAN, DASHBOARD_CRITERION, evidence, why)


# -- Hemdutt Rao: the real adapter ------------------------------------------- #

ADAPTER_CRITERION = (
    "the real optionmetrics source is available (its SPX panel is present) and clears "
    "conformance_report, the provider contract and structural invariants every source "
    "meets; SKIPPED is not a pass"
)


def adapter_item() -> GateItem:
    name = "Real data adapter conformance"
    reports = {r.source: r for r in conformance_suite()}
    real = reports["optionmetrics"]
    evidence = "; ".join(
        f"{r.source} {'PASS' if r.passed else 'SKIPPED' if not r.checked else 'FAIL'}"
        + (f" ({r.n_chains} chains)" if r.checked else "")
        for r in reports.values()
    )
    if not real.checked:
        why = "optionmetrics is unavailable (SKIPPED): no SPX panel for the adapter to serve"
        return _not_cleared(name, HEMDUTT, ADAPTER_CRITERION, evidence, why)
    if not real.passed:
        return _not_cleared(name, HEMDUTT, ADAPTER_CRITERION, evidence, real.violations[0])
    return GateItem(name, HEMDUTT, ADAPTER_CRITERION, evidence, passed=True)


# -- Dominick Dupuy: pilot, analysis validation, pre-registration ------------ #

PILOT_CRITERION = (
    f"an S x N pilot matrix ({PILOT_MATRIX}, S >= 2, N >= 2, finite) sits beside a valid "
    "manifest that records its digest; tau^2 and sigma^2 are reported with standard errors "
    "and turned into the detection floor at one-sided alpha = 0.05 and 80% power"
)


def pilot_item(
    pilot_root: Path,
    delta: float | None = None,
    *,
    train_steps_per_seed: int = 1_000_000,
    train_rate: float = 1_000.0,
    eval_rate: float = 1_800.0,
) -> GateItem:
    """Rates default to the M1-measured planning figures; the training budget is a placeholder."""
    name = "Variance-components pilot"
    matrix_path = pilot_root / PILOT_MATRIX
    if not matrix_path.exists():
        return _not_cleared(name, DOMINICK, PILOT_CRITERION, f"no {matrix_path}", "pilot not run")
    try:
        manifest = RunManifest.read(pilot_root)
    except FileNotFoundError:
        why = f"{matrix_path} has no {MANIFEST_FILENAME} beside it, so it is not reproducible"
        return _not_cleared(name, DOMINICK, PILOT_CRITERION, f"{matrix_path} found", why)
    except (IncompleteManifestError, ValueError) as exc:
        return _not_cleared(
            name, DOMINICK, PILOT_CRITERION, f"{matrix_path} found", f"manifest refused: {exc}"
        )
    recorded = manifest.artifacts.get(PILOT_MATRIX)
    if recorded != file_sha256(matrix_path):
        why = (
            f"the manifest does not record {PILOT_MATRIX}"
            if recorded is None
            else f"{PILOT_MATRIX} changed under its manifest"
        )
        return _not_cleared(name, DOMINICK, PILOT_CRITERION, f"run {manifest.run_id}", why)
    outcomes = np.load(matrix_path)
    shape = "x".join(str(d) for d in outcomes.shape)
    if outcomes.ndim != 2 or min(outcomes.shape) < 2 or not np.isfinite(outcomes).all():
        why = f"the matrix is {shape}, not S x N with S, N >= 2 and every entry finite"
        return _not_cleared(name, DOMINICK, PILOT_CRITERION, f"run {manifest.run_id}", why)
    vc = variance_components(outcomes)
    S, N = vc.n_seeds, vc.n_episodes
    floors = ", ".join(
        f"{detectable_effect(vc.tau2, vc.sigma2, s, N):,.4g} at S = {s}" for s in (5, 50)
    )
    evidence = (
        f"{S} x {N} pilot ({manifest.config.policy}, run {manifest.run_id}): tau^2 = "
        f"{vc.tau2:,.4g} (se {vc.se_tau2:,.4g}){' clamped at 0' if vc.clamped else ''}, "
        f"sigma^2 = {vc.sigma2:,.4g} (se {vc.se_sigma2:,.4g}); detection floor at N = {N}: "
        f"{floors}"
    )
    if delta is not None:
        designs = frontier(
            delta,
            vc.tau2,
            vc.sigma2,
            seeds=(2, 4, 8, 16, 32, 64),
            train_steps_per_seed=train_steps_per_seed,
            episode_days=manifest.config.env.episode_days,
            train_rate=train_rate,
            eval_rate=eval_rate,
        )
        costed = [
            (d.budget.total_hours, d.n_seeds, d.n_episodes, d.power)
            for d in designs
            if d.budget is not None and d.power is not None
        ]
        if costed:
            hours, seeds, episodes, power = min(costed)
            evidence += (
                f"; for delta = {delta:,.4g} the cheapest design is {seeds} seeds x {episodes} "
                f"episodes (power {power:.3f}, {hours:,.1f} core-hours at "
                f"{train_steps_per_seed:,} training steps per seed, a placeholder)"
            )
        else:
            evidence += f"; for delta = {delta:,.4g}: no design up to 64 seeds is feasible"
    return GateItem(name, DOMINICK, PILOT_CRITERION, evidence, passed=True)


ANALYSIS_CRITERION = (
    PASS_CRITERION
    + "; and each deliberately wrong rule ("
    + ", ".join(POISONED_PIPELINES)
    + ") fails the same check"
)


def analysis_item(replications: int = 10_000, seed: int = 0) -> GateItem:
    """The section 8 gate, live, with its negative controls.

    The controls also guard the replication count: below about 2,000 per effect the
    gate can no longer tell normal critical values from t ones at S = 10, and the
    item then fails on the control rather than passing on a gate too blunt to matter.
    """
    name = "Analysis pipeline validated against a known analytic case"
    result = validate_pipeline(DEFAULT_CASE, DEFAULT_DELTAS, replications, seed)
    accepted = [
        label
        for label, pipeline in POISONED_PIPELINES.items()
        if validate_pipeline(DEFAULT_CASE, DEFAULT_DELTAS, replications, seed, pipeline).passed
    ]
    rates = ", ".join(f"{p.empirical:.3f}/{p.analytic:.3f} at {p.delta:g}" for p in result.points)
    at = max(result.points, key=lambda p: p.normal - p.analytic)
    evidence = (
        f"{DEFAULT_CASE.describe()}; R = {replications:,} per effect; rejection rate/closed "
        f"form: {rates}; largest deviation {result.worst.z:+.2f} MC se ({result.worst.name}); "
        f"{len(POISONED_PIPELINES) - len(accepted)} of {len(POISONED_PIPELINES)} wrong rules "
        f"refused; the normal approximation reads {at.normal:.3f} against {at.analytic:.3f} "
        f"at delta = {at.delta:g}"
    )
    if not result.passed:
        c = result.failures[0]
        why = f"{c.name}: {c.observed:.4f} against {c.expected:.4f} (z = {c.z:.2f})"
        return _not_cleared(name, DOMINICK, ANALYSIS_CRITERION, evidence, why)
    if accepted:
        why = f"the gate accepted a wrong rule: {', '.join(accepted)}"
        return _not_cleared(name, DOMINICK, ANALYSIS_CRITERION, evidence, why)
    return GateItem(name, DOMINICK, ANALYSIS_CRITERION, evidence, passed=True)


PREREG_CRITERION = (
    f"{PREREG_PATH} is tracked with no uncommitted changes; the earliest "
    f"{PREREG_TAG_PATTERN} tag points at a commit on this history that contains it; and "
    "the file at HEAD begins with the tagged text byte for byte (amendments appended, "
    "never edited in place)"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def prereg_item(repo: Path) -> GateItem:
    name = "Pre-registration committed and tagged"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, PREREG_CRITERION, evidence, why)

    if _git(repo, "ls-files", "--error-unmatch", PREREG_PATH).returncode != 0:
        return fail(f"{repo}", f"{PREREG_PATH} is not tracked")
    head = _git(repo, "rev-parse", "--short=8", "HEAD").stdout.strip()
    if _git(repo, "diff", "--quiet", "HEAD", "--", PREREG_PATH).returncode != 0:
        return fail(f"tracked at {head}", f"{PREREG_PATH} has uncommitted changes")
    tags = _git(repo, "tag", "--list", PREREG_TAG_PATTERN, "--sort=creatordate").stdout.split()
    if not tags:
        return fail(f"committed at {head}", f"no {PREREG_TAG_PATTERN} tag: committed, not tagged")
    tag = tags[0]
    tagged = _git(repo, "show", f"{tag}^{{commit}}:{PREREG_PATH}")
    sha = _git(repo, "rev-parse", "--short=8", f"{tag}^{{commit}}").stdout.strip()
    when = _git(repo, "log", "-1", "--format=%cs", f"{tag}^{{commit}}").stdout.strip()
    evidence = f"{tag} at {sha} ({when})"
    if tagged.returncode != 0:
        return fail(evidence, f"{tag} points at a commit without {PREREG_PATH}")
    if _git(repo, "merge-base", "--is-ancestor", f"{tag}^{{commit}}", "HEAD").returncode != 0:
        return fail(evidence, f"{tag} is not on the history of HEAD ({head})")
    current = _git(repo, "show", f"HEAD:{PREREG_PATH}").stdout
    if not current.startswith(tagged.stdout):
        return fail(evidence, f"{PREREG_PATH} was edited in place after {tag}")
    grown = len(current) - len(tagged.stdout)
    evidence += (
        f"; HEAD {head} appends {grown:,} characters of amendments" if grown else "; unchanged"
    )
    return GateItem(name, DOMINICK, PREREG_CRITERION, evidence, passed=True)
