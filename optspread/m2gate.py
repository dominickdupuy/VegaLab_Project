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
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optspread.artifacts import RUN_FILENAME
from optspread.data import conformance_suite
from optspread.gatereport import GateItem
from optspread.manifest import MANIFEST_FILENAME, IncompleteManifestError, RunManifest, file_sha256
from optspread.orchestrate import PILOT_MATRIX, PILOT_ORACLE_MATRIX, read_grid
from optspread.power import (
    VarianceComponents,
    detectable_effect_t,
    frontier,
    variance_components,
)
from optspread.sweep import changed_artifacts
from optspread.validation import (
    DEFAULT_CASE,
    DEFAULT_DELTAS,
    NULL_WORLD_DRAG,
    PASS_CRITERION,
    POISONED_PIPELINES,
    PREREGISTERED_EVENT,
    REJECTED_EVENTS,
    validate_null_world,
    validate_pipeline,
)

# -- the names the parallel work is expected to land under ------------------- #

TRAIN_MODULE = "optspread.train"
TRAIN_FUNCTION = "train_ppo"
# () -> record with .values, .hash, .library_version, .matches_library, .differences (D3)
HYPERPARAMETERS = "frozen_hyperparameters"
REPLAY_MODULE = "optspread.replay"
REPLAY_FUNCTION = "replay_matches"  # (run_dir: Path) -> (matches, reason)
DASHBOARD_MODULE = "optspread.dashboard_data"
RUNS_FUNCTION = "scan_runs"  # (runs_root: Path) -> rows with .path, one per manifest.json
GATES_FUNCTION = "load_gate_boards"  # (docs_dir: Path) -> boards with .source, one per report

# The pilot's floor on its own size: below these the components are too noisy to plan on.
PILOT_MIN_SEEDS = 5
PILOT_MIN_EPISODES = 20
PREREG_PATH = "docs/PRE_REGISTRATION.md"
PREREG_TAG_PATTERN = "prereg-*"

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


def results_runs(runs_root: Path) -> list[tuple[Path, RunManifest]]:
    """The valid manifests of runs proper: those recording a ``run.json``.

    A pilot's matrices carry a manifest of their own (``orchestrate.write_pilot_matrices``)
    keyed by a PPO config, but that directory holds matrices, not a run; it is neither a
    training run nor something a replay can reproduce. A run whose files no longer
    match their recorded digests is not evidence either, so it is left out too.
    """
    return [
        (d, m)
        for d, m in run_manifests(runs_root)[0]
        if RUN_FILENAME in m.artifacts and not changed_artifacts(d, m)
    ]


def is_trained(manifest: RunManifest) -> bool:
    """A PPO run: the config names the learned arm and carries its training config.

    Library versions are no evidence: ``library_versions`` records torch and
    Stable-Baselines3 in every manifest written where they are installed.
    """
    return manifest.config.policy == "ppo" and manifest.config.train is not None


# -- Ilan Danial: harness, replay, dashboard --------------------------------- #

PPO_CRITERION = (
    f"{TRAIN_MODULE} exposes a {TRAIN_FUNCTION} entry point and one pinned "
    f"{HYPERPARAMETERS} record (D3) whose values equal the installed Stable-Baselines3 "
    "defaults (its matches_library is True; not installed is not a pass); and at least one "
    "PPO run (policy ppo with a training config) under the runs root has a valid manifest"
)


def ppo_harness_item(runs_root: Path) -> GateItem:
    name = "PPO harness at fixed D3 hyperparameters"
    train, why = _load(TRAIN_MODULE, TRAIN_FUNCTION)
    if train is None or not callable(train):
        why = why or f"{TRAIN_MODULE}.{TRAIN_FUNCTION} is not callable"
        return _not_cleared(name, ILAN, PPO_CRITERION, "no training entry point", why)
    loaded, why = _load(TRAIN_MODULE, HYPERPARAMETERS)
    record = loaded() if callable(loaded) else loaded
    pinned = getattr(record, "values", None)
    if not isinstance(pinned, Mapping) or not pinned:
        why = why or f"{HYPERPARAMETERS} is not a non-empty mapping"
        return _not_cleared(name, ILAN, PPO_CRITERION, "the hyperparameters are not pinned", why)
    version = getattr(record, "library_version", None)
    pinned_text = f"{len(pinned)} pinned values, hash {getattr(record, 'hash', '?')}"
    matches = getattr(record, "matches_library", None)
    if matches is None:
        why = "stable-baselines3 is not installed, so the pinned set was not compared"
        return _not_cleared(name, ILAN, PPO_CRITERION, pinned_text, why)
    if not matches:
        differences = ", ".join(str(d) for d in getattr(record, "differences", ())) or "?"
        why = f"the pinned set differs from the stable-baselines3 {version} defaults: {differences}"
        return _not_cleared(name, ILAN, PPO_CRITERION, pinned_text, why)
    trained = [m for _, m in results_runs(runs_root) if is_trained(m)]
    evidence = (
        f"{pinned_text}, equal to the stable-baselines3 {version} defaults; {len(trained)} "
        f"PPO run(s) under {runs_root}"
    )
    if not trained:
        why = f"no PPO run with a valid manifest under {runs_root}"
        return _not_cleared(name, ILAN, PPO_CRITERION, evidence, why)
    return GateItem(name, ILAN, PPO_CRITERION, evidence, passed=True)


REPLAY_CRITERION = (
    f"{REPLAY_MODULE}.{REPLAY_FUNCTION} re-runs a recorded run from its manifest (config "
    "hash and seed) and reproduces every artifact digest the manifest records; one PPO run "
    "and one scripted run are replayed when both kinds are under the runs root, at least "
    "one run in any case"
)


def replay_item(runs_root: Path) -> GateItem:
    name = "Deterministic replay from hash"
    replay, why = _load(REPLAY_MODULE, REPLAY_FUNCTION)
    if replay is None:
        return _not_cleared(name, ILAN, REPLAY_CRITERION, "no replay entry point", why)
    runs = results_runs(runs_root)
    if not runs:
        return _not_cleared(
            name,
            ILAN,
            REPLAY_CRITERION,
            "nothing to replay",
            f"no valid run manifest under {runs_root}",
        )
    kinds = {
        "PPO": [r for r in runs if is_trained(r[1])],
        "scripted": [r for r in runs if not is_trained(r[1])],
    }
    replayed: list[str] = []
    for kind, of_kind in kinds.items():
        if not of_kind:
            continue
        run_dir, manifest = of_kind[0]
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
        replayed.append(f"{manifest.run_id} ({kind}, {len(manifest.artifacts)} artifacts)")
    counts = ", ".join(f"{len(v)} {k}" for k, v in kinds.items())
    evidence = (
        f"replayed {' and '.join(replayed)} of {len(runs)} runs ({counts}) under {runs_root}; "
        "digests match"
    )
    absent = [k for k, v in kinds.items() if not v]
    if absent:
        evidence += f"; no {absent[0]} run under {runs_root} to replay"
    return GateItem(name, ILAN, REPLAY_CRITERION, evidence, passed=True)


DASHBOARD_CRITERION = (
    f"the dashboard's data layer ({DASHBOARD_MODULE}.{RUNS_FUNCTION} and {GATES_FUNCTION}) "
    "lists every run directory under the runs root that has a valid manifest recording "
    "its run.json, of which there is at least one, and reads every gate report JSON "
    "sidecar under docs/"
)


def dashboard_item(runs_root: Path, docs_dir: Path) -> GateItem:
    name = "Dashboard run browser and gate board"
    scan, why = _load(DASHBOARD_MODULE, RUNS_FUNCTION)
    load_boards, why_gates = _load(DASHBOARD_MODULE, GATES_FUNCTION)
    if scan is None or load_boards is None:
        return _not_cleared(
            name, ILAN, DASHBOARD_CRITERION, "no dashboard data layer", why or why_gates
        )
    # Runs proper only: a pilot's matrix manifest is not a run, and the browser skips it.
    runs = results_runs(runs_root)
    sidecars = sorted(docs_dir.glob("*_GATE_REPORT.json"))
    try:
        listed = {Path(row.path).resolve() for row in scan(runs_root)}
        read = {Path(board.source).resolve() for board in load_boards(docs_dir)}
    except Exception as exc:
        return _not_cleared(
            name, ILAN, DASHBOARD_CRITERION, "the data layer was called", f"it raised {exc!r}"
        )
    unlisted = [m.run_id for d, m in runs if d.resolve() not in listed]
    unread = [s.name for s in sidecars if s.resolve() not in read]
    evidence = (
        f"run browser lists {len(runs) - len(unlisted)} of {len(runs)} valid manifests under "
        f"{runs_root} ({len(listed)} rows in all); gate board reads "
        f"{len(sidecars) - len(unread)} of {len(sidecars)} sidecars"
    )
    if not runs:
        why = f"no valid run manifest under {runs_root}: nothing real to browse"
    elif unlisted:
        why = f"the run browser does not list {len(unlisted)} valid run(s): {', '.join(unlisted)}"
    elif unread:
        why = f"the gate board does not read {', '.join(unread)}"
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
    f"the pilot's S x N contrast matrices ({PILOT_MATRIX}, PPO at the pilot premium minus "
    f"PPO at zero, and {PILOT_ORACLE_MATRIX}, the fixed-delta seller's same contrast; the "
    "pre-registered detection event of #26) sit beside a valid manifest that records both "
    "digests and whose config is the PPO arm with its training config; "
    f"S >= {PILOT_MIN_SEEDS}, N >= {PILOT_MIN_EPISODES}, every entry finite, N equal to the "
    "manifest's n_episodes and S to the grid's seed count where the grid record is present; "
    "tau^2 and sigma^2 of the PPO contrast (what the pre-registered test tests) are reported "
    "with standard errors beside the seller's, and turned into the exact noncentral-t "
    "detection floor at one-sided alpha = 0.05 and 80% power"
)


def _components(label: str, vc: VarianceComponents) -> str:
    return (
        f"{label} tau^2 = {vc.tau2:,.4g} (se {vc.se_tau2:,.4g})"
        f"{' clamped at 0' if vc.clamped else ''}, sigma^2 = {vc.sigma2:,.4g} "
        f"(se {vc.se_sigma2:,.4g})"
    )


def _grid_seeds(pilot_root: Path, run_id: str) -> int | None:
    """The seed count of the grid record under ``pilot_root`` holding ``run_id``, if any."""
    for path in sorted(pilot_root.glob("grid-*.json")):
        try:
            grid = read_grid(path)
        except (OSError, ValueError):  # SchemaVersionError and pydantic's are ValueErrors
            continue
        if any(c.run_id == run_id for c in grid.cells):
            return grid.grid.n_seeds
    return None


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

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, PILOT_CRITERION, evidence, why)

    matrix_path = pilot_root / PILOT_MATRIX
    if not matrix_path.exists():
        return fail(f"no {matrix_path}", "pilot not run")
    try:
        manifest = RunManifest.read(pilot_root)
    except FileNotFoundError:
        why = f"{matrix_path} has no {MANIFEST_FILENAME} beside it, so it is not reproducible"
        return fail(f"{matrix_path} found", why)
    except (IncompleteManifestError, ValueError) as exc:
        return fail(f"{matrix_path} found", f"manifest refused: {exc}")
    at = f"run {manifest.run_id}"
    config = manifest.config
    if config.policy != "ppo" or config.train is None:
        why = f"the manifest's config is the {config.policy} arm, not PPO with a training config"
        return fail(at, why)
    arms: list[NDArray[np.float64]] = []
    for filename in (PILOT_MATRIX, PILOT_ORACLE_MATRIX):
        path = pilot_root / filename
        recorded = manifest.artifacts.get(filename)
        if recorded is None:
            return fail(at, f"the manifest does not record {filename}")
        if not path.exists():
            return fail(at, f"{filename} is recorded but missing")
        if recorded != file_sha256(path):
            return fail(at, f"{filename} changed under its manifest")
        arms.append(np.asarray(np.load(path), dtype=np.float64))
    ppo, oracle = arms
    shape = "x".join(str(d) for d in ppo.shape)
    if ppo.shape != oracle.shape:
        other = "x".join(str(d) for d in oracle.shape)
        return fail(at, f"{PILOT_MATRIX} is {shape} and {PILOT_ORACLE_MATRIX} {other}: not paired")
    if (
        ppo.ndim != 2
        or ppo.shape[0] < PILOT_MIN_SEEDS
        or ppo.shape[1] < PILOT_MIN_EPISODES
        or not (np.isfinite(ppo).all() and np.isfinite(oracle).all())
    ):
        why = (
            f"the matrix is {shape}, not S x N with S >= {PILOT_MIN_SEEDS}, "
            f"N >= {PILOT_MIN_EPISODES} and every entry finite"
        )
        return fail(at, why)
    S, N = int(ppo.shape[0]), int(ppo.shape[1])
    if config.n_episodes != N:
        return fail(
            at, f"the matrix has N = {N} episodes, the manifest's config {config.n_episodes}"
        )
    seeds = _grid_seeds(pilot_root, manifest.run_id)
    premium = f"{config.market.iv_premium_vol_pts:g}"
    if seeds is not None and seeds != S:
        return fail(at, f"the matrix has S = {S} seeds, the grid record {seeds}")
    vc = variance_components(ppo)  # already the contrast the event tests
    floors = ", ".join(
        f"{detectable_effect_t(vc.tau2, vc.sigma2, s, N):,.4g} at S = {s}" for s in (5, 50)
    )
    evidence = (
        f"{S} x {N} pilot ({at}, {config.train.total_timesteps:,} training steps per seed, "
        f"{'grid of ' + str(seeds) + ' seeds' if seeds is not None else 'no grid record'}): "
        f"{_components(f'contrast ppo at {premium} - ppo at 0', vc)}; "
        f"{_components(f'delta_seller at {premium} - at 0', variance_components(oracle))}; "
        f"exact-t detection floor of the PPO contrast at N = {N}: {floors}"
    )
    if delta is not None:
        designs = frontier(
            delta,
            vc.tau2,
            vc.sigma2,
            seeds=(2, 4, 8, 16, 32, 64),
            train_steps_per_seed=train_steps_per_seed,
            episode_days=config.env.episode_days,
            train_rate=train_rate,
            eval_rate=eval_rate,
        )
        costed = [
            (d.budget.total_hours, d.n_seeds, d.n_episodes, d.power)
            for d in designs
            if d.budget is not None and d.power is not None
        ]
        if costed:
            hours, n_seeds, episodes, power = min(costed)
            evidence += (
                f"; for delta = {delta:,.4g} the cheapest design is {n_seeds} seeds x {episodes} "
                f"episodes (normal-approximation power {power:.3f}, {hours:,.1f} core-hours at "
                f"{train_steps_per_seed:,} training steps per seed, a placeholder)"
            )
        else:
            evidence += f"; for delta = {delta:,.4g}: no design up to 64 seeds is feasible"
    return GateItem(name, DOMINICK, PILOT_CRITERION, evidence, passed=True)


ANALYSIS_CRITERION = (
    PASS_CRITERION
    + "; and each deliberately wrong rule ("
    + ", ".join(POISONED_PIPELINES)
    + ") fails the same check; and in the null world (a flat learner, a control paying "
    + f"a cost drag of {NULL_WORLD_DRAG:g}, nothing planted) the pre-registered event, "
    + PREREGISTERED_EVENT[0]
    + ", fires within the same tolerance of alpha while the event it replaced ("
    + ", ".join(REJECTED_EVENTS)
    + ") does not"
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
    null = validate_null_world(replications=replications, event=PREREGISTERED_EVENT[1])
    replaced = {
        label: validate_null_world(replications=replications, event=event)
        for label, event in REJECTED_EVENTS.items()
    }
    kept = [label for label, check in replaced.items() if check.within()]
    evidence = (
        f"{DEFAULT_CASE.describe()}; R = {replications:,} per effect; rejection rate/closed "
        f"form: {rates}; largest deviation {result.worst.z:+.2f} MC se ({result.worst.name}); "
        f"{len(POISONED_PIPELINES) - len(accepted)} of {len(POISONED_PIPELINES)} wrong rules "
        f"refused; the normal approximation reads {at.normal:.3f} against {at.analytic:.3f} "
        f"at delta = {at.delta:g}; null world: {PREREGISTERED_EVENT[0]} fires at "
        f"{null.observed:.3f} against alpha = {null.expected:g} ({null.z:+.2f} MC se), "
        + ", ".join(f"{label} at {c.observed:.3f}" for label, c in replaced.items())
    )
    if not result.passed:
        c = result.failures[0]
        why = f"{c.name}: {c.observed:.4f} against {c.expected:.4f} (z = {c.z:.2f})"
        return _not_cleared(name, DOMINICK, ANALYSIS_CRITERION, evidence, why)
    if accepted:
        why = f"the gate accepted a wrong rule: {', '.join(accepted)}"
        return _not_cleared(name, DOMINICK, ANALYSIS_CRITERION, evidence, why)
    if not null.within():
        why = (
            f"the pre-registered event fires at {null.observed:.4f} in the null world, "
            f"against alpha = {null.expected:g} (z = {null.z:.2f})"
        )
        return _not_cleared(name, DOMINICK, ANALYSIS_CRITERION, evidence, why)
    if kept:
        why = f"the null world did not refuse a replaced event: {', '.join(kept)}"
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
