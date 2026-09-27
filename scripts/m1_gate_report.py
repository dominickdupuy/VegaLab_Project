"""Build the M1 gate report from live runs and write docs/M1_GATE_REPORT.md.

Lane: Dominick Dupuy (statistical spine).

A milestone closes on a report, not on code that happens to run. Every item below
states the criterion it had to meet, the evidence, and on a failure what failed
first. The numbers come from running the gates here rather than from prose, so the
report cannot drift away from the tests, and the Presentation 1 slides are built
from the file this writes.

    python -m scripts.m1_gate_report
    python -m scripts.m1_gate_report --quick --out -
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from optspread.config import RunConfig
from optspread.data import conformance_suite
from optspread.gatereport import GateItem, build_report
from optspread.gates import wave0_gate
from optspread.leaks import default_env, field_coverage, leak_floor
from optspread.manifest import IncompleteManifestError, RunManifest
from optspread.scripted import cost_drag_check

DOC = Path("docs/M1_GATE_REPORT.md")


def wave0_item(episodes: int) -> GateItem:
    report = wave0_gate(n_episodes=episodes, seed=1000)
    ao_free = report.arms[("always_on", False)]
    ao_cost = report.arms[("always_on", True)]
    return GateItem(
        name="Wave 0 no-edge gate",
        owner="Hemdutt Rao",
        criterion=(
            "flat P&L exactly 0 with and without costs; no-cost always-on |t| <= 3; "
            "with-cost always-on negative and below no-cost; random bleeds fastest"
        ),
        evidence=(
            f"{episodes} episodes/arm: always-on {ao_free.mean_pnl:,.0f} free "
            f"(t={ao_free.t_stat:.2f}), {ao_cost.mean_pnl:,.0f} costed"
        ),
        passed=report.passed,
        failed_first="" if report.passed else "see make gate for the failing ordering",
    )


def drag_item(episodes: int) -> GateItem:
    report = cost_drag_check(n_episodes=episodes, seed=2000)
    worst = max(report.rows, key=lambda r: abs(r.drag_gap))
    return GateItem(
        name="Cost, fill and margin model",
        owner="Hemdutt Rao",
        criterion=(
            "for every scripted seller: costed P&L equals free P&L minus what was "
            "charged, per seed; a cash settlement charges nothing; no structure "
            "leaves the book unpaid"
        ),
        evidence=(
            f"{len(report.rows)} sellers, {episodes} episodes each; measured drag "
            f"matches prediction to {abs(worst.drag_gap):.2f} dollars"
        ),
        passed=report.passed,
        failed_first="" if report.passed else "a seller lost more or less than its modelled cost",
    )


def adapter_item() -> GateItem:
    reports = conformance_suite()
    available = [r for r in reports if r.checked]
    skipped = [r for r in reports if not r.checked]
    ok = bool(available) and all(r.passed for r in available)
    return GateItem(
        name="Data adapter boundary",
        owner="Hemdutt Rao",
        criterion=(
            "every available source clears the provider contract and the structural "
            "invariants; an unavailable source reports SKIPPED and never PASS"
        ),
        evidence=(
            f"{len(available)} sources conform ({', '.join(r.source for r in available)}); "
            f"{len(skipped)} skipped ({', '.join(r.source for r in skipped) or 'none'})"
        ),
        passed=ok,
        failed_first="" if ok else "; ".join(v for r in available for v in r.violations),
    )


def leak_item(seeds: int) -> GateItem:
    report = field_coverage(default_env, range(seeds))
    return GateItem(
        name="Leak detection suite",
        owner="Dominick Dupuy",
        criterion=(
            "no observation field at t predicts the true return into t+1 at a "
            "Bonferroni-adjusted alpha; every field in OBS_NAMES is tested under the "
            "probe or records why not; every field has a positive control that fires"
        ),
        evidence=(
            f"{report.clean.n_pairs:,} pairs over {seeds} seeds; "
            f"{report.clean.n_features_tested} of {len(report.controls)} fields tested, "
            f"untested: {', '.join(report.clean.untested) or 'none'}; all "
            f"{len(report.controls)} controls fire and are attributed correctly"
        ),
        passed=report.passed,
        failed_first=""
        if report.passed
        else f"uncovered: {', '.join(report.uncovered) or 'the clean world was flagged'}",
    )


def floor_item(seeds: int) -> GateItem:
    report = leak_floor(default_env, eps_grid=(1.0, 0.5, 0.25, 0.1, 0.05), seeds=range(seeds))
    found = report.floor is not None
    return GateItem(
        name="Leak detection floor",
        owner="Dominick Dupuy",
        criterion="the smallest planted leak the detector still catches is recorded",
        evidence=(
            f"grid {report.eps_grid}, caught {report.flagged}, floor eps = {report.floor}"
            if found
            else f"grid {report.eps_grid}: nothing was caught"
        ),
        passed=found,
        failed_first="" if found else "the detector caught nothing on the grid",
    )


def manifest_item() -> GateItem:
    manifest = RunManifest.create(RunConfig(policy="always_on", seed=1))
    complete = True
    try:
        manifest.validate_complete()
    except IncompleteManifestError:
        complete = False
    refused = False
    try:
        manifest.model_copy(update={"git_sha": "unknown"}).validate_complete()
    except IncompleteManifestError:
        refused = True
    ok = complete and refused
    failed = "a complete manifest failed validation" if not complete else "an incomplete one passed"
    return GateItem(
        name="Run artifacts and manifests",
        owner="Ilan Danial",
        criterion=(
            "every run writes a hashed manifest with config, commit, seed and package "
            "versions; a manifest missing any required field is refused, not written"
        ),
        evidence=(
            f"run_id {manifest.run_id}, git {manifest.git_sha[:8]}, "
            f"{len(manifest.versions)} versions recorded; an unknown commit is refused"
        ),
        passed=ok,
        failed_first="" if ok else failed,
    )


def throughput_item(steps: int) -> GateItem:
    from scripts.bench import bench_policy

    arm = bench_policy("always_on", steps)
    inside = 500.0 <= arm.per_second <= 3000.0
    return GateItem(
        name="Throughput, measured",
        owner="Ilan Danial",
        criterion=(
            "the schedule uses a measured steps/s figure for the steady-state "
            "workload, not the document's unvalidated 500-3,000 estimate"
        ),
        evidence=f"{arm.per_second:,.0f} steps/s/core over {arm.steps:,} steps (always_on)",
        passed=inside,
        failed_first="" if inside else f"{arm.per_second:,.0f} steps/s falls outside 500-3,000",
    )


def ppo_item() -> GateItem:
    try:
        import stable_baselines3  # noqa: F401

        available = True
    except ImportError:
        available = False
    return GateItem(
        name="PPO-inclusive throughput",
        owner="Ilan Danial",
        criterion="the training-update arm reports a measured steps/s figure",
        evidence="the ppo arm is wired at the fixed hyperparameters of decision D3",
        passed=available,
        failed_first=""
        if available
        else "stable-baselines3 is not installed, so the arm reports 'not measured'",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller samples, for a smoke run")
    parser.add_argument("--out", default=str(DOC), help="output path, or - for stdout")
    args = parser.parse_args()
    episodes, seeds, steps = (20, 5, 4_000) if args.quick else (100, 10, 20_000)

    report = build_report(
        "M1",
        [
            wave0_item(episodes),
            drag_item(episodes),
            adapter_item(),
            leak_item(seeds),
            floor_item(seeds),
            manifest_item(),
            throughput_item(steps),
            ppo_item(),
        ],
        date.today(),
    )
    rendered = report.render()
    if args.out == "-":
        print(rendered)
        return 0 if report.passed else 1
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    print(f"wrote {target} ({'PASS' if report.passed else 'NOT CLEARED'})")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
