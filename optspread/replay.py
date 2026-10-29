"""Deterministic replay: rerun a finished run from its manifest and compare bytes.

Lane: Ilan Danial (machinery). Project document 10.7, the reproducibility gate:
any completed run reproduces bit-for-bit from its hash.

The manifest carries the full resolved ``RunConfig`` (training included, for a PPO
run) and the seed is part of it, so the config alone is the recipe. ``replay_run``
retrains when the arm is learned, re-evaluates, renders ``run.json`` through the
same ``dump_run`` the writer uses, and compares the bytes with the file on disk.
Bytes, not a tolerance: a run that reproduces to the ninth decimal has still not
reproduced, and the difference would be the first sign of a nondeterminism nobody
has found yet.

Before replaying, every file the manifest hashed (``run.json``, and the saved
policy of a learned run) is checked against its recorded digest, so a file edited
or swapped after the fact reads as a failure of the run, not of the replay. A
mismatch reports the first episode that differs; the git SHA and library versions
recorded against those now running are reported either way, because they are the
first suspects when a replay fails.

The episodes are rerun on the data source the manifest records (the generator for
a manifest written before the source was recorded, which is every run made before
it was a parameter). A run on the real panel replays only against the panel it
read: the panel's files are hashed again and a rebuilt or missing panel is a
failure that says so, not a replay on different data. It replays as the run was made:
evaluation on the ``eval`` split and, for a learned run, retraining on ``train`` (the
dates of pre-registration A8; ``eval`` too for a real-data manifest from before the split
was recorded). A learned run without a recorded split was trained on the generator
(nothing trained on the panel before the split was recorded), and is retrained there. A
record of dates of its own (``DataRecord.ranges``, a surface of your own) names no root to
read them from, so it is refused rather than replayed on the team's panel.

A learned run is retrained only by the trainer its manifest names
(``manifest.trainer``): a run trained by a stand-in is reported as that, not as a P&L
difference. A run made with components of one's own replays only through the same ones
(``DataRecord.components`` against ``sweep.components_identity`` of ``runner``).

Bytes are the rule, and the default. Across platforms the last bit of libm can differ
(the sweep ran on Windows), so ``rtol`` asks instead for every number of ``run.json``
within a relative tolerance, every count and identifier exact: the verdict is then
IDENTICAL (the bytes still match), WITHIN (with the largest relative difference), or
DIFFERS. A manifest records the platform it was written on (``RunManifest.platform``),
and the reason names both when they differ. A tolerance does not make a retrained PPO
run comparable across platforms: training compounds a last-bit difference into
another policy (different trades, a different mean), so a PPO cell replays only on the
platform it was trained on; elsewhere it is checked statistically
(``optspread.reproduction``).

The replay never writes into the run directory it checks.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from optspread.artifacts import RUN_FILENAME, RunRecord, dump_run
from optspread.data import DataUnavailableError, make_source
from optspread.harness import run_episodes, summarize
from optspread.manifest import (
    DataRecord,
    RunManifest,
    callable_name,
    describe_platform,
    file_sha256,
    git_sha,
    library_versions,
    platform_record,
)
from optspread.sweep import Runner, changed_artifacts, components_identity, expected_data, run_id
from optspread.train import Trainer, check_disjoint, train_on, train_ppo

Verdict = Literal["IDENTICAL", "WITHIN", "DIFFERS"]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    run_dir: Path
    run_id: str
    matches: bool  # run.json reproduced byte for byte
    reason: str
    seconds: float
    rtol: float | None = None  # the tolerance asked for; None is bytes only
    within: bool = False  # not identical, every number within ``rtol``
    max_rel_diff: float | None = None  # over the numbers compared, with ``rtol``

    @property
    def reproduced(self) -> bool:
        """Identical, or within the tolerance asked for."""
        return self.matches or self.within

    @property
    def verdict(self) -> Verdict:
        if self.matches:
            return "IDENTICAL"
        return "WITHIN" if self.within else "DIFFERS"

    def render(self) -> str:
        head: str = self.verdict
        if self.within:
            head += f" rtol {self.rtol:g} (max rel diff {self.max_rel_diff:.2g})"
        return f"{head}  {self.run_id}  ({self.seconds:.1f}s)  {self.reason}"


def _provenance(manifest: RunManifest) -> str:
    notes = []
    now = git_sha()
    if now != manifest.git_sha:
        notes.append(f"recorded at git {manifest.git_sha[:8]}, replayed at {now[:8]}")
    if manifest.git_dirty:
        notes.append("recorded from a dirty tree")
    here = platform_record()
    if manifest.platform is not None and manifest.platform != here:
        notes.append(
            f"recorded on {describe_platform(manifest.platform)}, "
            f"replayed on {describe_platform(here)}"
        )
    current = library_versions()
    moved = sorted(
        f"{k} {v} -> {current.get(k, 'absent')}"
        for k, v in manifest.versions.items()
        if current.get(k) != v
    )
    if moved:
        notes.append("versions moved: " + ", ".join(moved))
    return "; ".join(notes)


def _first_difference(saved: RunRecord, replayed: RunRecord) -> str:
    for a, b in zip(saved.episodes, replayed.episodes, strict=False):
        if a != b:
            return f"episode seed {a.seed}: saved pnl {a.pnl!r}, replayed {b.pnl!r}"
    if len(saved.episodes) != len(replayed.episodes):
        return f"{len(saved.episodes)} episodes saved, {len(replayed.episodes)} replayed"
    for name in RunRecord.model_fields:
        if getattr(saved, name) != getattr(replayed, name):
            return f"{name}: saved {getattr(saved, name)!r}, replayed {getattr(replayed, name)!r}"
    return "the rendered bytes differ with every field equal"


def numeric_difference(saved: Any, replayed: Any, rtol: float, at: str = "") -> tuple[float, str]:
    """The largest relative difference between two parsed ``run.json`` documents, and
    the first place they differ beyond ``rtol`` or in kind ("" when nowhere).

    Floats are compared relatively (``|a - b| / max(|a|, |b|)``, zero when both are);
    integers, strings, booleans and the shape of the document must be equal."""
    if isinstance(saved, dict) and isinstance(replayed, dict):
        if saved.keys() != replayed.keys():
            return math.inf, f"{at or 'run.json'}: keys {sorted(saved)} against {sorted(replayed)}"
        worst, first = 0.0, ""
        for key in saved:
            diff, where = numeric_difference(saved[key], replayed[key], rtol, f"{at}.{key}")
            worst, first = max(worst, diff), first or where
        return worst, first
    if isinstance(saved, list) and isinstance(replayed, list):
        if len(saved) != len(replayed):
            return math.inf, f"{at}: {len(saved)} entries saved, {len(replayed)} replayed"
        worst, first = 0.0, ""
        for i, (a, b) in enumerate(zip(saved, replayed, strict=True)):
            diff, where = numeric_difference(a, b, rtol, f"{at}[{i}]")
            worst, first = max(worst, diff), first or where
        return worst, first
    floats = isinstance(saved, float) or isinstance(replayed, float)
    numbers = all(isinstance(x, int | float) and not isinstance(x, bool) for x in (saved, replayed))
    if floats and numbers:
        scale = max(abs(saved), abs(replayed))
        diff = 0.0 if saved == replayed else abs(saved - replayed) / scale
        if not diff <= rtol:  # NaN included
            return diff, f"{at}: saved {saved!r}, replayed {replayed!r}"
        return diff, ""
    if type(saved) is not type(replayed) or saved != replayed:
        return math.inf, f"{at}: saved {saved!r}, replayed {replayed!r}"
    return 0.0, ""


def replay_run(
    run_dir: Path,
    trainer: Trainer = train_ppo,
    runner: Runner = run_episodes,
    *,
    rtol: float | None = None,
) -> ReplayResult:
    """Rerun ``run_dir`` from its manifest and compare ``run.json`` byte for byte.

    ``runner`` is the one the run was made with: a run made with components of your
    own replays only through the same components, since its config does not name them,
    and a learned run only with the trainer its manifest names. With ``rtol``, a
    ``run.json`` that differs in its bytes is compared number by number (the module
    docstring): WITHIN when every number is within ``rtol``.
    """
    start = time.perf_counter()

    def result(rid: str, matches: bool, reason: str, **within: Any) -> ReplayResult:
        seconds = time.perf_counter() - start
        return ReplayResult(run_dir, rid, matches, reason, seconds, rtol, **within)

    try:
        manifest = RunManifest.read(run_dir)
        saved_bytes = (run_dir / RUN_FILENAME).read_bytes()
        saved = RunRecord.model_validate_json(saved_bytes)
    except (OSError, ValueError) as exc:
        return result(run_dir.name, False, f"not a readable run: {exc}")
    rid = manifest.run_id
    changed = changed_artifacts(run_dir, manifest)
    if RUN_FILENAME in changed:
        return result(rid, False, "run.json no longer matches the digest in its manifest")
    if changed:
        return result(rid, False, f"{', '.join(changed)} no longer match the manifest's digests")
    config = manifest.config
    if run_id(config) != rid or saved.run_id != rid:
        return result(rid, False, f"run id {rid} does not belong to the recorded config")
    data = manifest.data_record
    if (
        config.train is not None
        and manifest.trainer is not None
        and manifest.trainer != callable_name(trainer)
    ):
        return result(
            rid,
            False,
            f"trained by {manifest.trainer}, not {callable_name(trainer)}: replay it "
            "with the trainer that made it",
        )
    if data.components != components_identity(runner):
        made = data.components or "the default components"
        asked = components_identity(runner) or "the default components"
        return result(rid, False, f"made with {made}, replayed with {asked}")
    if data.components is not None:
        try:
            now = expected_data(runner, config, data.source)
        except DataUnavailableError as exc:
            return result(rid, False, f"cannot replay on {data.source}: {exc}")
        if now.panel != data.panel:
            return result(
                rid,
                False,
                f"the {data.source} panel is not the one the run read "
                f"(recorded {data.panel_id}, found {now.panel_id})",
            )
    elif data.ranges is not None or data.effective_split not in (None, "eval"):
        dates = data.ranges if data.ranges is not None else f"the {data.effective_split} split"
        return result(rid, False, f"ran on {dates}; the harness replays A8's eval split only")
    else:
        try:
            now = DataRecord.of(make_source(data.source, config.market, split="eval"))
        except DataUnavailableError as exc:
            return result(rid, False, f"cannot replay on {data.source}: {exc}")
        if now.panel != data.panel:
            return result(
                rid,
                False,
                f"the {data.source} panel is not the one the run read "
                f"(recorded {data.panel_id}, found {now.panel_id})",
            )
    # The default runner reruns on the recorded source; your own components bring theirs.
    run = partial(run_episodes, source=data.source) if runner is run_episodes else runner

    try:
        if config.train is None:
            episodes = run(config, None)
        else:
            check_disjoint(config)
            trained_on = data.source if data.split is not None else "synthetic"
            episodes = run(config, train_on(trainer, config, trained_on))
    except ImportError as exc:
        return result(rid, False, f"cannot retrain: {exc} (install the train extra)")
    replayed = RunRecord.of(config, summarize(episodes), episodes)
    context = _provenance(manifest)
    if manifest.platform is None:
        context = "; ".join(c for c in (context, "platform not recorded") if c)
    replayed_bytes = dump_run(replayed).encode()
    if replayed_bytes == saved_bytes:
        reason = f"run.json sha256 {file_sha256(run_dir / RUN_FILENAME)[:16]} reproduced"
        moved = _provenance(manifest)
        return result(rid, True, f"{reason}; {moved}" if moved else reason)
    if rtol is not None:
        worst, where = numeric_difference(json.loads(saved_bytes), json.loads(replayed_bytes), rtol)
        if not where:
            reason = f"every number of run.json within {rtol:g}, largest {worst:.2g}"
            return result(
                rid,
                False,
                f"{reason}; {context}" if context else reason,
                within=True,
                max_rel_diff=worst,
            )
        reason = f"beyond {rtol:g} at {where.lstrip('.')}"
        return result(rid, False, f"{reason}; {context}" if context else reason, max_rel_diff=worst)
    reason = _first_difference(saved, replayed)
    return result(rid, False, f"{reason}; {context}" if context else reason)


def platform_differs(run_dir: Path) -> bool:
    """Whether ``run_dir`` was written on another platform than this one, or on one its
    manifest does not record (an older manifest): where bytes are not to be expected."""
    try:
        recorded = RunManifest.read(run_dir).platform
    except (OSError, ValueError):
        return True
    return recorded != platform_record()


def replay_matches(run_dir: Path, trainer: Trainer = train_ppo) -> tuple[bool, str]:
    """``(identical, reason)`` for one run: the call the M2 gate report makes."""
    outcome = replay_run(run_dir, trainer)
    return outcome.matches, outcome.reason


def find_runs(path: Path) -> list[Path]:
    """``path`` if it is a run directory, else every run directory directly under it."""
    if (path / RUN_FILENAME).exists():
        return [path]
    return sorted(p for p in path.iterdir() if (p / RUN_FILENAME).exists())
