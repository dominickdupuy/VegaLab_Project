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

Before replaying, the saved file is checked against the digest its manifest
recorded, so a results file edited after the fact reads as a failure of the run,
not of the replay. A mismatch reports the first episode that differs; the git SHA
and library versions recorded against those now running are reported either way,
because they are the first suspects when a replay fails.

The replay never writes into the run directory it checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from optspread.artifacts import RUN_FILENAME, RunRecord, dump_run
from optspread.harness import run_episodes, summarize
from optspread.manifest import RunManifest, file_sha256, git_sha, library_versions
from optspread.sweep import run_id
from optspread.train import Trainer, check_disjoint, train_ppo


@dataclass(frozen=True, slots=True)
class ReplayResult:
    run_dir: Path
    run_id: str
    matches: bool
    reason: str
    seconds: float

    def render(self) -> str:
        verdict = "IDENTICAL" if self.matches else "DIFFERS"
        return f"{verdict}  {self.run_id}  ({self.seconds:.1f}s)  {self.reason}"


def _provenance(manifest: RunManifest) -> str:
    notes = []
    now = git_sha()
    if now != manifest.git_sha:
        notes.append(f"recorded at git {manifest.git_sha[:8]}, replayed at {now[:8]}")
    if manifest.git_dirty:
        notes.append("recorded from a dirty tree")
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


def replay_run(run_dir: Path, trainer: Trainer = train_ppo) -> ReplayResult:
    """Rerun ``run_dir`` from its manifest and compare ``run.json`` byte for byte."""
    start = time.perf_counter()

    def result(rid: str, matches: bool, reason: str) -> ReplayResult:
        return ReplayResult(run_dir, rid, matches, reason, time.perf_counter() - start)

    try:
        manifest = RunManifest.read(run_dir)
        saved_bytes = (run_dir / RUN_FILENAME).read_bytes()
        saved = RunRecord.model_validate_json(saved_bytes)
    except (OSError, ValueError) as exc:
        return result(run_dir.name, False, f"not a readable run: {exc}")
    rid = manifest.run_id
    if manifest.artifacts.get(RUN_FILENAME) != file_sha256(run_dir / RUN_FILENAME):
        return result(rid, False, "run.json no longer matches the digest in its manifest")
    config = manifest.config
    if run_id(config) != rid or saved.run_id != rid:
        return result(rid, False, f"run id {rid} does not belong to the recorded config")

    try:
        if config.train is None:
            episodes = run_episodes(config)
        else:
            check_disjoint(config)
            episodes = run_episodes(config, trainer(config))
    except ImportError as exc:
        return result(rid, False, f"cannot retrain: {exc} (install the train extra)")
    replayed = RunRecord.of(config, summarize(episodes), episodes)
    context = _provenance(manifest)
    if dump_run(replayed).encode() == saved_bytes:
        reason = f"run.json sha256 {file_sha256(run_dir / RUN_FILENAME)[:16]} reproduced"
        return result(rid, True, f"{reason}; {context}" if context else reason)
    reason = _first_difference(saved, replayed)
    return result(rid, False, f"{reason}; {context}" if context else reason)


def replay_matches(run_dir: Path, trainer: Trainer = train_ppo) -> tuple[bool, str]:
    """``(identical, reason)`` for one run: the call the M2 gate report makes."""
    outcome = replay_run(run_dir, trainer)
    return outcome.matches, outcome.reason


def find_runs(path: Path) -> list[Path]:
    """``path`` if it is a run directory, else every run directory directly under it."""
    if (path / RUN_FILENAME).exists():
        return [path]
    return sorted(p for p in path.iterdir() if (p / RUN_FILENAME).exists())
