"""A run directory's name, and the rule for reusing one already on disk.

Lane: Ilan Danial (machinery).

A run directory is named by ``run_id``: the config hash and the seed. ``valid_run`` is
the resume rule every orchestrator shares: a directory is read back rather than
recomputed only when it is present, readable, keyed by the expected hash and run id,
its data the record the run asked for would write (``expected_data``,
``data_mismatch``), every file its manifest hashed still the one it hashed
(``changed_artifacts``), and a learned run trained by the trainer asked for
(``recorded_trainer``). ``foreign_run`` says when a directory must not be written:
when it holds this config's finished run on other data or made with other components.

A cell made with components of one's own (a runner other than the harness) records them
(``components_identity``), and every reuse check here compares the whole record with the
one the run asked for would write (``expected_data``): the source, the split, the
components, and on the real panel its digests, so a cell read from a panel since rebuilt
is recomputed rather than mixed into a grid with cells of the new build.

Where runs live by default: ``default_runs_root`` (``$VEGALAB_DATA/runs``, else ``runs``
under the current directory), and under it ``source_root``, a real-data run's own folder,
because the source is not in a run id. ``JobSource`` names the sources a run can be
launched on.

``root_lock`` is the rule the orchestrators share about writers: one at a time computes
runs under a results root, holding ``<root>/.optspread.lock`` (``LOCK_FILENAME``); a
second one waits ``LOCK_TIMEOUT`` seconds and then stops with ``RootInUseError``, which
names the lock file. The lock is the operating system's, so it goes with its holder
however that exits, killed included. The file is a plain file named like no run artifact,
so nothing that looks for runs (``run.json``, ``manifest.json``, ``sweep.json``,
``grid-*.json``) takes it for one. On Windows it is removed on release; on Linux and macOS
the empty file stays. The worker processes of an orchestrator write under the lock their
parent holds and never take it.

Moved from ``optspread.sweep`` (the run-directory rules), ``optspread.orchestrate``
(the runs roots) and ``optspread.jobs`` (the sources), which re-export it.
"""

from __future__ import annotations

import os
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, get_args

from filelock import FileLock, Timeout

from optspread.artifacts import RUN_FILENAME, RunRecord, read_run
from optspread.config import RunConfig, config_hash
from optspread.data import ChainSource, DataUnavailableError, SourceName, make_source
from optspread.extend import describe
from optspread.harness import EpisodeResult, run_episodes
from optspread.manifest import (
    MANIFEST_FILENAME,
    DataRecord,
    RunManifest,
    callable_name,
    file_sha256,
)
from optspread.policies import Policy
from optspread.train import POLICY_FILENAME, train_ppo

# What the real learner records as a learned run's trainer (as ``optspread.m2gate``
# names it), and a member of every archive Stable-Baselines3 saves, which no stand-in
# writes: how a manifest from before the trainer was recorded is told apart (#31).
REAL_TRAINER = callable_name(train_ppo)
SB3_ARCHIVE_MEMBER = "_stable_baselines3_version"

# The one path from a config to episodes. Injected so a caller can count or wrap it;
# the default is the harness runner every gate uses.
Runner = Callable[[RunConfig, Policy | None], tuple[EpisodeResult, ...]]


def run_id(config: RunConfig) -> str:
    """``<config hash>-s<seed>``: the id ``RunRecord`` and ``RunManifest`` give the same run."""
    return f"{config_hash(config)}-s{config.seed}"


def changed_artifacts(run_dir: Path, manifest: RunManifest) -> tuple[str, ...]:
    """Recorded artifacts that are missing or no longer match their digest, in order.

    ``run.json`` counts as changed when the manifest does not record it at all: a
    manifest that hashes the policy but not the results vouches for nothing shown.
    """
    recorded = dict(manifest.artifacts)
    recorded.setdefault(RUN_FILENAME, "")
    return tuple(
        rel
        for rel, digest in sorted(recorded.items())
        if not (run_dir / rel).is_file() or file_sha256(run_dir / rel) != digest
    )


def recorded_trainer(cell_dir: Path, manifest: RunManifest) -> str | None:
    """The trainer a learned run's manifest names, or for one written before the trainer
    was recorded, ``REAL_TRAINER`` when its hashed policy is a Stable-Baselines3 archive
    (the rule of ``optspread.m2gate.not_real_learner``); None when nothing tells."""
    if manifest.trainer is not None:
        return manifest.trainer
    policy = cell_dir / POLICY_FILENAME
    recorded = manifest.artifacts.get(POLICY_FILENAME)
    if recorded is None or not policy.is_file() or recorded != file_sha256(policy):
        return None
    try:
        with zipfile.ZipFile(policy) as archive:
            return REAL_TRAINER if SB3_ARCHIVE_MEMBER in archive.namelist() else None
    except (OSError, zipfile.BadZipFile):
        return None


_NO_IDENTITY = object()


def components_identity(runner: Runner) -> str | None:
    """What a run made by ``runner`` records as its components (``DataRecord.components``).

    None for the harness, ``run_episodes`` itself or bound to a source, and for the
    default ``Components()``; a ``Components``' ``identity`` otherwise; and for any other
    runner its name (``extend.describe``), since nothing says it runs the default world.
    """
    if runner is run_episodes:
        return None
    if (
        isinstance(runner, partial)
        and runner.func is run_episodes
        and not runner.args
        and set(runner.keywords) <= {"source"}
    ):
        return None
    identity: Any = getattr(runner, "identity", _NO_IDENTITY)
    if identity is None or isinstance(identity, str):
        return identity
    return describe(runner)


def expected_data(
    runner: Runner, config: RunConfig, source: SourceName = "synthetic"
) -> DataRecord:
    """The data record a run of ``config`` by ``runner`` on ``source`` writes, as it stands
    now: on the real panel with the panel's digests (raises ``DataUnavailableError``
    when it is absent). A runner of one's own records its components, and a
    ``Components`` the source it builds (its panel too, when that is the real one)."""
    components = components_identity(runner)
    if components is None:
        return DataRecord.of(make_source(source, config.market, split="eval"))
    factory = getattr(runner, "source", None)
    if callable(factory):
        chains = factory(config.market)
        if isinstance(chains, ChainSource):
            return DataRecord.of(chains, components)
    return DataRecord(components=components)


def _components(name: str | None) -> str:
    return name if name is not None else "the default components"


def source_mismatch(manifest: RunManifest, source: SourceName) -> str:
    """Why ``manifest`` is not a default run on ``source``'s ``eval`` split; "" when it is.

    The source and the split are not in the config hash, so the same run id on other data,
    or on other dates of the same data, is a different run: the harness evaluates on
    ``eval`` under A8's ``SPLITS``, so a record of another split or of ranges of its own
    is not its run. Nor is a run made with components of one's own. A manifest from before
    the source was recorded ran on the generator, and a real-data one from before the split
    was recorded ran on ``eval``. The panel is not compared here (``data_mismatch``).
    """
    data = manifest.data_record
    if data.components is not None:
        return f"{manifest.run_id} was made with {data.components}, not the default components"
    if data.source != source:
        return f"{manifest.run_id} ran on {data.source}, not {source}"
    want = "eval" if source == "optionmetrics" else None
    if data.effective_split != want:
        return f"{manifest.run_id} ran under the split {data.effective_split}, not {want}"
    if data.ranges is not None:
        return f"{manifest.run_id} ran on dates of its own ({data.ranges}), not A8's splits"
    return ""


def data_mismatch(manifest: RunManifest, want: DataRecord, *, panel: bool = True) -> str:
    """Why ``manifest``'s data is not ``want`` (``expected_data``); "" when it is.

    The components, the source, the split and the dates, and with ``panel`` the panel's
    digests: a real-data run read from a panel since rebuilt is a run of other data, as
    the replay says too. A record from before the split was recorded ran on ``eval``.
    """
    data = manifest.data_record
    if data.components != want.components:
        return (
            f"{manifest.run_id} was made with {_components(data.components)}, "
            f"not {_components(want.components)}"
        )
    if data.source != want.source:
        return f"{manifest.run_id} ran on {data.source}, not {want.source}"
    if data.effective_split != want.effective_split:
        return (
            f"{manifest.run_id} ran under the split {data.effective_split}, "
            f"not {want.effective_split}"
        )
    if data.ranges != want.ranges:
        return f"{manifest.run_id} ran on dates {data.ranges}, not {want.ranges}"
    if panel and data.panel != want.panel:
        return (
            f"{manifest.run_id} read the {data.source} panel {data.panel_id or '(none)'}, "
            f"the panel now is {want.panel_id or '(none)'}"
        )
    return ""


def foreign_run(
    run_dir: Path, config: RunConfig, source: SourceName, runner: Runner = run_episodes
) -> str:
    """Why ``run_dir`` must not be written for ``config`` on ``source``; "" when it may.

    It may not when it already holds this config's run on another source or split, or
    made with other components: the run id is the same, and writing over it would
    destroy a finished run of other data or another world. A run of the same data on a
    panel since rebuilt may be written over: it is the cell the resume recomputes.
    """
    try:
        manifest = RunManifest.read(run_dir)
    except (OSError, ValueError):
        return ""
    if manifest.run_id != run_id(config):
        return ""
    components = components_identity(runner)
    if components is None:
        why = source_mismatch(manifest, source)
    else:
        try:
            why = data_mismatch(manifest, expected_data(runner, config, source), panel=False)
        except DataUnavailableError:
            why = ""
    if not why:
        return ""
    return (
        f"{run_dir} already holds {why}; runs on another source, or made with other "
        "components, go under their own root (the default is <runs root>/<source>), so it "
        "is not overwritten"
    )


def valid_run(
    cell_dir: Path,
    config: RunConfig,
    source: SourceName = "synthetic",
    *,
    trainer: str | None = REAL_TRAINER,
    runner: Runner = run_episodes,
) -> RunRecord | None:
    """The run already on disk for ``config``, or None when it has to be (re)computed.

    The resume rule every orchestrator here shares (``run_sweep`` and
    ``optspread.orchestrate``): present, readable, keyed by the expected hash and
    run id, its data the record a run by ``runner`` on ``source`` writes now
    (``expected_data``, ``data_mismatch``: the source, on the real source its ``eval``
    split and the panel's digests, and the components, so a run made with components of
    one's own is reused only by the same components), and every file its manifest hashed
    (the results, and the policy of a learned run) still the one it hashed. A learned
    run must also have been trained by ``trainer`` (a ``callable_name``; the real
    learner unless told otherwise, None to accept any): a stand-in's cell carries the
    same run id as the real learner's and is never reused for it (#31). A learned run
    that records no trainer and whose policy is not a Stable-Baselines3 archive is
    reused by no one asking for ``REAL_TRAINER``. The commit a run was written at is not
    compared: the freeze keeps the numbers of every commit after the cutoff the same,
    and the planner reports the commits it reuses (``orchestrate.Plan``).
    """
    if not (cell_dir / RUN_FILENAME).exists() or not (cell_dir / MANIFEST_FILENAME).exists():
        return None
    try:
        record = read_run(cell_dir)
        manifest = RunManifest.read(cell_dir)
    except (OSError, ValueError):  # unreadable, wrong schema, incomplete, or malformed JSON
        return None
    expected = config_hash(config)
    if record.config_hash != expected or manifest.config_hash != expected:
        return None
    if record.run_id != run_id(config) or manifest.run_id != run_id(config):
        return None
    try:
        want = expected_data(runner, config, source)
    except DataUnavailableError:  # the panel it would be checked against is not here
        return None
    if data_mismatch(manifest, want):
        return None
    if changed_artifacts(cell_dir, manifest):
        return None  # a recorded file changed under its manifest
    if trainer is not None and config.train is not None:
        by = recorded_trainer(cell_dir, manifest)
        if by != trainer and (by is not None or trainer == REAL_TRAINER):
            return None
    return record


DATA_ENV = "VEGALAB_DATA"


def default_runs_root(env: Mapping[str, str] | None = None) -> Path:
    """Where the pilot, the sweep and the reports that read them meet by default.

    ``$VEGALAB_DATA/runs`` when ``VEGALAB_DATA`` is set, else ``runs`` under the
    current directory. ``scripts/pilot.py`` writes under ``<this>/pilot``,
    ``scripts/sweep.py`` here, and ``make m2`` and ``make m4`` read the same places,
    so a run started without ``--root`` lands where its report looks.
    """
    data = (os.environ if env is None else env).get(DATA_ENV, "")
    return Path(data).expanduser() / "runs" if data else Path("runs")


def source_root(root: Path, source: SourceName) -> Path:
    """Where runs on ``source`` live under a runs root: the root itself for the generator,
    ``<root>/<source>`` otherwise. The source is not in a run id, so a real-data run and
    its synthetic twin share one; their own folders keep them apart by default."""
    return root if source == "synthetic" else root / source


# The data a job can run on: the generator, or the real SPX panel (optspread.data).
JobSource = Literal["synthetic", "optionmetrics"]
SOURCES: tuple[JobSource, ...] = get_args(JobSource)


LOCK_FILENAME = ".optspread.lock"
LOCK_TIMEOUT = 5.0  # seconds a second orchestrator on the same root waits before it stops


class RootInUseError(TimeoutError):
    """Another orchestrator holds the results root's lock (``root_lock``)."""


@contextmanager
def root_lock(root: Path, timeout: float = LOCK_TIMEOUT) -> Iterator[None]:
    """Hold ``root``'s lock, ``<root>/.optspread.lock``, for the body.

    Raises ``RootInUseError``, naming the lock file, when another process still holds it
    after ``timeout`` seconds. Released when the body ends, however it ends (an exception
    or an interrupt included); a holder that is killed releases it with its process.
    Worker processes write under the lock their parent holds and never take it.
    """
    lock = FileLock(root / LOCK_FILENAME, timeout=timeout)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RootInUseError(
            f"could not lock {root / LOCK_FILENAME} within {timeout:g}s: another run is "
            f"computing cells under {root}. Let it finish, or use another root."
        ) from exc
    try:
        yield
    finally:
        lock.release()
