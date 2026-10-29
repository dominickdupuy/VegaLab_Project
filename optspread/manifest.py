"""Run manifests: the provenance record every reported number carries.

Lane: Ilan Danial (machinery).

A manifest records what produced a run (git SHA, config hash, library versions,
the data it read) and what it produced (artifact hashes, headline results). The
dashboard reads these files directly; the reproducibility gate replays a run from
its hash and compares artifact digests.

``validate`` is the part that makes this a rule rather than an agreement. Every
field a replay needs is required, and ``write`` validates before it writes, so a
run nobody can reproduce cannot reach the results folder. A note in the team
contract would not have done that.

The data source is recorded here (``DataRecord``), not in ``RunConfig``: a field
there would re-key every config hash and every run already on disk. It is
optional on read, because manifests written before it was recorded do not carry
it, and every one of those ran on the generator (``harness.build_env`` built
nothing else until the source became a parameter).

A learned run also records which trainer produced its policy (``trainer``, the
function's ``module.qualname``), because a run trained by a test's stand-in carries
the same config, and so the same run id, as one trained by the real learner (#31).
It is optional for the same reason, and absent from a scripted run's manifest.

A run made with components of one's own (``optspread.extend.Components``, a planted
effect, a cost model or a runner of one's own) records what made it
(``DataRecord.components``, ``components_identity`` of the runner): its config hash is the
default world's at the same settings, so without it such a run would be taken for the
default one, and the reverse. The default components record nothing, so a default run's
manifest is the shape it always was.

A manifest also records the machine it was written on (``platform``: OS, architecture,
Python, numpy), because a replay on another platform may differ in the last bit of libm
and the reason should say so. Optional on read, absent from older manifests.

A run on the real source also records the split its episodes came from
(``DataRecord.split``, ``eval``; a learned arm trained on ``train``), since the split is
not in the config either. Those are the dates of ``optspread.realdata.SPLITS``
(pre-registration A8); a source over a surface of your own may have cut them on its own
``SplitRanges``, and then the record carries them (``DataRecord.ranges``), because two sets
of dates are two different runs. Both are absent for the generator, and a real-data
manifest written before the split was recorded ran on ``eval`` under ``SPLITS``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import numpy
import pydantic
import scipy
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from optspread import __version__
from optspread.config import RunConfig, config_hash
from optspread.data import (
    PANEL_FILES,
    ChainSource,
    DataUnavailableError,
    OptionMetricsSource,
    SourceName,
)
from optspread.realdata import Split, SplitRanges

MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1

# Everything a replay needs: without any one of these the run is not reproducible.
REQUIRED_FIELDS = ("run_id", "git_sha", "config_hash", "config", "versions")
REQUIRED_VERSIONS = ("python", "optspread", "numpy", "scipy", "pydantic")
# Recorded when installed, never required, so a manifest written before one was recorded
# still validates and replays: the environment's spaces and seeding (gymnasium), the real
# panel's reader (pandas, and pyarrow under it), and the learner (#31).
OPTIONAL_VERSIONS = ("gymnasium", "pandas", "pyarrow", "torch", "stable_baselines3")


# The checkout this code was imported from. Git is asked about it rather than about
# the process's working directory: a run records the commit of the code that made it,
# from wherever it was started, and a test or worker that changes directory cannot make
# that commit "unknown" (or another checkout's).
CODE_ROOT = Path(__file__).resolve().parent.parent


def git_sha(cwd: Path | None = None) -> str:
    """HEAD of the checkout at ``cwd``, by default the one this code was imported from."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd if cwd is not None else CODE_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def git_dirty(cwd: Path | None = None) -> bool:
    """Uncommitted changes in that same checkout; True when git cannot say."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd if cwd is not None else CODE_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.stdout.strip())


def callable_name(fn: object) -> str:
    """``module.qualname`` of a function: how a manifest names the trainer of a learned run.

    A function of a script (``__main__``) or one defined inside another (``<locals>``)
    is not told apart by that name: two scripts' ``__main__.trainer`` are two trainers.
    Its source file is added (``module.qualname@file``, the path relative to this
    checkout when it is inside it), so the two do not collide. A function of an
    importable module keeps its plain name, ``optspread.train.train_ppo`` included.
    """
    module = getattr(fn, "__module__", None) or type(fn).__module__
    name = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    if module != "__main__" and "<locals>" not in name:
        return f"{module}.{name}"
    target = fn if hasattr(fn, "__code__") else type(fn)
    try:
        source = Path(inspect.getfile(target))  # type: ignore[arg-type]
    except (TypeError, OSError):
        return f"{module}.{name}"
    try:
        shown = source.resolve().relative_to(CODE_ROOT).as_posix()
    except ValueError:
        shown = source.as_posix()
    return f"{module}.{name}@{shown}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Panel files are large and every reuse check of a real-data cell hashes them again
# (``sweep.valid_run``), so a digest is kept per file for as long as the file is the
# same one: same inode, size and modification time. A rebuilt panel changes at least one.
_PANEL_DIGESTS: dict[tuple[str, int, int, int], str] = {}


def panel_file_sha256(path: Path) -> str:
    """``file_sha256`` of a panel file, computed once per version of the file."""
    st = path.stat()
    key = (str(path.resolve()), st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _PANEL_DIGESTS:
        _PANEL_DIGESTS[key] = file_sha256(path)
    return _PANEL_DIGESTS[key]


def platform_record() -> dict[str, str]:
    """The machine a run is written on: what a replay elsewhere names when it differs."""
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
    }


def describe_platform(record: dict[str, str] | None) -> str:
    if not record:
        return "an unrecorded platform"
    return (
        f"{record.get('os', '?')} {record.get('arch', '?')} (python {record.get('python', '?')}, "
        f"numpy {record.get('numpy', '?')})"
    )


def library_versions() -> dict[str, str]:
    """Every library whose version can change a number, recorded per run.

    Beyond the required five, ``OPTIONAL_VERSIONS`` are recorded when present and
    simply absent otherwise. gymnasium is a core dependency, but manifests written
    before it was recorded do not carry it, so it is optional like the rest. pandas
    and pyarrow read the real panel (the ``data`` extra); a run on the generator never
    imports them, but a replay on the panel does. The learner is the ``train`` extra:
    torch and stable-baselines3 are not needed to reproduce a scripted-policy run.
    The presence of any of them says only that it was installed, not that the run used
    it; the config (``policy == "ppo"`` with ``train`` set) is what marks a training
    run. The replay reports a recorded version that has moved, and is silent about a
    library a manifest does not record.
    """
    versions = {
        "python": platform.python_version(),
        "optspread": __version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pydantic": pydantic.__version__,
    }
    for name in OPTIONAL_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


class DataRecord(BaseModel):
    """Where a run's chains came from: the source, and for real data the panel it read.

    ``panel`` maps every panel file the real source reads (``optspread.data.PANEL_FILES``)
    to its sha256, keyed as ``docs/DATA_MANIFEST.json`` keys it, so a run can be checked
    against the build it claims and a replay can refuse a panel rebuilt since. It is
    empty for the generator and the stub, which read no files.

    ``split`` is the split of ``optspread.realdata.SPLITS`` a run on the real source drew
    its episodes from: ``eval`` for every run the harness writes, a learned arm having
    been trained on ``train`` of the same panel (``optspread.train``). It is None for the
    generator and the stub, which keep training and evaluation apart by seed, and in a
    real-data record written before it was recorded (``effective_split`` reads that as
    ``eval``). ``ranges`` is set only when the source cut its splits on dates of its own
    (``SplitRanges``, a surface of your own); None means A8's ``SPLITS``. A record without
    either is written without the key, so a synthetic run's manifest is the same shape
    it always was.
    """

    model_config = ConfigDict(frozen=True)

    source: SourceName = "synthetic"
    panel: dict[str, str] = Field(default_factory=dict)  # "panel/<file>" -> sha256
    split: Split | None = None
    ranges: SplitRanges | None = None
    components: str | None = None

    @classmethod
    def of(cls, source: ChainSource, components: str | None = None) -> DataRecord:
        """The record for ``source`` as it stands now; the panel files are hashed on the spot.

        A source outside ``optspread.data``'s names (one of your own) is recorded as
        ``synthetic`` under the ``components`` that built it."""
        if not isinstance(source, OptionMetricsSource):
            name = source.name if source.name in get_args(SourceName) else "synthetic"
            return cls(source=name, components=components)
        if not source.available:
            raise DataUnavailableError(f"no SPX panel under {source.panel_dir} to record")
        return cls(
            source=source.name,
            panel={
                f"panel/{name}": panel_file_sha256(source.panel_dir / name) for name in PANEL_FILES
            },
            split=source.split,
            ranges=source.ranges,
            components=components,
        )

    @property
    def effective_split(self) -> Split | None:
        """The split the run's episodes came from: the recorded one, ``eval`` for a
        real-data record from before it was recorded, None for a source with no dates."""
        if self.source != "optionmetrics":
            return None
        return self.split if self.split is not None else "eval"

    @model_serializer(mode="wrap")
    def _omit_absent_split(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        for key in ("split", "ranges", "components"):
            if getattr(self, key) is None:
                out.pop(key, None)
        return out

    @property
    def panel_id(self) -> str:
        """First 16 hex chars of the SHA-256 over the panel's file digests; empty with no panel."""
        if not self.panel:
            return ""
        text = "".join(f"{name}:{digest}\n" for name, digest in sorted(self.panel.items()))
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class IncompleteManifestError(ValueError):
    """A manifest missing something a replay needs. Never written, never read past."""


class RunManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    config: RunConfig
    versions: dict[str, str]
    artifacts: dict[str, str] = Field(default_factory=dict)  # relative path -> sha256
    results: dict[str, float] = Field(default_factory=dict)
    data: DataRecord | None = None  # None only in a manifest written before it was recorded
    # The trainer of a learned run's policy (``callable_name``); None for a scripted run
    # and in a manifest written before it was recorded.
    trainer: str | None = None
    # The machine it was written on (``platform_record``); None in an older manifest.
    platform: dict[str, str] | None = None

    @classmethod
    def create(
        cls,
        config: RunConfig,
        *,
        repo: Path | None = None,
        data: DataRecord | None = None,
        trainer: str | None = None,
    ) -> RunManifest:
        """A manifest for ``config``. ``data`` defaults to the generator, as ``build_env`` does."""
        h = config_hash(config)
        return cls(
            run_id=f"{h}-s{config.seed}",
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=git_sha(repo),
            git_dirty=git_dirty(repo),
            config_hash=h,
            config=config,
            versions=library_versions(),
            data=data if data is not None else DataRecord(),
            trainer=trainer,
            platform=platform_record(),
        )

    @property
    def data_record(self) -> DataRecord:
        """The recorded source, or the generator for a manifest from before it was recorded."""
        return self.data if self.data is not None else DataRecord()

    @model_serializer(mode="wrap")
    def _omit_absent_data(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # An old manifest read and written again keeps the keys it was written with, and a
        # scripted run's manifest has no trainer key at all.
        out: dict[str, Any] = handler(self)
        if self.data is None:
            out.pop("data", None)
        if self.trainer is None:
            out.pop("trainer", None)
        if self.platform is None:
            out.pop("platform", None)
        return out

    def validate_complete(self) -> None:
        """Raise unless every field a replay needs is present and self-consistent."""
        missing = [f for f in REQUIRED_FIELDS if not getattr(self, f, None)]
        if missing:
            raise IncompleteManifestError(f"manifest is missing {', '.join(missing)}")
        if self.schema_version != SCHEMA_VERSION:
            raise IncompleteManifestError(
                f"manifest is schema v{self.schema_version}, this build writes v{SCHEMA_VERSION}"
            )
        absent = [k for k in REQUIRED_VERSIONS if not self.versions.get(k)]
        if absent:
            raise IncompleteManifestError(f"versions is missing {', '.join(absent)}")
        if self.config_hash != config_hash(self.config):
            raise IncompleteManifestError(
                f"config_hash {self.config_hash} does not match the recorded config "
                f"({config_hash(self.config)}): the two have drifted apart"
            )
        if self.git_sha == "unknown":
            raise IncompleteManifestError("git_sha is unknown, so the code is not identified")

    def add_artifact(self, run_dir: Path, path: Path) -> None:
        self.artifacts[str(path.relative_to(run_dir).as_posix())] = file_sha256(path)

    def write(self, run_dir: Path) -> Path:
        self.validate_complete()
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / MANIFEST_FILENAME
        target.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        return target

    @classmethod
    def read(cls, run_dir: Path) -> RunManifest:
        text = (run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        manifest = cls.model_validate_json(text)
        manifest.validate_complete()
        return manifest
