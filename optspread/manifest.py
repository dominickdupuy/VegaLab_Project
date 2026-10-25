"""Run manifests: the provenance record every reported number carries.

Lane: Ilan Danial (machinery).

A manifest records what produced a run (git SHA, config hash, library versions)
and what it produced (artifact hashes, headline results). The dashboard reads
these files directly; the reproducibility gate replays a run from its hash and
compares artifact digests.

``validate`` is the part that makes this a rule rather than an agreement. Every
field a replay needs is required, and ``write`` validates before it writes, so a
run nobody can reproduce cannot reach the results folder. A note in the team
contract would not have done that.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy
import pydantic
import scipy
from pydantic import BaseModel, Field

from optspread import __version__
from optspread.config import RunConfig, config_hash

MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1

# Everything a replay needs: without any one of these the run is not reproducible.
REQUIRED_FIELDS = ("run_id", "git_sha", "config_hash", "config", "versions")
REQUIRED_VERSIONS = ("python", "optspread", "numpy", "scipy", "pydantic")


def git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def git_dirty(cwd: Path | None = None) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.stdout.strip())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def library_versions() -> dict[str, str]:
    """Every library whose version can change a number, recorded per run.

    The learner is optional (the ``train`` extra), so torch and stable-baselines3
    are recorded when present and simply absent otherwise. They are not in
    REQUIRED_VERSIONS: a scripted-policy run is reproducible without them. Their
    presence says only that they were installed, not that the run trained anything;
    the config (``policy == "ppo"`` with ``train`` set) is what marks a training run.
    """
    versions = {
        "python": platform.python_version(),
        "optspread": __version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pydantic": pydantic.__version__,
    }
    for name in ("torch", "stable_baselines3"):
        try:
            versions[name] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


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

    @classmethod
    def create(cls, config: RunConfig, *, repo: Path | None = None) -> RunManifest:
        h = config_hash(config)
        return cls(
            run_id=f"{h}-s{config.seed}",
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=git_sha(repo),
            git_dirty=git_dirty(repo),
            config_hash=h,
            config=config,
            versions=library_versions(),
        )

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
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True), newline="\n"
        )
        return target

    @classmethod
    def read(cls, run_dir: Path) -> RunManifest:
        manifest = cls.model_validate_json((run_dir / MANIFEST_FILENAME).read_text())
        manifest.validate_complete()
        return manifest
