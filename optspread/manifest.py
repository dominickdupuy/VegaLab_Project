"""Run manifests: the provenance record every reported number carries.

Lane: Ilan Danial (machinery).

A manifest records what produced a run (git SHA, config hash, library versions)
and what it produced (artifact hashes, headline results). The dashboard reads
these files directly; the reproducibility gate replays a run from its hash and
compares artifact digests.
"""

from __future__ import annotations

import hashlib
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
    return {
        "python": platform.python_version(),
        "optspread": __version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pydantic": pydantic.__version__,
    }


class RunManifest(BaseModel):
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

    def add_artifact(self, run_dir: Path, path: Path) -> None:
        self.artifacts[str(path.relative_to(run_dir).as_posix())] = file_sha256(path)

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / MANIFEST_FILENAME
        target.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True))
        return target

    @classmethod
    def read(cls, run_dir: Path) -> RunManifest:
        return cls.model_validate_json((run_dir / MANIFEST_FILENAME).read_text())
