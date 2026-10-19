"""Ilan Danial: the Makefile finds the venv on every platform and passes ARGS through.

``make -n`` prints the commands without running them, so these read what a target
would do. A Windows venv has its interpreter under ``Scripts/``, every other one
under ``bin/``; the default ``PY`` has to pick whichever exists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="needs make")


def _dry_run(cwd: Path, *args: str) -> str:
    # Under `make check PY=...` the outer make passes PY down through MAKEFLAGS; the
    # default is only visible to a make that did not inherit it.
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
    env.pop("PY", None)
    out = subprocess.run(
        ["make", "-n", "-f", str(REPO / "Makefile"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


@pytest.mark.parametrize("scripts", [False, True])
def test_the_default_interpreter_is_the_venv_this_platform_made(
    tmp_path: Path, scripts: bool
) -> None:
    venv = tmp_path / ".venv" / ("Scripts" if scripts else "bin")
    venv.mkdir(parents=True)
    (venv / ("python.exe" if scripts else "python")).write_text("")
    want = ".venv/Scripts/python" if scripts else ".venv/bin/python"
    assert _dry_run(tmp_path, "gate").split()[0] == want
    assert _dry_run(tmp_path, "gate", "PY=py").split()[0] == "py"  # an explicit PY wins


def test_the_m2_target_passes_its_args_through(tmp_path: Path) -> None:
    line = _dry_run(tmp_path, "m2", "ARGS=--runs /tmp/r", "PY=py").strip()
    assert line == "py -m scripts.m2_gate_report --runs /tmp/r"
