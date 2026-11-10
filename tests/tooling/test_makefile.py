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

REPO = Path(__file__).resolve().parent.parent.parent

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


@pytest.mark.behaviour
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


@pytest.mark.behaviour
def test_the_m2_target_passes_its_args_through(tmp_path: Path) -> None:
    line = _dry_run(tmp_path, "m2", "ARGS=--runs /tmp/r", "PY=py").strip()
    assert line == "py -m scripts.m2_gate_report --runs /tmp/r"


@pytest.mark.behaviour
def test_the_m4_target_passes_its_args_through(tmp_path: Path) -> None:
    line = _dry_run(tmp_path, "m4", "ARGS=--root /tmp/r", "PY=py").strip()
    assert line == "py -m scripts.m4_analysis --root /tmp/r"


@pytest.mark.behaviour
def test_the_m4gate_target_passes_its_args_through(tmp_path: Path) -> None:
    line = _dry_run(tmp_path, "m4gate", "ARGS=--runs-root /tmp/r", "PY=py").strip()
    assert line == "py -m scripts.m4_gate_report --runs-root /tmp/r"


@pytest.mark.behaviour
def test_the_script_targets_import_the_checkout_they_live_in(tmp_path: Path) -> None:
    # `python scripts/x.py` puts scripts/ first on the path, and optspread then comes from
    # whatever a shared venv's editable install points at; `-m scripts.x` imports this one.
    for target, module in (("gate", "wave0_gate"), ("drag", "cost_drag"), ("bench", "bench")):
        assert _dry_run(tmp_path, target, "PY=py").strip() == f"py -m scripts.{module}"


def _fake_python(where: Path, name: str, version: tuple[int, int]) -> None:
    script = where / name
    script.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "-c" ]; then exit {1 if version < (3, 12) else 0}; fi\n'
        f"echo {name} {version}\n"
    )
    script.chmod(0o755)


@pytest.mark.behaviour
@pytest.mark.skipif(os.name == "nt", reason="the POSIX bootstrap probe")
def test_make_reproduce_bootstraps_with_a_python_new_enough_for_the_project(
    tmp_path: Path,
) -> None:
    import re

    requires = re.search(r'requires-python = ">=(\d+\.\d+)"', (REPO / "pyproject.toml").read_text())
    assert requires is not None
    make_min = re.search(r"^MIN_PYTHON := (\S+)$", (REPO / "Makefile").read_text(), re.M)
    assert make_min is not None and make_min[1] == requires[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_python(bin_dir, "python3", (3, 9))  # macOS's system python3
    _fake_python(bin_dir, "python3.12", (3, 12))
    # The probe tries python3.14 and python3.13 before python3.12, and the machine may have
    # real ones in /usr/bin (the CI image now does). Shadow them with ones that fail the
    # probe, so the test decides what is on the path, not the machine.
    for newer in ("python3.14", "python3.13"):
        (bin_dir / newer).write_text("#!/bin/sh\nexit 1\n")
        (bin_dir / newer).chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin"}
    make = shutil.which("make")
    assert make is not None
    out = subprocess.run(
        [make, "-n", "-f", str(REPO / "Makefile"), "reproduce"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "python3.12 scripts/reproduce.py"
    _fake_python(bin_dir, "python3", (3, 13))
    out = subprocess.run(
        [make, "-n", "-f", str(REPO / "Makefile"), "reproduce"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "python3 scripts/reproduce.py"
