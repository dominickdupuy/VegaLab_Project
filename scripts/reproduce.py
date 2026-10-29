"""Reproduce our results from a clean clone: one command on macOS, Linux or Windows (#10).

Lane: Dominick Dupuy (statistical spine).

    git clone https://github.com/dominickdupuy/VegaLab_Project.git
    cd VegaLab_Project
    python scripts/reproduce.py            # or: make reproduce

Until the venv exists this file uses the standard library only. It creates ``.venv``
beside the clone if there is none, installs the project into it (``pip install -e
".[dev,train]"``, with ``PIP_USER=0`` so a machine that forces ``--user`` installs still
installs into the venv; on Linux the CPU build of torch first, as the README does), and
runs itself again inside it with ``--skip-install``. There it runs, in order:

1. the parts of ``make check``: ruff, ``mypy --strict``, pytest with the coverage floor;
2. ``make gate`` (the Wave 0 sanity table), ``make drag`` (the cost drag against its
   prediction) and ``make leaks`` (the leak suite and its controls);
3. the poisoned-environment demo the dashboard serves, in-process;
4. ``--cells N`` fixed-delta seller cells of the confirmatory sweep regenerated from
   scratch and compared with ``docs/SWEEP_EXPECTED.json``, byte for byte and within the
   tolerance ``optspread.reproduction`` states and justifies (about 40 seconds a cell);
   ``--with-ppo`` adds the reference PPO cell (one million training steps, about 20 minutes),
   retrained and compared statistically, not byte for byte: PPO training is not
   bit-reproducible across platforms (``optspread.reproduction`` says how and why). It is
   optional; run, it must pass too.

It writes the record (``docs/M3_PARTNER_REPRODUCTION.json`` and a ``.md`` beside it, or
``--out``), prints a one-screen summary, and exits 0 only when every step passed. The
record is what the M3 gate's partner item reads: send both files back.

``--skip-install`` runs everything with the interpreter you started, which must already
have the project's ``dev`` and ``train`` extras installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"
EXTRAS = ".[dev,train]"
MIN_PYTHON = (3, 12)
CPU_TORCH = ("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
DEFAULT_OUT = Path("docs") / "M3_PARTNER_REPRODUCTION.json"
DEFAULT_CELLS = 3  # reproduction.DEFAULT_CELLS, kept here so the bootstrap imports nothing

Run = Callable[..., "subprocess.CompletedProcess[Any]"]


def venv_python(venv: Path, platform: str = sys.platform) -> Path:
    """The venv's interpreter: ``Scripts/python.exe`` on Windows, ``bin/python`` elsewhere."""
    if platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def bootstrap_commands(
    python: str, venv: Path, *, exists: bool, platform: str = sys.platform
) -> list[list[str]]:
    """Create the venv (unless it exists) and install the project into it."""
    vpy = str(venv_python(venv, platform))
    commands = [] if exists else [[python, "-m", "venv", str(venv)]]
    if platform.startswith("linux"):  # the default torch wheel on Linux is the CUDA one, GBs
        commands.append([vpy, "-m", "pip", "install", *CPU_TORCH])
    commands.append([vpy, "-m", "pip", "install", "-e", EXTRAS])
    return commands


def pip_env(base: Mapping[str, str]) -> dict[str, str]:
    """``PIP_USER=0``: a machine configured to force ``--user`` would install outside the venv."""
    env = dict(base)
    env["PIP_USER"] = "0"
    return env


def bootstrap(
    argv: Sequence[str],
    *,
    run: Run = subprocess.run,
    python: str = sys.executable,
    platform: str = sys.platform,
    version: tuple[int, ...] = tuple(sys.version_info[:2]),
    venv: Path = VENV,
    env: Mapping[str, str] | None = None,
) -> int:
    """Make the venv, install, and run this script again inside it; its exit status."""
    if tuple(version) < MIN_PYTHON:
        want = ".".join(map(str, MIN_PYTHON))
        print(
            f"Python {want} or newer is needed, this is {'.'.join(map(str, version))}. "
            f"Run it with a {want} interpreter, e.g. 'py -{want} scripts/reproduce.py' on "
            f"Windows or 'python{want} scripts/reproduce.py' elsewhere.",
            file=sys.stderr,
        )
        return 2
    environment = pip_env(os.environ if env is None else env)
    exists = venv_python(venv, platform).exists()
    print(f"{'using' if exists else 'creating'} {venv} and installing {EXTRAS} into it", flush=True)
    for command in bootstrap_commands(python, venv, exists=exists, platform=platform):
        print("$ " + " ".join(command), flush=True)
        code = run(command, cwd=REPO, env=environment).returncode
        if code != 0:
            print(f"failed (exit {code}): {' '.join(command)}", file=sys.stderr)
            return int(code)
    again = [str(venv_python(venv, platform)), str(Path(__file__).resolve()), *argv]
    return int(run([*again, "--skip-install"], cwd=REPO, env=environment).returncode)


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument(
        "--skip-install", action="store_true", help="run with this interpreter, install nothing"
    )
    parser.add_argument(
        "--cells", type=int, default=DEFAULT_CELLS, help="seller cells to regenerate (1 to 30)"
    )
    parser.add_argument(
        "--with-ppo",
        action="store_true",
        help="also the reference PPO cell (~20 min), compared statistically: PPO training is "
        "not bit-reproducible across platforms",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the record (.json)")
    parser.add_argument("--team", default="", help="your team's name, for the record")
    args = parser.parse_args(argv)
    if args.cells < 1:
        parser.error(f"--cells must be at least 1, got {args.cells}")
    return args


def say(text: str) -> None:
    print(text, flush=True)


def reproduce(args: argparse.Namespace) -> int:
    """Every step, the record and the summary. Needs the project installed."""
    sys.path.insert(0, str(REPO))
    try:
        from optspread import reproduction as rp
        from optspread.m4analysis import CONFIRMATORY_GRID
    except ImportError as exc:
        print(
            f"cannot import the project ({exc}): run without --skip-install, or install it "
            f'with pip install -e "{EXTRAS}" first',
            file=sys.stderr,
        )
        return 2
    start = time.perf_counter()
    out = args.out if args.out.is_absolute() else REPO / args.out
    out_md = out.with_suffix(".md")
    ignore = [p.relative_to(REPO).as_posix() for p in (out, out_md) if p.is_relative_to(REPO)]
    git = rp.git_facts(REPO, rp.CUTOFF_TAG, ignore)
    expected_path = REPO / "docs" / rp.EXPECTED_FILE
    expected = rp.read_expected(expected_path)
    sellers = expected.sellers(args.cells)
    plan = [*rp.STEPS, *([rp.PPO_STEP] if args.with_ppo else [])]
    say(f"reproducing {git.commit[:12]} on {rp.platform_facts()['platform']}: {', '.join(plan)}")

    steps, compared = rp.run_steps(
        plan,
        repo=REPO,
        python=sys.executable,
        grid=CONFIRMATORY_GRID,
        sellers=sellers,
        ppo=expected.ppo_reference(),
        report=say,
    )
    record = rp.PartnerRecord(
        team=args.team,
        generated=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=git.commit,
        dirty=git.dirty,
        tag=rp.CUTOFF_TAG,
        tag_commit=git.tag_commit,
        after_tag=git.after_tag,
        platform=rp.platform_facts(),
        versions=rp.versions_now(),
        expected_file=f"docs/{rp.EXPECTED_FILE}",
        expected_sha256=rp.text_sha256(expected_path.read_bytes()),
        tolerance=rp.tolerances(),
        steps=steps,
        cells=compared,
        seconds=time.perf_counter() - start,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(record.model_dump_json(indent=2) + "\n", newline="\n")
    shown = out.relative_to(REPO).as_posix() if out.is_relative_to(REPO) else str(out)
    summary = rp.render_summary(record, shown)
    out_md.write_text(summary, newline="\n")
    say("")
    say(summary)
    if git.dirty:
        say("note: the checkout has uncommitted changes, so the record cannot pass")
    if not git.tag_commit:
        say(f"note: no {rp.CUTOFF_TAG} tag in this clone (git fetch --tags), so it cannot pass")
    elif not git.after_tag:
        say(f"note: this commit is not at or after {rp.CUTOFF_TAG}, so it cannot pass")
    return 0 if record.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parse(raw)
    if not args.skip_install:
        return bootstrap(raw)
    return reproduce(args)


if __name__ == "__main__":
    sys.exit(main())
