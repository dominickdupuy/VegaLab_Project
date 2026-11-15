"""The research example: ``examples/sweep.py`` runs each arm against a planted drift."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from optspread.manifest import RunManifest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "sweep.py"


def _example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("example_sweep", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.behaviour
def test_the_example_sweep_runs_every_cell_as_a_launched_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--arms", "flat", "delta_seller", "--alphas", "0", "25", "--episodes", "2"]
    assert _example().main([*argv, "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "25 bps/day" in out and out.count("delta_seller") >= 3
    runs = sorted((tmp_path / "launched").iterdir())
    assert len(runs) == 4
    drifts = {RunManifest.read(r).config.market.alpha_bps for r in runs}
    assert drifts == {0.0, 25.0}
    # again: every cell is on disk, so read back rather than recomputed
    assert _example().main([*argv, "--root", str(tmp_path)]) == 0
    assert len(list((tmp_path / "launched").iterdir())) == 4
