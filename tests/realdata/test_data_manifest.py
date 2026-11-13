"""Hemdutt Rao: the data manifest records the raw inputs beside what the build wrote.

``docs/DATA_MANIFEST.json`` used to hash only the build's outputs, so a panel that
differed could not say whether the code or the ``opprcd``/``secprd`` inputs had
changed. The manifest stage runs here on a small tree under ``VEGALAB_DATA``; the
files are bytes, not real Parquet, because the stage only sizes and hashes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_spx_panel

TREE = {
    "databento/inventory.csv": b"tar,job_id\n",
    "databento/spx_close/year=2025/2025-07-01.parquet": b"close",
    "panel/spx/year=2010/part.parquet": b"panel 2010",
    "panel/spx_daily.parquet": b"daily",
    "panel/_summaries/v3/year=2010/term.parquet": b"cache",  # not a build output
    "opprcd/year=2010/SPX.parquet": b"opprcd 2010" * 100,
    "opprcd/year=2011/SPX.parquet": b"opprcd 2011",
    "opprcd/year=2012/SPX.empty": b"",  # a year with no rows: nothing to hash
    "secprd/year=2010/SPX.parquet": b"secprd 2010",
}
INPUTS = (
    "opprcd/year=2010/SPX.parquet",
    "opprcd/year=2011/SPX.parquet",
    "secprd/year=2010/SPX.parquet",
)


@pytest.fixture()
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "shared"
    for rel, body in TREE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(body)
    monkeypatch.setenv("VEGALAB_DATA", str(root))
    return root


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


@pytest.mark.invariant
def test_the_raw_inputs_are_sized_always_and_hashed_on_request(
    tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = tmp_path / "DATA_MANIFEST.json"
    build_spx_panel.data_manifest(copy)
    blob = json.loads(copy.read_text())
    assert blob["panel_version"] == build_spx_panel.PANEL_VERSION
    assert sorted(blob["inputs"]) == sorted(INPUTS)
    assert all(e["sha256"] is None for e in blob["inputs"].values())
    assert blob["inputs"][INPUTS[0]]["bytes"] == len(TREE[INPUTS[0]])
    assert blob["inputs_hashed"] is False
    assert "--hash-inputs" in capsys.readouterr().out, "says how to get the hashes"

    build_spx_panel.data_manifest(copy, hash_inputs=True)
    blob = json.loads(copy.read_text())
    assert blob["inputs_hashed"] is True
    for rel in INPUTS:
        assert blob["inputs"][rel] == {"sha256": _sha(TREE[rel]), "bytes": len(TREE[rel])}
    # The build's outputs are as before, and the shared copy is the repo copy.
    assert sorted(blob["files"]) == [
        "databento/inventory.csv",
        "databento/spx_close/year=2025/2025-07-01.parquet",
        "panel/spx/year=2010/part.parquet",
        "panel/spx_daily.parquet",
    ]
    assert blob["files"]["panel/spx_daily.parquet"]["sha256"] == _sha(b"daily")
    assert (tree / "MANIFEST.json").read_text() == copy.read_text()


@pytest.mark.behaviour
def test_the_manifest_stage_takes_the_flag(tree: Path, tmp_path: Path) -> None:
    copy = tmp_path / "m.json"
    args = ["manifest", "--repo-copy", str(copy)]
    assert build_spx_panel.main(args) == 0
    assert json.loads(copy.read_text())["inputs_hashed"] is False
    assert build_spx_panel.main([*args, "--hash-inputs"]) == 0
    entry = json.loads(copy.read_text())["inputs"]["secprd/year=2010/SPX.parquet"]
    assert entry["sha256"] == _sha(b"secprd 2010")
