"""The showcase plan: every question's runs share their paths and are valid job specs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from optspread.dashboard.jobs.spec import JobSpec
from scripts import showcase


@pytest.mark.behaviour
def test_the_plan_covers_every_question_on_paired_paths() -> None:
    runs = showcase.plan()
    groups = Counter(g for g, _, _, _ in runs)
    assert len(runs) == 76
    assert set(groups) == {
        "strikes",
        "wings",
        "tenor",
        "roll",
        "costs",
        "account",
        "regime",
        "real",
        "arms",
        "sample",
        "training",
    }
    for _, question, label, kw in runs:
        JobSpec(**kw)
        assert question and label and kw["seed"] == showcase.SEED
    # PPO runs go last, the slow ones.
    kinds = [kw.get("policy") == "ppo" for _, _, _, kw in runs]
    assert kinds == sorted(kinds)
    # The catalog's labels are unique, so a pair is named, never a run key.
    assert len({label for _, _, label, _ in runs}) == len(runs)


@pytest.mark.behaviour
def test_dry_run_prints_the_plan_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert showcase.main(["--root", str(tmp_path / "show"), "--dry-run", "--skip-ppo"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("68 runs") and "[strikes]" in out
    assert not (tmp_path / "show").exists()


@pytest.mark.behaviour
def test_a_run_writes_its_catalog_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(job_id: str, spec_json: str, root: str) -> str:
        calls.append(json.loads(spec_json)["policy"])
        return f"launched/fake-{len(calls)}"

    monkeypatch.setattr(showcase, "execute_job", fake)
    first_two = showcase.plan()[:2]
    monkeypatch.setattr(showcase, "plan", lambda: first_two)
    assert showcase.main(["--root", str(tmp_path)]) == 0
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog["seed"] == showcase.SEED
    assert [r["key"] for r in catalog["runs"]] == ["launched/fake-1", "launched/fake-2"]
    assert catalog["runs"][0]["group"] == "strikes" and calls == ["delta_seller"] * 2
