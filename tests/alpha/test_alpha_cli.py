"""``python -m scripts.alpha``: generate, train, score and report in one command."""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_learner import FakeLearned

from optspread.alpha import benchmark as bm
from optspread.alpha import plugins as plug
from optspread.alpha.dataset import list_datasets
from optspread.alpha.sweep import list_sweeps
from optspread.config import RunConfig
from optspread.types import ChainProvider
from scripts import alpha as cli

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "alpha_plugin.py"
TINY = ["--days", "20", "--train", "3", "--heldout", "10"]


@pytest.fixture(autouse=True)
def small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "FRESH_N", 16)
    monkeypatch.setenv(plug.PLUGINS_ENV, "")  # restored after register() writes it
    monkeypatch.delenv(plug.PLUGINS_FILE_ENV, raising=False)


def fake(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    assert config.train is not None
    return FakeLearned(config.train.seed)


@pytest.mark.slow
def test_one_command_runs_every_arm_and_an_edge_sweep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "run",
        "--root",
        str(tmp_path),
        "--set",
        "premium_pts=6",
        *TINY,
        "--arms",
        "ppo",
        "delta_seller",
        "iv_timer",
        "--plugin",
        f"{EXAMPLE}:iv_timer",
        "--timesteps",
        "2048",
        "--seeds",
        "2",
        "--sweep",
        "edge",
        "--values",
        "0",
        "6",
    ]
    assert cli.main(argv, trainer=fake) == 0
    out = capsys.readouterr().out
    (h,) = [m["hash"] for m in list_datasets(tmp_path) if m["spec"]["generator"]["premium_pts"]]
    assert out.startswith(f"dataset {h}: Constant variance premium")
    assert f"Summary, dataset {h}" in out
    assert "- ppo (2,048 steps, 2 seeds):" in out and "- iv_timer (no training):" in out
    assert "over the edge size (premium_pts; 2 seeds)" in out
    assert "over the edge size (premium_pts; 5 path blocks)" in out
    assert "the floor is rough" in out.lower() and f"/alpha/?dataset={h}" in out
    assert {s.spec.arm for s in list_sweeps(tmp_path, h)} == {"ppo", "delta_seller", "iv_timer"}
    assert len(bm.list_benchmarks(tmp_path, h)) == 3
    # a second run reads everything back
    assert cli.main(argv, trainer=fake) == 0


def test_a_train_size_sweep_and_the_arm_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["run", "--root", str(tmp_path), *TINY, "--arms", "flat", "--sweep", "n_train"]
    assert cli.main([*argv, "--values", "2", "3"]) == 0
    out = capsys.readouterr().out
    assert "flat over n_train:" in out and "  2: capture ratio" in out
    assert cli.main(["arms", "--plugin", f"{EXAMPLE}:iv_picker"]) == 0
    listing = capsys.readouterr().out
    assert "delta_seller" in listing and "iv_picker      plug-in, trainable" in listing


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["arms", "--plugin", "nowhere.at_all:x"], "cannot be imported"),
        (["run", "--arms", "mystery"], "unknown arm 'mystery'"),
        (["run", "--arms", "flat", "--set", "premium_pts=99"], "the dataset"),
        (["run", "--arms", "flat", "--sweep", "edge", "--values", "1"], "two --values"),
        (
            [
                "run",
                "--generator",
                "spot_mr",
                "--arms",
                "flat",
                "--sweep",
                "edge",
                "--values",
                "0",
                "1",
            ],
            "no edge size",
        ),
        (["run", "--arms", "flat", "--seeds", "0"], "--seeds"),
        (["run", "--set", "nonsense"], "knob=value"),
    ],
)
def test_bad_flags_are_refused_with_the_reason(
    argv: list[str], message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if argv[0] == "run":
        argv = [*argv, "--root", str(tmp_path)]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv, trainer=fake)
    assert exc.value.code == 2 and message in capsys.readouterr().err


def test_the_summary_follows_the_benchmarks_verdict_on_the_edge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = cli.run_benchmark

    def weak(*args: object, **kwargs: object) -> bm.Benchmark:
        b = real(*args, **kwargs)  # type: ignore[arg-type]
        # a nonzero attainable figure the z test does not call an edge
        stat = bm.BenchStat(value=150.0, se=100.0, verdict=bm.NO_EDGE)
        return b.model_copy(update={"attainable": stat})

    monkeypatch.setattr(cli, "run_benchmark", weak)
    argv = ["run", "--root", str(tmp_path), *TINY, "--arms", "delta_seller"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "The oracle finds no attainable edge" in out and "shows a" not in out


def test_an_edge_sweep_out_of_range_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["run", "--root", str(tmp_path), *TINY, "--sweep", "edge", "--values", "0", "30"]
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert "edge size 30 is out of range for premium_pts" in capsys.readouterr().err
    assert list_datasets(tmp_path) == []
