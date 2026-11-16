"""Plug-in arms: registered by import path, refused clearly, benchmarked like the built-ins."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from optspread.alpha import benchmark as bm
from optspread.alpha import plugins as plug
from optspread.alpha.dataset import SynthSpec, build_dataset
from optspread.alpha.generators import ConstVrpParams
from optspread.alpha.jobs import BenchSpec
from optspread.alpha.sweep import AlphaSweepSpec, load_sweep, run_sweep, sweep_path
from optspread.policies import FlatPolicy

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "alpha_plugin.py"
TIMER, PICKER = f"{EXAMPLE}:iv_timer", f"{EXAMPLE}:iv_picker"
SMALL = {"n_days": 20, "n_train": 4, "n_heldout": 10, "seed": 3}

# Plug-ins the tests import by dotted path: pytest puts this file's folder on sys.path.
flat_rule = plug.AlphaPlugin(name="flat_rule", label="Flat rule", policy=FlatPolicy)
not_a_plugin = FlatPolicy


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # set, not deleted, so the teardown restores it even after register() writes it
    monkeypatch.setenv(plug.PLUGINS_ENV, "")
    monkeypatch.delenv(plug.PLUGINS_FILE_ENV, raising=False)
    monkeypatch.setattr(bm, "FRESH_N", 16)
    yield


def test_the_example_file_registers_both_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(plug.PLUGINS_ENV, f"{TIMER}, {PICKER}")
    found = plug.plugins()
    assert list(found) == ["iv_timer", "iv_picker"]
    assert not found["iv_timer"].plugin.trainable and found["iv_picker"].plugin.trainable
    assert found["iv_timer"].digest == found["iv_picker"].digest  # one file, one digest
    assert bm.arm_names()[-2:] == ("iv_timer", "iv_picker")
    assert bm.is_learned("iv_picker") and not bm.is_learned("iv_timer")
    assert bm.arm_key("iv_timer") == f"iv_timer.{found['iv_timer'].digest}"
    assert bm.arm_key("ppo") == "ppo" and plug.find("ppo") is None


def test_a_dotted_path_and_a_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listed = tmp_path / "plugins.json"
    listed.write_text(json.dumps({"plugins": ["test_alpha_plugins:flat_rule", TIMER]}))
    monkeypatch.setenv(plug.PLUGINS_FILE_ENV, str(listed))
    assert set(plug.plugins()) == {"flat_rule", "iv_timer"}
    assert plug.find("flat_rule") is not None
    assert plug.plugins()["flat_rule"].plugin.title == "Flat rule"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("no_colon_here", "not an import path"),
        ("optspread.alpha.nothing_here:x", "cannot be imported"),
        ("optspread.alpha.plugins:nothing", "has no attribute 'nothing'"),
        ("test_alpha_plugins:not_a_plugin", "not an AlphaPlugin"),
        ("/no/such/file.py:x", "does not exist"),
    ],
)
def test_a_bad_import_path_says_what_is_wrong(path: str, message: str) -> None:
    with pytest.raises(plug.PluginError, match=message):
        plug.load_plugin(path)


def test_a_broken_file_and_a_bad_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(plug.PluginError, match=r"failed to import.*boom"):
        plug.load_plugin(f"{broken}:x")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    monkeypatch.setenv(plug.PLUGINS_FILE_ENV, str(bad))
    with pytest.raises(plug.PluginError, match="JSON file"):
        plug.configured_paths()
    bad.write_text(json.dumps({"plugins": [1]}))
    with pytest.raises(plug.PluginError, match="list of strings"):
        plug.configured_paths()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"name": "Bad Name", "policy": FlatPolicy}, "lower case"),
        ({"name": "ppo", "policy": FlatPolicy}, "built-in"),
        ({"name": "neither"}, "exactly one"),
    ],
)
def test_a_plugin_must_be_well_formed(fields: dict[str, object], message: str) -> None:
    with pytest.raises(plug.PluginError, match=message):
        plug.AlphaPlugin(**fields)  # type: ignore[arg-type]


def test_two_plugins_cannot_share_a_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other.py"
    other.write_text(
        "from optspread.alpha.plugins import AlphaPlugin\n"
        "from optspread.policies import FlatPolicy\n"
        "iv_timer = AlphaPlugin(name='iv_timer', policy=FlatPolicy)\n"
    )
    monkeypatch.setenv(plug.PLUGINS_ENV, TIMER)
    with pytest.raises(plug.PluginError, match="two plug-ins are named"):
        plug.register(f"{other}:iv_timer")
    assert os.environ[plug.PLUGINS_ENV] == TIMER  # the refused one was not left listed


def test_an_unknown_arm_names_what_there_is() -> None:
    with pytest.raises(ValueError, match=r"unknown arm 'mystery'.*plug-ins: none"):
        bm.check_arm("mystery")
    with pytest.raises(ValueError, match="unknown arm"):
        BenchSpec(dataset="0123456789abcdef", arm="mystery", total_timesteps=None)


def test_a_plugin_is_benchmarked_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plug.register(TIMER)
    plug.register(PICKER)
    ds = build_dataset(SynthSpec(generator=ConstVrpParams(premium_pts=6.0), **SMALL), tmp_path)
    rule = bm.run_benchmark(ds, "iv_timer", None, tmp_path)
    assert rule.arm == "iv_timer" and rule.noise_control is False
    assert rule.plugin is not None and rule.plugin.startswith(TIMER + "@")
    # implied vol never moves on const_vrp, so the timer never sells: nothing captured
    assert rule.captured.value == pytest.approx(0.0) and len(rule.blocks) == bm.BLOCKS
    learner = bm.run_benchmark(ds, "iv_picker", 2048, tmp_path, n_seeds=2)
    assert learner.noise_control is True and len(learner.seeds) == 2
    assert {(r.policy, r.world, r.split) for r in learner.rows} >= {
        ("arm", "signal", "heldout"),
        ("noise_control", "twin", "train"),
    }
    path = bm.bench_path(tmp_path, ds.hash, "iv_picker", 2048, 2)
    assert path.is_file() and "iv_picker." in path.parent.name
    assert bm.list_benchmarks(tmp_path, ds.hash)[0].arm in {"iv_timer", "iv_picker"}
    # and it sweeps like any arm, with its floor
    spec = AlphaSweepSpec(dataset=ds.hash, arm="iv_timer", axis="edge", values=(0, 3, 6))
    sweep = run_sweep(spec, tmp_path)
    assert sweep.floor is not None and sweep.floor.status == "not_reached"
    where = sweep_path(tmp_path, spec)
    assert where.parent.name.endswith(plug.find("iv_timer").digest)  # type: ignore[union-attr]
    # read back after the plug-in is gone, the sweep still loads
    monkeypatch.delenv(plug.PLUGINS_ENV)
    assert load_sweep(where) == sweep


def test_a_plugin_bench_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(plug.PLUGINS_ENV, f"{TIMER},{PICKER}")
    BenchSpec(dataset="0123456789abcdef", arm="iv_timer", total_timesteps=None)
    with pytest.raises(ValueError, match="deterministic"):
        BenchSpec(dataset="0123456789abcdef", arm="iv_timer", total_timesteps=None, n_seeds=2)
    spec = BenchSpec(dataset="0123456789abcdef", arm="iv_picker", total_timesteps=4096, n_seeds=3)
    assert spec.stages == 2 * 3 + 4
