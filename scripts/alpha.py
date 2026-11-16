"""Synthetic alpha in one command: generate, train, score and report, for any arms.

Lane: Dominick Dupuy.

    python -m scripts.alpha arms --plugin examples/alpha_plugin.py:iv_timer
    python -m scripts.alpha run --root runs --generator const_vrp --set premium_pts=4 \\
        --arms ppo delta_seller iv_timer --plugin examples/alpha_plugin.py:iv_timer \\
        --timesteps 20480 --seeds 3
    python -m scripts.alpha run --root runs --arms delta_seller iv_picker \\
        --plugin examples/alpha_plugin.py:iv_picker --timesteps 4096 --seeds 3 \\
        --sweep edge --values 0 0.5 1 2 4

``run`` builds (or reuses) the frozen dataset the flags describe, benchmarks every arm on
it (``optspread.alpha.benchmark``: train per seed on the train paths and on the twin's,
score on held-out paths against the oracle and the twin), optionally runs one sweep per
arm (``--sweep``: the planted edge's size, the number of train paths or the training
budget), and prints what came out in plain words. Everything is written where the
Synthetic alpha page reads it, under ``<root>/alpha/``, so ``python -m optspread.server
--runs <root>`` shows it at ``/alpha/?dataset=<hash>``; a second run with the same flags
recomputes nothing.

Arms are the built-ins (``ppo``, ``a2c``, ``dqn`` need the ``train`` extra) and any plug-in
(``optspread.alpha.plugins``): ``--plugin`` registers one for this command, and the server
reads ``OPTSPREAD_ALPHA_PLUGINS``. A learned arm, or a plug-in with a trainer, takes
``--timesteps`` and ``--seeds``; a rule takes neither.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

from optspread.alpha import plugins as plug
from optspread.alpha.benchmark import (
    ARMS,
    MAX_SEEDS,
    Benchmark,
    BenchStat,
    SynthTrainer,
    arm_names,
    check_arm,
    edge_shown,
    is_learned,
    learned_trainer,
    run_benchmark,
)
from optspread.alpha.dataset import SynthSpec, build_dataset
from optspread.alpha.generators import GENERATORS, GeneratorKind
from optspread.alpha.sweep import AlphaSweep, AlphaSweepSpec, SweepAxis, check_edge, run_sweep
from optspread.config import LEARNED_ARMS
from optspread.dashboard.jobs.spec import ppo_available
from optspread.runs.rundir import default_runs_root

DEFAULT_TIMESTEPS = 20_480
DEFAULT_SEEDS = 3


def _value(text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from exc


def _setting(text: str) -> tuple[str, float]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"--set takes knob=value, got {text!r}")
    return key, _value(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    sub = parser.add_subparsers(dest="command", required=True)

    def plugin_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--plugin",
            action="append",
            default=[],
            metavar="PATH",
            help="register a plug-in arm, package.module:attr or path/to/file.py:attr "
            "(repeatable; added to those $OPTSPREAD_ALPHA_PLUGINS lists)",
        )

    arms = sub.add_parser("arms", help="list the arms a benchmark can run")
    plugin_flag(arms)

    run = sub.add_parser("run", help="generate, train, score and report")
    plugin_flag(run)
    run.add_argument(
        "--root",
        type=Path,
        default=None,
        help="runs root the page reads (default: $VEGALAB_DATA/runs, else runs)",
    )
    run.add_argument(
        "--generator", choices=get_args(GeneratorKind), default="const_vrp", help="the edge"
    )
    run.add_argument(
        "--set",
        type=_setting,
        action="append",
        default=[],
        metavar="KNOB=VALUE",
        help="a generator knob, e.g. premium_pts=4 (repeatable)",
    )
    run.add_argument("--sigma", type=float, default=0.20, help="path volatility, decimal")
    run.add_argument("--days", type=int, default=63, help="days per path")
    run.add_argument("--train", type=int, default=64, help="frozen train paths")
    run.add_argument("--heldout", type=int, default=256, help="frozen held-out paths")
    run.add_argument("--seed", type=int, default=0, help="first path seed")
    run.add_argument(
        "--arms", nargs="+", default=["delta_seller"], help="arms to benchmark, built-in or plug-in"
    )
    run.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"training steps per learner (default {DEFAULT_TIMESTEPS:,})",
    )
    run.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"training seeds per learner, 1 to {MAX_SEEDS} (default {DEFAULT_SEEDS})",
    )
    run.add_argument(
        "--sweep",
        choices=get_args(SweepAxis),
        default=None,
        help="also sweep one setting per arm: edge (the planted size; gives each arm's "
        "detection floor), n_train or total_timesteps",
    )
    run.add_argument("--values", nargs="+", type=_value, default=[], help="the sweep's values")
    return parser


def _usd(x: float | None) -> str:
    return "none" if x is None else f"${x:,.0f}" if x >= 0 else f"-${-x:,.0f}"


def _pct(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.0f}%"


def _pm(s: BenchStat, fmt: Any) -> str:
    if s.value is None:
        return "none"
    return fmt(s.value) if s.se is None else f"{fmt(s.value)} ± {fmt(2 * s.se)}"


def _setting_text(b: Benchmark) -> str:
    if b.total_timesteps is None:
        return "no training"
    return f"{b.total_timesteps:,} steps, {b.n_seeds} seed{'s' if b.n_seeds != 1 else ''}"


def bench_line(b: Benchmark) -> str:
    """One arm's result in a sentence: how much of the edge, then overfit and noise."""
    head = f"{b.arm} ({_setting_text(b)}):"
    if b.capture_ratio.value is not None:
        took = f"took {_pm(b.capture_ratio, _pct)} of the edge ({_pm(b.captured, _usd)} a path)"
    else:
        took = f"earned {_pm(b.captured, _usd)} a path from the signal"
    parts = [took, b.captured.verdict, b.overfit_gap.verdict]
    if b.noise_fit is not None:
        parts.append(b.noise_fit.verdict)
    return f"{head} {'; '.join(parts)}."


def sweep_lines(s: AlphaSweep) -> list[str]:
    spec = s.spec
    if s.floor is not None:
        f = s.floor
        what = f"{f.n} {f.unit}" if f.n else "no test"
        out = [f"{spec.arm} over the edge size ({s.edge_knob}; {what}): {f.reading}"]
        for p in s.points:
            if p.test is None:
                continue
            verdict = "detected" if p.test.detected else "not detected"
            t = "inf" if p.test.t is None and p.test.estimate else f"{p.test.t or 0.0:.2f}"
            out.append(
                f"  {p.value:g} {s.edge_unit}: captured {_usd(p.test.estimate)} "
                f"(se {_usd(p.test.se)}), t = {t} on {p.test.df} df: {verdict}; "
                f"power {_pct(p.test.power)}"
            )
        return out
    out = [f"{spec.arm} over {spec.axis}:"]
    for p in s.points:
        out.append(
            f"  {p.value:,}: capture ratio {_pm(p.capture_ratio, _pct)}, overfit gap "
            f"{_pm(p.overfit_gap, _usd)}"
            + (f", noise fit {_pm(p.noise_fit, _usd)}" if p.noise_fit is not None else "")
        )
    return out


def _cmd_arms() -> int:
    for a in ARMS:
        kind = "learned" if a in LEARNED_ARMS else "scripted"
        note = "" if a not in LEARNED_ARMS or ppo_available() else " (needs the train extra)"
        print(f"{a:<14} built-in, {kind}{note}")
    for p in plug.plugins().values():
        kind = "trainable" if p.plugin.trainable else "rule"
        print(f"{p.name:<14} plug-in, {kind}: {p.source}@{p.digest}")
    return 0


def _cmd_run(
    parser: argparse.ArgumentParser, args: argparse.Namespace, trainer: SynthTrainer
) -> int:
    root = Path(args.root) if args.root is not None else default_runs_root()
    for a in args.arms:
        try:
            check_arm(a)
        except ValueError as exc:
            parser.error(str(exc))
        if a in LEARNED_ARMS and trainer is learned_trainer and not ppo_available():
            parser.error(f"the {a} arm needs the train extra (pip install -e '.[train]')")
    if not 1 <= args.seeds <= MAX_SEEDS:
        parser.error(f"--seeds must be 1 to {MAX_SEEDS}")
    try:
        spec = SynthSpec.model_validate(
            {
                "generator": {"kind": args.generator, **dict(args.set)},
                "sigma": args.sigma,
                "n_days": args.days,
                "n_train": args.train,
                "n_heldout": args.heldout,
                "seed": args.seed,
            }
        )
    except ValueError as exc:
        parser.error(f"the dataset: {exc}")
    sweep: SweepAxis | None = args.sweep
    if sweep is not None:
        if len(args.values) < 2:
            parser.error("--sweep needs at least two --values")
        if sweep == "edge":
            try:
                check_edge(spec, args.values)
            except ValueError as exc:
                parser.error(str(exc))

    ds = build_dataset(spec, root)
    info = GENERATORS[spec.generator.kind]
    print(
        f"dataset {ds.hash}: {info.title}, {ds.n_paths('train')} train and "
        f"{ds.n_paths('heldout')} held-out paths of {spec.n_days} days, under {root}/alpha"
    )

    def say(line: str) -> None:
        print(f"  {line}", flush=True)

    results: list[Benchmark] = []
    for a in args.arms:
        learned = is_learned(a)
        print(f"{a}: benchmarking", flush=True)
        results.append(
            run_benchmark(
                ds,
                a,
                args.timesteps if learned else None,
                root,
                trainer,
                say,
                args.seeds if learned else 1,
            )
        )
    sweeps: list[AlphaSweep] = []
    for a in args.arms if sweep is not None else []:
        learned = is_learned(a)
        try:
            sspec = AlphaSweepSpec(
                dataset=ds.hash,
                arm=a,
                axis=sweep,
                values=tuple(args.values),
                total_timesteps=None
                if not learned or sweep == "total_timesteps"
                else args.timesteps,
                n_seeds=args.seeds if learned else 1,
            )
        except ValueError as exc:
            parser.error(f"the {a} sweep: {exc}")
        print(f"{a}: sweeping {sweep} over {', '.join(f'{v:g}' for v in args.values)}", flush=True)
        sweeps.append(run_sweep(sspec, root, trainer, say))

    first = results[0]
    print()
    print(f"Summary, dataset {ds.hash} ({info.title.lower()}):")
    if edge_shown(first):
        print(
            f"The oracle ({first.oracle}) shows a {_pm(first.attainable, _usd)} edge a path "
            "on held-out paths (signal minus twin)."
        )
    else:
        print(
            f"The oracle finds no attainable edge on this dataset ({first.oracle}: "
            f"{_pm(first.attainable, _usd)} a path, signal minus twin, not 2 se above zero)."
        )
    for b in results:
        print(f"- {bench_line(b)}")
    for s in sweeps:
        for line in sweep_lines(s):
            print(f"- {line}" if not line.startswith("  ") else line)
    if any(s.floor is not None for s in sweeps):
        floor = next(s.floor for s in sweeps if s.floor is not None)
        print(f"Note: {floor.note}")
    print(f"See it: python -m optspread.server --runs {root}, then /alpha/?dataset={ds.hash}")
    return 0


def main(argv: Sequence[str] | None = None, trainer: SynthTrainer = learned_trainer) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for p in args.plugin:
        try:
            plug.register(p)
        except plug.PluginError as exc:
            parser.error(str(exc))
    try:
        arm_names()
    except plug.PluginError as exc:
        parser.error(str(exc))
    if args.command == "arms":
        return _cmd_arms()
    return _cmd_run(parser, args, trainer)


if __name__ == "__main__":
    sys.exit(main())
