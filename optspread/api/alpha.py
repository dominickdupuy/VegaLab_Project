"""API payloads for the Synthetic alpha page: plant an edge, freeze it, score an agent on it.

Lane: Dominick Dupuy.

The page builds a frozen dataset with a known, planted edge (``optspread.alpha``), trains
an arm on exactly its train paths, and reads back a benchmark: what the oracle could take,
what the arm took, and what it made on the null twin and on noise. Every model here is
what one route returns, so the site's types (``web/lib/schema.d.ts``) are generated from
them.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import Field

from optspread.alpha import plugins as plug
from optspread.alpha.benchmark import ARMS, BENCH_FILE, MAX_SEEDS, Benchmark, load_benchmark
from optspread.alpha.dataset import DATASETS_DIR, PATHS_FILE, SynthDataset, SynthSpec, preview
from optspread.alpha.generators import GENERATORS, GeneratorParams
from optspread.alpha.jobs import DEFAULT_SEEDS, DEFAULT_TIMESTEPS
from optspread.alpha.sweep import EDGE_KNOB, AlphaSweep, load_sweep
from optspread.api._base import Out
from optspread.api.jobs import JobOut, job_out
from optspread.config import LEARNED_ARMS
from optspread.dashboard.jobs.queue import JobStatus
from optspread.dashboard.jobs.spec import ppo_available

# What the form calls each field, with its unit, and the one line it shows on focus.
_FIELDS: dict[str, tuple[str, str]] = {
    "premium_pts": ("Premium (vol pts)", "Vol points the chain is priced over the path's vol."),
    "mean_pts": ("Mean premium (vol pts)", "Long-run premium; 0 is rich and cheap equally often."),
    "vol_pts": ("Premium s.d. (vol pts)", "Stationary standard deviation of the premium."),
    "half_life_days": ("Half-life (days)", "Days for a deviation to halve."),
    "bps_per_day": ("Drift (bps per day)", "Drift of the underlying, basis points a day."),
    "sigma": ("Path volatility", "Annualised vol driving the path, decimal."),
    "n_days": ("Days per path", "Path and episode length in trading days."),
    "n_train": ("Train paths", "Frozen paths the learner trains on, cycled repeatedly."),
    "n_heldout": ("Held-out paths", "Frozen paths it is scored on and never trains on."),
    "seed": ("First seed", "Train paths start here; held-out paths follow on."),
}


class AlphaKnobOut(Out):
    name: str
    label: str
    kind: Literal["number", "integer"]
    default: float
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""


def _knob(name: str, info: Any) -> AlphaKnobOut:
    bounds: dict[str, float | None] = {"minimum": None, "maximum": None}
    for meta in info.metadata:
        for attr, key in (
            ("ge", "minimum"),
            ("gt", "minimum"),
            ("le", "maximum"),
            ("lt", "maximum"),
        ):
            value = getattr(meta, attr, None)
            if value is not None:
                bounds[key] = float(value)
    label, text = _FIELDS.get(name, (name.replace("_", " ").capitalize(), ""))
    kind: Literal["number", "integer"] = "integer" if info.annotation is int else "number"
    return AlphaKnobOut(
        name=name, label=label, kind=kind, default=float(info.default), help=text, **bounds
    )


class AlphaGeneratorOut(Out):
    kind: str
    title: str
    edge: str = Field(description="What was planted.")
    taker: str = Field(description="Which policy can take it, and whether the agent sees it.")
    signal_label: str
    knobs: list[AlphaKnobOut]
    edge_knob: str | None = Field(
        default=None, description="The knob an edge-size sweep moves; None if it has none."
    )
    edge_unit: str | None = None


class AlphaArmOut(Out):
    """One arm the page can benchmark: a built-in or a plug-in registered on the server."""

    name: str
    label: str
    kind: Literal["learned", "scripted", "plugin"]
    trainable: bool = Field(description="Trained per seed with a budget, like PPO.")
    description: str = ""
    source: str | None = Field(default=None, description="A plug-in's import path and code digest.")


class AlphaOptionsOut(Out):
    generators: list[AlphaGeneratorOut]
    spec_knobs: list[AlphaKnobOut]
    arms: list[str]
    learned_arms: list[str] = Field(
        description="The arms that train per seed with a budget: built-in learners and "
        "plug-ins with a trainer."
    )
    learners_available: bool
    arm_details: list[AlphaArmOut] = Field(default_factory=list)
    plugin_error: str | None = Field(
        default=None, description="Why the configured plug-ins could not be loaded."
    )
    default_timesteps: int
    default_seeds: int
    max_seeds: int


_BUILTIN_LABEL = {
    "ppo": "PPO",
    "a2c": "A2C",
    "dqn": "DQN",
    "delta_seller": "Fixed-delta seller",
    "always_on": "Always-on condor",
    "flat": "Flat",
}


def alpha_arms() -> tuple[list[AlphaArmOut], str | None]:
    """The built-in arms this checkout can run, then every registered plug-in."""
    available = ppo_available()
    out = [
        AlphaArmOut(
            name=a,
            label=_BUILTIN_LABEL.get(a, a),
            kind="learned" if a in LEARNED_ARMS else "scripted",
            trainable=a in LEARNED_ARMS,
        )
        for a in ARMS
        if available or a not in LEARNED_ARMS
    ]
    try:
        loaded = plug.plugins()
    except plug.PluginError as exc:
        return out, str(exc)
    for p in loaded.values():
        out.append(
            AlphaArmOut(
                name=p.name,
                label=f"{p.plugin.title} (plug-in)",
                kind="plugin",
                trainable=p.plugin.trainable,
                description=p.plugin.description,
                source=f"{p.source}@{p.digest}",
            )
        )
    return out, None


def alpha_options() -> AlphaOptionsOut:
    union = get_args(get_args(GeneratorParams)[0])
    gens = []
    for model in union:
        kind = model.model_fields["kind"].default
        info = GENERATORS[kind]
        gens.append(
            AlphaGeneratorOut(
                kind=kind,
                title=info.title,
                edge=info.edge,
                taker=info.taker,
                signal_label=info.signal_label,
                knobs=[_knob(k, v) for k, v in model.model_fields.items() if k != "kind"],
                edge_knob=EDGE_KNOB[kind][0] if kind in EDGE_KNOB else None,
                edge_unit=EDGE_KNOB[kind][1] if kind in EDGE_KNOB else None,
            )
        )
    available = ppo_available()
    arms, error = alpha_arms()
    return AlphaOptionsOut(
        generators=gens,
        spec_knobs=[_knob(k, v) for k, v in SynthSpec.model_fields.items() if k != "generator"],
        arms=[a.name for a in arms],
        learned_arms=[*LEARNED_ARMS, *(a.name for a in arms if a.kind == "plugin" and a.trainable)],
        learners_available=available,
        arm_details=arms,
        plugin_error=error,
        default_timesteps=DEFAULT_TIMESTEPS,
        default_seeds=DEFAULT_SEEDS,
        max_seeds=MAX_SEEDS,
    )


class AlphaWorldOut(Out):
    spot: list[list[float]]
    iv: list[list[float]]
    signal: list[list[float]]


class AlphaPreviewOut(Out):
    days: list[int]
    signal: AlphaWorldOut
    twin: AlphaWorldOut


class AlphaDatasetOut(Out):
    hash: str
    spec: SynthSpec
    title: str
    edge: str
    taker: str
    signal_label: str
    n_train: int
    n_heldout: int
    preview: AlphaPreviewOut
    source: str


def alpha_dataset(ds: SynthDataset, n: int = 20) -> AlphaDatasetOut:
    info = GENERATORS[ds.spec.generator.kind]
    return AlphaDatasetOut(
        hash=ds.hash,
        spec=ds.spec,
        title=info.title,
        edge=info.edge,
        taker=info.taker,
        signal_label=info.signal_label,
        n_train=ds.n_paths("train"),
        n_heldout=ds.n_paths("heldout"),
        preview=AlphaPreviewOut.model_validate(preview(ds, n)),
        source=f"<runs root>/{DATASETS_DIR.as_posix()}/{ds.hash}/{PATHS_FILE}",
    )


class AlphaManifestOut(Out):
    hash: str
    kind: str
    title: str
    n_train: int
    n_heldout: int
    created_at: str


def alpha_manifests(manifests: list[dict[str, Any]]) -> list[AlphaManifestOut]:
    out = []
    for m in manifests:
        spec = m.get("spec", {})
        kind = str(spec.get("generator", {}).get("kind", ""))
        info = GENERATORS.get(kind)
        out.append(
            AlphaManifestOut(
                hash=str(m.get("hash", "")),
                kind=kind,
                title=info.title if info else kind,
                n_train=int(spec.get("n_train", 0)),
                n_heldout=int(spec.get("n_heldout", 0)),
                created_at=str(m.get("created_at", "")),
            )
        )
    return out


class AlphaJobOut(JobOut):
    kind: Literal["benchmark", "sweep"] = "benchmark"
    benchmark: Benchmark | None = None
    sweep: AlphaSweep | None = None


def alpha_job(status: JobStatus, root: Any, kind: Literal["benchmark", "sweep"]) -> AlphaJobOut:
    """A benchmark or sweep job's status, with its result once it is written."""
    bench, sweep = None, None
    if status.state == "done" and status.run_key:
        path = root / status.run_key
        if path.name == BENCH_FILE:
            bench = load_benchmark(path)
        else:
            sweep = load_sweep(path)
    return AlphaJobOut(**job_out(status).model_dump(), kind=kind, benchmark=bench, sweep=sweep)
