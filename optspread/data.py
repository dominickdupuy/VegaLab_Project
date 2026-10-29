"""The data adapter boundary: where a chain comes from, kept separate from what it is.

Lane: Hemdutt Rao (market layer).

Everything downstream of this file consumes a ``ChainProvider`` and cannot tell
whether the chains were generated or loaded. That is the point. Real data was
designed in as an upgrade rather than a dependency, and it has now arrived: the
``optionmetrics`` source serves historical SPX chains from the stitched panel
(``optspread.realdata``), and nothing else in the package changed to take it.

Three sources:

    synthetic            the fair-IV generator (``optspread.market``)
    optionmetrics_stub   the real-data adapter's shape, served by the generator
    optionmetrics        real SPX chains: OptionMetrics to 2025-08-29, OPRA after;
                         unavailable (and says why) when the shared data folder is absent

The stub is what makes the boundary testable before the data exists. It sits at
the same interface the real adapter will occupy, so ``conformance_report`` proves
the downstream code is source-agnostic now, not after the fact.

A source of your own is any object with this shape (docs/EXTENDING.md). Its ``name``
is any string; the three above are the ones ``make_source`` builds. The two
generator-backed sources take a ``PlantedEffect`` (``optspread.effects``), the
constant premium unless told otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from optspread.config import MarketConfig
from optspread.effects import CONSTANT_PREMIUM, PlantedEffect
from optspread.leaks import sequence_violations
from optspread.market import SyntheticChainProvider
from optspread.realdata import PANEL_FILES as PANEL_FILES  # the manifest reads it from here
from optspread.realdata import Split, SplitRanges, check_split
from optspread.rng import SeededStepRNG
from optspread.types import ChainProvider

SourceName = Literal["synthetic", "optionmetrics_stub", "optionmetrics"]


class DataUnavailableError(RuntimeError):
    """A source that cannot serve chains yet. Never a silent fallback."""


@runtime_checkable
class ChainSource(Protocol):
    """A factory for providers. One source, many episodes, one provider each."""

    @property
    def name(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def provider(self) -> ChainProvider:
        """A fresh provider. Raises ``DataUnavailableError`` when ``available`` is False."""
        ...


@dataclass(frozen=True, slots=True)
class SyntheticSource:
    config: MarketConfig = field(default_factory=MarketConfig)
    effect: PlantedEffect = CONSTANT_PREMIUM

    @property
    def name(self) -> SourceName:
        return "synthetic"

    @property
    def available(self) -> bool:
        return True

    def provider(self) -> ChainProvider:
        return SyntheticChainProvider(self.config, self.effect)


@dataclass(frozen=True, slots=True)
class OptionMetricsStubSource:
    """The real adapter's interface, served by the generator.

    Conformance-passing by construction, which is the whole job: it keeps the
    OptionMetrics path exercised in CI while the WRDS request is outstanding. It
    is not a claim about real data, and it never masquerades as one -- ``name``
    says ``optionmetrics_stub`` and every artifact records it.
    """

    config: MarketConfig = field(default_factory=MarketConfig)
    effect: PlantedEffect = CONSTANT_PREMIUM

    @property
    def name(self) -> SourceName:
        return "optionmetrics_stub"

    @property
    def available(self) -> bool:
        return True

    def provider(self) -> ChainProvider:
        return SyntheticChainProvider(self.config, self.effect)


@dataclass(frozen=True, slots=True)
class OptionMetricsSource:
    """Real SPX chains from the stitched panel in the shared data folder.

    ``root`` defaults to ``optspread.realdata.data_root()``. Available exactly when
    the panel's surface grid and daily series are present; pandas is imported only
    when a provider is built, so the core package does not need it. ``split`` is the
    date range its provider serves (#28): ``train`` or ``eval``, disjoint, and
    neither reaches the dates after the evaluation range. Anything else, ``None``
    included, is a ValueError here, so no record can carry it.

    ``ranges`` replaces those dates (``optspread.realdata.SPLITS``, pre-registration
    A8) for a surface of your own under its own ``root`` (docs/EXTENDING.md). It is
    refused for the team's SPX panel, known by path (no ``root``, ``data_root()`` or
    ``DEFAULT_ROOT``) or by content (either panel file, symlinks followed, has a digest
    a team manifest recorded: ``optspread.realdata.is_team_panel``), here and again when
    the provider is built, so A8's dates cannot be widened there.
    """

    config: MarketConfig = field(default_factory=MarketConfig)
    root: Path | None = None
    split: Split = "train"
    ranges: SplitRanges | None = None

    def __post_init__(self) -> None:
        check_split(self.split)
        if self.ranges is None:
            return
        from optspread.realdata import refuse_team_ranges

        if self.root is None:
            raise ValueError(
                "date ranges of your own are for a surface under a root of your own; the "
                "default root is the team's SPX panel, which serves only the splits "
                "pre-registration A8 fixed (optspread.realdata.SPLITS)"
            )
        refuse_team_ranges(self.panel_dir, self.ranges)

    @property
    def name(self) -> SourceName:
        return "optionmetrics"

    @property
    def panel_dir(self) -> Path:
        from optspread.realdata import data_root

        return (self.root if self.root is not None else data_root()) / "panel"

    @property
    def available(self) -> bool:
        return all((self.panel_dir / f).exists() for f in PANEL_FILES)

    def provider(self) -> ChainProvider:
        if not self.available:
            raise DataUnavailableError(
                f"no SPX panel under {self.panel_dir} (set VEGALAB_DATA to the shared "
                "folder, or build it with scripts/build_spx_panel.py). Use "
                "'optionmetrics_stub' to exercise this boundary, or 'synthetic' to run."
            )
        from optspread.realdata.provider import HistoricalChainProvider, load_store

        store = load_store(self.panel_dir, self.split, self.ranges)
        return HistoricalChainProvider(self.config, store)


def make_source(
    name: SourceName, config: MarketConfig | None = None, *, split: Split = "train"
) -> ChainSource:
    """The named source. ``split`` picks the real source's date range
    (``optspread.realdata.SPLITS``) and is checked for every source, so a caller that
    passes ``None`` hears about it whichever source it asked for; the generated sources
    keep training and evaluation apart by seed and otherwise ignore it."""
    check_split(split)
    cfg = config if config is not None else MarketConfig()
    if name == "synthetic":
        return SyntheticSource(cfg)
    if name == "optionmetrics_stub":
        return OptionMetricsStubSource(cfg)
    return OptionMetricsSource(cfg, split=split)


# -- conformance: what any source must satisfy to sit at this boundary ------- #


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    source: str
    checked: bool  # False when the source is unavailable; not a pass
    n_chains: int
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.checked and not self.violations

    def render(self) -> str:
        if not self.checked:
            return f"{self.source:<20} SKIPPED (unavailable)"
        verdict = "PASS" if self.passed else "FAIL"
        head = f"{self.source:<20} {verdict} ({self.n_chains} chains)"
        return "\n".join([head, *(f"    - {v}" for v in self.violations)])


def conformance_report(source: ChainSource, seed: int = 0) -> ConformanceReport:
    """Drive a source through one full path and check the provider contract.

    Structural invariants come from ``optspread.leaks``: the same checker the leak
    suite uses, so a real adapter has to clear the bar the generator clears.
    """
    if not source.available:
        return ConformanceReport(source.name, checked=False, n_chains=0, violations=())

    violations: list[str] = []
    provider = source.provider()
    first = provider.reset(SeededStepRNG(seed))
    if first.t != 0:
        violations.append(f"reset must return the chain at t=0, got t={first.t}")
    if provider.done:
        violations.append("a fresh provider must not report done")

    chains = [first]
    try:
        for _ in range(provider.n_days):
            chains.append(provider.step())
    except Exception as exc:
        violations.append(f"step() failed before the horizon: {exc!r}")
    else:
        if not provider.done:
            violations.append(f"provider must report done after n_days={provider.n_days} steps")
    violations += sequence_violations(chains)

    # A second provider from the same source and seed must reproduce the path.
    replay = source.provider()
    try:
        replay.reset(SeededStepRNG(seed))
        same = all(replay.step().spot == c.spot for c in chains[1:])
    except Exception as exc:
        violations.append(f"replay failed: {exc!r}")
    else:
        if not same:
            violations.append("same source and seed must reproduce the same path")

    return ConformanceReport(source.name, True, len(chains), tuple(violations))


def conformance_suite(config: MarketConfig | None = None) -> tuple[ConformanceReport, ...]:
    """Every source at the boundary, in one table for the gate report."""
    names: tuple[SourceName, ...] = ("synthetic", "optionmetrics_stub", "optionmetrics")
    return tuple(conformance_report(make_source(n, config)) for n in names)
