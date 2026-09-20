"""Hemdutt Rao: the data adapter boundary (week 2).

The conformance checker is only worth having if a non-conforming source fails it,
so each check here has a deliberately broken source as its positive control.
"""

from __future__ import annotations

import pytest

from optspread.config import MarketConfig
from optspread.data import (
    ConformanceReport,
    DataUnavailableError,
    OptionMetricsSource,
    OptionMetricsStubSource,
    SyntheticSource,
    conformance_report,
    conformance_suite,
    make_source,
)
from optspread.market import SyntheticChainProvider
from optspread.rng import StepRNG
from optspread.types import Chain, ChainProvider

SHORT = MarketConfig(n_days=15, tenor_days=(8, 15))


# -- the boundary holds for every source we can serve ----------------------- #


def test_every_available_source_conforms() -> None:
    reports = conformance_suite(SHORT)
    by_name = {r.source: r for r in reports}
    assert by_name["synthetic"].passed
    assert by_name["optionmetrics_stub"].passed


def test_the_pending_real_source_is_skipped_not_passed() -> None:
    report = conformance_suite(SHORT)[-1]
    assert report.source == "optionmetrics"
    assert not report.checked
    assert not report.passed, "an unavailable source must never read as a pass"
    assert "SKIPPED" in report.render()


def test_the_real_source_refuses_loudly() -> None:
    with pytest.raises(DataUnavailableError, match="WRDS"):
        OptionMetricsSource().provider()


def test_make_source_returns_the_named_source() -> None:
    assert isinstance(make_source("synthetic", SHORT), SyntheticSource)
    assert isinstance(make_source("optionmetrics_stub", SHORT), OptionMetricsStubSource)
    assert isinstance(make_source("optionmetrics", SHORT), OptionMetricsSource)


def test_the_stub_never_claims_to_be_real_data() -> None:
    assert OptionMetricsStubSource(SHORT).name == "optionmetrics_stub"


# -- positive controls: broken sources must fail ---------------------------- #


class _LateStart:
    """Returns its first chain at t=1: a provider that skips the present."""

    def __init__(self, config: MarketConfig) -> None:
        self._inner = SyntheticChainProvider(config)

    def reset(self, rng: StepRNG) -> Chain:
        self._inner.reset(rng)
        return self._inner.step()

    def step(self) -> Chain:
        return self._inner.step()

    @property
    def n_days(self) -> int:
        return self._inner.n_days

    @property
    def done(self) -> bool:
        return self._inner.done


class _Drifting:
    """Ignores the seed after reset, so two runs of one seed disagree."""

    def __init__(self, config: MarketConfig) -> None:
        self.config = config
        self._inner = SyntheticChainProvider(config)
        self._offset = 0.0

    def reset(self, rng: StepRNG) -> Chain:
        self._offset += 1.0
        return self._inner.reset(rng)

    def step(self) -> Chain:
        chain = self._inner.step()
        return Chain(
            chain.t, chain.spot + self._offset, chain.r, chain.strikes, chain.expiry_days, chain.ivs
        )

    @property
    def n_days(self) -> int:
        return self._inner.n_days

    @property
    def done(self) -> bool:
        return self._inner.done


class _Source:
    def __init__(self, provider: ChainProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def available(self) -> bool:
        return True

    def provider(self) -> ChainProvider:
        return self._provider


def _violations(provider: ChainProvider) -> tuple[str, ...]:
    report = conformance_report(_Source(provider))  # type: ignore[arg-type]
    assert isinstance(report, ConformanceReport)
    return report.violations


def test_a_provider_that_starts_late_is_caught() -> None:
    violations = _violations(_LateStart(SHORT))
    assert any("t=0" in v for v in violations), violations


def test_a_provider_that_ignores_its_seed_is_caught() -> None:
    violations = _violations(_Drifting(SHORT))
    assert any("reproduce" in v for v in violations), violations


def test_a_clean_provider_has_no_violations() -> None:
    assert _violations(SyntheticChainProvider(SHORT)) == ()


def test_determinism_check_is_not_fooled_by_a_shared_provider() -> None:
    # Same provider object returned twice: the replay pass must still line up,
    # which is what makes the _Drifting failure above meaningful.
    assert conformance_report(SyntheticSource(SHORT)).passed
    assert conformance_report(SyntheticSource(SHORT), seed=99).passed
