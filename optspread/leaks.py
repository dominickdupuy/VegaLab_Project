"""Leak detection suite (project document 3.2, 10.4).

Lane: Dominick Dupuy (statistical spine).

Three layers, plus a floor for the detector itself:

1. Perturbation property: resample every RNG stream after step ``t`` and the
   chain at ``t`` must not move (``perturbation_check``). Needs the per-step
   stream contract in ``optspread.rng``.
2. Structural invariants: ordered strikes and expiries, positive vols,
   no-arbitrage bounds and put-call parity (``chain_violations``).
3. Poisoned environment: ``PoisonedChainProvider`` contaminates the spot shown
   at ``t`` with ``eps`` times the return into ``t+1``. The perturbation check
   catches it exactly (the chain at ``t`` now moves when stream ``t+1`` is
   resampled). ``detect_leak`` is the statistical detector: no observation
   feature at ``t`` may predict the path's TRUE return into ``t+1``. It reads
   the true return from the provider, so it is a simulator instrument, not a
   black-box one: at ``eps = 1`` the shown series is a clean GBM shifted by a
   day and observations alone cannot tell.

``leak_floor`` plants leaks of decreasing size and records the smallest one the
detector still catches. Passing this suite shows we catch the modes it is
built to catch; it does not prove the absence of unknown leakage.

``PASS_CRITERION`` states the rule. Writing the gate report showed this suite had
no criterion anyone else could check: the author knew what a pass looked like, so
it read as a pass every time it ran.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from optspread.config import CostConfig, EnvConfig, MarketConfig
from optspread.costs import CostModel, make_cost_model
from optspread.env import OBS_NAMES, SpreadEnv
from optspread.market import SyntheticChainProvider
from optspread.policies import Policy, RandomPolicy
from optspread.pricing import bs_price
from optspread.rng import SeededStepRNG, SplicedStepRNG, StepRNG
from optspread.types import Chain, ChainProvider

PASS_CRITERION = """A run of the leak suite passes when all three hold:

1. No observation feature at t correlates with the path's true log return into
   t+1, at a Bonferroni-adjusted alpha across the features actually tested.
2. Every name in OBS_NAMES is either tested under the probe policy or recorded as
   constant by construction, with the reason given. A field that is silently
   skipped is not covered.
3. Every name in OBS_NAMES has a positive control: a leak planted in that field
   alone is detected. A detector that has never fired on a field cannot claim it.

Failing 1 means the world leaks. Failing 2 or 3 means the detector does not cover
what it reports on, which is the worse outcome, because it reads as a pass."""


def default_env(
    eps: float = 0.0, market: MarketConfig | None = None, env_config: EnvConfig | None = None
) -> SpreadEnv:
    """The env the suite runs against: clean at ``eps = 0``, spot-poisoned above it.

    Costs are off, because the detector asks whether the world leaks and a cost
    model cannot create or hide look-ahead. Lives here rather than in the tests so
    the gate report and the test suite probe exactly the same environment.
    """
    cfg = market if market is not None else MarketConfig()
    provider: ChainProvider = (
        PoisonedChainProvider(cfg, eps) if eps else SyntheticChainProvider(cfg)
    )
    return SpreadEnv(
        provider,
        make_cost_model(CostConfig(enabled=False)),
        env_config if env_config is not None else EnvConfig(),
    )


# -- 1. perturbation property --------------------------------------------- #


def chains_equal(a: Chain, b: Chain) -> bool:
    return (
        a.t == b.t
        and a.spot == b.spot
        and a.r == b.r
        and np.array_equal(a.strikes, b.strikes)
        and np.array_equal(a.expiry_days, b.expiry_days)
        and np.array_equal(a.ivs, b.ivs)
    )


def run_to(provider: ChainProvider, rng: StepRNG, t: int) -> Chain:
    chain = provider.reset(rng)
    for _ in range(t):
        chain = provider.step()
    return chain


def perturbation_check(
    make_provider: Callable[[], ChainProvider], seed: int, other_seed: int, t: int
) -> tuple[bool, bool]:
    """(past unchanged at t, future actually resampled at t+1) for a split at ``t``."""
    base, spliced = (
        SeededStepRNG(seed),
        SplicedStepRNG(SeededStepRNG(seed), SeededStepRNG(other_seed), t),
    )
    past_same = chains_equal(run_to(make_provider(), base, t), run_to(make_provider(), spliced, t))
    provider = make_provider()
    if t >= provider.n_days:
        return past_same, True
    future_differs = not chains_equal(
        run_to(provider, base, t + 1), run_to(make_provider(), spliced, t + 1)
    )
    return past_same, future_differs


# -- 2. structural invariants ---------------------------------------------- #


def chain_violations(chain: Chain, tol: float = 1e-6) -> list[str]:
    out: list[str] = []
    if not chain.spot > 0:
        out.append("spot must be positive")
    if np.any(np.diff(chain.strikes) <= 0):
        out.append("strikes not strictly increasing")
    if np.any(chain.expiry_days <= chain.t) or np.any(np.diff(chain.expiry_days) <= 0):
        out.append("expiries must be ascending and after t")
    if not (np.isfinite(chain.ivs).all() and np.all(chain.ivs > 0)):
        out.append("ivs must be finite and positive")
    for row, expiry in enumerate(chain.expiry_days):
        T = chain.tau(int(expiry))
        disc = float(np.exp(-chain.r * T))
        C = bs_price("C", chain.spot, chain.strikes, chain.r, chain.ivs[row], T)
        P = bs_price("P", chain.spot, chain.strikes, chain.r, chain.ivs[row], T)
        lower_c, lower_p = (
            np.maximum(chain.spot - chain.strikes * disc, 0),
            np.maximum(chain.strikes * disc - chain.spot, 0),
        )
        if np.any(chain.spot + tol < C) or np.any(lower_c - tol > C):
            out.append(f"call bounds violated at expiry row {row}")
        if np.any(chain.strikes * disc + tol < P) or np.any(lower_p - tol > P):
            out.append(f"put bounds violated at expiry row {row}")
        if np.any(np.abs(C - P - (chain.spot - chain.strikes * disc)) > tol):
            out.append(f"put-call parity violated at expiry row {row}")
    return out


def sequence_violations(chains: Sequence[Chain]) -> list[str]:
    out = [v for c in chains for v in chain_violations(c)]
    ts = [c.t for c in chains]
    if any(b - a != 1 for a, b in pairwise(ts)):
        out.append("timestamps not consecutive")
    return out


# -- 3. poisoned environment and detector ---------------------------------- #


class PoisonedChainProvider(SyntheticChainProvider):
    """Deliberate one-step look-ahead: the spot shown at ``t`` carries ``eps`` times
    the log return into ``t+1``. ``eps = 0`` is the clean provider."""

    def __init__(self, config: MarketConfig, eps: float) -> None:
        super().__init__(config)
        self.eps = eps

    def _chain(self, spot: float) -> Chain:
        if self.eps != 0.0 and self.day < self.n_days:
            spot *= float(np.exp(self.eps * self.log_return(self.day + 1)))
        return super()._chain(spot)


class FieldPoisonedEnv(SpreadEnv):
    """Look-ahead injected into ONE observation field, and nothing else.

    The positive control per field. ``PoisonedChainProvider`` contaminates the spot
    and so moves many features at once; this moves exactly the named one, which is
    what shows the detector can fire on that field rather than on its neighbours.
    """

    def __init__(
        self,
        provider: ChainProvider,
        costs: CostModel,
        config: EnvConfig,
        field: str,
        eps: float,
        scale: float = 1.0,
    ) -> None:
        super().__init__(provider, costs, config)
        if field not in OBS_NAMES:
            raise ValueError(f"{field!r} is not an observation field")
        self.field = field
        self.field_index = OBS_NAMES.index(field)
        self.eps = eps
        self.scale = scale

    def _obs(self) -> NDArray[np.float32]:
        obs = super()._obs()
        provider = self.provider
        if self.eps == 0.0 or not isinstance(provider, SyntheticChainProvider):
            return obs
        day = round(float(obs[0]) * self.config.episode_days)
        if day >= provider.n_days:
            return obs
        # The planted signal is eps standard deviations of the field's own variation,
        # so a control means the same thing in every field. next_ret is standardised
        # by the path's daily vol first.
        cfg = provider.config
        daily = cfg.sigma / float(np.sqrt(cfg.trading_days_per_year))
        z = provider.log_return(day + 1) / daily
        obs[self.field_index] += np.float32(self.eps * self.scale * z)
        return obs


def poison_field(
    env: SpreadEnv, field: str, eps: float = 1.0, scale: float = 1.0
) -> FieldPoisonedEnv:
    """Rebuild ``env`` with look-ahead in one observation field.

    ``scale`` is the field's own standard deviation, so ``eps`` is a signal-to-noise
    ratio rather than an absolute number of index points. Without it a single eps
    means something different in every field, and four of the fourteen silently
    failed their control because their natural variance dwarfed the planted leak.
    """
    return FieldPoisonedEnv(env.provider, env.costs, env.config, field, eps, scale)


@dataclass(frozen=True, slots=True)
class FeatureResult:
    name: str
    tested: bool
    t_stat: float
    p_value: float
    reason: str = ""  # why it was not tested


@dataclass(frozen=True, slots=True)
class LeakReport:
    n_pairs: int
    n_features_tested: int
    worst_feature: str
    worst_t: float
    p_value: float
    threshold: float  # Bonferroni-adjusted alpha
    flagged: bool
    features: tuple[FeatureResult, ...] = ()

    @property
    def untested(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if not f.tested)

    @property
    def covers_every_field(self) -> bool:
        """Criterion 2: nothing silently skipped. A constant field states its reason."""
        return len(self.features) == len(OBS_NAMES) and all(
            f.tested or f.reason for f in self.features
        )

    def render(self) -> str:
        lines = [f"{'field':<20}{'tested':>8}{'t':>10}{'p':>12}   reason"]
        for f in self.features:
            t = f"{f.t_stat:>10.2f}" if f.tested else f"{'--':>10}"
            pv = f"{f.p_value:>12.2e}" if f.tested else f"{'--':>12}"
            lines.append(f"{f.name:<20}{f.tested!s:>8}{t}{pv}   {f.reason}")
        lines.append(
            f"{self.n_pairs:,} pairs, {self.n_features_tested} of {len(OBS_NAMES)} fields "
            f"tested, threshold {self.threshold:.2e}"
        )
        lines.append(f"worst: {self.worst_feature or '--'} p={self.p_value:.2e}")
        lines.append("LEAK DETECTED" if self.flagged else "no leak detected")
        return chr(10).join(lines)


def detect_leak(
    make_env: Callable[[], SpreadEnv],
    seeds: Sequence[int],
    alpha: float = 0.01,
    policy: Policy | None = None,
) -> LeakReport:
    """Does any observation feature at ``t`` predict the path's true return into ``t+1``?

    The probe policy matters. A flat book leaves nine of the fourteen fields at a
    constant zero -- every inventory, greek and margin slot -- so a flat-only probe
    silently tests the price fields and reports a pass for the rest. The default
    probe trades, which gives those fields variance to correlate on. Its actions
    come from ``POLICY_CHANNEL``, independent of the path, so any correlation with
    the next return has to come from leakage rather than from the policy.

    Pooled pairs are independent under GBM, so a per-feature Pearson t-test with a
    Bonferroni correction across the fields actually tested is valid at ``alpha``.
    Every field is reported either way: tested with its statistic, or untested with
    the reason. Nothing is skipped in silence.
    """
    probe = policy if policy is not None else RandomPolicy()
    xs: list[NDArray[np.float32]] = []
    ys: list[float] = []
    for seed in seeds:
        env = make_env()
        provider = env.provider
        if not isinstance(provider, SyntheticChainProvider):
            raise TypeError("detect_leak needs a provider that exposes the true log_return")
        obs, info = env.reset(seed=seed)
        probe.reset(seed)
        done = False
        while not done:
            xs.append(obs)
            ys.append(provider.log_return(int(info["day"]) + 1))
            obs, _, done, _, info = env.step(probe.act(obs, info))
    X, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    n = len(y)

    features: list[FeatureResult] = []
    for j, name in enumerate(OBS_NAMES):
        if X[:, j].std() == 0.0:
            features.append(
                FeatureResult(name, False, 0.0, 1.0, "constant under the probe: carries no signal")
            )
            continue
        r = float(np.corrcoef(X[:, j], y)[0, 1])
        t = r * np.sqrt((n - 2) / max(1.0 - r * r, 1e-12))
        p = float(2.0 * stats.t.sf(abs(t), df=n - 2))
        features.append(FeatureResult(name, True, float(t), p))

    tested = [f for f in features if f.tested]
    threshold = alpha / max(len(tested), 1)
    worst = min(tested, key=lambda f: f.p_value) if tested else None
    return LeakReport(
        n_pairs=n,
        n_features_tested=len(tested),
        worst_feature=worst.name if worst else "",
        worst_t=worst.t_stat if worst else 0.0,
        p_value=worst.p_value if worst else 1.0,
        threshold=threshold,
        flagged=bool(worst and worst.p_value < threshold),
        features=tuple(features),
    )


@dataclass(frozen=True, slots=True)
class FloorReport:
    eps_grid: tuple[float, ...]
    flagged: tuple[bool, ...]
    floor: float | None  # smallest planted eps still detected; None if none were


def leak_floor(
    make_env: Callable[[float], SpreadEnv],
    eps_grid: Sequence[float],
    seeds: Sequence[int],
    alpha: float = 0.01,
) -> FloorReport:
    """Plant leaks of decreasing size and record the smallest the detector still catches."""
    grid = tuple(sorted(eps_grid, reverse=True))
    flags = tuple(detect_leak(partial(make_env, eps), seeds, alpha).flagged for eps in grid)
    caught = [eps for eps, f in zip(grid, flags, strict=True) if f]
    return FloorReport(grid, flags, min(caught) if caught else None)


# -- 4. per-field coverage: the detector must be able to fire on each field -- #


@dataclass(frozen=True, slots=True)
class FieldControl:
    field: str
    flagged: bool
    named_correctly: bool  # the flagged feature is the one that was poisoned
    p_value: float


@dataclass(frozen=True, slots=True)
class CoverageReport:
    controls: tuple[FieldControl, ...]
    clean: LeakReport

    @property
    def uncovered(self) -> tuple[str, ...]:
        """Fields where a planted leak went undetected: fields we cannot claim."""
        return tuple(c.field for c in self.controls if not c.flagged)

    @property
    def misattributed(self) -> tuple[str, ...]:
        return tuple(c.field for c in self.controls if c.flagged and not c.named_correctly)

    @property
    def passed(self) -> bool:
        """The stated criterion: clean is quiet, nothing is skipped, every field fires."""
        return (
            not self.clean.flagged
            and self.clean.covers_every_field
            and len(self.controls) == len(OBS_NAMES)
            and not self.uncovered
        )

    def render(self) -> str:
        lines = [f"{'field':<20}{'planted leak caught':>21}{'named':>8}{'p':>12}"]
        for c in self.controls:
            lines.append(
                f"{c.field:<20}{c.flagged!s:>21}{c.named_correctly!s:>8}{c.p_value:>12.2e}"
            )
        checks = (
            ("clean world is quiet", not self.clean.flagged),
            ("every field tested or explained", self.clean.covers_every_field),
            ("every field has a working control", not self.uncovered),
        )
        lines += [f"  [{'PASS' if ok else 'FAIL'}] {label}" for label, ok in checks]
        if self.uncovered:
            lines.append(f"  uncovered: {', '.join(self.uncovered)}")
        if self.misattributed:
            lines.append(f"  flagged but attributed elsewhere: {', '.join(self.misattributed)}")
        lines.append(f"Leak suite {'PASSES' if self.passed else 'FAILS'} its stated criterion.")
        return chr(10).join(lines)


def _poisoned_factory(
    make_env: Callable[[], SpreadEnv], field: str, eps: float, scale: float
) -> SpreadEnv:
    return poison_field(make_env(), field, eps, scale)


def field_scales(make_env: Callable[[], SpreadEnv], seeds: Sequence[int]) -> dict[str, float]:
    """Each field's own standard deviation under the probe, so controls are comparable.

    A field that is constant gets a scale of 1.0: there is no natural variation to
    take a fraction of, so the control plants an absolute signal instead.
    """
    xs: list[NDArray[np.float32]] = []
    probe = RandomPolicy()
    for seed in seeds:
        env = make_env()
        obs, info = env.reset(seed=seed)
        probe.reset(seed)
        done = False
        while not done:
            xs.append(obs)
            obs, _, done, _, info = env.step(probe.act(obs, info))
    X = np.asarray(xs, dtype=np.float64)
    stds = X.std(axis=0)
    return {name: float(stds[j]) if stds[j] > 0.0 else 1.0 for j, name in enumerate(OBS_NAMES)}


def field_coverage(
    make_env: Callable[[], SpreadEnv],
    seeds: Sequence[int],
    eps: float = 0.25,
    alpha: float = 0.01,
) -> CoverageReport:
    """Plant a leak in each observation field in turn and check the detector fires.

    Criterion 3. A detector that has never fired on a field cannot claim to cover
    it, and a flat-book probe never gives most of these fields any variance at all.
    Poisoning one field at a time also shows the finding is attributed to the right
    field rather than to a neighbour that moved with it.
    """
    clean = detect_leak(make_env, seeds, alpha)
    scales = field_scales(make_env, seeds)
    controls: list[FieldControl] = []
    for name in OBS_NAMES:
        make_poisoned = partial(_poisoned_factory, make_env, name, eps, scales[name])
        report = detect_leak(make_poisoned, seeds, alpha)
        controls.append(
            FieldControl(
                field=name,
                flagged=report.flagged,
                named_correctly=report.worst_feature == name,
                p_value=report.p_value,
            )
        )
    return CoverageReport(tuple(controls), clean)
