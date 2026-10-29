"""Planted effects: the deliberate departure from the null world, as an interface.

Lane: Hemdutt Rao (market layer).

The null world prices every contract at the vol that drives the path. A planted
effect is what the market lays on that fair surface when it builds a chain. The
effect fixes the shape; the size is ``MarketConfig.iv_premium_vol_pts``, in vol
points, so the sweep's premium axis drives any effect unchanged and zero premium is
the null world whichever effect is planted. The detection event (#26), each arm at
premium p against itself at zero, therefore needs nothing new to test another effect.

``ConstantPremium`` is the confirmatory design's effect (D2) and the default
everywhere: the whole surface shifted by the premium, the same floats the market
produced before this interface existed. ``SkewPremium`` and ``TermPremium`` are
examples for research, not part of the confirmatory design. Nothing in a config
selects them: an effect is passed to the market in Python (``SyntheticChainProvider``,
``SyntheticSource``), so the config hash of a run does not record which effect it
planted. docs/EXTENDING.md says how to keep such runs apart.

An effect is a pure function of the fair chain at ``t`` and the size. It reads no
RNG stream and no later chain, so the no-look-ahead contract the leak suite checks
holds for every effect by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from optspread.rng import StepRNG
from optspread.types import Chain, ChainProvider, FloatArray


@runtime_checkable
class PlantedEffect(Protocol):
    def apply(self, fair: Chain, vol_pts: float) -> FloatArray:
        """The planted IV surface for ``fair``, shape ``fair.ivs.shape``, at size ``vol_pts``.

        Returns a new array and leaves ``fair`` alone. ``vol_pts == 0`` must return the
        fair surface, so the null world is the same under every effect.
        """
        ...


@dataclass(frozen=True, slots=True)
class ConstantPremium:
    """Every contract ``vol_pts`` vol points richer: the planted variance risk premium (D2)."""

    def apply(self, fair: Chain, vol_pts: float) -> FloatArray:
        return fair.ivs + vol_pts / 100.0


@dataclass(frozen=True, slots=True)
class SkewPremium:
    """Example: the premium rises into the downside, as the SPX premium does.

    ``vol_pts`` at the money, growing by ``put_slope`` times itself per standard
    deviation of downside moneyness, ``x = ln(K / F) / (iv sqrt(tau))``, and by
    ``call_slope`` per standard deviation on the upside. At the default a put two
    standard deviations out carries twice the at-the-money premium and the calls
    carry the at-the-money premium alone.
    """

    put_slope: float = 0.5
    call_slope: float = 0.0

    def apply(self, fair: Chain, vol_pts: float) -> FloatArray:
        taus = np.array([fair.tau(int(e)) for e in fair.expiry_days], dtype=np.float64)
        forwards = fair.spot * np.exp(fair.r * taus)
        x = np.log(fair.strikes[None, :] / forwards[:, None]) / (fair.ivs * np.sqrt(taus)[:, None])
        scale = 1.0 + self.put_slope * np.maximum(-x, 0.0) + self.call_slope * np.maximum(x, 0.0)
        return np.asarray(fair.ivs + vol_pts / 100.0 * scale, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TermPremium:
    """Example: the premium is largest at short tenors and decays with time to expiry.

    ``vol_pts * (ref_days / d) ** decay`` on an expiry ``d`` trading days out: the full
    premium at ``ref_days``, more inside it, less beyond. At the default an expiry
    four times as far out carries half the premium.

    A held leg is marked on the nearest listed tenor (``Chain.iv_at``), so as it ages
    it moves onto a row with another premium: a step in its mark that comes from the
    lookup, not the effect (#27 describes the same step on real chains).
    """

    ref_days: int = 21
    decay: float = 0.5

    def apply(self, fair: Chain, vol_pts: float) -> FloatArray:
        days = np.maximum(fair.expiry_days - fair.t, 1).astype(np.float64)
        scale = (self.ref_days / days) ** self.decay
        return np.asarray(fair.ivs + vol_pts / 100.0 * scale[:, None], dtype=np.float64)


CONSTANT_PREMIUM = ConstantPremium()


def plant(effect: PlantedEffect, fair: Chain, vol_pts: float) -> Chain:
    """``fair`` with its surface replaced by the planted one; every other field kept."""
    return replace(fair, ivs=np.asarray(effect.apply(fair, vol_pts), dtype=np.float64))


class PlantedChainProvider:
    """Any ``ChainProvider`` with an effect planted on every chain it serves.

    For a source that plants nothing itself, such as real chains
    (``optspread.data.OptionMetricsSource`` serves the market's own surface): the
    wrapped chain at ``t`` is the inner one at ``t`` with its surface replaced, so the
    wrapper reads no stream the inner provider did not and keeps its contract. The
    synthetic provider plants its own effect, so pass it the effect instead: wrapped,
    it plants twice unless its own premium is zero.
    """

    def __init__(self, inner: ChainProvider, effect: PlantedEffect, vol_pts: float) -> None:
        self.inner = inner
        self.effect = effect
        self.vol_pts = vol_pts

    @property
    def n_days(self) -> int:
        return self.inner.n_days

    @property
    def done(self) -> bool:
        return self.inner.done

    def reset(self, rng: StepRNG) -> Chain:
        return plant(self.effect, self.inner.reset(rng), self.vol_pts)

    def step(self) -> Chain:
        return plant(self.effect, self.inner.step(), self.vol_pts)
