"""Black-Scholes pricing and analytic greeks.

Lane: Hemdutt Rao (market layer).

No dividend yield: SPX is priced on the forward and a flat ``q`` adds nothing to
the null world. Pure, vectorised, NaN-free at ``T -> 0`` and ``sigma -> 0`` where
the value collapses to intrinsic. Theta is per year; vega per unit vol.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

F = NDArray[np.float64]
B = NDArray[np.bool_]


def _norm_pdf(x: F) -> F:
    return np.asarray(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi), dtype=np.float64)


def _prep(
    right: str, S: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> tuple[F, F, F, F, F, B, F, F]:
    if right not in ("C", "P"):
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    S_, K_, r_, sig, T_ = (np.asarray(x, dtype=np.float64) for x in (S, K, r, sigma, T))
    live = (T_ > 0.0) & (sig > 0.0)
    vol = np.where(live, sig * np.sqrt(T_), 1.0)
    d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig * sig) * T_) / vol
    d2 = d1 - vol
    return S_, K_, r_, sig, T_, live, d1, d2


def intrinsic(right: str, S: ArrayLike, K: ArrayLike) -> F:
    S_, K_ = np.asarray(S, dtype=np.float64), np.asarray(K, dtype=np.float64)
    return np.maximum(S_ - K_, 0.0) if right == "C" else np.maximum(K_ - S_, 0.0)


def bs_price(
    right: str, S: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> F:
    S_, K_, r_, _, T_, live, d1, d2 = _prep(right, S, K, r, sigma, T)
    disc = np.exp(-r_ * T_)
    if right == "C":
        value = S_ * ndtr(d1) - K_ * disc * ndtr(d2)
    else:
        value = K_ * disc * ndtr(-d2) - S_ * ndtr(-d1)
    return np.asarray(np.where(live, value, intrinsic(right, S_, K_)), dtype=np.float64)


def bs_delta(
    right: str, S: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> F:
    S_, K_, _, _, _, live, d1, _ = _prep(right, S, K, r, sigma, T)
    if right == "C":
        value, dead = ndtr(d1), np.where(S_ > K_, 1.0, 0.0)
    else:
        value, dead = -ndtr(-d1), np.where(S_ < K_, -1.0, 0.0)
    return np.asarray(np.where(live, value, dead), dtype=np.float64)


def bs_greeks(
    right: str, S: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> tuple[F, F, F, F]:
    """(delta, gamma, vega, theta). Gamma, vega and theta are zero once expired."""
    S_, K_, r_, sig, T_, live, d1, d2 = _prep(right, S, K, r, sigma, T)
    sqrt_T = np.where(live, np.sqrt(T_), 1.0)
    pdf = _norm_pdf(d1)
    disc = np.exp(-r_ * T_)
    gamma = np.where(live, pdf / (S_ * sig * sqrt_T), 0.0)
    vega = np.where(live, S_ * pdf * sqrt_T, 0.0)
    decay = -S_ * pdf * sig / (2.0 * sqrt_T)
    if right == "C":
        theta = decay - r_ * K_ * disc * ndtr(d2)
    else:
        theta = decay + r_ * K_ * disc * ndtr(-d2)
    theta = np.where(live, theta, 0.0)
    delta = bs_delta(right, S, K, r, sigma, T)
    return delta, np.asarray(gamma), np.asarray(vega), np.asarray(theta)
