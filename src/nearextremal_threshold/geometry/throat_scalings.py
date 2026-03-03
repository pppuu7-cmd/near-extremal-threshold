from __future__ import annotations

import math


def _clamp_nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0


def T_H_scaling(M: float, Delta: float) -> float:
    """
    Universal near-extremal scaling used in the manuscript:
      T_H(Δ) ~ sqrt(Δ) / M
    Natural units.

    Numerically stable:
      - clamp Δ to 0 if tiny negative from rounding
      - validate M>0
    """
    if M <= 0.0:
        raise ValueError("M must be positive")
    if Delta < 0.0:
        # allow tiny negative due to upstream rounding; otherwise error
        if Delta > -1e-18:
            Delta = 0.0
        else:
            raise ValueError("Delta must be nonnegative")
    return math.sqrt(_clamp_nonneg(Delta)) / M


def L2_scaling(M: float, Delta: float) -> float:
    """
    Throat scale:
      L2(Δ) ~ 1/T_H ~ M/sqrt(Δ)
    """
    TH = T_H_scaling(M, Delta)
    if TH == 0.0:
        return float("inf")
    return 1.0 / TH


def redshift_Z_scaling(M: float, Delta: float) -> float:
    """
    Redshift factor scaling:
      Z(Δ) ~ L2/M ~ 1/sqrt(Δ)
    """
    if Delta < 0.0:
        if Delta > -1e-18:
            Delta = 0.0
        else:
            raise ValueError("Delta must be nonnegative")
    d = _clamp_nonneg(Delta)
    if d == 0.0:
        return float("inf")
    return 1.0 / math.sqrt(d)