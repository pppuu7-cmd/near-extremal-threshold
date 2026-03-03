from __future__ import annotations

import math
from dataclasses import dataclass


def _one_minus_chi2(chi: float) -> float:
    """
    Numerically stable computation of 1 - chi^2 using fused multiply-add when available:
      1 - chi^2 = fma(-chi, chi, 1).
    This avoids catastrophic cancellation for chi ~ 1.
    """
    # Python's math.fma is available in modern versions; if not, fall back.
    fma = getattr(math, "fma", None)
    if fma is not None:
        return fma(-chi, chi, 1.0)
    return 1.0 - chi * chi


@dataclass(frozen=True)
class KerrParams:
    """
    Kerr black hole in natural units (G=c=1).

    M: mass
    chi: dimensionless spin chi = a/M in [0,1)
    """
    M: float
    chi: float

    @property
    def a(self) -> float:
        return self.chi * self.M

    def Delta_from_chi(self) -> float:
        """
        Near-extremality parameter Δ = 1 - chi^2, computed stably.
        """
        if not (0.0 <= self.chi < 1.0):
            raise ValueError("chi must satisfy 0 <= chi < 1 for Kerr BH")
        d = _one_minus_chi2(self.chi)
        # Guard against tiny negative due to rounding near chi~1
        return d if d > 0.0 else 0.0

    def horizon_radii(self) -> tuple[float, float]:
        """
        r± = M ± sqrt(M^2 - a^2) = M ± M*sqrt(1-chi^2)
        Compute discriminant in a stable way using Δ = 1-chi^2.
        """
        if self.M <= 0.0:
            raise ValueError("M must be positive")
        if not (0.0 <= self.chi < 1.0):
            raise ValueError("chi must satisfy 0 <= chi < 1 for Kerr BH")

        Delta = self.Delta_from_chi()
        # disc = M^2 * Δ
        disc = (self.M * self.M) * Delta
        s = math.sqrt(disc) if disc > 0.0 else 0.0
        return (self.M + s, self.M - s)

    def kappa(self) -> float:
        """
        Surface gravity:
          κ = (r+ - r-) / (2 (r+^2 + a^2))
        """
        r_plus, r_minus = self.horizon_radii()
        denom = 2.0 * (r_plus * r_plus + self.a * self.a)
        if denom == 0.0:
            return 0.0
        return (r_plus - r_minus) / denom

    def T_H(self) -> float:
        """
        Hawking temperature:
          T_H = κ / (2π)
        """
        return self.kappa() / (2.0 * math.pi)