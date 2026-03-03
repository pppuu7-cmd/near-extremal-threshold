from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KerrNewmanParams:
    """
    Dimensionless parameters in natural units (G=c=1):
    M: mass
    a: spin parameter (J = a M)
    Q: charge
    """
    M: float
    a: float = 0.0
    Q: float = 0.0

    def horizon_radii(self) -> tuple[float, float]:
        """
        r± = M ± sqrt(M^2 - a^2 - Q^2)
        """
        disc = self.M * self.M - self.a * self.a - self.Q * self.Q
        if disc < 0:
            raise ValueError("Naked singularity: M^2 < a^2 + Q^2")
        s = math.sqrt(disc)
        return (self.M + s, self.M - s)

    def omega_H(self) -> float:
        r_plus, _ = self.horizon_radii()
        return self.a / (r_plus * r_plus + self.a * self.a)

    def T_H(self) -> float:
        """
        Hawking temperature:
        T = (r+ - r-) / (4π (r+^2 + a^2))
        """
        r_plus, r_minus = self.horizon_radii()
        return (r_plus - r_minus) / (4.0 * math.pi * (r_plus * r_plus + self.a * self.a))


def delta_from_horizon_gap(M: float, r_plus_minus_gap: float) -> float:
    """
    Define a dimensionless near-extremality Δ by:
    r+ - r- = 2 M sqrt(Δ)
    => Δ = ((r+ - r-) / (2M))^2
    """
    if M <= 0:
        raise ValueError("M must be positive")
    return (r_plus_minus_gap / (2.0 * M)) ** 2