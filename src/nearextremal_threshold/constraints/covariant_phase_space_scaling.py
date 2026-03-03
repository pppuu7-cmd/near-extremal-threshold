from __future__ import annotations

from dataclasses import dataclass

from nearextremal_threshold.geometry.throat_scalings import L2_scaling, redshift_Z_scaling


@dataclass(frozen=True)
class CPSScalingModel:
    """
    Covariant phase space-inspired scaling model for operational non-factorization.

    C(Δ) ~ c * (ℓ_P^2/M^2) * (L2/M) * Z(Δ)

    In natural units: ℓ_P = 1.
    """
    c: float = 1.0
    lP: float = 1.0

    def C_delta(self, M: float, Delta: float) -> float:
        if M <= 0:
            raise ValueError("M must be positive")
        if Delta <= 0:
            return float("inf")
        L2 = L2_scaling(M, Delta)
        Z = redshift_Z_scaling(M, Delta)
        return self.c * (self.lP * self.lP / (M * M)) * (L2 / M) * Z

    def Delta_crit(self, M: float) -> float:
        """
        Solve C(Δ)=1 for Δ under the scaling formula:
        C = c * lP^2/(M^2) * (1/Δ)
        => Δ_crit = c * lP^2 / M^2
        """
        return self.c * (self.lP * self.lP) / (M * M)