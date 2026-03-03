from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import gamma


@dataclass(frozen=True)
class ThermalBath:
    """
    Thermal bath spectral density models.

    Two options:
      - ohmic: Δ_B = 1 -> S(ω) = N_B * ω / (1 - exp(-β ω))
      - conformal_1d: general Δ_B using |Γ(Δ_B + i ω/2πT)|^2 structure (up to normalization)
    """
    N_B: float = 1.0
    Delta_B: float = 1.0

    def S_ohmic(self, omega: float, T: float) -> float:
        if T <= 0:
            raise ValueError("T must be positive")

        if omega == 0.0:
            # limit ω/(1-e^{-βω}) -> 1/β = T
            return self.N_B * T

        beta = 1.0 / T
        x = beta * omega

        # Use expm1 for stability: 1 - exp(-x) = -expm1(-x)
        denom = -math.expm1(-x)

        # In extreme cases denom can underflow to 0; then use series limit safely.
        if denom == 0.0:
            # ω/(1-e^{-βω}) ~ 1/β = T
            return self.N_B * T

        return self.N_B * (omega / denom)

    def S_conformal_1d(self, omega: float, T: float) -> float:
        """
        Conformal 1D thermal spectral density (shape) for operator dimension Δ_B.
        Returns S(ω) with an overall N_B normalization.
        """
        if T <= 0:
            raise ValueError("T must be positive")
        DB = float(self.Delta_B)

        # x = ω / (2πT)
        x = omega / (2.0 * math.pi * T)

        # e^{βω/2} |Γ(Δ + i x)|^2 (2πT)^{2Δ-1}
        # Here ω/T might be large in other uses; guard exp overflow.
        y = 0.5 * (omega / T)
        if y > 700:  # exp(709) ~ 8e307 near float max
            factor = float("inf")
        else:
            factor = math.exp(y)

        pref = (2.0 * math.pi * T) ** (2.0 * DB - 1.0)
        val = abs(gamma(DB + 1j * x)) ** 2
        return self.N_B * pref * factor * val

    def S(self, omega: float, T: float) -> float:
        if abs(self.Delta_B - 1.0) < 1e-12:
            return self.S_ohmic(omega, T)
        return self.S_conformal_1d(omega, T)