from __future__ import annotations

import math
from dataclasses import dataclass

from nearextremal_threshold.geometry.throat_scalings import T_H_scaling
from nearextremal_threshold.jt.bath_spectral import ThermalBath


@dataclass(frozen=True)
class GKLSRateModel:
    """
    Γ(Δ) = g^2 S_B(ω*)
    with ω* = 2π T_H as the dominant Schwarzian thermal scale.

    Provides:
      - Γ(Δ)
      - Γ_min = Γ(Δ_crit)
    """
    g: float = 1.0
    bath: ThermalBath = ThermalBath(N_B=1.0, Delta_B=1.0)

    def omega_star(self, T: float) -> float:
        return 2.0 * math.pi * T

    def Gamma(self, M: float, Delta: float) -> float:
        T = T_H_scaling(M, Delta)
        if T == 0:
            return 0.0
        w = self.omega_star(T)
        return (self.g * self.g) * self.bath.S(w, T)

    def Gamma_ohmic_closed_form(self, M: float, Delta: float) -> float:
        """
        Explicit ohmic formula used in manuscript:
          Γ = g^2 N_B * (2π T) / (1 - e^{-2π})
        """
        if abs(self.bath.Delta_B - 1.0) > 1e-12:
            raise ValueError("Closed form is only for Delta_B=1 (ohmic).")
        T = T_H_scaling(M, Delta)
        if T == 0:
            return 0.0
        return (self.g * self.g) * self.bath.N_B * (2.0 * math.pi * T) / (1.0 - math.exp(-2.0 * math.pi))