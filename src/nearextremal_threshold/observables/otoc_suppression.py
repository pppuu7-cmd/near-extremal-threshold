from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from nearextremal_threshold.geometry.throat_scalings import T_H_scaling
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel


@dataclass(frozen=True)
class OTOCSuppressionModel:
    """
    λ_eff(Δ) = λ_L(Δ) - Γ(Δ)
    with λ_L ≈ 2π T_H for Schwarzian/JT-like throat sector.

    This is the operational prediction for throat-supported chaotic growth.
    """
    gkls: GKLSRateModel = GKLSRateModel()

    def lambda_L(self, M: float, Delta: float) -> float:
        T = T_H_scaling(M, Delta)
        return 2.0 * math.pi * T

    def lambda_eff(self, M: float, Delta: float) -> float:
        return self.lambda_L(M, Delta) - self.gkls.Gamma(M, Delta)

    def curve(self, M: float, Deltas: np.ndarray) -> dict:
        lamL = np.array([self.lambda_L(M, d) for d in Deltas], dtype=float)
        lamE = np.array([self.lambda_eff(M, d) for d in Deltas], dtype=float)
        Gam = np.array([self.gkls.Gamma(M, d) for d in Deltas], dtype=float)
        return {"lambda_L": lamL, "lambda_eff": lamE, "Gamma": Gam}