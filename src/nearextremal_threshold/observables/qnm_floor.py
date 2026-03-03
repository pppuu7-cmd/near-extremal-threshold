from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel


@dataclass(frozen=True)
class QNMFloorModel:
    """
    Operational QNM-width floor model for throat-supported modes.

    We represent:
      - classical width ~ a0 * T_H (toy model)
      - operationally allowed region: Δ >= Δ_crit (predictive semiclassics)
      - dissipative addition: Γ(Δ) from GKLS model
    """
    a0: float = 1.0  # toy proportionality for classical Im ω ~ a0 T_H
    cps: CPSScalingModel = CPSScalingModel(c=1.0, lP=1.0)
    gkls: GKLSRateModel = GKLSRateModel()

    def Delta_crit(self, M: float) -> float:
        return self.cps.Delta_crit(M)

    def width_total(self, M: float, Delta: float, T_H: float) -> float:
        # total width = classical + dissipative (minimal operational)
        return self.a0 * T_H + self.gkls.Gamma(M, Delta)

    def predict_width_curve(self, M: float, Deltas: np.ndarray, T_H_func) -> dict:
        Dc = self.Delta_crit(M)
        widths = []
        predictable = []
        for d in Deltas:
            TH = T_H_func(M, d)
            widths.append(self.width_total(M, d, TH))
            predictable.append(d >= Dc)
        return {"Delta_crit": Dc, "widths": np.array(widths), "predictable": np.array(predictable, dtype=bool)}