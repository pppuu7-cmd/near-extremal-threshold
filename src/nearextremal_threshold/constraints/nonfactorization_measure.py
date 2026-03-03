from __future__ import annotations

import numpy as np

from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel


def predictivity_zone_mask(M: float, Deltas: np.ndarray, model: CPSScalingModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      predictable: boolean mask where C(Δ) < 1
      unpredictable: boolean mask where C(Δ) >= 1
    """
    Cvals = np.array([model.C_delta(M, d) for d in Deltas], dtype=float)
    predictable = Cvals < 1.0
    unpredictable = ~predictable
    return predictable, unpredictable


def C_values(M: float, Deltas: np.ndarray, model: CPSScalingModel) -> np.ndarray:
    return np.array([model.C_delta(M, d) for d in Deltas], dtype=float)