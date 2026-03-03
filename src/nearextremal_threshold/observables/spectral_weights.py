from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralWeight:
    """
    Simple model spectral weights W_O(ω) to test robustness.

    We provide:
      - gaussian peak around ω*
      - broad lorentzian
    """
    kind: str = "gaussian"  # "gaussian" or "lorentzian"
    width: float = 1.0      # width in units of ω*
    normalize: bool = True

    def W(self, omega: np.ndarray, omega_star: float) -> np.ndarray:
        if omega_star <= 0:
            raise ValueError("omega_star must be positive")
        x = omega / omega_star
        if self.kind == "gaussian":
            w = np.exp(-0.5 * (x - 1.0) ** 2 / (self.width ** 2))
        elif self.kind == "lorentzian":
            w = 1.0 / (1.0 + ((x - 1.0) / self.width) ** 2)
        else:
            raise ValueError(f"Unknown weight kind: {self.kind}")
        if self.normalize:
            area = np.trapz(w, omega)
            if area > 0:
                w = w / area
        return w