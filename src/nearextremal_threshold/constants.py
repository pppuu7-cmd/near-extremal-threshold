from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NaturalConstants:
    """
    Natural units defaults: G = ħ = c = k_B = 1.

    We still keep explicit names for readability.
    """
    G: float = 1.0
    hbar: float = 1.0
    c: float = 1.0
    kB: float = 1.0

    @property
    def lP(self) -> float:
        # ℓ_P = sqrt(G ħ / c^3) = 1 in natural units
        return math.sqrt(self.G * self.hbar / (self.c**3))

    @property
    def mP(self) -> float:
        # m_P = sqrt(ħ c / G) = 1 in natural units
        return math.sqrt(self.hbar * self.c / self.G)


NAT = NaturalConstants()