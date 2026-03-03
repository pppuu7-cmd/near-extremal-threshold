from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SIConstants:
    # CODATA-like values (hardcoded). Good enough for illustrative plots.
    c: float = 299_792_458.0
    hbar: float = 1.054_571_817e-34
    G: float = 6.674_30e-11
    kB: float = 1.380_649e-23

    @property
    def lP(self) -> float:
        return math.sqrt(self.G * self.hbar / (self.c**3))

    @property
    def tP(self) -> float:
        return self.lP / self.c

    @property
    def mP(self) -> float:
        return math.sqrt(self.hbar * self.c / self.G)


SI = SIConstants()


def mass_solar_kg() -> float:
    return 1.98847e30


def M_solar_in_planck_mass() -> float:
    """
    Convert 1 solar mass to Planck mass units (natural units).
    """
    return mass_solar_kg() / SI.mP


def to_planck_mass_units(mass_kg: float) -> float:
    return mass_kg / SI.mP


def from_planck_mass_units(M: float) -> float:
    return M * SI.mP