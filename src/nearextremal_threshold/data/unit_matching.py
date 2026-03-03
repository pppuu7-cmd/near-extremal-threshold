from __future__ import annotations

from dataclasses import dataclass

from nearextremal_threshold.units import SI, mass_solar_kg


@dataclass(frozen=True)
class UnitMatcher:
    """
    Convert rates computed in natural/Planck units to dimensionless combinations comparable
    with Kerr QNM tables in terms of ω M (dimensionless).

    Key identity:
      (Γ M)_dimless = Γ_SI * (G M_SI / c^3)
    """

    def geom_time_of_mass(self, M_kg: float) -> float:
        # t_M = G M / c^3  [seconds]
        return SI.G * M_kg / (SI.c ** 3)

    def rate_to_dimless_GammaM(self, Gamma_SI: float, M_kg: float) -> float:
        return Gamma_SI * self.geom_time_of_mass(M_kg)

    def planck_rate_to_SI(self, Gamma_planck: float) -> float:
        # 1 (Planck time)^-1 = 1/t_P in SI
        return Gamma_planck / SI.tP

    def solar_mass_kg(self, M_solar: float) -> float:
        return M_solar * mass_solar_kg()