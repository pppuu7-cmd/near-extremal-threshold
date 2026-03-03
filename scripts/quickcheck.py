from __future__ import annotations

import os
import sys
import numpy as np

# Allow running without `pip install -e .`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel
from nearextremal_threshold.jt.bath_spectral import ThermalBath
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel


def main() -> None:
    M = 1e6
    cps = CPSScalingModel(c=1.0, lP=1.0)
    Dc = cps.Delta_crit(M)
    print("M =", M)
    print("Delta_crit =", Dc, "(expected ~ 1/M^2 =", 1.0 / (M * M), ")")

    bath = ThermalBath(N_B=1.0, Delta_B=1.0)
    gk = GKLSRateModel(g=1.0, bath=bath)
    Gam_at_Dc = gk.Gamma_ohmic_closed_form(M, Dc)
    print("Gamma_min (ohmic closed form) =", Gam_at_Dc)

    Deltas = np.array([Dc, 10 * Dc, 100 * Dc], dtype=float)
    Gs = [gk.Gamma_ohmic_closed_form(M, d) for d in Deltas]
    print("Gamma at [Dc,10Dc,100Dc] =", Gs)
    assert Gs[0] < Gs[1] < Gs[2], "Expected monotonic increase with Delta"

    print("Quickcheck passed.")


if __name__ == "__main__":
    main()