from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel
from nearextremal_threshold.geometry.throat_scalings import T_H_scaling, L2_scaling, redshift_Z_scaling
from nearextremal_threshold.jt.bath_spectral import ThermalBath
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel
from nearextremal_threshold.geometry.kerr import KerrParams


def assert_monotone_increasing(arr: np.ndarray, name: str):
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        raise AssertionError(f"{name} is not strictly increasing.")


def main():
    # ----- Claim 1: Delta_crit scaling -----
    cps = CPSScalingModel(c=1.0, lP=1.0)
    Ms = np.array([1e3, 1e4, 1e5, 1e6], dtype=float)
    Dcs = np.array([cps.Delta_crit(M) for M in Ms], dtype=float)
    expected = 1.0 / (Ms * Ms)
    rel = np.max(np.abs(Dcs - expected) / expected)
    print("Claim 1: Delta_crit ~ 1/M^2  | max rel err:", rel)
    assert rel < 1e-12

    # ----- Claim 2: throat scaling consistency -----
    M = 1e6
    Deltas = np.array([1e-12, 1e-10, 1e-8], dtype=float)
    TH = np.array([T_H_scaling(M, d) for d in Deltas], dtype=float)
    L2 = np.array([L2_scaling(M, d) for d in Deltas], dtype=float)
    Z = np.array([redshift_Z_scaling(M, d) for d in Deltas], dtype=float)

    # TH should increase with Delta, L2 and Z should decrease with Delta
    assert_monotone_increasing(TH, "T_H(Delta)")
    if not np.all(np.diff(L2) < 0):
        raise AssertionError("L2(Delta) not decreasing with Delta.")
    if not np.all(np.diff(Z) < 0):
        raise AssertionError("Z(Delta) not decreasing with Delta.")
    print("Claim 2: throat scalings monotonic OK")

    # ----- Claim 3: C(Delta) ~ 1/Delta and Delta_crit solves C=1 -----
    Cvals = np.array([cps.C_delta(M, d) for d in Deltas], dtype=float)
    # Check ratios C(d1)/C(d2) ~ d2/d1
    ratio = Cvals[0] / Cvals[1]
    expected_ratio = Deltas[1] / Deltas[0]
    print("Claim 3: C ratio:", ratio, "expected:", expected_ratio)
    assert abs(ratio / expected_ratio - 1.0) < 1e-12

    Dc = cps.Delta_crit(M)
    C_at_Dc = cps.C_delta(M, Dc)
    print("C(Delta_crit) =", C_at_Dc)
    assert abs(C_at_Dc - 1.0) < 1e-12
    print("Claim 3 OK")

    # ----- Claim 4: GKLS ohmic closed-form matches general for Delta_B=1 -----
    bath = ThermalBath(N_B=1.0, Delta_B=1.0)
    gk = GKLSRateModel(g=1.0, bath=bath)
    for d in [1e-12, 1e-10, 1e-8]:
        G1 = gk.Gamma(M, d)
        G2 = gk.Gamma_ohmic_closed_form(M, d)
        if abs(G1 - G2) / max(1e-30, abs(G2)) > 1e-10:
            raise AssertionError("Ohmic Gamma mismatch")
    print("Claim 4 OK: ohmic closed form matches.")

    # ----- Claim 5: Kerr mapping Delta = 1 - chi^2, and TH->0 as chi->1 -----
    chis = np.array([0.9, 0.99, 0.999], dtype=float)
    Dk = np.array([KerrParams(M=1.0, chi=chi).Delta_from_chi() for chi in chis], dtype=float)
    THk = np.array([KerrParams(M=1.0, chi=chi).T_H() for chi in chis], dtype=float)
    if not np.all(np.diff(Dk) < 0):
        raise AssertionError("Delta(chi) should decrease as chi increases")
    if not np.all(np.diff(THk) < 0):
        raise AssertionError("T_H should decrease as chi increases")
    print("Claim 5 OK: Kerr Delta mapping + TH behavior.")

    print("\nAll validation checks PASSED.")


if __name__ == "__main__":
    main()