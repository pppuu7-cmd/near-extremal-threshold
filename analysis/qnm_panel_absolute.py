from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nearextremal_threshold.data.qnm_loader import load_qnm_csv, filter_modes, newest_qnm_csv
from nearextremal_threshold.geometry.kerr import KerrParams
from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel
from nearextremal_threshold.jt.bath_spectral import ThermalBath
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel
from nearextremal_threshold.data.unit_matching import UnitMatcher
from nearextremal_threshold.plots.plotting import PlotConfig, new_fig
from nearextremal_threshold.units import SI


def pick_csv() -> str:
    data_dir = os.path.join(ROOT, "data", "qnm")
    bundle = os.path.join(data_dir, "qnm_kerr_bundle.csv")
    if os.path.exists(bundle):
        return bundle
    return str(newest_qnm_csv(data_dir))


def main():
    csv_path = pick_csv()
    print("Using QNM CSV:", csv_path)

    outdir = os.path.join(ROOT, "figures", "qnm_absolute_panel")
    cfg = PlotConfig(outdir=outdir, dpi=260)

    families = [(2,2,0), (2,2,1), (2,2,2), (3,3,0), (3,3,1), (4,4,0)]

    M_solar = 10.0
    c = 1.0
    g = 1.0
    N_B = 1.0
    Delta_B = 1.0

    rows_all = load_qnm_csv(csv_path)

    um = UnitMatcher()
    M_kg = um.solar_mass_kg(M_solar)
    M_planck = M_kg / SI.mP

    cps = CPSScalingModel(c=c, lP=1.0)
    Delta_crit = cps.Delta_crit(M_planck)

    bath = ThermalBath(N_B=N_B, Delta_B=Delta_B)
    gk = GKLSRateModel(g=g, bath=bath)

    Gamma_min_planck = gk.Gamma(M_planck, Delta_crit)
    Gamma_min_SI = um.planck_rate_to_SI(Gamma_min_planck)
    Gamma_min_M = um.rate_to_dimless_GammaM(Gamma_min_SI, M_kg)

    made = 0
    for ell, m, n in families:
        rows = filter_modes(rows_all, ell=ell, m=m, n=n)
        if len(rows) < 20:
            continue

        chis = np.array([r.chi for r in rows], dtype=float)
        Imw = np.abs(np.array([r.Imw for r in rows], dtype=float))
        Delta = np.array([KerrParams(M=1.0, chi=chi).Delta_from_chi() for chi in chis], dtype=float)

        Gamma_planck = np.array([gk.Gamma(M_planck, d) for d in Delta], dtype=float)
        Gamma_SI = np.array([um.planck_rate_to_SI(Gp) for Gp in Gamma_planck], dtype=float)
        GammaM = np.array([um.rate_to_dimless_GammaM(Gsi, M_kg) for Gsi in Gamma_SI], dtype=float)

        fig, ax = new_fig()
        ax.scatter(Delta, Imw, s=10, label=r"data $|\mathrm{Im}\,\omega M|$")
        ax.plot(Delta, GammaM, label=r"model $(\Gamma M)(\Delta)$")
        ax.axhline(Gamma_min_M, linestyle="--", label=r"$(\Gamma_{\min} M)$")
        ax.axvline(Delta_crit, linestyle="--", label=r"$\Delta_{\rm crit}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta=1-\chi^2$")
        ax.set_ylabel(r"dimensionless")
        ax.set_title(fr"M={M_solar} $M_\odot$, mode (ell={ell},m={m},n={n})")
        ax.legend()
        cfg.savefig(fig, f"panel_abs_vs_Delta_ell{ell}_m{m}_n{n}.png")
        made += 1

    print("Saved panel figures into:", outdir)
    print("Panels made:", made)
    print("Delta_crit =", Delta_crit, "Gamma_min*M =", Gamma_min_M)


if __name__ == "__main__":
    main()