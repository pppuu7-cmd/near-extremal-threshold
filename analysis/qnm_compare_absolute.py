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


def available_families(rows) -> list[tuple[int,int,int]]:
    fams = sorted({(r.ell, r.m, r.n) for r in rows})
    return fams


def main():
    csv_path = pick_csv()
    print("Using QNM CSV:", csv_path)

    # preferred mode to plot
    preferred = (2, 2, 0)

    # physical mass for unit matching
    M_solar = 10.0

    # operational model params
    c = 1.0
    g = 1.0
    N_B = 1.0
    Delta_B = 1.0  # ohmic

    outdir = os.path.join(ROOT, "figures", "qnm_absolute")
    cfg = PlotConfig(outdir=outdir, dpi=260)

    rows_all = load_qnm_csv(csv_path)
    fams = available_families(rows_all)
    if not fams:
        raise RuntimeError("No data rows found in CSV.")

    ell, m, n = preferred
    rows = filter_modes(rows_all, ell=ell, m=m, n=n)
    if not rows:
        print("Preferred mode not found:", preferred)
        print("Available modes (ell,m,n):", fams[:30], ("..." if len(fams) > 30 else ""))
        ell, m, n = fams[0]
        rows = filter_modes(rows_all, ell=ell, m=m, n=n)
        print("Falling back to first available mode:", (ell, m, n))

    chis = np.array([r.chi for r in rows], dtype=float)
    Imw = np.abs(np.array([r.Imw for r in rows], dtype=float))

    Delta = np.array([KerrParams(M=1.0, chi=chi).Delta_from_chi() for chi in chis], dtype=float)
    THM = np.array([KerrParams(M=1.0, chi=chi).T_H() for chi in chis], dtype=float)

    # unit matching
    um = UnitMatcher()
    M_kg = um.solar_mass_kg(M_solar)
    M_planck = M_kg / SI.mP

    cps = CPSScalingModel(c=c, lP=1.0)
    Delta_crit = cps.Delta_crit(M_planck)

    bath = ThermalBath(N_B=N_B, Delta_B=Delta_B)
    gk = GKLSRateModel(g=g, bath=bath)

    Gamma_planck = np.array([gk.Gamma(M_planck, d) for d in Delta], dtype=float)
    Gamma_SI = np.array([um.planck_rate_to_SI(Gp) for Gp in Gamma_planck], dtype=float)
    GammaM = np.array([um.rate_to_dimless_GammaM(Gsi, M_kg) for Gsi in Gamma_SI], dtype=float)

    Gamma_min_planck = gk.Gamma(M_planck, Delta_crit)
    Gamma_min_SI = um.planck_rate_to_SI(Gamma_min_planck)
    Gamma_min_M = um.rate_to_dimless_GammaM(Gamma_min_SI, M_kg)

    # plot: vs Delta
    fig, ax = new_fig()
    ax.scatter(Delta, Imw, s=12, label=r"data $|\mathrm{Im}\,\omega M|$")
    ax.plot(Delta, GammaM, label=r"model $(\Gamma M)(\Delta)$")
    ax.axhline(Gamma_min_M, linestyle="--", label=r"$(\Gamma_{\min} M)$")
    ax.axvline(Delta_crit, linestyle="--", label=r"$\Delta_{\rm crit}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta=1-\chi^2$")
    ax.set_ylabel(r"dimensionless")
    ax.set_title(fr"M={M_solar} $M_\odot$, mode (ell={ell},m={m},n={n})")
    ax.legend()
    cfg.savefig(fig, f"qnm_abs_vs_Delta_ell{ell}_m{m}_n{n}.png")

    # plot: vs THM
    fig, ax = new_fig()
    ax.scatter(THM, Imw, s=12, label=r"data $|\mathrm{Im}\,\omega M|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_H M$")
    ax.set_ylabel(r"$|\mathrm{Im}\,\omega M|$")
    ax.set_title("Scaling diagnostic: |Im ωM| vs T_H M")
    ax.legend()
    cfg.savefig(fig, f"qnm_abs_vs_THM_ell{ell}_m{m}_n{n}.png")

    print("Saved figures into:", outdir)
    print("Chosen mode:", (ell, m, n))
    print("M_solar =", M_solar, " => M_planck =", M_planck)
    print("Delta_crit =", Delta_crit)
    print("Gamma_min*M (dimless) =", Gamma_min_M)


if __name__ == "__main__":
    main()