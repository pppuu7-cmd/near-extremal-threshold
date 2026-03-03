from __future__ import annotations

import os
import sys
import csv
import numpy as np

# allow running without install
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nearextremal_threshold.data.qnm_loader import load_qnm_csv, filter_modes
from nearextremal_threshold.geometry.kerr import KerrParams
from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel
from nearextremal_threshold.jt.bath_spectral import ThermalBath
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel
from nearextremal_threshold.plots.plotting import PlotConfig, new_fig


def ensure_template(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chi","ell","m","n","Rew","Imw","source","note"])
        w.writerow([0.90,2,2,0,0.50,-0.05,"template","example row"])
        w.writerow([0.99,2,2,0,0.45,-0.01,"template","example row"])
    print(f"[created template] {csv_path}")
    print("Put your real Kerr QNM CSVs into data/qnm/ and update csv_path in this script.")


def main():
    csv_path = os.path.join(ROOT, "data", "qnm", "qnm_template.csv")
    ensure_template(csv_path)

    # ---- user inputs ----
    ell, m, n = 2, 2, 0
    M_planck = 1e6     # mass in Planck units (only for operational Delta_crit here)
    c = 1.0
    g = 1.0
    N_B = 1.0
    Delta_B = 1.0

    outdir = os.path.join(ROOT, "figures", "qnm")
    cfg = PlotConfig(outdir=outdir, dpi=240)

    # ---- load data ----
    rows = load_qnm_csv(csv_path)
    rows = filter_modes(rows, ell=ell, m=m, n=n)
    if not rows:
        raise RuntimeError("No rows after filtering. Check (ell,m,n) and CSV content.")

    chis = np.array([r.chi for r in rows], dtype=float)
    Imw = np.array([r.Imw for r in rows], dtype=float)

    # ---- derived (dimensionless for QNM): Delta=1-chi^2 and T_H * M (with M=1) ----
    Delta = np.array([KerrParams(M=1.0, chi=chi).Delta_from_chi() for chi in chis], dtype=float)
    THM = np.array([KerrParams(M=1.0, chi=chi).T_H() for chi in chis], dtype=float)

    # ---- operational threshold for chosen macroscopic mass (Planck units) ----
    cps = CPSScalingModel(c=c, lP=1.0)
    Delta_crit = cps.Delta_crit(M_planck)

    bath = ThermalBath(N_B=N_B, Delta_B=Delta_B)
    gk = GKLSRateModel(g=g, bath=bath)

    # A purely illustrative model curve (not unit-matched to ωM yet)
    Gamma_model = np.array([gk.Gamma(M_planck, d) for d in Delta], dtype=float)
    Gamma_min = gk.Gamma(M_planck, Delta_crit)

    # ---- figure 1: |Im ωM| vs Delta ----
    fig, ax = new_fig()
    ax.scatter(Delta, np.abs(Imw), label=r"data $|\mathrm{Im}\,\omega M|$")
    ax.plot(Delta, Gamma_model, label=r"model $\Gamma(\Delta)$ (illustrative units)")
    ax.axhline(Gamma_min, linestyle="--", label=r"$\Gamma_{\min}=\Gamma(\Delta_{\rm crit})$")
    ax.axvline(Delta_crit, linestyle="--", label=r"$\Delta_{\rm crit}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta = 1-\chi^2$")
    ax.set_ylabel(r"dimensionless")
    ax.set_title(fr"Kerr QNM mode (ell={ell}, m={m}, n={n})")
    ax.legend()
    cfg.savefig(fig, f"qnm_vs_Delta_ell{ell}_m{m}_n{n}.png")

    # ---- figure 2: |Im ωM| vs T_H M ----
    fig, ax = new_fig()
    ax.scatter(THM, np.abs(Imw), label=r"data $|\mathrm{Im}\,\omega M|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_H M$")
    ax.set_ylabel(r"$|\mathrm{Im}\,\omega M|$")
    ax.set_title("Scaling diagnostic: width vs Hawking temperature")
    ax.legend()
    cfg.savefig(fig, f"qnm_vs_THM_ell{ell}_m{m}_n{n}.png")

    print("Saved figures into:", outdir)
    print("Delta_crit(M_planck) =", Delta_crit, "for M_planck =", M_planck)


if __name__ == "__main__":
    main()