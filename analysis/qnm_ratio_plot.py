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
from nearextremal_threshold.plots.plotting import PlotConfig, new_fig


def pick_csv() -> str:
    data_dir = os.path.join(ROOT, "data", "qnm")
    bundle = os.path.join(data_dir, "qnm_kerr_bundle.csv")
    if os.path.exists(bundle):
        return bundle
    return str(newest_qnm_csv(data_dir))


def main():
    csv_path = pick_csv()
    print("Using QNM CSV:", csv_path)

    outdir = os.path.join(ROOT, "figures", "qnm_ratio")
    cfg = PlotConfig(outdir=outdir, dpi=260)

    rows_all = load_qnm_csv(csv_path)

    # Families to plot (if present)
    families = [(2,2,0), (2,2,1), (2,2,2), (3,3,0), (3,3,1), (4,4,0)]

    # Optional: focus on near-extremal window only
    chi_min_plot = 0.90  # set to 0.95 or 0.99 for stricter near-extremal
    eps = 1e-30

    fig, ax = new_fig()

    made = 0
    for ell, m, n in families:
        rows = filter_modes(rows_all, ell=ell, m=m, n=n)
        if len(rows) < 30:
            continue

        chis = np.array([r.chi for r in rows], dtype=float)
        mask = chis >= chi_min_plot
        chis = chis[mask]
        if len(chis) < 20:
            continue

        Imw = np.abs(np.array([r.Imw for r in rows], dtype=float))[mask]
        Delta = np.array([KerrParams(M=1.0, chi=chi).Delta_from_chi() for chi in chis], dtype=float)
        THM = np.array([KerrParams(M=1.0, chi=chi).T_H() for chi in chis], dtype=float)

        ratio = Imw / np.maximum(THM, eps)

        # sort by Delta for smooth lines
        idx = np.argsort(Delta)
        Delta = Delta[idx]
        ratio = ratio[idx]

        ax.plot(Delta, ratio, label=fr"$(\ell,m,n)=({ell},{m},{n})$")
        made += 1

    if made == 0:
        raise RuntimeError("No mode families found with enough points. Generate bundle first.")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta = 1-\chi^2$")
    ax.set_ylabel(r"$|\mathrm{Im}\,\omega M| /(T_H M)$")
    ax.set_title(fr"Near-extremal linearity test (chi >= {chi_min_plot})")
    ax.legend()

    cfg.savefig(fig, "Imw_over_THM_vs_Delta.png")
    print("Saved:", os.path.join(outdir, "Imw_over_THM_vs_Delta.png"))


if __name__ == "__main__":
    main()