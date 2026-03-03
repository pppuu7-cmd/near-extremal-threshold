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


def fit_powerlaw_THM(Imw: np.ndarray, THM: np.ndarray, thm_floor: float = 1e-12) -> tuple[float, float]:
    """
    Fit: Imw ~ A * THM^alpha in log-log.
    Robustness:
      - drop nonpositive values
      - enforce THM > thm_floor to avoid log(0) / extreme underflow near chi->1
    Returns (A, alpha).
    """
    mask = (Imw > 0) & (THM > thm_floor) & np.isfinite(Imw) & np.isfinite(THM)
    Imw = Imw[mask]
    THM = THM[mask]
    if len(Imw) < 20:
        raise ValueError("Not enough points for fit after masking")
    X = np.log(THM)
    Y = np.log(Imw)
    alpha, logA = np.polyfit(X, Y, 1)
    return float(np.exp(logA)), float(alpha)


def main():
    csv_path = pick_csv()
    print("Using QNM CSV:", csv_path)

    outdir = os.path.join(ROOT, "figures", "qnm_alpha_window")
    cfg = PlotConfig(outdir=outdir, dpi=260)

    rows_all = load_qnm_csv(csv_path)

    # mode families to analyze (those present will be used)
    families = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (3, 3, 0), (3, 3, 1), (4, 4, 0)]

    # Sliding windows toward extremality
    chi_mins = np.array([0.0, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999], dtype=float)

    # Avoid log underflow for extremely small T_H M
    thm_floor = 1e-12

    fig, ax = new_fig()
    made = 0

    for ell, m, n in families:
        rows = filter_modes(rows_all, ell=ell, m=m, n=n)
        if len(rows) < 40:
            continue

        chis_full = np.array([r.chi for r in rows], dtype=float)
        Imw_full = np.abs(np.array([r.Imw for r in rows], dtype=float))
        THM_full = np.array([KerrParams(M=1.0, chi=chi).T_H() for chi in chis_full], dtype=float)

        xs = []
        alphas = []

        for chi_min in chi_mins:
            mask = chis_full >= chi_min
            if np.sum(mask) < 25:
                continue
            try:
                _, alpha = fit_powerlaw_THM(Imw_full[mask], THM_full[mask], thm_floor=thm_floor)
            except ValueError:
                continue
            xs.append(float(chi_min))
            alphas.append(float(alpha))

        if len(alphas) < 3:
            continue

        ax.plot(xs, alphas, marker="o", label=f"(ell,m,n)=({ell},{m},{n})")
        made += 1

    if made == 0:
        ax.text(
            0.5, 0.5,
            "No modes passed fit filters.\nTry lowering thm_floor or loosening chi_mins.",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Sliding-window scaling test (no valid fits)")
    else:
        ax.set_title("Sliding-window scaling test toward extremality")

    # IMPORTANT: Avoid LaTeX-only symbols like \\ge to prevent mathtext parse errors
    ax.set_xlabel("chi_min  (fit window: chi >= chi_min)")
    ax.set_ylabel("fitted exponent alpha in  |Im(ωM)| ~ (T_H M)^alpha")

    if made > 0:
        ax.legend()

    cfg.savefig(fig, "alpha_vs_chimin.png")
    print("Saved:", os.path.join(outdir, "alpha_vs_chimin.png"))
    print("Modes plotted:", made)
    print("THM floor used:", thm_floor)


if __name__ == "__main__":
    main()