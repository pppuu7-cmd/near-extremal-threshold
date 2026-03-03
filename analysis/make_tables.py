from __future__ import annotations

import os
import sys
import csv as csvlib
import numpy as np

# allow running without install
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nearextremal_threshold.data.qnm_loader import load_qnm_csv, filter_modes
from nearextremal_threshold.geometry.kerr import KerrParams


def powerlaw_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y ~ A x^alpha in log-log via least squares.
    Returns (A, alpha).
    """
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        raise ValueError("Not enough points for fit.")
    X = np.log(x)
    Y = np.log(y)
    alpha, logA = np.polyfit(X, Y, 1)
    A = np.exp(logA)
    return A, alpha


def write_latex_table(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{cccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"$\ell$ & $m$ & $n$ & $A$ & $\alpha$ & points \\" + "\n")
        f.write(r"\hline" + "\n")
        for r in rows:
            f.write(f"{r['ell']} & {r['m']} & {r['n']} & {r['A']:.3e} & {r['alpha']:.3f} & {r['N']} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")


def main():
    csv_path = os.path.join(ROOT, "data", "qnm", "qnm_template.csv")  # replace
    outdir = os.path.join(ROOT, "figures", "qnm", "tables")
    os.makedirs(outdir, exist_ok=True)

    rows = load_qnm_csv(csv_path)

    # choose a few mode families to summarize
    families = [(2,2,0), (2,2,1), (3,3,0)]

    summary_rows: list[dict] = []
    for ell, m, n in families:
        sub = filter_modes(rows, ell=ell, m=m, n=n)
        if len(sub) < 5:
            continue
        chis = np.array([r.chi for r in sub], dtype=float)
        Imw = np.abs(np.array([r.Imw for r in sub], dtype=float))

        THM = np.array([KerrParams(M=1.0, chi=chi).T_H() for chi in chis], dtype=float)

        A, alpha = powerlaw_fit(THM, Imw)
        summary_rows.append({"ell": ell, "m": m, "n": n, "A": A, "alpha": alpha, "N": len(sub)})

    # write CSV summary
    csv_out = os.path.join(outdir, "qnm_scaling_summary.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csvlib.DictWriter(f, fieldnames=["ell","m","n","A","alpha","N"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # write LaTeX table
    tex_out = os.path.join(outdir, "qnm_scaling_summary.tex")
    write_latex_table(tex_out, summary_rows)

    print("Saved:", csv_out)
    print("Saved:", tex_out)
    if not summary_rows:
        print("No families had enough points; adjust families or provide more data.")


if __name__ == "__main__":
    main()