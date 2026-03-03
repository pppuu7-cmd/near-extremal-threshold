#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "qnm"
FIG_DIR = REPO_ROOT / "figures" / "qnm_absolute_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "qnm_kerr_bundle.csv"
ENRICHED_CSV = DATA_DIR / "qnm_kerr_bundle_enriched.csv"


def one_minus_chi2(chi: float) -> float:
    """Stable 1-chi^2 using fma when available."""
    fma = getattr(math, "fma", None)
    if fma is not None:
        return fma(-chi, chi, 1.0)
    return 1.0 - chi * chi


def thm_from_chi(chi: float) -> float:
    """
    Dimensionless Hawking temperature T_H * M for Kerr (M=1):
      r_+ = 1 + s,  s = sqrt(1-chi^2) = sqrt(Delta)
      T_H M = s / (4*pi*(1+s))
    Checks: chi=0 -> 1/(8pi)=0.0397887...
    """
    Delta = one_minus_chi2(chi)
    if Delta < 0.0 and Delta > -1e-18:
        Delta = 0.0
    if Delta < 0.0:
        raise ValueError(f"Delta<0 from chi={chi}: {Delta}")
    s = math.sqrt(Delta)
    denom = 4.0 * math.pi * (1.0 + s)
    if denom == 0.0:
        return 0.0
    return s / denom


def ensure_enriched_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Delta, THM, absImwM, R if missing. Assumes input frequencies are dimensionless with M=1."""
    out = df.copy()

    if "Delta" not in out.columns:
        out["Delta"] = out["chi"].astype(float).map(one_minus_chi2)

    if "THM" not in out.columns:
        out["THM"] = out["chi"].astype(float).map(thm_from_chi)

    if "absImwM" not in out.columns:
        out["absImwM"] = out["Imw"].astype(float).abs()  # M=1

    if "R" not in out.columns:
        thm = out["THM"].astype(float).to_numpy()
        num = out["absImwM"].astype(float).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            R = num / thm
        out["R"] = R

    return out


def choose_input_csv(explicit: Path | None = None) -> Tuple[Path, str]:
    """Prefer enriched unless explicit path is given."""
    if explicit is not None:
        return explicit, "explicit"
    if ENRICHED_CSV.exists():
        return ENRICHED_CSV, "enriched"
    return RAW_CSV, "raw"


def fit_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit log10(y) = rho * log10(x) + intercept.
    Returns (rho, intercept, r_loglog).
    """
    lx = np.log10(x)
    ly = np.log10(y)

    # slope/intercept by polyfit
    rho, intercept = np.polyfit(lx, ly, 1)

    # Pearson r in log-log
    r = float(np.corrcoef(lx, ly)[0, 1]) if len(lx) >= 2 else float("nan")
    return float(rho), float(intercept), r


def to_latex_table(summary: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"$(s,\ell,m,n)$ & $N$ & $\Delta_{\max}$ & $\rho$ & $A$ & $r_{\log\log}$ & $R^2$ \\")
    lines.append(r"\midrule")

    for _, row in summary.iterrows():
        s = int(row["s"])
        ell = int(row["ell"])
        m = int(row["m"])
        n = int(row["n"])
        N = int(row["N"])
        dmax = row["delta_max"]
        rho = row["rho"]
        A = row["A"]
        rlog = row["r_loglog"]
        r2 = row["r2_loglog"]

        fam = f"({s},{ell},{m},{n})"
        lines.append(
            rf"{fam} & {N} & {dmax:.2f} & {rho:.6f} & {A:.6g} & {rlog:.6f} & {r2:.6f} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Absolute scaling fit on Kerr QNMs: $|{\rm Im}(\omega M)| \approx A\,(T_H M)^{\rho}$ "
        r"performed in log--log over the restricted near-extremal range $\Delta\le\Delta_{\max}$. "
        r"Here $r_{\log\log}$ is the Pearson correlation coefficient in log space (not the exponent).}"
    )
    lines.append(r"\label{tab:qnm_abs_scaling}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Optional explicit CSV path.")
    ap.add_argument("--delta-max", type=float, default=0.19, help="Fit only points with Delta <= delta-max.")
    ap.add_argument("--spin-weight", type=int, default=-2, help="Spin weight s (metadata for table).")
    args = ap.parse_args()

    csv_path, mode = choose_input_csv(Path(args.csv) if args.csv else None)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Using QNM CSV: {csv_path} (mode={mode})")
    df = pd.read_csv(csv_path)
    df = ensure_enriched_columns(df)

    # Filter fit region
    dmax = float(args.delta_max)
    df_fit = df[(df["Delta"] <= dmax) & np.isfinite(df["THM"]) & np.isfinite(df["absImwM"])].copy()
    df_fit = df_fit[(df_fit["THM"] > 0) & (df_fit["absImwM"] > 0)]

    # Define families: by (ell,m,n,source)
    fam_cols = ["ell", "m", "n", "source"]
    groups = df_fit.groupby(fam_cols, dropna=False)

    rows = []
    for (ell, m, n, source), g in groups:
        x = g["THM"].astype(float).to_numpy()
        y = g["absImwM"].astype(float).to_numpy()
        if len(x) < 5:
            continue

        rho, intercept, rlog = fit_loglog(x, y)
        A = 10.0 ** intercept
        r2 = rlog * rlog if np.isfinite(rlog) else float("nan")

        rows.append(
            {
                "s": int(args.spin_weight),
                "ell": int(ell),
                "m": int(m),
                "n": int(n),
                "source": str(source),
                "N": int(len(g)),
                "delta_max": dmax,
                "rho": rho,
                "intercept_log10A": intercept,
                "A": A,
                "r_loglog": rlog,
                "r2_loglog": r2,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No family had enough points after filtering. Try increasing --delta-max.")

    # Sort
    summary = summary.sort_values(["ell", "m", "n", "source"]).reset_index(drop=True)

    out_csv = FIG_DIR / "qnm_abs_scaling_summary.csv"
    out_tex = FIG_DIR / "qnm_abs_scaling_summary.tex"

    summary.to_csv(out_csv, index=False)
    out_tex.write_text(to_latex_table(summary), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_tex}")
    print(f"Rows in summary: {len(summary)}")


if __name__ == "__main__":
    main()