#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_qnm_abs_scaling_table.py

Reads figures/qnm_absolute_tables/qnm_abs_scaling_summary.csv
and writes a LaTeX table (figures/qnm_absolute_tables/qnm_abs_scaling_summary.tex).

Compatibility fix:
- If column 'alpha' is missing, we set alpha := rho.
  (Historically alpha duplicated rho in your tables; we keep the pipeline robust.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures" / "qnm_absolute_tables"
CSV_PATH = FIG_DIR / "qnm_abs_scaling_summary.csv"
OUT_TEX = FIG_DIR / "qnm_abs_scaling_summary.tex"


REQUIRED_BASE = {
    "s",
    "ell",
    "m",
    "n",
    "source",
    "N",
    "delta_max",
    "rho",
    "A",
    "r_loglog",
    "r2_loglog",
}

# Optional columns (we may include them if present)
OPTIONAL = {
    "intercept_log10A",
    "alpha",  # may be missing; we will synthesize from rho
}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_BASE - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    out = df.copy()

    # Backward compatibility: alpha
    if "alpha" not in out.columns:
        out["alpha"] = out["rho"]

    # Ensure numeric columns are numeric (robust formatting)
    for c in ["rho", "alpha", "A", "r_loglog", "r2_loglog", "delta_max"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["N"] = pd.to_numeric(out["N"], errors="coerce").fillna(0).astype(int)
    out["s"] = pd.to_numeric(out["s"], errors="coerce").fillna(-2).astype(int)
    out["ell"] = pd.to_numeric(out["ell"], errors="coerce").astype(int)
    out["m"] = pd.to_numeric(out["m"], errors="coerce").astype(int)
    out["n"] = pd.to_numeric(out["n"], errors="coerce").astype(int)

    return out


def _fmt(x: float, nd: int = 6) -> str:
    if x is None or pd.isna(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def make_table_tex(df: pd.DataFrame, include_optional: bool = True) -> str:
    cols = list(REQUIRED_BASE | {"alpha"})
    # Control order
    order = [
        "s", "ell", "m", "n",
        "N", "delta_max",
        "rho", "alpha", "A",
        "r_loglog", "r2_loglog",
        "source",
    ]

    for c in order:
        if c not in df.columns:
            raise ValueError(f"Internal error: expected column {c} after normalization")

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ccccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"$s$ & $\ell$ & $m$ & $n$ & $N$ & $\Delta_{\max}$ & $\rho$ & $\alpha$ & $A$ & $r_{\log\log}$ & $R^2$ \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        lines.append(
            rf"{int(r['s'])} & {int(r['ell'])} & {int(r['m'])} & {int(r['n'])} & "
            rf"{int(r['N'])} & {_fmt(r['delta_max'], 2)} & "
            rf"{_fmt(r['rho'], 6)} & {_fmt(r['alpha'], 6)} & "
            rf"{(f'{float(r['A']):.6g}' if pd.notna(r['A']) else 'nan')} & "
            rf"{_fmt(r['r_loglog'], 6)} & {_fmt(r['r2_loglog'], 6)} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Absolute scaling fit for Kerr QNMs over the restricted near-extremal window "
        r"$\Delta \le \Delta_{\max}$: $|{\rm Im}(\omega M)| \approx A\,(T_H M)^{\rho}$. "
        r"Here $r_{\log\log}$ is the Pearson correlation coefficient in log space (not an exponent). "
        r"The auxiliary exponent $\alpha$ is included for backward compatibility; if not independently fit, we set $\alpha=\rho$.}"
    )
    lines.append(r"\label{tab:qnm_abs_scaling}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV_PATH), help="Input summary CSV.")
    ap.add_argument("--out-tex", type=str, default=str(OUT_TEX), help="Output LaTeX file.")
    ap.add_argument("--no-optional", action="store_true", help="Ignore optional columns.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_tex = Path(args.out_tex)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)

    # Stable sort
    df = df.sort_values(["ell", "m", "n", "source"]).reset_index(drop=True)

    tex = make_table_tex(df, include_optional=not args.no_optional)
    out_tex.write_text(tex, encoding="utf-8")

    print(f"[ok] wrote {out_tex}")


if __name__ == "__main__":
    main()