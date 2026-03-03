#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qnm_ratio_collapse_table.py

Out-of-the-box:
  python analysis/qnm_ratio_collapse_table.py

Reads:
  figures/qnm_ratio_collapse/ratio_collapse_binned.csv

Accepts BOTH schemas:

Schema A (expected):
  bin, Delta_lo, Delta_hi, Delta_med, N, R_median, R_p16, R_p84

Schema B (legacy/alt):
  bin, bin_lo, bin_hi, (Delta_med optional), N, median, p16, p84
  (and similar close variants)

Writes:
  figures/qnm_ratio_collapse/ratio_collapse_binned_table.tex
  figures/qnm_ratio_collapse/ratio_collapse_caption.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize df to:
      bin, Delta_lo, Delta_hi, Delta_med, N, R_median, R_p16, R_p84
    """
    out = df.copy()

    # bin
    if "bin" not in out.columns:
        c = _pick(out, ["Bin", "BIN", "b"])
        if c is None:
            out["bin"] = np.arange(len(out), dtype=int)
        else:
            out["bin"] = out[c].astype(int)

    # Delta_lo / Delta_hi
    if "Delta_lo" not in out.columns:
        c = _pick(out, ["bin_lo", "DeltaLow", "delta_lo", "lo", "Delta_min"])
        if c is None:
            raise SystemExit("[error] binned CSV missing Delta_lo (or bin_lo/lo/Delta_min)")
        out["Delta_lo"] = out[c].astype(float)

    if "Delta_hi" not in out.columns:
        c = _pick(out, ["bin_hi", "DeltaHigh", "delta_hi", "hi", "Delta_max"])
        if c is None:
            raise SystemExit("[error] binned CSV missing Delta_hi (or bin_hi/hi/Delta_max)")
        out["Delta_hi"] = out[c].astype(float)

    # Delta_med
    if "Delta_med" not in out.columns:
        c = _pick(out, ["Delta_mid", "delta_med", "mid", "Delta_center"])
        if c is not None:
            out["Delta_med"] = out[c].astype(float)
        else:
            lo = out["Delta_lo"].to_numpy(dtype=float)
            hi = out["Delta_hi"].to_numpy(dtype=float)
            med = np.where((lo > 0) & (hi > 0), np.sqrt(lo * hi), 0.5 * (lo + hi))
            out["Delta_med"] = med

    # N
    if "N" not in out.columns:
        c = _pick(out, ["count", "n", "Nsamples", "N_bin"])
        if c is None:
            raise SystemExit("[error] binned CSV missing N (or count/n/Nsamples)")
        out["N"] = out[c].astype(int)

    # R stats
    if "R_median" not in out.columns:
        c = _pick(out, ["median", "R_med", "R50", "p50"])
        if c is None:
            raise SystemExit("[error] binned CSV missing R_median (or median/R_med/p50)")
        out["R_median"] = out[c].astype(float)

    if "R_p16" not in out.columns:
        c = _pick(out, ["p16", "R16", "R_p16", "q16"])
        if c is None:
            raise SystemExit("[error] binned CSV missing R_p16 (or p16/R16/q16)")
        out["R_p16"] = out[c].astype(float)

    if "R_p84" not in out.columns:
        c = _pick(out, ["p84", "R84", "R_p84", "q84"])
        if c is None:
            raise SystemExit("[error] binned CSV missing R_p84 (or p84/R84/q84)")
        out["R_p84"] = out[c].astype(float)

    keep = ["bin", "Delta_lo", "Delta_hi", "Delta_med", "N", "R_median", "R_p16", "R_p84"]
    return out[keep].sort_values("bin").reset_index(drop=True)


def _fmt(x: float, digits: int = 6) -> str:
    if not np.isfinite(x):
        return "nan"
    # scientific for tiny, else fixed
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-4 or ax >= 1e4):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def main() -> None:
    root = repo_root_from_here()
    in_csv = root / "figures" / "qnm_ratio_collapse" / "ratio_collapse_binned.csv"
    out_tex = root / "figures" / "qnm_ratio_collapse" / "ratio_collapse_binned_table.tex"
    out_cap = root / "figures" / "qnm_ratio_collapse" / "ratio_collapse_caption.txt"

    if not in_csv.exists():
        raise SystemExit(f"[error] missing input: {in_csv}")

    df = pd.read_csv(in_csv)
    df = _ensure_schema(df)

    # Table (LaTeX)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(r"\hline")
    lines.append(r"bin & $\Delta_{\rm lo}$ & $\Delta_{\rm hi}$ & $\Delta_{\rm med}$ & $N$ & $R_{\rm med}$ & $[R_{16},R_{84}]$ \\")
    lines.append(r"\hline")

    for _, r in df.iterrows():
        lines.append(
            f"{int(r['bin'])} & "
            f"{_fmt(r['Delta_lo'], 6)} & {_fmt(r['Delta_hi'], 6)} & {_fmt(r['Delta_med'], 6)} & "
            f"{int(r['N'])} & {_fmt(r['R_median'], 6)} & "
            f"[{_fmt(r['R_p16'], 6)},{_fmt(r['R_p84'], 6)}] \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Binned summary of the near-extremal collapse ratio $R(\Delta)=|\mathrm{Im}(\omega M)|/(T_H M)$ over $\Delta\le 0.19$.}")
    lines.append(r"\label{tab:ratio_collapse_binned}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Caption helper text
    pooled_N = int(df["N"].sum())
    cap = (
        f"Near-extremal collapse binned summary (pooled N={pooled_N}): "
        r"$R=|\mathrm{Im}(\omega M)|/(T_H M)$ shows weak $\Delta$-dependence in the pooled region."
    )
    out_cap.write_text(cap + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_tex}")
    print(f"[ok] wrote {out_cap}")


if __name__ == "__main__":
    main()