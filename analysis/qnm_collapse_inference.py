#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnm_collapse_inference.py

Statistical inference layer on top of the near-extremal collapse:
  R(Δ) = |Im(ωM)| / (T_H M),  Δ = 1 - chi^2,  THM = sqrt(Δ)/[4π(1+sqrt(Δ))].

Computes (for several Δ cutoffs):
  - pooled N
  - median(R) and percentile bands
  - bootstrap 95% CI for median(R)
  - log–log slope beta from log10(R)=beta log10(Δ)+c and bootstrap 95% CI for beta
  - LOFO (leave-one-family-out) stability: recompute median(R) after removing each (ell,m,n) family

Outputs:
  - figures/qnm_ratio_collapse/ratio_collapse_inference.json
  - figures/qnm_ratio_collapse/ratio_collapse_inference.tex (RevTeX-friendly)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def kerr_thm_from_chi(chi: np.ndarray) -> np.ndarray:
    delta = 1.0 - np.asarray(chi, dtype=float) ** 2
    delta = np.clip(delta, 0.0, None)
    s = np.sqrt(delta)
    return s / (4.0 * math.pi * (1.0 + s))


def percentile_band(x: np.ndarray, p_lo: float = 16.0, p_hi: float = 84.0) -> Tuple[float, float, float]:
    med = float(np.nanmedian(x))
    lo = float(np.nanpercentile(x, p_lo))
    hi = float(np.nanpercentile(x, p_hi))
    return med, lo, hi


def bootstrap_ci_median(x: np.ndarray, nboot: int = 5000, alpha: float = 0.05, seed: int = 1) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    meds = np.empty(nboot, dtype=float)
    for i in range(nboot):
        samp = rng.choice(x, size=n, replace=True)
        meds[i] = np.median(samp)
    lo = float(np.quantile(meds, alpha / 2))
    hi = float(np.quantile(meds, 1 - alpha / 2))
    return lo, hi


def fit_beta(D: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
    m = (D > 0) & (R > 0) & np.isfinite(D) & np.isfinite(R)
    D = D[m]
    R = R[m]
    if len(D) < 3:
        return float("nan"), float("nan")
    x = np.log10(D)
    y = np.log10(R)
    beta, intercept = np.polyfit(x, y, 1)
    return float(beta), float(intercept)


def bootstrap_ci_beta(D: np.ndarray, R: np.ndarray, nboot: int = 5000, alpha: float = 0.05, seed: int = 2) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = (D > 0) & (R > 0) & np.isfinite(D) & np.isfinite(R)
    D = D[m]
    R = R[m]
    n = len(D)
    if n < 3:
        return float("nan"), float("nan")
    betas = np.empty(nboot, dtype=float)
    idx = np.arange(n)
    for i in range(nboot):
        j = rng.choice(idx, size=n, replace=True)
        b, _ = fit_beta(D[j], R[j])
        betas[i] = b
    lo = float(np.quantile(betas, alpha / 2))
    hi = float(np.quantile(betas, 1 - alpha / 2))
    return lo, hi


def fmt_float(x: float, ndp: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{ndp}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap/LOFO inference for QNM ratio collapse.")
    p.add_argument("--qnm_csv", type=str, default=str(Path("data") / "qnm" / "qnm_kerr_bundle.csv"))
    p.add_argument("--outdir", type=str, default=str(Path("figures") / "qnm_ratio_collapse"))
    p.add_argument("--cuts", type=str, default="0.19,0.10,0.05,0.02")
    p.add_argument("--nboot", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qnm_csv = Path(args.qnm_csv)
    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    cuts = [float(s.strip()) for s in args.cuts.split(",") if s.strip()]
    nboot = int(args.nboot)

    df = pd.read_csv(qnm_csv)
    for col in ("chi", "Imw", "ell", "m", "n"):
        if col not in df.columns:
            raise SystemExit(f"[error] missing column '{col}' in {qnm_csv}")

    chi = df["chi"].to_numpy(dtype=float)
    Delta = 1.0 - chi**2
    THM = kerr_thm_from_chi(chi)
    R = np.abs(df["Imw"].to_numpy(dtype=float)) / THM

    df = df.copy()
    df["Delta"] = Delta
    df["R"] = R
    df["family"] = df.apply(lambda r: (int(r["ell"]), int(r["m"]), int(r["n"])), axis=1)

    out = {"qnm_csv": str(qnm_csv), "cuts": cuts, "nboot": nboot, "results": []}

    for c in cuts:
        pooled = df[(df["Delta"] > 0) & (df["Delta"] <= c) & np.isfinite(df["R"]) & (df["R"] > 0)].copy()
        Dp = pooled["Delta"].to_numpy(dtype=float)
        Rp = pooled["R"].to_numpy(dtype=float)

        med, p16, p84 = percentile_band(Rp, 16.0, 84.0)
        ci_lo, ci_hi = bootstrap_ci_median(Rp, nboot=nboot, alpha=0.05, seed=11)
        beta, intercept = fit_beta(Dp, Rp)
        b_lo, b_hi = bootstrap_ci_beta(Dp, Rp, nboot=nboot, alpha=0.05, seed=22)

        # LOFO: drop each family and recompute median(R)
        lofo = []
        fams = sorted(pooled["family"].unique().tolist())
        for fam in fams:
            sub = pooled[pooled["family"] != fam]
            rr = sub["R"].to_numpy(dtype=float)
            if len(rr) == 0:
                continue
            lofo_med = float(np.median(rr))
            lofo.append({"dropped_family": list(fam), "median_R": lofo_med})

        out["results"].append(
            dict(
                delta_max=c,
                N=int(len(pooled)),
                R_median=med,
                R_p16=p16,
                R_p84=p84,
                R_median_boot95=[ci_lo, ci_hi],
                beta=beta,
                beta_boot95=[b_lo, b_hi],
                lofo=lofo,
            )
        )

    # Write JSON
    out_json = outdir / "ratio_collapse_inference.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Write a compact RevTeX-friendly table (one row per cut)
    out_tex = outdir / "ratio_collapse_inference.tex"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ccccc}")
    lines.append(r"\hline\hline")
    lines.append(r"$\Delta_{\max}$ & $N$ & $R_{\rm med}$ & $R_{\rm med}$ (boot 95\%) & $\beta$ (boot 95\%) \\")
    lines.append(r"\hline")
    for r in out["results"]:
        c = r["delta_max"]
        N = r["N"]
        Rm = r["R_median"]
        ci = r["R_median_boot95"]
        b = r["beta"]
        bci = r["beta_boot95"]
        lines.append(
            rf"{c:g} & {N:d} & {fmt_float(Rm,3)} & [{fmt_float(ci[0],3)},{fmt_float(ci[1],3)}] & "
            rf"{fmt_float(b,4)}\,[{fmt_float(bci[0],4)},{fmt_float(bci[1],4)}] \\"
        )
    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Bootstrap inference for the near-extremal collapse ratio $R(\Delta)=|\mathrm{Im}(\omega M)|/(T_H M)$ and log--log slope $\beta$ in $R\propto \Delta^{\beta}$ over pooled samples.}")
    lines.append(r"\label{tab:ratio_collapse_inference}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Print Abstract-ready one-liner (pick Δ≤0.05 if available)
    pick = None
    for r in out["results"]:
        if abs(r["delta_max"] - 0.05) < 1e-12:
            pick = r
            break
    if pick is None:
        pick = out["results"][0]

    c = pick["delta_max"]
    N = pick["N"]
    Rm = pick["R_median"]
    ci = pick["R_median_boot95"]
    b = pick["beta"]
    bci = pick["beta_boot95"]
    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_tex}")
    print("")
    print(
        f"Collapse inference: for Δ≤{c:g} (N={N}), median R={Rm:.3f} "
        f"with bootstrap 95% CI [{ci[0]:.3f},{ci[1]:.3f}]; "
        f"log–log slope β={b:.4f} with bootstrap 95% CI [{bci[0]:.4f},{bci[1]:.4f}]."
    )


if __name__ == "__main__":
    main()