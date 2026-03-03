#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnm_R0_extrapolate.py

Skeptic-facing diagnostic:
  Estimate R0 := lim_{Δ->0} R(Δ), where R(Δ)=|Im(ωM)|/(T_H M), Δ=1-chi^2,
  and test consistency with 2π.

Method:
  1) Build pooled near-extremal dataset (Δ<=delta_max, default 0.19)
  2) Bin in logΔ into nbins (default 5)
  3) Use binned medians as robust summary points (Δ_med, R_med)
  4) Fit correction model R(Δ) = R0 + a Δ^γ (γ>0) with:
       - grid search over γ in [γ_min, γ_max]
       - for each γ, solve linear least squares for (R0, a)
       - choose γ minimizing SSE on binned medians
  5) Bootstrap:
       resample pooled points with replacement, re-bin, refit -> distribution of R0

Outputs:
  - figures/qnm_ratio_collapse/ratio_extrapolate_R0.png
  - figures/qnm_ratio_collapse/ratio_extrapolate_R0.json
  - figures/qnm_ratio_collapse/ratio_extrapolate_R0_one_liner.txt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def kerr_thm_from_chi(chi: np.ndarray) -> np.ndarray:
    delta = 1.0 - np.asarray(chi, dtype=float) ** 2
    delta = np.clip(delta, 0.0, None)
    s = np.sqrt(delta)
    return s / (4.0 * math.pi * (1.0 + s))


def logspace_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not (xmin > 0 and xmax > 0 and xmax > xmin):
        raise ValueError(f"Invalid range for log bins: xmin={xmin}, xmax={xmax}")
    return np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)


def percentile_band(x: np.ndarray, p_lo: float = 16.0, p_hi: float = 84.0) -> Tuple[float, float, float]:
    med = float(np.nanmedian(x))
    lo = float(np.nanpercentile(x, p_lo))
    hi = float(np.nanpercentile(x, p_hi))
    return med, lo, hi


def bin_medians(D: np.ndarray, R: np.ndarray, nbins: int) -> pd.DataFrame:
    edges = logspace_edges(D, nbins)
    rows: List[Dict[str, float]] = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (D >= lo) & (D <= hi if i == nbins - 1 else D < hi)
        if not np.any(m):
            continue
        d_med = float(np.nanmedian(D[m]))
        r_med, r16, r84 = percentile_band(R[m], 16.0, 84.0)
        rows.append(
            dict(
                bin=i,
                Delta_lo=float(lo),
                Delta_hi=float(hi),
                Delta_med=d_med,
                N=int(np.sum(m)),
                R_median=r_med,
                R_p16=r16,
                R_p84=r84,
            )
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("Delta_med").reset_index(drop=True)
    return out


def fit_R0_a_for_gamma(d: np.ndarray, r: np.ndarray, gamma: float) -> Tuple[float, float, float]:
    """
    Fit r ≈ R0 + a d^gamma by linear least squares at fixed gamma.
    Returns (R0, a, sse).
    """
    x = d ** gamma
    A = np.vstack([np.ones_like(x), x]).T  # [1, d^gamma]
    # Solve min ||A theta - r||^2
    theta, *_ = np.linalg.lstsq(A, r, rcond=None)
    R0, a = float(theta[0]), float(theta[1])
    resid = (A @ theta) - r
    sse = float(np.sum(resid ** 2))
    return R0, a, sse


def grid_fit(d: np.ndarray, r: np.ndarray, gamma_min: float, gamma_max: float, gamma_steps: int) -> Dict[str, float]:
    gammas = np.linspace(gamma_min, gamma_max, gamma_steps)
    best = dict(sse=float("inf"), gamma=float("nan"), R0=float("nan"), a=float("nan"))
    for g in gammas:
        R0, a, sse = fit_R0_a_for_gamma(d, r, float(g))
        if sse < best["sse"]:
            best = dict(sse=sse, gamma=float(g), R0=R0, a=a)
    return best


def bootstrap_R0(D: np.ndarray, R: np.ndarray, nbins: int, gamma_min: float, gamma_max: float, gamma_steps: int,
                 nboot: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(D)
    out = np.empty(nboot, dtype=float)
    idx = np.arange(n)
    for i in range(nboot):
        j = rng.choice(idx, size=n, replace=True)
        Db = D[j]
        Rb = R[j]
        # bin medians
        b = bin_medians(Db, Rb, nbins)
        if len(b) < 2:
            out[i] = np.nan
            continue
        d = b["Delta_med"].to_numpy(dtype=float)
        r = b["R_median"].to_numpy(dtype=float)
        best = grid_fit(d, r, gamma_min, gamma_max, gamma_steps)
        out[i] = best["R0"]
    return out[np.isfinite(out)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrapolate R0 = lim_{Δ->0} R(Δ) and compare to 2π.")
    p.add_argument("--qnm_csv", type=str, default=str(Path("data") / "qnm" / "qnm_kerr_bundle.csv"))
    p.add_argument("--outdir", type=str, default=str(Path("figures") / "qnm_ratio_collapse"))
    p.add_argument("--delta_max", type=float, default=0.19)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--gamma_min", type=float, default=0.1)
    p.add_argument("--gamma_max", type=float, default=2.0)
    p.add_argument("--gamma_steps", type=int, default=200)
    p.add_argument("--nboot", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qnm_csv = Path(args.qnm_csv)
    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    df = pd.read_csv(qnm_csv)
    for col in ("chi", "Imw"):
        if col not in df.columns:
            raise SystemExit(f"[error] missing column '{col}' in {qnm_csv}")

    chi = df["chi"].to_numpy(dtype=float)
    Delta = 1.0 - chi**2
    THM = kerr_thm_from_chi(chi)
    R = np.abs(df["Imw"].to_numpy(dtype=float)) / THM

    m = (Delta > 0) & (Delta <= float(args.delta_max)) & np.isfinite(R) & (R > 0) & np.isfinite(Delta)
    Dp = Delta[m]
    Rp = R[m]

    # Bin medians
    b = bin_medians(Dp, Rp, int(args.bins))
    d = b["Delta_med"].to_numpy(dtype=float)
    r = b["R_median"].to_numpy(dtype=float)

    best = grid_fit(d, r, float(args.gamma_min), float(args.gamma_max), int(args.gamma_steps))
    R0_hat, a_hat, g_hat, sse_hat = best["R0"], best["a"], best["gamma"], best["sse"]

    # Bootstrap CI for R0
    R0_boot = bootstrap_R0(Dp, Rp, int(args.bins),
                           float(args.gamma_min), float(args.gamma_max), int(args.gamma_steps),
                           int(args.nboot), seed=2026)
    lo = float(np.quantile(R0_boot, 0.025)) if len(R0_boot) else float("nan")
    hi = float(np.quantile(R0_boot, 0.975)) if len(R0_boot) else float("nan")
    med = float(np.median(R0_boot)) if len(R0_boot) else float("nan")

    two_pi = 2.0 * math.pi
    # "sigma" style distance using bootstrap std (rough)
    sigma = float(np.std(R0_boot, ddof=1)) if len(R0_boot) > 2 else float("nan")
    z = (R0_hat - two_pi) / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")

    # Save JSON
    out_json = outdir / "ratio_extrapolate_R0.json"
    payload = {
        "qnm_csv": str(qnm_csv),
        "delta_max": float(args.delta_max),
        "bins": int(args.bins),
        "pooled_N": int(len(Dp)),
        "fit_model": "R(Δ) = R0 + a Δ^γ (grid γ, linear LS for R0,a)",
        "best_fit": {"R0": R0_hat, "a": a_hat, "gamma": g_hat, "sse": sse_hat},
        "bootstrap": {"nboot": int(args.nboot), "R0_median": med, "R0_ci95": [lo, hi], "R0_std": sigma},
        "compare_to_2pi": {"two_pi": two_pi, "z_score_approx": z},
        "binned": b.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # One-liner
    out_txt = outdir / "ratio_extrapolate_R0_one_liner.txt"
    one_liner = (
        f"Extrapolation R(Δ)=|Im(ωM)|/(T_H M) → R0 using R=R0+aΔ^γ over Δ≤{args.delta_max:g}: "
        f"best-fit R0={R0_hat:.3f} (bootstrap 95% CI [{lo:.3f},{hi:.3f}]); "
        f"2π={two_pi:.3f}, (R0-2π)/σ≈{z:.2f}."
    )
    out_txt.write_text(one_liner + "\n", encoding="utf-8")

    # Plot
    out_png = outdir / "ratio_extrapolate_R0.png"
    # best-fit curve on a dense grid within binned Δ-range
    dd = np.logspace(np.log10(np.min(d)), np.log10(np.max(d)), 300)
    rr = R0_hat + a_hat * (dd ** g_hat)

    plt.figure(figsize=(7.2, 4.3))
    plt.xscale("log")
    plt.yscale("log")

    # binned median + band
    plt.fill_between(d, b["R_p16"].to_numpy(dtype=float), b["R_p84"].to_numpy(dtype=float), alpha=0.25)
    plt.plot(d, r, linewidth=2.0)

    # fit curve
    plt.plot(dd, rr, linewidth=2.0)

    # 2π reference
    plt.axhline(two_pi, linewidth=2.0)

    plt.xlabel(r"$\Delta = 1-\chi^2$")
    plt.ylabel(r"$R(\Delta)=|\mathrm{Im}(\omega M)|/(T_H M)$")
    plt.title(rf"Extrapolation to $R_0$: best $R_0={R0_hat:.3f}$, $\gamma={g_hat:.3f}$ (Δ≤{args.delta_max:g})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_txt}")
    print("")
    print(one_liner)


if __name__ == "__main__":
    main()