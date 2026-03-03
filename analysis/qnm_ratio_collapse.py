#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qnm_ratio_collapse.py

Out-of-the-box:
  python analysis/qnm_ratio_collapse.py

Also supports:
  python analysis/qnm_ratio_collapse.py --in data/qnm/qnm_kerr_bundle_enriched.csv
  python analysis/qnm_ratio_collapse.py --qnm_csv data/qnm/qnm_kerr_bundle_enriched.csv

Writes (default):
  figures/qnm_ratio_collapse/ratio_collapse_median_band.png
  figures/qnm_ratio_collapse/ratio_collapse_binned.csv
  figures/qnm_ratio_collapse/ratio_collapse_summary.json

Input CSV must contain at least:
  chi, ell, m, n, Rew, Imw
If enriched: also Delta, THM, absImwM, R
If not enriched: will compute Delta, THM, absImwM, R on the fly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def hawking_THM_from_chi(chi: np.ndarray) -> np.ndarray:
    """
    Kerr Hawking temperature times M (dimensionless):
      T_H * M = sqrt(1-chi^2) / [4*pi*(1 + sqrt(1-chi^2))]
    """
    chi = np.asarray(chi, dtype=float)
    Delta = 1.0 - chi**2
    s = np.sqrt(np.maximum(Delta, 0.0))
    THM = s / (4.0 * math.pi * (1.0 + s))
    return THM


def enrich_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    need = ["Delta", "THM", "absImwM", "R"]
    if all(c in df.columns for c in need):
        return df

    if "chi" not in df.columns:
        raise ValueError("[error] missing required column: chi")
    if "Imw" not in df.columns:
        raise ValueError("[error] missing required column: Imw")

    chi = df["chi"].to_numpy(dtype=float)
    Delta = 1.0 - chi**2
    THM = hawking_THM_from_chi(chi)
    absImwM = np.abs(df["Imw"].to_numpy(dtype=float))  # Im(omega M)
    R = absImwM / np.maximum(THM, 1e-300)

    out = df.copy()
    out["Delta"] = Delta
    out["THM"] = THM
    out["absImwM"] = absImwM
    out["R"] = R
    return out


def percentile_band(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return (
        float(np.nanpercentile(x, 50)),
        float(np.nanpercentile(x, 16)),
        float(np.nanpercentile(x, 84)),
    )


def loglog_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    lx = np.log10(x)
    ly = np.log10(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    beta, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = beta * lx + intercept
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "N": int(len(x)),
        "beta": float(beta),
        "intercept": float(intercept),
        "r2": float(r2),
        "delta_min": float(np.min(x)) if len(x) else float("nan"),
        "delta_max": float(np.max(x)) if len(x) else float("nan"),
    }


def make_binned(df: pd.DataFrame, delta_max: float, bins: int) -> pd.DataFrame:
    d = df[(df["Delta"] > 0) & (df["Delta"] <= delta_max)].copy()
    if len(d) == 0:
        raise ValueError(f"[error] no rows after filtering Delta<= {delta_max}")

    # log-spaced bins in Delta
    dmin = float(d["Delta"].min())
    dmax = float(delta_max)
    edges = np.logspace(np.log10(dmin), np.log10(dmax), bins + 1)

    rows: List[Dict[str, Any]] = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        sel = d[(d["Delta"] >= lo) & (d["Delta"] <= hi)]
        if len(sel) == 0:
            continue
        med, p16, p84 = percentile_band(sel["R"].to_numpy())
        dmed = float(np.sqrt(lo * hi)) if lo > 0 and hi > 0 else float(0.5 * (lo + hi))
        rows.append(
            {
                "bin": int(i),
                "Delta_lo": float(lo),
                "Delta_hi": float(hi),
                "Delta_med": dmed,
                "N": int(len(sel)),
                "R_median": med,
                "R_p16": p16,
                "R_p84": p84,
            }
        )

    out = pd.DataFrame(rows)
    return out


def plot_median_band(df_binned: pd.DataFrame, out_png: Path) -> None:
    x = df_binned["Delta_med"].to_numpy(dtype=float)
    y = df_binned["R_median"].to_numpy(dtype=float)
    y16 = df_binned["R_p16"].to_numpy(dtype=float)
    y84 = df_binned["R_p84"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, y, marker="o", linewidth=2)
    plt.fill_between(x, y16, y84, alpha=0.2)
    plt.xlabel(r"$\Delta=1-\chi^2$")
    plt.ylabel(r"$R(\Delta)=|\mathrm{Im}(\omega M)|/(T_H M)$")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    root = repo_root_from_here()

    # defaults “из коробки”
    default_in_enriched = root / "data" / "qnm" / "qnm_kerr_bundle_enriched.csv"
    default_in_raw = root / "data" / "qnm" / "qnm_kerr_bundle.csv"
    default_in = default_in_enriched if default_in_enriched.exists() else default_in_raw

    default_outdir = root / "figures" / "qnm_ratio_collapse"

    ap = argparse.ArgumentParser()
    ap.add_argument("--qnm_csv", default=str(default_in), help="input QNM csv")
    ap.add_argument("--in", dest="qnm_csv", help="alias for --qnm_csv")
    ap.add_argument("--outdir", default=str(default_outdir), help="output directory")
    ap.add_argument("--delta_max", type=float, default=0.19, help="max Delta to include")
    ap.add_argument("--bins", type=int, default=5, help="number of log-bins in Delta")
    args = ap.parse_args()

    qnm_csv = Path(args.qnm_csv)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"Using QNM CSV: {qnm_csv.resolve()}")
    df = pd.read_csv(qnm_csv)
    df = enrich_if_needed(df)

    # pooled filter for summary/fit
    pooled = df[(df["Delta"] > 0) & (df["Delta"] <= float(args.delta_max))].copy()
    pooled_N = int(len(pooled))

    df_binned = make_binned(df, delta_max=float(args.delta_max), bins=int(args.bins))

    out_png = outdir / "ratio_collapse_median_band.png"
    plot_median_band(df_binned, out_png)
    print(f"Saved: {out_png.resolve()}")
    print(f"Binned points: {len(df_binned)} Total pooled samples: {pooled_N}")

    # plateau metrics
    plateau_cuts = [0.19, 0.10, 0.05, 0.02]
    plateau_metrics: Dict[str, Any] = {}
    for cut in plateau_cuts:
        dd = df[(df["Delta"] > 0) & (df["Delta"] <= cut)]
        if len(dd) == 0:
            continue
        med, p16, p84 = percentile_band(dd["R"].to_numpy())
        plateau_metrics[f"Delta<={cut}"] = {"N": int(len(dd)), "median": med, "p16": p16, "p84": p84}

    # log-log fit on pooled
    fit = loglog_fit(pooled["Delta"].to_numpy(dtype=float), pooled["R"].to_numpy(dtype=float))

    print("\nPlateau metrics for R = |Im(wM)|/(T_H M):")
    for k, v in plateau_metrics.items():
        print(f"  {k}: N={v['N']:4d}  median={v['median']:.6g}  [p16,p84]=[{v['p16']:.6g}, {v['p84']:.6g}]")

    print("\nLog-log fit over pooled region: log10(R) = beta*log10(Delta) + intercept")
    print(f"  beta={fit['beta']:.9g}  intercept={fit['intercept']:.9g}  r2={fit['r2']:.9g}  (N={fit['N']})")
    print(f"  Delta range: [{fit['delta_min']:.9g}, {fit['delta_max']:.9g}]")

    # write binned csv with “ожидаемой” схемой
    out_binned = outdir / "ratio_collapse_binned.csv"
    df_binned.to_csv(out_binned, index=False)
    print(f"\nWrote: {out_binned.resolve()}")

    out_summary = outdir / "ratio_collapse_summary.json"
    payload = {
        "input_csv": str(qnm_csv.resolve()),
        "delta_max": float(args.delta_max),
        "bins": int(args.bins),
        "plateau_metrics": plateau_metrics,
        "loglog_fit": fit,
        "binned_points": int(len(df_binned)),
        "pooled_N": pooled_N,
    }
    out_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_summary.resolve()}")


if __name__ == "__main__":
    main()