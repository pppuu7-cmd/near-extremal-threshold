#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "qnm"
FIG_DIR = REPO_ROOT / "figures" / "qnm_absolute_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "qnm_kerr_bundle.csv"
ENRICHED_CSV = DATA_DIR / "qnm_kerr_bundle_enriched.csv"


def choose_input_csv(explicit: Path | None = None) -> tuple[Path, str]:
    if explicit is not None:
        return explicit, "explicit"
    if ENRICHED_CSV.exists():
        return ENRICHED_CSV, "enriched"
    return RAW_CSV, "raw"


def fit_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lx = np.log10(x)
    ly = np.log10(y)
    rho, _ = np.polyfit(lx, ly, 1)
    r = float(np.corrcoef(lx, ly)[0, 1]) if len(lx) >= 2 else float("nan")
    return float(rho), r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Optional explicit CSV path.")
    ap.add_argument("--delta-max", type=float, default=0.19, help="Restrict to Delta <= delta-max.")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum points per family to fit.")
    ap.add_argument("--write-one-liner", action="store_true", help="Write abstract_one_liner.txt.")
    args = ap.parse_args()

    csv_path, mode = choose_input_csv(Path(args.csv) if args.csv else None)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Delta", "THM", "absImwM", "ell", "m", "n", "source"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CSV missing required columns {sorted(missing)}. "
            f"Run analysis/enrich_qnm_bundle.py to generate the enriched CSV."
        )

    dmax = float(args.delta_max)
    dff = df[(df["Delta"] <= dmax) & np.isfinite(df["THM"]) & np.isfinite(df["absImwM"])].copy()
    dff = dff[(dff["THM"] > 0) & (dff["absImwM"] > 0)]

    fam_cols = ["ell", "m", "n", "source"]
    rows = []
    for (ell, m, n, source), g in dff.groupby(fam_cols, dropna=False):
        if len(g) < args.min_n:
            continue
        x = g["THM"].astype(float).to_numpy()
        y = g["absImwM"].astype(float).to_numpy()
        rho, rlog = fit_loglog(x, y)
        rows.append(
            {
                "ell": int(ell),
                "m": int(m),
                "n": int(n),
                "source": str(source),
                "N": int(len(g)),
                "delta_max": dmax,
                "rho": rho,
                "r_loglog": rlog,
                "r2_loglog": (rlog * rlog) if np.isfinite(rlog) else float("nan"),
            }
        )

    if not rows:
        # Provide a helpful diagnostic instead of crashing
        # Count max points per family after filtering
        counts = dff.groupby(fam_cols, dropna=False).size().sort_values(ascending=False)
        max_n = int(counts.iloc[0]) if len(counts) else 0
        msg = (
            f"No families have >= min_n={args.min_n} points after filtering Delta<= {dmax}.\n"
            f"Max points in any family after filtering: {max_n}.\n"
            f"Fix: re-run with smaller --min-n (e.g. 5 or 3), or increase --delta-max."
        )
        raise SystemExit(msg)

    out = pd.DataFrame(rows).sort_values(["ell", "m", "n", "source"]).reset_index(drop=True)

    # Plot
    labels = [f"({r.ell},{r.m},{r.n})" for r in out.itertuples(index=False)]
    rho_vals = out["rho"].to_numpy()

    plt.figure(figsize=(9, 4.5))
    plt.plot(range(len(rho_vals)), rho_vals, marker="o", linestyle="None")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("rho in |Im(wM)| ∝ (T_H M)^rho")
    plt.title(f"Per-family exponent over Delta <= {dmax} (CSV={mode})")
    plt.gcf().set_constrained_layout(True)

    out_png = FIG_DIR / "rho_by_family.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    rho_mean = float(np.mean(rho_vals))
    rho_std = float(np.std(rho_vals, ddof=1)) if len(rho_vals) > 1 else 0.0
    rho_min = float(np.min(rho_vals))
    rho_max = float(np.max(rho_vals))

    total_N = int(out["N"].sum())
    num_fam = len(out)

    print(f"[ok] wrote {out_png}")
    print(
        f"Across {num_fam} Kerr QNM families (total N={total_N}), "
        f"fit over Delta<= {dmax}: rho_mean={rho_mean:.6f} ± {rho_std:.6f} "
        f"(family spread; range [{rho_min:.6f},{rho_max:.6f}])."
    )

    if args.write_one_liner:
        one_liner = (
            f"Across {num_fam} Kerr QNM families (total N={total_N}), "
            f"we fit |Im(ωM)| ∝ (T_H M)^ρ over Δ≤{dmax:.2f}, obtaining "
            f"ρ̄={rho_mean:.6f}±{rho_std:.6f} (family spread; range [{rho_min:.6f},{rho_max:.6f}])."
        )
        out_txt = FIG_DIR / "abstract_one_liner.txt"
        out_txt.write_text(one_liner + "\n", encoding="utf-8")
        print(f"[ok] wrote {out_txt}")


if __name__ == "__main__":
    main()