#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
davies_gap_aggregate.py

Out-of-the-box:
  python analysis/davies_gap_aggregate.py

Reads:
  figures/qnm_markov_gap/gap_scan_index.csv

Writes:
  figures/qnm_markov_gap/gap_scan_slope_hist_ohmic_like.png
  figures/qnm_markov_gap/gap_scan_slope_hist_super_p3.png
  figures/qnm_markov_gap/gap_scan_summary_ohmic_like.json
  figures/qnm_markov_gap/gap_scan_summary_super_p3.json
  figures/qnm_markov_gap/gap_scan_slope_table_ohmic_like.tex
  figures/qnm_markov_gap/gap_scan_slope_table_super_p3.tex
  figures/qnm_markov_gap/gap_scan_slope_by_tag_table.tex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def percentile_band(x: np.ndarray, p_lo: float = 16.0, p_hi: float = 84.0) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.median(x)), float(np.percentile(x, p_lo)), float(np.percentile(x, p_hi))


def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
    slopes = df["slope_loglog"].to_numpy(dtype=float)
    r2s = df["r2"].to_numpy(dtype=float)

    med, p16, p84 = percentile_band(slopes) if len(slopes) else (float("nan"), float("nan"), float("nan"))
    out: Dict[str, Any] = {
        "N": int(len(df)),
        "slope": {
            "mean": float(np.mean(slopes)) if len(slopes) else float("nan"),
            "std": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else float("nan"),
            "median": med,
            "p16": p16,
            "p84": p84,
            "min": float(np.min(slopes)) if len(slopes) else float("nan"),
            "max": float(np.max(slopes)) if len(slopes) else float("nan"),
            "unique": int(len(np.unique(slopes))) if len(slopes) else 0,
        },
        "r2": {
            "mean": float(np.mean(r2s)) if len(r2s) else float("nan"),
            "median": float(np.median(r2s)) if len(r2s) else float("nan"),
            "min": float(np.min(r2s)) if len(r2s) else float("nan"),
            "max": float(np.max(r2s)) if len(r2s) else float("nan"),
        },
    }
    return out


def _safe_hist(slopes: np.ndarray, bins: int) -> Tuple[int, Tuple[float, float] | None]:
    slopes = np.asarray(slopes, dtype=float)
    if len(slopes) == 0:
        return 1, (0.0, 1.0)

    uniq = np.unique(slopes)
    if len(uniq) == 1:
        v = float(uniq[0])
        eps = 0.05 if abs(v) < 1.0 else 0.01 * abs(v)
        return 1, (v - eps, v + eps)

    bins_eff = int(bins)
    bins_eff = max(1, min(bins_eff, max(2, len(uniq))))
    return bins_eff, None


def _write_hist_png(df: pd.DataFrame, out_png: Path, title: str, bins: int) -> None:
    slopes = df["slope_loglog"].to_numpy(dtype=float)
    summ = _summarize(df)
    mu = summ["slope"]["mean"]
    med = summ["slope"]["median"]

    bins_eff, hist_range = _safe_hist(slopes, bins=bins)

    plt.figure(figsize=(7.2, 4.3))
    if hist_range is None:
        plt.hist(slopes, bins=bins_eff)
    else:
        plt.hist(slopes, bins=bins_eff, range=hist_range)

    plt.axvline(mu)
    plt.axvline(med)

    plt.xlabel("log–log slope of gap vs T")
    plt.ylabel("count")
    extra = " (all equal)" if summ["slope"]["unique"] == 1 else ""
    plt.title(f"{title}: mean={mu:.4f}, median={med:.4f}, N={len(slopes)}{extra}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _latex_escape(s: str) -> str:
    # минимально, чтобы не ломать таблицу
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
    )


def _format_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        # аккуратно: наклоны и r2 как в логах
        return f"{float(v):.6f}"
    return str(v)


def _write_latex_table_manual(
    df: pd.DataFrame,
    out_tex: Path,
    caption: str,
    label: str,
    col_order: Optional[List[str]] = None,
    col_align: Optional[str] = None,
) -> None:
    df2 = df.copy()

    if col_order is not None:
        cols = [c for c in col_order if c in df2.columns]
        rest = [c for c in df2.columns if c not in cols]
        df2 = df2[cols + rest]

    cols = list(df2.columns)
    if col_align is None:
        # l для текстовых, r для чисел
        aligns = []
        for c in cols:
            if pd.api.types.is_numeric_dtype(df2[c]):
                aligns.append("r")
            else:
                aligns.append("l")
        col_align = "".join(aligns)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\hline")
    # header
    header = " & ".join(_latex_escape(str(c)) for c in cols) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    # rows
    for _, row in df2.iterrows():
        cells = []
        for c in cols:
            cell = _format_cell(row[c])
            cell = _latex_escape(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_by_tag_table(df: pd.DataFrame, out_tex: Path) -> None:
    out_rows = []
    for tag, g in df.groupby("tag"):
        slopes = g["slope_loglog"].to_numpy(dtype=float)
        med, p16, p84 = percentile_band(slopes)
        out_rows.append(
            {
                "tag": str(tag),
                "N": int(len(g)),
                "median": med,
                "p16": p16,
                "p84": p84,
                "min": float(np.min(slopes)),
                "max": float(np.max(slopes)),
            }
        )
    tab = pd.DataFrame(out_rows).sort_values(["tag"]).reset_index(drop=True)
    _write_latex_table_manual(
        tab,
        out_tex,
        caption="Davies-gap scan: log--log slope summary by tag.",
        label="tab:gap_scan_slope_by_tag",
        col_order=["tag", "N", "median", "p16", "p84", "min", "max"],
        col_align="lrrrrrr",
    )


def main() -> None:
    root = repo_root_from_here()
    out_base = root / "figures" / "qnm_markov_gap"
    ensure_dir(out_base)

    index_path = out_base / "gap_scan_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index CSV: {index_path}. Run analysis/davies_gap_scan.py first.")

    df = pd.read_csv(index_path)
    need_cols = {"tag", "slope_loglog", "r2"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"gap_scan_index.csv must contain columns: {sorted(list(need_cols))}")

    is_super = df["tag"].astype(str).str.contains("super_p3", regex=False)
    super_p3 = df[is_super].copy()
    ohmic_like = df[~is_super].copy()

    summ_ohmic = _summarize(ohmic_like)
    summ_super = _summarize(super_p3)

    print(f"[ok] read {index_path} (N_total={len(df)})")
    print(f"[ok] ohmic_like: N={summ_ohmic['N']} slope_median={summ_ohmic['slope']['median']:.4f} [{summ_ohmic['slope']['p16']:.4f},{summ_ohmic['slope']['p84']:.4f}]")
    print(f"[ok] super_p3  : N={summ_super['N']} slope_median={summ_super['slope']['median']:.4f} [{summ_super['slope']['p16']:.4f},{summ_super['slope']['p84']:.4f}]")

    png1 = out_base / "gap_scan_slope_hist_ohmic_like.png"
    png2 = out_base / "gap_scan_slope_hist_super_p3.png"
    _write_hist_png(ohmic_like, png1, "Ohmic-like (ohmic+drude)", bins=10)
    _write_hist_png(super_p3, png2, "Super-Ohmic p=3 (control)", bins=10)

    j1 = out_base / "gap_scan_summary_ohmic_like.json"
    j2 = out_base / "gap_scan_summary_super_p3.json"
    j1.write_text(json.dumps(summ_ohmic, indent=2), encoding="utf-8")
    j2.write_text(json.dumps(summ_super, indent=2), encoding="utf-8")

    t1 = out_base / "gap_scan_slope_table_ohmic_like.tex"
    t2 = out_base / "gap_scan_slope_table_super_p3.tex"
    col_order = ["tag", "seed", "bath", "p", "nS", "dephase", "d", "eta", "omega_c", "Tmin", "Tmax", "nT", "slope_loglog", "r2"]

    _write_latex_table_manual(
        ohmic_like,
        t1,
        caption="Davies-gap scan slopes (Ohmic-like).",
        label="tab:gap_scan_ohmic_like",
        col_order=col_order,
    )
    _write_latex_table_manual(
        super_p3,
        t2,
        caption="Davies-gap scan slopes (Super-Ohmic p=3 control).",
        label="tab:gap_scan_super_p3",
        col_order=col_order,
    )

    t3 = out_base / "gap_scan_slope_by_tag_table.tex"
    _write_by_tag_table(df, t3)

    print(f"[ok] wrote {png1}")
    print(f"[ok] wrote {png2}")
    print(f"[ok] wrote {j1}")
    print(f"[ok] wrote {j2}")
    print(f"[ok] wrote {t1}")
    print(f"[ok] wrote {t2}")
    print(f"[ok] wrote {t3}")


if __name__ == "__main__":
    main()