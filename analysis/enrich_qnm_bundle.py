#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enrich_qnm_bundle.py

Out-of-the-box:
  python analysis/enrich_qnm_bundle.py

Reads:
  <repo>/data/qnm/qnm_kerr_bundle.csv

Writes:
  <repo>/data/qnm/qnm_kerr_bundle_enriched.csv

Adds columns:
  Delta = 1 - chi^2
  THM   = (T_H * M) for Kerr with M=1: THM = sqrt(1-chi^2) / (4*pi*(1 + sqrt(1-chi^2)))
  absImwM = |Imw|
  R = absImwM / THM
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for d in [start] + list(start.parents):
        if (d / "analysis").is_dir() and (d / "data").is_dir():
            return d
    cwd = Path.cwd().resolve()
    return cwd


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing required columns: {missing}. Available: {list(df.columns)}")


def compute_THM_from_chi(chi: np.ndarray) -> np.ndarray:
    """
    Kerr Hawking temperature: T_H = (r_+ - r_-)/(4*pi*(r_+^2 + a^2)), with M=1, a=chi
    r_+ = 1 + sqrt(1-a^2), r_- = 1 - sqrt(1-a^2)
    => r_+ - r_- = 2*sqrt(1-a^2)
    r_+^2 + a^2 = (1 + s)^2 + a^2 = 1 + 2s + s^2 + a^2 = 2 + 2s   since s^2 + a^2 = 1
    => T_H = 2s / (4*pi*(2+2s)) = s / (4*pi*(1+s))
    Thus THM = T_H * M = s/(4*pi*(1+s)) for M=1.
    """
    s = np.sqrt(np.clip(1.0 - chi**2, 0.0, None))
    THM = s / (4.0 * math.pi * (1.0 + s))
    return THM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich Kerr QNM bundle with Delta, THM, absImwM, R.")
    p.add_argument("--in", dest="input_csv", default=None, help="Input CSV (default: data/qnm/qnm_kerr_bundle.csv)")
    p.add_argument("--out", dest="output_csv", default=None, help="Output CSV (default: data/qnm/qnm_kerr_bundle_enriched.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = find_repo_root(Path(__file__).resolve())
    in_csv = Path(args.input_csv) if args.input_csv else (root / "data" / "qnm" / "qnm_kerr_bundle.csv")
    out_csv = Path(args.output_csv) if args.output_csv else (root / "data" / "qnm" / "qnm_kerr_bundle_enriched.csv")

    if not in_csv.exists():
        raise SystemExit(f"[error] input not found: {in_csv}")

    df = pd.read_csv(in_csv)
    require_cols(df, ["chi", "Rew", "Imw", "ell", "m", "n"])

    chi = pd.to_numeric(df["chi"], errors="coerce").to_numpy(dtype=float)
    imw = pd.to_numeric(df["Imw"], errors="coerce").to_numpy(dtype=float)

    df["Delta"] = 1.0 - chi**2
    df["THM"] = compute_THM_from_chi(chi)
    df["absImwM"] = np.abs(imw)
    df["R"] = df["absImwM"] / df["THM"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[ok] read : {in_csv}  (rows={len(df)})")
    print(f"[ok] wrote: {out_csv}")
    print(f"[cols] {list(df.columns)}")


if __name__ == "__main__":
    main()