#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_all.py

One-command reproduction (no args needed):
  python analysis/run_all.py

Pipeline:
  1) enrich_qnm_bundle.py
  2) qnm_ratio_collapse.py
  3) lofo_from_bundle.py
  4) qnm_ratio_collapse_table.py
  5) qnm_R0_extrapolate.py
  6) qnm_rho_by_family.py            (forced consistent --delta-max)
  7) qnm_tables_absolute.py          (forced consistent --delta-max)
  8) make_qnm_abs_scaling_table.py   (if present)
  9) davies_gap_scan.py              (if present)
 10) davies_gap_aggregate.py         (if present)

All called via subprocess so it matches how you normally run scripts in venv.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ---- Single source of truth for the near-extremal window used in "absolute" fits ----
DELTA_MAX = "0.19"  # change to "0.05" if you want strict near-extremal only


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for d in [start] + list(start.parents):
        if (d / "analysis").is_dir() and (d / "data").is_dir():
            return d
    return Path.cwd().resolve()


def run(script: Path, *args: str) -> int:
    cmd = [sys.executable, str(script), *args]
    print("\n" + "=" * 90)
    print("[run]", " ".join(cmd))
    print("=" * 90)
    p = subprocess.run(cmd, cwd=str(script.parent.parent))
    return int(p.returncode)


def main() -> None:
    root = find_repo_root(Path(__file__).resolve())
    analysis = root / "analysis"

    # Always enrich first. Downstream scripts should prefer the enriched CSV.
    pipeline: list[tuple[Path, list[str]]] = [
        (analysis / "enrich_qnm_bundle.py", []),
        (analysis / "qnm_ratio_collapse.py", []),
        (analysis / "lofo_from_bundle.py", []),
        (analysis / "qnm_ratio_collapse_table.py", []),
        (analysis / "qnm_R0_extrapolate.py", []),
    ]

    # Force consistent delta cut for "absolute scaling" style scripts (if present)
    qnm_rho = analysis / "qnm_rho_by_family.py"
    if qnm_rho.exists():
        pipeline.append((qnm_rho, ["--delta-max", DELTA_MAX, "--write-one-liner"]))
    else:
        print(f"[skip] missing: {qnm_rho.name}")

    qnm_abs = analysis / "qnm_tables_absolute.py"
    if qnm_abs.exists():
        pipeline.append((qnm_abs, ["--delta-max", DELTA_MAX]))
    else:
        print(f"[skip] missing: {qnm_abs.name}")

    # Remaining optional scripts
    optional = [
        (analysis / "make_qnm_abs_scaling_table.py", []),
        (analysis / "davies_gap_scan.py", []),
        (analysis / "davies_gap_aggregate.py", []),
    ]
    pipeline.extend(optional)

    for script, args in pipeline:
        if not script.exists():
            print(f"[skip] missing: {script.name}")
            continue
        code = run(script, *args)
        if code != 0:
            raise SystemExit(f"[error] {script.name} failed with code={code}")

    print("\n[ok] pipeline finished")


if __name__ == "__main__":
    main()