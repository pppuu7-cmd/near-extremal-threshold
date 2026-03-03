from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class QNMRow:
    chi: float
    ell: int
    m: int
    n: int
    Rew: float   # Re(ω M)
    Imw: float   # Im(ω M), typically negative
    source: str = ""
    note: str = ""


def load_qnm_csv(path: str | Path) -> list[QNMRow]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    rows: list[QNMRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"chi", "ell", "m", "n", "Rew", "Imw"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        for r in reader:
            rows.append(
                QNMRow(
                    chi=float(r["chi"]),
                    ell=int(r["ell"]),
                    m=int(r["m"]),
                    n=int(r["n"]),
                    Rew=float(r["Rew"]),
                    Imw=float(r["Imw"]),
                    source=str(r.get("source", "")),
                    note=str(r.get("note", "")),
                )
            )
    return rows


def filter_modes(
    rows: Iterable[QNMRow],
    ell: Optional[int] = None,
    m: Optional[int] = None,
    n: Optional[int] = None
) -> list[QNMRow]:
    out: list[QNMRow] = []
    for r in rows:
        if ell is not None and r.ell != ell:
            continue
        if m is not None and r.m != m:
            continue
        if n is not None and r.n != n:
            continue
        out.append(r)
    return out


def newest_qnm_csv(data_dir: str | Path) -> Path:
    """
    Returns the newest CSV (by mtime) in data_dir.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files[0]