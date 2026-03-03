from __future__ import annotations

import os
import sys
import csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import qnm  # pip install qnm


def main():
    # Download cached sequences (first time only)
    qnm.download_data()

    outdir = os.path.join(ROOT, "data", "qnm")
    os.makedirs(outdir, exist_ok=True)

    # Dense spin coverage up to near-extremal
    a_min = 0.0
    a_max = 0.9999
    num = 260
    chis = np.linspace(a_min, a_max, num)

    bundle_path = os.path.join(outdir, "qnm_kerr_bundle.csv")

    # Modes included in the bundle
    modes = [
        (-2, 2, 2, 0),
        (-2, 2, 2, 1),
        (-2, 2, 2, 2),
        (-2, 3, 3, 0),
        (-2, 3, 3, 1),
        (-2, 4, 4, 0),
    ]

    ksc = qnm.cached.KerrSeqCache(init_schw=True)

    with open(bundle_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chi", "ell", "m", "n", "Rew", "Imw", "source", "note"])

        for s, ell, m, n in modes:
            seq = ksc(s=s, l=ell, m=m, n=n)
            for chi in chis:
                omega, A, C = seq(a=float(chi))  # omega is complex ωM
                w.writerow([
                    float(chi),
                    int(ell),
                    int(m),
                    int(n),
                    float(np.real(omega)),
                    float(np.imag(omega)),
                    "qnm",
                    f"s={s} from qnm package"
                ])

    print("Wrote bundle dataset:")
    print(" ", bundle_path)
    print("\nNow run:")
    print(r"  python analysis\qnm_compare_absolute.py")
    print(r"  python analysis\qnm_panel_absolute.py")
    print(r"  python analysis\qnm_tables_absolute.py")
    print(r"  python analysis\qnm_ratio_plot.py")


if __name__ == "__main__":
    main()