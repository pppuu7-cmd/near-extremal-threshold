from __future__ import annotations

import os
import sys

# Allow running without `pip install -e .`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nearextremal_threshold.plots.make_figures import make_all


def main() -> None:
    paths = make_all(M=1e6, Delta_min=1e-18, Delta_max=1e-3, n=700, outdir="figures")
    print("Saved figures:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()