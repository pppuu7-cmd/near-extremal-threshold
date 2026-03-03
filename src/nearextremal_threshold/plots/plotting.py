from __future__ import annotations

import os
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    outdir: str = "figures"
    dpi: int = 180

    def ensure_outdir(self) -> None:
        os.makedirs(self.outdir, exist_ok=True)

    def savefig(self, fig, name: str) -> str:
        self.ensure_outdir()
        path = os.path.join(self.outdir, name)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        return path


def new_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax