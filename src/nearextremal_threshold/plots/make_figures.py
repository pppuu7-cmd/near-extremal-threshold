from __future__ import annotations

import numpy as np

from nearextremal_threshold.constraints.covariant_phase_space_scaling import CPSScalingModel
from nearextremal_threshold.constraints.nonfactorization_measure import C_values
from nearextremal_threshold.geometry.throat_scalings import T_H_scaling
from nearextremal_threshold.jt.bath_spectral import ThermalBath
from nearextremal_threshold.jt.gkls_rates import GKLSRateModel
from nearextremal_threshold.observables.otoc_suppression import OTOCSuppressionModel
from nearextremal_threshold.plots.plotting import PlotConfig, new_fig


def fig_C_vs_Delta(M: float, Deltas: np.ndarray, cfg: PlotConfig, c: float = 1.0) -> str:
    cps = CPSScalingModel(c=c, lP=1.0)
    C = C_values(M, Deltas, cps)
    fig, ax = new_fig()
    ax.plot(Deltas, C)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$\mathcal{C}(\Delta)$")
    ax.set_title(r"Operational non-factorization scaling")
    # mark Delta_crit
    Dc = cps.Delta_crit(M)
    ax.axvline(Dc, linestyle="--")
    ax.text(Dc, np.nanmax(C[np.isfinite(C)]) * 0.2, r"$\Delta_{\rm crit}$", rotation=90)
    return cfg.savefig(fig, "C_vs_Delta.png")


def fig_Gamma_vs_Delta(M: float, Deltas: np.ndarray, cfg: PlotConfig,
                      g: float = 1.0, N_B: float = 1.0, Delta_B: float = 1.0) -> str:
    bath = ThermalBath(N_B=N_B, Delta_B=Delta_B)
    gk = GKLSRateModel(g=g, bath=bath)
    Gam = np.array([gk.Gamma(M, d) for d in Deltas], dtype=float)
    TH = np.array([T_H_scaling(M, d) for d in Deltas], dtype=float)

    fig, ax = new_fig()
    ax.plot(Deltas, Gam, label=r"$\Gamma(\Delta)$")
    ax.plot(Deltas, TH, label=r"$T_H(\Delta)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"rate / scale")
    ax.set_title(r"JT+Bath dissipation rate vs near-extremality")
    ax.legend()
    return cfg.savefig(fig, "Gamma_vs_Delta.png")


def fig_lambda_eff_vs_Delta(M: float, Deltas: np.ndarray, cfg: PlotConfig,
                           g: float = 1.0, N_B: float = 1.0, Delta_B: float = 1.0) -> str:
    bath = ThermalBath(N_B=N_B, Delta_B=Delta_B)
    gk = GKLSRateModel(g=g, bath=bath)
    model = OTOCSuppressionModel(gkls=gk)
    data = model.curve(M, Deltas)

    fig, ax = new_fig()
    ax.plot(Deltas, data["lambda_L"], label=r"$\lambda_L$")
    ax.plot(Deltas, data["lambda_eff"], label=r"$\lambda_{\rm eff}$")
    ax.plot(Deltas, data["Gamma"], label=r"$\Gamma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"rate")
    ax.set_title(r"Chaos suppression: $\lambda_{\rm eff}=\lambda_L-\Gamma$")
    ax.legend()
    return cfg.savefig(fig, "lambda_eff_vs_Delta.png")


def fig_Gamma_min_vs_M(Ms: np.ndarray, cfg: PlotConfig,
                       c: float = 1.0, g: float = 1.0, N_B: float = 1.0, Delta_B: float = 1.0) -> str:
    cps = CPSScalingModel(c=c, lP=1.0)
    bath = ThermalBath(N_B=N_B, Delta_B=Delta_B)
    gk = GKLSRateModel(g=g, bath=bath)

    Gam_min = []
    for M in Ms:
        Dc = cps.Delta_crit(M)
        Gam_min.append(gk.Gamma(M, Dc))
    Gam_min = np.array(Gam_min, dtype=float)

    fig, ax = new_fig()
    ax.plot(Ms, Gam_min)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\Gamma_{\min}(M)$")
    ax.set_title(r"Minimal dissipation scale vs mass")
    return cfg.savefig(fig, "Gamma_min_vs_M.png")


def make_all(M: float = 1e6,
             Delta_min: float = 1e-16,
             Delta_max: float = 1e-2,
             n: int = 500,
             outdir: str = "figures") -> list[str]:
    cfg = PlotConfig(outdir=outdir, dpi=200)
    Deltas = np.logspace(np.log10(Delta_min), np.log10(Delta_max), n)

    paths = []
    paths.append(fig_C_vs_Delta(M, Deltas, cfg, c=1.0))
    paths.append(fig_Gamma_vs_Delta(M, Deltas, cfg, g=1.0, N_B=1.0, Delta_B=1.0))
    paths.append(fig_lambda_eff_vs_Delta(M, Deltas, cfg, g=1.0, N_B=1.0, Delta_B=1.0))
    Ms = np.logspace(2, 9, 200)
    paths.append(fig_Gamma_min_vs_M(Ms, cfg, c=1.0, g=1.0, N_B=1.0, Delta_B=1.0))
    return paths