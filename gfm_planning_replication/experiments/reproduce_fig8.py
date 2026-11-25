"""Reproduce Fig. 8: GFL power responses for GFM plans reaching gSCR = 2,
~CgSCR and 3.5 (Scenario 1, c_i = 1), after a 20 ms, 0.05 p.u. SG voltage sag at 0.2 s.

Also sweeps the planned gSCR and records the slowest closed-loop mode of the
full multi-infeed model, to check the paper's claim that the system is stable
iff gSCR > CgSCR.

Outputs go to results/: fig8_time_domain.png, fig8_time_domain.csv,
fig8_gscr_sweep.png, fig8_gscr_sweep.csv.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from experiments.reproduce_fig6_fig7 import RAMP, RESULTS, SURFACE, TEXT, TEXT_2, style
from network.build_B_matrix import build_B
from network.load_system import load_ieee39
from optimization.scenario1 import algorithm1
from stability.calculate_gscr import gscr_with_gfm
from stability.gfl_model import GFLParams, critical_gscr
from validation.dynamic_simulation import build_model, modal_analysis, simulate

SWEEP = np.round(np.arange(1.6, 3.51, 0.1), 2)
SAG = 0.05  # p.u.; the paper does not give the sag depth


def plan(system, target, B):
    """GFM sizing reaching `target` (none needed below the baseline gSCR)."""
    if gscr_with_gfm(system, B=B)[0] >= target:
        return np.zeros(len(system.candidate_nodes))
    return algorithm1(system, target=target, B=B).s_gfm


def sweep(system, B, params):
    rows = []
    for target in SWEEP:
        model = build_model(system, plan(system, target, B), params, B=B)
        ev, damping = modal_analysis(model)
        rows.append({"target_gscr": target, "max_real": ev[0].real,
                     "freq_hz": abs(ev[0].imag) / (2 * np.pi), "min_damping": damping.min()})
    return pd.DataFrame(rows)


def boundary(system, B, params, lo, hi):
    """Planned gSCR at which the multi-infeed system becomes critically stable."""
    def abscissa(target):
        return modal_analysis(build_model(system, plan(system, target, B), params, B=B))[0][0].real
    return brentq(abscissa, lo, hi, xtol=1e-4)


def fig8(system, B, params, targets):
    fig, axes = plt.subplots(len(targets), 1, figsize=(6.4, 6.4), sharex=True)
    rows = []
    for ax, (label, target) in zip(axes, targets.items()):
        style(ax, fig)
        model = build_model(system, plan(system, target, B), params, B=B)
        t, P, _ = simulate(model, t_end=1.0, sag=SAG)
        valid = ~np.isnan(P[0])
        if not valid.all():
            t_trip = t[valid][-1]
            ax.axvline(t_trip, color=TEXT_2, linewidth=1, linestyle="--")
            ax.text(t_trip, 1.85, f"  trip (|I| > 2 p.u.) at {t_trip:.2f} s",
                    color=TEXT_2, fontsize=8, va="top")
        for k in range(P.shape[0]):
            ax.plot(t, P[k], color=RAMP[2], linewidth=1.2, alpha=0.8)
            rows += [{"case": label, "gfl_node": k + 1, "t": tk, "p": pk} for tk, pk in zip(t, P[k])]
        ax.set_ylim(0.0, 2.0)
        ax.set_ylabel("P (p.u.)", color=TEXT)
        ax.set_title(f"{label} gSCR = {target:.2f}", color=TEXT, fontsize=10, loc="left")
    axes[-1].set_xlabel("Time (s)", color=TEXT)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    system = load_ieee39()
    B = build_B(system)
    params = GFLParams()

    cg = critical_gscr(params)
    print(f"CgSCR of the single-IBR system: {cg:.4f}")

    data = sweep(system, B, params)
    data.to_csv(RESULTS / "fig8_gscr_sweep.csv", index=False)
    print(data.round(4).to_string(index=False))
    unstable = data[data.max_real > 0].target_gscr
    if len(unstable) and unstable.max() < SWEEP[-1]:
        g_star = boundary(system, B, params, unstable.max(), unstable.max() + 0.1)
        print(f"multi-infeed stability boundary: gSCR = {g_star:.4f} (CgSCR {cg:.4f})")

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    style(ax, fig)
    ax.plot(data.target_gscr, data.max_real, color=RAMP[3], linewidth=2, marker="o",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(0, color=TEXT_2, linewidth=1)
    ax.axvline(cg, color=TEXT_2, linewidth=1, linestyle="--")
    ax.annotate(f"CgSCR = {cg:.2f}", (cg, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=TEXT, fontsize=9)
    ax.set_xlabel("Planned gSCR", color=TEXT)
    ax.set_ylabel("Max real part (1/s)", color=TEXT)
    ax.set_title("Slowest closed-loop mode vs planned gSCR (IEEE 39-bus)",
                 color=TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig8_gscr_sweep.png", dpi=200)

    f8, d8 = fig8(system, B, params, {"(a)": 2.0, "(b)": cg, "(c)": 3.5})
    f8.savefig(RESULTS / "fig8_time_domain.png", dpi=200)
    d8.to_csv(RESULTS / "fig8_time_domain.csv", index=False)
