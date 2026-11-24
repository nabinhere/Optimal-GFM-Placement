"""Reproduce Fig. 6 (Algorithm 1 convergence vs gamma) and Fig. 7 (optimal
cost vs B_BES for several gSCR targets) on the modified IEEE 39-bus system.

Both use Scenario 1 with c_i = 1. Outputs (PNG + CSV data table) go to results/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network.build_B_matrix import B_BES, build_B
from network.load_system import load_ieee39
from optimization.scenario1 import EPS, algorithm1, solve_sdp

RESULTS = Path(__file__).resolve().parents[1] / "results"

GAMMAS = [-1, -3, -5, -7]
TARGETS = [2.0, 2.5, 3.0, 3.5]
B_BES_VALUES = np.arange(4.0, 10.01, 1.0)

# Ordinal blue ramp (light -> dark), validated with the dataviz palette checks
RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig6(system, B):
    """||S_{k+1} - S_k|| per iteration for each gamma (fixed gamma, as in the paper)."""
    rows, crossings = [], {}
    for gamma in GAMMAS:
        r = algorithm1(system, gamma=gamma, eps=EPS / 10, adapt_gamma=False, B=B)
        err = np.array(r.history)
        crossings[gamma] = int(np.argmax(err <= EPS)) + 1
        rows += [{"gamma": gamma, "iteration": k + 1, "error": e} for k, e in enumerate(err)]
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    style(ax, fig)
    for gamma, color in zip(GAMMAS, RAMP):
        d = data[data.gamma == gamma]
        ax.semilogy(d.iteration, d.error, color=color, linewidth=2, label=f"γ = {gamma}")
        k = crossings[gamma]
        ax.plot(k, EPS, "o", color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(f"{k}", (k, EPS), xytext=(4, 6), textcoords="offset points",
                    color=TEXT, fontsize=9)
    ax.axhline(EPS, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(ax.get_xlim()[1], EPS, " ε = 1e-5", va="center", ha="left", color=TEXT_2, fontsize=9)
    ax.set_xlabel("Iteration", color=TEXT)
    ax.set_ylabel(r"$\|S^{k+1}_{GFM} - S^{k}_{GFM}\|$", color=TEXT)
    ax.set_title("Algorithm 1 convergence under different γ (c_i = 1, gSCR = 3.5)",
                 color=TEXT, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    return fig, data, crossings


def fig7(system, B):
    """Optimal cost vs B_BES for each gSCR target, by Algorithm 1 (checked by SDP)."""
    rows = []
    for target in TARGETS:
        for b in B_BES_VALUES:
            r = algorithm1(system, target=target, b_bes=b, B=B)
            sdp = solve_sdp(system, target=target, b_bes=b, B=B)
            rows.append({"target_gscr": target, "b_bes": b, "cost": r.cost,
                         "cost_sdp": sdp.cost, "iterations": r.iterations, "gamma": r.gamma})
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    style(ax, fig)
    for target, color in zip(TARGETS, RAMP):
        d = data[data.target_gscr == target]
        ax.plot(d.b_bes, d.cost, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"gSCR = {target:g}")
        ax.annotate(f"gSCR = {target:g}", (d.b_bes.iloc[-1], d.cost.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=TEXT, fontsize=9)
    ax.set_xlim(B_BES_VALUES[0] - 0.2, B_BES_VALUES[-1] + 1.3)
    ax.set_xticks(B_BES_VALUES)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$B_{BES}$ (p.u.)", color=TEXT)
    ax.set_ylabel("Optimal cost (p.u.)", color=TEXT)
    ax.set_title("Optimal GFM ESS cost vs unit susceptance (c_i = 1)",
                 color=TEXT, fontsize=10, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, labelcolor=TEXT,
              loc="upper right", bbox_to_anchor=(0.86, 1.0))
    fig.tight_layout()
    return fig, data


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    system = load_ieee39()
    B = build_B(system)

    f6, d6, crossings = fig6(system, B)
    f6.savefig(RESULTS / "fig6_convergence.png", dpi=200)
    d6.to_csv(RESULTS / "fig6_convergence.csv", index=False)
    print("Fig. 6 iterations to reach eps = 1e-5:", crossings, "(paper: -1:30, -3:86, -5:135, -7:181)")

    f7, d7 = fig7(system, B)
    f7.savefig(RESULTS / "fig7_cost_vs_bbes.png", dpi=200)
    d7.to_csv(RESULTS / "fig7_cost_vs_bbes.csv", index=False)
    print(d7.pivot(index="b_bes", columns="target_gscr", values="cost").round(4))
    print(f"max |Alg.1 - SDP| = {np.abs(d7.cost - d7.cost_sdp).max():.1e}")
    # Only B_BES * S_GFM enters the model, so with non-binding bounds the optimal
    # cost scales exactly as 1 / B_BES.
    ref = d7[np.isclose(d7.b_bes, 5.0)].set_index("target_gscr").cost
    scaled = d7.apply(lambda r: ref[r.target_gscr] * 5.0 / r.b_bes, axis=1)
    print(f"max |cost - cost(5) * 5 / B_BES| = {np.abs(d7.cost - scaled).max():.1e}")
