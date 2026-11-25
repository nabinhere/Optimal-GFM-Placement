"""Reproduce Fig. 9 and Table III (Scenario 2, modified IEEE 39-bus system).

Fig. 9: (a) best cost Cost_best^kappa vs number of enumerated combinations,
with the brute-force global optimum dashed; (b) cumulative time; (c) GFM ESS
ratings of the best combination per N_max. All runs use c_i = 1, gSCR = 3.5.

Global optima come from results/brute_force_nmax{N}.csv (written by
experiments.brute_force_scenario2); N_max values without a complete file get
no dashed line.

Outputs go to results/: fig9_scenario2.png, fig9_traces.csv,
fig9_ratings.csv and table_III.csv.
"""

from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reproduce_fig6_fig7 import RAMP, RESULTS, SURFACE, TEXT, TEXT_2, style
from network.build_B_matrix import build_B
from network.load_system import load_ieee39
from optimization.scenario2 import centralized_benchmark, two_layer

N_MAX = [2, 3, 4]
ITER_MAX = [1, 50, 100, 150]
COLORS = dict(zip(N_MAX, [RAMP[0], RAMP[2], RAMP[3]]))
PAPER = {  # Table III: (combinations, cost) per (N_max, iter_max)
    (4, 1): (3, 2.7503), (4, 50): (70, 2.6303), (4, 100): (120, 2.6303), (4, 150): (281, 2.597),
    (3, 1): (4, 2.9341), (3, 50): (55, 2.7988), (3, 100): (190, 2.7561), (3, 150): (240, 2.7561),
    (2, 1): (4, 3.1578), (2, 50): (95, 2.9549), (2, 100): (145, 2.9549), (2, 150): (195, 2.9549),
}


def global_optimum(system, n_max):
    """Best cost in the brute-force file, or None if it is missing or incomplete."""
    path = RESULTS / f"brute_force_nmax{n_max}.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if len(data) < comb(len(system.candidate_nodes), n_max):
        return None
    return float(data.cost.min())


def table3(system, B, bench):
    rows, runs = [], {}
    for n_max in N_MAX:
        for iter_max in ITER_MAX:
            r = two_layer(system, n_max, iter_max, B=B)
            runs[n_max, iter_max] = r
            paper_combos, paper_cost = PAPER[n_max, iter_max]
            rows.append({"n_max": n_max, "iter_max": iter_max, "combinations": r.evaluated,
                         "best_at": r.best_index, "cost": round(r.cost, 4),
                         "reduction_pct": round(100 * (1 - r.cost / bench.cost), 2),
                         "time_s": round(r.elapsed, 1), "paper_combinations": paper_combos,
                         "paper_cost": paper_cost})
    return pd.DataFrame(rows), runs


def fig9(system, runs, optima):
    fig = plt.figure(figsize=(9.6, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    for ax in (ax_a, ax_b, ax_c):
        style(ax, fig)

    traces = []
    for n_max in N_MAX:
        r = runs[n_max, ITER_MAX[-1]]
        kappa = np.arange(1, r.evaluated + 1)
        color = COLORS[n_max]
        ax_a.step(kappa, r.best_cost_trace, where="post", color=color, linewidth=2,
                  label=f"N_max = {n_max}")
        ax_b.plot(kappa, r.time_trace, color=color, linewidth=2, label=f"N_max = {n_max}")
        if optima[n_max] is not None:
            ax_a.axhline(optima[n_max], color=color, linewidth=1, linestyle="--")
        ax_a.annotate(f"{r.cost:.4f}", (r.evaluated, r.cost), xytext=(4, -3),
                      textcoords="offset points", va="top", color=TEXT, fontsize=8)
        traces += [{"n_max": n_max, "kappa": k, "best_cost": c, "time_s": t}
                   for k, c, t in zip(kappa, r.best_cost_trace, r.time_trace)]

    ax_a.set_ylim(2.4, 3.6)
    ax_a.set_xlabel("Number of combinations κ", color=TEXT)
    ax_a.set_ylabel("Best cost (p.u.)", color=TEXT)
    ax_a.set_title("(a) Best cost; dashed = brute-force optimum", color=TEXT, fontsize=10, loc="left")
    ax_a.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax_b.set_xlabel("Number of combinations κ", color=TEXT)
    ax_b.set_ylabel("Time (s)", color=TEXT)
    ax_b.set_title(f"(b) Cumulative time, iter_max = {ITER_MAX[-1]}", color=TEXT, fontsize=10, loc="left")
    ax_b.legend(frameon=False, fontsize=8, labelcolor=TEXT)

    # (c) ratings of the best combination, grouped by node
    ratings = {n_max: dict(zip(system.candidate_nodes, runs[n_max, ITER_MAX[-1]].best.s_gfm))
               for n_max in N_MAX}
    nodes = sorted({node for n_max in N_MAX for node, s in ratings[n_max].items() if s > 1e-4})
    width = 0.26
    x = np.arange(len(nodes))
    rows = []
    for j, n_max in enumerate(N_MAX):
        values = [ratings[n_max][node] for node in nodes]
        ax_c.bar(x + (j - 1) * width, values, width - 0.03, color=COLORS[n_max],
                 edgecolor=SURFACE, linewidth=1, label=f"N_max = {n_max}")
        rows += [{"n_max": n_max, "node": node, "rating": v} for node, v in zip(nodes, values)]
    ax_c.set_xticks(x, [str(node) for node in nodes])
    ax_c.set_xlabel("Node number", color=TEXT)
    ax_c.set_ylabel("Capacity (p.u.)", color=TEXT)
    ax_c.set_title("(c) GFM ESS ratings of the best combination", color=TEXT, fontsize=10, loc="left")
    ax_c.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncols=3)
    fig.tight_layout()
    return fig, pd.DataFrame(traces), pd.DataFrame(rows)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    system = load_ieee39()
    B = build_B(system)
    bench, node = centralized_benchmark(system, B=B)
    print(f"centralized benchmark: node {node}, cost {bench.cost:.4f}")

    table, runs = table3(system, B, bench)
    table.to_csv(RESULTS / "table_III.csv", index=False)
    print(table.to_string(index=False))

    optima = {n_max: global_optimum(system, n_max) for n_max in N_MAX}
    print("brute-force optima:", optima)
    fig, traces, ratings = fig9(system, runs, optima)
    fig.savefig(RESULTS / "fig9_scenario2.png", dpi=200)
    traces.to_csv(RESULTS / "fig9_traces.csv", index=False)
    ratings.to_csv(RESULTS / "fig9_ratings.csv", index=False)
    print(ratings.pivot(index="node", columns="n_max", values="rating").round(4))
