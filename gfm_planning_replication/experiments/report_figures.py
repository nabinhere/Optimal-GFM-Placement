"""Two figures for the report that are not in the paper.

1. gSCR against the total GFM ESS capacity on the modified IEEE 39-bus system
   (c_i = 1) for three ways of placing the capacity: the Scenario 1 optimum,
   the proportional benchmark and a single site at the node of the centralized
   benchmark.
2. Investment cost of every plan and benchmark on both test systems, ours
   against the values reported in the paper.

Figure 2 reads results/table_III.csv, results/table_IV.csv and
results/ieee118_scenario1.csv, so run reproduce_fig9 and reproduce_table4
first. Outputs (PNG + CSV data table) go to results/.

    python -m experiments.report_figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.reproduce_fig6_fig7 import RAMP, RESULTS, SURFACE, TEXT, TEXT_2, style
from network.build_B_matrix import B_BES, build_B
from network.load_system import load_ieee39, load_ieee118
from optimization.scenario1 import TARGET_GSCR, algorithm1, proportional_benchmark
from optimization.scenario2 import centralized_benchmark
from stability.calculate_gscr import gscr_with_gfm

CGSCR = 2.5       # critical gSCR of the GFL IBRs, Sec. V-A
TARGET_118 = 3.0  # gSCR target of the 118-bus case, Sec. V-C

# Values reported in the paper that are not in table_III.csv / table_IV.csv
PAPER_39 = {"centralized": 4.4125,   # Sec. V-B
            "proportional": 2.8656,  # Sec. V-A
            "scenario1": 2.4516}     # Table I, c_i = 1
PAPER_118_CENTRALIZED = 2.0522       # Sec. V-C

# Categorical colours: the project's blue plus two hues of the dataviz palette.
# Checked with its validator on SURFACE (CVD and normal-vision separation pass;
# orange and aqua are below 3:1 contrast, so every curve is also labelled).
BLUE, ORANGE, AQUA = RAMP[2], "#f47230", "#00b894"


# ----- Figure 1: gSCR against the installed capacity ------------------------

def strength_data(system, B):
    """gSCR reached per p.u. of GFM ESS capacity for the three placements."""
    n_cand = len(system.candidate_nodes)
    baseline = gscr_with_gfm(system, None, B_BES, B)[0]
    _, node = centralized_benchmark(system, B=B)
    k = system.candidate_nodes.index(node)

    rows = [{"plan": plan, "capacity": 0.0, "gscr": baseline}
            for plan in ("optimal", "proportional", "one_site")]
    for target in np.round(np.arange(1.6, 4.001, 0.1), 2):
        rows.append({"plan": "optimal", "gscr": target,
                     "capacity": algorithm1(system, target=target, B=B).cost})
        rows.append({"plan": "proportional", "gscr": target,
                     "capacity": proportional_benchmark(system, target=target, B=B).cost})
    for capacity in np.linspace(0.05, 5.0, 100):
        s = np.zeros(n_cand)
        s[k] = capacity
        rows.append({"plan": "one_site", "capacity": capacity,
                     "gscr": gscr_with_gfm(system, s, B_BES, B)[0]})
    return pd.DataFrame(rows), baseline, node


def strength_figure(system, B):
    data, baseline, node = strength_data(system, B)
    at_target = {
        "optimal": algorithm1(system, B=B).cost,
        "proportional": proportional_benchmark(system, B=B).cost,
        "one_site": centralized_benchmark(system, B=B)[0].cost,
    }
    plans = [("optimal", BLUE, "Scenario 1 optimum"),
             ("proportional", ORANGE, "Proportional benchmark"),
             ("one_site", AQUA, f"One site (node {node})")]

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    style(ax, fig)
    ax.axhspan(1.2, CGSCR, color=TEXT_2, alpha=0.07, linewidth=0)
    ax.axhline(CGSCR, color=TEXT_2, linewidth=1, linestyle="--")
    ax.axhline(TARGET_GSCR, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(5.02, CGSCR - 0.05, "unstable below CgSCR = 2.5", ha="right", va="top",
            color=TEXT_2, fontsize=8)
    ax.text(0.06, TARGET_GSCR + 0.05, "target 3.5", ha="left", va="bottom",
            color=TEXT_2, fontsize=8)

    for plan, color, label in plans:
        d = data[data.plan == plan].sort_values("capacity")
        ax.plot(d.capacity, d.gscr, color=color, linewidth=2, label=label,
                solid_capstyle="round", solid_joinstyle="round")
    for plan, color, _ in plans:
        ax.plot(at_target[plan], TARGET_GSCR, "o", color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.plot(0, baseline, "o", color=TEXT, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=3, clip_on=False)
    ax.annotate(f"no GFM ESS: {baseline:.2f}", (0, baseline), xytext=(8, -4),
                textcoords="offset points", va="top", color=TEXT, fontsize=8.5)

    # capacity needed for the target, written next to each marker
    offsets = {"optimal": ((-6, 7), "right"), "proportional": ((6, -8), "left"),
               "one_site": ((0, 8), "center")}
    for plan, _, _ in plans:
        (dx, dy), ha = offsets[plan]
        ax.annotate(f"{at_target[plan]:.4f}", (at_target[plan], TARGET_GSCR),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="bottom" if dy > 0 else "top", color=TEXT, fontsize=8.5)

    ax.set_xlim(0, 5.05)
    ax.set_ylim(1.2, 4.15)
    ax.set_xlabel("Total GFM ESS capacity (p.u.)", color=TEXT)
    ax.set_ylabel("gSCR", color=TEXT)
    ax.set_title("gSCR against installed GFM ESS capacity (IEEE 39 bus, c_i = 1)",
                 color=TEXT, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), borderaxespad=0.3)
    fig.tight_layout()
    return fig, data, at_target


# ----- Figure 2: cost of every plan, ours against the paper -----------------

def cost_table():
    """One row per plan: system, item, our cost, cost reported in the paper."""
    s39 = load_ieee39()
    B39 = build_B(s39)
    scenario1 = algorithm1(s39, B=B39)
    t3 = pd.read_csv(RESULTS / "table_III.csv")
    t3 = t3[t3.iter_max == t3.iter_max.max()].set_index("n_max")
    rows = [
        ("39", "Centralized benchmark (1 site)", centralized_benchmark(s39, B=B39)[0].cost,
         PAPER_39["centralized"]),
        ("39", f"Proportional benchmark ({len(s39.gfl_nodes)} sites)",
         proportional_benchmark(s39, B=B39).cost, PAPER_39["proportional"]),
        ("39", f"Scenario 1 ({len(scenario1.locations(s39))} sites)", scenario1.cost,
         PAPER_39["scenario1"]),
    ]
    rows += [("39", f"Two layer, $N_{{max}}$ = {n}", t3.cost[n], t3.paper_cost[n]) for n in t3.index]

    s118 = load_ieee118()
    t4 = pd.read_csv(RESULTS / "table_IV.csv")
    two_layer = t4[t4.iter_max == t4.iter_max.max()].set_index("n_max")
    top = t4[t4.iter_max.isna()].set_index("n_max")
    ratings = pd.read_csv(RESULTS / "ieee118_scenario1.csv").rating
    rows.append(("118", "Centralized benchmark (1 site)",
                 centralized_benchmark(s118, target=TARGET_118)[0].cost, PAPER_118_CENTRALIZED))
    rows += [("118", f"Two layer, $N_{{max}}$ = {n}", two_layer.cost[n], two_layer.paper_cost[n])
             for n in two_layer.index]
    rows += [("118", f"Best of {int(top.combinations[n])}, $N_{{max}}$ = {n}", top.cost[n],
              top.paper_cost[n]) for n in top.index]
    # the paper gives the Scenario 1 result of this case only as a saving in per cent
    rows.append(("118", f"Scenario 1 ({int((ratings > 1e-4).sum())} sites)", ratings.sum(), np.nan))
    return pd.DataFrame(rows, columns=["system", "item", "cost", "paper_cost"])


def cost_figure(table):
    panels = [("39", f"(a) IEEE 39 bus system, gSCR target {TARGET_GSCR:.1f}"),
              ("118", f"(b) IEEE 118 bus system, gSCR target {TARGET_118:.1f}")]
    sizes = [int((table.system == key).sum()) for key, _ in panels]
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 0.34 * sum(sizes) + 2.0),
                             gridspec_kw={"height_ratios": sizes})
    for ax, (key, title) in zip(axes, panels):
        d = table[table.system == key].sort_values("cost", ascending=False, kind="stable")
        y = np.arange(len(d))[::-1]
        style(ax, fig)
        ax.grid(axis="y", visible=False)
        ax.barh(y, d.cost, height=0.56, color=BLUE, edgecolor=SURFACE, linewidth=1)
        for width, color in ((4.5, SURFACE), (2.0, TEXT)):  # surface ring, then the mark
            ax.plot(d.paper_cost, y, linestyle="none", marker="|", markersize=13,
                    markeredgewidth=width, color=color, zorder=3)
        reach = np.fmax(d.cost, d.paper_cost.fillna(0)).to_numpy()
        for yi, cost, x in zip(y, d.cost, reach):
            ax.text(x, yi, f"  {cost:.4f}", va="center", ha="left", color=TEXT, fontsize=8.5)
        ax.set_yticks(y, d.item)
        ax.tick_params(axis="y", length=0, labelsize=8.5, colors=TEXT)
        ax.set_xlim(0, 1.2 * reach.max())
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("Investment cost (p.u.)", color=TEXT, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    left = axes[0].get_position().x0
    fig.legend(handles=[Patch(color=BLUE, label="This work"),
                        Line2D([], [], linestyle="none", marker="|", markersize=11,
                               markeredgewidth=2, color=TEXT, label="Reported in the paper")],
               frameon=False, fontsize=8.5, labelcolor=TEXT, ncols=2, loc="upper left",
               bbox_to_anchor=(left, 1.0), borderaxespad=0.4, handletextpad=0.5,
               columnspacing=1.6)
    return fig


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    system = load_ieee39()
    B = build_B(system)

    f1, curves, at_target = strength_figure(system, B)
    f1.savefig(RESULTS / "report_gscr_vs_capacity.png", dpi=200)
    curves.to_csv(RESULTS / "report_gscr_vs_capacity.csv", index=False)
    print("capacity needed for gSCR = 3.5:", {k: round(v, 4) for k, v in at_target.items()})

    table = cost_table()
    table.to_csv(RESULTS / "report_cost_comparison.csv", index=False)
    cost_figure(table).savefig(RESULTS / "report_cost_comparison.png", dpi=200)
    print(table.round(4).to_string(index=False))
