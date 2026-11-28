"""Print the numbers quoted in the report text that the other scripts do not print.

Run from gfm_planning_replication/:  python -m experiments.report_checks

Each block names the report section that uses the numbers. Results that take
long to compute are read from results/ (written by the reproduce_* scripts).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from network.build_B_matrix import B_BES, build_B
from network.load_system import load_ieee39
from optimization.scenario1 import algorithm1, proportional_benchmark, solve_sdp
from optimization.scenario2 import centralized_benchmark, ranked_combinations, solve_combination
from stability.calculate_gscr import gscr_with_gfm
from stability.sensitivity import check_gradient, gscr_gradient, site_sensitivity

RESULTS = Path(__file__).resolve().parents[1] / "results"
warnings.filterwarnings("ignore")  # solver accuracy warnings; the SDP cross-check covers them


def hessian_min_eig(system, B, s, h=1e-5):
    """Smallest eigenvalue of the Hessian of lambda_min at s (central differences of the gradient)."""
    n = len(s)
    H = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        H[:, k] = (gscr_gradient(system, s + e, B_BES, B)[1]
                   - gscr_gradient(system, s - e, B_BES, B)[1]) / (2 * h)
    return float(np.linalg.eigvalsh(0.5 * (H + H.T))[0])


def fixed_gamma(system, B, gamma, cost=1.0):
    """Algorithm 1 as in the paper (no safeguard). Returns the iteration count or None."""
    try:
        return algorithm1(system, cost=cost, gamma=gamma, adapt_gamma=False, B=B).iterations
    except RuntimeError:
        return None


def main():
    system = load_ieee39()
    B = build_B(system)
    n, n_cand = len(system.gfl_nodes), len(system.candidate_nodes)
    opt = algorithm1(system, B=B)

    print("Sec. V-A  gradient check at S = 0: max relative error %.1e" % check_gradient(system, B=B)[2])
    for name, s in [("S = 0", np.zeros(n_cand)), ("Scenario 1 optimum", opt.s_gfm),
                    ("starting point (upper limits)", np.full(n_cand, 10.0))]:
        print("Sec. V-A  smallest Hessian eigenvalue at %-30s %.2f" % (name + ":", hessian_min_eig(system, B, s)))

    print("Sec. VI-A fixed gamma, c_i = 1:", {g: fixed_gamma(system, B, g) for g in (-1.0, -0.8, -0.7)},
          "(iterations; None = no convergence in 5000)")
    cost09 = np.r_[np.ones(n), np.full(n_cand - n, 0.9)]
    print("Sec. VI-A fixed gamma = -1, c_i = 0.9 at interior nodes:", fixed_gamma(system, B, -1.0, cost09))
    r09, sdp09 = algorithm1(system, cost=cost09, B=B), solve_sdp(system, cost=cost09, B=B)
    print("Sec. VI-A safeguard: cost %.4f, %d iterations, final gamma %g, %d sites %s; SDP %.4f; rating at node 20 %.1e"
          % (r09.cost, r09.iterations, r09.gamma, len(r09.locations(system)), r09.locations(system), sdp09.cost,
             r09.s_gfm[system.candidate_nodes.index(20)]))

    prop = proportional_benchmark(system, B=B)
    cen, node = centralized_benchmark(system, B=B)
    print("Sec. VI-A proportional %.4f (%.3f per GFL node), centralized %.4f at node %d, Scenario 1 %.4f"
          % (prop.cost, prop.cost / n, cen.cost, node, opt.cost))
    print("Sec. VI-A saving of Scenario 1: %.2f%% (proportional), %.1f%% (centralized); ratings %.3f to %.3f at nodes %s"
          % (100 * (1 - opt.cost / prop.cost), 100 * (1 - opt.cost / cen.cost),
             min(s for s in opt.s_gfm if s > 1e-4), opt.s_gfm.max(), opt.locations(system)))

    f7 = pd.read_csv(RESULTS / "fig7_cost_vs_bbes.csv")
    law = f7.cost * f7.b_bes
    print("Sec. VI-A Fig. 7: %d cases, max |Alg. 1 - SDP| %.1e, cost x B_BES constant per target to %.1e, safeguard used in %d cases (B_BES >= %g)"
          % (len(f7), (f7.cost - f7.cost_sdp).abs().max(), law.groupby(f7.target_gscr).agg(np.ptp).max() / f7.b_bes.min(),
             (f7.gamma != -1).sum(), f7.b_bes[f7.gamma != -1].min()))

    sw = pd.read_csv(RESULTS / "fig8_gscr_sweep.csv").set_index("target_gscr")
    print("Sec. VI-B mode frequency %.1f Hz at planned gSCR 2.0 and %.1f Hz at 2.5; smallest damping ratio at 3.5: %.2f"
          % (sw.freq_hz[2.0], sw.freq_hz[2.5], sw.min_damping[3.5]))
    td = pd.read_csv(RESULTS / "fig8_time_domain.csv").dropna()

    def amplitude(case, t0, t1):
        w = td[(td.case == case) & (td.t >= t0) & (td.t <= t1)].groupby("gfl_node").p
        return ((w.max() - w.min()) / 2).round(4).tolist()

    print("Sec. VI-B (a) last sample at %.3f s" % td[td.case == "(a)"].t.max())
    print("Sec. VI-B (b) amplitude per converter, 0.85 to 1.0 s:", amplitude("(b)", 0.85, 1.0))
    print("Sec. VI-B (c) amplitude per converter, 0.40 to 0.55 s:", amplitude("(c)", 0.40, 0.55))

    _, ranking = site_sensitivity(system, np.ones(n_cand), B=B)
    for kappa, sites in zip(range(1, 5), ranked_combinations(ranking, 3)):
        r = solve_combination(system, sites, B=B)
        print("Sec. VI-C N_max = 3, combination %d %s: cost %.4f (%s)" % (kappa, sites, r.cost, r.solver))
    for n_max in (2, 3, 4):
        bf = pd.read_csv(RESULTS / f"brute_force_nmax{n_max}.csv")
        print("Sec. VI-C exhaustive search, N_max = %d: %d combinations, %d solved as SDP (Algorithm 1 not converged), %d cannot reach the target"
              % (n_max, len(bf), (bf.solver == "sdp").sum(), (bf.solver == "infeasible").sum()))
    t3 = pd.read_csv(RESULTS / "table_III.csv")
    best = t3[t3.iter_max == 150]
    print("Sec. VI-C optimum found at combination", best.best_at.tolist(), "saving vs centralized (%):",
          [round(100 * (1 - c / cen.cost), 1) for c in best.cost])

    t4 = RESULTS / "table_IV.csv"
    if t4.exists():
        t4 = pd.read_csv(t4)
        two = t4[t4.iter_max == 150].cost.to_numpy()
        top = t4[t4.iter_max.isna()].cost.to_numpy()
        paper_top = t4[t4.iter_max.isna()].paper_cost.to_numpy()
        print("Sec. VI-D two-layer", two, "best of 40000", top, "-> best is cheaper by (%)", np.round(100 * (1 - top / two), 1),
              "; two-layer above best by (%)", np.round(100 * (two / top - 1), 1))
        print("Sec. VI-D reported best of 40000 below ours by (%)", np.round(100 * (1 - paper_top / top), 1),
              "; centralized: %.1f" % (100 * (1 - 2.0522 / 2.0895)))
    for target, base, label in [(3.5, 1.51, "39-bus, reported baseline"), (3.0, 1.69, "118-bus, reported baseline"),
                                (3.0, 1.6055, "118-bus, our baseline")]:
        print("Sec. VI-D Eq. (11), %s: %.4f" % (label, 9 * (target - base) / B_BES))
    print("Sec. VI-D 1.8864 x (1 - 0.3318) = %.4f; our Scenario 1 saving: %.2f%%"
          % (1.8864 * (1 - 0.3318), 100 * (1 - 1.2719 / (9 * (3.0 - 1.6055) / B_BES))))


if __name__ == "__main__":
    main()
