"""Scenario 2: limited number of grid-scale GFM ESS stations (P2).

    min  sum_i c_i S_GFMi
    s.t. lambda_min(A(S_GFM)) >= CgSCR + dg0                  (15b)
         S_min u_i <= S_GFMi <= S_max u_i,  sum_i u_i <= N_max (16b-c)

Two-layer approach (Sec. IV-C, Fig. 4):
  outer: candidate sites ranked by the sensitivity p_i (Eq. 27); all
         N_max-site combinations of ranks ordered colexicographically
         (Steps 2-4, Eq. 28), so combinations of high-sensitivity sites come
         first; enumeration stops once the best cost has not improved for
         iter_max consecutive combinations (Eq. 29).
  inner: Algorithm 1 sizes the GFM ESSs of one combination (u fixed).

Since cost and gSCR are monotone in S_GFM, using all N_max sites never hurts
(a site can still be sized ~0), so only combinations of exactly N_max sites
are enumerated, as in the paper.
"""

import time
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import brentq

from network.build_B_matrix import B_BES, build_B
from optimization.scenario1 import S_MAX, TARGET_GSCR, Result, _costs, algorithm1, solve_sdp
from stability.calculate_gscr import gscr_with_gfm
from stability.sensitivity import site_sensitivity


@dataclass
class PlanningResult:
    best: Result                    # sizing of the best combination (None if none feasible)
    sites: tuple                    # best combination (node labels)
    evaluated: int                  # combinations evaluated before stopping
    best_index: int                 # 1-based index of the combination that gave `best`
    elapsed: float                  # seconds
    best_cost_trace: list = field(default_factory=list)  # Cost_best^kappa per kappa
    time_trace: list = field(default_factory=list)       # cumulative seconds per kappa

    @property
    def cost(self):
        return np.inf if self.best is None else self.best.cost


def ranked_combinations(ranking, n_max):
    """Site combinations in the order of Steps 2-4 / Eq. 28.

    `ranking` lists node labels from highest to lowest sensitivity. Each
    combination of ranks is sorted ascending (Step 3), and combinations are
    ordered colexicographically: compare the last (lowest-sensitivity) rank
    first, then the one before it, and so on (Step 4).

    Generated lazily: C(116, 5) combinations for the 118-bus system would not
    fit in memory, and only the first few hundred are usually needed.
    """
    for ranks in _colex(len(ranking), n_max):
        yield tuple(ranking[r] for r in ranks)


def _colex(n, k):
    """k-subsets of range(n) in colexicographic order (largest element last)."""
    if k == 0:
        yield ()
        return
    for top in range(k - 1, n):
        for rest in _colex(top, k - 1):
            yield rest + (top,)


def solve_combination(system, sites, target=TARGET_GSCR, cost=1.0, s_max=S_MAX,
                      b_bes=B_BES, B=None, max_iter=1000, **alg1_kwargs):
    """Inner layer: size GFM ESSs restricted to `sites` with Algorithm 1.

    Returns None if the combination cannot reach the target even at s_max.
    A few poorly conditioned combinations (e.g. sites 4 and 5, which are
    tightly coupled) make Algorithm 1 crawl; if it has not converged after
    max_iter iterations, the exact SDP restricted to `sites` is used instead
    (Result.solver == "sdp").
    """
    nodes = system.candidate_nodes
    upper = np.zeros(len(nodes))
    upper[[nodes.index(site) for site in sites]] = s_max
    if gscr_with_gfm(system, upper, b_bes, B)[0] < target:
        return None
    try:
        return algorithm1(system, target, cost, 0.0, upper, s_init=upper,
                          max_iter=max_iter, b_bes=b_bes, B=B, **alg1_kwargs)
    except RuntimeError:
        return solve_sdp(system, target, cost, 0.0, upper, b_bes, B)


def _enumerate(system, combos, target, cost, s_max, b_bes, B, iter_max=None, **alg1_kwargs):
    """Evaluate combinations in order, tracking the best (Fig. 4)."""
    start = time.perf_counter()
    best, best_sites, best_index, stall, trace, times = None, None, 0, 0, [], []
    for kappa, sites in enumerate(combos, start=1):
        r = solve_combination(system, sites, target, cost, s_max, b_bes, B, **alg1_kwargs)
        if r is not None and (best is None or r.cost < best.cost - 1e-9):
            best, best_sites, best_index, stall = r, sites, kappa, 0
        else:
            stall += 1
        trace.append(np.inf if best is None else best.cost)
        times.append(time.perf_counter() - start)
        if iter_max is not None and best is not None and stall >= iter_max:
            break
    return PlanningResult(best, best_sites, kappa, best_index,
                          time.perf_counter() - start, trace, times)


def stop_point(best_cost_trace, iter_max):
    """1-based kappa at which Eq. 29 stops a run, given a longer run's trace.

    A run with a larger iter_max passes through the same combinations, so the
    results for every smaller iter_max can be read off its trace. Returns None
    if the trace ends before the criterion is met.
    """
    stall, best = 0, np.inf
    for kappa, cost in enumerate(best_cost_trace, start=1):
        if cost < best - 1e-9:
            best, stall = cost, 0
        else:
            stall += 1
        if np.isfinite(best) and stall >= iter_max:
            return kappa
    return None


def two_layer(system, n_max, iter_max, target=TARGET_GSCR, cost=1.0, s_max=S_MAX,
              b_bes=B_BES, B=None, **alg1_kwargs):
    """Outer partial enumeration + inner Algorithm 1; stop per Eq. 29."""
    if B is None:
        B = build_B(system)
    _, ranking = site_sensitivity(system, _costs(cost, len(system.candidate_nodes)), b_bes, B)
    combos = ranked_combinations(ranking, n_max)
    return _enumerate(system, combos, target, cost, s_max, b_bes, B, iter_max, **alg1_kwargs)


def brute_force(system, n_max, target=TARGET_GSCR, cost=1.0, s_max=S_MAX,
                b_bes=B_BES, B=None, **alg1_kwargs):
    """Exact reference: evaluate every N_max-site combination."""
    if B is None:
        B = build_B(system)
    combos = combinations(system.candidate_nodes, n_max)
    return _enumerate(system, combos, target, cost, s_max, b_bes, B, None, **alg1_kwargs)


def centralized_benchmark(system, target=TARGET_GSCR, cost=1.0, s_max=S_MAX,
                          b_bes=B_BES, B=None):
    """Benchmark: single best site (paper: node 25, 4.4125 p.u.).

    With one site the problem is 1-D and gSCR is monotone in its rating, so
    each site's minimum rating is found by root finding.
    """
    if B is None:
        B = build_B(system)
    nodes = system.candidate_nodes
    c = _costs(cost, len(nodes))
    best = None
    for k, node in enumerate(nodes):
        s = np.zeros(len(nodes))

        def margin(rating):
            s[k] = rating
            return gscr_with_gfm(system, s, b_bes, B)[0] - target

        if margin(s_max) < 0:
            continue
        rating = brentq(margin, 0.0, s_max, xtol=1e-12)
        s[k] = rating
        if best is None or c[k] * rating < best[0].cost:
            best = (Result(s.copy(), float(c[k] * rating), target, solver="bisection"), node)
    return best


if __name__ == "__main__":
    from network.load_system import load_ieee39

    system = load_ieee39()
    B = build_B(system)
    bench, node = centralized_benchmark(system, B=B)
    print(f"centralized benchmark: node {node}, cost {bench.cost:.4f} (paper: node 25, 4.4125)")

    for n_max in (2, 3, 4):
        print(f"N_max = {n_max} ({comb(len(system.candidate_nodes), n_max)} combinations)")
        for iter_max in (1, 50, 100, 150):
            r = two_layer(system, n_max, iter_max, B=B)
            ratings = {n: round(v, 4) for n, v in zip(system.candidate_nodes, r.best.s_gfm) if v > 1e-4}
            print(f"  iter_max {iter_max:3d}: {r.evaluated:3d} combos (best at #{r.best_index}), "
                  f"cost {r.cost:.4f}, saving {100 * (1 - r.cost / bench.cost):.2f}%, "
                  f"{r.elapsed:.1f}s, {ratings}")
