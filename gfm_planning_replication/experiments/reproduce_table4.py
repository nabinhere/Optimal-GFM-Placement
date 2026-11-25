"""Reproduce Table IV (Scenario 2 on the modified IEEE 118-bus system, gSCR = 3).

Usage: python -m experiments.reproduce_table4 [--top K] [--workers W]

  * centralized benchmark (paper: 2.0522 p.u.) and Scenario 1 for reference;
  * two-layer method for N_max = 3, 4, 5. One run with iter_max = 150 per N_max
    gives the rows for iter_max = 50 and 100 too (scenario2.stop_point);
  * with --top K, the first K ranked combinations (paper: 40000) are sized in
    parallel as the "/" reference rows; per-combination results are appended to
    results/ieee118_top{K}_nmax{N}.csv, so an interrupted run resumes.

Outputs: results/table_IV.csv and results/ieee118_scenario1.csv.
"""

import argparse
import csv
import os
import time
from itertools import islice
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.reproduce_fig6_fig7 import RESULTS  # noqa: E402
from network.build_B_matrix import build_B  # noqa: E402
from network.load_system import load_ieee118  # noqa: E402
from optimization.scenario1 import algorithm1, proportional_benchmark, solve_sdp  # noqa: E402
from optimization.scenario2 import (centralized_benchmark, ranked_combinations,  # noqa: E402
                                    solve_combination, stop_point, two_layer)
from stability.sensitivity import site_sensitivity  # noqa: E402

TARGET = 3.0
N_MAX = [3, 4, 5]
ITER_MAX = [50, 100, 150]
PAPER = {  # Table IV: (combinations, time s, cost) per (N_max, iter_max)
    (3, 50): (156, 3759, 1.2681), (3, 100): (206, 4960, 1.2681),
    (3, 150): (256, 6160, 1.2681), (3, None): (40000, 1424100, 1.2681),
    (4, 50): (109, 2730, 1.5346), (4, 100): (277, 6936, 1.2615),
    (4, 150): (327, 8188, 1.2615), (4, None): (40000, 808330, 1.2615),
    (5, 50): (89, 2216, 1.5346), (5, 100): (288, 7171.2, 1.2615),
    (5, 150): (338, 8416.2, 1.2615), (5, None): (40000, 995889, 1.2605),
}
PAPER_BENCHMARK = 2.0522

_system = _B = None


def _init():
    global _system, _B
    _system = load_ieee118()
    _B = build_B(_system)


def _solve(sites):
    try:
        r = solve_combination(_system, sites, target=TARGET, B=_B)
    except (RuntimeError, np.linalg.LinAlgError):
        return {"sites": " ".join(map(str, sites)), "cost": np.inf, "solver": "failed"}
    return {"sites": " ".join(map(str, sites)),
            "cost": np.inf if r is None else r.cost,
            "solver": "infeasible" if r is None else r.solver}


def top_k(system, n_max, k, workers):
    """Size the first k ranked combinations in parallel; return the best row."""
    _, ranking = site_sensitivity(system, 1.0, B=build_B(system))
    out = RESULTS / f"ieee118_top{k}_nmax{n_max}.csv"
    done = set()
    if out.exists():
        done = set(pd.read_csv(out, dtype={"sites": str}).sites)
    todo = [c for c in islice(ranked_combinations(ranking, n_max), k)
            if " ".join(map(str, c)) not in done]
    print(f"  top {k}, N_max = {n_max}: {len(done)} done, {len(todo)} to go", flush=True)
    start = time.perf_counter()
    with open(out, "a", newline="") as f, Pool(workers, initializer=_init) as pool:
        writer = csv.DictWriter(f, ["sites", "cost", "solver"])
        if not done:
            writer.writeheader()
        for j, row in enumerate(pool.imap_unordered(_solve, todo, chunksize=8), start=1):
            writer.writerow(row)
            if j % 2000 == 0:
                f.flush()
                print(f"    {j}/{len(todo)}, {time.perf_counter() - start:.0f}s", flush=True)
    data = pd.read_csv(out, dtype={"sites": str})
    return data.loc[data.cost.idxmin()], time.perf_counter() - start


def main(top, workers):
    RESULTS.mkdir(exist_ok=True)
    system = load_ieee118()
    B = build_B(system)

    bench, node = centralized_benchmark(system, target=TARGET, B=B)
    print(f"centralized benchmark: node {node}, cost {bench.cost:.4f} (paper {PAPER_BENCHMARK})")

    prop = proportional_benchmark(system, target=TARGET, B=B)
    alg1 = algorithm1(system, target=TARGET, B=B)
    sdp = solve_sdp(system, target=TARGET, B=B)
    print(f"Scenario 1: proportional {prop.cost:.4f}, Alg. 1 {alg1.cost:.4f} "
          f"({alg1.iterations} iters, gamma {alg1.gamma}), SDP {sdp.cost:.4f}, "
          f"reduction {100 * (1 - alg1.cost / prop.cost):.2f}% (paper 33.18%)")
    pd.DataFrame({"node": system.candidate_nodes, "rating": alg1.s_gfm}).query("rating > 1e-4") \
        .to_csv(RESULTS / "ieee118_scenario1.csv", index=False)

    rows = []
    for n_max in N_MAX:
        r = two_layer(system, n_max, max(ITER_MAX), target=TARGET, B=B)
        for iter_max in ITER_MAX:
            kappa = stop_point(r.best_cost_trace, iter_max)
            cost = r.best_cost_trace[kappa - 1]
            paper_combos, paper_time, paper_cost = PAPER[n_max, iter_max]
            rows.append({"n_max": n_max, "iter_max": iter_max, "combinations": kappa,
                         "time_s": round(r.time_trace[kappa - 1], 1), "cost": round(cost, 4),
                         "reduction_pct": round(100 * (1 - cost / bench.cost), 2),
                         "paper_combinations": paper_combos, "paper_time_s": paper_time,
                         "paper_cost": paper_cost})
        print(f"N_max = {n_max}: best {r.cost:.4f} at {r.sites} (combination #{r.best_index})",
              flush=True)
        if top:
            best, elapsed = top_k(system, n_max, top, workers)
            paper_combos, paper_time, paper_cost = PAPER[n_max, None]
            rows.append({"n_max": n_max, "iter_max": None, "combinations": top,
                         "time_s": round(elapsed, 1), "cost": round(best.cost, 4),
                         "reduction_pct": round(100 * (1 - best.cost / bench.cost), 2),
                         "paper_combinations": paper_combos, "paper_time_s": paper_time,
                         "paper_cost": paper_cost})
            print(f"  top {top}: best {best.cost:.4f} at ({best.sites})", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "table_IV.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 6))
    args = parser.parse_args()
    main(args.top, args.workers)
