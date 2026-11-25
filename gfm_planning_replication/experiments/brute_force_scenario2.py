"""Brute-force reference for Scenario 2: size every N_max-site combination.

Usage: python -m experiments.brute_force_scenario2 N_MAX [WORKERS]

Combinations are spread over a process pool and each result is appended to
results/brute_force_nmax{N}.csv as soon as it is done, so an interrupted run
resumes where it stopped. When all combinations are done, the global optimum
is printed and compared with the two-layer result.
"""

import csv
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

# One BLAS thread per worker; set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIELDS = ["sites", "cost", "gscr", "solver", "iterations", "ratings"]

_system = _B = None


def _init():
    global _system, _B
    from network.build_B_matrix import build_B
    from network.load_system import load_ieee39
    _system = load_ieee39()
    _B = build_B(_system)


def _solve(sites):
    from optimization.scenario2 import solve_combination
    try:
        r = solve_combination(_system, sites, B=_B)
    except (RuntimeError, np.linalg.LinAlgError):
        return {"sites": " ".join(map(str, sites)), "cost": np.inf, "gscr": "",
                "solver": "failed", "iterations": 0, "ratings": ""}
    if r is None:
        return {"sites": " ".join(map(str, sites)), "cost": np.inf, "gscr": "",
                "solver": "infeasible", "iterations": 0, "ratings": ""}
    idx = [_system.candidate_nodes.index(site) for site in sites]
    return {"sites": " ".join(map(str, sites)), "cost": r.cost, "gscr": r.gscr,
            "solver": r.solver, "iterations": r.iterations,
            "ratings": " ".join(f"{r.s_gfm[i]:.6f}" for i in idx)}


def main(n_max, workers):
    from network.load_system import load_ieee39
    nodes = load_ieee39().candidate_nodes
    out = RESULTS / f"brute_force_nmax{n_max}.csv"

    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {row["sites"] for row in csv.DictReader(f)}
    todo = [c for c in combinations(nodes, n_max) if " ".join(map(str, c)) not in done]
    total = len(done) + len(todo)
    print(f"N_max = {n_max}: {total} combinations, {len(done)} already done, "
          f"{workers} workers", flush=True)

    start = time.perf_counter()
    with open(out, "a", newline="") as f, Pool(workers, initializer=_init) as pool:
        writer = csv.DictWriter(f, FIELDS)
        if not done:
            writer.writeheader()
        for k, row in enumerate(pool.imap_unordered(_solve, todo, chunksize=8), start=1):
            writer.writerow(row)
            if k % 500 == 0 or k == len(todo):
                f.flush()
                rate = k / (time.perf_counter() - start)
                print(f"  {len(done) + k}/{total} done, "
                      f"{(len(todo) - k) / rate / 60:.1f} min left", flush=True)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda row: float(row["cost"]))
    solvers = {}
    for row in rows:
        solvers[row["solver"]] = solvers.get(row["solver"], 0) + 1
    print(f"global optimum {float(best['cost']):.4f} at ({best['sites']}), "
          f"ratings {best['ratings']}; solvers {solvers}; "
          f"{time.perf_counter() - start:.0f}s this run", flush=True)


if __name__ == "__main__":
    n_max = int(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, os.cpu_count() - 6)
    main(n_max, workers)
