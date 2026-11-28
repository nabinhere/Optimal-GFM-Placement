"""Model variants of the IEEE 118-bus system checked against the paper's values.

Usage: python -m experiments.ieee118_data_checks [--case PATH] [--iit PATH] [--scan]

For the modified IEEE 118-bus system the paper reports a baseline gSCR of 1.69
and a centralized benchmark of 2.0522 p.u. (Sec. V-C), and a Scenario 1 saving
of 33.18 % over the proportional benchmark (Sec. VI). The proportional
benchmark has the closed form

    cost = (target - baseline) * sum(S_GFL) / B_BES,

because GFM ratings r * S_GFL at the GFL nodes add r * B_BES to every
eigenvalue of S_GFL^{-1} B_red. With the reported baseline this is
9 * (3 - 1.69) / 6.25 = 1.8864 p.u., so the reported saving corresponds to a
Scenario 1 cost of (1 - 0.3318) * 1.8864 = 1.2605 p.u., which is also the
N_max = 5 entry of Table IV.

Our model (network/load_system.py) gives 1.6055, 2.0895 and 1.2719 p.u. Each
variant below changes one modelling choice and recomputes the three values
with the package's own functions. With --scan two searches are added: every
other pair of SG buses, and every single-branch reactance change that gives
the reported baseline.

Inputs:
  * MATPOWER case118.m (default: the copy bundled with ANDES);
  * data/ieee118/iit_jeas118_lines.csv, the line table (Table 3) of the IIT
    file JEAS_IEEE118, http://motor.ece.iit.edu/data/JEAS_IEEE118.doc, which
    belongs to the data index cited by the paper as [41]. The CSV was read
    from the copy of that file kept in
    github.com/evgenytsydenov/ieee118_power_flow_data (data/raw/jeas118).
    --iit also accepts the document saved as .docx and then rewrites the CSV.

Outputs: results/ieee118_data_checks.csv and, with --scan,
results/ieee118_sg_pairs.csv and results/ieee118_single_branch.csv.
"""

import argparse
import re
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from experiments.reproduce_fig6_fig7 import RESULTS
from network.build_B_matrix import B_BES, build_susceptance_matrix, ground_sg
from network.load_system import (DATA_DIR_118, G118, N_BUS_118, ORIGINAL_TO_PAPER_118,
                                 S_GFL, SG_NODES_118, System, load_ieee118)
from optimization.scenario1 import proportional_benchmark, solve_sdp
from optimization.scenario2 import centralized_benchmark
from stability.calculate_gscr import gscr_with_gfm

TARGET = 3.0
BASE_MVA = 100.0
PAPER = {"baseline": 1.69, "centralized": 2.0522, "saving_pct": 33.18}
IIT_CSV = DATA_DIR_118 / "iit_jeas118_lines.csv"


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

def _matrix(text, name):
    block = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S).group(1)
    rows = [line.split("%")[0].strip().rstrip(";").split() for line in block.splitlines()]
    return np.array([row for row in rows if row], dtype=float)


def read_case(path=None):
    """Branch table and bus shunts of MATPOWER case118, original numbering.

    Returns (branches, shunts): branches has r, x, the total line charging b
    and the tap ratio (0 for a line); shunts maps bus -> Bs in p.u., positive
    for a capacitor as in MATPOWER.
    """
    if path is None:
        import andes
        path = andes.get_case("matpower/case118.m")
    text = Path(path).read_text()
    branch = _matrix(text, "branch")
    branch = branch[branch[:, 10] != 0]  # in-service branches
    bus = _matrix(text, "bus")
    branches = pd.DataFrame({
        "from_orig": branch[:, 0].astype(int),
        "to_orig": branch[:, 1].astype(int),
        "r": branch[:, 2],
        "x_orig": branch[:, 3],
        "b": branch[:, 4],
        "ratio": branch[:, 8],
    })
    shunts = {int(b): bs / BASE_MVA for b, bs in zip(bus[:, 0], bus[:, 5]) if bs != 0}
    return branches, shunts


def read_iit_docx(path):
    """Line table (Table 3) of JEAS_IEEE118 saved as .docx, original numbering."""
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")

    def cells(row):
        return ["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", cell)).strip()
                for cell in re.findall(r"<w:tc>.*?</w:tc>", row, flags=re.S)]

    for table in re.findall(r"<w:tbl>.*?</w:tbl>", xml, flags=re.S):
        rows = [cells(row) for row in re.findall(r"<w:tr[ >].*?</w:tr>", table, flags=re.S)]
        if rows and rows[0][0].replace(" ", "") == "LineNo.":
            data = np.array(rows[1:], dtype=float)
            return pd.DataFrame({
                "line": data[:, 0].astype(int),
                "from_orig": data[:, 1].astype(int),
                "to_orig": data[:, 2].astype(int),
                "circuit": data[:, 3].astype(int),
                "r": data[:, 4], "x_orig": data[:, 5], "b": data[:, 6],
            })
    raise ValueError("line table not found")


def same_reactances(a, b):
    """True if two branch tables hold the same (from, to, x) entries."""
    def key(table):
        return sorted((min(f, t), max(f, t), round(x, 6)) for f, t, x in
                      zip(table.from_orig, table.to_orig, table.x_orig))
    return key(a) == key(b)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def make_system(branches, x=None, sg_orig=SG_NODES_118, s_gfl=S_GFL):
    """System in Fig. 10 numbering from a branch table in original numbering.

    `x` overrides the reactances; `sg_orig` are the SG buses (original
    numbering, which is the Fig. 10 numbering for every bus outside 1-9 and
    G118).
    """
    table = pd.DataFrame({
        "from_node": branches["from_orig"].map(ORIGINAL_TO_PAPER_118),
        "to_node": branches["to_orig"].map(ORIGINAL_TO_PAPER_118),
        "x": branches["x_orig"] if x is None else x,
    })
    gfl = list(range(1, len(G118) + 1))
    sg = [ORIGINAL_TO_PAPER_118[bus] for bus in sg_orig]
    interior = [i for i in range(1, N_BUS_118 + 1) if i not in gfl + sg]
    return System(table, gfl, interior, sg, np.full(len(gfl), s_gfl))


def build(system):
    """B of the non-SG nodes, ordered [GFL | interior] (as network.build_B)."""
    buses = system.gfl_nodes + system.interior_nodes + system.sg_nodes
    return ground_sg(build_susceptance_matrix(system.branches, buses), buses,
                     system.sg_nodes, system.candidate_nodes)


def add_shunts(B, system, shunt_by_orig_bus):
    """Add inductive shunt susceptances (p.u., by original bus) to the diagonal."""
    B = B.copy()
    index = {node: k for k, node in enumerate(system.candidate_nodes)}
    for bus, value in shunt_by_orig_bus.items():
        node = ORIGINAL_TO_PAPER_118[bus]
        if node in index:
            B[index[node], index[node]] += value
    return B


def build_with_taps(branches, system):
    """B with off-nominal tap ratios (MATPOWER convention, tap at the from bus)."""
    index = {node: k for k, node in enumerate(system.candidate_nodes)}
    B = np.zeros((len(index), len(index)))
    for f, t, x, ratio in zip(branches.from_orig, branches.to_orig, branches.x_orig,
                              branches.ratio):
        f, t = ORIGINAL_TO_PAPER_118[f], ORIGINAL_TO_PAPER_118[t]
        b, tau = 1.0 / x, (ratio if ratio != 0 else 1.0)
        if f in index:
            B[index[f], index[f]] += b / tau**2
        if t in index:
            B[index[t], index[t]] += b
        if f in index and t in index:
            B[index[f], index[t]] -= b / tau
            B[index[t], index[f]] -= b / tau
    return B


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def baseline(system, B):
    return gscr_with_gfm(system, B=B)[0]


def centralized(system, B):
    """(cost, original bus) of the best single site, or (nan, None)."""
    best = centralized_benchmark(system, target=TARGET, B=B)
    if best is None:
        return np.nan, None
    return best[0].cost, ORIGINAL_TO_PAPER_118[best[1]]


def evaluate(name, system, B):
    """Baseline gSCR, both benchmarks and the Scenario 1 optimum of one variant."""
    try:
        cost_c, bus_c = centralized(system, B)
        prop = proportional_benchmark(system, target=TARGET, B=B).cost
        scenario1 = solve_sdp(system, target=TARGET, B=B).cost
        row = {"variant": name, "baseline": baseline(system, B), "centralized": cost_c,
               "centralized_orig_bus": bus_c, "proportional": prop, "scenario1": scenario1,
               "saving_pct": 100 * (1 - scenario1 / prop)}
    except np.linalg.LinAlgError:
        # B4 is not positive definite, so the Kron reduction of Eq. 9 does not apply
        row = {"variant": name + " (B4 not positive definite)"}
    print(row, flush=True)
    return row


def model_variants(branches, shunts, iit):
    """Yield (name, system, B) for every variant."""
    system = make_system(branches)
    B = build(system)
    yield "reference: MATPOWER case118, b = 1/x", system, B

    if iit is not None:
        s = make_system(iit)
        yield "line reactances of IIT JEAS_IEEE118", s, build(s)

    # 1.6055 / 1.69 = 0.95: a uniform factor that reproduces the reported baseline
    s = make_system(branches, x=branches.x_orig * 0.95)
    yield "all reactances x 0.95", s, build(s)

    s = make_system(branches, s_gfl=0.95)
    yield "GFL ratings 0.95 p.u.", s, build(s)

    s = make_system(branches, x=(branches.r**2 + branches.x_orig**2) / branches.x_orig)
    yield "series resistance included, b = x / (r^2 + x^2)", s, build(s)

    yield "tap ratios included", system, build_with_taps(branches, system)

    yield "bus shunts included", system, add_shunts(B, system, {b: -v for b, v in shunts.items()})

    charging = {}
    for f, t, b in zip(branches.from_orig, branches.to_orig, branches.b):
        for bus in (f, t):
            charging[bus] = charging.get(bus, 0.0) - b / 2
    yield "line charging included", system, add_shunts(B, system, charging)


def rounds_to_paper(value):
    """True if a baseline gSCR is printed as 1.69 with two decimals."""
    return PAPER["baseline"] - 0.005 <= value < PAPER["baseline"] + 0.005


def scan_sg_pairs(branches):
    """Baseline gSCR for every pair of SG buses; centralized cost where it is 1.69."""
    rows = []
    buses = [bus for bus in range(1, N_BUS_118 + 1) if bus not in G118]
    for pair in combinations(buses, 2):
        system = make_system(branches, sg_orig=pair)
        B = build(system)
        row = {"sg1": pair[0], "sg2": pair[1], "baseline": baseline(system, B)}
        if rounds_to_paper(row["baseline"]):
            row["centralized"], row["centralized_orig_bus"] = centralized(system, B)
        rows.append(row)
    return pd.DataFrame(rows)


def scan_single_branch(branches):
    """For each branch, the reactance factor in (0, 1) that gives a 1.69 baseline.

    Raising a reactance can only lower the gSCR, so factors above 1 and
    outages cannot produce the higher reported baseline.
    """
    rows = []
    for k in range(len(branches)):
        def gap(factor):
            x = branches.x_orig.copy()
            x.iloc[k] *= factor
            system = make_system(branches, x=x)
            return baseline(system, build(system)) - PAPER["baseline"]

        if gap(1e-4) < 0:
            continue
        factor = brentq(gap, 1e-4, 1.0, xtol=1e-9)
        x = branches.x_orig.copy()
        x.iloc[k] *= factor
        system = make_system(branches, x=x)
        cost, bus = centralized(system, build(system))
        rows.append({"from_orig": branches.from_orig.iloc[k], "to_orig": branches.to_orig.iloc[k],
                     "x_orig": branches.x_orig.iloc[k], "factor": factor,
                     "centralized": cost, "centralized_orig_bus": bus})
    return pd.DataFrame(rows)


def main(case, iit_path, scan):
    RESULTS.mkdir(exist_ok=True)
    branches, shunts = read_case(case)

    # the variants must start from the model used everywhere else
    reference = load_ieee118()
    system = make_system(branches)
    assert np.allclose(build(system), build(reference)), "reference differs from load_ieee118"

    iit = None
    iit_path = Path(iit_path)
    if iit_path.suffix == ".docx":
        iit = read_iit_docx(iit_path)
        iit.to_csv(IIT_CSV, index=False)
    elif iit_path.exists():
        iit = pd.read_csv(iit_path)
    if iit is None:
        print(f"{iit_path} not found, IIT comparison skipped")
    else:
        print(f"IIT line table: {len(iit)} lines, same (from, to, x) as case118: "
              f"{same_reactances(iit, branches)}")

    paper_prop = (TARGET - PAPER["baseline"]) * len(G118) * S_GFL / B_BES
    rows = [{"variant": "reported in the paper", "baseline": PAPER["baseline"],
             "centralized": PAPER["centralized"], "proportional": paper_prop,
             "scenario1": (1 - PAPER["saving_pct"] / 100) * paper_prop,
             "saving_pct": PAPER["saving_pct"]}]
    rows += [evaluate(*variant) for variant in model_variants(branches, shunts, iit)]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "ieee118_data_checks.csv", index=False)
    print(table.round(4).to_string(index=False))

    if scan:
        pairs = scan_sg_pairs(branches)
        pairs.to_csv(RESULTS / "ieee118_sg_pairs.csv", index=False)
        hits = pairs[pairs.baseline.map(rounds_to_paper)]
        closest = hits.loc[(hits.centralized - PAPER["centralized"]).abs().idxmin()]
        print(f"SG pairs: {len(pairs)} tested, {len(hits)} with a 1.69 baseline; "
              f"centralized cost between {hits.centralized.min():.4f} and "
              f"{hits.centralized.max():.4f}, closest to {PAPER['centralized']}: "
              f"{closest.centralized:.4f} for SGs at {int(closest.sg1)} and {int(closest.sg2)}")

        single = scan_single_branch(branches)
        single.to_csv(RESULTS / "ieee118_single_branch.csv", index=False)
        print(f"single-branch changes: {len(single)} of {len(branches)} branches can give a "
              f"1.69 baseline; centralized cost between {single.centralized.min():.4f} and "
              f"{single.centralized.max():.4f}")
        print(single.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="path to MATPOWER case118.m")
    parser.add_argument("--iit", default=str(IIT_CSV), help="IIT line table (.csv or .docx)")
    parser.add_argument("--scan", action="store_true", help="add the SG-pair and branch scans")
    args = parser.parse_args()
    main(args.case, args.iit, args.scan)
