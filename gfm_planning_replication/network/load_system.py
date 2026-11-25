"""Load the modified IEEE 39-bus system used in the paper.

Branch reactances come from the ANDES ieee39 case and every branch (lines and
transformers) is scaled by 1.5 to make the grid weak (Sec. V-A). Buses are
renumbered to match Fig. 5 of the paper:

    paper 1-9   <- original generator buses 30-35, 37-39 (GFL IBRs)
    paper 10-38 <- original buses 1-29 (shift by 9)
    paper 39    <- original generator bus 36 (generator removed, interior node)

The single SG sits at paper node 37 (original bus 28) and is modeled as an
ideal voltage source, i.e. that node is grounded. Resistances, line charging,
loads and tap ratios are ignored; b_ij = 1 / x_ij.

With these choices the baseline gSCR is 1.5101 (paper: 1.51).

The scaled branch table is exported once to data/ieee39/branches.csv so the
rest of the code does not depend on ANDES.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REACTANCE_SCALE = 1.5
S_BASE = 100.0  # MVA system base
S_GFL = 1.0  # p.u. rating of every GFL IBR (on S_BASE)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ieee39"
BRANCH_CSV = DATA_DIR / "branches.csv"

N_BUS = 39
GFL_NODES = list(range(1, 10))
SG_NODES = [37]

# Fig. 5 numbering: paper node -> original IEEE 39-bus number
PAPER_TO_ORIGINAL = {
    **{i + 9: i for i in range(1, 30)},
    1: 30, 2: 31, 3: 32, 4: 33, 5: 34, 6: 35, 7: 37, 8: 38, 9: 39,
    39: 36,
}
ORIGINAL_TO_PAPER = {o: p for p, o in PAPER_TO_ORIGINAL.items()}


@dataclass
class System:
    """Network data in paper numbering (1-based node labels)."""

    branches: pd.DataFrame  # columns: from_node, to_node, x
    gfl_nodes: list         # n GFL IBR nodes {1..n}
    interior_nodes: list    # m interior nodes {n+1..n+m}
    sg_nodes: list          # k SG nodes (grounded)
    s_gfl: np.ndarray       # GFL power ratings, aligned with gfl_nodes

    @property
    def candidate_nodes(self):
        """Nodes where a GFM ESS may be installed (GFL + interior)."""
        return self.gfl_nodes + self.interior_nodes


def read_andes_branches():
    """Raw (unscaled) branch data from the ANDES ieee39 case, original numbering.

    ANDES stores x on each device's own Sn base; it is converted to the
    S_BASE system base here (a no-op for ieee39, where every Sn is 100 MVA).
    """
    import andes

    line = pd.read_excel(andes.get_case("ieee39/ieee39.xlsx"), sheet_name="Line")
    return pd.DataFrame({
        "from_orig": line["bus1"].astype(int),
        "to_orig": line["bus2"].astype(int),
        "x_orig": line["x"].astype(float) * S_BASE / line["Sn"].astype(float),
        "is_transformer": line["trans"].astype(bool),
    })


def build_branch_table(reactance_scale=REACTANCE_SCALE):
    """Scaled branch table carrying both paper and original bus numbers."""
    raw = read_andes_branches()
    return pd.DataFrame({
        "from_node": raw["from_orig"].map(ORIGINAL_TO_PAPER),
        "to_node": raw["to_orig"].map(ORIGINAL_TO_PAPER),
        "x": raw["x_orig"] * reactance_scale,
        "from_orig": raw["from_orig"],
        "to_orig": raw["to_orig"],
        "x_orig": raw["x_orig"],
        "is_transformer": raw["is_transformer"],
    })


def export_branch_table(path=BRANCH_CSV, reactance_scale=REACTANCE_SCALE):
    """Write the scaled branch table to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_branch_table(reactance_scale)
    table.to_csv(path, index=False)
    return path


def load_ieee39(reactance_scale=REACTANCE_SCALE, path=BRANCH_CSV):
    """Return the modified IEEE 39-bus system in paper numbering.

    Reads the exported CSV when it exists, otherwise builds it from ANDES.
    """
    path = Path(path)
    if path.exists():
        table = pd.read_csv(path)
        table["x"] = table["x_orig"] * reactance_scale
    else:
        table = build_branch_table(reactance_scale)

    interior = [i for i in range(1, N_BUS + 1) if i not in GFL_NODES + SG_NODES]
    return System(
        branches=table[["from_node", "to_node", "x"]].reset_index(drop=True),
        gfl_nodes=list(GFL_NODES),
        interior_nodes=interior,
        sg_nodes=list(SG_NODES),
        s_gfl=np.full(len(GFL_NODES), S_GFL),
    )


# ---------------------------------------------------------------------------
# Modified IEEE 118-bus system (Sec. V-C, Fig. 10)
#
# Reactances are the standard MATPOWER case118 values (bundled with ANDES),
# unscaled. Fig. 10 swaps labels between paper nodes 1-9 and the original
# generator buses that host the GFL IBRs; all other buses keep their number:
#
#     paper i <-> original G118[i-1],  i = 1..9
#
# Two SGs (ideal sources) stay at buses 69 and 111; the other generators are
# removed. Fig. 10 counts all 118 buses as candidates, but the SG buses are
# grounded here, leaving 116 candidates.
#
# The baseline gSCR is 1.6055 (paper: 1.69). The mapping was checked against
# every GFL node's neighbours in Fig. 10; tap, SG-set and numbering variants
# did not close the gap, so [41] likely carries slightly different data.
# ---------------------------------------------------------------------------

DATA_DIR_118 = DATA_DIR.parent / "ieee118"
BRANCH_CSV_118 = DATA_DIR_118 / "branches.csv"
N_BUS_118 = 118
G118 = [10, 12, 25, 26, 31, 46, 49, 54, 59]
SG_NODES_118 = [69, 111]
ORIGINAL_TO_PAPER_118 = {o: o for o in range(1, N_BUS_118 + 1)}
for _i, _g in enumerate(G118, start=1):
    ORIGINAL_TO_PAPER_118[_g], ORIGINAL_TO_PAPER_118[_i] = _i, _g


def read_matpower_branches(path=None):
    """In-service branches of MATPOWER case118 (x already on the 100 MVA base)."""
    import re

    if path is None:
        import andes
        path = andes.get_case("matpower/case118.m")
    text = Path(path).read_text()
    block = re.search(r"mpc\.branch\s*=\s*\[(.*?)\];", text, re.S).group(1)
    rows = [line.split("%")[0].strip().rstrip(";").split() for line in block.splitlines()]
    data = np.array([row for row in rows if row], dtype=float)
    data = data[data[:, 10] != 0]  # status column
    return pd.DataFrame({
        "from_orig": data[:, 0].astype(int),
        "to_orig": data[:, 1].astype(int),
        "x_orig": data[:, 3],
        "is_transformer": data[:, 8] != 0,  # nonzero tap ratio
    })


def export_branch_table_118(path=BRANCH_CSV_118):
    raw = read_matpower_branches()
    table = pd.DataFrame({
        "from_node": raw["from_orig"].map(ORIGINAL_TO_PAPER_118),
        "to_node": raw["to_orig"].map(ORIGINAL_TO_PAPER_118),
        "x": raw["x_orig"],
        **raw,
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_ieee118(path=BRANCH_CSV_118):
    """Return the modified IEEE 118-bus system in Fig. 10 numbering."""
    path = Path(path)
    if not path.exists():
        export_branch_table_118(path)
    table = pd.read_csv(path)
    gfl = list(range(1, 10))
    interior = [i for i in range(1, N_BUS_118 + 1) if i not in gfl + SG_NODES_118]
    return System(
        branches=table[["from_node", "to_node", "x"]].reset_index(drop=True),
        gfl_nodes=gfl,
        interior_nodes=interior,
        sg_nodes=list(SG_NODES_118),
        s_gfl=np.full(len(gfl), S_GFL),
    )


if __name__ == "__main__":
    print(f"Wrote {export_branch_table()}")
    print(f"Wrote {export_branch_table_118()}")
