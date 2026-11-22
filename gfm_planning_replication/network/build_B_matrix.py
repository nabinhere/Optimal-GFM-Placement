"""Construct the network susceptance matrix B and its partition (Eqs. 6-7).

Inductive network approximation: every branch (line or transformer) is a pure
series reactance, so b_ij = 1 / x_ij in p.u. on the system base. Resistance,
line charging, shunts, loads and off-nominal tap ratios are neglected, which
keeps B a weighted graph Laplacian (symmetric, zero row sums).

SG buses are ideal voltage sources (dU = 0), i.e. grounded. Their rows and
columns are removed, but branches to them stay in the diagonal of their
neighbours: B_ii = sum_{j=1}^{n+m+k} b_ij (paper, below Eq. 6). The resulting
(n+m)x(n+m) matrix is ordered [GFL nodes | interior nodes].

GFM ESS equivalent susceptance model (Eqs. 3, 5, 7): with the converter
impedance dominated by its step-up transformer, a GFM ESS of rating S_GFMi
behaves as a shunt susceptance B_BES * S_GFMi from node i to ground, with
B_BES ~= B_T = 6.25 p.u. (Sec. V).
"""

import numpy as np

B_BES = 6.25  # p.u. unit susceptance of a GFM ESS grid-integration system


def build_susceptance_matrix(branches, buses):
    """Laplacian over all `buses` (SG included): B_ij = -b_ij, B_ii = sum_j b_ij.

    `branches` has columns from_node, to_node, x; parallel branches add up.
    Returns B with rows/columns in the order of `buses`.
    """
    index = {bus: k for k, bus in enumerate(buses)}
    i = branches["from_node"].map(index)
    j = branches["to_node"].map(index)
    if i.isna().any() or j.isna().any():
        raise ValueError("branch endpoint not in bus list")
    i, j = i.to_numpy(dtype=int), j.to_numpy(dtype=int)
    x = branches["x"].to_numpy(dtype=float)
    if np.any(i == j) or np.any(x <= 0):
        raise ValueError("self-loop or non-inductive branch")

    b = 1.0 / x
    B = np.zeros((len(buses), len(buses)))
    np.add.at(B, (i, j), -b)
    np.add.at(B, (j, i), -b)
    np.add.at(B, (i, i), b)
    np.add.at(B, (j, j), b)
    return B


def ground_sg(B_full, buses, sg_buses, keep):
    """Drop the grounded SG rows/columns and reorder to `keep`."""
    index = {bus: k for k, bus in enumerate(buses)}
    if set(keep) & set(sg_buses):
        raise ValueError("SG bus listed among kept buses")
    rows = [index[bus] for bus in keep]
    return B_full[np.ix_(rows, rows)]


def build_B(system):
    """(n+m)x(n+m) susceptance matrix of the non-SG nodes, ordered [GFL | interior]."""
    buses = system.gfl_nodes + system.interior_nodes + system.sg_nodes
    B_full = build_susceptance_matrix(system.branches, buses)
    return ground_sg(B_full, buses, system.sg_nodes, system.candidate_nodes)


def partition(B, n):
    """Split B into B1 (n x n, GFL), B2, B3, B4 (interior) as in Eq. 6."""
    return B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]


def add_gfm(B1, B4, s_gfm_n, s_gfm_m, b_bes=B_BES):
    """Merge GFM ESS shunt susceptances into the diagonals (Eq. 7).

    B1 + B_BES S_GFMn and B4 + B_BES S_GFMm, with S_GFMn / S_GFMm the diagonal
    matrices of GFM ratings at the GFL / interior nodes.
    """
    return (B1 + b_bes * np.diag(s_gfm_n),
            B4 + b_bes * np.diag(s_gfm_m))


def validate_B(B, system, atol=1e-9):
    """Check B against the branch table; raise AssertionError on any mismatch.

    - symmetric, positive diagonal, non-positive off-diagonal
    - every off-diagonal equals -sum(1/x) over the branches joining that pair
    - row sums equal the total susceptance to SG buses (zero elsewhere)
    - positive definite (grounding through the SG makes B invertible)
    """
    nodes = system.candidate_nodes
    index = {bus: k for k, bus in enumerate(nodes)}
    sg = set(system.sg_nodes)

    expected_off = np.zeros_like(B)
    to_sg = np.zeros(len(nodes))
    for f, t, x in system.branches[["from_node", "to_node", "x"]].itertuples(index=False):
        if f in sg and t in sg:
            continue
        if f in sg or t in sg:
            to_sg[index[t if f in sg else f]] += 1.0 / x
            continue
        expected_off[index[f], index[t]] -= 1.0 / x
        expected_off[index[t], index[f]] -= 1.0 / x

    off = B - np.diag(np.diag(B))
    assert np.allclose(B, B.T, atol=atol), "B is not symmetric"
    assert np.all(np.diag(B) > 0), "non-positive diagonal entry"
    assert np.all(off <= atol), "positive off-diagonal entry"
    assert np.allclose(off, expected_off, atol=atol), "off-diagonals differ from branch table"
    assert np.allclose(B.sum(axis=1), to_sg, atol=atol), "row sums differ from SG coupling"
    assert np.linalg.eigvalsh(B).min() > 0, "B is not positive definite"
    return {
        "shape": B.shape,
        "nonzeros_offdiag": int(np.count_nonzero(off)) // 2,
        "sg_neighbours": {nodes[k]: round(v, 4) for k, v in enumerate(to_sg) if v},
        "cond": float(np.linalg.cond(B)),
    }


if __name__ == "__main__":
    from network.load_system import load_ieee39

    system = load_ieee39()
    B = build_B(system)
    print(validate_B(B, system))
