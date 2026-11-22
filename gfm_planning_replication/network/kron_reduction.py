"""Kron reduction onto the GFL nodes (Eqs. 8-9).

Interior nodes carry no current injection (GFM ESSs are already merged into
the diagonal as shunts), so they can be eliminated:

    B = [B1 B2; B3 B4],   B_red = B1 - B2 B4^{-1} B3

B4 is symmetric positive definite (grounded through the SG and GFM shunts),
so the solve uses a Cholesky factorisation.
"""

import numpy as np
from scipy.linalg import solve

from network.build_B_matrix import B_BES, add_gfm, partition


def kron_reduce(B1, B2, B3, B4):
    """B_red = B1 - B2 B4^{-1} B3."""
    B_red = B1 - B2 @ solve(B4, B3, assume_a="pos")
    return 0.5 * (B_red + B_red.T)  # remove round-off asymmetry


def reduce_network(B, n, s_gfm=None, b_bes=B_BES):
    """B_red(S_GFM) = B1 + B_BES S_GFMn - B2 (B4 + B_BES S_GFMm)^{-1} B3 (Eq. 9).

    `s_gfm` holds the GFM ratings of all n+m candidate nodes in B's ordering
    ([GFL | interior]); None means no GFM.
    """
    B1, B2, B3, B4 = partition(B, n)
    if s_gfm is not None:
        s_gfm = np.asarray(s_gfm, dtype=float)
        B1, B4 = add_gfm(B1, B4, s_gfm[:n], s_gfm[n:], b_bes)
    return kron_reduce(B1, B2, B3, B4)


def validate_B_red(B_red, B, n, atol=1e-9):
    """Check B_red; raise AssertionError on any mismatch.

    - n x n, symmetric, positive definite
    - non-positive off-diagonals, non-negative row sums (Kron reduction keeps
      the grounded-Laplacian structure)
    - independent identity: B_red^{-1} equals the GFL block of B^{-1}
    """
    off = B_red - np.diag(np.diag(B_red))
    assert B_red.shape == (n, n), f"expected {n}x{n}, got {B_red.shape}"
    assert np.allclose(B_red, B_red.T, atol=atol), "B_red is not symmetric"
    assert np.linalg.eigvalsh(B_red).min() > 0, "B_red is not positive definite"
    assert np.all(off <= atol), "positive off-diagonal entry"
    assert np.all(B_red.sum(axis=1) >= -atol), "negative row sum"
    assert np.allclose(np.linalg.inv(B_red), np.linalg.inv(B)[:n, :n], atol=1e-8), \
        "B_red^{-1} differs from GFL block of B^{-1}"
    return {
        "shape": B_red.shape,
        "eig_min": float(np.linalg.eigvalsh(B_red).min()),
        "row_sums_to_SG": np.round(B_red.sum(axis=1), 4).tolist(),
    }


if __name__ == "__main__":
    from network.build_B_matrix import build_B
    from network.load_system import load_ieee39

    system = load_ieee39()
    B = build_B(system)
    n = len(system.gfl_nodes)
    B_red = reduce_network(B, n)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(B_red)
    print(validate_B_red(B_red, B, n))
