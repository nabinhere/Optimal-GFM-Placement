"""Generalized short-circuit ratio (Eq. 13).

    gSCR = lambda_min(S_GFL^{-1} B_red)

S_GFL^{-1} B_red is not symmetric, so the similar symmetric matrix
A = S_GFL^{-1/2} B_red S_GFL^{-1/2} is used instead (Sec. IV-A). Both have the
same eigenvalues, and A allows a symmetric eigensolver with real, orthonormal
eigenvectors. The eigenvector x1 of A is what Eq. 22 needs for the gradient.
"""

import numpy as np
from scipy.linalg import eigh

from network.build_B_matrix import build_B
from network.kron_reduction import reduce_network


def symmetric_matrix(B_red, s_gfl):
    """A = S_GFL^{-1/2} B_red S_GFL^{-1/2}, same spectrum as S_GFL^{-1} B_red."""
    d = 1.0 / np.sqrt(np.asarray(s_gfl, dtype=float))
    A = d[:, None] * B_red * d[None, :]
    return 0.5 * (A + A.T)


def calculate_gscr(B_red, s_gfl):
    """gSCR = lambda_min(A) and the unit eigenvector x1 of A (Eq. 13).

    The sign of x1 is fixed so its entries sum to a positive number.
    """
    A = symmetric_matrix(B_red, s_gfl)
    lam, vec = eigh(A, subset_by_index=[0, 0])
    x1 = vec[:, 0]
    if x1.sum() < 0:
        x1 = -x1
    return float(lam[0]), x1


def gscr_with_gfm(system, s_gfm=None, b_bes=6.25, B=None):
    """gSCR of `system` with GFM ratings `s_gfm` on its candidate nodes.

    `s_gfm` is ordered like system.candidate_nodes ([GFL | interior]);
    None means no GFM. Pass a prebuilt B to skip rebuilding it.
    """
    if B is None:
        B = build_B(system)
    B_red = reduce_network(B, len(system.gfl_nodes), s_gfm, b_bes)
    return calculate_gscr(B_red, system.s_gfl)


if __name__ == "__main__":
    from network.load_system import load_ieee39

    system = load_ieee39()
    gscr, x1 = gscr_with_gfm(system)
    np.set_printoptions(precision=4, suppress=True)
    print(f"baseline gSCR = {gscr:.4f}  (paper: 1.51)")
    print(f"x1 = {x1}")
