"""gSCR gradients and per-cost site sensitivities (Eqs. 22, 23, 27).

By Eq. 22, d lambda_min / d S_GFMi = x1^T (dA/dS_GFMi) x1, with x1 the unit
eigenvector of A = D B_red D, D = S_GFL^{-1/2}. Evaluating Eq. 23 per node and
using B2 = B3^T gives closed forms that need no per-node matrices:

    GFL node i:       B_BES * (D x1)_i^2
    interior node j:  B_BES * w_j^2,   w = (B4 + B_BES S_GFMm)^{-1} B3 D x1

Both are non-negative: adding GFM never lowers the gSCR.
"""

import numpy as np
from scipy.linalg import solve

from network.build_B_matrix import B_BES, build_B, partition
from network.kron_reduction import reduce_network
from stability.calculate_gscr import calculate_gscr, gscr_with_gfm, symmetric_matrix


def gscr_gradient(system, s_gfm=None, b_bes=B_BES, B=None):
    """gSCR and d lambda_min / d S_GFMi for every candidate node (Eqs. 22-23).

    Returns (gscr, grad) with grad ordered like system.candidate_nodes.
    Warns if lambda_min is (nearly) repeated, where the gradient is not unique.
    """
    if B is None:
        B = build_B(system)
    n = len(system.gfl_nodes)
    s_gfm = np.zeros(B.shape[0]) if s_gfm is None else np.asarray(s_gfm, dtype=float)

    B_red = reduce_network(B, n, s_gfm, b_bes)
    gscr, x1 = calculate_gscr(B_red, system.s_gfl)
    eig = np.linalg.eigvalsh(symmetric_matrix(B_red, system.s_gfl))
    if eig[1] - eig[0] < 1e-6 * max(1.0, abs(eig[0])):
        import warnings
        warnings.warn("lambda_min is (nearly) repeated; gradient is not unique")

    _, _, B3, B4 = partition(B, n)
    y = x1 / np.sqrt(system.s_gfl)  # D x1
    w = solve(B4 + b_bes * np.diag(s_gfm[n:]), B3 @ y, assume_a="pos")
    return gscr, b_bes * np.concatenate([y**2, w**2])


def site_sensitivity(system, cost, b_bes=B_BES, B=None):
    """p_i = (d lambda_min / d S_GFMi) / c_i at S_GFM = 0 (Eq. 27).

    Returns (p, ranking): p ordered like system.candidate_nodes, and the
    candidate nodes sorted by p from largest to smallest.
    """
    _, grad = gscr_gradient(system, None, b_bes, B)
    p = grad / np.broadcast_to(np.asarray(cost, dtype=float), grad.shape)
    order = np.argsort(-p, kind="stable")
    return p, [system.candidate_nodes[k] for k in order]


def check_gradient(system, s_gfm=None, h=1e-6, b_bes=B_BES, B=None):
    """Compare the analytic gradient against central finite differences.

    Returns (analytic, numeric, max relative error).
    """
    if B is None:
        B = build_B(system)
    s_gfm = np.zeros(B.shape[0]) if s_gfm is None else np.asarray(s_gfm, dtype=float)
    _, analytic = gscr_gradient(system, s_gfm, b_bes, B)

    numeric = np.empty_like(analytic)
    for k in range(len(s_gfm)):
        step = np.zeros_like(s_gfm)
        step[k] = h
        up = gscr_with_gfm(system, s_gfm + step, b_bes, B)[0]
        down = gscr_with_gfm(system, s_gfm - step, b_bes, B)[0]
        numeric[k] = (up - down) / (2 * h)

    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-12)
    return analytic, numeric, float(rel.max())


if __name__ == "__main__":
    from network.load_system import load_ieee39

    system = load_ieee39()
    B = build_B(system)
    _, _, err = check_gradient(system, B=B)
    print(f"gradient check at S_GFM = 0: max relative error {err:.2e}")

    p, ranking = site_sensitivity(system, cost=1.0, B=B)
    top = sorted(zip(p, system.candidate_nodes), reverse=True)[:10]
    print("top-10 sensitivity (c_i = 1):", [(node, round(v, 4)) for v, node in top])
