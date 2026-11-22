"""gSCR gradients and per-cost site sensitivities (Eqs. 22, 23, 27)."""

from network.build_B_matrix import B_BES


def gscr_gradient(system, s_gfm, b_bes=B_BES):
    """d lambda_min / d S_GFMi for every candidate node (Eqs. 22-23)."""
    raise NotImplementedError


def site_sensitivity(system, cost, b_bes=B_BES):
    """p_i = (d lambda_min / d S_GFMi) / c_i evaluated at S_GFM = 0 (Eq. 27)."""
    raise NotImplementedError


def check_gradient(system, s_gfm, h=1e-6):
    """Compare the analytic gradient against central finite differences."""
    raise NotImplementedError
