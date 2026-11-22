"""Generalized short-circuit ratio (Eq. 13)."""


def symmetric_matrix(B_red, s_gfl):
    """A = S_GFL^{-1/2} B_red S_GFL^{-1/2}, same spectrum as S_GFL^{-1} B_red."""
    raise NotImplementedError


def calculate_gscr(B_red, s_gfl):
    """gSCR = lambda_min(A); also return the corresponding eigenvector."""
    raise NotImplementedError


def gscr_with_gfm(system, s_gfm, b_bes=6.25):
    """Convenience wrapper: add GFM, Kron-reduce, and return gSCR."""
    raise NotImplementedError
