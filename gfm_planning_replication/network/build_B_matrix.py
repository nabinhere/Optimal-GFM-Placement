"""Construct the network susceptance matrix B and its partition (Eq. 6)."""


def build_susceptance_matrix(branches, buses):
    """B_ij = -b_ij (i != j), B_ii = sum_j b_ij, including branches to SG buses."""
    raise NotImplementedError


def partition(B, gfl_buses, interior_buses):
    """Split B into B1 (GFL), B2, B3, B4 (interior); SG buses are grounded and dropped."""
    raise NotImplementedError


def add_gfm(B1, B4, s_gfm_n, s_gfm_m, b_bes=6.25):
    """Merge GFM ESS shunt susceptances B_BES * S_GFM into the diagonal (Eq. 7)."""
    raise NotImplementedError
