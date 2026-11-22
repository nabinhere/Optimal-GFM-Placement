"""Load the modified IEEE 39-bus system used in the paper.

Branch reactances come from the ANDES ieee39 case, are scaled by 1.5 to make the
grid weak (Sec. V-A), and are exported once to data/ieee39/ so the rest of the
code does not depend on ANDES.
"""

REACTANCE_SCALE = 1.5
S_GFL = 1.0  # p.u. rating of every GFL IBR


def load_ieee39(reactance_scale=REACTANCE_SCALE):
    """Return the branch table (from, to, x) and the GFL / interior / SG bus sets."""
    raise NotImplementedError


def export_branch_table(path):
    """Write the scaled branch table to CSV for reproducibility."""
    raise NotImplementedError
