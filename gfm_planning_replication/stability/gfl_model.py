"""GFL IBR impedance model and critical gSCR (Eqs. 1, 14).

The paper does not list the GFL control structure or parameters (it cites
[24], [25]). A standard SRF-PLL + PI current control + L filter is assumed and
tuned so that CgSCR ~= 2.5, matching Fig. 8. All GFL units are homogeneous,
so one per-unit admittance Y_GFL(s) serves every node.

gSCR and the planning problems do not need this module; only CgSCR (and thus
the target CgSCR + dg0 = 3.5) and the dynamic validation do.
"""


def gfl_admittance(s, params):
    """Per-unit 2x2 GFL admittance matrix Y_GFL(s) in the xy frame (Eq. 1)."""
    raise NotImplementedError


def critical_gscr(params):
    """Solve Eq. 14 for CgSCR on the single-IBR equivalent (target ~2.5)."""
    raise NotImplementedError
