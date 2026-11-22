"""Small-signal and time-domain validation (Fig. 8).

Uses a dq-frame state-space model (GFL PLL + current control, RL network,
GFM as shunt susceptance) instead of ANDES, since RMS phasor models do not
capture the PLL weak-grid oscillation.
"""


def gfl_admittance(s, params):
    """Per-unit GFL admittance matrix Y_GFL(s)."""
    raise NotImplementedError


def critical_gscr(params):
    """Solve Eq. 14 for CgSCR on the single-IBR equivalent (target ~2.5)."""
    raise NotImplementedError


def build_state_space(system, s_gfm, params):
    """Assemble the closed-loop multi-IBR state-space matrices."""
    raise NotImplementedError


def modal_analysis(A):
    """Eigenvalues and damping ratios of the closed-loop system."""
    raise NotImplementedError


def simulate(system, s_gfm, params, t_end=1.0):
    """Voltage sag at 0.2 s cleared after 0.02 s; integrate with solve_ivp."""
    raise NotImplementedError
