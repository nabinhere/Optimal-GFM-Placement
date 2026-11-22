"""Small-signal and time-domain validation (Fig. 8).

Uses a dq-frame state-space model (GFL PLL + current control, RL network,
GFM as shunt susceptance) instead of ANDES, since RMS phasor models do not
capture the PLL weak-grid oscillation. The GFL model itself lives in
stability.gfl_model.
"""

from stability.gfl_model import critical_gscr, gfl_admittance  # noqa: F401


def build_state_space(system, s_gfm, params):
    """Assemble the closed-loop multi-IBR state-space matrices."""
    raise NotImplementedError


def modal_analysis(A):
    """Eigenvalues and damping ratios of the closed-loop system."""
    raise NotImplementedError


def simulate(system, s_gfm, params, t_end=1.0):
    """Voltage sag at 0.2 s cleared after 0.02 s; integrate with solve_ivp."""
    raise NotImplementedError
