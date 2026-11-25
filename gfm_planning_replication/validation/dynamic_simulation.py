"""Small-signal and time-domain validation (Fig. 8).

The planned network (B, GFM ESSs as ideal EMFs behind B_BES * S_GFM, SG as an
ideal source) is reduced to the GFL nodes,

    V = Z (dI_sys/dt / w0 + j I_sys) + H E,   Z = B_red(S_GFM)^-1,

and closed with the nonlinear GFL model of stability.gfl_model. E holds the
source EMFs: column 0 is the SG (all SG nodes share one EMF), the remaining
columns are the GFM ESSs. A voltage sag is applied to the SG EMF.
"""

import numpy as np
from scipy.integrate import solve_ivp

from network.build_B_matrix import B_BES, build_B
from network.kron_reduction import reduce_network
from stability.gfl_model import GFLParams, MultiInfeed


def thevenin(system, s_gfm=None, b_bes=B_BES, B=None):
    """Z = B_red^-1 at the GFL nodes and H mapping source EMFs to V_th."""
    if B is None:
        B = build_B(system)
    n_cand = len(system.candidate_nodes)
    n = len(system.gfl_nodes)
    s = np.zeros(n_cand) if s_gfm is None else np.asarray(s_gfm, dtype=float)
    gfm = np.flatnonzero(s > 1e-9)

    b_sg = B.sum(axis=1)                    # coupling of every node to the SG(s)
    shunt = b_bes * s
    B_nn = B + np.diag(shunt)
    sources = np.column_stack([b_sg] + [shunt[k] * np.eye(n_cand)[:, k] for k in gfm])
    H = np.linalg.solve(B_nn, sources)[:n]  # V_th = H E at the GFL nodes
    Z = np.linalg.inv(reduce_network(B, n, s, b_bes))
    return Z, H, gfm


def build_model(system, s_gfm=None, params=None, b_bes=B_BES, B=None):
    Z, H, _ = thevenin(system, s_gfm, b_bes, B)
    return MultiInfeed(Z, system.s_gfl, H, params or GFLParams())


def modal_analysis(model):
    """Eigenvalues sorted by real part (descending), with damping ratios."""
    ev = model.eigenvalues()
    ev = ev[np.argsort(-ev.real)]
    damping = -ev.real / np.maximum(np.abs(ev), 1e-12)
    return ev, damping


def simulate(model, t_end=1.0, t_fault=0.2, duration=0.02, sag=0.2, n_points=2001,
             i_trip=2.0):
    """SG voltage sag of `sag` p.u. at t_fault, cleared after `duration`.

    Returns t, GFL active powers (own base, one row per unit) and the
    integrated states. If the system is unstable the oscillation grows until
    the model diverges, where the solver would take ever smaller steps; the run
    therefore stops (like overcurrent protection) once any GFL current exceeds
    i_trip p.u., and later samples are NaN.
    """
    x0 = model.equilibrium()
    E0 = model.default_E()
    E_fault = E0.copy()
    E_fault[0] = 1.0 - sag
    segments = [(0.0, t_fault, E0), (t_fault, t_fault + duration, E_fault),
                (t_fault + duration, t_end, E0)]
    t_eval = np.linspace(0.0, t_end, n_points)

    def trip(t, x, E):
        return i_trip - np.abs(model.unpack(x)[0]).max()
    trip.terminal = True

    X = np.full((len(x0), n_points), np.nan)
    sources = [E0] * n_points
    x = x0
    for k, (t0, t1, E) in enumerate(segments):
        last = k == len(segments) - 1
        mask = (t_eval >= t0) & ((t_eval <= t1) if last else (t_eval < t1))
        # t_eval only (no dense output) keeps memory bounded
        sol = solve_ivp(model.rhs, (t0, t1), x, args=(E,), method="LSODA",
                        t_eval=t_eval[mask], events=trip, rtol=1e-7, atol=1e-9,
                        max_step=1e-3)
        if sol.status == -1:
            raise RuntimeError(sol.message)
        idx = np.flatnonzero(mask)[:sol.y.shape[1]]
        X[:, idx] = sol.y
        for j in idx:
            sources[j] = E
        if sol.status == 1:  # tripped
            break
        x = sol.y[:, -1] if sol.t[-1] == t1 else \
            solve_ivp(model.rhs, (sol.t[-1], t1), sol.y[:, -1], args=(E,), method="LSODA",
                      rtol=1e-7, atol=1e-9, max_step=1e-3).y[:, -1]
    P = np.full((model.n, n_points), np.nan)
    for j in np.flatnonzero(~np.isnan(X[0])):
        P[:, j] = model.active_power(X[:, j], sources[j])
    return t_eval, P, X
