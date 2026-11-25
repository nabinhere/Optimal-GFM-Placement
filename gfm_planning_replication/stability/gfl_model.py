"""GFL IBR model, multi-infeed closed loop and critical gSCR (Eqs. 1, 10, 14).

The paper does not list the GFL control structure or parameters (it cites
[24], [25]); a standard grid-following converter is assumed and tuned so that
CgSCR ~= 2.5, matching Sec. V-A and Fig. 8. Per unit on each IBR's own rating:

    L filter x_f        x_f (dI/dt / w0 + j I) = V_c - V     (global xy frame)
    SRF-PLL             d(theta)/dt = kp_pll v_q + z,  dz/dt = ki_pll v_q
    measurement LPFs    p_m, v_m <- p, |v| with time constant tau_m
    P / AC-V loops      i_d* = PI(P* - p_m),  i_q* = -PI(V* - v_m)
    current control     v_c = PI(i* - i) + j x_f i            (PLL dq frame,
                                                             no voltage feed-forward)

so each unit has 10 states [Re I, Im I, x_cd, x_cq, theta, z, x_P, x_V, p_m, v_m].
Terminal-voltage control (not Q* = 0) is needed: at unity power factor a
1 p.u. GFL has a static power-flow limit at SCR = 2, above the paper's
baseline gSCR of 1.51.

Network. All branches are inductive with the same X/R = inf, so the network
operator is B (x) F(s), i.e. in complex phasor form I = B h(s) V with
h(s) = 1 / (s/w0 + j). Eliminating interior nodes and ideal sources (SG, GFM
internal EMFs) leaves, at the GFL nodes (system base),

    V = Z (dI_sys/dt / w0 + j I_sys) + H E,    Z = B_red^-1,

where I_sys = S_GFL * I and H E is the Thevenin voltage driven by the source
EMFs E (rows of H sum to 1). With w = dI/dt / w0 + j I (own base), the
filter and network equations give

    (diag(x_f) + Z diag(S)) w = rot v_c - H E,    V = Z (S w) + H E,

so dI/dt and V are explicit functions of the states. (An ideal, unfiltered
voltage feed-forward would decouple the converter from the grid and push
CgSCR down to ~1.2, so it is left out.)

A single IBR on an SCR-g grid is the n = 1 case with Z = 1/g, H = 1; CgSCR
(Eq. 14) is the smallest g for which its linearisation is stable.

Unlike the paper's derivation of (12), the operating point here is the true
nonlinear power-flow solution, so voltage angles differ between nodes and the
modal decoupling is only approximate; validation.dynamic_simulation measures
how well gSCR > CgSCR still predicts stability.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

N_STATES = 10


@dataclass
class GFLParams:
    f0: float = 50.0       # Hz
    x_f: float = 0.10      # filter reactance (p.u., own base)
    kp_pll: float = 152.0  # rad/s per p.u. voltage; tuned so CgSCR = 2.50
    ki_pll: float = 11552.0  # kp^2 / 2 (damping ratio 0.71 on a stiff grid)
    tau_m: float = 0.01    # s, power / voltage measurement filter
    kp_p: float = 0.25
    ki_p: float = 25.0
    kp_v: float = 0.25
    ki_v: float = 25.0
    kp_c: float = 0.5      # current loop, p.u. voltage per p.u. current
    ki_c: float = 50.0
    p_ref: float = 1.0
    v_ref: float = 1.0

    @property
    def w0(self):
        return 2 * np.pi * self.f0


class MultiInfeed:
    """n homogeneous GFL IBRs behind V = Z (dI_sys/dt / w0 + j I_sys) + H E."""

    def __init__(self, Z, s_gfl, H, params=None):
        self.p = params or GFLParams()
        self.Z = np.atleast_2d(np.asarray(Z, dtype=float))
        self.s = np.atleast_1d(np.asarray(s_gfl, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.n = len(self.s)
        self.M = np.diag(np.full(self.n, self.p.x_f)) + self.Z * self.s[None, :]

    def default_E(self):
        return np.ones(self.H.shape[1], dtype=complex)

    # ---- state packing ----------------------------------------------------
    def unpack(self, x):
        x = np.reshape(x, (self.n, N_STATES))
        return (x[:, 0] + 1j * x[:, 1], x[:, 2] + 1j * x[:, 3],
                x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9])

    @staticmethod
    def pack(I, xc, theta, z, xp, xv, pm, vm):
        return np.column_stack([I.real, I.imag, xc.real, xc.imag,
                                theta, z, xp, xv, pm, vm]).ravel()

    # ---- dynamics -----------------------------------------------------------
    def outputs(self, x, E):
        """Terminal voltage V, current reference i*, converter voltage u (PLL
        frame) and w = dI/dt / w0 + j I."""
        p = self.p
        I, xc, theta, z, xp, xv, pm, vm = self.unpack(x)
        rot = np.exp(1j * theta)
        i_loc = I / rot
        istar = (p.kp_p * (p.p_ref - pm) + xp) - 1j * (p.kp_v * (p.v_ref - vm) + xv)
        u = p.kp_c * (istar - i_loc) + xc + 1j * p.x_f * i_loc
        Vth = self.H @ E
        w = np.linalg.solve(self.M, rot * u - Vth)
        V = self.Z @ (self.s * w) + Vth
        return V, istar, u, w

    def rhs(self, t, x, E):
        """dx/dt for source EMFs E (complex vector, one per column of H)."""
        p = self.p
        I, xc, theta, z, xp, xv, pm, vm = self.unpack(x)
        V, istar, u, w = self.outputs(x, E)
        rot = np.exp(1j * theta)
        i_loc = I / rot
        v_loc = V / rot
        power = (v_loc * np.conj(i_loc)).real
        return self.pack(
            p.w0 * (w - 1j * I),                       # filter current
            p.ki_c * (istar - i_loc),                  # current PI
            p.kp_pll * v_loc.imag + z,                 # PLL angle
            p.ki_pll * v_loc.imag,                     # PLL integrator
            p.ki_p * (p.p_ref - pm),                   # P loop
            p.ki_v * (p.v_ref - vm),                   # AC-voltage loop
            (power - pm) / p.tau_m,                    # filters
            (np.abs(V) - vm) / p.tau_m,
        )

    def active_power(self, x, E):
        I = self.unpack(x)[0]
        V = self.outputs(x, E)[0]
        return (V * np.conj(I)).real

    # ---- operating point & linearisation -----------------------------------
    def power_flow(self, E=None):
        """GFL terminal voltages with every unit at P* and |V| = V* (PV nodes).

        Static network: I_sys = -j Z^-1 (V - H E). Raises RuntimeError if no
        solution exists (grid too weak for P*).
        """
        E = self.default_E() if E is None else E
        Vth = self.H @ E
        Y = np.linalg.inv(self.Z)
        p = self.p

        def voltages(delta):
            return p.v_ref * np.exp(1j * delta)

        def mismatch(delta):
            V = voltages(delta)
            I_sys = -1j * Y @ (V - Vth)
            return (V * np.conj(I_sys)).real - self.s * p.p_ref

        delta0 = np.angle(Vth) + np.arcsin(np.clip(np.diag(self.Z) * self.s * p.p_ref, -1, 1))
        delta, _, ok, msg = fsolve(mismatch, delta0, full_output=True, xtol=1e-13)
        if ok != 1 or np.max(np.abs(mismatch(delta))) > 1e-9:
            raise RuntimeError(f"no power-flow solution: {msg}")
        V = voltages(delta)
        return V, -1j * Y @ (V - Vth) / self.s

    def equilibrium(self, E=None):
        """Steady state with source EMFs E (default 1 p.u.).

        PLL aligned with V (v_q = 0, z = 0), i = i*, so the current
        integrators hold v (u = v + j x_f i) and the outer integrators hold
        i_d and -i_q.
        """
        E = self.default_E() if E is None else E
        V, I = self.power_flow(E)
        theta = np.angle(V)
        i_loc = I * np.exp(-1j * theta)
        x = self.pack(I, np.abs(V).astype(complex), theta, np.zeros(self.n),
                      i_loc.real, -i_loc.imag, np.full(self.n, self.p.p_ref),
                      np.full(self.n, self.p.v_ref))
        residual = np.max(np.abs(self.rhs(0, x, E)))
        assert residual < 1e-7, residual
        return x

    def jacobian(self, x0, E=None, h=1e-7):
        """Central-difference Jacobian of rhs at x0."""
        E = self.default_E() if E is None else E
        J = np.empty((len(x0), len(x0)))
        for k in range(len(x0)):
            dx = np.zeros(len(x0))
            dx[k] = h
            J[:, k] = (self.rhs(0, x0 + dx, E) - self.rhs(0, x0 - dx, E)) / (2 * h)
        return J

    def eigenvalues(self, E=None):
        return np.linalg.eigvals(self.jacobian(self.equilibrium(E), E))


def single_ibr(scr, params=None):
    """One IBR on a grid of short-circuit ratio `scr` (Z = 1/scr, own base)."""
    return MultiInfeed([[1.0 / scr]], [1.0], [[1.0]], params)


def max_real_eig(scr, params=None):
    """Spectral abscissa of the linearised single-IBR system."""
    return np.max(single_ibr(scr, params).eigenvalues().real)


def critical_gscr(params=None, lo=1.05, hi=10.0):
    """CgSCR (Eq. 14): SCR at which the single-IBR system is critically stable.

    The system is assumed unstable at `lo` and stable at `hi`.
    """
    return brentq(lambda g: max_real_eig(g, params), lo, hi, xtol=1e-8)


def gfl_admittance(s, params=None, scr=10.0, eps=1e-6):
    """Per-unit 2x2 GFL admittance Y_GFL(s) in the xy frame (Eq. 1).

    Linearised at the operating point on an SCR-`scr` grid: the terminal
    voltage is made an input (Z -> 0, H -> 1 with E = U), and Y maps
    [dU_x, dU_y] to -[dI_x, dI_y] as in dI = -S Y dU.
    """
    model = single_ibr(scr, params)
    x0 = model.equilibrium()
    V0 = model.outputs(x0, model.default_E())[0][0]

    # Source-driven model: the terminal voltage is imposed directly.
    stiff = MultiInfeed([[0.0]], [1.0], [[1.0]], model.p)
    E0 = np.array([V0])
    n = len(x0)
    A = stiff.jacobian(x0, E0)
    f0 = stiff.rhs(0, x0, E0)
    B = np.column_stack([(stiff.rhs(0, x0, E0 + d) - f0) / eps
                         for d in (np.array([eps]), np.array([1j * eps]))])
    C = np.zeros((2, n))
    C[0, 0] = C[1, 1] = 1.0  # outputs: Re I, Im I
    s = np.atleast_1d(s)
    return np.array([-(C @ np.linalg.solve(sk * np.eye(n) - A, B)) for sk in s])
