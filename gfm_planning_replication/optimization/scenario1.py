"""Scenario 1: distributed GFM ESS planning with unrestricted sites (P1).

    min  sum_i c_i S_GFMi
    s.t. lambda_min(A(S_GFM)) >= CgSCR + dg0           (15b)
         S_min <= S_GFMi <= S_max                      (15c)

Algorithm 1 replaces (15b) by the quadratic support function (19) built at
the current iterate S_k,

    q_k(S) = lambda_k + grad_k^T (S - S_k) + gamma/2 ||S - S_k||^2,

which under-estimates lambda_min when gamma <= inf lambda_min(Hessian) (24).
With gamma < 0, q_k(S) >= target is a convex (second-order cone) constraint,
so each (P-SF_k) is solved exactly by CVXPY, and every iterate stays feasible
for (P1).

solve_sdp gives an independent global optimum: lambda_min(A) >= t is
equivalent to the LMI  B + B_BES diag(S) - t blkdiag(S_GFL, 0) >= 0
(Schur complement on B4 + B_BES S_GFMm > 0), which is linear in S.
"""

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq

from network.build_B_matrix import B_BES, build_B
from stability.calculate_gscr import gscr_with_gfm
from stability.sensitivity import gscr_gradient

GAMMA = -1.0
EPS = 1e-5
TARGET_GSCR = 3.5  # CgSCR (2.5) + dg0 (1.0), Sec. V-A
S_MAX = 10.0  # p.u.; not given in the paper, chosen non-binding


@dataclass
class Result:
    s_gfm: np.ndarray
    cost: float
    gscr: float
    iterations: int = 0
    history: list = field(default_factory=list)  # ||S_{k+1} - S_k|| per iteration
    gamma: float = None  # final gamma used by Algorithm 1

    def locations(self, system, tol=1e-4):
        """Candidate nodes with a non-negligible GFM rating."""
        return [node for node, s in zip(system.candidate_nodes, self.s_gfm) if s > tol]


def _bounds(n_cand, s_min, s_max):
    return (np.broadcast_to(np.asarray(s_min, dtype=float), (n_cand,)),
            np.broadcast_to(np.asarray(s_max, dtype=float), (n_cand,)))


def _costs(cost, n_cand):
    return np.broadcast_to(np.asarray(cost, dtype=float), (n_cand,)).copy()


class SupportFunctionStep:
    """(P-SF_k), Eq. 25, as a parametrised CVXPY problem reused across iterations."""

    def __init__(self, cost, target, s_min, s_max, gamma=GAMMA):
        n_cand = len(cost)
        self.S = cp.Variable(n_cand)
        self.s_k = cp.Parameter(n_cand)
        self.grad = cp.Parameter(n_cand)
        self.offset = cp.Parameter()  # lambda_k - grad_k^T s_k (keeps the problem DPP)
        support = (self.offset + self.grad @ self.S
                   + gamma / 2 * cp.sum_squares(self.S - self.s_k))
        self.problem = cp.Problem(
            cp.Minimize(cost @ self.S),
            [support >= target, self.S >= s_min, self.S <= s_max],
        )

    def solve(self, s_k, lam_k, grad_k, solver=cp.CLARABEL):
        self.s_k.value, self.grad.value = s_k, grad_k
        self.offset.value = lam_k - grad_k @ s_k
        self.problem.solve(solver=solver)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"(P-SF_k) {self.problem.status}")
        return np.asarray(self.S.value)


def solve_support_function_step(system, s_k, target, cost, s_min, s_max,
                                gamma=GAMMA, b_bes=B_BES, B=None):
    """Solve one (P-SF_k) around s_k (Eq. 25). Returns S_{k+1}."""
    n_cand = len(system.candidate_nodes)
    lo, hi = _bounds(n_cand, s_min, s_max)
    lam_k, grad_k = gscr_gradient(system, s_k, b_bes, B)
    step = SupportFunctionStep(_costs(cost, n_cand), target, lo, hi, gamma)
    return step.solve(np.asarray(s_k, dtype=float), lam_k, grad_k)


def algorithm1(system, target=TARGET_GSCR, cost=1.0, s_min=0.0, s_max=S_MAX,
               s_init=None, gamma=GAMMA, eps=EPS, max_iter=5000,
               adapt_gamma=True, b_bes=B_BES, B=None):
    """Iterate (P-SF_k) until ||S_{k+1} - S_k|| <= eps (Algorithm 1).

    s_init must be feasible for (P1); by default every candidate gets s_max.

    If gamma violates (24), q_k over-estimates lambda_min and S_{k+1} can be
    infeasible (true gSCR < target); the iterates then cycle below the optimum.
    With adapt_gamma, such a step is rejected and gamma is doubled before
    retrying from S_k. With adapt_gamma=False the paper's fixed-gamma
    iteration is run as is.
    """
    if B is None:
        B = build_B(system)
    n_cand = len(system.candidate_nodes)
    c = _costs(cost, n_cand)
    lo, hi = _bounds(n_cand, s_min, s_max)

    s_k = hi.copy() if s_init is None else np.asarray(s_init, dtype=float)
    lam_k, grad_k = gscr_gradient(system, s_k, b_bes, B)
    if lam_k < target:
        raise ValueError(f"initial point infeasible: gSCR {lam_k:.4f} < {target}")

    step = SupportFunctionStep(c, target, lo, hi, gamma)
    history = []
    for k in range(1, max_iter + 1):
        s_next = step.solve(s_k, lam_k, grad_k)
        lam_next, grad_next = gscr_gradient(system, s_next, b_bes, B)
        if adapt_gamma and lam_next < target - 1e-6:
            gamma *= 2
            step = SupportFunctionStep(c, target, lo, hi, gamma)
            continue
        history.append(float(np.linalg.norm(s_next - s_k)))
        s_k, lam_k, grad_k = s_next, lam_next, grad_next
        if history[-1] <= eps:
            break
    else:
        raise RuntimeError(f"no convergence in {max_iter} iterations (gamma = {gamma})")

    return Result(s_k, float(c @ s_k), lam_k, k, history, gamma)


def solve_sdp(system, target=TARGET_GSCR, cost=1.0, s_min=0.0, s_max=S_MAX,
              b_bes=B_BES, B=None):
    """Exact reference: P1 as one LMI via the Schur complement of the full B.

    Solved with SCS at tight tolerances; Clarabel 0.11.1 stalls on this PSD cone.
    """
    if B is None:
        B = build_B(system)
    n_cand = len(system.candidate_nodes)
    c = _costs(cost, n_cand)
    lo, hi = _bounds(n_cand, s_min, s_max)
    n = len(system.gfl_nodes)

    shift = np.zeros(n_cand)
    shift[:n] = target * np.asarray(system.s_gfl, dtype=float)
    S = cp.Variable(n_cand)
    M = B - np.diag(shift) + b_bes * cp.diag(S)
    problem = cp.Problem(cp.Minimize(c @ S),
                         [0.5 * (M + M.T) >> 0, S >= lo, S <= hi])
    problem.solve(solver=cp.SCS, eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP {problem.status}")
    s = np.asarray(S.value)
    return Result(s, float(c @ s), gscr_with_gfm(system, s, b_bes, B)[0])


def proportional_benchmark(system, target=TARGET_GSCR, cost=1.0, b_bes=B_BES, B=None):
    """Benchmark: GFM at GFL nodes proportional to GFL ratings (paper: 2.8656 p.u.)."""
    if B is None:
        B = build_B(system)
    n_cand = len(system.candidate_nodes)
    n = len(system.gfl_nodes)
    c = _costs(cost, n_cand)

    def s_of(ratio):
        s = np.zeros(n_cand)
        s[:n] = ratio * np.asarray(system.s_gfl, dtype=float)
        return s

    ratio = brentq(lambda r: gscr_with_gfm(system, s_of(r), b_bes, B)[0] - target, 0.0, 100.0)
    s = s_of(ratio)
    return Result(s, float(c @ s), gscr_with_gfm(system, s, b_bes, B)[0])


if __name__ == "__main__":
    from network.load_system import load_ieee39

    system = load_ieee39()
    B = build_B(system)
    bench = proportional_benchmark(system, B=B)
    print(f"benchmark: cost {bench.cost:.4f} (paper 2.8656)")

    n = len(system.gfl_nodes)
    cases = {
        "c_i = 1": (1.0, "2.4516 @ 1-7, 9"),
        "c_i = 0.9 interior": (np.r_[np.ones(n), np.full(len(system.interior_nodes), 0.9)],
                               "2.4203 @ 1-6, 9-10, 15, 18-20, 28, 29, 31"),
    }
    for name, (cost, paper) in cases.items():
        r = algorithm1(system, cost=cost, B=B)
        sdp = solve_sdp(system, cost=cost, B=B)
        print(f"{name}: Alg.1 cost {r.cost:.4f}, gSCR {r.gscr:.4f}, {r.iterations} iters "
              f"(gamma {r.gamma}), "
              f"saving {100 * (1 - r.cost / bench.cost):.2f}% | SDP {sdp.cost:.4f} | paper {paper}")
        print(f"   locations: {r.locations(system)}")
