"""Scenario 1: distributed GFM ESS planning with unrestricted sites (P1)."""

GAMMA = -1.0
EPS = 1e-5


def solve_support_function_step(system, s_k, target, cost, s_min, s_max, gamma=GAMMA):
    """Solve the convex subproblem (P-SF_k), Eq. 25, with CVXPY."""
    raise NotImplementedError


def algorithm1(system, target, cost, s_min, s_max, s_init, gamma=GAMMA, eps=EPS):
    """Iterate (P-SF_k) until ||S_{k+1} - S_k|| <= eps (Algorithm 1)."""
    raise NotImplementedError


def solve_sdp(system, target, cost, s_min, s_max):
    """Exact reference: P1 as one LMI via the Schur complement of the full B."""
    raise NotImplementedError


def proportional_benchmark(system, target):
    """Benchmark: GFM at GFL nodes proportional to GFL ratings (paper: 2.8656 p.u.)."""
    raise NotImplementedError
