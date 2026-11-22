"""Scenario 2: limited number of grid-scale GFM ESS stations (P2)."""


def ranked_combinations(sensitivity, n_max):
    """Site combinations in colexicographic order of sensitivity rank (Steps 1-4, Eq. 28)."""
    raise NotImplementedError


def two_layer(system, target, cost, s_min, s_max, n_max, iter_max):
    """Outer partial enumeration + inner Algorithm 1; stop per Eq. 29."""
    raise NotImplementedError


def brute_force(system, target, cost, s_min, s_max, n_max):
    """Exact reference: one SDP per site combination."""
    raise NotImplementedError


def centralized_benchmark(system, target, cost):
    """Benchmark: single best site (paper: node 25, 4.4125 p.u.)."""
    raise NotImplementedError
