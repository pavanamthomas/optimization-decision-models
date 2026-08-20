"""Decisions under discrete scenario uncertainty.

A single production quantity is chosen before demand is realised. Two
criteria are compared: expected cost under given scenario probabilities,
and maximin (minimax cost) which ignores those probabilities.

cost(x, d) = c x + h (x - d)_+ + pi (d - x)_+

Decision variables, parameters, and limitations: ``docs/model_catalogue.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class UncertaintyResult:
    x: float
    value: float
    criterion: str
    success: bool


def _as_vector(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def scenario_cost(
    x: float,
    demand: float,
    unit_cost: float,
    holding: float,
    penalty: float,
) -> float:
    """Linear production, holding, and shortage cost in one scenario."""
    if min(unit_cost, holding, penalty) < 0.0:
        raise ValueError("cost coefficients must be non-negative")
    over = max(x - demand, 0.0)
    under = max(demand - x, 0.0)
    return float(unit_cost * x + holding * over + penalty * under)


def expected_cost(
    x: float,
    demands: ArrayLike,
    probabilities: ArrayLike,
    unit_cost: float,
    holding: float,
    penalty: float,
) -> float:
    d = _as_vector(demands)
    p = _as_vector(probabilities)
    if d.size != p.size:
        raise ValueError("demands and probabilities must have the same length")
    if np.any(p < 0.0) or abs(float(p.sum()) - 1.0) > 1e-10:
        raise ValueError("probabilities must be non-negative and sum to 1")
    return float(
        sum(
            pk * scenario_cost(x, dk, unit_cost, holding, penalty)
            for pk, dk in zip(p, d)
        )
    )


def worst_case_cost(
    x: float,
    demands: ArrayLike,
    unit_cost: float,
    holding: float,
    penalty: float,
) -> float:
    d = _as_vector(demands)
    return float(
        max(scenario_cost(x, dk, unit_cost, holding, penalty) for dk in d)
    )


def _bounds(demands: np.ndarray) -> tuple[float, float]:
    lo = 0.0
    hi = float(np.max(demands)) * 1.5 + 1.0
    return lo, hi


def minimize_expected_cost(
    demands: ArrayLike,
    probabilities: ArrayLike,
    unit_cost: float,
    holding: float,
    penalty: float,
) -> UncertaintyResult:
    """Choose x to minimise probability-weighted scenario cost.

    For a continuous demand distribution the critical fractile is
    (pi - c) / (pi + h) when pi > c. With finitely many scenarios the
    same first-order condition is checked on the empirical cdf in tests
    by a neighbourhood comparison, not by claiming a unique closed form.
    """
    d = _as_vector(demands)
    p = _as_vector(probabilities)
    lo, hi = _bounds(d)

    def obj(x: float) -> float:
        return expected_cost(float(x), d, p, unit_cost, holding, penalty)

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
    return UncertaintyResult(
        x=float(res.x),
        value=float(res.fun),
        criterion="expected_cost",
        success=bool(res.success),
    )


def minimize_maximin_cost(
    demands: ArrayLike,
    unit_cost: float,
    holding: float,
    penalty: float,
) -> UncertaintyResult:
    """Choose x to minimise the worst-case scenario cost (maximin on -cost)."""
    d = _as_vector(demands)
    lo, hi = _bounds(d)

    def obj(x: float) -> float:
        return worst_case_cost(float(x), d, unit_cost, holding, penalty)

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
    return UncertaintyResult(
        x=float(res.x),
        value=float(res.fun),
        criterion="maximin",
        success=bool(res.success),
    )


def sensitivity_penalty(
    demands: ArrayLike,
    probabilities: ArrayLike,
    unit_cost: float,
    holding: float,
    penalties: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected-cost x*(pi) and value as the shortage penalty varies."""
    pi_grid = _as_vector(penalties)
    xs = np.zeros_like(pi_grid)
    vals = np.zeros_like(pi_grid)
    for i, pi in enumerate(pi_grid):
        res = minimize_expected_cost(demands, probabilities, unit_cost, holding, float(pi))
        xs[i] = res.x
        vals[i] = res.value
    return pi_grid, xs, vals


def example_demand_scenarios() -> tuple[np.ndarray, np.ndarray]:
    demands = np.array([10.0, 20.0, 30.0])
    probabilities = np.array([0.2, 0.5, 0.3])
    return demands, probabilities


@dataclass(frozen=True)
class SAAResult:
    x: float
    saa_value: float
    n_sample: int
    seed: int
    success: bool
    feasible: bool
    demands: np.ndarray
    note: str


def sample_average_newsvendor(
    n_sample: int,
    *,
    demand_mean: float = 20.0,
    demand_sd: float = 6.0,
    unit_cost: float = 1.0,
    holding: float = 0.4,
    penalty: float = 3.0,
    seed: int = 2026,
) -> SAAResult:
    """Sample-average approximation of the newsvendor expectation.

    Demand is drawn iid Normal(mean, sd), truncated at zero. The SAA
    programme uses equal weights 1/N on those draws. N is the sample size,
    not a hardcoded three-scenario list. Solver success is not a statement
    that the SAA objective has converged in N.
    """
    if n_sample < 2:
        raise ValueError("n_sample must be at least 2")
    rng = np.random.default_rng(int(seed))
    demands = np.maximum(rng.normal(demand_mean, demand_sd, size=n_sample), 0.0)
    probabilities = np.full(n_sample, 1.0 / n_sample)
    res = minimize_expected_cost(demands, probabilities, unit_cost, holding, penalty)
    lo, hi = _bounds(demands)
    feasible = bool(lo - 1e-9 <= res.x <= hi + 1e-9)
    return SAAResult(
        x=float(res.x),
        saa_value=float(res.value),
        n_sample=int(n_sample),
        seed=int(seed),
        success=bool(res.success),
        feasible=feasible,
        demands=demands,
        note=(
            "SAA uses 1/N weights on N draws. Feasibility is substitution into "
            "the sampled programme. Increasing N is a convergence diagnostic, "
            "not implied by success=True."
        ),
    )
