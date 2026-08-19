"""0-1 integer programmes: project selection and knapsack.

The working model is maximise v^T x subject to w^T x <= W, x in {0,1}^n,
with optional pairwise exclusion. PuLP/CBC is the solver; complete
enumeration is the validation method on small n.

Decision variables, parameters, and limitations: ``docs/model_catalogue.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class IPResult:
    x: np.ndarray
    objective: float
    resource_use: float
    success: bool
    status: str
    names: tuple[str, ...]


def _as_vector(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _solver():
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=30)


def example_projects() -> tuple[np.ndarray, np.ndarray, float, tuple[str, ...]]:
    """Six projects, one capacity constraint, one exclusive pair (A,B)."""
    names = ("A", "B", "C", "D", "E", "F")
    values = np.array([16.0, 15.0, 10.0, 8.0, 7.0, 4.0])
    costs = np.array([8.0, 7.0, 5.0, 4.0, 3.0, 2.0])
    budget = 14.0
    return values, costs, budget, names


def solve_knapsack(
    values: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    *,
    names: tuple[str, ...] | None = None,
    exclusive_pairs: tuple[tuple[int, int], ...] = (),
) -> IPResult:
    """0-1 knapsack / project selection.

    Decision variables: x in {0,1}^n.
    Objective: maximise v^T x.
    Constraints: w^T x <= W; optional x_i + x_j <= 1.
    Solution method: PuLP model, CBC.
    """
    v = _as_vector(values)
    w = _as_vector(weights)
    n = v.size
    if w.size != n:
        raise ValueError("values and weights must have the same length")
    if capacity < 0.0:
        raise ValueError("capacity must be non-negative")
    if names is None:
        names = tuple(f"x{i}" for i in range(n))
    if len(names) != n:
        raise ValueError("names length must match the number of items")

    prob = pulp.LpProblem("project_selection", pulp.LpMaximize)
    xs = [pulp.LpVariable(names[i], cat="Binary") for i in range(n)]
    prob += pulp.lpDot(v.tolist(), xs)
    prob += pulp.lpDot(w.tolist(), xs) <= float(capacity), "capacity"
    for a, b in exclusive_pairs:
        if not (0 <= a < n and 0 <= b < n and a != b):
            raise ValueError("exclusive pair indices out of range")
        prob += xs[a] + xs[b] <= 1, f"excl_{a}_{b}"

    status_code = prob.solve(_solver())
    status = pulp.LpStatus.get(status_code, str(status_code))
    x = np.array([float(pulp.value(xi) or 0.0) for xi in xs], dtype=float)
    obj = float(v @ x)
    use = float(w @ x)
    success = status == "Optimal" and np.all(np.isfinite(x))
    return IPResult(
        x=x,
        objective=obj,
        resource_use=use,
        success=success,
        status=status,
        names=names,
    )


def enumerate_knapsack(
    values: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    *,
    exclusive_pairs: tuple[tuple[int, int], ...] = (),
) -> IPResult:
    """Complete enumeration. Requires n <= 20."""
    v = _as_vector(values)
    w = _as_vector(weights)
    n = v.size
    if w.size != n:
        raise ValueError("values and weights must have the same length")
    if n > 20:
        raise ValueError("enumeration is limited to n <= 20")

    best_val = -np.inf
    best_x = np.zeros(n)
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        if float(w @ x) > capacity + 1e-12:
            continue
        illegal = False
        for a, b in exclusive_pairs:
            if x[a] + x[b] > 1.0 + 1e-12:
                illegal = True
                break
        if illegal:
            continue
        val = float(v @ x)
        if val > best_val:
            best_val = val
            best_x = x
    return IPResult(
        x=best_x,
        objective=float(best_val if np.isfinite(best_val) else 0.0),
        resource_use=float(w @ best_x),
        success=True,
        status="Enumerated",
        names=tuple(f"x{i}" for i in range(n)),
    )


def solve_project_selection(
    values: ArrayLike | None = None,
    costs: ArrayLike | None = None,
    budget: float | None = None,
    *,
    exclusive_pairs: tuple[tuple[int, int], ...] = ((0, 1),),
) -> IPResult:
    """Default six-project instance; A and B are mutually exclusive."""
    names: tuple[str, ...] | None
    if values is None or costs is None or budget is None:
        values, costs, budget, names = example_projects()
    else:
        names = None
    return solve_knapsack(
        values,
        costs,
        budget,
        names=names,
        exclusive_pairs=exclusive_pairs,
    )
