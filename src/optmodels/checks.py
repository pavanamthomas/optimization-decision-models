"""Independent substitution of a candidate point into f and g.

These routines do not call a solver. They recompute the objective, slacks,
gradient residuals, Marshallian identities, and integrality at a supplied
point so that a status string is never treated as evidence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from optmodels.constrained import expenditure_shares
from optmodels.unconstrained import quadratic_gradient, quadratic_objective, rosenbrock_gradient


def central_difference_gradient(fun, x: ArrayLike, h: float = 1e-6) -> np.ndarray:
    """Component-wise central difference of a scalar function."""
    x0 = np.asarray(x, dtype=float).reshape(-1)
    g = np.zeros_like(x0)
    for i in range(x0.size):
        e = np.zeros_like(x0)
        e[i] = h
        g[i] = (fun(x0 + e) - fun(x0 - e)) / (2.0 * h)
    return g


def quadratic_stationarity(x: ArrayLike, Q: ArrayLike, c: ArrayLike, tol: float = 1e-8) -> bool:
    grad = quadratic_gradient(x, Q, c)
    return bool(np.linalg.norm(grad, ord=np.inf) <= tol)


def quadratic_objective_at(x: ArrayLike, Q: ArrayLike, c: ArrayLike) -> float:
    return quadratic_objective(x, Q, c)


def rosenbrock_gradient_residual(x: ArrayLike, h: float = 1e-6) -> float:
    from optmodels.unconstrained import rosenbrock

    analytic = rosenbrock_gradient(x)
    numeric = central_difference_gradient(rosenbrock, x, h=h)
    return float(np.linalg.norm(analytic - numeric, ord=np.inf))


def lp_primal_feasible(
    x: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    *,
    tol: float = 1e-8,
) -> bool:
    xv = np.asarray(x, dtype=float).reshape(-1)
    A_ub = np.asarray(A, dtype=float)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if np.any(xv < -tol):
        return False
    return bool(np.all(A_ub @ xv <= bv + tol))


def lp_strong_duality(primal_obj: float, dual_obj: float, *, tol: float = 1e-6) -> bool:
    return bool(abs(primal_obj - dual_obj) <= tol * max(1.0, abs(primal_obj)))


def knapsack_feasible(
    x: ArrayLike,
    weights: ArrayLike,
    capacity: float,
    *,
    exclusive_pairs: tuple[tuple[int, int], ...] = (),
    tol: float = 1e-8,
) -> bool:
    xv = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if np.any((xv < -tol) | (xv > 1.0 + tol)):
        return False
    if np.any(np.abs(xv - np.round(xv)) > tol):
        return False
    if float(w @ xv) > capacity + tol:
        return False
    for a, b in exclusive_pairs:
        if xv[a] + xv[b] > 1.0 + tol:
            return False
    return True


def transportation_balanced(
    flows: ArrayLike,
    supply: ArrayLike,
    demand: ArrayLike,
    *,
    tol: float = 1e-6,
) -> bool:
    x = np.asarray(flows, dtype=float)
    s = np.asarray(supply, dtype=float).reshape(-1)
    d = np.asarray(demand, dtype=float).reshape(-1)
    if np.any(x < -tol):
        return False
    return bool(np.allclose(x.sum(axis=1), s, atol=tol) and np.allclose(x.sum(axis=0), d, atol=tol))


def cobb_douglas_budget_exhausted(
    x: ArrayLike,
    p1: float,
    p2: float,
    m: float,
    *,
    tol: float = 1e-6,
) -> bool:
    xv = np.asarray(x, dtype=float).reshape(-1)
    return bool(abs(p1 * xv[0] + p2 * xv[1] - m) <= tol * max(1.0, m))


def cobb_douglas_shares_match_alpha(
    x: ArrayLike,
    alpha: float,
    p1: float,
    p2: float,
    m: float,
    *,
    tol: float = 1e-5,
) -> bool:
    shares = expenditure_shares(x, [p1, p2], m)
    return bool(abs(shares[0] - alpha) <= tol and abs(shares[1] - (1.0 - alpha)) <= tol)


def cake_accounting(policy: np.ndarray, initial_wealth: int) -> bool:
    """Remainder is never negative along the stored integer policy."""
    w = initial_wealth
    t_max, w_max = policy.shape
    if not (0 <= initial_wealth < w_max):
        return False
    for t in range(t_max):
        c = int(policy[t, w])
        if c < 0 or c > w:
            return False
        w = w - c
    return w == 0
