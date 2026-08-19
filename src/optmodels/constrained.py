"""Equality- and inequality-constrained nonlinear programmes.

The working example is two-good Cobb-Douglas utility maximisation on a
linear budget. Marshallian demands are available in closed form, which
makes the SLSQP path a numerical check rather than a discovery procedure.

Decision variables, parameters, and limitations: ``docs/model_catalogue.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


@dataclass(frozen=True)
class ConstrainedResult:
    x: np.ndarray
    utility: float
    multiplier: float | None
    success: bool
    message: str
    budget_residual: float
    shares: np.ndarray
    binding_floor: bool = False


def _as_vector(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def cobb_douglas_utility(x: ArrayLike, alpha: float) -> float:
    """u(x1, x2) = x1^alpha * x2^(1-alpha), x > 0, alpha in (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    xv = _as_vector(x)
    if xv.size != 2:
        raise ValueError("two-good Cobb-Douglas requires x in R^2")
    if np.any(xv <= 0.0):
        return 0.0
    return float(xv[0] ** alpha * xv[1] ** (1.0 - alpha))


def cobb_douglas_log_utility(x: ArrayLike, alpha: float) -> float:
    """log u = alpha log x1 + (1-alpha) log x2, defined for x > 0."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    xv = _as_vector(x)
    if xv.size != 2 or np.any(xv <= 0.0):
        return -np.inf
    return float(alpha * np.log(xv[0]) + (1.0 - alpha) * np.log(xv[1]))


def cobb_douglas_log_gradient(x: ArrayLike, alpha: float) -> np.ndarray:
    xv = _as_vector(x)
    return np.array([alpha / xv[0], (1.0 - alpha) / xv[1]], dtype=float)


def marshallian_cobb_douglas(
    alpha: float,
    p1: float,
    p2: float,
    m: float,
) -> np.ndarray:
    """Interior Marshallian demand: x1 = alpha m / p1, x2 = (1-alpha) m / p2.

    Decision variables: x = (x1, x2) > 0.
    Parameters: alpha in (0,1), prices p > 0, income m > 0.
    Objective: maximise x1^alpha x2^(1-alpha).
    Constraint: p1 x1 + p2 x2 = m.
    Assumptions: local nonsatiation, interior solution (no binding floors).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if min(p1, p2, m) <= 0.0:
        raise ValueError("prices and income must be positive")
    return np.array([alpha * m / p1, (1.0 - alpha) * m / p2], dtype=float)


def lagrangian_multiplier_cobb_douglas(
    alpha: float,
    p1: float,
    p2: float,
    m: float,
) -> float:
    """Interior multiplier for L = u + lambda (m - p·x).

    At the Marshallian point, lambda* = u*/m = MU_i / p_i.
    """
    x = marshallian_cobb_douglas(alpha, p1, p2, m)
    u = cobb_douglas_utility(x, alpha)
    return float(u / m)


def expenditure_shares(x: ArrayLike, prices: ArrayLike, m: float) -> np.ndarray:
    xv = _as_vector(x)
    p = _as_vector(prices)
    if m <= 0.0:
        raise ValueError("income must be positive")
    return (p * xv) / m


def maximize_cobb_douglas(
    alpha: float,
    p1: float,
    p2: float,
    m: float,
    *,
    budget_type: str = "eq",
    x1_min: float = 0.0,
    x0: ArrayLike | None = None,
) -> ConstrainedResult:
    """Numerical maximisation of Cobb-Douglas utility on a linear budget.

    ``budget_type`` is ``eq`` (p·x = m) or ``ineq`` (p·x <= m). Local
    nonsatiation makes the inequality bind at an interior optimum.
    ``x1_min`` is a consumption floor; if it exceeds Marshallian x1 the
    solution is on the boundary x1 = x1_min, x2 = (m - p1 x1_min)/p2.
    """
    if budget_type not in {"eq", "ineq"}:
        raise ValueError("budget_type must be 'eq' or 'ineq'")
    if min(p1, p2, m) <= 0.0:
        raise ValueError("prices and income must be positive")
    if x1_min < 0.0:
        raise ValueError("x1_min must be non-negative")
    if p1 * x1_min >= m:
        raise ValueError("floor x1_min exhausts or exceeds income")

    lo1 = max(1e-10, x1_min)
    bounds = [(lo1, None), (1e-10, None)]
    if x0 is None:
        remaining = m - p1 * lo1
        x0_vec = np.array([lo1 + 0.25 * (m / p1 - lo1), 0.5 * remaining / p2], dtype=float)
        x0_vec = np.maximum(x0_vec, [lo1, 1e-10])
    else:
        x0_vec = _as_vector(x0)

    p = np.array([p1, p2], dtype=float)

    def nlogu(x: np.ndarray) -> float:
        val = cobb_douglas_log_utility(x, alpha)
        return -val if np.isfinite(val) else 1e30

    def nlogu_jac(x: np.ndarray) -> np.ndarray:
        return -cobb_douglas_log_gradient(x, alpha)

    if budget_type == "eq":
        constraints = {"type": "eq", "fun": lambda x: m - float(p @ x)}
    else:
        constraints = {"type": "ineq", "fun": lambda x: m - float(p @ x)}

    res = minimize(
        nlogu,
        x0_vec,
        jac=nlogu_jac,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 400, "disp": False},
    )
    x = np.asarray(res.x, dtype=float)
    u = cobb_douglas_utility(x, alpha)
    residual = float(m - p @ x)
    shares = expenditure_shares(x, p, m)
    floor_gap = float(x[0] - x1_min) if x1_min > 0.0 else np.inf
    binding = bool(x1_min > 0.0 and floor_gap <= 1e-6)

    multiplier = None
    if residual <= 1e-7 * max(1.0, m) and x1_min == 0.0:
        multiplier = float(u / m)

    return ConstrainedResult(
        x=x,
        utility=u,
        multiplier=multiplier,
        success=bool(res.success),
        message=str(res.message),
        budget_residual=residual,
        shares=shares,
        binding_floor=binding,
    )


def boundary_cobb_douglas(
    alpha: float,
    p1: float,
    p2: float,
    m: float,
    x1_min: float,
) -> np.ndarray:
    """Exact demand when x1 >= x1_min is imposed on the budget line.

    If the unconstrained Marshallian point satisfies the floor, return it.
    Otherwise the floor binds: x1 = x1_min, x2 = (m - p1 x1_min) / p2.
    """
    interior = marshallian_cobb_douglas(alpha, p1, p2, m)
    if interior[0] >= x1_min - 1e-15:
        return interior
    x2 = (m - p1 * x1_min) / p2
    if x2 <= 0.0:
        raise ValueError("binding floor leaves non-positive residual income")
    return np.array([x1_min, x2], dtype=float)
