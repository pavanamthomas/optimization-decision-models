"""Linear programmes: resource allocation in standard form.

Primal: maximise c^T x subject to A x <= b, x >= 0.
The dual is solved as a separate LP so that strong duality can be checked
without relying on a solver's marginal-value field.

Decision variables, parameters, and limitations: ``docs/model_catalogue.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog


@dataclass(frozen=True)
class LPResult:
    x: np.ndarray
    objective: float
    slacks: np.ndarray
    dual: np.ndarray | None
    dual_objective: float | None
    success: bool
    message: str
    solver_marginals: np.ndarray | None = None


def _as_vector(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _as_matrix(A: ArrayLike) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-d array")
    return A


def textbook_resource_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-activity, three-resource illustration.

    maximise  3 x1 + 5 x2
    subject to  x1        <= 4
               2 x2       <= 12
               3 x1 + 2 x2 <= 18
               x >= 0

    The unique vertex optimum is x = (2, 6) with value 36. The first
    resource is slack; the other two bind.
    """
    c = np.array([3.0, 5.0])
    A = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 2.0],
        ]
    )
    b = np.array([4.0, 12.0, 18.0])
    return c, A, b


def solve_lp_max(
    c: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    *,
    method: str = "highs",
) -> LPResult:
    """Maximise c^T x s.t. A x <= b, x >= 0.

    Decision variables: x >= 0.
    Objective: maximise c^T x.
    Constraints: A x <= b.
    Solution method: scipy.optimize.linprog on -c, plus an independent dual.
    """
    cv = _as_vector(c)
    A_ub = _as_matrix(A)
    bv = _as_vector(b)
    if A_ub.shape != (bv.size, cv.size):
        raise ValueError("incompatible shapes for c, A, b")

    res = linprog(-cv, A_ub=A_ub, b_ub=bv, bounds=(0.0, None), method=method)
    if res.x is None:
        return LPResult(
            x=np.full(cv.size, np.nan),
            objective=float("nan"),
            slacks=np.full(bv.size, np.nan),
            dual=None,
            dual_objective=None,
            success=False,
            message=str(res.message),
        )

    x = np.asarray(res.x, dtype=float)
    objective = float(cv @ x)
    slacks = bv - A_ub @ x

    dual_res = solve_lp_dual(cv, A_ub, bv, method=method)
    marginals = None
    ineq = getattr(res, "ineqlin", None)
    if ineq is not None and getattr(ineq, "marginals", None) is not None:
        # linprog minimises -c^T x, so d(min obj)/db = - d(primal max)/db.
        marginals = -np.asarray(ineq.marginals, dtype=float)

    return LPResult(
        x=x,
        objective=objective,
        slacks=np.asarray(slacks, dtype=float),
        dual=dual_res.x if dual_res.success else None,
        dual_objective=dual_res.objective if dual_res.success else None,
        success=bool(res.success) and dual_res.success,
        message=str(res.message),
        solver_marginals=marginals,
    )


def solve_lp_dual(
    c: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    *,
    method: str = "highs",
) -> LPResult:
    """Dual of maximise c^T x, A x <= b, x >= 0: minimise b^T y, A^T y >= c, y >= 0."""
    cv = _as_vector(c)
    A_ub = _as_matrix(A)
    bv = _as_vector(b)
    # min b^T y s.t. -A^T y <= -c, y >= 0
    res = linprog(
        bv,
        A_ub=-A_ub.T,
        b_ub=-cv,
        bounds=(0.0, None),
        method=method,
    )
    if res.x is None:
        return LPResult(
            x=np.full(bv.size, np.nan),
            objective=float("nan"),
            slacks=np.full(cv.size, np.nan),
            dual=None,
            dual_objective=None,
            success=False,
            message=str(res.message),
        )
    y = np.asarray(res.x, dtype=float)
    return LPResult(
        x=y,
        objective=float(bv @ y),
        slacks=A_ub.T @ y - cv,
        dual=None,
        dual_objective=None,
        success=bool(res.success),
        message=str(res.message),
    )


def solve_resource_allocation(
    c: ArrayLike | None = None,
    A: ArrayLike | None = None,
    b: ArrayLike | None = None,
) -> LPResult:
    """Resource-allocation LP; defaults to the two-activity textbook instance."""
    if c is None or A is None or b is None:
        c, A, b = textbook_resource_data()
    return solve_lp_max(c, A, b)
