"""Unconstrained nonlinear programmes.

Convex quadratic programmes, a smooth test function with a known global
minimizer (Rosenbrock), and a one-dimensional double well used to contrast
local search with multi-start.

Decision variables, parameters, and limitations for each model are recorded
in ``docs/model_catalogue.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


@dataclass(frozen=True)
class UnconstrainedResult:
    x: np.ndarray
    fun: float
    success: bool
    nfev: int
    message: str
    starts_fun: tuple[float, ...] = ()
    starts_x: tuple[tuple[float, ...], ...] = ()


def _as_vector(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _symmetrize(Q: ArrayLike) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    return 0.5 * (Q + Q.T)


def quadratic_objective(x: ArrayLike, Q: ArrayLike, c: ArrayLike) -> float:
    """f(x) = (1/2) x^T Q x + c^T x with Q taken symmetric."""
    xv = _as_vector(x)
    Qs = _symmetrize(Q)
    cv = _as_vector(c)
    if Qs.shape[0] != xv.size or cv.size != xv.size:
        raise ValueError("incompatible shapes for Q, c, x")
    return float(0.5 * xv @ Qs @ xv + cv @ xv)


def quadratic_gradient(x: ArrayLike, Q: ArrayLike, c: ArrayLike) -> np.ndarray:
    xv = _as_vector(x)
    Qs = _symmetrize(Q)
    cv = _as_vector(c)
    return Qs @ xv + cv


def quadratic_closed_form(Q: ArrayLike, c: ArrayLike) -> np.ndarray:
    """Unique minimizer x* = -Q^{-1} c when Q is positive definite.

    Decision variables: x in R^n.
    Objective: minimise (1/2) x^T Q x + c^T x.
    Assumptions: Q symmetric positive definite.
    Solution method: linear solve, not an iterative local search.
    """
    Qs = _symmetrize(Q)
    cv = _as_vector(c)
    if Qs.shape[0] != cv.size:
        raise ValueError("Q and c have incompatible shapes")
    evals = np.linalg.eigvalsh(Qs)
    if float(np.min(evals)) <= 1e-12:
        raise ValueError("closed form requires a positive-definite Hessian")
    return -np.linalg.solve(Qs, cv)


def minimize_quadratic_bfgs(
    Q: ArrayLike,
    c: ArrayLike,
    x0: ArrayLike | None = None,
) -> UnconstrainedResult:
    """BFGS on the same quadratic; used to compare with the closed form."""
    Qs = _symmetrize(Q)
    cv = _as_vector(c)
    n = cv.size
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = _as_vector(x0)

    def fun(x: np.ndarray) -> float:
        return quadratic_objective(x, Qs, cv)

    def jac(x: np.ndarray) -> np.ndarray:
        return quadratic_gradient(x, Qs, cv)

    res = minimize(fun, x0, jac=jac, method="BFGS")
    return UnconstrainedResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        nfev=int(res.nfev),
        message=str(res.message),
    )


def rosenbrock(x: ArrayLike, a: float = 1.0, b: float = 100.0) -> float:
    """Classic Rosenbrock banana: unique global min 0 at (a, a^2)."""
    xv = _as_vector(x)
    if xv.size != 2:
        raise ValueError("rosenbrock is defined on R^2")
    return float((a - xv[0]) ** 2 + b * (xv[1] - xv[0] ** 2) ** 2)


def rosenbrock_gradient(x: ArrayLike, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    xv = _as_vector(x)
    if xv.size != 2:
        raise ValueError("rosenbrock is defined on R^2")
    x1, x2 = float(xv[0]), float(xv[1])
    dfdx = -2.0 * (a - x1) - 4.0 * b * x1 * (x2 - x1**2)
    dfdy = 2.0 * b * (x2 - x1**2)
    return np.array([dfdx, dfdy], dtype=float)


def rosenbrock_minimizer(a: float = 1.0) -> np.ndarray:
    return np.array([a, a * a], dtype=float)


def minimize_rosenbrock(x0: ArrayLike) -> UnconstrainedResult:
    x0 = _as_vector(x0)
    res = minimize(rosenbrock, x0, jac=rosenbrock_gradient, method="BFGS")
    return UnconstrainedResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        nfev=int(res.nfev),
        message=str(res.message),
    )


def double_well(x: ArrayLike) -> float:
    """f(x) = x^4 - 2 x^2 + 0.5 x.

    Two local minima; the linear term breaks symmetry so one is strictly
    global. Critical points of the cubic 4x^3 - 4x + 0.5 can be listed
    exactly via ``numpy.roots``.
    """
    z = float(_as_vector(x)[0])
    return float(z**4 - 2.0 * z**2 + 0.5 * z)


def double_well_gradient(x: ArrayLike) -> np.ndarray:
    z = float(_as_vector(x)[0])
    return np.array([4.0 * z**3 - 4.0 * z + 0.5], dtype=float)


def double_well_critical_points() -> np.ndarray:
    """Real roots of f'(x) = 4x^3 - 4x + 0.5."""
    roots = np.roots(np.array([4.0, 0.0, -4.0, 0.5], dtype=float))
    real = np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)])
    return np.sort(real.astype(float))


def double_well_global_minimizer() -> float:
    pts = double_well_critical_points()
    values = np.array([double_well(p) for p in pts])
    return float(pts[int(np.argmin(values))])


def minimize_unconstrained(
    fun,
    x0: ArrayLike,
    jac=None,
    method: str = "BFGS",
) -> UnconstrainedResult:
    x0 = _as_vector(x0)
    res = minimize(fun, x0, jac=jac, method=method)
    return UnconstrainedResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        nfev=int(res.nfev),
        message=str(res.message),
    )


def multi_start_minimize(
    fun,
    starts: ArrayLike,
    jac=None,
    method: str = "BFGS",
) -> UnconstrainedResult:
    """Run local search from each row of ``starts``; keep the least f.

    This is a heuristic comparison of basins, not a global certificate,
    unless every relevant critical point is known independently.
    """
    starts_arr = np.asarray(starts, dtype=float)
    if starts_arr.ndim == 1:
        starts_arr = starts_arr.reshape(-1, 1)
    best: UnconstrainedResult | None = None
    recorded_fun: list[float] = []
    recorded_x: list[tuple[float, ...]] = []
    for row in starts_arr:
        res = minimize_unconstrained(fun, row, jac=jac, method=method)
        recorded_fun.append(res.fun)
        recorded_x.append(tuple(float(v) for v in res.x))
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return UnconstrainedResult(
        x=best.x,
        fun=best.fun,
        success=best.success,
        nfev=best.nfev,
        message=best.message,
        starts_fun=tuple(recorded_fun),
        starts_x=tuple(recorded_x),
    )
