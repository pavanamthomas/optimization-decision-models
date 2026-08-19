"""Closed-form and gradient checks for unconstrained programmes."""

from __future__ import annotations

import numpy as np

from optmodels.checks import quadratic_stationarity, rosenbrock_gradient_residual
from optmodels.unconstrained import (
    double_well_global_minimizer,
    minimize_unconstrained,
    multi_start_minimize,
    quadratic_closed_form,
    quadratic_objective,
    minimize_quadratic_bfgs,
    rosenbrock_minimizer,
    minimize_rosenbrock,
    double_well,
    double_well_gradient,
)


def test_quadratic_closed_form_matches_stationarity() -> None:
    Q = np.array([[4.0, 1.0], [1.0, 3.0]])
    c = np.array([-2.0, 1.0])
    x = quadratic_closed_form(Q, c)
    assert quadratic_stationarity(x, Q, c)
    bfgs = minimize_quadratic_bfgs(Q, c)
    assert bfgs.success
    np.testing.assert_allclose(bfgs.x, x, atol=1e-6)


def test_quadratic_objective_at_minimizer_is_below_origin() -> None:
    Q = np.eye(2) * 2.0
    c = np.array([2.0, -4.0])
    x = quadratic_closed_form(Q, c)
    assert quadratic_objective(x, Q, c) < quadratic_objective(np.zeros(2), Q, c)


def test_rosenbrock_recovered_from_offset_start() -> None:
    star = rosenbrock_minimizer()
    res = minimize_rosenbrock([-1.2, 1.0])
    assert res.success
    np.testing.assert_allclose(res.x, star, atol=1e-5)
    assert rosenbrock_gradient_residual(res.x) < 1e-5


def test_multi_start_finds_better_well_than_poor_local_start() -> None:
    global_x = double_well_global_minimizer()
    poor = minimize_unconstrained(double_well, [1.2], jac=double_well_gradient)
    multi = multi_start_minimize(double_well, [[-1.5], [0.0], [1.5]], jac=double_well_gradient)
    assert multi.fun <= poor.fun + 1e-10
    assert abs(float(multi.x[0]) - global_x) < 1e-5
