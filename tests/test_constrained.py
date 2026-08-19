"""Cobb-Douglas identities and boundary solutions."""

from __future__ import annotations

import numpy as np

from optmodels.checks import cobb_douglas_budget_exhausted, cobb_douglas_shares_match_alpha
from optmodels.constrained import (
    boundary_cobb_douglas,
    marshallian_cobb_douglas,
    maximize_cobb_douglas,
)


def test_marshallian_shares_and_budget() -> None:
    alpha, p1, p2, m = 0.35, 2.0, 3.0, 12.0
    x = marshallian_cobb_douglas(alpha, p1, p2, m)
    assert cobb_douglas_budget_exhausted(x, p1, p2, m)
    assert cobb_douglas_shares_match_alpha(x, alpha, p1, p2, m)
    np.testing.assert_allclose(x, [alpha * m / p1, (1.0 - alpha) * m / p2])


def test_slsqp_matches_marshallian_on_equality_budget() -> None:
    alpha, p1, p2, m = 0.4, 1.5, 2.5, 10.0
    closed = marshallian_cobb_douglas(alpha, p1, p2, m)
    num = maximize_cobb_douglas(alpha, p1, p2, m, budget_type="eq")
    assert num.success
    np.testing.assert_allclose(num.x, closed, atol=1e-5)
    assert cobb_douglas_budget_exhausted(num.x, p1, p2, m)


def test_inequality_budget_binds_under_local_nonsatiation() -> None:
    num = maximize_cobb_douglas(0.5, 1.0, 1.0, 8.0, budget_type="ineq")
    assert abs(num.budget_residual) < 1e-5


def test_binding_floor_is_a_boundary_solution() -> None:
    alpha, p1, p2, m = 0.2, 1.0, 1.0, 10.0
    x1_min = 5.0
    interior = marshallian_cobb_douglas(alpha, p1, p2, m)
    assert interior[0] < x1_min
    exact = boundary_cobb_douglas(alpha, p1, p2, m, x1_min)
    num = maximize_cobb_douglas(alpha, p1, p2, m, x1_min=x1_min)
    np.testing.assert_allclose(exact, [x1_min, m - p1 * x1_min], atol=1e-10)
    np.testing.assert_allclose(num.x, exact, atol=1e-4)
    assert num.binding_floor
