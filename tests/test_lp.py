"""Linear programme feasibility and independent dual check."""

from __future__ import annotations

import numpy as np

from optmodels.checks import lp_primal_feasible, lp_strong_duality
from optmodels.linear_program import solve_resource_allocation, textbook_resource_data


def test_textbook_vertex_and_feasibility() -> None:
    c, A, b = textbook_resource_data()
    res = solve_resource_allocation()
    assert res.success
    np.testing.assert_allclose(res.x, [2.0, 6.0], atol=1e-7)
    np.testing.assert_allclose(res.objective, 36.0, atol=1e-7)
    assert lp_primal_feasible(res.x, A, b)
    assert res.dual is not None
    assert lp_strong_duality(res.objective, float(res.dual_objective))
    # First resource slack; others bind.
    assert res.slacks[0] > 1e-6
    np.testing.assert_allclose(res.slacks[1:], 0.0, atol=1e-7)


def test_nonnegativity_of_activities() -> None:
    res = solve_resource_allocation()
    assert np.all(res.x >= -1e-12)
