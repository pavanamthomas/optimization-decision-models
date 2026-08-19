"""Integer project selection: integrality, feasibility, enumeration."""

from __future__ import annotations

import numpy as np

from optmodels.checks import knapsack_feasible
from optmodels.integer_program import enumerate_knapsack, example_projects, solve_project_selection


def test_solver_matches_enumeration_on_default_instance() -> None:
    values, costs, budget, _names = example_projects()
    pairs = ((0, 1),)
    solved = solve_project_selection()
    enumerated = enumerate_knapsack(values, costs, budget, exclusive_pairs=pairs)
    assert solved.success
    assert knapsack_feasible(solved.x, costs, budget, exclusive_pairs=pairs)
    assert np.all(np.isin(np.round(solved.x), [0.0, 1.0]))
    np.testing.assert_allclose(solved.objective, enumerated.objective, atol=1e-8)
    np.testing.assert_allclose(solved.x, enumerated.x, atol=1e-8)


def test_capacity_is_respected() -> None:
    values, costs, budget, _ = example_projects()
    res = solve_project_selection()
    assert res.resource_use <= budget + 1e-8
    assert knapsack_feasible(res.x, costs, budget, exclusive_pairs=((0, 1),))
