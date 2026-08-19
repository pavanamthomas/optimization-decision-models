"""Integer project selection: integrality, feasibility, enumeration."""

from __future__ import annotations

import numpy as np

from optmodels.checks import knapsack_feasible
from optmodels.integer_program import (
    enumerate_knapsack,
    example_projects,
    solve_knapsack_scipy_milp,
    solve_project_selection,
)


def test_solver_matches_enumeration_on_default_instance() -> None:
    values, costs, budget, _names = example_projects()
    pairs = ((0, 1),)
    solved = solve_project_selection()
    enumerated = enumerate_knapsack(values, costs, budget, exclusive_pairs=pairs)
    assert solved.success
    assert knapsack_feasible(solved.x, costs, budget, exclusive_pairs=pairs)
    assert np.max(np.abs(solved.x - np.round(solved.x))) < 1e-8
    assert np.all(np.isin(np.round(solved.x), [0.0, 1.0]))
    np.testing.assert_allclose(solved.objective, enumerated.objective, atol=1e-8)
    np.testing.assert_allclose(solved.x, enumerated.x, atol=1e-8)


def test_capacity_is_respected() -> None:
    values, costs, budget, _ = example_projects()
    res = solve_project_selection()
    assert res.resource_use <= budget + 1e-8
    assert knapsack_feasible(res.x, costs, budget, exclusive_pairs=((0, 1),))


def test_scipy_milp_matches_enumeration_on_default_instance() -> None:
    values, costs, budget, _names = example_projects()
    pairs = ((0, 1),)
    milp_res = solve_knapsack_scipy_milp(
        values, costs, budget, exclusive_pairs=pairs
    )
    enumerated = enumerate_knapsack(values, costs, budget, exclusive_pairs=pairs)
    assert milp_res.success
    assert knapsack_feasible(milp_res.x, costs, budget, exclusive_pairs=pairs)
    np.testing.assert_allclose(milp_res.objective, enumerated.objective, atol=1e-6)
    np.testing.assert_allclose(milp_res.x, enumerated.x, atol=1e-6)
