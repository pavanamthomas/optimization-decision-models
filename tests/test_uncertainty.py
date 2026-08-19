"""Expected-cost versus maximin under discrete demand scenarios."""

from __future__ import annotations

import numpy as np

from optmodels.uncertainty import (
    example_demand_scenarios,
    expected_cost,
    minimize_expected_cost,
    minimize_maximin_cost,
    sensitivity_penalty,
    worst_case_cost,
)


def test_expected_cost_minimizer_beats_neighbourhood() -> None:
    demands, probs = example_demand_scenarios()
    res = minimize_expected_cost(demands, probs, unit_cost=1.0, holding=0.4, penalty=3.0)
    assert res.success
    for shift in (-2.0, -0.5, 0.5, 2.0):
        alt = expected_cost(res.x + shift, demands, probs, 1.0, 0.4, 3.0)
        assert res.value <= alt + 1e-6


def test_maximin_is_not_the_same_as_expected_cost() -> None:
    demands, probs = example_demand_scenarios()
    ev = minimize_expected_cost(demands, probs, 1.0, 0.4, 3.0)
    mm = minimize_maximin_cost(demands, 1.0, 0.4, 3.0)
    assert ev.success and mm.success
    # Criteria differ: the maximin value is a worst-case cost, not an expectation.
    assert abs(ev.x - mm.x) > 1e-3 or abs(ev.value - mm.value) > 1e-3
    assert mm.value >= worst_case_cost(mm.x, demands, 1.0, 0.4, 3.0) - 1e-6
    assert ev.value <= expected_cost(mm.x, demands, probs, 1.0, 0.4, 3.0) + 1e-6


def test_expected_cost_quantity_rises_with_shortage_penalty() -> None:
    demands, probs = example_demand_scenarios()
    penalties = np.array([1.5, 2.5, 4.0, 8.0])
    _grid, xs, _vals = sensitivity_penalty(demands, probs, 1.0, 0.4, penalties)
    assert np.all(np.diff(xs) >= -1e-6)
    assert xs[-1] > xs[0] + 0.5
