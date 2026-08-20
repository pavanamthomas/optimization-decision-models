"""Expected-cost versus maximin under discrete demand scenarios."""

from __future__ import annotations

import numpy as np

from optmodels.uncertainty import (
    example_demand_scenarios,
    expected_cost,
    minimize_expected_cost,
    minimize_maximin_cost,
    sample_average_newsvendor,
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


def test_saa_n_is_the_sample_size_not_a_fixed_scenario_list() -> None:
    small = sample_average_newsvendor(n_sample=6, seed=2026)
    large = sample_average_newsvendor(n_sample=40, seed=2026)
    assert small.n_sample == 6
    assert large.n_sample == 40
    assert small.demands.size == 6
    assert large.demands.size == 40
    assert small.x != large.x or small.saa_value != large.saa_value


def test_saa_point_is_feasible_on_the_sampled_programme() -> None:
    res = sample_average_newsvendor(n_sample=20, seed=7)
    assert res.success
    assert res.feasible
    lo = 0.0
    hi = float(np.max(res.demands)) * 1.5 + 1.0
    assert lo <= res.x <= hi
    # Solver success does not imply that two sample sizes agree.
    other = sample_average_newsvendor(n_sample=8, seed=7)
    assert other.success
    assert abs(res.saa_value - other.saa_value) > 1e-8 or abs(res.x - other.x) > 1e-8
