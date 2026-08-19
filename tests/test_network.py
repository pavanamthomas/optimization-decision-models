"""Dijkstra identities and transportation balance."""

from __future__ import annotations

import numpy as np

from optmodels.checks import transportation_balanced
from optmodels.network import (
    dijkstra,
    example_transportation,
    path_length,
    reconstruct_path,
    solve_transportation,
    textbook_dijkstra_graph,
)


def test_dijkstra_matches_hand_computed_graph() -> None:
    n, edges, source = textbook_dijkstra_graph()
    result = dijkstra(n, edges, source)
    np.testing.assert_allclose(result.distances, [0.0, 1.0, 3.0, 6.0], atol=1e-12)
    path = reconstruct_path(result, 3)
    assert path == [0, 1, 2, 3]
    assert path_length(path, edges) == result.distances[3]


def test_transportation_meets_supply_and_demand() -> None:
    supply, demand, cost = example_transportation()
    res = solve_transportation(supply, demand, cost)
    assert res.success
    assert transportation_balanced(res.flows, supply, demand)
    np.testing.assert_allclose(res.cost, float(np.sum(cost * res.flows)), atol=1e-8)
    assert res.cost < 8.0 * 20 + 12.0 * 25
