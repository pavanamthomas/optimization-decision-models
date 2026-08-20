"""Optimization and decision models: formulation, solution, independent checks."""

from optmodels.constrained import (
    boundary_cobb_douglas,
    marshallian_cobb_douglas,
    maximize_cobb_douglas,
)
from optmodels.dynamic_program import cake_eating_bruteforce, cake_eating_dp
from optmodels.integer_program import enumerate_knapsack, solve_project_selection
from optmodels.linear_program import solve_resource_allocation, textbook_resource_data
from optmodels.network import dijkstra, solve_transportation, textbook_dijkstra_graph
from optmodels.unconstrained import quadratic_closed_form, quadratic_objective
from optmodels.uncertainty import minimize_expected_cost, minimize_maximin_cost, sample_average_newsvendor

__all__ = [
    "boundary_cobb_douglas",
    "cake_eating_bruteforce",
    "cake_eating_dp",
    "dijkstra",
    "enumerate_knapsack",
    "marshallian_cobb_douglas",
    "maximize_cobb_douglas",
    "sample_average_newsvendor",
    "minimize_expected_cost",
    "minimize_maximin_cost",
    "quadratic_closed_form",
    "quadratic_objective",
    "solve_project_selection",
    "solve_resource_allocation",
    "solve_transportation",
    "textbook_dijkstra_graph",
    "textbook_resource_data",
]
