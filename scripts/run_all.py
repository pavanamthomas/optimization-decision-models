"""Reproduce illustrative solutions, tables, and figures."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from optmodels.constrained import marshallian_cobb_douglas, maximize_cobb_douglas
from optmodels.dynamic_program import cake_eating_dp
from optmodels.integer_program import example_projects, solve_project_selection
from optmodels.linear_program import solve_resource_allocation
from optmodels.network import example_transportation, solve_transportation, textbook_dijkstra_graph, dijkstra
from optmodels.plots import (
    plot_budget_set,
    plot_cake_value,
    plot_lp_feasible,
    plot_penalty_sensitivity,
    plot_quadratic_contours,
)
from optmodels.unconstrained import quadratic_closed_form
from optmodels.uncertainty import example_demand_scenarios, minimize_expected_cost, minimize_maximin_cost

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    Q = np.array([[4.0, 1.0], [1.0, 3.0]])
    c = np.array([-2.0, 1.0])
    xq = quadratic_closed_form(Q, c)
    _write_csv(TAB / "quadratic_minimizer.csv", [{"x1": xq[0], "x2": xq[1]}])

    alpha, p1, p2, m = 0.4, 2.0, 1.0, 10.0
    marsh = marshallian_cobb_douglas(alpha, p1, p2, m)
    num = maximize_cobb_douglas(alpha, p1, p2, m)
    _write_csv(
        TAB / "cobb_douglas.csv",
        [
            {"source": "marshallian", "x1": marsh[0], "x2": marsh[1]},
            {"source": "slsqp", "x1": num.x[0], "x2": num.x[1]},
        ],
    )

    lp = solve_resource_allocation()
    _write_csv(
        TAB / "resource_allocation.csv",
        [{"x1": lp.x[0], "x2": lp.x[1], "objective": lp.objective, "dual_objective": lp.dual_objective}],
    )

    ip = solve_project_selection()
    values, costs, budget, names = example_projects()
    _write_csv(
        TAB / "project_selection.csv",
        [{"name": names[i], "selected": int(round(ip.x[i])), "value": values[i], "cost": costs[i]} for i in range(len(names))],
    )

    n, edges, source = textbook_dijkstra_graph()
    sp = dijkstra(n, edges, source)
    _write_csv(
        TAB / "shortest_path.csv",
        [{"node": i, "distance": sp.distances[i]} for i in range(n)],
    )

    supply, demand, cost = example_transportation()
    tr = solve_transportation(supply, demand, cost)
    _write_csv(
        TAB / "transportation.csv",
        [{"i": i, "j": j, "flow": tr.flows[i, j]} for i in range(tr.flows.shape[0]) for j in range(tr.flows.shape[1])],
    )

    cake = cake_eating_dp(8, 4, 0.9)
    _write_csv(
        TAB / "cake_eating_value.csv",
        [{"wealth": w, "value0": cake.value[0, w]} for w in range(cake.wealth_cap + 1)],
    )

    demands, probs = example_demand_scenarios()
    ev = minimize_expected_cost(demands, probs, 1.0, 0.4, 3.0)
    mm = minimize_maximin_cost(demands, 1.0, 0.4, 3.0)
    _write_csv(
        TAB / "uncertainty.csv",
        [
            {"criterion": ev.criterion, "x": ev.x, "value": ev.value},
            {"criterion": mm.criterion, "x": mm.x, "value": mm.value},
        ],
    )

    plot_quadratic_contours(FIG / "quadratic_contours.png")
    plot_budget_set(FIG / "budget_set.png")
    plot_lp_feasible(FIG / "lp_feasible.png")
    plot_cake_value(FIG / "cake_value.png")
    plot_penalty_sensitivity(FIG / "penalty_sensitivity.png")

    print("Problem -> formalization -> assumptions -> computation -> validation -> interpretation -> limitations")
    print("Wrote figures to", FIG)
    print("Wrote tables to", TAB)
    print("Illustrative parameters only; not empirical estimates.")


if __name__ == "__main__":
    main()
