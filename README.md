# optimization-decision-models

[![CI](https://github.com/pavanamthomas/optimization-decision-models/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/optimization-decision-models/actions)

Mathematical optimization, operations research, decision modelling, sensitivity analysis, and numerical verification.

This repository states each model as a mathematical programme, computes a solution by a declared method, and checks the reported point against the original objective and constraints. Parameters in scripts, tests, and the notebook are **illustrative**. They are not estimated from a plant, a market, or a survey.

Related work:

- Formal statements of programmes and comparative statics in Lean: [lean4-optimization-economics](https://github.com/pavanamthomas/lean4-optimization-economics)
- Valuation, mean–variance algebra, and risk identities that sometimes appear as inner problems here: [quantitative-finance-models](https://github.com/pavanamthomas/quantitative-finance-models)

## Problem this repository addresses

A decision is written as: choose a point in a stated domain to extremize a stated objective subject to stated constraints. The library covers unconstrained smooth programmes, equality- and inequality-constrained nonlinear programmes, linear and 0–1 integer programmes, shortest paths, transportation, finite-horizon dynamic programmes, scenario-based choice under uncertainty, and a sample-average newsvendor with sample size N.

The scientific claim is limited: given the symbols as written, the computed point satisfies the first-order or discrete optimality conditions that the chosen method is designed to enforce, up to documented numerical tolerance. The claim is not that the same numbers should be implemented in an organisation.

## What to inspect first

If the goal is to see how the work is specified and checked, read in this order:

1. `tests/` — feasibility, integrality, closed-form agreement, and independent residual checks
2. `MODEL_AUDIT_CHECKLIST.md` — questions that must be answered for every model
3. `docs/model_catalogue.md` — decision variables, parameters, domain, objective, constraints, assumptions, method, validation, interpretation, limitations

Then `src/optmodels/` and `scripts/run_all.py`. Remaining bounds: `ROADMAP.md` (issues #3 and #4 are closed). Recorded failures: `docs/failures_and_corrections.md`.

## What is exercised

- Translating a verbal allocation or routing task into variables, a feasible set, and an objective
- Distinguishing convex quadratic programmes with a unique minimizer from multimodal local search
- Reading a Lagrange multiplier or an LP dual variable as a marginal value *inside the model*
- Checking a solver output by substituting it back into the original functions (`optmodels.checks`)
- Enumerating a tiny discrete programme to confirm a Bellman recursion or a 0–1 knapsack
- Separating expected-cost optimality from maximin (minimax-cost) optimality under scenarios, and SAA from a fixed three-scenario list

## Methods

| Class | Module | Method |
| --- | --- | --- |
| Convex quadratic | `optmodels.unconstrained` | Closed form \(x^\star=-Q^{-1}c\) when \(Q\succ 0\); BFGS for comparison |
| Smooth unconstrained | `optmodels.unconstrained` | Analytic gradient, central-difference check, multi-start local search |
| Cobb–Douglas demand | `optmodels.constrained` | Marshallian closed form; SLSQP on equality and inequality budget sets |
| Linear programme | `optmodels.linear_program` | Primal via `scipy.optimize.linprog`; dual solved independently |
| 0–1 integer | `optmodels.integer_program` | PuLP/CBC; SciPy MILP cross-check; brute-force enumeration on small instances |
| Shortest path | `optmodels.network` | Dijkstra (non-negative lengths; no NetworkX) |
| Transportation | `optmodels.network` | Balanced transportation as an equality-constrained LP |
| Dynamic programme | `optmodels.dynamic_program` | Backward Bellman recursion on a finite grid; brute-force on tiny \(T,W\) |
| Uncertainty | `optmodels.uncertainty` | Scenario expected-cost minimisation versus maximin; one-parameter sensitivity |

`optmodels.checks` does not call a solver. It evaluates objectives, slacks, Marshallian shares, integrality, and finite-difference gradients at a supplied point.

## Data

There is no empirical dataset. Resource coefficients, utilities, distances, project values, and demand scenarios are numbers chosen so that closed forms, duals, or complete enumerations exist. Treat them as a worked example, not as a calibration.

## Assumptions (shared)

- The feasible set and the objective are those written in the catalogue, not an undocumented shop-floor constraint.
- Coefficients are known and constant unless a sensitivity sweep is explicit.
- Linear programmes treat activity as divisible; integer programmes do not.
- Dijkstra requires non-negative arc lengths.
- Scenario probabilities used in expected-cost problems are part of the model; maximin does not use them.
- Numerical methods stop at finite tolerance. Agreement with a closed form is a check of implementation, not a proof that the economic model is true.

Model-specific assumptions are in `docs/model_catalogue.md`.

## Methodology

Each programme is written as variables, a feasible set, and an objective.
A solver status is not the validation step: the reported point is substituted
back into the original functions (`optmodels.checks`). The catalogue records
what the solution does and does not license.

## Contents

| Area | Module | Object |
| --- | --- | --- |
| Unconstrained | `optmodels.unconstrained` | Convex quadratic, Rosenbrock, double-well multi-start |
| Constrained | `optmodels.constrained` | Two-good Cobb–Douglas, budget equality/inequality, binding floors |
| Linear | `optmodels.linear_program` | Resource allocation, primal feasibility, independent dual |
| Integer | `optmodels.integer_program` | Project selection / 0–1 knapsack |
| Network | `optmodels.network` | Dijkstra; transportation LP |
| Dynamic | `optmodels.dynamic_program` | Finite-horizon cake-eating |
| Uncertainty | `optmodels.uncertainty` | Scenario expected cost, maximin, sensitivity |
| Checks | `optmodels.checks` | Independent substitution into f and g |
| Figures | `optmodels.plots` | Contours, budget sets, feasible regions, value functions, sensitivity |

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Dependencies are `numpy`, `scipy`, `matplotlib`, `pulp`, and (for tests) `pytest`. Integer programmes use PuLP's bundled CBC interface; the default knapsack is also solved with `scipy.optimize.milp` and checked against enumeration.

## Reproducing the demonstrations

```bash
python scripts/run_all.py
python -m pytest -q
```

`scripts/run_all.py` writes figures under `outputs/figures/` and tables under `outputs/tables/`. Those directories are output locations; PNG and CSV files are not source data. Numerical values belong in that output, not in this README.

The notebook `notebooks/01_resource_allocation.ipynb` walks through a two-activity linear programme using the same methodology sequence.

## Tests

Tests are written as properties the mathematics requires:

- The quadratic minimizer matches \(-Q^{-1}c\) when \(Q\) is positive definite.
- Analytic and central-difference gradients agree on Rosenbrock.
- Multi-start recovers the better critical point of a double well that a poor local start misses.
- Cobb–Douglas numerical demand exhausts the budget and matches Marshallian shares \(x_1=\alpha m/p_1\).
- A binding consumption floor produces a boundary solution with slack-free budget.
- The LP primal is feasible and the independently solved dual matches the primal objective (strong duality in the example).
- The integer solution is binary and feasible; on a small knapsack PuLP, SciPy MILP, and complete enumeration agree.
- Dijkstra distances match a hand-computed graph with non-negative lengths.
- The transportation plan meets supply and demand and attains the LP objective.
- The cake-eating Bellman value matches brute-force enumeration on a tiny state space.
- `checks.py` recomputes objective and constraint residuals without trusting solver status strings.

## Continuous integration

`.github/workflows/ci.yml` runs on `ubuntu-latest` with Python 3.11: `pip install -e ".[dev]"`, `pytest -q`, then `python scripts/run_all.py`.

## Limitations

- Illustrative coefficients are not a production plan, a capital budget, or an estimated demand system.
- Local search can stop at a local minimizer; multi-start is a heuristic, not a global certificate, except on the tiny enumerated instances.
- Shadow prices are derivatives of the *modelled* optimal value with respect to a right-hand side, valid in a neighbourhood where the basis (or active set) stays the same.
- Dijkstra is not a method for negative lengths; transportation here is uncapacitated and balanced.
- Finite-grid dynamic programming approximates a continuous state; the brute-force check is on the same grid, not on the underlying continuous programme.
- Expected-cost optimality is not maximin optimality. Neither is a forecast of demand.
- A correct solution of a wrong programme is still a wrong recommendation.

## Author

Dr. Pavanam Thomas  
GitHub: [pavanamthomas](https://github.com/pavanamthomas)  
Email: thomaspavanam@gmail.com

## Citation

See `CITATION.cff`.

## License

Copyright 2026 Dr. Pavanam Thomas. MIT License; see `LICENSE`.
