# Roadmap

Current as of August 2026.

## In scope now

- Unconstrained quadratic and multimodal local search; Cobb–Douglas demand; LP with an independently solved dual; 0–1 knapsack with enumeration on small instances; Dijkstra; transportation LP; cake-eating DP; scenario expected-cost versus maximin.
- Independent substitution checks in `optmodels.checks` (not a second call to the same solver).
- `MODEL_AUDIT_CHECKLIST.md` and `docs/model_catalogue.md`.

## Failures that are part of the design

- A poor local start on a double well is not the global minimiser; multi-start is required to reach the better well.
- Expected-cost optimality is not maximin optimality under the same scenarios.
- Mathematical optimality is not organisational optimality.

Details: `docs/failures_and_corrections.md`.

## Open (issues)

1. Multi-start is not a global certificate except on enumerated instances. Branch-and-bound for the smooth nonconvex programme is not implemented.
2. Integer programmes larger than the enumerated knapsack are solved by CBC; independent enumeration is then impossible. A second solver for cross-check is not in the dependency list.
3. Stochastic programmes are discrete-scenario only. Sample-average approximation with convergence diagnostics is not implemented.
4. Network modules assume non-negative lengths (Dijkstra) and a balanced transportation tableau.

## Explicitly not in scope

- Calibrating coefficients from a plant or a market.
- Claiming uniqueness without a convexity or enumeration argument.
- OR-Tools, unless a second solver becomes a stated validation method.

Close an issue only with a check in `tests/` or a limitation sentence in the catalogue.
