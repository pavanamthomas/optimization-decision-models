# Roadmap

Current as of August 2026.

## In scope now

- Unconstrained quadratic and multimodal local search; Cobb–Douglas demand; LP with an independently solved dual; 0–1 knapsack with enumeration on small instances; Dijkstra; transportation LP; cake-eating DP; scenario expected-cost versus maximin; SAA newsvendor with sample size N.
- Independent substitution checks in `optmodels.checks` (not a second call to the same solver).
- `MODEL_AUDIT_CHECKLIST.md` and `docs/model_catalogue.md`.

## Failures that are part of the design

- A poor local start on a double well is not the global minimiser; multi-start is required to reach the better well.
- Expected-cost optimality is not maximin optimality under the same scenarios.
- Mathematical optimality is not organisational optimality.

Details: `docs/failures_and_corrections.md`.

## Remaining bounds

Issues #3 and #4 were closed after `solve_knapsack_scipy_milp` matched enumeration
on the default instance and `sample_average_newsvendor` used sample size N.
Still unimplemented:

1. Multi-start is still not a global certificate except on enumerated instances.
2. SAA is illustrated for a truncated-normal newsvendor. Convergence rates and CVaR/expected-utility criteria are not implemented.
3. Dijkstra still requires non-negative lengths.

## Explicitly not in scope

- Calibrating coefficients from a plant or a market.
- Claiming uniqueness without a convexity or enumeration argument.
- OR-Tools, unless a second solver becomes a stated validation method.

Close an issue only with a check in `tests/` or a limitation sentence in the catalogue.
