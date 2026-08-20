# Model catalogue

Copyright 2026 Dr. Pavanam Thomas

Each entry follows `MODEL_AUDIT_CHECKLIST.md`. Parameters are illustrative.
Related formal developments: [lean4-optimization-economics](https://github.com/pavanamthomas/lean4-optimization-economics).
Related valuation identities that can appear as inner objectives:
[quantitative-finance-models](https://github.com/pavanamthomas/quantitative-finance-models).

## 1. Convex quadratic programme (`optmodels.unconstrained`)

- **Decision variables.** \(x\in\mathbb{R}^n\).
- **Parameters.** Symmetric matrix \(Q\in\mathbb{R}^{n\times n}\), vector \(c\in\mathbb{R}^n\).
- **Domain.** Unconstrained, \(\mathbb{R}^n\).
- **Objective.** Minimise \(\tfrac12 x^\top Q x + c^\top x\).
- **Constraints.** None.
- **Assumptions.** \(Q\succ 0\), so the programme is strictly convex and the critical point is the unique global minimizer.
- **Solution method.** Closed form \(x^\star=-Q^{-1}c\); BFGS from an arbitrary start for comparison.
- **Validation.** Numerical \(x\) versus the linear solve; Hessian eigenvalues; gradient residual \(Qx+c\).
- **Interpretation.** \(x^\star\) is the unique minimizer of this quadratic, not a statistical estimate.
- **Limitations.** If \(Q\) is indefinite the closed form is not a minimizer. The quadratic is a local model of a smooth \(f\) only near the expansion point.

## 2. Rosenbrock and double-well local search (`optmodels.unconstrained`)

- **Decision variables.** \(x\in\mathbb{R}^n\) (Rosenbrock: \(n=2\); double well: \(n=1\)).
- **Parameters.** Rosenbrock constants \(a=1\), \(b=100\); double-well polynomial coefficients as coded.
- **Domain.** Unconstrained, or a box used only to draw multi-start seeds.
- **Objective.** Rosenbrock \(f(x,y)=(a-x)^2+b(y-x^2)^2\), global min \(0\) at \((a,a^2)\). Double well \(x^4-2x^2+0.5x\), two local minima.
- **Constraints.** None in the programme; box bounds apply only to sampling starts.
- **Assumptions.** Smoothness so that BFGS with an analytic gradient is applicable. Multi-start assumes the basin of the global min is hit at least once.
- **Solution method.** BFGS; central-difference gradient check; independent starts.
- **Validation.** Rosenbrock point versus \((1,1)\); analytic versus finite-difference gradient; double-well multi-start versus evaluation of all real critical points of the cubic derivative.
- **Interpretation.** A local solver reports a stationary point of the start's basin. Multi-start is a comparison, not a proof of globality, except when all critical points are known.
- **Limitations.** Heuristic globalisation; no certificate for general multimodal \(f\).

## 3. Cobb–Douglas utility maximisation (`optmodels.constrained`)

- **Decision variables.** Interior consumption \(x=(x_1,x_2)\).
- **Parameters.** Share \(\alpha\in(0,1)\), prices \(p_1,p_2>0\), income \(m>0\); optional floor \(x_1\ge x_1^{\min}\).
- **Domain.** \(x>0\) (log form of Cobb–Douglas is undefined at 0).
- **Objective.** Maximise \(u(x)=x_1^\alpha x_2^{1-\alpha}\).
- **Constraints.** Equality budget \(p\cdot x=m\), or inequality \(p\cdot x\le m\). Non-negativity. Optional linear floors.
- **Assumptions.** Local nonsatiation (so the inequality budget binds); interior Marshallian demand when floors are inactive; prices and income known.
- **Solution method.** Marshallian closed form \(x_1=\alpha m/p_1\), \(x_2=(1-\alpha)m/p_2\); SLSQP on \(-\log u\) or \(-u\); KKT multiplier recovered as \(u^\star/m\) when the constraint is the budget and demand is interior.
- **Validation.** Budget residual; expenditure shares \(\alpha\) and \(1-\alpha\); numerical versus closed form; binding floor versus the truncated Marshallian point.
- **Interpretation.** \((x_1^\star,x_2^\star)\) is Marshallian demand for this utility and budget. The multiplier is the marginal utility of income *in this problem*.
- **Limitations.** Two goods, one period, no labour supply, no estimation of \(\alpha\). A floor that binds is a different programme, not a "failure" of Cobb–Douglas.

## 4. Resource-allocation LP (`optmodels.linear_program`)

- **Decision variables.** Activity levels \(x\in\mathbb{R}^n_+\).
- **Parameters.** Unit contributions \(c\), technology matrix \(A\), endowments \(b\).
- **Domain.** \(x\ge 0\).
- **Objective.** Maximise \(c^\top x\).
- **Constraints.** \(Ax\le b\).
- **Assumptions.** Linear technology, constant returns, known coefficients, divisibility of activity.
- **Solution method.** `scipy.optimize.linprog` (HiGHS) on the minimisation of \(-c^\top x\). Dual \(\min b^\top y\) s.t. \(A^\top y\ge c\), \(y\ge 0\), solved as a separate LP.
- **Validation.** Independent slacks \(b-Ax\ge 0\); primal objective \(c^\top x\); strong duality \(c^\top x^\star=b^\top y^\star\) on the textbook instance.
- **Interpretation.** \(x^\star\) is a planned mix under the LP. Dual \(y^\star\) is the marginal value of a resource inside the LP, for perturbations that do not change the optimal basis.
- **Limitations.** Fractional activity may be meaningless; omitted constraints (setups, ramping, integer crews) are not "approximately dualised". Coefficients are not estimated.

## 5. Project selection / 0–1 knapsack (`optmodels.integer_program`)

- **Decision variables.** \(x\in\{0,1\}^n\), include or exclude project \(i\).
- **Parameters.** Values \(v\), resource uses \(w\), capacity \(W\); optional pairs with \(x_i+x_j\le 1\).
- **Domain.** Binary hypercube.
- **Objective.** Maximise \(v^\top x\).
- **Constraints.** \(w^\top x\le W\); optional mutual exclusion.
- **Assumptions.** Projects are indivisible; values and uses are additive; no sequencing or option value over time.
- **Solution method.** 0–1 IP via PuLP with CBC; complete enumeration for \(n\le 20\).
- **Validation.** Integrality; knapsack inequality; match with enumeration on small \(n\).
- **Interpretation.** A set of accepted projects under the stated budget. Not an NPV model of stochastic cash flows (see the finance repository for valuation identities used as *inputs* \(v_i\)).
- **Limitations.** CBC returns a feasible integer point to solver tolerance; capital rationing in practice includes timing, mutually exclusive real options, and mismeasured \(v_i\).

## 6. Shortest path, Dijkstra (`optmodels.network`)

- **Decision variables.** A walk from source \(s\) to a node \(t\), or equivalently labels \(d_v\).
- **Parameters.** Directed (or undirected) graph with finitely many nodes and non-negative finite lengths \(w_{uv}\).
- **Domain.** Simple walks; Dijkstra returns a tree of shortest paths from \(s\).
- **Objective.** Minimise path length \(\sum w_{uv}\) along the walk.
- **Constraints.** Incidence of a walk; no negative lengths.
- **Assumptions.** Lengths are additive and known; no time windows or capacities.
- **Solution method.** Dijkstra with a binary heap, implemented in this repository (not NetworkX).
- **Validation.** Distances on a four-node graph against a hand expansion of the frontier.
- **Interpretation.** \(d_t\) is the least-length \(s\)–\(t\) walk in this weighted graph.
- **Limitations.** Negative lengths require a different algorithm; the method is not a traffic equilibrium.

## 7. Transportation LP (`optmodels.network`)

- **Decision variables.** Shipments \(x_{ij}\ge 0\) from source \(i\) to sink \(j\).
- **Parameters.** Supplies \(s_i\), demands \(d_j\), unit costs \(c_{ij}\), with \(\sum s_i=\sum d_j\).
- **Domain.** Non-negative reals (divisible freight).
- **Objective.** Minimise \(\sum_{i,j} c_{ij}x_{ij}\).
- **Constraints.** Row sums \(s\), column sums \(d\).
- **Assumptions.** Balanced problem; uncapacitated arcs; linear cost; bipartite structure. Uncapacitated bipartite min-cost flow is this transportation problem.
- **Solution method.** Equality-constrained LP via `linprog`.
- **Validation.** Supply and demand residuals; independent cost \(\sum c_{ij}x_{ij}\).
- **Limitations.** Capacitated transshipment on a general digraph is a different LP; integer tons require an IP.

## 8. Finite-horizon cake-eating (`optmodels.dynamic_program`)

- **Decision variables.** Consumption \(c_t\) and remainder \(w_{t+1}=w_t-c_t\) for \(t=0,\ldots,T-1\).
- **Parameters.** Horizon \(T\), discount \(\beta\in(0,1]\), initial cake \(W\), utility \(u(c)=\sqrt{c}\) on the integer grid (and \(\log\) for the textbook closed form on a positive continuum).
- **Domain.** Discrete cake \(w\in\{0,1,\ldots,W\}\) for the Bellman and the brute-force check; \(c\in\{0,\ldots,w\}\).
- **Objective.** Maximise \(\sum_{t=0}^{T-1}\beta^t u(c_t)\) with \(c_{T-1}=w_{T-1}\).
- **Constraints.** Cake accounting; no borrowing the future.
- **Assumptions.** Deterministic remaining stock; time-separable utility; grid is the true state space for the discrete model.
- **Solution method.** Backward induction on the grid; complete enumeration of consumption sequences for tiny \(T,W\). Log-utility closed form \(c_t=(1-\beta)w_t/(1-\beta^{T-t})\) as a separate continuum identity.
- **Validation.** Bellman value at \((t,W)\) equals brute-force value on the same grid and \(u\).
- **Interpretation.** An optimal consumption policy *on this grid and utility*. Inventory with leftover stock as the state is the same recursion with a different name for \(w\).
- **Limitations.** Grid approximation of a continuous cake is a different object; the brute-force test does not validate the continuum. No price process, no labour.

## 9. Scenario expected cost and maximin (`optmodels.uncertainty`)

- **Decision variables.** Production (or order) quantity \(x\ge 0\) chosen before demand is known.
- **Parameters.** Scenario demands \(d_s\), probabilities \(p_s\) (expected-cost only), unit production cost \(c\), holding \(h\), shortage penalty \(\pi\).
- **Domain.** \(x\ge 0\), typically in \([\min d_s,\max d_s]\) for the interesting region.
- **Objective.** Expected-cost: minimise \(\sum_s p_s\,\mathrm{cost}(x,d_s)\). Maximin: minimise \(\max_s\mathrm{cost}(x,d_s)\). \(\mathrm{cost}(x,d)=c x+h(x-d)_++\pi(d-x)_+\).
- **Constraints.** Non-negativity; no recourse production in the base model.
- **Assumptions.** Demand is one of the listed scenarios. Expected-cost uses \(p_s\) as preferences over scenarios. Maximin ignores \(p_s\). Linear holding and shortage.
- **Solution method.** Bounded scalar minimisation; critical-fractile comparison when the empirical distribution is a convenient check; one-at-a-time sweeps of \(\pi\).
- **Validation.** Grid comparison: expected cost at \(x^\star\) is no worse than at neighbouring grid points; maximin worst-case cost is no worse than at those neighbours; monotonicity of \(x^\star\) in \(\pi\) on the example.
- **Interpretation.** Two different decision criteria. Neither is a statistical forecast. A more conservative maximin quantity is a property of \(\max_s\mathrm{cost}\), not a moral claim.
- **Limitations.** Missed scenarios are unmodelled. Risk measures other than expectation and maximin (CVaR, expected utility) are not implemented here.

## 10. Sample-average newsvendor (`optmodels.uncertainty.sample_average_newsvendor`)

- **Decision variables.** The same \(x\ge 0\) as the scenario programme.
- **Parameters.** Sample size \(N\), Normal demand mean and sd (truncated at 0), \(c,h,\pi\), seed.
- **Domain.** Bounded interval built from the sampled demands.
- **Objective.** SAA: minimise \(N^{-1}\sum_{i=1}^N \mathrm{cost}(x,d_i)\).
- **Assumptions.** Draws are iid from the stated truncated-normal sampling measure. \(N\) is the sample size, not a fixed three-scenario list.
- **Solution method.** The same bounded scalar minimiser as expected-cost, with equal weights \(1/N\).
- **Validation.** Feasibility of \(x\) in the sampled bounds; different \(N\) under the same seed are different programmes (`tests/test_uncertainty.py`).
- **Interpretation.** An SAA point is optimal for the sampled programme. Solver `success` is not convergence in \(N\).
- **Limitations.** No SAA rate theorem is proved. CVaR is not implemented.
