# Failures and corrections

A solver return is not a result until it is substituted back into the stated programme. Several “optimal” reports are false or incomplete on the laboratory examples.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| Local unconstrained start in the worse well of a double-well objective | Stationary point is not the global minimiser | Compare `fun` to multi-start | Multi-start, or restrict to a convex programme | `tests/test_unconstrained.py::test_multi_start_finds_better_well_than_poor_local_start` | Number of starts needed on other landscapes |
| Reading expected-cost quantity as a maximin hedge | The two programmes choose different points | Separate objectives | State the criterion; do not mix them | `tests/test_uncertainty.py::test_maximin_is_not_the_same_as_expected_cost` | Ambiguity aversion beyond maximin |
| Accepting an LP point without the dual | Primal feasibility alone does not check complementary slackness | Independent dual solve; residual checks | `optmodels.checks` on the reported point | `tests/test_lp.py` | Degeneracy and multiple optima on other tableaus |
| Treating a 0–1 CBC solution as integer without a check | Floating-point reports can look continuous | Integrality residual; enumeration on the default instance | Assert 0–1 and match enumeration | `tests/test_ip.py` | Large MIPs where enumeration is impossible |
| Reporting Dijkstra on a graph with negative lengths | Algorithm assumption violated | Documented non-negativity | Refuse, or switch method | `src/optmodels/network.py` docstring; `tests/test_network.py` | Bellman–Ford is not implemented |

Process: `docs/lab_process.md`. Open extensions: `ROADMAP.md`.
