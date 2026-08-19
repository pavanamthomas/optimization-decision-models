# Model audit checklist

Copyright 2026 Dr. Pavanam Thomas

This checklist is the working protocol for every programme in the repository.
It is a sequence of modelling decisions, not a list of solver flags. Shared
methodology: Problem → formalization → assumptions → computation/estimation →
validation → interpretation → limitations.

Filled answers for the implemented models are in `docs/model_catalogue.md`.
Independent numerical substitution is in `src/optmodels/checks.py`.

## 1. Problem

State the decision in one sentence: who chooses what, over what horizon, and
what is counted as success. Separate a descriptive question (what was chosen)
from a prescriptive one (what the programme says to choose). Optimization
answers the latter, and only under the model that follows.

## 2. Decision variables

Name the vector \(x\) (or policy \(\pi\)) and its meaning in units. Record
whether coordinates are continuous, integer, or binary. If the object is a
sequence, write the time index.

## 3. Parameters

List coefficients that the decision maker does not choose: prices, resource
endowments, arc lengths, discount factors, scenario probabilities. State the
source. In this repository the source is "illustrative, chosen for a closed
form or a complete enumeration", not a statistical estimate.

## 4. Domain

Write the ambient space (\(x\in\mathbb{R}^n\), \(x\in\{0,1\}^n\), \(x\ge 0\))
before adding functional constraints. Note bounds that are physical (hours
cannot be negative) versus conventional (a modelling truncation of a grid).

## 5. Objective

Write \(f(x)\) or \(\mathbb{E}[f(x,\tilde s)]\) or \(\max_s f(x,s)\). State
whether the problem is max or min. Record units. Do not change the objective
after seeing the solver output.

## 6. Constraints

Write equalities and inequalities as functions of \(x\). Record which are
technological, accounting identities (budget, flow balance), and logical
(mutual exclusion). For networks, state conservation and capacity separately.

## 7. Assumptions

List conditions that make the programme a valid prescription: convexity or
unimodularity if claimed; known and constant coefficients; divisibility;
non-negative lengths; risk attitude (expected cost versus maximin); perfect
recall in a dynamic programme. If an assumption is used only to obtain a
closed form, say so.

## 8. Solution method

Name the algorithm and the software entry point (closed form, BFGS, SLSQP,
HiGHS via `linprog`, CBC via PuLP, Dijkstra, backward induction). Record
tolerances and, for local search, the starting rule and the multi-start
protocol. A method that assumes convexity is misapplied on a multimodal
objective.

## 9. Validation

At least one check that does not reuse the solver's own optimality flag:

- substitute \(x^\star\) into \(f\) and into every constraint (`checks.py`)
- compare with a closed form (quadratic, Marshallian shares, log cake rule)
- compare with an independently solved dual (LP strong duality)
- compare with complete enumeration on a tiny discrete instance
- compare analytic gradients with central differences

A small residual is evidence that *this* instance was solved as written. It
is not evidence that the programme describes an organisation.

## 10. Interpretation

State what \(x^\star\) is allowed to mean. Dual variables and Lagrange
multipliers are marginal values of relaxing a constraint *in the model*,
at the computed point, holding the rest of the programme fixed. Do not read
them as market prices unless the model has a market.

## 11. Sensitivity

Change one parameter at a time. Record whether the active set or the support
of a discrete solution jumps. Expected-cost and maximin solutions need not
move together. Sensitivity inside a misspecified programme is still
misspecified.

## 12. Limitations

Write what the solution cannot answer: integer restrictions ignored by an LP;
global optimality on a nonconvex objective; behaviour off the demand
scenarios; values far from a discrete wealth grid; implementation costs and
constraints omitted from \(f\) and \(g\).

## 13. Descriptive versus normative

Keep the language exact. "The unique minimizer of this convex quadratic is
\(x^\star\)" is a mathematical statement. "A manager should set production to
\(x^\star\)" is a normative statement that requires the programme to be the
right one. This repository tests the former. It does not establish the latter.
