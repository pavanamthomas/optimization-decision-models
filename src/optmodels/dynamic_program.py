"""Finite-horizon dynamic programmes.

The discrete cake-eating problem is solved by backward induction on an
integer wealth grid. Complete enumeration of consumption sequences is the
validation method on tiny (T, W). Log utility on a continuum has a
textbook closed form used as a separate identity, not as a check of the
integer-grid model.

Inventory with leftover stock as the state is the same recursion with a
different name for remaining cake.

Decision variables, parameters, and limitations: ``docs/model_catalogue.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CakeResult:
    value: np.ndarray
    policy: np.ndarray
    beta: float
    n_periods: int
    wealth_cap: int


def sqrt_utility(c: float) -> float:
    if c < 0.0:
        return -np.inf
    return float(np.sqrt(c))


def log_utility(c: float) -> float:
    if c <= 0.0:
        return -np.inf
    return float(np.log(c))


def cake_eating_dp(
    wealth_cap: int,
    n_periods: int,
    beta: float,
    *,
    utility=sqrt_utility,
) -> CakeResult:
    """Backward induction on w in {0,...,W}, c in {0,...,w}.

    Decision variables: consumption c_t and remainder w_{t+1} = w_t - c_t.
    Objective: maximise sum_{t=0}^{T-1} beta^t u(c_t), last period eats all.
    Constraints: cake accounting; no borrowing.
    Solution method: Bellman recursion on the integer grid.
    """
    if wealth_cap < 0 or n_periods < 1:
        raise ValueError("wealth_cap >= 0 and n_periods >= 1 required")
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must lie in (0, 1]")

    W = wealth_cap
    T = n_periods
    value = np.full((T, W + 1), -np.inf, dtype=float)
    policy = np.zeros((T, W + 1), dtype=int)

    for w in range(W + 1):
        value[T - 1, w] = utility(float(w))
        policy[T - 1, w] = w

    for t in range(T - 2, -1, -1):
        for w in range(W + 1):
            best = -np.inf
            best_c = 0
            for c in range(w + 1):
                rem = w - c
                val = utility(float(c)) + beta * value[t + 1, rem]
                if val > best:
                    best = val
                    best_c = c
            value[t, w] = best
            policy[t, w] = best_c

    return CakeResult(
        value=value,
        policy=policy,
        beta=beta,
        n_periods=T,
        wealth_cap=W,
    )


def cake_eating_bruteforce(
    wealth: int,
    n_periods: int,
    beta: float,
    *,
    utility=sqrt_utility,
) -> float:
    """Enumerate all integer consumption sequences of a cake of size W.

    Value is sum_t beta^t u(c_t) with c_{T-1} equal to the remainder.
    Intended for tiny T and W only.
    """
    if wealth < 0 or n_periods < 1:
        raise ValueError("wealth >= 0 and n_periods >= 1 required")
    if n_periods > 6 or wealth > 12:
        raise ValueError("brute-force instance is too large")

    best = -np.inf

    def rec(t: int, remaining: int, acc: float) -> None:
        nonlocal best
        if t == n_periods - 1:
            val = acc + (beta**t) * utility(float(remaining))
            if val > best:
                best = val
            return
        for c in range(remaining + 1):
            rec(t + 1, remaining - c, acc + (beta**t) * utility(float(c)))

    rec(0, wealth, 0.0)
    return float(best)


def policy_value(
    result: CakeResult,
    initial_wealth: int,
    *,
    utility=sqrt_utility,
) -> float:
    """Evaluate the stored policy from a given initial cake."""
    if not (0 <= initial_wealth <= result.wealth_cap):
        raise ValueError("initial wealth outside the grid")
    w = initial_wealth
    acc = 0.0
    for t in range(result.n_periods):
        c = int(result.policy[t, w])
        acc += (result.beta**t) * utility(float(c))
        w = w - c
        if w < 0:
            raise RuntimeError("policy consumed more than remaining cake")
    return float(acc)


def cake_eating_log_closed_form(
    wealth: float,
    n_periods: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuum identity for u = log, finite horizon, no grid.

    c_t = (1 - beta) / (1 - beta^{T-t}) * w_t when beta != 1, and
    c_t = w_t / (T - t) when beta = 1. Returns consumption and wealth paths.
    This is not a check of the integer-grid DP with sqrt utility.
    """
    if wealth <= 0.0 or n_periods < 1:
        raise ValueError("positive wealth and n_periods >= 1 required")
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must lie in (0, 1]")
    c = np.zeros(n_periods)
    w = np.zeros(n_periods + 1)
    w[0] = wealth
    for t in range(n_periods):
        remaining_periods = n_periods - t
        if abs(beta - 1.0) < 1e-15:
            c[t] = w[t] / remaining_periods
        else:
            disc = 1.0 - beta**remaining_periods
            c[t] = (1.0 - beta) / disc * w[t]
        w[t + 1] = w[t] - c[t]
    return c, w
