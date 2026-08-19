"""Bellman recursion versus brute-force enumeration."""

from __future__ import annotations

from optmodels.checks import cake_accounting
from optmodels.dynamic_program import cake_eating_bruteforce, cake_eating_dp, policy_value


def test_bellman_matches_bruteforce_on_tiny_instance() -> None:
    W, T, beta = 5, 3, 0.9
    dp = cake_eating_dp(wealth_cap=W, n_periods=T, beta=beta)
    brute = cake_eating_bruteforce(wealth=W, n_periods=T, beta=beta)
    assert abs(dp.value[0, W] - brute) < 1e-10
    assert abs(policy_value(dp, W) - brute) < 1e-10
    assert cake_accounting(dp.policy, W)


def test_last_period_eats_the_remainder() -> None:
    dp = cake_eating_dp(wealth_cap=4, n_periods=2, beta=1.0)
    for w in range(5):
        assert int(dp.policy[-1, w]) == w
