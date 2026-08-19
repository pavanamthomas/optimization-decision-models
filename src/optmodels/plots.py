"""Restrained figures for the illustrative programmes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optmodels.constrained import marshallian_cobb_douglas
from optmodels.dynamic_program import cake_eating_dp
from optmodels.linear_program import textbook_resource_data
from optmodels.unconstrained import quadratic_objective
from optmodels.uncertainty import example_demand_scenarios, sensitivity_penalty


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_quadratic_contours(path: Path) -> Path:
    Q = np.array([[2.0, 0.4], [0.4, 1.0]])
    c = np.array([-1.0, 0.5])
    xs = np.linspace(-2.0, 2.0, 80)
    ys = np.linspace(-2.0, 2.0, 80)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = quadratic_objective([xx[i, j], yy[i, j]], Q, c)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cs = ax.contour(xx, yy, zz, levels=12, colors="0.25", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%1.1f")
    xstar = -np.linalg.solve(0.5 * (Q + Q.T), c)
    ax.plot(xstar[0], xstar[1], "k+", markersize=10, label="closed-form minimizer")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Convex quadratic level sets (illustrative Q, c)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_budget_set(path: Path) -> Path:
    alpha, p1, p2, m = 0.4, 2.0, 1.0, 10.0
    x = marshallian_cobb_douglas(alpha, p1, p2, m)
    x1 = np.linspace(0.05, m / p1, 80)
    x2 = (m - p1 * x1) / p2
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(x1, x2, "k-", label="budget p·x = m")
    ax.plot(x[0], x[1], "ko", label="Marshallian demand")
    ax.set_xlabel("good 1 (units)")
    ax.set_ylabel("good 2 (units)")
    ax.set_title("Cobb-Douglas interior demand on a linear budget")
    ax.legend(frameon=False)
    ax.set_xlim(0, m / p1 * 1.05)
    ax.set_ylim(0, m / p2 * 1.05)
    return _save(fig, path)


def plot_lp_feasible(path: Path) -> Path:
    _c, A, b = textbook_resource_data()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x1 = np.linspace(0.0, 6.0, 200)
    ax.plot(x1, np.full_like(x1, b[1] / A[1, 1]), "k--", linewidth=0.9, label="x2 resource")
    ax.axvline(b[0], color="0.3", linestyle=":", linewidth=0.9, label="x1 resource")
    ax.plot(x1, (b[2] - A[2, 0] * x1) / A[2, 1], "k-", linewidth=0.9, label="joint resource")
    ax.plot(2.0, 6.0, "ks", label="vertex optimum (2, 6)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 8)
    ax.set_xlabel("activity 1")
    ax.set_ylabel("activity 2")
    ax.set_title("Resource-allocation feasible set (illustrative)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_cake_value(path: Path) -> Path:
    result = cake_eating_dp(wealth_cap=8, n_periods=4, beta=0.9)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    wealth = np.arange(result.wealth_cap + 1)
    ax.plot(wealth, result.value[0], "k-o", markersize=4)
    ax.set_xlabel("initial cake (integer units)")
    ax.set_ylabel("value at t = 0")
    ax.set_title("Cake-eating Bellman value on the integer grid")
    return _save(fig, path)


def plot_penalty_sensitivity(path: Path) -> Path:
    demands, probs = example_demand_scenarios()
    penalties = np.linspace(1.0, 8.0, 12)
    grid, xs, _vals = sensitivity_penalty(demands, probs, unit_cost=1.0, holding=0.5, penalties=penalties)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(grid, xs, "k-o", markersize=4)
    ax.set_xlabel("shortage penalty")
    ax.set_ylabel("expected-cost production x*")
    ax.set_title("Newsvendor-type sensitivity (illustrative scenarios)")
    return _save(fig, path)
