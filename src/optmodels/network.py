"""Network models: Dijkstra shortest paths and transportation LPs.

Dijkstra is implemented here with a binary heap. There is no NetworkX
dependency. Uncapacitated bipartite min-cost flow is the transportation
problem and is solved as an LP.

Decision variables, parameters, and limitations: ``docs/model_catalogue.md``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog


@dataclass(frozen=True)
class ShortestPathResult:
    distances: np.ndarray
    predecessors: np.ndarray
    source: int


@dataclass(frozen=True)
class TransportationResult:
    flows: np.ndarray
    cost: float
    success: bool
    message: str


def textbook_dijkstra_graph() -> tuple[int, list[tuple[int, int, float]], int]:
    """Four-node directed graph with non-negative lengths.

    Edges: 0->1 (1), 0->2 (4), 1->2 (2), 1->3 (6), 2->3 (3).
    From source 0 the shortest-path distances are 0, 1, 3, 6.
    Unique shortest path 0 -> 3 is 0-1-2-3.
    """
    edges = [
        (0, 1, 1.0),
        (0, 2, 4.0),
        (1, 2, 2.0),
        (1, 3, 6.0),
        (2, 3, 3.0),
    ]
    return 4, edges, 0


def dijkstra(
    n: int,
    edges: list[tuple[int, int, float]],
    source: int,
    *,
    directed: bool = True,
) -> ShortestPathResult:
    """Single-source shortest paths on a graph with non-negative lengths.

    Decision variables: a walk from ``source`` (equivalently distance labels).
    Parameters: n nodes, list of (u, v, length) with length >= 0.
    Objective: minimise path length.
    Assumptions: no negative lengths; missing nodes are unreachable (inf).
    Solution method: Dijkstra with a binary heap.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= source < n):
        raise ValueError("source out of range")
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge endpoint out of range")
        if w < 0.0:
            raise ValueError("Dijkstra requires non-negative lengths")
        adj[u].append((v, float(w)))
        if not directed:
            adj[v].append((u, float(w)))

    dist = np.full(n, np.inf, dtype=float)
    pred = np.full(n, -1, dtype=int)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u] + 1e-15:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd + 1e-15 < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return ShortestPathResult(distances=dist, predecessors=pred, source=source)


def reconstruct_path(result: ShortestPathResult, target: int) -> list[int] | None:
    """Walk source -> target using predecessor labels; None if unreachable."""
    n = result.distances.size
    if not (0 <= target < n):
        raise ValueError("target out of range")
    if not np.isfinite(result.distances[target]):
        return None
    path = [target]
    v = target
    seen = {target}
    while v != result.source:
        v = int(result.predecessors[v])
        if v < 0 or v in seen:
            return None
        path.append(v)
        seen.add(v)
    path.reverse()
    return path


def path_length(path: list[int], edges: list[tuple[int, int, float]]) -> float:
    """Length of an explicit node list on a directed edge list."""
    weight = {(u, v): float(w) for u, v, w in edges}
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if (a, b) not in weight:
            raise ValueError(f"missing arc {(a, b)}")
        total += weight[(a, b)]
    return total


def example_transportation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two sources, three sinks, balanced supply and demand."""
    supply = np.array([20.0, 25.0])
    demand = np.array([15.0, 15.0, 15.0])
    cost = np.array(
        [
            [8.0, 6.0, 10.0],
            [9.0, 12.0, 7.0],
        ]
    )
    return supply, demand, cost


def solve_transportation(
    supply: ArrayLike,
    demand: ArrayLike,
    cost: ArrayLike,
    *,
    method: str = "highs",
) -> TransportationResult:
    """Balanced transportation LP.

    Decision variables: x_ij >= 0.
    Objective: minimise sum c_ij x_ij.
    Constraints: row sums = supply, column sums = demand.
    Assumptions: sum supply = sum demand; uncapacitated arcs.
    """
    s = np.asarray(supply, dtype=float).reshape(-1)
    d = np.asarray(demand, dtype=float).reshape(-1)
    c = np.asarray(cost, dtype=float)
    m, n = s.size, d.size
    if c.shape != (m, n):
        raise ValueError("cost must have shape (n_sources, n_sinks)")
    if s.min() < 0.0 or d.min() < 0.0:
        raise ValueError("supply and demand must be non-negative")
    if abs(float(s.sum()) - float(d.sum())) > 1e-8:
        raise ValueError("transportation problem is not balanced")

    cvec = c.reshape(-1)
    n_var = m * n
    A_eq = np.zeros((m + n, n_var))
    b_eq = np.zeros(m + n)
    for i in range(m):
        for j in range(n):
            A_eq[i, i * n + j] = 1.0
        b_eq[i] = s[i]
    for j in range(n):
        for i in range(m):
            A_eq[m + j, i * n + j] = 1.0
        b_eq[m + j] = d[j]

    res = linprog(cvec, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method=method)
    if res.x is None:
        return TransportationResult(
            flows=np.full((m, n), np.nan),
            cost=float("nan"),
            success=False,
            message=str(res.message),
        )
    flows = np.asarray(res.x, dtype=float).reshape(m, n)
    return TransportationResult(
        flows=flows,
        cost=float(np.sum(c * flows)),
        success=bool(res.success),
        message=str(res.message),
    )


def min_cost_bipartite_flow(
    supply: ArrayLike,
    demand: ArrayLike,
    cost: ArrayLike,
) -> TransportationResult:
    """Uncapacitated bipartite min-cost flow: the transportation problem."""
    return solve_transportation(supply, demand, cost)
