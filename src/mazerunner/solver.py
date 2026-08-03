"""Reference solvers and route certification.

Adjacency maps node id -> list of (neighbor id, pixel-length weight). All
retained routes are certified with pixel-weighted Dijkstra so that the stored
reference length is a geometric optimum over the graph's exact edge geometry,
not just a step count. A step-count optimum would let a route with few long
hops certify as "shortest" while a visibly shorter one existed.
"""

from __future__ import annotations

import heapq
from collections import deque

Adjacency = dict[int, list[tuple[int, float]]]


def bfs_path(adj: Adjacency, start: int, goal: int) -> list[int] | None:
    """Unweighted shortest path in edge steps."""
    prev: dict[int, int] = {start: start}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == goal:
            break
        for nbr, _w in adj[node]:
            if nbr not in prev:
                prev[nbr] = node
                q.append(nbr)
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def dijkstra(adj: Adjacency, start: int) -> tuple[dict[int, float], dict[int, int]]:
    """Pixel-weighted shortest distances and predecessors from `start`."""
    dist: dict[int, float] = {start: 0.0}
    prev: dict[int, int] = {}
    pq: list[tuple[float, int]] = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        for nbr, w in adj[node]:
            nd = d + w
            if nd < dist.get(nbr, float("inf")) - 1e-12:
                dist[nbr] = nd
                prev[nbr] = node
                heapq.heappush(pq, (nd, nbr))
    return dist, prev


def dijkstra_path(adj: Adjacency, start: int, goal: int) -> tuple[list[int] | None, float]:
    """(path, pixel length), or (None, inf) when the goal is unreachable."""
    dist, prev = dijkstra(adj, start)
    if goal not in dist:
        return None, float("inf")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1], dist[goal]


def farthest_node(adj: Adjacency, start: int) -> int:
    """Farthest node by pixel distance; used with a double sweep for endpoints."""
    dist, _ = dijkstra(adj, start)
    return max(dist, key=lambda n: dist[n])


def route_length(adj: Adjacency, path: list[int]) -> float:
    """Pixel length of a node path. Raises if any consecutive pair is not an
    edge, so an invented shortcut can never be measured as a real route."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        weights = [w for nbr, w in adj[a] if nbr == b]
        if not weights:
            raise ValueError(f"consecutive path nodes {a}->{b} are not a graph edge")
        total += min(weights)
    return total


def certify_route(adj: Adjacency, path: list[int]) -> float:
    """Fail closed unless `path` is edge-valid and Dijkstra-optimal.

    Returns the certified pixel length.
    """
    length = route_length(adj, path)  # raises if any hop is not an edge
    _opt_path, opt_len = dijkstra_path(adj, path[0], path[-1])
    if abs(length - opt_len) > 1e-6:
        raise ValueError(
            f"retained route length {length:.3f}px is not the weighted optimum "
            f"{opt_len:.3f}px"
        )
    return length


def connected(adj: Adjacency) -> bool:
    """Whether every node is reachable. An empty graph is not connected."""
    if not adj:
        return False
    start = next(iter(adj))
    seen = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for nbr, _w in adj[node]:
            if nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return len(seen) == len(adj)
