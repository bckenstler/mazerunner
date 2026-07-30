"""Canonical world model.

A World is the single navigable structure from which everything else derives:
the hidden mask, the styled render, the adjacency graph, the certified
reference route, and the evaluator artifacts (the doc's section 20 invariant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import solver as solver_mod
from .evaluator import check_path_collision, min_clearance
from .geometry import Point, densify_polyline, polyline_length

DEFAULT_POINTER_RADIUS = 3
DEFAULT_ENDPOINT_RADIUS = 28.0
MIN_ENDPOINT_RADIUS = 12.0
# Extra pixels of certified clearance beyond the pointer radius along the
# reference route (hardening fix 6).
CLEARANCE_MARGIN = 0.5


@dataclass
class Edge:
    a: int
    b: int
    geometry: list[Point]  # pixel-space polyline from node a to node b
    width_px: float


# Extra open primitives beyond corridors: ("disk", x, y, r) or
# ("rect", x0, y0, x1, y1), both in pixel space.
OpenPrim = tuple


@dataclass
class World:
    id: str
    type: str
    style: str
    state_representation: str  # "GRAPH" or "RASTER"
    width: int
    height: int
    seed: int
    generator: str
    solver: str
    nodes: dict[int, Point]
    edges: list[Edge]
    start_node: int
    goal_node: int
    node_pad_radius: dict[int, float] = field(default_factory=dict)
    node_rects: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    extra_open: list[OpenPrim] = field(default_factory=list)
    path_nodes: list[int] = field(default_factory=list)
    pointer_radius_px: int = DEFAULT_POINTER_RADIUS
    start_radius_px: float = DEFAULT_ENDPOINT_RADIUS
    goal_radius_px: float = DEFAULT_ENDPOINT_RADIUS
    # RASTER worlds (cave) derive adjacency directly from cell geometry, where
    # touching corridors ARE the connectivity; graph worlds must keep
    # non-adjacent corridors separated so the mask never contains shortcuts
    # the graph does not model.
    check_edge_separation: bool = True

    @property
    def start_px(self) -> Point:
        return self.nodes[self.start_node]

    @property
    def goal_px(self) -> Point:
        return self.nodes[self.goal_node]


def adjacency(world: World) -> solver_mod.Adjacency:
    adj: solver_mod.Adjacency = {n: [] for n in world.nodes}
    for e in world.edges:
        w = polyline_length(e.geometry)
        adj[e.a].append((e.b, w))
        adj[e.b].append((e.a, w))
    return adj


def _edge_lookup(world: World) -> dict[tuple[int, int], Edge]:
    lookup: dict[tuple[int, int], Edge] = {}
    for e in world.edges:
        lookup[(e.a, e.b)] = e
        lookup[(e.b, e.a)] = e
    return lookup


def reference_polyline(world: World, spacing: float = 2.0) -> list[Point]:
    """Densified continuous centerline along the retained route."""
    lookup = _edge_lookup(world)
    pts: list[Point] = []
    for a, b in zip(world.path_nodes, world.path_nodes[1:]):
        edge = lookup[(a, b)]
        geom = edge.geometry if edge.a == a else edge.geometry[::-1]
        segment = densify_polyline(geom, spacing)
        pts.extend(segment if not pts else segment[1:])
    return pts


def open_mask(world: World) -> np.ndarray:
    """Binary traversability mask rasterized without antialiasing.

    The renderer uses this exact array as its corridor stencil, so the visibly
    open region always equals the scored region (hardening fix 1).
    """
    img = Image.new("L", (world.width, world.height), 0)
    draw = ImageDraw.Draw(img)
    for e in world.edges:
        half = e.width_px / 2.0
        geom = densify_polyline(e.geometry, 4.0)
        for p, q in zip(geom, geom[1:]):
            draw.line([p, q], fill=255, width=max(1, round(e.width_px)))
        for x, y in geom:
            draw.ellipse([x - half, y - half, x + half, y + half], fill=255)
    for node, (x, y) in world.nodes.items():
        pad = world.node_pad_radius.get(node)
        if pad:
            draw.ellipse([x - pad, y - pad, x + pad, y + pad], fill=255)
    for rect in world.node_rects.values():
        draw.rectangle(list(rect), fill=255)
    for prim in world.extra_open:
        if prim[0] == "disk":
            _, x, y, r = prim
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        elif prim[0] == "rect":
            _, x0, y0, x1, y1 = prim
            draw.rectangle([x0, y0, x1, y1], fill=255)
        else:
            raise ValueError(f"unknown open primitive {prim[0]!r}")
    return np.asarray(img) > 127


def _disambiguated_radius(
    mask: np.ndarray, center: Point, default_radius: float
) -> float:
    """Largest acceptance radius whose in-disk mask pixels all connect locally
    to the endpoint center (hardening fix 3)."""
    h, w = mask.shape
    r = int(math.ceil(default_radius))
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    window = mask[y0:y1, x0:x1]
    labels, _n = ndimage.label(window)
    center_label = labels[cy - y0, cx - x0]
    if center_label == 0:
        raise ValueError("endpoint center is not on traversable mask")
    ys, xs = np.nonzero(window)
    dist = np.hypot(xs + x0 - center[0], ys + y0 - center[1])
    foreign = labels[ys, xs] != center_label
    in_disk_foreign = foreign & (dist <= default_radius)
    if not in_disk_foreign.any():
        return default_radius
    return max(0.0, float(dist[in_disk_foreign].min()) - 1.0)


EDGE_SEPARATION_GAP = 4.0


def _point_rect_distance(px: float, py: float, rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def check_edge_separation(world: World) -> None:
    """Non-adjacent corridors and foreign node areas must not touch or overlap.

    An unmodeled overlap would open a traversable shortcut that the graph (and
    therefore the certified reference) does not know about.
    """
    dense = [
        (e, np.array(densify_polyline(e.geometry, 3.0), dtype=float)) for e in world.edges
    ]
    for i, (e1, pts1) in enumerate(dense):
        for e2, pts2 in dense[i + 1 :]:
            if {e1.a, e1.b} & {e2.a, e2.b}:
                continue
            deltas = pts1[:, None, :] - pts2[None, :, :]
            min_d = float(np.sqrt((deltas**2).sum(axis=2)).min())
            required = (e1.width_px + e2.width_px) / 2 + EDGE_SEPARATION_GAP
            if min_d < required:
                raise ValueError(
                    f"{world.id}: corridors {e1.a}-{e1.b} and {e2.a}-{e2.b} pass "
                    f"{min_d:.1f}px apart (need {required:.1f})"
                )
    for e, pts in dense:
        for node in world.nodes:
            if node in (e.a, e.b):
                continue
            required = e.width_px / 2 + EDGE_SEPARATION_GAP
            pad = world.node_pad_radius.get(node)
            if pad:
                nx, ny = world.nodes[node]
                d = float(np.hypot(pts[:, 0] - nx, pts[:, 1] - ny).min()) - pad
                if d < required:
                    raise ValueError(
                        f"{world.id}: corridor {e.a}-{e.b} passes {d:.1f}px from "
                        f"node {node}'s open pad (need {required:.1f})"
                    )
            rect = world.node_rects.get(node)
            if rect:
                d = min(_point_rect_distance(x, y, rect) for x, y in pts)
                if d < required:
                    raise ValueError(
                        f"{world.id}: corridor {e.a}-{e.b} passes {d:.1f}px from "
                        f"node {node}'s room (need {required:.1f})"
                    )


@dataclass
class WorldValidation:
    reference_points_px: list[Point]
    reference_length_px: float
    reference_steps: int
    min_clearance_px: float
    start_radius_px: float
    goal_radius_px: float
    geodesic_length_px: float = 0.0
    geodesic_points_px: list[Point] = field(default_factory=list)


def validate_world(world: World, mask: np.ndarray) -> WorldValidation:
    """Run every fail-closed generation check; raise on any violation."""
    adj = adjacency(world)

    if not solver_mod.connected(adj):
        raise ValueError(f"{world.id}: graph is not connected")

    if not world.path_nodes:
        raise ValueError(f"{world.id}: no retained path")
    if world.path_nodes[0] != world.start_node or world.path_nodes[-1] != world.goal_node:
        raise ValueError(f"{world.id}: retained path does not join start to goal")

    # Edge validity + weighted-optimality certification (hardening fix 4).
    ref_len = solver_mod.certify_route(adj, world.path_nodes)

    if world.check_edge_separation:
        check_edge_separation(world)

    # The densified reference must pass the exact swept-disk scorer used for
    # model output.
    ref_pts = reference_polyline(world)
    collision_free, first = check_path_collision(mask, ref_pts, world.pointer_radius_px)
    if not collision_free:
        raise ValueError(f"{world.id}: reference route fails the scorer at {first}")

    # Min corridor clearance along the route (hardening fix 6).
    clearance = min_clearance(mask, ref_pts)
    if clearance < world.pointer_radius_px + CLEARANCE_MARGIN:
        raise ValueError(
            f"{world.id}: reference clearance {clearance:.2f}px is below "
            f"pointer radius {world.pointer_radius_px} + {CLEARANCE_MARGIN}"
        )

    # Mask-certified geometric optimum: the true shortest legal route through
    # the eroded mask. Certifies the graph route is never shorter than the
    # geometry allows, and becomes the efficiency denominator.
    from .geodesic import geodesic_optimum

    geo_len, geo_pts = geodesic_optimum(
        mask, world.start_px, world.goal_px, world.pointer_radius_px
    )
    if geo_len > ref_len + 1.0:
        raise ValueError(
            f"{world.id}: geodesic optimum {geo_len:.1f}px exceeds the graph "
            f"route {ref_len:.1f}px — mask and graph disagree"
        )

    # Endpoint disambiguation (hardening fix 3): shrink acceptance radii until
    # unambiguous, and fail closed if they collapse below the minimum.
    start_r = _disambiguated_radius(mask, world.start_px, world.start_radius_px)
    goal_r = _disambiguated_radius(mask, world.goal_px, world.goal_radius_px)
    for name, r in (("start", start_r), ("goal", goal_r)):
        if r < MIN_ENDPOINT_RADIUS:
            raise ValueError(
                f"{world.id}: {name} acceptance radius shrank to {r:.1f}px "
                f"(< {MIN_ENDPOINT_RADIUS}); another corridor passes too close "
                "to the marker"
            )

    return WorldValidation(
        reference_points_px=ref_pts,
        reference_length_px=ref_len,
        reference_steps=len(world.path_nodes) - 1,
        min_clearance_px=clearance,
        start_radius_px=start_r,
        goal_radius_px=goal_r,
        geodesic_length_px=geo_len,
        geodesic_points_px=geo_pts,
    )
