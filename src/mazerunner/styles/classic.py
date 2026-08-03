"""The eight original smoke styles, reworked as parameterized archetypes.

Every constant that used to be hard-coded is now sampled in `sample()` and
recorded in the params dict, so two style seeds of the same archetype look
related but distinct.
"""

from __future__ import annotations

import numpy as np
from PIL import ImageFont

from ..geometry import densify_polyline
from . import base
from .base import Archetype, as_tuple


class Notebook(Archetype):
    """Ruled notebook paper: the corridor is the bare page, walls are ink."""

    name = "notebook"

    def sample(self, rng):
        return {
            "paper": base.hsv_color(rng, 0.12, 0.04, (0.05, 0.12), (0.93, 0.98)),
            "rule": base.hsv_color(rng, 0.58, 0.06, (0.15, 0.35), (0.75, 0.92)),
            "rule_spacing": int(rng.integers(22, 32)),
            "margin": base.hsv_color(rng, 0.0, 0.03, (0.3, 0.5), (0.75, 0.9)),
            "corridor_fill": base.hsv_color(rng, 0.12, 0.03, (0.0, 0.04), (0.97, 1.0)),
            "outline_color": base.hsv_color(rng, rng.choice([0.6, 0.62, 0.55, 0.08]), 0.03, (0.4, 0.7), (0.2, 0.42)),
            "outline_px": int(rng.integers(3, 6)),
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["paper"])
        base.hstripes(arr, params["rule_spacing"], params["rule"], offset=12)
        base.vstripes(arr, arr.shape[1] * 2, params["margin"], offset=64, thickness=2)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])


class DungeonPebble(Archetype):
    """Dark stone dungeon: speckled floor, pebbles scattered in the wall band.

    The floor speckle is declared in corridor_extra so certification reads it
    as corridor rather than a wall crossing it.
    """

    name = "dungeon-pebble"

    def sample(self, rng):
        stone_h = rng.uniform(0.07, 0.13)
        floor = base.hsv_color(rng, stone_h, 0.02, (0.05, 0.12), (0.22, 0.3))
        return {
            "bg": base.hsv_color(rng, stone_h, 0.02, (0.05, 0.12), (0.1, 0.16)),
            "corridor_fill": floor,
            "corridor_noise": base.shift(floor, -8),
            "corridor_extra": [base.shift(floor, -8)],
            "pebble_shades": [
                base.hsv_color(rng, stone_h, 0.03, (0.06, 0.16), (0.45, 0.62)) for _ in range(4)
            ],
            "pebble_outline": base.hsv_color(rng, stone_h, 0.02, (0.05, 0.12), (0.06, 0.1)),
            "outline_color": base.hsv_color(rng, stone_h, 0.02, (0.05, 0.12), (0.05, 0.09)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.speckle(arr, rng, base.shift(params["bg"], 8), 0.18)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, params["corridor_noise"], 0.15, where=mask)
        base.pebbles(arr, rng, mask, 16, params["pebble_shades"], params["pebble_outline"])


class BlueprintRooms(Archetype):
    """Architectural blueprint: light corridor on a gridded blue ground.

    Room labels composite through `mask`, so text can never sit on a wall and
    suggest open space where there is none.
    """

    name = "blueprint-rooms"

    def sample(self, rng):
        bg_h = rng.uniform(0.55, 0.63)
        return {
            "bg": base.hsv_color(rng, bg_h, 0.02, (0.6, 0.85), (0.24, 0.34)),
            "grid": None,  # filled in paint from bg
            "grid_spacing": int(rng.integers(32, 52)),
            "corridor_fill": base.hsv_color(rng, bg_h, 0.02, (0.02, 0.08), (0.9, 0.97)),
            "dash": base.hsv_color(rng, bg_h, 0.02, (0.3, 0.5), (0.55, 0.72)),
            "label": base.hsv_color(rng, bg_h, 0.02, (0.5, 0.7), (0.3, 0.5)),
            "outline_color": base.hsv_color(rng, bg_h, 0.02, (0.3, 0.5), (0.65, 0.88)),
            "outline_px": 2,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.grid_lines(arr, params["grid_spacing"], base.shift(params["bg"], 14))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.edge_dashes(arr, world, mask, params["dash"])
        if world.node_rects:
            font = ImageFont.load_default(size=17)
            layer, draw = base.layer_for(arr)
            for node in sorted(world.node_rects):
                if node in (world.start_node, world.goal_node):
                    continue
                x, y = world.nodes[node]
                draw.text((x, y), f"R{node + 1}", fill=(*as_tuple(params["label"]), 255), font=font, anchor="mm")
            base.composite_layer(arr, layer, mask)


class ForestPath(Archetype):
    """Woodland trail: dirt path through dense canopy."""

    name = "forest-path"

    def sample(self, rng):
        green_h = rng.uniform(0.28, 0.38)
        path = base.hsv_color(rng, 0.1, 0.03, (0.22, 0.34), (0.8, 0.92))
        return {
            "bg": base.hsv_color(rng, green_h, 0.02, (0.45, 0.65), (0.16, 0.24)),
            "foliage": [base.hsv_color(rng, green_h, 0.04, (0.4, 0.7), v) for v in ((0.2, 0.28), (0.12, 0.18), (0.3, 0.42), (0.34, 0.5))],
            "foliage_count": int(rng.integers(1800, 3200)),
            "corridor_fill": path,
            "corridor_noise": base.shift(path, -16),
            "corridor_light": base.shift(path, 14),
            "corridor_extra": [base.shift(path, -16), base.shift(path, 14)],
            "outline_color": base.hsv_color(rng, green_h, 0.02, (0.4, 0.6), (0.28, 0.4)),
            "outline_px": 4,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.dots(arr, rng, np.ones(arr.shape[:2], dtype=bool), params["foliage"], params["foliage_count"], (2, 7))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, params["corridor_noise"], 0.14, where=mask)
        base.speckle(arr, rng, params["corridor_light"], 0.06, where=mask)


class GlowCavern(Archetype):
    """Luminous cave: glowing floor against near-black rock."""

    name = "glow-cavern"

    def sample(self, rng):
        glow_h = rng.choice([0.48, 0.52, 0.36, 0.78])
        floor_a = base.hsv_color(rng, 0.28, 0.05, (0.06, 0.14), (0.42, 0.52))
        return {
            "bg": base.hsv_color(rng, 0.62, 0.05, (0.3, 0.55), (0.05, 0.09)),
            "floor_a": floor_a,
            "floor_b": base.shift(floor_a, -10),
            "corridor_fill": floor_a,
            "corridor_extra": [base.shift(floor_a, -10)],
            "sparkle": base.hsv_color(rng, float(glow_h), 0.02, (0.7, 0.9), (0.75, 0.92)),
            "tile": int(rng.integers(16, 26)),
            "outline_color": base.hsv_color(rng, 0.45, 0.04, (0.2, 0.4), (0.14, 0.22)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.speckle(arr, rng, base.shift(params["bg"], 10), 0.06)
        layer, draw = base.layer_for(arr)
        h, w = arr.shape[:2]
        for _ in range(90):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            s = rng.uniform(2.5, 5.5)
            draw.polygon(
                [(x, y - s), (x - s * 0.7, y + s * 0.6), (x + s * 0.7, y + s * 0.6)],
                fill=(*as_tuple(params["sparkle"]), 255),
            )
        base.composite_layer(arr, layer, ~mask)
        self.paint_outline(arr, mask, params)
        base.checker(arr, params["tile"], params["floor_a"], params["floor_b"], mask)
        base.speckle(arr, rng, params["sparkle"], 0.004, where=mask)


class ParchmentChart(Archetype):
    """Aged parchment chart, corridor as the drawn route."""

    name = "parchment-chart"

    def sample(self, rng):
        parchment = base.hsv_color(rng, 0.11, 0.02, (0.14, 0.24), (0.88, 0.96))
        return {
            "bg": base.hsv_color(rng, rng.uniform(0.6, 0.72), 0.02, (0.4, 0.6), (0.08, 0.14)),
            "star": [235, 232, 244],
            "star_faint": base.hsv_color(rng, 0.68, 0.04, (0.25, 0.4), (0.35, 0.55)),
            "corridor_fill": parchment,
            "corridor_noise": base.shift(parchment, -12),
            "corridor_extra": [base.shift(parchment, -12)],
            "outline_color": base.hsv_color(rng, rng.choice([0.75, 0.08, 0.62]), 0.03, (0.4, 0.6), (0.22, 0.38)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.speckle(arr, rng, params["star"], 0.0015)
        base.speckle(arr, rng, params["star_faint"], 0.005)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, params["corridor_noise"], 0.08, where=mask)


class WatercolorArchipelago(Archetype):
    """Watercolor islands: sand corridors in a washed sea.

    Two sand shades are declared in corridor_extra; without both, the wash
    variation reads as walls inside the corridor.
    """

    name = "watercolor-archipelago"

    def sample(self, rng):
        sea_h = rng.uniform(0.45, 0.58)
        sea = base.hsv_color(rng, sea_h, 0.02, (0.5, 0.68), (0.6, 0.74))
        sand = base.hsv_color(rng, 0.11, 0.02, (0.18, 0.28), (0.88, 0.96))
        return {
            "sea": sea,
            "wave_light": base.shift(sea, 14),
            "wave_dark": base.shift(sea, -12),
            "wave_spacing": int(rng.integers(13, 20)),
            "corridor_fill": sand,
            "corridor_noise": base.shift(sand, -12),
            "stone": base.shift(sand, -18),
            "stone_rim": base.shift(sand, -46),
            "corridor_extra": [base.shift(sand, -12), base.shift(sand, -18)],
            "outline_color": base.hsv_color(rng, sea_h, 0.02, (0.5, 0.7), (0.4, 0.56)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["sea"])
        base.hstripes(arr, params["wave_spacing"], params["wave_light"], offset=5)
        base.hstripes(arr, params["wave_spacing"], params["wave_dark"], offset=13)
        base.speckle(arr, rng, base.shift(params["sea"], 20), 0.04)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, params["corridor_noise"], 0.1, where=mask)
        base.edge_stones(arr, world, mask, params["stone"], params["stone_rim"])


class NeonPipes(Archetype):
    """Neon industrial pipework, corridor as the lit pipe run.

    Uses corridor_extra_from so the whole sampled palette of pipe bodies and
    cores counts as corridor — the brightest style in the set, and the one
    most able to look traversable where it is not.
    """

    name = "neon-pipes"

    def sample(self, rng):
        picked = [float(h) for h in rng.choice([0.5, 0.58, 0.68, 0.83, 0.9, 0.38], size=5, replace=False)]
        hues, outline_hue = picked[:4], picked[4]
        return {
            "bg": base.hsv_color(rng, 0.66, 0.05, (0.3, 0.5), (0.06, 0.1)),
            "grid_spacing": int(rng.integers(42, 60)),
            "corridor_fill": base.hsv_color(rng, 0.66, 0.04, (0.12, 0.22), (0.2, 0.28)),
            "pipe_bodies": [base.hsv_color(rng, h, 0.02, (0.6, 0.8), (0.75, 0.9)) for h in hues],
            "pipe_cores": [base.hsv_color(rng, h, 0.02, (0.25, 0.4), (0.92, 1.0)) for h in hues],
            "joint": [228, 231, 240],
            "corridor_extra_from": ["pipe_bodies", "pipe_cores"],
            "outline_color": base.hsv_color(rng, outline_hue, 0.01, (0.75, 0.95), (0.85, 1.0)),
            "outline_px": 2,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.grid_lines(arr, params["grid_spacing"], base.shift(params["bg"], 12))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        layer, draw = base.layer_for(arr)
        bodies, cores = params["pipe_bodies"], params["pipe_cores"]
        for i, e in enumerate(sorted(world.edges, key=lambda e: (e.a, e.b))):
            body, core = bodies[i % len(bodies)], cores[i % len(cores)]
            geom = densify_polyline(e.geometry, 4.0)
            for p, q in zip(geom, geom[1:]):
                draw.line([p, q], fill=(*as_tuple(body), 255), width=max(4, round(e.width_px) - 6))
            for p, q in zip(geom, geom[1:]):
                draw.line([p, q], fill=(*as_tuple(core), 255), width=3)
        joint = as_tuple(params["joint"])
        dark = as_tuple(params["corridor_fill"])
        for x, y in world.nodes.values():
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(*dark, 255), outline=(*joint, 255), width=2)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(*joint, 255))
        base.composite_layer(arr, layer, mask)


CLASSIC = [
    Notebook(),
    DungeonPebble(),
    BlueprintRooms(),
    ForestPath(),
    GlowCavern(),
    ParchmentChart(),
    WatercolorArchipelago(),
    NeonPipes(),
]
