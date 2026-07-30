"""Twelve additional archetypes for the dataset's visual-diversity engine.

Same rules as classic.py: all painting goes through the mask stencil, every
sampled value lands in the params dict, corridor-like colors are declared via
corridor_extra so the fairness certifier can hunt for fakes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry import densify_polyline
from . import base
from .base import Archetype, as_tuple


def _outer_band(mask, inner: int, outer: int) -> np.ndarray:
    """Wall region between `inner` and `outer` px from the corridor — decor
    placed here can never touch the boundary ring."""
    return ndimage.binary_dilation(mask, iterations=outer) & ~ndimage.binary_dilation(
        mask, iterations=inner
    )


class CircuitBoard(Archetype):
    name = "circuit-board"

    def sample(self, rng):
        copper = base.hsv_color(rng, 0.09, 0.02, (0.55, 0.75), (0.7, 0.85))
        return {
            "bg": base.hsv_color(rng, 0.36, 0.03, (0.5, 0.75), (0.1, 0.18)),
            "grid_spacing": int(rng.integers(24, 40)),
            "corridor_fill": copper,
            "corridor_extra": [base.shift(copper, 25)],
            "pad": [232, 226, 210],
            "outline_color": base.hsv_color(rng, 0.36, 0.03, (0.5, 0.75), (0.04, 0.08)),
            "outline_px": 2,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.grid_lines(arr, params["grid_spacing"], base.shift(params["bg"], 10))
        base.speckle(arr, rng, base.shift(params["bg"], 16), 0.01)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.edge_dashes(arr, world, mask, base.shift(params["corridor_fill"], 25), dash=8, gap_every=1, width=2)
        # Pads at every junction of a dense raster world would blanket the
        # corridor with high-contrast rings; the certifier rejects that.
        if len(world.nodes) <= 80:
            layer, draw = base.layer_for(arr)
            for x, y in world.nodes.values():
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(*as_tuple(params["pad"]), 255), width=3)
            base.composite_layer(arr, layer, mask)


class HedgeGarden(Archetype):
    name = "hedge-garden"

    def sample(self, rng):
        hedge_h = rng.uniform(0.3, 0.38)
        gravel = base.hsv_color(rng, 0.1, 0.03, (0.08, 0.16), (0.75, 0.88))
        return {
            "grass": base.hsv_color(rng, hedge_h, 0.02, (0.45, 0.6), (0.5, 0.64)),
            "hedge": [base.hsv_color(rng, hedge_h, 0.03, (0.55, 0.75), v) for v in ((0.2, 0.28), (0.26, 0.36), (0.14, 0.2))],
            "corridor_fill": gravel,
            "corridor_extra": [base.shift(gravel, -12)],
            "outline_color": base.hsv_color(rng, hedge_h, 0.02, (0.55, 0.75), (0.1, 0.15)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["grass"])
        base.speckle(arr, rng, base.shift(params["grass"], -14), 0.08)
        base.dots(arr, rng, base.wall_band(mask, 14), params["hedge"], 2600, (3, 8))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, base.shift(params["corridor_fill"], -12), 0.12, where=mask)


class MetroMap(Archetype):
    name = "metro-map"

    def sample(self, rng):
        hues = [float(h) for h in rng.choice([0.0, 0.08, 0.15, 0.33, 0.55, 0.62, 0.78], size=4, replace=False)]
        lines = [base.hsv_color(rng, h, 0.02, (0.7, 0.9), (0.65, 0.85)) for h in hues]
        return {
            "bg": base.hsv_color(rng, 0.13, 0.04, (0.02, 0.06), (0.93, 0.98)),
            "grid_spacing": int(rng.integers(36, 56)),
            "corridor_fill": base.hsv_color(rng, 0.13, 0.03, (0.02, 0.06), (0.8, 0.88)),
            "lines": lines,
            "corridor_extra_from": ["lines"],
            "station": [252, 252, 252],
            "station_ring": [40, 42, 48],
            "outline_color": base.hsv_color(rng, 0.6, 0.05, (0.1, 0.25), (0.25, 0.4)),
            "outline_px": 2,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.grid_lines(arr, params["grid_spacing"], base.shift(params["bg"], -8))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        layer, draw = base.layer_for(arr)
        for i, e in enumerate(sorted(world.edges, key=lambda e: (e.a, e.b))):
            color = as_tuple(params["lines"][i % len(params["lines"])])
            geom = densify_polyline(e.geometry, 4.0)
            for p, q in zip(geom, geom[1:]):
                draw.line([p, q], fill=(*color, 255), width=max(4, round(e.width_px) - 8))
        if len(world.nodes) <= 80:
            ring = as_tuple(params["station_ring"])
            station = as_tuple(params["station"])
            for x, y in world.nodes.values():
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(*station, 255), outline=(*ring, 255), width=2)
        base.composite_layer(arr, layer, mask)


class IceGlacier(Archetype):
    name = "ice-glacier"

    def sample(self, rng):
        ice = base.hsv_color(rng, 0.55, 0.03, (0.08, 0.18), (0.88, 0.97))
        return {
            "bg": base.hsv_color(rng, 0.58, 0.03, (0.5, 0.7), (0.25, 0.38)),
            "corridor_fill": ice,
            "corridor_extra": [base.shift(ice, -14)],
            "crack": [235, 244, 250],
            "outline_color": base.hsv_color(rng, 0.58, 0.03, (0.55, 0.75), (0.12, 0.2)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.speckle(arr, rng, base.shift(params["bg"], 12), 0.05)
        # Cracks live deep in the wall, never touching the boundary ring.
        band = _outer_band(mask, 6, 22)
        layer, draw = base.layer_for(arr)
        ys, xs = np.nonzero(band)
        for i in rng.permutation(len(ys))[: len(ys) // 600]:
            x, y = float(xs[i]), float(ys[i])
            angle = rng.uniform(0, np.pi)
            length = rng.uniform(6, 16)
            dx, dy = np.cos(angle) * length, np.sin(angle) * length
            draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(*as_tuple(params["crack"]), 255), width=1)
        base.composite_layer(arr, layer, band)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, base.shift(params["corridor_fill"], -14), 0.07, where=mask)


class TreasureMap(Archetype):
    name = "treasure-map"

    def sample(self, rng):
        aged = base.hsv_color(rng, 0.1, 0.02, (0.25, 0.38), (0.6, 0.72))
        trail = base.hsv_color(rng, 0.1, 0.02, (0.12, 0.2), (0.9, 0.97))
        return {
            "bg": aged,
            "stain": base.shift(aged, -22),
            "corridor_fill": trail,
            "corridor_extra": [base.shift(trail, -10)],
            "route_dash": base.hsv_color(rng, 0.99, 0.02, (0.5, 0.7), (0.45, 0.6)),
            "x_mark": base.hsv_color(rng, 0.99, 0.02, (0.6, 0.8), (0.35, 0.5)),
            "outline_color": base.hsv_color(rng, 0.08, 0.02, (0.4, 0.6), (0.2, 0.3)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.dots(arr, rng, ~mask, [params["stain"], base.shift(params["bg"], 12)], 260, (4, 14))
        band = _outer_band(mask, 6, 26)
        layer, draw = base.layer_for(arr)
        ys, xs = np.nonzero(band)
        for i in rng.permutation(len(ys))[: max(1, len(ys) // 4000)]:
            x, y = float(xs[i]), float(ys[i])
            c = as_tuple(params["x_mark"])
            draw.line([(x - 5, y - 5), (x + 5, y + 5)], fill=(*c, 255), width=3)
            draw.line([(x - 5, y + 5), (x + 5, y - 5)], fill=(*c, 255), width=3)
        base.composite_layer(arr, layer, band)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, base.shift(params["corridor_fill"], -10), 0.06, where=mask)
        base.edge_dashes(arr, world, mask, params["route_dash"], dash=7, gap_every=2, width=2)


class PaperCutout(Archetype):
    name = "paper-cutout"

    def sample(self, rng):
        card_h = rng.uniform(0, 1)
        return {
            "bg": base.hsv_color(rng, card_h, 0.03, (0.35, 0.55), (0.6, 0.78)),
            "corridor_fill": base.hsv_color(rng, card_h, 0.02, (0.0, 0.05), (0.95, 1.0)),
            "shadow_shift": int(rng.integers(3, 6)),
            "outline_color": base.hsv_color(rng, card_h, 0.03, (0.4, 0.6), (0.25, 0.4)),
            "outline_px": 2,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        # Drop shadow: the corridor's cutout shape offset down-right.
        s = params["shadow_shift"]
        shadow = np.zeros_like(mask)
        shadow[s:, s:] = mask[:-s, :-s]
        arr[shadow & ~mask] = as_tuple(base.shift(params["bg"], -40))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])


class Volcanic(Archetype):
    name = "volcanic"

    def sample(self, rng):
        stone = base.hsv_color(rng, 0.05, 0.02, (0.04, 0.1), (0.42, 0.55))
        return {
            "bg": base.hsv_color(rng, 0.03, 0.02, (0.2, 0.4), (0.07, 0.12)),
            "ember": base.hsv_color(rng, 0.05, 0.02, (0.85, 1.0), (0.8, 0.95)),
            "corridor_fill": stone,
            "corridor_extra": [base.shift(stone, -12)],
            "outline_color": base.hsv_color(rng, 0.04, 0.02, (0.9, 1.0), (0.55, 0.75)),
            "outline_px": 2,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.speckle(arr, rng, base.shift(params["bg"], 10), 0.1)
        base.dots(arr, rng, _outer_band(mask, 5, 30), [params["ember"], base.shift(params["ember"], -60)], 320, (1, 3))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, base.shift(params["corridor_fill"], -12), 0.12, where=mask)


class CandyPastel(Archetype):
    name = "candy-pastel"

    def sample(self, rng):
        bg_h = rng.choice([0.95, 0.85, 0.6, 0.13])
        mint = base.hsv_color(rng, 0.42, 0.03, (0.06, 0.14), (0.95, 1.0))
        return {
            "bg": base.hsv_color(rng, float(bg_h), 0.02, (0.18, 0.3), (0.85, 0.95)),
            "stripe_spacing": int(rng.integers(18, 30)),
            "corridor_fill": mint,
            "sprinkles": [base.hsv_color(rng, float(h), 0.02, (0.6, 0.85), (0.75, 0.92)) for h in (0.0, 0.13, 0.55, 0.8)],
            "outline_color": base.hsv_color(rng, 0.07, 0.02, (0.55, 0.75), (0.25, 0.38)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.hstripes(arr, params["stripe_spacing"], base.shift(params["bg"], -14), thickness=6)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.dots(arr, rng, mask, params["sprinkles"], 240, (1, 2))


class MosaicTile(Archetype):
    name = "mosaic-tile"

    def sample(self, rng):
        tile_h = rng.uniform(0.5, 0.62)
        floor = base.hsv_color(rng, 0.13, 0.03, (0.1, 0.2), (0.8, 0.92))
        return {
            "bg_a": base.hsv_color(rng, tile_h, 0.02, (0.5, 0.7), (0.3, 0.42)),
            "bg_b": base.hsv_color(rng, tile_h, 0.02, (0.5, 0.7), (0.22, 0.3)),
            "tile": int(rng.integers(8, 14)),
            "corridor_fill": floor,
            "corridor_extra": [base.shift(floor, -10)],
            "outline_color": base.hsv_color(rng, tile_h, 0.02, (0.4, 0.6), (0.1, 0.16)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.checker(arr, params["tile"], params["bg_a"], params["bg_b"], np.ones(arr.shape[:2], dtype=bool))
        self.paint_outline(arr, mask, params)
        base.checker(arr, params["tile"], params["corridor_fill"], base.shift(params["corridor_fill"], -10), mask)


class PencilSketch(Archetype):
    name = "pencil-sketch"

    def sample(self, rng):
        paper = base.hsv_color(rng, 0.12, 0.03, (0.02, 0.06), (0.93, 0.98))
        return {
            "paper": paper,
            "corridor_fill": paper,
            "hatch": base.hsv_color(rng, 0.6, 0.05, (0.02, 0.08), (0.5, 0.65)),
            "hatch_spacing": int(rng.integers(5, 9)),
            "outline_color": base.hsv_color(rng, 0.62, 0.04, (0.05, 0.12), (0.18, 0.3)),
            "outline_px": int(rng.integers(3, 5)),
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["paper"])
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((xx + yy) % params["hatch_spacing"]) == 0
        arr[hatch & ~mask] = as_tuple(params["hatch"])
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])


class DesertCanyon(Archetype):
    name = "desert-canyon"

    def sample(self, rng):
        sand = base.hsv_color(rng, 0.09, 0.02, (0.14, 0.24), (0.88, 0.96))
        strata_h = rng.uniform(0.03, 0.08)
        return {
            "strata": [base.hsv_color(rng, strata_h, 0.015, (0.45, 0.7), v) for v in ((0.3, 0.4), (0.42, 0.52), (0.5, 0.62), (0.36, 0.46))],
            "band": [int(rng.integers(8, 14)), int(rng.integers(16, 30))],
            "corridor_fill": sand,
            "corridor_extra": [base.shift(sand, -12)],
            "outline_color": base.hsv_color(rng, strata_h, 0.02, (0.5, 0.7), (0.16, 0.24)),
            "outline_px": 3,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["strata"][0])
        base.band_stripes(arr, rng, params["strata"], tuple(params["band"]), np.ones(arr.shape[:2], dtype=bool))
        base.speckle(arr, rng, base.shift(params["strata"][1], -16), 0.03)
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, base.shift(params["corridor_fill"], -12), 0.09, where=mask)


class BiolabVessels(Archetype):
    name = "biolab-vessels"

    def sample(self, rng):
        tissue_h = rng.uniform(0.85, 0.95)
        membrane = base.hsv_color(rng, tissue_h, 0.02, (0.06, 0.14), (0.92, 0.98))
        return {
            "bg": base.hsv_color(rng, tissue_h, 0.03, (0.35, 0.55), (0.25, 0.36)),
            "organelles": [base.hsv_color(rng, tissue_h, 0.05, (0.4, 0.6), v) for v in ((0.32, 0.42), (0.18, 0.26), (0.42, 0.52))],
            "corridor_fill": membrane,
            "corridor_extra": [base.shift(membrane, -12)],
            "capillary": base.shift(membrane, -28),
            "outline_color": base.hsv_color(rng, tissue_h, 0.03, (0.5, 0.7), (0.14, 0.22)),
            "outline_px": 4,
        }

    def paint(self, arr, world, mask, params, rng):
        base.fill(arr, params["bg"])
        base.dots(arr, rng, ~mask, params["organelles"], 900, (3, 11))
        self.paint_outline(arr, mask, params)
        arr[mask] = as_tuple(params["corridor_fill"])
        base.speckle(arr, rng, base.shift(params["corridor_fill"], -12), 0.06, where=mask)
        base.edge_dashes(arr, world, mask, params["capillary"], dash=9, gap_every=2, width=1)


EXPANDED = [
    CircuitBoard(),
    HedgeGarden(),
    MetroMap(),
    IceGlacier(),
    TreasureMap(),
    PaperCutout(),
    Volcanic(),
    CandyPastel(),
    MosaicTile(),
    PencilSketch(),
    DesertCanyon(),
    BiolabVessels(),
]
