"""Archetype base and shared painting utilities.

An archetype is a parameterized visual family. `sample(rng)` resolves every
random style decision into a JSON-serializable params dict (logged with the
task for reproducibility); `paint(arr, world, mask, params, rng)` renders it.
Painting must go through the mask stencil: outside-mask decor composites
through `~mask`, inside-mask decor through `mask`, and nothing else may touch
traversability. Every params dict must include:

- "corridor_fill": [r, g, b] — the corridor's base color, used by the
  fairness certifier to hunt for corridor-colored fakes outside the mask;
- "outline_color": [r, g, b] and "outline_px": int.

Every archetype paints in the same order, and the order is load-bearing:

1. fill the background;
2. background decor, composited through `~mask`;
3. `paint_outline` — the wall band around the corridor;
4. `arr[mask] = corridor_fill` — the corridor itself;
5. in-corridor decor, confined to `mask`.

Filling the corridor before drawing the outline would let the wall band eat
into the corridor: the render would still certify (the mask is unchanged) but
the visible corridor would be narrower than the scored one, which is precisely
the unfairness certification exists to prevent.

Any decor color that reads as traversable must be declared in
"corridor_extra", or the certifier will count it as a wall crossing the
corridor and reject the sample.
"""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

Color = tuple[int, int, int]


# --- color sampling ---


def hsv_color(rng, h: float, h_jitter: float, s: tuple[float, float], v: tuple[float, float]) -> list[int]:
    """Sample an RGB color (as a JSON-friendly list) around an HSV anchor."""
    hue = (h + rng.uniform(-h_jitter, h_jitter)) % 1.0
    sat = rng.uniform(*s)
    val = rng.uniform(*v)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return [round(r * 255), round(g * 255), round(b * 255)]


def shift(color, dv: int) -> list[int]:
    """Value-shift a color, clamped."""
    return [max(0, min(255, c + dv)) for c in color]


def as_tuple(color) -> Color:
    return tuple(int(c) for c in color)


# --- array painters ---


def fill(arr: np.ndarray, color) -> None:
    arr[:] = as_tuple(color)


def speckle(arr, rng, color, density: float, where=None) -> None:
    sel = rng.random(arr.shape[:2]) < density
    if where is not None:
        sel &= where
    arr[sel] = as_tuple(color)


def hstripes(arr, spacing: int, color, offset: int = 0, thickness: int = 1) -> None:
    for y in range(offset, arr.shape[0], spacing):
        arr[y : y + thickness, :] = as_tuple(color)


def vstripes(arr, spacing: int, color, offset: int = 0, thickness: int = 1) -> None:
    for x in range(offset, arr.shape[1], spacing):
        arr[:, x : x + thickness] = as_tuple(color)


def grid_lines(arr, spacing: int, color) -> None:
    hstripes(arr, spacing, color)
    vstripes(arr, spacing, color)


def checker(arr, cell: int, color_a, color_b, where) -> None:
    """Two-tone checkerboard, painted only where `where` is true — pass the
    mask (or its inverse) to keep the pattern on one side of the wall."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    parity = ((xx // cell) + (yy // cell)) % 2 == 0
    arr[where & parity] = as_tuple(color_a)
    arr[where & ~parity] = as_tuple(color_b)


def band_stripes(arr, rng, colors, band: tuple[int, int], where) -> None:
    """Horizontal strata of random thickness (canyon/glacier bands)."""
    y = 0
    h = arr.shape[0]
    i = 0
    while y < h:
        thickness = int(rng.integers(*band))
        arr[y : y + thickness][where[y : y + thickness]] = as_tuple(colors[i % len(colors)])
        y += thickness
        i += 1


# --- layered (RGBA) painters, composited through a region ---


def composite_layer(arr: np.ndarray, layer: Image.Image, region: np.ndarray) -> None:
    """Copy layer pixels into arr wherever the layer was drawn AND region holds."""
    layer_arr = np.asarray(layer)
    drawn = layer_arr[:, :, 3] > 0
    sel = drawn & region
    arr[sel] = layer_arr[sel][:, :3]


def layer_for(arr) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """A transparent RGBA scratch layer matching `arr`, for props that need
    real drawing (ellipses, strokes) before being composited through a mask."""
    h, w = arr.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return layer, ImageDraw.Draw(layer)


def dots(arr, rng, region, colors, count: int, radius: tuple[float, float]) -> None:
    """Scatter dots (foliage, spray, bubbles) composited into `region` only."""
    h, w = arr.shape[:2]
    layer, draw = layer_for(arr)
    for _ in range(count):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(*radius)
        color = as_tuple(colors[int(rng.integers(len(colors)))])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*color, 255))
    composite_layer(arr, layer, region)


def wall_band(mask: np.ndarray, depth: int) -> np.ndarray:
    """The wall region within `depth` px of the corridor — where wall props go."""
    return ndimage.binary_dilation(mask, iterations=depth) & ~mask


def pebbles(arr, rng, mask, depth: int, shades, outline, density: float = 1 / 90) -> None:
    """Scatter pebbles through the wall band `depth` pixels deep.

    `density` is a fraction of band pixels, not a count, so a style keeps its
    look across canvas sizes. Pebbles composite through `~mask`, so no prop can
    narrow the corridor.
    """
    band = wall_band(mask, depth)
    layer, draw = layer_for(arr)
    ys, xs = np.nonzero(band)
    order = rng.permutation(len(ys))[: max(1, int(len(ys) * density))]
    for i in order:
        x, y = xs[i] + rng.uniform(-3, 3), ys[i] + rng.uniform(-3, 3)
        rx, ry = rng.uniform(7, 12), rng.uniform(5, 9)
        shade = as_tuple(shades[int(rng.integers(len(shades)))])
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(*shade, 255), outline=(*as_tuple(outline), 255), width=2)
    composite_layer(arr, layer, ~mask)


def edge_dashes(arr, world, mask, color, dash: float = 6.0, gap_every: int = 2, width: int = 2) -> None:
    """Dashed centerline along every edge, clipped to the corridor."""
    from ..geometry import densify_polyline

    layer, draw = layer_for(arr)
    for e in world.edges:
        pts = densify_polyline(e.geometry, dash)
        for i in range(0, len(pts) - 1, gap_every):
            draw.line([pts[i], pts[i + 1]], fill=(*as_tuple(color), 255), width=width)
    composite_layer(arr, layer, mask)


def edge_stones(arr, world, mask, fill_color, rim_color, spacing: float = 13.0, rx: float = 4, ry: float = 3) -> None:
    """Stepping-stone pearls along each route, clipped to the corridor."""
    from ..geometry import densify_polyline

    layer, draw = layer_for(arr)
    for e in world.edges:
        for x, y in densify_polyline(e.geometry, spacing):
            draw.ellipse(
                [x - rx, y - ry, x + rx, y + ry],
                fill=(*as_tuple(fill_color), 255),
                outline=(*as_tuple(rim_color), 255),
            )
    composite_layer(arr, layer, mask)


# --- archetype contract ---


class Archetype:
    """The style plugin interface. Subclass, set `name`, implement sample+paint.

    `supports` narrows the state representations a style claims to handle;
    a style that assumes designed corridors should drop "RASTER".
    """

    name: str = ""
    supports = frozenset({"GRAPH", "RASTER"})

    def sample(self, rng: np.random.Generator) -> dict:
        """Resolve every random style decision into a JSON-serializable params
        dict. Deterministic given `rng`, and stored with the task — a render
        that cannot be reproduced from its params is not reproducible."""
        raise NotImplementedError

    def paint(self, arr: np.ndarray, world, mask: np.ndarray, params: dict, rng: np.random.Generator) -> None:
        """Paint the render in place, in the order given in the module
        docstring. Must never change what is traversable: the mask is the
        stencil, not a suggestion."""
        raise NotImplementedError

    def paint_outline(self, arr, mask, params) -> None:
        """The wall band hugging the corridor. Call before filling the
        corridor, or the band will eat into it."""
        outline = ndimage.binary_dilation(mask, iterations=int(params["outline_px"])) & ~mask
        arr[outline] = as_tuple(params["outline_color"])
