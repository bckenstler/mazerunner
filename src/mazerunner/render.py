"""Rendering: compose an archetype's paint over the scored mask, add markers.

The mask is the stencil for everything an archetype paints, so the visibly
open region always equals the traversable region by construction (hardening
fix 1). The returned style record contains the archetype name, style seed,
and every resolved sampled parameter — logged with the task so any render is
byte-reproducible from its record.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .styles import ARCHETYPES
from .styles.base import Archetype
from .world import World

Color = tuple[int, int, int]

START_COLOR: Color = (0, 188, 212)  # cyan badge
GOAL_COLOR: Color = (255, 179, 0)  # amber badge


def _draw_badge(draw: ImageDraw.ImageDraw, x: float, y: float, color: Color, r: int, halo: int) -> None:
    draw.ellipse([x - r - halo, y - r - halo, x + r + halo, y + r + halo], fill=(250, 250, 250))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(40, 40, 45), width=2)


def _draw_start(draw: ImageDraw.ImageDraw, x: float, y: float, r: int, halo: int) -> None:
    _draw_badge(draw, x, y, START_COLOR, r, halo)
    s = r - 7
    draw.polygon([(x - s + 1, y - s - 1), (x + s + 2, y), (x - s + 1, y + s + 1)], fill=(255, 255, 255))


def _draw_goal(draw: ImageDraw.ImageDraw, x: float, y: float, r: int, halo: int) -> None:
    _draw_badge(draw, x, y, GOAL_COLOR, r, halo)
    s = r - 7
    draw.rectangle([x - s - 1, y - s + 1, x + s + 1, y + s], fill=(255, 251, 240), outline=(109, 76, 0), width=1)
    draw.line([(x - s - 1, y), (x + s + 1, y)], fill=(109, 76, 0), width=2)
    draw.rectangle([x - 1, y - 1, x + 1, y + 2], fill=(109, 76, 0))


def render_world(
    world: World,
    mask: np.ndarray,
    archetype: Archetype | str | None = None,
    style_seed: int | None = None,
) -> tuple[Image.Image, dict]:
    """Render `world` through `mask` with the given archetype and style seed.

    Defaults preserve the smoke set: the family's classic archetype with the
    topology seed doubling as the style seed.
    """
    from .styles import CLASSIC_FOR_FAMILY

    if archetype is None:
        archetype = CLASSIC_FOR_FAMILY[world.id]
    if isinstance(archetype, str):
        archetype = ARCHETYPES[archetype]
    if style_seed is None:
        style_seed = world.seed
    if world.state_representation not in archetype.supports:
        raise ValueError(f"{archetype.name} does not support {world.state_representation} worlds")

    rng = np.random.default_rng(np.random.SeedSequence([style_seed, 0xC0FFEE]))
    params = archetype.sample(rng)

    arr = np.zeros((world.height, world.width, 3), dtype=np.uint8)
    archetype.paint(arr, world, mask, params, rng)

    marker = {"radius": int(rng.integers(12, 15)), "halo": int(rng.integers(2, 4))}
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    _draw_start(draw, *world.start_px, marker["radius"], marker["halo"])
    _draw_goal(draw, *world.goal_px, marker["radius"], marker["halo"])

    record = {
        "archetype": archetype.name,
        "style_seed": int(style_seed),
        "marker": marker,
        "params": params,
    }
    return img, record
