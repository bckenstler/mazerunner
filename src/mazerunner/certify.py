"""Pixel-level fairness certification of rendered tasks.

Construction (mask-as-stencil) guarantees visually-open == traversable, but
style sampling could still produce renders that are unfair to the model:
walls that don't contrast with the corridor, decor outside the mask that
reads as extra corridor width, or decor inside the corridor that reads as a
wall. Certification checks the final pixels and fails closed — a failing
style sample is rejected and resampled, never hand-fixed.

The checks (thresholds recorded with every task):

1. boundary  — along the corridor boundary, the pixels just outside the mask
   must contrast with the corridor palette; every point of the true corridor
   is visibly delimited.
2. extension — no corridor-colored region outside the mask near the corridor
   boundary; the corridor never *looks* wider or longer than it is.
   (Route-suggestive decor such as dashes/pearls is additionally confined to
   the mask by construction in styles/base.py.)
3. interior  — inside the mask, strongly contrasting decor is bounded; a
   texture may dress a corridor but never read as a wall across it.
4. markers   — start/goal badges contrast with their local surroundings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class Thresholds:
    """Tuned constants for the four checks, serialized with every task so a
    later reader can tell which certification a task actually passed."""

    boundary_similar_tau: float = 32.0  # RGB distance considered "corridor-like"
    boundary_max_similar_fraction: float = 0.03
    extension_band_px: int = 18  # how far outside the mask we hunt for fakes
    extension_min_area_px: int = 60  # smallest boundary-touching blob that counts
    interior_contrast_tau: float = 110.0  # distance considered "wall-like" inside
    interior_max_contrast_fraction: float = 0.15
    marker_min_contrast: float = 30.0
    marker_exclusion_px: int = 22


@dataclass
class Certification:
    """One render's verdict, stored in the task's provenance. `failures` names
    every check that tripped, not just the first, so a rejected style sample
    can be diagnosed without re-running it."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)


def corridor_palette(style_record: dict) -> list[list[int]]:
    """Colors that read as traversable for this style.

    corridor_fill always; archetypes list additional fills/texture colors
    under "corridor_extra" (see styles/*).
    """
    params = style_record["params"]
    palette = [params["corridor_fill"]]
    palette.extend(params.get("corridor_extra", []))
    for key in params.get("corridor_extra_from", []):
        palette.extend(params[key])
    return palette


def _distance_to_palette(pixels: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    """Min Euclidean RGB distance from each pixel to the palette."""
    best = None
    flat = pixels.astype(np.int32)
    for color in palette:
        d = np.sqrt(((flat - np.array(color, dtype=np.int32)) ** 2).sum(axis=-1))
        best = d if best is None else np.minimum(best, d)
    return best


def _marker_exclusion(shape, world, radius: int) -> np.ndarray:
    excl = np.zeros(shape, dtype=bool)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y in (world.start_px, world.goal_px):
        excl |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return excl


def certify_render(
    image: Image.Image,
    mask: np.ndarray,
    world,
    style_record: dict,
    thresholds: Thresholds = Thresholds(),
) -> Certification:
    """Run the four fairness checks against the final pixels.

    Construction already guarantees the mask is the render's stencil; these
    checks catch what construction cannot — a style whose sampled colors make
    the walls invisible, extend the apparent corridor, or dress the corridor
    in something that reads as a wall.

    Reports every failure rather than stopping at the first, and never
    modifies the render: a failing sample is resampled by `certified_render`,
    never repaired.
    """
    arr = np.asarray(image.convert("RGB"))
    palette = corridor_palette(style_record)
    dist = _distance_to_palette(arr, palette)
    excl = _marker_exclusion(mask.shape, world, thresholds.marker_exclusion_px)

    failures: list[str] = []
    metrics: dict = {}

    # 1. Boundary legibility: the ring just outside the corridor must not be
    # corridor-colored, or the wall is invisible there.
    outer_ring = (ndimage.binary_dilation(mask, iterations=2) & ~mask) & ~excl
    ring_similar = (dist < thresholds.boundary_similar_tau) & outer_ring
    ring_fraction = ring_similar.sum() / max(1, outer_ring.sum())
    metrics["boundary_similar_fraction"] = round(float(ring_fraction), 5)
    if ring_fraction > thresholds.boundary_max_similar_fraction:
        failures.append(
            f"boundary: {ring_fraction:.1%} of the corridor boundary is "
            "corridor-colored (invisible wall)"
        )

    # 2. No visual extensions: a corridor-colored blob outside the mask is
    # deceptive when it TOUCHES the boundary ring — it reads as a gap in the
    # wall or as extra corridor width. Look-alike territory that sits fully
    # beyond a visible wall (e.g. paper beyond an ink line) is legible and
    # allowed; check 1 already guarantees the wall is visible.
    near_band = (
        ndimage.binary_dilation(mask, iterations=thresholds.extension_band_px) & ~mask & ~excl
    )
    fake = (dist < thresholds.boundary_similar_tau) & near_band
    labels, count = ndimage.label(fake)
    max_area = 0
    if count:
        touching = np.unique(labels[ring_similar | (outer_ring & fake)])
        touching = touching[touching > 0]
        if touching.size:
            areas = ndimage.sum_labels(np.ones_like(labels), labels, index=touching)
            max_area = int(np.max(areas))
    metrics["largest_fake_corridor_px"] = max_area
    if max_area > thresholds.extension_min_area_px:
        failures.append(
            f"extension: corridor-colored region of {max_area}px touches the "
            "corridor boundary and suggests traversable space that is not"
        )

    # 3. Interior restraint: decor inside the corridor must not read as wall.
    interior = mask & ~excl
    contrasty = (dist > thresholds.interior_contrast_tau) & interior
    interior_fraction = contrasty.sum() / max(1, interior.sum())
    metrics["interior_contrast_fraction"] = round(float(interior_fraction), 5)
    if interior_fraction > thresholds.interior_max_contrast_fraction:
        failures.append(
            f"interior: {interior_fraction:.1%} of corridor pixels contrast "
            "strongly with the corridor fill (reads as walls)"
        )

    # 4. Marker visibility: a badge is visible if either its disk or its halo
    # ring contrasts with the local surroundings (the white halo is the
    # designed rescue when the badge fill matches the scene).
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    marker = style_record.get("marker", {})
    marker_r = marker.get("radius", 13)
    halo = marker.get("halo", 3)
    for label, (mx, my) in (("start", world.start_px), ("goal", world.goal_px)):
        d2 = (xx - mx) ** 2 + (yy - my) ** 2
        disk = d2 <= marker_r**2
        halo_ring = (d2 > marker_r**2) & (d2 <= (marker_r + halo) ** 2)
        annulus = (d2 > (marker_r + halo + 2) ** 2) & (d2 <= (marker_r + halo + 10) ** 2)
        surround = arr[annulus].mean(axis=0).astype(np.float64)
        disk_contrast = float(np.abs(arr[disk].mean(axis=0) - surround).mean())
        halo_contrast = float(np.abs(arr[halo_ring].mean(axis=0) - surround).mean())
        contrast = max(disk_contrast, halo_contrast)
        metrics[f"{label}_marker_contrast"] = round(contrast, 2)
        if contrast < thresholds.marker_min_contrast:
            failures.append(f"markers: {label} badge contrast {contrast:.1f} below threshold")

    return Certification(
        ok=not failures,
        failures=failures,
        metrics=metrics,
        thresholds=asdict(thresholds),
    )


MAX_STYLE_RESAMPLES = 6


def certified_render(
    world,
    mask: np.ndarray,
    archetype,
    style_seed: int,
    thresholds: Thresholds = Thresholds(),
):
    """Render with fail-closed certification, resampling the style seed.

    Returns (image, style_record, certification, rejections). Raises if no
    style sample passes within MAX_STYLE_RESAMPLES attempts.
    """
    from .render import render_world

    rejections: list[dict] = []
    for attempt in range(MAX_STYLE_RESAMPLES):
        seed = style_seed + attempt * 1_000_003
        image, record = render_world(world, mask, archetype, seed)
        certification = certify_render(image, mask, world, record, thresholds)
        if certification.ok:
            record["style_rejections"] = rejections
            return image, record, certification, rejections
        rejections.append({"style_seed": seed, "failures": certification.failures})
    raise ValueError(
        f"{world.id}/{getattr(archetype, 'name', archetype)}: no style sample "
        f"passed certification in {MAX_STYLE_RESAMPLES} attempts: {rejections}"
    )
