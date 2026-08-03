"""What image the model actually receives.

Scoring never touches these bytes — submissions are normalized coordinates
evaluated against the task's true mask — so substituting the image changes only
what the model can perceive, which is exactly what the blind (H4) and
input-resolution ablations need.

Modes:
  real        the task's own render (the main run)
  blank       a neutral canvas at the task's true dimensions
  mismatched  a *different* task's render, tier-matched
  rescale     the task's render resized by `scale`
"""

from __future__ import annotations

import io as _io
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

# Mid-grey: no hue that could bias a guess toward any archetype's palette.
BLANK_FILL = (128, 128, 128)
RESAMPLE = {"lanczos": Image.LANCZOS, "nearest": Image.NEAREST, "bicubic": Image.BICUBIC}


@dataclass(frozen=True)
class ImageSpec:
    """Which image a run actually sends, for the perception ablations.

    "real" is the benchmark; "blank" and "mismatched" are the controls that
    establish the score floor, and "rescale" resends the same maze at another
    resolution. Validated on construction, so an impossible combination
    (rescale at 1.0x) fails at configuration rather than mid-run.
    """

    mode: str = "real"
    scale: float = 1.0
    resample: str = "lanczos"
    seed: int | None = None

    def __post_init__(self):
        if self.mode not in ("real", "blank", "mismatched", "rescale"):
            raise ValueError(f"unknown image mode {self.mode!r}")
        if self.mode == "rescale" and self.scale == 1.0:
            raise ValueError("rescale mode needs a scale != 1.0")
        if self.resample not in RESAMPLE:
            raise ValueError(f"unknown resample filter {self.resample!r}")

    @property
    def label(self) -> str:
        return f"{self.mode}@{self.scale:g}x" if self.mode == "rescale" else self.mode


def blank_png(width: int, height: int, fill: tuple[int, int, int] = BLANK_FILL) -> bytes:
    """A featureless canvas at the task's true size.

    Keeping the true dimensions holds aspect ratio constant, so the blind
    condition isolates image *content* rather than also removing shape cues.
    """
    buffer = _io.BytesIO()
    Image.new("RGB", (width, height), fill).save(buffer, format="PNG")
    return buffer.getvalue()


def rescale_png(png_bytes: bytes, scale: float, resample: str = "lanczos") -> bytes:
    """Resize the render. Mask and ground truth are untouched by construction."""
    image = Image.open(_io.BytesIO(png_bytes))
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    out = image.resize((width, height), RESAMPLE[resample])
    buffer = _io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


def tier_matched_derangement(
    task_ids: list[str],
    tiers: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """Map each task to a *different* task's image, matched on difficulty tier.

    Fixed-point-free within each tier, so no task ever receives its own image
    and the wrong image carries comparable difficulty priors — otherwise a model
    shown an easy maze in place of a hard one could score above the true floor
    for reasons that have nothing to do with vision.

    A tier holding a single task cannot be deranged within itself; those tasks
    fall back to the global pool (still never themselves).
    """
    rng = np.random.default_rng(seed)
    mapping: dict[str, str] = {}
    by_tier: dict[str, list[str]] = {}
    for task_id in task_ids:
        by_tier.setdefault(tiers[task_id], []).append(task_id)

    for tier, members in sorted(by_tier.items()):
        if len(members) < 2:
            for task_id in members:
                others = [t for t in task_ids if t != task_id]
                mapping[task_id] = others[int(rng.integers(len(others)))]
            continue
        # Rotate by a random non-zero offset: guaranteed fixed-point-free.
        order = list(members)
        rng.shuffle(order)
        offset = int(rng.integers(1, len(order)))
        for i, task_id in enumerate(order):
            mapping[task_id] = order[(i + offset) % len(order)]
    return mapping


class ImageSource:
    """Resolves the bytes for one attempt, plus provenance for the record."""

    def __init__(self, spec: ImageSpec, sources: list[tuple[str, Path]] | None = None,
                 tiers: dict[str, str] | None = None):
        self.spec = spec
        self._dirs = dict(sources or [])
        self._mapping: dict[str, str] = {}
        if spec.mode == "mismatched":
            if not sources:
                raise ValueError("mismatched mode needs the task pool")
            task_ids = [name for name, _dir in sources]
            resolved_tiers = tiers or {t: "unknown" for t in task_ids}
            self._mapping = tier_matched_derangement(
                task_ids, resolved_tiers, spec.seed if spec.seed is not None else 0
            )

    def bytes_for(self, task_id: str, task_dir: Path, task: dict) -> tuple[bytes, dict]:
        spec = self.spec
        provenance = {"mode": spec.mode, "label": spec.label}

        if spec.mode == "blank":
            return blank_png(task["width"], task["height"]), {
                **provenance, "width": task["width"], "height": task["height"],
            }

        if spec.mode == "mismatched":
            other = self._mapping[task_id]
            other_dir = self._dirs[other]
            other_task_png = other_dir / "input.png"
            return other_task_png.read_bytes(), {
                **provenance, "source_task": other, "seed": spec.seed,
            }

        raw = (task_dir / task["image_file"]).read_bytes()
        if spec.mode == "rescale":
            scaled = rescale_png(raw, spec.scale, spec.resample)
            image = Image.open(_io.BytesIO(scaled))
            return scaled, {
                **provenance, "scale": spec.scale, "resample": spec.resample,
                "sent_width": image.width, "sent_height": image.height,
                "true_width": task["width"], "true_height": task["height"],
            }
        return raw, provenance


def spec_from_args(variant: str, scale: float, seed: int | None) -> ImageSpec:
    """CLI flags -> ImageSpec, with rescale inferred from a non-unit scale."""
    if variant == "real" and scale != 1.0:
        variant = "rescale"
    return ImageSpec(mode=variant, scale=scale, seed=seed)
