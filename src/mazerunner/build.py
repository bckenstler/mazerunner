"""Asset building and offline validation for the full smoke set."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .evaluator import evaluate_task
from .generators import FAMILIES
from .io import file_sha256, load_task, mask_sha256, save_task
from .overlay import render_overlay
from .render import render_world
from .world import open_mask, validate_world

MAZES_DIR = Path("mazes")


def build_family(name: str):
    module = FAMILIES[name]
    world = module.build()
    mask = open_mask(world)
    validation = validate_world(world, mask)
    image, style_record = render_world(world, mask)
    return world, mask, validation, image, style_record


def _sheet(base_dir: Path, file_name: str, out_name: str, scale: float = 0.42) -> Path:
    thumbs = []
    font = ImageFont.load_default(size=16)
    for name in FAMILIES:
        img = Image.open(base_dir / name / file_name).convert("RGB")
        w, h = round(img.width * scale), round(img.height * scale)
        thumb = img.resize((w, h), Image.LANCZOS)
        framed = Image.new("RGB", (w + 8, h + 34), (24, 24, 28))
        framed.paste(thumb, (4, 4))
        draw = ImageDraw.Draw(framed)
        draw.text((8, h + 10), name, fill=(230, 230, 235), font=font)
        thumbs.append(framed)
    cols = 4
    tw, th = thumbs[0].size
    sheet = Image.new("RGB", (cols * tw, 2 * th), (24, 24, 28))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
    out = base_dir / out_name
    sheet.save(out)
    return out


def build_all(base_dir: Path = MAZES_DIR) -> dict:
    """Generate every family, write task artifacts, sheets, and the manifest."""
    entries = []
    for name in FAMILIES:
        world, mask, validation, image, style_record = build_family(name)
        task_dir = base_dir / name
        save_task(task_dir, world, mask, image, validation, style_record)
        overlay = render_overlay(image, validation.geodesic_points_px, success=True)
        overlay.save(task_dir / "reference-overlay.png")
        entries.append(
            {
                "id": name,
                "type": world.type,
                "style": world.style,
                "seed": world.seed,
                "nodes": len(world.nodes),
                "edges": len(world.edges),
                "optimal_length_steps": validation.reference_steps,
                "optimal_length_px": round(validation.reference_length_px, 2),
                "mask_sha256": mask_sha256(mask),
                "input_sha256": file_sha256(task_dir / "input.png"),
            }
        )
    _sheet(base_dir, "input.png", "contact-sheet.png")
    _sheet(base_dir, "reference-overlay.png", "reference-sheet.png")
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            pkg: metadata.version(pkg) for pkg in ("numpy", "pillow", "scipy")
        },
        "tasks": entries,
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_all(base_dir: Path = MAZES_DIR) -> list[str]:
    """Offline validation. Returns a list of failure strings (empty = green).

    - Rebuilds every world twice and demands byte-identical masks (fix 5).
    - If a committed manifest exists, demands the rebuilt mask hashes match it.
    - Runs every fail-closed generation check (inside validate_world).
    - Scores the *saved* reference route of every task through the exact
      evaluator used for model output.
    """
    failures: list[str] = []

    old_manifest = None
    manifest_path = base_dir / "manifest.json"
    if manifest_path.exists():
        old_manifest = {t["id"]: t for t in json.loads(manifest_path.read_text())["tasks"]}

    for name in FAMILIES:
        try:
            world, mask, validation, image, style_record = build_family(name)
        except ValueError as exc:
            failures.append(f"{name}: generation failed closed: {exc}")
            continue

        rebuilt = open_mask(FAMILIES[name].build())
        if mask_sha256(rebuilt) != mask_sha256(mask):
            failures.append(f"{name}: mask is not deterministic across rebuilds")
        if old_manifest and name in old_manifest:
            if old_manifest[name]["mask_sha256"] != mask_sha256(mask):
                failures.append(
                    f"{name}: rebuilt mask differs from committed manifest hash"
                )
    if failures:
        return failures

    # Rewrite artifacts, then score every saved reference through the scorer.
    build_all(base_dir)
    for name in FAMILIES:
        task, mask = load_task(base_dir / name)
        submission = {"points": task["reference"]["optimal_path"]}
        result = evaluate_task(task, mask, submission)
        if not result.success:
            failures.append(
                f"{name}: saved reference route fails the scorer: {result.to_dict()}"
            )
        elif result.efficiency_canary:
            failures.append(
                f"{name}: efficiency canary fired on the reference itself: "
                f"raw={result.efficiency_raw:.3f}"
            )
    return failures
