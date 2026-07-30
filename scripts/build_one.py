"""Dev helper: build a single family end-to-end and save its artifacts."""

import sys
from pathlib import Path

from mazerunner.io import save_task
from mazerunner.overlay import render_overlay
from mazerunner.render import render_world
from mazerunner.world import open_mask, validate_world

from importlib import import_module

family = sys.argv[1] if len(sys.argv) > 1 else "rectilinear"
module = import_module(f"mazerunner.generators.{family}")

world = module.build()
mask = open_mask(world)
validation = validate_world(world, mask)
print(
    f"{world.id}: nodes={len(world.nodes)} edges={len(world.edges)} "
    f"steps={validation.reference_steps} ref_px={validation.reference_length_px:.1f} "
    f"clearance={validation.min_clearance_px:.2f} "
    f"start_r={validation.start_radius_px:.1f} goal_r={validation.goal_radius_px:.1f}"
)
image = render_world(world, mask)
task_dir = Path("mazes") / world.id
save_task(task_dir, world, mask, image, validation)
overlay = render_overlay(image, validation.reference_points_px, success=True)
overlay.save(task_dir / "reference-overlay.png")
print(f"saved {task_dir}/")
