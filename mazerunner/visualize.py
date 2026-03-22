"""CLI entry point for batch maze rendering."""

import argparse
import glob
import os
import sys

from mazerunner.renderer.base import (
    DragRenderConfig,
    GridRenderConfig,
    load_instance,
)
from mazerunner.renderer.text_grid import render_text_grid
from mazerunner.renderer.vision_drag import render_vision_drag
from mazerunner.renderer.vision_grid import render_vision_grid

MODES = ["vision_drag", "vision_grid", "text_grid", "all"]


def _auto_scale_config(instance: dict) -> tuple[DragRenderConfig, GridRenderConfig]:
    """Auto-scale render configs based on grid size."""
    rows = instance["grid_rows"]
    cols = instance["grid_cols"]
    max_dim = max(rows, cols)

    if max_dim <= 12:
        wt, cw = 6, 28
    elif max_dim <= 22:
        wt, cw = 5, 22
    else:
        wt, cw = 4, 16

    drag_cfg = DragRenderConfig(wall_thickness=wt, corridor_width=cw)
    grid_cfg = GridRenderConfig(cell_size=cw, wall_thickness=wt)
    return drag_cfg, grid_cfg


def render_batch(input_dir: str, output_dir: str, mode: str) -> None:
    """Discover maze JSONs and render them."""
    instances_dir = os.path.join(input_dir, "instances")
    json_files = sorted(glob.glob(os.path.join(instances_dir, "*.json")))

    if not json_files:
        print(f"No JSON files found in {instances_dir}", file=sys.stderr)
        sys.exit(1)

    modes = ["vision_drag", "vision_grid", "text_grid"] if mode == "all" else [mode]

    for m in modes:
        out = os.path.join(output_dir, m)
        os.makedirs(out, exist_ok=True)

    for json_path in json_files:
        instance = load_instance(json_path)
        basename = os.path.splitext(os.path.basename(json_path))[0]
        drag_cfg, grid_cfg = _auto_scale_config(instance)

        for m in modes:
            if m == "text_grid":
                text = render_text_grid(instance)
                out_path = os.path.join(output_dir, m, f"{basename}.txt")
                with open(out_path, "w") as f:
                    f.write(text)
            elif m == "vision_drag":
                img = render_vision_drag(instance, config=drag_cfg)
                out_path = os.path.join(output_dir, m, f"{basename}.png")
                img.save(out_path)
            elif m == "vision_grid":
                img = render_vision_grid(instance, config=grid_cfg)
                out_path = os.path.join(output_dir, m, f"{basename}.png")
                img.save(out_path)

    print(f"Rendered {len(json_files)} mazes ({', '.join(modes)}) to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch-render maze instances")
    parser.add_argument("--input-dir", required=True, help="Directory containing instances/ folder")
    parser.add_argument("--output-dir", required=True, help="Output directory for renderings")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="all",
        help="Rendering mode (default: all)",
    )

    args = parser.parse_args()
    render_batch(args.input_dir, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
