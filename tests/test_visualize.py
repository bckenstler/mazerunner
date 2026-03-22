"""Integration tests for the visualize CLI batch rendering."""

import json
import os
import tempfile

import pytest

from mazerunner.visualize import render_batch


def _make_simple_instance_dict(rows: int, cols: int, maze_id: str) -> dict:
    """Build a fully-connected maze instance dict."""
    adjacency = {}
    for r in range(rows):
        for c in range(cols):
            adjacency[f"{r},{c}"] = []

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                adjacency[f"{r},{c}"].append(f"{r},{c + 1}")
                adjacency[f"{r},{c + 1}"].append(f"{r},{c}")
            if r + 1 < rows:
                adjacency[f"{r},{c}"].append(f"{r + 1},{c}")
                adjacency[f"{r + 1},{c}"].append(f"{r},{c}")

    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))

    return {
        "id": maze_id,
        "grid_rows": rows,
        "grid_cols": cols,
        "start": "0,0",
        "goal": f"{rows - 1},{cols - 1}",
        "adjacency": adjacency,
        "shortest_path_cells": ["0,0", f"{rows - 1},{cols - 1}"],
        "metadata": {
            "color_schema": {
                "name": "classic",
                "wall": "#1a1a2e",
                "corridor": "#e8e8e8",
                "start": "#22c55e",
                "goal": "#ef4444",
                "solution_path": "#3b82f6",
                "background": "#f5f5f5",
            }
        },
    }


@pytest.fixture
def sample_input_dir(tmp_path):
    """Create a temporary input directory with a few maze JSON files."""
    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    for i in range(3):
        instance = _make_simple_instance_dict(5, 7, f"maze_{i:06d}")
        path = instances_dir / f"maze_{i:06d}.json"
        path.write_text(json.dumps(instance))
    return str(tmp_path)


class TestVisualizeBatch:
    def test_vision_drag_produces_pngs(self, sample_input_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        render_batch(sample_input_dir, output_dir, "vision_drag")
        drag_dir = os.path.join(output_dir, "vision_drag")
        files = sorted(os.listdir(drag_dir))
        assert len(files) == 3
        assert all(f.endswith(".png") for f in files)

    def test_text_grid_produces_txts(self, sample_input_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        render_batch(sample_input_dir, output_dir, "text_grid")
        text_dir = os.path.join(output_dir, "text_grid")
        files = sorted(os.listdir(text_dir))
        assert len(files) == 3
        assert all(f.endswith(".txt") for f in files)

    def test_vision_grid_produces_pngs(self, sample_input_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        render_batch(sample_input_dir, output_dir, "vision_grid")
        grid_dir = os.path.join(output_dir, "vision_grid")
        files = sorted(os.listdir(grid_dir))
        assert len(files) == 3
        assert all(f.endswith(".png") for f in files)

    def test_mode_all_produces_three_subdirs(self, sample_input_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        render_batch(sample_input_dir, output_dir, "all")
        subdirs = sorted(os.listdir(output_dir))
        assert "vision_drag" in subdirs
        assert "vision_grid" in subdirs
        assert "text_grid" in subdirs

    def test_filenames_match_input_ids(self, sample_input_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        render_batch(sample_input_dir, output_dir, "vision_drag")
        drag_dir = os.path.join(output_dir, "vision_drag")
        files = sorted(os.listdir(drag_dir))
        assert files == ["maze_000000.png", "maze_000001.png", "maze_000002.png"]

    def test_text_files_contain_maze(self, sample_input_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        render_batch(sample_input_dir, output_dir, "text_grid")
        text_dir = os.path.join(output_dir, "text_grid")
        content = open(os.path.join(text_dir, "maze_000000.txt")).read()
        assert " S " in content
        assert " G " in content
