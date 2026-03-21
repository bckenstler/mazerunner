"""Predefined color schemas for maze rendering."""

from typing import Dict, List

import numpy as np

# Each schema defines colors for: wall, corridor, start, goal, solution_path, background
# Colors are hex strings for JSON serialization and renderer portability.
COLOR_SCHEMAS: List[Dict[str, str]] = [
    {
        "name": "classic",
        "wall": "#1a1a2e",
        "corridor": "#e8e8e8",
        "start": "#22c55e",
        "goal": "#ef4444",
        "solution_path": "#3b82f6",
        "background": "#f5f5f5",
    },
    {
        "name": "ocean",
        "wall": "#1e3a5f",
        "corridor": "#d4e6f1",
        "start": "#2ecc71",
        "goal": "#e74c3c",
        "solution_path": "#f39c12",
        "background": "#eaf2f8",
    },
    {
        "name": "forest",
        "wall": "#2d4a22",
        "corridor": "#e8f5e9",
        "start": "#ff9800",
        "goal": "#d32f2f",
        "solution_path": "#7c4dff",
        "background": "#f1f8e9",
    },
    {
        "name": "slate",
        "wall": "#37474f",
        "corridor": "#eceff1",
        "start": "#00c853",
        "goal": "#ff1744",
        "solution_path": "#00b0ff",
        "background": "#fafafa",
    },
    {
        "name": "midnight",
        "wall": "#0d0d0d",
        "corridor": "#2c2c3a",
        "start": "#00e676",
        "goal": "#ff5252",
        "solution_path": "#448aff",
        "background": "#1a1a2e",
    },
    {
        "name": "sandstone",
        "wall": "#5d4037",
        "corridor": "#fff3e0",
        "start": "#4caf50",
        "goal": "#f44336",
        "solution_path": "#2196f3",
        "background": "#fbe9e7",
    },
    {
        "name": "blueprint",
        "wall": "#1565c0",
        "corridor": "#e3f2fd",
        "start": "#ff6f00",
        "goal": "#d50000",
        "solution_path": "#00c853",
        "background": "#bbdefb",
    },
    {
        "name": "charcoal",
        "wall": "#263238",
        "corridor": "#cfd8dc",
        "start": "#76ff03",
        "goal": "#ff3d00",
        "solution_path": "#40c4ff",
        "background": "#eceff1",
    },
    {
        "name": "lavender",
        "wall": "#4a148c",
        "corridor": "#f3e5f5",
        "start": "#00e676",
        "goal": "#ff1744",
        "solution_path": "#ffab00",
        "background": "#ede7f6",
    },
    {
        "name": "ember",
        "wall": "#b71c1c",
        "corridor": "#fce4ec",
        "start": "#00c853",
        "goal": "#6200ea",
        "solution_path": "#ff6d00",
        "background": "#fff8e1",
    },
]

SCHEMA_NAMES = [s["name"] for s in COLOR_SCHEMAS]


def sample_color_schema(rng: np.random.Generator) -> Dict[str, str]:
    """Sample a random color schema from the predefined list."""
    idx = int(rng.integers(len(COLOR_SCHEMAS)))
    return COLOR_SCHEMAS[idx].copy()


def get_color_schema(name: str) -> Dict[str, str]:
    """Get a color schema by name."""
    for schema in COLOR_SCHEMAS:
        if schema["name"] == name:
            return schema.copy()
    raise ValueError(f"Unknown color schema: {name}. Available: {SCHEMA_NAMES}")
