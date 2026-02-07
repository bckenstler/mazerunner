"""Color themes for maze rendering."""

import numpy as np

THEMES = {
    "light_classic": {
        "wall": (40, 40, 40),
        "corridor": (255, 255, 255),
        "chrome_bg": (232, 232, 232),
        "chrome_border": (200, 200, 200),
        "title_text": (60, 60, 60),
        "start_marker": (34, 197, 94),
        "goal_marker": (239, 68, 68),
        "traffic_close": (255, 95, 87),
        "traffic_minimize": (255, 189, 46),
        "traffic_maximize": (39, 201, 63),
    },
    "dark_modern": {
        "wall": (30, 30, 30),
        "corridor": (55, 65, 81),
        "chrome_bg": (17, 24, 39),
        "chrome_border": (55, 65, 81),
        "title_text": (209, 213, 219),
        "start_marker": (34, 197, 94),
        "goal_marker": (248, 113, 113),
        "traffic_close": (255, 95, 87),
        "traffic_minimize": (255, 189, 46),
        "traffic_maximize": (39, 201, 63),
    },
    "blueprint": {
        "wall": (200, 220, 255),
        "corridor": (20, 50, 120),
        "chrome_bg": (15, 40, 100),
        "chrome_border": (40, 80, 160),
        "title_text": (200, 220, 255),
        "start_marker": (0, 255, 120),
        "goal_marker": (255, 80, 80),
        "traffic_close": (255, 95, 87),
        "traffic_minimize": (255, 189, 46),
        "traffic_maximize": (39, 201, 63),
    },
    "parchment": {
        "wall": (101, 67, 33),
        "corridor": (245, 230, 200),
        "chrome_bg": (210, 190, 160),
        "chrome_border": (180, 160, 130),
        "title_text": (80, 50, 20),
        "start_marker": (34, 180, 80),
        "goal_marker": (200, 50, 50),
        "traffic_close": (255, 95, 87),
        "traffic_minimize": (255, 189, 46),
        "traffic_maximize": (39, 201, 63),
    },
}


def pick_theme(rng: np.random.Generator) -> str:
    """Pick a random theme name."""
    names = list(THEMES.keys())
    return names[int(rng.integers(0, len(names)))]


def get_theme(name: str) -> dict:
    """Get theme dict by name."""
    return THEMES[name]
