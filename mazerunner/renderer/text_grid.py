"""ASCII text rendering of maze instances."""

from typing import Any, Dict

from mazerunner.renderer.base import get_color_schema, get_opening_side, has_wall, parse_cell


def render_text_grid(instance: Dict[str, Any]) -> str:
    """Render a maze instance as an ASCII text grid.

    Output format (2*rows+1 lines, 4*cols+1 chars per line)::

        +---+---+---+
        | S |   |   |
        +   +---+   +
        |       | G |
        +---+---+---+

    Edge start/goal cells get border openings (wall replaced with space).

    Args:
        instance: Maze instance dict (as loaded from JSON).

    Returns:
        Multi-line ASCII string representing the maze.
    """
    rows = instance["grid_rows"]
    cols = instance["grid_cols"]
    adjacency = instance["adjacency"]
    start = parse_cell(instance["start"])
    goal = parse_cell(instance["goal"])

    # Collect the single opening side for start and goal
    openings: dict[tuple[int, int], str] = {}
    for cell in [start, goal]:
        side = get_opening_side(cell[0], cell[1], rows, cols)
        if side is not None:
            openings[cell] = side

    lines = []

    for text_row in range(2 * rows + 1):
        line = []
        if text_row % 2 == 0:
            # Wall row
            grid_row_above = text_row // 2 - 1
            grid_row_below = text_row // 2
            for c in range(cols):
                line.append("+")
                # Horizontal wall between (grid_row_above, c) and (grid_row_below, c)
                if grid_row_above < 0 or grid_row_below >= rows:
                    # Border wall — check for opening
                    if grid_row_above < 0 and openings.get((0, c)) == "top":
                        line.append("   ")
                    elif grid_row_below >= rows and openings.get((rows - 1, c)) == "bottom":
                        line.append("   ")
                    else:
                        line.append("---")
                elif has_wall(adjacency, (grid_row_above, c), (grid_row_below, c)):
                    line.append("---")
                else:
                    line.append("   ")
            line.append("+")
        else:
            # Cell row
            r = text_row // 2
            for c in range(cols):
                # Vertical wall between (r, c-1) and (r, c)
                if c == 0:
                    # Left border — check for opening
                    if openings.get((r, 0)) == "left":
                        line.append(" ")
                    else:
                        line.append("|")
                elif has_wall(adjacency, (r, c - 1), (r, c)):
                    line.append("|")
                else:
                    line.append(" ")

                # Cell content
                if (r, c) == start:
                    line.append(" S ")
                elif (r, c) == goal:
                    line.append(" G ")
                else:
                    line.append("   ")
            # Right border — check for opening
            if openings.get((r, cols - 1)) == "right":
                line.append(" ")
            else:
                line.append("|")

        lines.append("".join(line))

    return "\n".join(lines)
