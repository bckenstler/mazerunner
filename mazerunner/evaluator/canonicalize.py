"""Convert various submission encodings to dense polylines."""

from typing import Dict, List, Optional, Tuple


def _parse_polyline(data: dict) -> List[Tuple[float, float]]:
    """Parse polyline encoding: data has "points" key with list of [x, y] pairs."""
    if "points" not in data:
        raise ValueError("Polyline data missing 'points' key")
    points = data["points"]
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError("Polyline 'points' must be a non-empty list")
    result = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"Polyline point {i} must be [x, y], got {p}")
        result.append((float(p[0]), float(p[1])))
    return result


def _parse_delta(data: dict) -> List[Tuple[float, float]]:
    """Parse delta encoding: data has "start" [x,y] and "deltas" list of [dx,dy]."""
    if "start" not in data:
        raise ValueError("Delta data missing 'start' key")
    if "deltas" not in data:
        raise ValueError("Delta data missing 'deltas' key")

    start = data["start"]
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        raise ValueError(f"Delta 'start' must be [x, y], got {start}")

    deltas = data["deltas"]
    if not isinstance(deltas, list):
        raise ValueError("Delta 'deltas' must be a list")

    x, y = float(start[0]), float(start[1])
    result = [(x, y)]
    for i, d in enumerate(deltas):
        if not isinstance(d, (list, tuple)) or len(d) != 2:
            raise ValueError(f"Delta entry {i} must be [dx, dy], got {d}")
        x += float(d[0])
        y += float(d[1])
        result.append((x, y))
    return result


def _parse_cell_route(data: dict, render_config: dict) -> List[Tuple[float, float]]:
    """Parse cell_route encoding: data has "cells" list of [row, col].

    Convert each cell to pixel center using render_config info.
    """
    if "cells" not in data:
        raise ValueError("Cell route data missing 'cells' key")
    cells = data["cells"]
    if not isinstance(cells, list) or len(cells) == 0:
        raise ValueError("Cell route 'cells' must be a non-empty list")
    if render_config is None:
        raise ValueError("cell_route encoding requires render_config in GT data")

    corridor_width = render_config["corridor_width"]
    wall_thickness = render_config["wall_thickness"]
    chrome_height_top = render_config["chrome_height_top"]
    chrome_width_left = render_config["chrome_width_left"]

    cell_size = corridor_width + wall_thickness
    maze_origin_x = chrome_width_left + wall_thickness
    maze_origin_y = chrome_height_top + wall_thickness

    result = []
    for i, c in enumerate(cells):
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise ValueError(f"Cell {i} must be [row, col], got {c}")
        row, col = int(c[0]), int(c[1])
        x = maze_origin_x + col * cell_size + corridor_width / 2
        y = maze_origin_y + row * cell_size + corridor_width / 2
        result.append((float(x), float(y)))
    return result


def _parse_multi_segment(
    data: dict, render_config: dict = None
) -> List[Tuple[float, float]]:
    """Parse multi_segment encoding: data has "segments" list.

    Each segment has its own "encoding" and "data". Parse each recursively,
    concatenate. Check continuity: gap > 5px between consecutive segments
    raises ValueError.
    """
    if "segments" not in data:
        raise ValueError("Multi-segment data missing 'segments' key")
    segments = data["segments"]
    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError("Multi-segment 'segments' must be a non-empty list")

    combined = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"Segment {i} must be a dict")
        if "encoding" not in seg or "data" not in seg:
            raise ValueError(f"Segment {i} missing 'encoding' or 'data'")

        encoding = seg["encoding"]
        seg_data = seg["data"]

        if encoding == "polyline":
            points = _parse_polyline(seg_data)
        elif encoding == "delta":
            points = _parse_delta(seg_data)
        elif encoding == "cell_route":
            points = _parse_cell_route(seg_data, render_config)
        elif encoding == "multi_segment":
            points = _parse_multi_segment(seg_data, render_config)
        else:
            raise ValueError(f"Segment {i} has unknown encoding '{encoding}'")

        # Check continuity with previous segment
        if combined and points:
            last = combined[-1]
            first = points[0]
            dist = ((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2) ** 0.5
            if dist > 5.0:
                raise ValueError(
                    f"Discontinuity between segment {i - 1} and {i}: "
                    f"gap={dist:.1f}px (max 5px)"
                )

        combined.extend(points)

    return combined


def clamp_points(
    polyline: List[Tuple[float, float]], W: int, H: int
) -> List[Tuple[float, float]]:
    """Clamp each point to [0, W-1] x [0, H-1]."""
    result = []
    for x, y in polyline:
        cx = max(0.0, min(float(W - 1), x))
        cy = max(0.0, min(float(H - 1), y))
        result.append((cx, cy))
    return result


def densify(
    polyline: List[Tuple[float, float]], max_gap: float = 2.0
) -> List[Tuple[float, float]]:
    """Walk along polyline, insert intermediate points via linear interpolation
    wherever consecutive points are > max_gap apart.
    """
    if len(polyline) < 2:
        return list(polyline)

    result = [polyline[0]]
    for i in range(1, len(polyline)):
        x0, y0 = result[-1]
        x1, y1 = polyline[i]
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5

        if dist > max_gap:
            n_steps = int(dist / max_gap) + 1
            for s in range(1, n_steps):
                t = s / n_steps
                result.append((x0 + dx * t, y0 + dy * t))

        result.append((x1, y1))

    return result


def canonicalize(
    prediction: dict,
    image_size: dict,
    render_config: dict = None,
    max_gap: float = 2.0,
) -> List[Tuple[float, float]]:
    """Master canonicalization function.

    1. Parse based on prediction["encoding"]
    2. Clamp to image bounds
    3. Densify
    4. Return dense polyline
    """
    encoding = prediction.get("encoding")
    data = prediction.get("data")

    if encoding is None or data is None:
        raise ValueError("Prediction must have 'encoding' and 'data'")

    if encoding == "polyline":
        polyline = _parse_polyline(data)
    elif encoding == "delta":
        polyline = _parse_delta(data)
    elif encoding == "cell_route":
        polyline = _parse_cell_route(data, render_config)
    elif encoding == "multi_segment":
        polyline = _parse_multi_segment(data, render_config)
    else:
        raise ValueError(f"Unknown encoding '{encoding}'")

    if len(polyline) == 0:
        raise ValueError("Parsed polyline is empty")

    W = image_size["w"]
    H = image_size["h"]

    polyline = clamp_points(polyline, W, H)
    polyline = densify(polyline, max_gap)

    return polyline
