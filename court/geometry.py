"""
court/geometry.py — official NBA court geometry + the 14-keypoint scheme.

Coordinate system: center-origin, feet. x runs baseline→baseline in [-47, 47],
y runs sideline→sideline in [-25, 25]. Reused from the old project (the geometry
was correct; only the labeling was missing).

The 14 keypoints are crisp, clickable line intersections (paint corners, half-court
sideline points, three-point corner breaks) — chosen because they localize
precisely, unlike basket centers or arc midpoints. ≥4 visible non-collinear points
give a homography; more over-constrain it for a robust RANSAC fit.

This module also draws the court (as polylines in feet) — used both for the
labeler's guide diagram and for overlaying the projected court on frames to
visually verify the homography.
"""
from __future__ import annotations

import math

import numpy as np

# ── Court constants (feet) ────────────────────────────────────────────────────
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
HALF_LENGTH = COURT_LENGTH / 2.0   # 47
HALF_WIDTH = COURT_WIDTH / 2.0     # 25
PAINT_DEPTH = 19.0
PAINT_HALF_W = 8.0
THREE_RADIUS = 23.75
THREE_CORNER_Y = 22.0
THREE_CORNER_DX = math.sqrt(THREE_RADIUS ** 2 - THREE_CORNER_Y ** 2)  # ≈ 8.95
BASKET_LEFT = np.array([-41.75, 0.0])
BASKET_RIGHT = np.array([41.75, 0.0])
THREE_LEFT_X = BASKET_LEFT[0] + THREE_CORNER_DX    # ≈ -32.80
THREE_RIGHT_X = BASKET_RIGHT[0] - THREE_CORNER_DX  # ≈  32.80

# ── The 14 keypoints (ordered; index = YOLO-pose keypoint index) ──────────────
# Names: paint corners are {home/away}_paint_{b=near-sideline/t=far-sideline}{l=baseline/r=FT-line}.
KEYPOINT_NAMES = [
    "home_paint_bl", "home_paint_tl", "home_paint_br", "home_paint_tr",   # 0-3
    "away_paint_bl", "away_paint_tl", "away_paint_br", "away_paint_tr",   # 4-7
    "half_bot", "half_top",                                               # 8-9   half line ∩ sidelines
    "center_bot", "center_top",                                           # 10-11 half line ∩ center circle
    "home_corner3_b", "home_corner3_t",                                   # 12-13 corner-3 line ∩ home baseline
    "away_corner3_b", "away_corner3_t",                                   # 14-15 corner-3 line ∩ away baseline
    "home_corner_b", "home_corner_t",                                     # 16-17 court corner (baseline ∩ sideline)
    "away_corner_b", "away_corner_t",                                     # 18-19 court corner (baseline ∩ sideline)
]

COURT_KEYPOINTS: dict[str, np.ndarray] = {
    "home_paint_bl": np.array([-HALF_LENGTH, -PAINT_HALF_W]),               # (-47, -8)
    "home_paint_tl": np.array([-HALF_LENGTH,  PAINT_HALF_W]),               # (-47, +8)
    "home_paint_br": np.array([-HALF_LENGTH + PAINT_DEPTH, -PAINT_HALF_W]), # (-28, -8)
    "home_paint_tr": np.array([-HALF_LENGTH + PAINT_DEPTH,  PAINT_HALF_W]), # (-28, +8)
    "away_paint_bl": np.array([HALF_LENGTH - PAINT_DEPTH, -PAINT_HALF_W]),  # (+28, -8)
    "away_paint_tl": np.array([HALF_LENGTH - PAINT_DEPTH,  PAINT_HALF_W]),  # (+28, +8)
    "away_paint_br": np.array([HALF_LENGTH, -PAINT_HALF_W]),                # (+47, -8)
    "away_paint_tr": np.array([HALF_LENGTH,  PAINT_HALF_W]),                # (+47, +8)
    "half_bot": np.array([0.0, -HALF_WIDTH]),                              # (0, -25)
    "half_top": np.array([0.0,  HALF_WIDTH]),                              # (0, +25)
    # half-court line ∩ center circle (radius 6 ft) — crisp, central, conditions H well
    "center_bot": np.array([0.0, -6.0]),                                   # (0, -6)
    "center_top": np.array([0.0,  6.0]),                                   # (0, +6)
    # corner-3 line ∩ baseline — crisp T-intersections (replace fuzzy arc-break points)
    "home_corner3_b": np.array([-HALF_LENGTH, -THREE_CORNER_Y]),           # (-47, -22)
    "home_corner3_t": np.array([-HALF_LENGTH,  THREE_CORNER_Y]),           # (-47, +22)
    "away_corner3_b": np.array([HALF_LENGTH, -THREE_CORNER_Y]),            # (+47, -22)
    "away_corner3_t": np.array([HALF_LENGTH,  THREE_CORNER_Y]),            # (+47, +22)
    # court corners — baseline ∩ sideline; crispest points, anchor the sidelines
    "home_corner_b": np.array([-HALF_LENGTH, -HALF_WIDTH]),                # (-47, -25)
    "home_corner_t": np.array([-HALF_LENGTH,  HALF_WIDTH]),                # (-47, +25)
    "away_corner_b": np.array([HALF_LENGTH, -HALF_WIDTH]),                 # (+47, -25)
    "away_corner_t": np.array([HALF_LENGTH,  HALF_WIDTH]),                 # (+47, +25)
}

# Flip pairs for left↔right mirror augmentation (home ↔ away; on-axis points self-map).
FLIP_IDX = [6, 7, 4, 5, 2, 3, 0, 1, 8, 9, 10, 11, 14, 15, 12, 13, 18, 19, 16, 17]


def keypoint_court_array() -> np.ndarray:
    """(N, 2) court coords in KEYPOINT_NAMES order."""
    return np.array([COURT_KEYPOINTS[n] for n in KEYPOINT_NAMES], dtype=np.float32)


def court_line_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Straight court line segments as (P1, P2) endpoint pairs in court feet — used
    for line-based homography refinement. Excludes the 3pt arcs and center circle
    (those are conics, not lines)."""
    L, W, p = HALF_LENGTH, HALF_WIDTH, PAINT_HALF_W
    d = PAINT_DEPTH

    def S(ax, ay, bx, by):
        return (np.array([ax, ay], np.float32), np.array([bx, by], np.float32))

    return [
        S(-L, -W, -L, W), S(L, -W, L, W),                 # baselines
        S(-L, -W, L, -W), S(-L, W, L, W),                 # sidelines
        S(0, -W, 0, W),                                   # half-court line
        S(-L, -p, -L + d, -p), S(-L, p, -L + d, p), S(-L + d, -p, -L + d, p),  # home paint + FT line
        S(L, -p, L - d, -p), S(L, p, L - d, p), S(L - d, -p, L - d, p),        # away paint + FT line
        S(-L, -THREE_CORNER_Y, THREE_LEFT_X, -THREE_CORNER_Y),                 # home corner-3 straights
        S(-L, THREE_CORNER_Y, THREE_LEFT_X, THREE_CORNER_Y),
        S(L, -THREE_CORNER_Y, THREE_RIGHT_X, -THREE_CORNER_Y),                 # away corner-3 straights
        S(L, THREE_CORNER_Y, THREE_RIGHT_X, THREE_CORNER_Y),
    ]


def is_on_court(court_xy, margin_ft: float = 0.0) -> bool:
    """True if a court coordinate [x_ft, y_ft] lies within the 94×50 ft court + margin.
    Used for court-masking: detections whose foot-point projects off-court are crowd/bench."""
    x, y = float(court_xy[0]), float(court_xy[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return abs(x) <= HALF_LENGTH + margin_ft and abs(y) <= HALF_WIDTH + margin_ft


# ── Court drawing (polylines in feet) ─────────────────────────────────────────
def _arc(center, radius, a0, a1, n=40):
    ang = np.linspace(math.radians(a0), math.radians(a1), n)
    return np.stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)], axis=1)


def court_polylines() -> list[np.ndarray]:
    """List of (N,2) polylines (feet) tracing the court — for diagram + overlay."""
    L, W = HALF_LENGTH, HALF_WIDTH
    lines = [
        np.array([[-L, -W], [L, -W], [L, W], [-L, W], [-L, -W]]),   # boundary
        np.array([[0, -W], [0, W]]),                                 # half-court line
        np.array([[-L, -PAINT_HALF_W], [-L + PAINT_DEPTH, -PAINT_HALF_W],
                  [-L + PAINT_DEPTH, PAINT_HALF_W], [-L, PAINT_HALF_W]]),   # left paint
        np.array([[L, -PAINT_HALF_W], [L - PAINT_DEPTH, -PAINT_HALF_W],
                  [L - PAINT_DEPTH, PAINT_HALF_W], [L, PAINT_HALF_W]]),     # right paint
    ]
    lines.append(_arc((0, 0), 6.0, 0, 360))  # center circle
    # Left 3PT: corner straights + arc
    lines.append(np.array([[-L, -THREE_CORNER_Y], [THREE_LEFT_X, -THREE_CORNER_Y]]))
    lines.append(np.array([[-L, THREE_CORNER_Y], [THREE_LEFT_X, THREE_CORNER_Y]]))
    a = math.degrees(math.atan2(-THREE_CORNER_Y, THREE_LEFT_X - BASKET_LEFT[0]))
    lines.append(_arc(BASKET_LEFT, THREE_RADIUS, a, -a))
    # Right 3PT
    lines.append(np.array([[L, -THREE_CORNER_Y], [THREE_RIGHT_X, -THREE_CORNER_Y]]))
    lines.append(np.array([[L, THREE_CORNER_Y], [THREE_RIGHT_X, THREE_CORNER_Y]]))
    a2 = math.degrees(math.atan2(THREE_CORNER_Y, THREE_RIGHT_X - BASKET_RIGHT[0]))
    lines.append(_arc(BASKET_RIGHT, THREE_RADIUS, 180 - a2, 180 + a2))
    return lines


def court_to_diagram(pts_ft: np.ndarray, w: int, h: int, margin: int = 20) -> np.ndarray:
    """Map court-feet coords → pixel coords of a w×h schematic diagram."""
    pts_ft = np.atleast_2d(pts_ft)
    sx = (w - 2 * margin) / COURT_LENGTH
    sy = (h - 2 * margin) / COURT_WIDTH
    s = min(sx, sy)
    cx, cy = w / 2.0, h / 2.0
    out = np.empty_like(pts_ft, dtype=np.float32)
    out[:, 0] = cx + pts_ft[:, 0] * s
    out[:, 1] = cy - pts_ft[:, 1] * s   # flip y so +y is up
    return out


def draw_court_diagram(highlight: str | None = None, w: int = 470, h: int = 250):
    """Render the court schematic; if `highlight` names a keypoint, mark it red."""
    import cv2

    img = np.full((h, w, 3), 30, np.uint8)
    for poly in court_polylines():
        px = court_to_diagram(poly, w, h).astype(np.int32)
        cv2.polylines(img, [px], False, (180, 180, 180), 1, cv2.LINE_AA)
    # all keypoints as small dots
    for name in KEYPOINT_NAMES:
        p = court_to_diagram(COURT_KEYPOINTS[name], w, h)[0].astype(int)
        cv2.circle(img, tuple(p), 3, (90, 200, 90), -1)
    if highlight is not None and highlight in COURT_KEYPOINTS:
        p = court_to_diagram(COURT_KEYPOINTS[highlight], w, h)[0].astype(int)
        cv2.circle(img, tuple(p), 7, (60, 60, 255), 2)
        cv2.circle(img, tuple(p), 2, (60, 60, 255), -1)
    return img
