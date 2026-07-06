"""
court/court33.py — the 33-keypoint court scheme from the court-detection-2 dataset.

These are the exact court-feet coordinates (NBA) that the Roboflow
basketball-court-detection-2 dataset's 33 keypoints correspond to, lifted from
roboflow/sports `CourtConfiguration(NBA, FEET).vertices` (the library that created
the dataset). Index here == keypoint index in the dataset / the trained pose model.

Coordinate system: CORNER-ORIGIN — x runs along the court LENGTH 0→94 ft, y across
the WIDTH 0→50 ft. (Different origin from the legacy 20-pt center-origin scheme;
fine for homography, which only needs a consistent court frame.) We adopt this
scheme because its 850-image dataset is a far stronger training base than our 400
hand-labels, and the points are well-defined.
"""
from __future__ import annotations

import numpy as np

# 33 court vertices in feet (corner-origin), index = pose-model keypoint index.
COURT_VERTICES_33 = np.array([
    (0.00, 0.00), (0.00, 2.99), (0.00, 17.00), (0.00, 33.01), (0.00, 47.02), (0.00, 50.00),   # 0-5  left baseline
    (5.25, 25.00),                                                                              # 6    left rim center
    (13.92, 2.99), (13.92, 47.02),                                                              # 7-8  left 3pt straight ends
    (19.00, 17.00), (19.00, 25.00), (19.00, 33.01),                                             # 9-11 free-throw line / paint
    (27.40, 0.00), (29.01, 25.00), (27.40, 50.00),                                              # 12-14 3pt extent / top of key
    (46.99, 0.00), (46.99, 25.00), (46.99, 50.00),                                              # 15-17 half-court line
    (66.61, 0.00), (65.00, 25.00), (66.61, 50.00),                                              # 18-20 (mirror) top of key
    (75.00, 17.00), (75.00, 25.00), (75.00, 33.01),                                             # 21-23 free-throw line / paint
    (80.09, 2.99), (80.09, 47.02),                                                              # 24-25 right 3pt straight ends
    (88.75, 25.00),                                                                             # 26   right rim center
    (94.00, 0.00), (94.00, 2.99), (94.00, 17.00), (94.00, 33.01), (94.00, 47.02), (94.00, 50.00),  # 27-32 right baseline
], dtype=np.float32)

# Straight court-line segments as (i, j) vertex-index pairs — for drawing / line refinement.
EDGES_33 = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),          # left sideline (x=0)
    (2, 9), (11, 3), (9, 10), (10, 11),               # left paint + free-throw line
    (1, 4),                                            # left 3pt straight (chord)
    (0, 12), (12, 15), (15, 18), (18, 27),            # top boundary (y=0) full length
    (15, 16), (16, 17),                               # center line
    (5, 14), (14, 17), (17, 20), (20, 32),            # bottom boundary (y=50) full length
    (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), # right sideline (x=94)
    (28, 31),                                          # right 3pt straight (chord)
    (29, 21), (21, 22), (22, 23), (23, 30),           # right paint + free-throw line
]

N_KP_33 = len(COURT_VERTICES_33)   # 33
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0


def court33_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Straight court-line segments as (P1, P2) feet endpoint pairs (from EDGES_33) —
    for drawing the court overlay and for line-based refinement."""
    return [(COURT_VERTICES_33[a], COURT_VERTICES_33[b]) for a, b in EDGES_33]


def _circumcircle(pa, pb, pc):
    """Center and radius of the circle through three points."""
    ax, ay = float(pa[0]), float(pa[1])
    bx, by = float(pb[0]), float(pb[1])
    cx, cy = float(pc[0]), float(pc[1])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy), float(np.hypot(ax - ux, ay - uy))


def _arc(center, r, a_from, a_to, step_deg):
    n = max(2, int(abs(a_to - a_from) / np.radians(step_deg)) + 1)
    t = np.linspace(a_from, a_to, n)
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], axis=1).astype(np.float32)


def court33_curves(step_deg: float = 4.0) -> list[np.ndarray]:
    """Curved court features (plus the corner-3 straights, which EDGES_33 lacks) as
    sampled polylines in court-feet — for overlay drawing only, NOT part of the
    33-keypoint scheme. Each 3-pt arc is the circle fit through its three dataset
    vertices (straight-end pair + apex) so it passes exactly through them."""
    curves = []
    for i1, im, i2 in ((7, 13, 8), (24, 19, 25)):          # left / right 3-pt arc
        pa, pb, pc = COURT_VERTICES_33[i1], COURT_VERTICES_33[im], COURT_VERTICES_33[i2]
        c, r = _circumcircle(pa, pb, pc)
        a1 = np.arctan2(pa[1] - c[1], pa[0] - c[0])
        am = np.arctan2(pb[1] - c[1], pb[0] - c[0])
        a2 = np.arctan2(pc[1] - c[1], pc[0] - c[0])
        ccw_full = (a2 - a1) % (2 * np.pi)                  # sweep a1->a2 the way that passes through the apex
        ccw_mid = (am - a1) % (2 * np.pi)
        curves.append(_arc(c, r, a1, a1 + ccw_full, step_deg) if ccw_mid <= ccw_full
                      else _arc(c, r, a1, a1 - (2 * np.pi - ccw_full), step_deg))
    for i1, i2 in ((1, 7), (4, 8), (28, 24), (31, 25)):    # corner-3 straight segments
        curves.append(np.stack([COURT_VERTICES_33[i1], COURT_VERTICES_33[i2]]).astype(np.float32))
    for ctr, r in (((46.99, 25.0), 6.0),                   # center circle
                   ((19.0, 25.0), 6.0), ((75.0, 25.0), 6.0)):  # free-throw circles
        curves.append(_arc(ctr, r, 0.0, 2 * np.pi, step_deg))
    return curves


def court33_boundary(margin_ft: float = 0.0) -> np.ndarray:
    """The court rectangle (+margin) as 4 corner points in feet (corner-origin)."""
    m = margin_ft
    return np.array([[-m, -m], [COURT_LENGTH_FT + m, -m],
                     [COURT_LENGTH_FT + m, COURT_WIDTH_FT + m], [-m, COURT_WIDTH_FT + m]], np.float32)


def court_ft_to_px(pts, scale: float = 8.0, margin: int = 20) -> np.ndarray:
    """Map court-feet (corner-origin) → top-down diagram pixels."""
    pts = np.atleast_2d(np.asarray(pts, np.float32))
    out = pts * scale + margin
    return out.astype(np.int32)


def draw_court_topdown(scale: float = 8.0, margin: int = 20):
    """Render a blank top-down court (length horizontal). Returns (img, scale, margin)."""
    import cv2

    w = int(COURT_LENGTH_FT * scale + 2 * margin)
    h = int(COURT_WIDTH_FT * scale + 2 * margin)
    img = np.full((h, w, 3), 30, np.uint8)
    for P1, P2 in court33_segments():
        a, b = court_ft_to_px(P1, scale, margin)[0], court_ft_to_px(P2, scale, margin)[0]
        cv2.line(img, tuple(a), tuple(b), (170, 170, 170), 2, cv2.LINE_AA)
    for poly in court33_curves():
        cv2.polylines(img, [court_ft_to_px(poly, scale, margin)], False, (170, 170, 170), 2, cv2.LINE_AA)
    return img, scale, margin
