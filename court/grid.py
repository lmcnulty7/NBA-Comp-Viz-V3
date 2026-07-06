"""
court/grid.py — the 13×7 court-grid keypoint scheme (91 points).

Canonical definition of the grid used by the grid-pose court detector
(models/court_grid_snapped.pt). Index convention: idx = iy * GRID_NX + ix,
x along court length (0→94 ft), y across width (0→50 ft) — matches
data/court_labels_grid/grid_def.tsv and the training data.yaml flip_idx.

Unlike the 33-vertex scheme, grid points are NOT painted landmarks — they're
arbitrary court positions the model learns to localize from context. Their value
is redundancy: ~37 visible points/frame (vs ~14) makes the RANSAC homography fit
far more robust, especially in tight shots where few true vertices are visible.
"""
from __future__ import annotations

import numpy as np

from .court33 import COURT_LENGTH_FT, COURT_WIDTH_FT

GRID_NX, GRID_NY = 13, 7
GRID_FT = np.array([(x, y)
                    for y in np.linspace(0.0, COURT_WIDTH_FT, GRID_NY)
                    for x in np.linspace(0.0, COURT_LENGTH_FT, GRID_NX)], np.float32)
N_KP_GRID = len(GRID_FT)   # 91
