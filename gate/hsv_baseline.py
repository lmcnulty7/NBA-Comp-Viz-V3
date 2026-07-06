"""
gate/hsv_baseline.py — the gate that ACTUALLY SHIPPED in the old project,
recovered verbatim from Basketball_Defensive_Vision/src/pipeline/runner.py:479
(`_is_court_visible`). Reimplemented fresh here (not copied) as a measurable
baseline-to-beat.

Method: convert BGR→HSV, count "maple-floor-colored" pixels, accept the frame
if the floor fraction falls inside a band (too low = crowd/ad/face closeup;
too high = floor-level closeup).

Note on the shared interface: the decision is a BAND, not a monotonic threshold
on a single score, so this gate is reported at its fixed recovered operating
point rather than swept on a PR curve. `score()` returns the floor fraction for
inspection; `is_court_visible()` applies the band.
"""
from __future__ import annotations

import numpy as np

from .common import load_bgr


class HsvGate:
    def __init__(
        self,
        lo=(8, 25, 90),
        hi=(38, 210, 245),
        min_floor_fraction: float = 0.12,
        max_floor_fraction: float = 0.55,
    ):
        self.lo = np.array(lo, dtype=np.uint8)
        self.hi = np.array(hi, dtype=np.uint8)
        self.min_floor_fraction = min_floor_fraction
        self.max_floor_fraction = max_floor_fraction

    def floor_fraction(self, frame_bgr) -> float:
        import cv2

        if not isinstance(frame_bgr, np.ndarray):
            frame_bgr = load_bgr(frame_bgr)  # accepts str / Path
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        return float(mask.sum()) / (255.0 * frame_bgr.shape[0] * frame_bgr.shape[1])

    def score(self, frame_bgr) -> float:
        """Raw floor fraction (for inspection — NOT a calibrated P(live))."""
        return self.floor_fraction(frame_bgr)

    def is_court_visible(self, frame_bgr, threshold=None) -> bool:
        # `threshold` is ignored: the rule is a fixed band, kept for interface parity.
        frac = self.floor_fraction(frame_bgr)
        return self.min_floor_fraction <= frac <= self.max_floor_fraction

    def predict_from_fraction(self, fracs: np.ndarray) -> np.ndarray:
        """Vectorized band decision over precomputed floor fractions → 0/1 (live)."""
        fracs = np.asarray(fracs)
        return ((fracs >= self.min_floor_fraction) & (fracs <= self.max_floor_fraction)).astype(int)
