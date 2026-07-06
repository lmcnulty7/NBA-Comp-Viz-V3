"""
detect/types.py — the Track data structure shared across the tracking component
and (later) downstream stages.

Kept deliberately rich: team_id / court_pos are placeholders that downstream
components (team classification, homography) will fill, so a Track can flow
through the whole pipeline without changing shape — the same "stable interface"
idea that let the gate drop in cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Track:
    track_id: int                 # persistent ID from BoT-SORT (stable within a camera shot)
    bbox: np.ndarray              # (4,) xyxy in original pixels, float32
    center: np.ndarray            # (2,) box center, float32
    confidence: float             # YOLO detection confidence this frame
    frame_idx: int                # absolute source-video frame index
    timestamp_sec: float
    # ── downstream placeholders (None until a later component fills them) ──
    team_id: Optional[int] = None        # 0=home, 1=away, 2=ref
    court_pos: Optional[np.ndarray] = None  # [x_ft, y_ft] after homography

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def foot_point(self) -> np.ndarray:
        """Bottom-center of the box — where the player meets the floor.
        This is the point downstream homography will project to court coords."""
        return np.array([(self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3]], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": [round(float(v), 2) for v in self.bbox],
            "center": [round(float(v), 2) for v in self.center],
            "confidence": round(float(self.confidence), 4),
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "team_id": self.team_id,
            "court_pos": self.court_pos.tolist() if self.court_pos is not None else None,
        }
